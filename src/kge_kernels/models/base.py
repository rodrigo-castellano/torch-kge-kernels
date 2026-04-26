"""Abstract base class for KGE models in tkk.

A ``KGEModel`` is a ``torch.nn.Module`` with the following contract.
**Every method below is part of the contract** — callers (training
pipeline, eval pipeline, scoring helpers) may rely on any of them.
Some are abstract (subclass must implement); the rest have a default
implementation in this base class.

Why training/eval orchestration lives on the model (and not as free
functions over a model): they are the model's ``torch.compile`` boundary.
Subclasses with a different atom layout (reasoners that build a
``[B*(1+K), 3]`` pool for the grounder rather than the static-flat
``[B + B*K, 3]`` KGE-only pool) override the methods to keep one
compiled graph per model. Free functions can't do that polymorphic
dispatch, so the contract bakes in the override points.

Required attributes (set in ``__init__``):
  - ``num_entities: int``
  - ``num_relations: int``
  - ``dim: int``
  - ``entity_embeddings: nn.Embedding``  (the entity table)

Abstract — subclass MUST implement:
  - ``score(h, r, t, *, d_chunk=None) -> Tensor`` — single scoring entry
    point. ``h, r, t`` all bound: returns ``[B]`` triple scores.
    ``t=None``: returns ``[B, E]`` ranking against all tails. ``h=None``:
    returns ``[B, E]`` ranking against all heads. ``d_chunk=None``
    (default) means score every entity in one pass; an integer enables
    memory-chunked scoring on models that support it (RotatE / RotatENS).
  - ``compose(h, r, t) -> Tensor`` — fused per-atom embedding consumed
    by ``KGEEmbedAtom`` (the layer that turns atoms into MLP features).
  - ``reset_parameters() -> None`` — re-init weights (called from
    ``__init__`` and from external repro/seed setups).

Provided by base — override only when a model needs to:
  - ``train_step(pos, neg, mask, pos_valid) -> scalar`` — unified
    compile-boundary training loss. Default: mask-aware BCE on top of
    ``score``. RotatE/RotatENS flip ``_train_loss_is_from_logits=False``
    so the default uses sigmoid+BCE instead of BCE-with-logits. ns's
    ``ReasonerModel`` overrides for the reasoner atom-pool layout.
  - ``eval_scores(q_buf, cand_buf, mode) -> [B, C]`` — unified eval hook
    consumed by :mod:`kge_kernels.eval.unified`. Default: calls
    ``score(h, r, None)`` / ``score(None, r, t)`` and gathers the ``C``
    candidates per query. ns's ``ReasonerModel`` overrides for the
    candidate-pool replay path.
  - ``recommended_eval_batch_size(num_entities, budget_gb=2.0) -> int``
    — memory hint used by the eval pipeline to pick a safe ``B``.
    Default assumes matmul-style ``[B, |E|]`` peak; RotatE overrides to
    factor in ``half_dim``.
  - ``forward(h, r, t) -> Tensor`` — ``nn.Module`` convention; delegates
    to ``score``.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


Mode = Literal["tail", "head"]


class KGEModel(nn.Module):
    """Base class for KGE models. See module docstring for the contract."""

    num_entities: int
    num_relations: int
    dim: int

    @abstractmethod
    def score(
        self,
        h: Optional[Tensor],
        r: Tensor,
        t: Optional[Tensor],
        *,
        d_chunk: Optional[int] = None,
    ) -> Tensor:
        """Score (h, r, t) triples or rank against all heads/tails.

        Modes (decided by which of ``h`` / ``t`` are bound):

        - ``h, r, t`` all bound: returns ``[B]`` per-triple scores.
        - ``h, r`` bound, ``t=None``: returns ``[B, E]`` ranking all tails.
        - ``h=None``, ``r, t`` bound: returns ``[B, E]`` ranking all heads.

        ``d_chunk``:
          - ``None`` (default): score all entities in **one pass** (full
            broadcast / matmul). Use this whenever memory is not the
            bottleneck.
          - ``int``: enables memory-chunked exhaustive scoring along the
            embedding dimension. Currently implemented by RotatE /
            RotatENS — other models ignore the value and still do
            one-pass.
        """

    @abstractmethod
    def compose(self, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
        """Fused per-atom embedding ``[B, *]`` consumed by KGEEmbedAtom."""

    @abstractmethod
    def reset_parameters(self) -> None:
        """Re-initialise weights."""

    # Read by ``train_step``. Override on subclasses that need
    # sigmoid+BCE instead of BCE-with-logits (RotatE family).
    _train_loss_is_from_logits: bool = True

    def train_step(
        self,
        pos: Tensor,           # [B, 3]       (r, h, t) positive triples
        neg: Tensor,           # [B, K, 3]    candidate negatives
        mask: Tensor,          # [B, K]       which negatives are valid
        pos_valid: Tensor,     # [B]          which positives are real (not tail-pad)
    ) -> Tensor:               # scalar loss
        """Unified training-step hook — the compile boundary for per-batch
        loss computation. Mask-aware BCE on top of ``score``.

        Every shape is a compile-time constant: ``[B, 3]``, ``[B, K, 3]``,
        ``[B, K]``, ``[B]`` never vary, so this traces into one CUDA graph
        per ``(model, B, K)`` combo and replays across every step.

        Subclasses with a different atom layout (e.g. ns's
        ``ReasonerModel`` building a ``[B*(1+K), 3]`` pool for the
        grounder) override to keep their own compiled graph.
        """
        B, K, _ = neg.shape
        # Flat, static-shape atom layout: [B + B*K, 3], cols (r, h, t).
        neg_flat = neg.reshape(B * K, 3)
        all_items = torch.cat([pos, neg_flat], dim=0)
        # score consumes (h, r, t); items are in (r, h, t).
        scores = self.score(all_items[:, 1], all_items[:, 0], all_items[:, 2])
        pos_scores = scores[:B]                          # [B]
        neg_scores = scores[B:]                          # [B*K]

        pos_w = pos_valid.to(dtype=pos_scores.dtype)     # [B]
        neg_w = mask.to(dtype=neg_scores.dtype).reshape(-1)   # [B*K]

        pos_labels = torch.ones_like(pos_scores)
        neg_labels = torch.zeros_like(neg_scores)
        if self._train_loss_is_from_logits:
            pos_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_labels, reduction="none")
            neg_loss = F.binary_cross_entropy_with_logits(neg_scores, neg_labels, reduction="none")
        else:
            pos_loss = F.binary_cross_entropy(torch.sigmoid(pos_scores), pos_labels, reduction="none")
            neg_loss = F.binary_cross_entropy(torch.sigmoid(neg_scores), neg_labels, reduction="none")

        numer = (pos_loss * pos_w).sum() + (neg_loss * neg_w).sum()
        denom = (pos_w.sum() + neg_w.sum()).clamp_min(1.0)
        return numer / denom

    def recommended_eval_batch_size(
        self, num_entities: int, budget_gb: float = 2.0,
    ) -> int:
        """Largest safe compile-graph batch size for :meth:`eval_scores`.

        Default assumes matmul-style ``[B, |E|]`` output (ComplEx,
        DistMult, TransE, ModE, TuckER): peak intermediate is bounded
        by ``B × |E| × 4 bytes``. RotatE overrides to factor in
        ``half_dim`` (its broadcast is ``[B, |E|, half_dim]``).
        """
        bytes_per_B = 4 * int(num_entities)                 # [B, |E|] float32
        budget_b = budget_gb * (1 << 30)
        return max(16, min(4096, int(budget_b / max(bytes_per_B, 1))))

    def eval_scores(
        self,
        q_buf: Tensor,     # [B, 3]  int64, columns (r, h, t)
        cand_buf: Tensor,  # [B, C]  int64, entity indices to score
        mode: Mode,
    ) -> Tensor:           # [B, C]  float
        """Score candidates against a query batch — the compile boundary
        for :func:`kge_kernels.eval.unified.evaluate`.

        Calls ``score`` with the matmul fast path (``[B, |E|]`` in one
        BLAS call) and gathers the ``C`` candidates per query. Subclasses
        with a different scoring path (e.g. ns's ``ReasonerModel``
        candidate-pool replay) override.
        """
        r = q_buf[:, 0]
        if mode == "tail":
            h = q_buf[:, 1]
            all_scores = self.score(h, r, None)                    # [B, |E|]
        else:
            t = q_buf[:, 2]
            all_scores = self.score(None, r, t)                    # [B, |E|]
        return all_scores.gather(1, cand_buf)                      # [B, C]

    def forward(
        self,
        h: Optional[Tensor],
        r: Tensor,
        t: Optional[Tensor] = None,
    ) -> Tensor:  # noqa: D401
        """Default forward delegates to ``score`` for nn.Module compatibility."""
        return self.score(h, r, t)


__all__ = ["KGEModel", "Mode"]
