"""Abstract base class for KGE models in tkk.

A ``KGEModel`` is a torch ``nn.Module`` that knows how to:
  - score specific triples       → ``score_triples(h, r, t) -> [N]``
  - score all tails for ``(h, r)`` → ``score_all_tails(h, r) -> [N, E]``
  - score all heads for ``(r, t)`` → ``score_all_heads(r, t) -> [N, E]``
  - compose a fused embedding    → ``compose(h, r, t) -> [N, E]``

The first three methods plug directly into ``KGEBackend`` (the existing
``kge_kernels.scoring`` protocol). The ``compose`` method is what
``KGEEmbedAtom`` consumes — it's the fused per-atom embedding (TransE's
``h+r-t``, ComplEx's bilinear form, etc.) that used to be duplicated in
``DpRL.kge_experiments.nn.atom_embedders`` and ``torch-ns.ns_lib.nn.kge_layers``.

``score(h, r, t=None)`` is a convenience dispatch provided by the base:
``t is None`` → all tails, ``h is None`` → all heads, otherwise specific
triples. This matches the call convention used by DpRL's inference and
evaluation paths.

``eval_scores(q_buf, cand_buf, mode)`` is the unified-eval hook consumed
by :mod:`kge_kernels.eval.unified` — a ``[B, 3]`` query batch plus a
``[B, C]`` candidate-entity matrix returns ``[B, C]`` scores. The base
implementation expands to per-atom ``score_triples`` so every subclass
works out of the box; a subclass that wants a faster specialisation
(e.g. one matmul for exhaustive mode via ``score_all_tails``) can
override. Same shape contract either way.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


Mode = Literal["tail", "head"]


class KGEModel(nn.Module):
    """Base class for KGE models with the four required scoring methods."""

    num_entities: int
    num_relations: int
    dim: int

    @abstractmethod
    def score_triples(self, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
        """Score a batch of fully ground triples → ``[N]``."""

    @abstractmethod
    def score_all_tails(self, h: Tensor, r: Tensor) -> Tensor:
        """Score all entities as tail for each ``(h, r)`` pair → ``[N, E]``."""

    @abstractmethod
    def score_all_heads(self, r: Tensor, t: Tensor) -> Tensor:
        """Score all entities as head for each ``(r, t)`` pair → ``[N, E]``."""

    @abstractmethod
    def compose(self, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
        """Fused per-atom embedding ``[N, E]`` consumed by KGEEmbedAtom."""

    def score(
        self,
        h: Optional[Tensor],
        r: Tensor,
        t: Optional[Tensor] = None,
    ) -> Tensor:
        """Unified dispatch: specific triples, all tails, or all heads.

        - ``h`` and ``t`` both non-None → ``score_triples`` → ``[N]``
        - ``t is None`` and ``h`` non-None → ``score_all_tails`` → ``[N, E]``
        - ``h is None`` and ``t`` non-None → ``score_all_heads`` → ``[N, E]``
        """
        if h is not None and t is not None:
            return self.score_triples(h, r, t)
        if t is None:
            if h is None:
                raise ValueError("score() requires at least one of h or t to be non-None")
            return self.score_all_tails(h, r)
        return self.score_all_heads(r, t)

    # -- Override this to pick a different loss (e.g. sigmoid+BCE for
    # -- RotatE whose raw scores saturate BCE_with_logits on large KGs).
    _train_loss_is_from_logits: bool = True

    def train_step(
        self,
        pos: Tensor,           # [B, 3]       (r, h, t) positive triples
        neg: Tensor,           # [B, K, 3]    candidate negatives
        mask: Tensor,          # [B, K]       which negatives are valid
        pos_valid: Tensor,     # [B]          which positives are real (not tail-pad)
    ) -> Tensor:               # scalar loss
        """Unified training-step hook — the single compile boundary for
        per-batch loss computation shared by ns and tkk trainers.

        Every shape is a compile-time constant: ``[B, 3]``, ``[B, K, 3]``,
        ``[B, K]``, ``[B]`` never vary. Mask-aware BCE reduction gates
        invalid negatives (``mask=False``) and padded positives
        (``pos_valid=False``, only non-trivial for the last partial
        batch of an epoch). This makes the whole method
        ``torch.compile(fullgraph=True, mode='reduce-overhead')``
        compatible — one CUDA graph per model, reused across every
        training step.

        The cost vs the old lean ``cat([pos, neg[mask]])`` path is
        scoring ``K`` atoms per row instead of ``mask.sum(dim=1)`` per
        row — typically 0-10% extra FLOPs when filter/unique evict a
        few candidates. That's the price of fullgraph compile.

        Subclasses can override for model-specific layouts (e.g.
        reasoners building a ``[B*(1+K), 3]`` atom pool for the
        grounder) — see ``ns.experiments.model.ReasonerModel``.
        """
        B, K, _ = neg.shape
        # Flat, static-shape atom layout: [B + B*K, 3], cols (r, h, t).
        neg_flat = neg.reshape(B * K, 3)
        all_items = torch.cat([pos, neg_flat], dim=0)
        # score_triples consumes (h, r, t); items are in (r, h, t).
        scores = self.score_triples(all_items[:, 1], all_items[:, 0], all_items[:, 2])
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
        by ``B × |E| × 4 bytes``, so ``B ≤ budget_gb × 2^30 / (4|E|)``.
        RotatE overrides because its ``score_all_heads`` broadcasts a
        ``[B, |E|, half_dim]`` intermediate — memory scales with
        ``half_dim`` too.

        Callers in the unified eval path use this to pick the compile-
        graph B so a long sweep doesn't OOM on the first matmul. Outer-
        loop chunking inside :func:`kge_kernels.eval.unified.evaluate`
        iterates over all triples in chunks of this size.
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
        """Score candidates against a query batch — unified eval hook.

        The compile boundary for :func:`kge_kernels.eval.unified.evaluate`.
        Shapes are fixed across calls (``B``, ``C`` constant within a
        single :func:`evaluate` call) so this traces into one CUDA graph.

        Uses the matmul fast path when available: ``score_all_tails`` /
        ``score_all_heads`` compute all-entity scores ``[B, |E|]`` in
        one BLAS call, and we gather the ``C`` candidates. For large
        candidate sets this is orders of magnitude faster than per-atom
        ``score_triples`` (one matmul vs ``B*C`` embedding lookups +
        per-dim math).

        The matmul path expects the full candidate pool per call — so
        :func:`evaluate` must pass ``chunk_size = K_fixed`` (one chunk
        covers all candidates). Caller-side chunking is still supported
        via the per-atom fallback below for models that don't expose
        ``score_all_tails`` (e.g., reasoners that would need to re-run
        grounding per entity).

        ``mode`` is a Python string, not a tensor: ``torch.compile``
        specialises one graph per value, keeping the compiled region
        branch-free.
        """
        B, C = cand_buf.shape
        r = q_buf[:, 0]
        if mode == "tail":
            h = q_buf[:, 1]
            all_scores = self.score_all_tails(h, r)                # [B, |E|]
        else:
            t = q_buf[:, 2]
            all_scores = self.score_all_heads(r, t)                # [B, |E|]
        # Gather the C candidate scores per query. ``cand_buf`` is
        # already clamped to valid entity indices by the provider; the
        # validity mask outside this call zeroes out pad slots before
        # ranking.
        return all_scores.gather(1, cand_buf)                      # [B, C]

    def embed_entities(self, indices: Tensor) -> Tensor:
        """Return the full entity representation for the given indices.

        For models with a single entity embedding table (TransE, DistMult,
        ModE, TuckER): returns ``entity_embeddings(indices)``.
        For models with split real/imaginary tables (ComplEx, RotatE):
        returns ``cat(ent_re(indices), ent_im(indices), dim=-1)``.

        Override in subclasses with non-standard embedding layout.
        """
        if hasattr(self, "entity_embeddings"):
            return self.entity_embeddings(indices)
        if hasattr(self, "ent_re") and hasattr(self, "ent_im"):
            import torch
            return torch.cat([self.ent_re(indices), self.ent_im(indices)], dim=-1)
        raise NotImplementedError("Subclass must implement embed_entities or have entity_embeddings / ent_re+ent_im")

    def forward(
        self,
        h: Optional[Tensor],
        r: Tensor,
        t: Optional[Tensor] = None,
    ) -> Tensor:  # noqa: D401
        """Default forward delegates to ``score`` for nn.Module compatibility."""
        return self.score(h, r, t)


__all__ = ["KGEModel", "Mode"]
