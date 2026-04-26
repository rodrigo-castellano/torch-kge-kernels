"""Abstract base class for KGE models in tkk.

A ``KGEModel`` is a ``torch.nn.Module`` whose contract is just the
KGE math. Training and eval orchestration are free functions in
:mod:`kge_kernels.training` and :mod:`kge_kernels.eval` that operate
over a model.

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

Provided by base:
  - ``forward(h, r, t)`` — ``nn.Module`` convention; delegates to
    ``score``.

Optional class flag:
  - ``_train_loss_is_from_logits: bool = True`` — read by
    :func:`kge_kernels.training.train_step` to decide BCE-with-logits
    vs sigmoid+BCE. RotatE / RotatENS set this to ``False``.

External orchestration (free functions over a model):
  - :func:`kge_kernels.training.train_step` — mask-aware BCE loss
    (unified compile boundary).
  - :func:`kge_kernels.models.kge_default_scorer` — adapt ``score`` to
    the evaluator's ``(q_buf, pool_buf, mode) -> [B, P]`` ScoreFn shape.
  - :func:`kge_kernels.models.recommended_eval_batch_size` — memory-aware
    batch-size hint.

Subclasses with a different atom layout (e.g. ns's ``ReasonerModel``
building a ``[B*(1+K), 3]`` pool for the grounder) override the
training / eval hooks by **defining their own ``train_step`` /
``eval_scores`` methods on the model**: tkk's
:func:`kge_kernels.training.train_epoch` and
:func:`kge_kernels.eval.evaluate.evaluate` prefer ``model.train_step`` /
``model.eval_scores`` when present, and fall back to the free-function
defaults otherwise.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Optional

from torch import Tensor, nn


class KGEModel(nn.Module):
    """Base class for KGE models. See module docstring for the contract."""

    num_entities: int
    num_relations: int
    dim: int

    # Read by kge_kernels.training.train_step. Override on subclasses
    # that need sigmoid+BCE instead of BCE-with-logits (RotatE family).
    _train_loss_is_from_logits: bool = True

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

    def forward(
        self,
        h: Optional[Tensor],
        r: Tensor,
        t: Optional[Tensor] = None,
    ) -> Tensor:  # noqa: D401
        """Default forward delegates to ``score`` for nn.Module compatibility."""
        return self.score(h, r, t)


__all__ = ["KGEModel"]
