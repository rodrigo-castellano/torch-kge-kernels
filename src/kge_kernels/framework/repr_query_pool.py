"""Pool→query primitives for the rule-based reasoning path.

After :class:`RuleTrajRepr` returns the updated atom pool, the
``query_repr`` step gathers the query atoms' entries from the pool to
produce per-query scalar scores.

Two implementations:

* :class:`LookupAtPool` — scalar pool ``[N_pool]`` (SBR / DCR). The pool
  already stores per-atom scores; just gather at the query indices.
* :class:`OutputLayerAtPool` — embedding pool ``[N_pool, E]`` (R2N). The
  pool stores per-atom embeddings; gather at the query indices and
  apply the KGE model's ``output_layer`` to convert embedding → score.

Both honor the ``ever_written [N_pool]`` boolean mask returned by
:class:`RuleTrajRepr`. Queries that land on an atom that was *not*
written by any valid scatter (the rule program never derived that atom)
are tied at a constant score so they rank uniformly random — matching
keras-ns's "no-grounding → tied last" semantic.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor


class LookupAtPool(nn.Module):
    """Score-pool gather. Used by SBR/DCR (scalar pool ``[N_pool]``).

    For unwritten queries (``ever_written[query_pool_idx]=False``)
    returns the default ``unwritten_score`` (0.0 by default — neutral
    score that ties unwritten queries at the same value). Pass
    ``unwritten_score=None`` to leave their KGE-init values intact (the
    fallback chosen by the per-tree path).
    """

    def __init__(self, unwritten_score: float = 0.0) -> None:
        super().__init__()
        self.unwritten_score = unwritten_score

    def forward(
        self,
        pool: Tensor,                    # [N_pool] or [N_pool, 1]
        query_pool_idx: Tensor,          # [B]
        ever_written: Tensor,            # [N_pool] bool
    ) -> Tensor:
        if pool.dim() == 2 and pool.shape[-1] == 1:
            pool = pool.squeeze(-1)
        scores = pool[query_pool_idx]                          # [B]
        is_written = ever_written[query_pool_idx]              # [B] bool
        if self.unwritten_score is None:
            return scores
        scores = torch.where(
            is_written,
            scores,
            torch.full_like(scores, float(self.unwritten_score)),
        )
        return scores


class OutputLayerAtPool(nn.Module):
    """Embedding-pool gather + KGE output_layer. Used by R2N (``[N_pool, E]``).

    Gathers the query atoms' embeddings from the pool and applies
    ``output_layer`` to convert embedding → score.

    Queries that land on an unwritten atom get their embedding zeroed
    before ``output_layer``; ``output_layer(0) = sigmoid(bias)`` is a
    constant — the same value for every unwritten query — so they all
    tie at the same score and rank uniformly random. This matches
    keras-ns's R2N "no-grounding → tied last" semantic without leaking
    the atom's KGE-baseline embedding (which would masquerade as a KGE
    fallback even when the rule MLP never wrote that slot).
    """

    def __init__(self, output_layer: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self.output_layer = output_layer

    def forward(
        self,
        pool: Tensor,                    # [N_pool, E]
        query_pool_idx: Tensor,          # [B]
        ever_written: Tensor,            # [N_pool] bool
    ) -> Tensor:
        emb = pool[query_pool_idx]                              # [B, E]
        is_written = ever_written[query_pool_idx]               # [B] bool
        emb = torch.where(
            is_written.unsqueeze(-1),
            emb,
            torch.zeros_like(emb),
        )
        scores = self.output_layer(emb)                         # [B] or [B, 1]
        if scores.dim() == 2 and scores.shape[-1] == 1:
            scores = scores.squeeze(-1)
        return scores


__all__ = ["LookupAtPool", "OutputLayerAtPool"]
