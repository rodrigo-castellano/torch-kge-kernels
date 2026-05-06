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

Default semantics match keras-ns's pool-iter: queries gather *whatever*
the pool stores at their slot — KGE-init for atoms no rule firing
touched (including provable-as-fact queries) and the K-iter rule-MLP
update for atoms that rule firings did rewrite. Optional
``unwritten_value`` overrides this for callers that want to tie
unwritten queries at a constant (e.g. inference-time random-rank
ablations); ReasonerModel never sets it.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor


class LookupAtPool(nn.Module):
    """Score-pool gather. Used by SBR/DCR (scalar pool ``[N_pool]``).

    Returns ``pool[query_pool_idx]`` directly — including for queries
    that no rule firing wrote (their pool slot still holds the
    KGE-init score from ``atom_repr``). This matches keras-ns: a
    fact-only query (no rule fires, head appears as a fact in the KB)
    keeps its KGE score as the reasoning answer; an unprovable query
    likewise reads its KGE-init.

    ``unwritten_score`` overrides the gather result for unwritten
    queries when set to a float (e.g. 0.0 to tie them at random
    rank). ``None`` (default) lets the pool value through unchanged —
    the keras-faithful semantics.
    """

    def __init__(self, unwritten_score: Optional[float] = None) -> None:
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
        if self.unwritten_score is None:
            return scores
        is_written = ever_written[query_pool_idx]              # [B] bool
        scores = torch.where(
            is_written,
            scores,
            torch.full_like(scores, float(self.unwritten_score)),
        )
        return scores


class OutputLayerAtPool(nn.Module):
    """Embedding-pool gather + KGE output_layer. Used by R2N (``[N_pool, E]``).

    Gathers ``pool[query_pool_idx]`` and applies ``output_layer`` to
    convert embedding → score. Default behavior keeps the pool's
    embedding as-is even for queries no rule firing rewrote (their
    pool slot holds the KGE-pair embedding from ``atom_repr``,
    matching keras-ns's pool-iter semantics).

    ``zero_unwritten=True`` reverts to the legacy "tie unwritten
    queries at sigmoid(bias)" behavior (zero the embedding before
    ``output_layer``); off by default.
    """

    def __init__(
        self,
        output_layer: Callable[[Tensor], Tensor],
        *,
        zero_unwritten: bool = False,
    ) -> None:
        super().__init__()
        self.output_layer = output_layer
        self.zero_unwritten = zero_unwritten

    def forward(
        self,
        pool: Tensor,                    # [N_pool, E]
        query_pool_idx: Tensor,          # [B]
        ever_written: Tensor,            # [N_pool] bool
    ) -> Tensor:
        emb = pool[query_pool_idx]                              # [B, E]
        if self.zero_unwritten:
            is_written = ever_written[query_pool_idx]           # [B] bool
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
