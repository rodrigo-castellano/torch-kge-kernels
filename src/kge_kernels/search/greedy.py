"""GreedySearcher: argmax-based sequential proof search.

Composes framework primitives in the canonical loop with
``GreedySelect`` (argmax over per-state scores). The loop body is
``framework.search_and_score`` — this class is a thin configuration
wrapper that satisfies the :class:`Searcher` Protocol.

Optional Gumbel noise on selection scores via ``set_gumbel_scale``
(used by :class:`MultiRolloutSearcher` composition). When the buffer
is a static-address tensor, mutating it does not break CUDA-graph
capture.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..framework import (
    AtomRepr,
    QueryRepr,
    ResolutionOp,
    StateRepr,
    TrajRepr,
)
from ..framework.scorer import search_and_score
from ..framework.select import GreedySelect, StateFactory


class GreedySearcher(nn.Module):
    """Greedy proof search returning ``{name: [N]}`` per-query scores."""

    def __init__(
        self,
        *,
        resolve: ResolutionOp,
        atom_repr: AtomRepr,
        state_repr: StateRepr,
        traj_repr: TrajRepr,
        query_repr: QueryRepr,
        model: Any = None,
        max_depth: int = 20,
        name: str = "greedy",
        state_factory: Optional[StateFactory] = None,
        gumbel_scale_buf: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        self.resolve = resolve
        self.atom_repr = atom_repr
        self.state_repr = state_repr
        self.select = GreedySelect(state_factory=state_factory)
        self.traj_repr = traj_repr
        self.query_repr = query_repr
        self.model = model
        self.max_depth = max_depth
        self.name = name
        self._gumbel_scale_buf = gumbel_scale_buf

    @torch.no_grad()
    def __call__(self, queries: Tensor) -> Dict[str, Tensor]:
        scores = search_and_score(
            queries,
            resolve=self.resolve,
            atom_repr=self.atom_repr,
            state_repr=self.state_repr,
            select=self.select,
            traj_repr=self.traj_repr,
            query_repr=self.query_repr,
            model=self.model,
            max_depth=self.max_depth,
        )
        return {self.name: scores}

    def set_gumbel_scale(self, scale: float) -> None:
        """Mutate noise level for MultiRolloutSearcher composition.

        No-op if no static-address buffer was supplied. When supplied,
        the buffer is filled in place — CUDA-graph-safe if the
        downstream ``select`` reads through the same buffer reference.
        """
        if self._gumbel_scale_buf is not None:
            self._gumbel_scale_buf.fill_(scale)


__all__ = ["GreedySearcher"]
