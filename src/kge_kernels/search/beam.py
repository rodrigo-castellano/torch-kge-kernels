"""BeamSearcher: top-k sequential proof search.

Composes framework primitives with ``BeamSelect``. Like
:class:`GreedySearcher` but keeps the top-k per-query candidates at
each step. The aggregation across beams happens in ``query_repr``
(typically ``MaxQueryRepr`` so the best beam wins).
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
from ..framework.select import BeamSelect, StateFactory


class BeamSearcher(nn.Module):
    """Beam proof search returning ``{name: [N]}`` per-query scores."""

    def __init__(
        self,
        *,
        resolve: ResolutionOp,
        atom_repr: AtomRepr,
        state_repr: StateRepr,
        traj_repr: TrajRepr,
        query_repr: QueryRepr,
        beam_width: int,
        model: Any = None,
        max_depth: int = 20,
        name: str = "beam",
        state_factory: Optional[StateFactory] = None,
        gumbel_scale_buf: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        self.resolve = resolve
        self.atom_repr = atom_repr
        self.state_repr = state_repr
        self.select = BeamSelect(k=beam_width, state_factory=state_factory)
        self.traj_repr = traj_repr
        self.query_repr = query_repr
        self.model = model
        self.max_depth = max_depth
        self.name = name
        self.beam_width = beam_width
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
        if self._gumbel_scale_buf is not None:
            self._gumbel_scale_buf.fill_(scale)


__all__ = ["BeamSearcher"]
