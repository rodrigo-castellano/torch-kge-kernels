"""ExhaustiveSearcher: one-shot proof search using ExhaustiveSelect.

The canonical SBR/DCR/R2N pattern (framework.pdf §6.5): ``resolve``
returns all groundings across all depths in a single call,
``ExhaustiveSelect`` is identity, and the loop runs once
(``max_depth=1``). ``traj_repr.forward`` reduces over the depth dim
in batch.

torch-ns keeps its fused reasoning layers as compile-safe
specializations (per framework.pdf §9 architectural contract); this
class is the **reference** implementation that any consumer can use
when speed is not critical.
"""
from __future__ import annotations

from typing import Any, Dict

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
from ..framework.select import ExhaustiveSelect


class ExhaustiveSearcher(nn.Module):
    """Exhaustive (one-shot) proof search returning ``{name: [N]}`` scores."""

    def __init__(
        self,
        *,
        resolve: ResolutionOp,
        atom_repr: AtomRepr,
        state_repr: StateRepr,
        traj_repr: TrajRepr,
        query_repr: QueryRepr,
        model: Any = None,
        name: str = "exhaustive",
    ) -> None:
        super().__init__()
        self.resolve = resolve
        self.atom_repr = atom_repr
        self.state_repr = state_repr
        self.select = ExhaustiveSelect()
        self.traj_repr = traj_repr
        self.query_repr = query_repr
        self.model = model
        self.name = name

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
            max_depth=1,
        )
        return {self.name: scores}


__all__ = ["ExhaustiveSearcher"]
