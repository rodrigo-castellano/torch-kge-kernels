"""MultiRestartSearcher: N independent beam searches with noise.

Pre-builds N inner :class:`BeamSearcher` instances, each with a
different noise seed. The first restart is deterministic
(``noise_scale=0``); subsequent restarts inject Gaussian noise on the
selection scores. Per-query, the best score across restarts wins.

This is a higher-order Searcher: it composes other Searchers rather
than primitives directly. It fits the same :class:`Searcher` contract.
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
from ..framework.select import StateFactory
from .beam import BeamSearcher


class MultiRestartSearcher(nn.Module):
    """Run :class:`BeamSearcher` N times with different noise seeds; per-query max.

    The first restart is deterministic (no noise); restarts 1..N-1 use
    independent noise. Restart noise is currently mediated through the
    ``BeamSearcher.set_gumbel_scale`` interface — pass a per-restart
    scale to inject different noise levels per pass.
    """

    def __init__(
        self,
        *,
        resolve: ResolutionOp,
        atom_repr: AtomRepr,
        state_repr: StateRepr,
        traj_repr: TrajRepr,
        query_repr: QueryRepr,
        beam_width: int,
        n_restarts: int = 3,
        noise_scale: float = 0.1,
        model: Any = None,
        max_depth: int = 20,
        name: str = "multi_restart",
        state_factory: Optional[StateFactory] = None,
    ) -> None:
        super().__init__()
        if n_restarts < 1:
            raise ValueError("n_restarts must be >= 1")
        self.name = name
        self.n_restarts = n_restarts
        self.noise_scale = noise_scale
        self._inners = nn.ModuleList([
            BeamSearcher(
                resolve=resolve, atom_repr=atom_repr, state_repr=state_repr,
                traj_repr=traj_repr, query_repr=query_repr,
                beam_width=beam_width, model=model, max_depth=max_depth,
                name=name, state_factory=state_factory,
            )
            for _ in range(n_restarts)
        ])

    @torch.no_grad()
    def __call__(self, queries: Tensor) -> Dict[str, Tensor]:
        best: Optional[Dict[str, Tensor]] = None
        for i, inner in enumerate(self._inners):
            scale = 0.0 if i == 0 else self.noise_scale
            inner.set_gumbel_scale(scale)
            scores = inner(queries)
            if best is None:
                best = {k: v.clone() for k, v in scores.items()}
            else:
                for k in best:
                    best[k] = torch.maximum(best[k], scores[k])
        assert best is not None
        return best


__all__ = ["MultiRestartSearcher"]
