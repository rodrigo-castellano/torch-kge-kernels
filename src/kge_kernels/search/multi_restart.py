"""MultiRestartSearcher: N independent UnifiedSearcher beam runs with noise.

Pre-builds N inner :class:`UnifiedSearcher` instances backed by
:class:`BeamSelect`, each with its own static-address Gumbel scale
buffer. The first restart is deterministic (``noise_scale=0``);
subsequent restarts inject Gumbel-max noise at ``noise_scale``.
Per-query, the best score across restarts wins.

Higher-order Searcher: composes Searcher instances rather than
primitives directly. Same :class:`Searcher` contract.
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
from ..framework.select import BeamSelect, StateFactory
from .searcher import CaptureMode, SearchSpec, UnifiedSearcher


class MultiRestartSearcher(nn.Module):
    """Run :class:`UnifiedSearcher` (beam) N times with different noise; per-query max.

    The first restart is deterministic (no noise); restarts 1..N-1 use
    independent Gumbel noise at ``noise_scale``. Restart noise is
    injected via :meth:`UnifiedSearcher.set_gumbel_scale` between calls.
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
        capture: CaptureMode = "dynamic",
        spec: Optional[SearchSpec] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        if n_restarts < 1:
            raise ValueError("n_restarts must be >= 1")
        self.name = name
        self.n_restarts = n_restarts
        self.noise_scale = noise_scale

        if spec is None:
            spec = SearchSpec(
                batch_size=batch_size if batch_size is not None else 1,
                max_depth=max_depth,
                beam_width=beam_width,
            )

        inners = []
        for _ in range(n_restarts):
            buf = torch.zeros(())
            select = BeamSelect(
                k=beam_width,
                state_factory=state_factory,
                gumbel_scale_buf=buf,
            )
            inners.append(
                UnifiedSearcher(
                    resolve=resolve,
                    atom_repr=atom_repr,
                    state_repr=state_repr,
                    select=select,
                    traj_repr=traj_repr,
                    query_repr=query_repr,
                    spec=spec,
                    model=model,
                    name=name,
                    capture=capture,
                )
            )
        self._inners = nn.ModuleList(inners)

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
