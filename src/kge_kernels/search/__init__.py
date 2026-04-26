"""High-level Searcher compositions for proof-based query ranking.

Per framework.pdf §11, every method is a 6-tuple
``(resolve, atom_repr, state_repr, select, traj_repr, query_repr)``.
This module provides reference :class:`Searcher` classes that compose
those primitives in the canonical search loop, parameterized by which
:class:`Select` they use:

  - :class:`ExhaustiveSearcher` — ``ExhaustiveSelect``, max_depth=1
    (canonical for SBR / DCR / R2N).
  - :class:`GreedySearcher` — ``GreedySelect`` (DpRL eval default).
  - :class:`BeamSearcher` — ``BeamSelect``.
  - :class:`MultiRestartSearcher` — N beam runs with noise.
  - :class:`MultiRolloutSearcher` — higher-order, wraps any base
    Searcher and runs it K times with different Gumbel scales.
  - :class:`DirectSearcher` — no search, AtomRepr only (baseline).

The :class:`Searcher` Protocol and :func:`make_scorer_from_searcher`
adapter let any Searcher feed
:class:`kge_kernels.eval.RankingEvaluator` directly.

Method-specific compiled specializations (e.g., DpRL's
``PolicyRolloutSearcher``) live with their methods and satisfy the
same Protocol — per framework.pdf §9 architectural contract.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

from .beam import BeamSearcher
from .direct import DirectSearcher
from .exhaustive import ExhaustiveSearcher
from .greedy import GreedySearcher
from .multi_restart import MultiRestartSearcher
from .multirollout import MultiRolloutSearcher
from .searcher import Mode, ScoreFn, Searcher, make_scorer_from_searcher


def make_searcher(
    strategy: str,
    *,
    K_rollouts: int = 1,
    multirollout_scales: Optional[Sequence[float]] = None,
    multirollout_scale: float = 0.3,
    **kw: Any,
) -> Searcher:
    """Factory: build a :class:`Searcher` by string name.

    Recognized strategies:
      - ``"greedy"``: :class:`GreedySearcher`
      - ``"beam"``: :class:`BeamSearcher` (requires ``beam_width=...``)
      - ``"multi_restart"``: :class:`MultiRestartSearcher`
      - ``"exhaustive"``: :class:`ExhaustiveSearcher`
      - ``"direct"``: :class:`DirectSearcher`

    When ``K_rollouts > 1``, the result is auto-wrapped in
    :class:`MultiRolloutSearcher`. ``multirollout_scales`` overrides
    the auto-built ``[0.0, scale, scale, ...]`` schedule.
    """
    if strategy == "greedy":
        base: Searcher = GreedySearcher(**kw)
    elif strategy == "beam":
        base = BeamSearcher(**kw)
    elif strategy == "multi_restart":
        base = MultiRestartSearcher(**kw)
    elif strategy == "exhaustive":
        base = ExhaustiveSearcher(**kw)
    elif strategy == "direct":
        base = DirectSearcher(**kw)
    else:
        raise ValueError(
            f"Unknown strategy: {strategy!r}. "
            f"Expected one of: greedy, beam, multi_restart, exhaustive, direct."
        )
    if K_rollouts > 1:
        scales = (
            tuple(multirollout_scales)
            if multirollout_scales is not None
            else (0.0,) + (multirollout_scale,) * (K_rollouts - 1)
        )
        base = MultiRolloutSearcher(base, scales=scales)
    return base


__all__ = [
    "BeamSearcher",
    "DirectSearcher",
    "ExhaustiveSearcher",
    "GreedySearcher",
    "MultiRestartSearcher",
    "MultiRolloutSearcher",
    "Mode",
    "ScoreFn",
    "Searcher",
    "make_scorer_from_searcher",
    "make_searcher",
]
