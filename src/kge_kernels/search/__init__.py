"""High-level Searcher compositions for proof-based query ranking.

Per framework.pdf §11, every method is a 6-tuple
``(resolve, atom_repr, state_repr, select, traj_repr, query_repr)``.
The unified :class:`UnifiedSearcher` (see ``searcher.py``) composes
that 6-tuple plus a :class:`SearchSpec` and a capture-mode flag. The
:func:`make_searcher` factory below dispatches on a string strategy
name to choose the right :class:`Select` and wire it into a
``UnifiedSearcher`` instance.

  - ``"exhaustive"``: ``ExhaustiveSelect`` + ``max_depth=1``
    (canonical for SBR / DCR / R2N).
  - ``"greedy"``: ``GreedySelect`` (DpRL eval default).
  - ``"beam"``: ``BeamSelect`` (requires ``beam_width=...``).
  - ``"multi_restart"``: N independent ``UnifiedSearcher`` instances,
    per-query max across restarts; the first restart is deterministic.

Higher-order Searchers:

  - :class:`MultiRolloutSearcher` — wraps any base Searcher; runs it K
    times with different Gumbel scales, takes the per-query max.
  - :class:`DirectSearcher` — baseline that bypasses search and scores
    triples directly via an :class:`AtomRepr`. Kept distinct from
    :class:`UnifiedSearcher` because it doesn't compose the 6-tuple.

The :class:`Searcher` Protocol and :func:`make_scorer_from_searcher`
adapter let any Searcher feed
:class:`kge_kernels.eval.RankingEvaluator` directly.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

from ..framework.select import (
    BeamSelect,
    ExhaustiveSelect,
    GreedySelect,
    StateFactory,
)
from .direct import DirectSearcher
from .multi_restart import MultiRestartSearcher
from .multirollout import MultiRolloutSearcher
from .searcher import (
    CaptureMode,
    Mode,
    ScoreFn,
    Searcher,
    SearchSpec,
    UnifiedSearcher,
    make_scorer_from_searcher,
)


def _build_unified(
    *,
    select,
    spec: Optional[SearchSpec],
    batch_size: Optional[int],
    capture: CaptureMode,
    name: str,
    **kw: Any,
) -> UnifiedSearcher:
    """Internal helper: construct a ``UnifiedSearcher`` from raw kwargs.

    ``spec`` may be supplied directly; otherwise a default
    :class:`SearchSpec` is built from ``batch_size`` (required when
    ``spec`` is None and ``capture="static"``).
    """
    if spec is None:
        if batch_size is None:
            # Dynamic-mode default; static-mode raises later if reached.
            spec = SearchSpec(batch_size=1)
        else:
            spec = SearchSpec(batch_size=batch_size)
    return UnifiedSearcher(
        select=select,
        spec=spec,
        capture=capture,
        name=name,
        **kw,
    )


def make_searcher(
    strategy: str,
    *,
    K_rollouts: int = 1,
    multirollout_scales: Optional[Sequence[float]] = None,
    multirollout_scale: float = 0.3,
    capture: CaptureMode = "dynamic",
    spec: Optional[SearchSpec] = None,
    batch_size: Optional[int] = None,
    state_factory: Optional[StateFactory] = None,
    gumbel_scale_buf: Optional[Any] = None,
    beam_width: int = 1,
    n_restarts: int = 3,
    noise_scale: float = 0.1,
    max_depth: int = 20,
    name: Optional[str] = None,
    **kw: Any,
) -> Searcher:
    """Factory: build a :class:`Searcher` by string name.

    Recognized strategies:
      - ``"exhaustive"``: ``UnifiedSearcher`` with ``ExhaustiveSelect``.
      - ``"greedy"``: ``UnifiedSearcher`` with ``GreedySelect``.
      - ``"beam"``: ``UnifiedSearcher`` with ``BeamSelect`` (requires ``beam_width``).
      - ``"multi_restart"``: :class:`MultiRestartSearcher` over N
        beam-equipped ``UnifiedSearcher`` instances.
      - ``"direct"``: :class:`DirectSearcher` (baseline).

    When ``K_rollouts > 1``, the result is auto-wrapped in
    :class:`MultiRolloutSearcher`. ``multirollout_scales`` overrides
    the auto-built ``[0.0, scale, scale, ...]`` schedule.

    ``capture`` defaults to ``"dynamic"`` for the factory to keep
    eager-mode behavior; pass ``capture="static"`` to opt into
    CUDA-graph cheap-replay (requires ``spec`` or ``batch_size``).
    """
    if strategy == "exhaustive":
        from_kw = {**kw}
        if "max_depth" in from_kw:
            from_kw.pop("max_depth")  # exhaustive forces max_depth=1
        if spec is None and batch_size is not None:
            spec = SearchSpec(batch_size=batch_size, max_depth=1)
        elif spec is not None and spec.max_depth != 1:
            spec = SearchSpec(
                batch_size=spec.batch_size,
                max_depth=1,
                pool_size=spec.pool_size,
                n_corruptions=spec.n_corruptions,
                beam_width=spec.beam_width,
                embed_dim=spec.embed_dim,
                max_P=spec.max_P,
                max_D=spec.max_D,
                max_M=spec.max_M,
            )
        base: Searcher = _build_unified(
            select=ExhaustiveSelect(),
            spec=spec,
            batch_size=batch_size,
            capture=capture,
            name=name or "exhaustive",
            **from_kw,
        )
    elif strategy == "greedy":
        base = _build_unified(
            select=GreedySelect(state_factory=state_factory, gumbel_scale_buf=gumbel_scale_buf),
            spec=spec or (
                SearchSpec(batch_size=batch_size, max_depth=max_depth)
                if batch_size is not None else None
            ),
            batch_size=batch_size,
            capture=capture,
            name=name or "greedy",
            **kw,
        )
    elif strategy == "beam":
        base = _build_unified(
            select=BeamSelect(
                k=beam_width,
                state_factory=state_factory,
                gumbel_scale_buf=gumbel_scale_buf,
            ),
            spec=spec or (
                SearchSpec(batch_size=batch_size, max_depth=max_depth, beam_width=beam_width)
                if batch_size is not None else None
            ),
            batch_size=batch_size,
            capture=capture,
            name=name or "beam",
            **kw,
        )
    elif strategy == "multi_restart":
        base = MultiRestartSearcher(
            beam_width=beam_width,
            n_restarts=n_restarts,
            noise_scale=noise_scale,
            max_depth=max_depth,
            capture=capture,
            spec=spec,
            batch_size=batch_size,
            state_factory=state_factory,
            name=name or "multi_restart",
            **kw,
        )
    elif strategy == "direct":
        base = DirectSearcher(name=name or "direct", **kw)
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
    "CaptureMode",
    "DirectSearcher",
    "Mode",
    "MultiRestartSearcher",
    "MultiRolloutSearcher",
    "ScoreFn",
    "SearchSpec",
    "Searcher",
    "UnifiedSearcher",
    "make_scorer_from_searcher",
    "make_searcher",
]
