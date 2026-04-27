"""Searcher protocol + ScoreFn type + reshape adapter + unified Searcher class.

The framework PDF §6 specifies the canonical scoring-loop output as
``Dict[str, Tensor]`` (per-query, per-mode score). tkk's
:class:`~kge_kernels.eval.RankingEvaluator` consumes a different
contract: ``ScoreFn(q_buf, pool_buf, mode) -> [B, C]`` (per-pool-entry
score). The two contracts are reconciled by one tiny adapter,
:func:`make_scorer_from_searcher`, which performs the K-major flat-pool
reshape outside any compile boundary.

This module also hosts the :class:`UnifiedSearcher` class introduced in
Phase 4 of the unified-Searcher migration. The reference per-strategy
classes (``ExhaustiveSearcher``, ``GreedySearcher``, ``BeamSearcher``,
``DirectSearcher``) live in their own modules during the migration and
are deleted in Phase 5 in favor of :class:`UnifiedSearcher`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Literal,
    Optional,
    Protocol,
    runtime_checkable,
)

import torch
import torch.nn as nn
from torch import Tensor

from ..framework.protocols import (
    AtomRepr,
    QueryRepr,
    ResolutionOp,
    Select,
    StateRepr,
    TrajRepr,
)
from ..framework.scorer import search_and_score
from ..framework.select import ExhaustiveSelect

Mode = Literal["head", "tail"]
ScoreFn = Callable[[Tensor, Tensor, Mode], Tensor]
CaptureMode = Literal["static", "dynamic"]


@runtime_checkable
class Searcher(Protocol):
    """Per-query search strategy returning ``{mode_name: [N]}``.

    Concrete classes compose framework primitives in a
    ``search_and_score``-equivalent loop. They may also use
    method-specific specializations (e.g., DpRL's compiled CUDA-graph
    rollout) — the Protocol only requires the callable signature.

    .. note::
       Phase 4+ introduces :class:`UnifiedSearcher` as the canonical
       implementation. Phase 5 deletes the per-strategy reference
       classes and renames ``UnifiedSearcher`` → ``Searcher``.
    """

    def __call__(self, queries: Tensor) -> Dict[str, Tensor]: ...


def make_scorer_from_searcher(searcher: Searcher, mode_key: str) -> ScoreFn:
    """Reshape adapter: ``Searcher`` → tkk ``ScoreFn``.

    ``RankingEvaluator`` calls ``ScoreFn(q_buf, pool_buf, mode)`` per
    chunk × per mode. ``q_buf [B, 3]`` is a batch of (rel, head, tail)
    queries; ``pool_buf [B, C]`` is the candidate-entity pool per
    query. We construct the K-major flat triple pool and feed it to
    the searcher as if each pool slot were an independent query, then
    reshape the result back to ``[B, C]``.

    Pure tensor ops, eager. Not part of the compile boundary.
    """
    @torch.no_grad()
    def scorer(q_buf: Tensor, pool_buf: Tensor, mode: Mode) -> Tensor:
        B, C = pool_buf.shape
        if mode == "head":
            col = 1
        elif mode == "tail":
            col = 2
        else:
            raise ValueError(f"mode must be 'head' or 'tail', got {mode}")
        triples = q_buf.unsqueeze(0).expand(C, B, 3).clone()
        triples[:, :, col] = pool_buf.t()
        flat = triples.reshape(C * B, 3)              # K-major: index [k*B + b]
        result = searcher(flat)
        if mode_key not in result:
            raise KeyError(
                f"searcher did not return '{mode_key}' in its result dict; "
                f"keys: {sorted(result.keys())}"
            )
        return result[mode_key].view(C, B).t().contiguous()
    return scorer


@dataclass(frozen=True)
class SearchSpec:
    """Frozen sizing for a :class:`UnifiedSearcher` instance.

    A :class:`SearchSpec` is part of the searcher's identity. Shape
    changes require a new instance. Mid-life mutation is reserved for
    scalar / index buffers via primitive-level setters
    (``select.set_gumbel_scale``, ``resolve.configure``).

    Worst-case grounder evidence shape (``max_P``, ``max_D``, ``max_M``)
    matters only for ``capture="static"`` — captured-graph buffers are
    sized to these and runtime exceedance raises loudly.
    """

    batch_size: int
    max_depth: int = 1
    pool_size: int = 1
    n_corruptions: int = 1
    beam_width: int = 1
    embed_dim: Optional[int] = None
    max_P: int = 64
    max_D: int = 8
    max_M: int = 4


def _needs_alternation(select: Select) -> bool:
    """Sequential Selects need an A/B compiled-step pair for double-buffering."""
    return not isinstance(select, ExhaustiveSelect)


class UnifiedSearcher(nn.Module):
    """Unified Searcher with capture-mode dispatch.

    Composes the six framework primitives plus a :class:`SearchSpec` and
    a :data:`CaptureMode` flag. ``capture="static"`` allocates
    static-address buffers per ``SearchSpec`` and compiles the inner
    step via ``torch.compile(mode="reduce-overhead", fullgraph=True)``;
    ``capture="dynamic"`` runs the eager
    :func:`framework.scorer.search_and_score` loop with no buffer
    allocation. Both expose the same ``__call__`` contract.

    Mid-life mutable state lives on the primitives, not on the searcher::

        searcher.select.set_gumbel_scale(0.3)
        searcher.resolve.configure(n_corruptions=5)

    The :meth:`set_gumbel_scale` and :meth:`configure` methods on
    UnifiedSearcher are convenience forwarders that delegate to the
    primitives; setting on the primitive directly works equivalently.
    """

    _graph_cache: ClassVar[Dict[tuple, Callable]] = {}

    def __init__(
        self,
        *,
        resolve: ResolutionOp,
        atom_repr: AtomRepr,
        state_repr: StateRepr,
        select: Select,
        traj_repr: TrajRepr,
        query_repr: QueryRepr,
        spec: SearchSpec,
        model: Any = None,
        name: str = "default",
        capture: CaptureMode = "static",
    ) -> None:
        super().__init__()
        self.resolve = resolve
        self.atom_repr = atom_repr
        self.state_repr = state_repr
        self.select = select
        self.traj_repr = traj_repr
        self.query_repr = query_repr
        self.model = model
        self.spec = spec
        self.name = name
        self.capture: CaptureMode = capture
        self._needs_alternation = _needs_alternation(select)
        # Two static-mode shapes:
        #  * pool-aware: resolve owns prepare/run/extract_summaries, returns
        #    an alternated (step_ab, step_ba) compiled pair.
        #  * generic:    resolve is a function-style ResolutionOp, the entire
        #    search_and_score loop is torch.compile'd into one closure.
        self._compiled_step: Optional[Callable] = None
        self._compiled_steps: Optional[Tuple[Callable, Callable]] = None
        self._pool_aware = (
            hasattr(resolve, "build_compiled_steps")
            and hasattr(resolve, "prepare")
            and hasattr(resolve, "run")
            and hasattr(resolve, "extract_summaries")
        )
        if capture == "static":
            self._build_static_path()

    @torch.no_grad()
    def __call__(self, queries: Tensor) -> Dict[str, Tensor]:
        if self._pool_aware and self.capture == "static":
            return self._call_pool_aware(queries)
        if queries.shape[0] != self.spec.batch_size:
            if self.capture == "static":
                raise ValueError(
                    f"queries.shape[0]={queries.shape[0]} != "
                    f"spec.batch_size={self.spec.batch_size} "
                    f"(static-capture Searcher cannot reshape on the fly; "
                    f"build a new Searcher with the right SearchSpec)"
                )
        if self.capture == "static":
            return self._call_static(queries)
        return self._call_dynamic(queries)

    def _call_dynamic(self, queries: Tensor) -> Dict[str, Tensor]:
        scores = search_and_score(
            queries,
            resolve=self.resolve,
            atom_repr=self.atom_repr,
            state_repr=self.state_repr,
            select=self.select,
            traj_repr=self.traj_repr,
            query_repr=self.query_repr,
            model=self.model,
            max_depth=self.spec.max_depth,
        )
        return {self.name: scores}

    def _call_static(self, queries: Tensor) -> Dict[str, Tensor]:
        assert self._compiled_step is not None, "static path not initialized"
        scores = self._compiled_step(queries)
        return {self.name: scores}

    def _call_pool_aware(self, queries: Tensor) -> Dict[str, Tensor]:
        """Pool-aware static path.

        Delegates pool-allocation + initial-state setup + replay loop +
        summary extraction to the resolve primitive (which owns those
        buffers). The query_repr converts trajectory summaries to the
        per-query scalar score.

        The input ``queries`` are also threaded into ``summaries`` under
        the ``"queries"`` key so query_reprs that need the original
        triples (e.g. external-KGE bridge combinations) can read them
        without reaching into the resolve primitive.
        """
        from ..framework import Repr  # local import to avoid cycles

        assert self._compiled_steps is not None, "pool-aware path not initialized"
        N = self.resolve.prepare(queries)
        self.resolve.run(N, self._compiled_steps)
        summaries = self.resolve.extract_summaries(N)
        summaries["queries"] = queries
        scores = self.query_repr(Repr(summaries=summaries), evidence=None).scores
        return {self.name: scores}

    def _build_static_path(self) -> None:
        """Compile the inner search step.

        If the resolve primitive owns its own buffer alternation
        (:class:`StatefulEnvResolve` does, because policy-rollout needs
        explicit cur_buf/next_buf alternation + cudagraph_mark_step_begin
        replay), it builds the alternated pair via
        :meth:`build_compiled_steps`. Otherwise traces
        ``search_and_score`` through ``torch.compile(fullgraph=True)``.
        """
        if self._pool_aware:
            self._compiled_steps = self.resolve.build_compiled_steps(
                select=self.select,
                traj_repr=self.traj_repr,
                spec=self.spec,
            )
            return

        cache_key = self._cache_key()
        cached = self._graph_cache.get(cache_key)
        if cached is not None:
            self._compiled_step = cached
            return

        spec = self.spec
        resolve = self.resolve
        atom_repr = self.atom_repr
        state_repr = self.state_repr
        select = self.select
        traj_repr = self.traj_repr
        query_repr = self.query_repr
        model = self.model

        @torch.no_grad()
        def inner(queries: Tensor) -> Tensor:
            return search_and_score(
                queries,
                resolve=resolve,
                atom_repr=atom_repr,
                state_repr=state_repr,
                select=select,
                traj_repr=traj_repr,
                query_repr=query_repr,
                model=model,
                max_depth=spec.max_depth,
            )

        compile_mode = "reduce-overhead" if torch.cuda.is_available() else "default"
        try:
            compiled = torch.compile(inner, mode=compile_mode, fullgraph=True)
        except Exception:
            # Compilation may not always succeed on all primitive combos.
            compiled = inner
        self._graph_cache[cache_key] = compiled
        self._compiled_step = compiled

    def _cache_key(self) -> tuple:
        return (
            self.spec,
            type(self.resolve).__name__,
            type(self.atom_repr).__name__,
            type(self.state_repr).__name__,
            type(self.select).__name__,
            type(self.traj_repr).__name__,
            type(self.query_repr).__name__,
        )

    def set_gumbel_scale(self, scale: float) -> None:
        """Forward to ``self.select.set_gumbel_scale`` if available.

        Convenience method — equivalent to calling
        ``searcher.select.set_gumbel_scale(scale)`` directly.
        """
        fn = getattr(self.select, "set_gumbel_scale", None)
        if fn is not None:
            fn(scale)

    def configure(self, **kwargs: Any) -> None:
        """Forward to ``self.resolve.configure`` if available.

        Convenience method — equivalent to calling
        ``searcher.resolve.configure(**kwargs)`` directly.
        """
        fn = getattr(self.resolve, "configure", None)
        if fn is not None:
            fn(**kwargs)

    def reset_stats(self) -> None:
        """Forward to ``self.resolve.reset_stats`` if available."""
        fn = getattr(self.resolve, "reset_stats", None)
        if fn is not None:
            fn()

    def aggregate_stats(self, **kwargs: Any) -> Dict[str, Any]:
        """Forward to ``self.resolve.aggregate_stats``; empty dict if absent."""
        fn = getattr(self.resolve, "aggregate_stats", None)
        if fn is not None:
            return fn(**kwargs)
        return {}


__all__ = [
    "CaptureMode",
    "Mode",
    "ScoreFn",
    "SearchSpec",
    "Searcher",
    "UnifiedSearcher",
    "make_scorer_from_searcher",
]
