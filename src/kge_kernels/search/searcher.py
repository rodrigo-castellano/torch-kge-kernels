"""Searcher protocol + ScoreFn type + reshape adapter + :class:`ProofScorer`.

The framework PDF §6 specifies the canonical scoring-loop output as
``Dict[str, Tensor]`` (per-query, per-mode score). tkk's
:class:`~kge_kernels.eval.RankingEvaluator` consumes a different
contract: ``ScoreFn(q_buf, pool_buf, mode) -> [B, C]`` (per-pool-entry
score). The two contracts are reconciled by one tiny adapter,
:func:`make_scorer_from_searcher`, which performs the K-major flat-pool
reshape outside any compile boundary.

This module also hosts :class:`ProofScorer`: the canonical concrete
implementation of the framework's 6-tuple. The base class composes the
six primitives plus a :class:`SearchSpec` and a :data:`CaptureMode` flag,
exposes the :meth:`search_and_score` loop as a method, and provides a
minimal compilation contract — :meth:`_allocate_buffers` and
:meth:`_compile` hooks called from ``__init__`` — that subclasses (e.g.,
DpRL's PPO/Lookahead proof scorers) override to plug in specialized
rollouts without re-deriving the framework boilerplate.
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
    RuleStateRepr,
    RuleTrajRepr,
    Select,
    StateRepr,
    TrajRepr,
)
from ..framework.repr import Repr
from ..framework.types import (
    ProofEvidence,
    ProofState,
    RuleGroundings,
    build_firings_from_rule_groundings,
)

Mode = Literal["head", "tail"]
ScoreFn = Callable[[Tensor, Tensor, Mode], Tensor]
CaptureMode = Literal["static", "dynamic"]


@runtime_checkable
class Searcher(Protocol):
    """Per-query search strategy returning ``{mode_name: [N]}``.

    Concrete classes compose framework primitives in a
    :meth:`ProofScorer.search_and_score`-equivalent loop. They may also
    use method-specific specializations (e.g., DpRL's compiled
    CUDA-graph rollout) — the Protocol only requires the callable
    signature.
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
    """Frozen sizing for a :class:`ProofScorer` instance.

    A :class:`SearchSpec` is part of the scorer's identity. Shape
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


def _atoms_from_evidence(evidence: ProofEvidence):
    """Split ``evidence.body`` into ``(preds, subjs, objs)`` index tensors."""
    body = evidence.body                                 # [B, P, D, M, 3] or [B, P, G, 3]
    preds = body[..., 0]
    subjs = body[..., 1]
    objs = body[..., 2]
    return preds, subjs, objs


def _canonical_loop(
    query: Any,
    *,
    resolve: ResolutionOp,
    atom_repr: AtomRepr,
    state_repr: StateRepr,
    select: Select,
    traj_repr: TrajRepr,
    query_repr: QueryRepr,
    model: Any,
    max_depth: int = 1,
    initial_state: Optional[ProofState] = None,
) -> Tensor:
    """Reference scoring loop composing the six framework primitives.

    This is the canonical body referenced as ``framework.tex §6.5``.
    :class:`ProofScorer` calls it from :meth:`search_and_score` (eager
    or torch.compile'd via :meth:`_compile`). Subclasses with
    method-specific rollouts override :meth:`search_and_score` and
    therefore do not call this helper.

    Args:
        query: Initial query (consumed by ``resolve`` if ``initial_state``
            is None — the resolution op is responsible for converting
            queries into ProofStates).
        resolve / atom_repr / state_repr / select / traj_repr / query_repr:
            framework primitives.
        model: KGE model (or backend) passed positionally to ``atom_repr``.
        max_depth: Number of resolution steps. Use ``1`` for exhaustive
            (one-shot) scoring with ``ExhaustiveSelect``.
        initial_state: Optional pre-built ProofState. If None, ``resolve``
            is expected to accept ``query`` directly on the first call.

    Returns:
        ``[B]`` per-query scalar scores.
    """
    state = initial_state if initial_state is not None else query
    accum: Optional[Repr] = None
    final_evidence: Optional[ProofEvidence] = None

    for d in range(max_depth):
        evidence = resolve(state)
        final_evidence = evidence

        preds, subjs, objs = _atoms_from_evidence(evidence)
        a_repr = atom_repr(preds, subjs, objs, model)
        s_repr = state_repr(a_repr, evidence)

        next_state, info = select(evidence, s_repr)

        if d == 0 and max_depth == 1:
            # Exhaustive path: traj_repr reduces over the full depth dim
            # in a single batch call.
            accum = traj_repr(s_repr, evidence)
        else:
            if accum is None:
                B = s_repr.scores.shape[0] if s_repr.has_scores else s_repr.embeddings.shape[0]
                device = (s_repr.scores if s_repr.has_scores else s_repr.embeddings).device
                accum = traj_repr.init(B, device)
            accum = traj_repr.step(accum, s_repr, info)

        if next_state is None:
            break
        state = next_state

    if accum is None or final_evidence is None:
        raise RuntimeError("ProofScorer canonical loop produced no accumulator (max_depth must be >= 1)")

    out = query_repr(accum, final_evidence)
    if not out.has_scores:
        raise RuntimeError("query_repr must return Repr with scores")
    return out.scores


def _rule_loop(
    pool_init: Tensor,
    rule_groundings: RuleGroundings,
    *,
    traj_repr: RuleTrajRepr,
    state_repr_fn: RuleStateRepr,
    query_repr: Callable[[Tensor, Tensor, Tensor], Tensor],
    M_max: Optional[int] = None,
    pad_idx: int = 0,
) -> Tensor:
    """Reference scoring loop for the rule-based reasoning path.

    Runs the K-iteration pool-iter loop and gathers per-query scores
    from the final pool. Used by exhaustive methods (SBR / DCR / R2N
    via :func:`grounder.bc.bc.BCGrounder.run_bc`).

    Args:
        pool_init: ``[N_pool, E]`` (R2N) or ``[N_pool]`` (SBR/DCR) — the
            initial pool, typically ``atom_repr`` over
            ``rule_groundings.atom_table``.
        rule_groundings: object satisfying :class:`RuleGroundings`
            Protocol (e.g., grounder's :class:`RuleGroundings`
            dataclass with ``query_pool_idx`` populated).
        traj_repr: K-iteration pool loop (:class:`RuleTrajRepr`).
        state_repr_fn: per-firing rule operator (:class:`RuleStateRepr`).
        query_repr: callable ``(pool, query_pool_idx, ever_written) →
            [B]`` (e.g., :class:`LookupAtPool` or
            :class:`OutputLayerAtPool`).
        M_max: optional pad target body width; defaults to max-M over
            present rules in ``rule_groundings``.
        pad_idx: pool slot to use for invalid body atom positions
            (default 0 — the padding slot).

    Returns:
        ``[B]`` per-query scalar scores.
    """
    firings = build_firings_from_rule_groundings(
        rule_groundings, M_max=M_max, pad_idx=pad_idx,
    )
    pool, ever_written = traj_repr(pool_init, firings, state_repr_fn)
    if rule_groundings.query_pool_idx is None:
        raise RuntimeError(
            "_rule_loop requires rule_groundings.query_pool_idx populated. "
            "Use BCGrounder.run_bc to obtain RuleGroundings with queries pinned."
        )
    return query_repr(pool, rule_groundings.query_pool_idx, ever_written)


class ProofScorer(nn.Module):
    """Canonical concrete implementation of the framework's 6-tuple.

    Composes the six framework primitives plus a :class:`SearchSpec`
    and a :data:`CaptureMode` flag. ``capture="static"`` allocates
    static-address buffers and compiles :meth:`search_and_score` via
    ``torch.compile(mode="reduce-overhead", fullgraph=True)``;
    ``capture="dynamic"`` runs the eager :func:`_canonical_loop` with no
    buffer allocation. Both expose the same ``__call__`` contract.

    Mid-life mutable state lives on the primitives, not on the scorer::

        scorer.select.set_gumbel_scale(0.3)
        scorer.resolve.configure(n_corruptions=5)

    The :meth:`set_gumbel_scale` and :meth:`configure` methods on
    ``ProofScorer`` are convenience forwarders that delegate to the
    primitives; setting on the primitive directly works equivalently.

    Compilation contract — subclasses override these two hooks to plug
    in specialized rollouts:

    * :meth:`_allocate_buffers` — allocate persistent tensors owned by
      this scorer. Base: no-op (primitives own theirs).
    * :meth:`_compile` — build compiled bodies / closures. Base:
      :func:`torch.compile` over the canonical 6-tuple loop.

    Subclasses that fully override :meth:`search_and_score` (e.g.,
    DpRL's ``PPOProofScorer`` running an alternated-buffer rollout) may
    bypass the parent's compiled artifact entirely; they still benefit
    from the lifecycle hooks for inventory clarity.
    """

    _graph_cache: ClassVar[Dict[tuple, Callable]] = {}

    def __init__(
        self,
        *,
        spec: SearchSpec,
        name: str = "default",
        capture: CaptureMode = "static",
        resolve: Optional[ResolutionOp] = None,
        atom_repr: Optional[AtomRepr] = None,
        state_repr: Optional[StateRepr] = None,
        select: Optional[Select] = None,
        traj_repr: Optional[TrajRepr] = None,
        query_repr: Optional[QueryRepr] = None,
        model: Any = None,
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
        self._compiled_step: Optional[Callable] = None
        self._allocate_buffers()
        if capture == "static":
            self._compile()

    # ── Compilation contract — subclasses override ─────────────────────

    def _allocate_buffers(self) -> None:
        """Allocate persistent tensors owned by this scorer.

        Base: no-op (primitives own their own buffers). Override in
        subclasses to add scorer-level tensors (rollout state pair,
        result scratchpads, etc.).
        """
        pass

    def _compile(self) -> None:
        """Build compiled bodies / closures.

        Base: ``torch.compile`` the canonical 6-tuple loop. Requires
        all six primitives to be present. Override in subclasses with
        specialized compiled paths (e.g., alternated-buffer step pair).
        """
        if any(p is None for p in (
            self.resolve, self.atom_repr, self.state_repr,
            self.select, self.traj_repr, self.query_repr,
        )):
            # Subclass that overrides search_and_score may pass None for
            # primitives the base class would compose; their _compile
            # runs the specialized path instead. Nothing to do here.
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
            return _canonical_loop(
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

    # ── Search loop ────────────────────────────────────────────────────

    @torch.no_grad()
    def __call__(self, queries: Tensor) -> Dict[str, Tensor]:
        return {self.name: self.search_and_score(queries)}

    def search_and_score(self, queries: Tensor) -> Tensor:
        """Run the canonical 6-tuple scoring loop.

        Default: dispatches between the compiled artifact (static) and
        the eager :func:`_canonical_loop` body (dynamic). Subclasses
        override entirely for method-specific rollouts.
        """
        if self.capture == "static":
            if queries.shape[0] != self.spec.batch_size:
                raise ValueError(
                    f"queries.shape[0]={queries.shape[0]} != "
                    f"spec.batch_size={self.spec.batch_size} "
                    f"(static-capture ProofScorer cannot reshape on the fly; "
                    f"build a new ProofScorer with the right SearchSpec)"
                )
            assert self._compiled_step is not None, "static path not initialized"
            return self._compiled_step(queries)
        return _canonical_loop(
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

    # ── Mid-life convenience forwarders ────────────────────────────────

    def set_gumbel_scale(self, scale: float) -> None:
        """Forward to ``self.select.set_gumbel_scale`` if available.

        Convenience method — equivalent to calling
        ``scorer.select.set_gumbel_scale(scale)`` directly.
        """
        fn = getattr(self.select, "set_gumbel_scale", None) if self.select is not None else None
        if fn is not None:
            fn(scale)

    def configure(self, **kwargs: Any) -> None:
        """Forward to ``self.resolve.configure`` if available.

        Convenience method — equivalent to calling
        ``scorer.resolve.configure(**kwargs)`` directly.
        """
        fn = getattr(self.resolve, "configure", None) if self.resolve is not None else None
        if fn is not None:
            fn(**kwargs)

    def reset_stats(self) -> None:
        """Forward to ``self.resolve.reset_stats`` if available."""
        fn = getattr(self.resolve, "reset_stats", None) if self.resolve is not None else None
        if fn is not None:
            fn()

    def aggregate_stats(self, **kwargs: Any) -> Dict[str, Any]:
        """Forward to ``self.resolve.aggregate_stats``; empty dict if absent."""
        fn = getattr(self.resolve, "aggregate_stats", None) if self.resolve is not None else None
        if fn is not None:
            return fn(**kwargs)
        return {}


__all__ = [
    "CaptureMode",
    "Mode",
    "ProofScorer",
    "ScoreFn",
    "SearchSpec",
    "Searcher",
    "_rule_loop",
    "make_scorer_from_searcher",
]
