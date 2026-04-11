"""Reference ``search_and_score`` implementation.

Composes the six framework primitives in the canonical loop from
``framework.tex`` §6.5. This is the **reference** implementation —
torch-ns and any new exhaustive baseline can use it directly.

DPrL keeps its own optimized scorers (``beam_scorer.py`` etc.) because
its compiled CUDA-graph closures cannot tolerate a generic Python loop.
That's fine: the architectural contract is that each method ships its
own ``queries -> scores`` function and reuses tkk primitives where
practical, NOT that every method calls ``search_and_score``.

Two control structures are supported:

  Exhaustive (one shot, used by SBR/DCR/R2N):
      ``select=ExhaustiveSelect()`` and ``max_depth=1``. The grounder
      produces all evidence in a single ``resolve`` call; the loop runs
      one iteration and the trajectory reduction happens via
      ``traj_repr.forward`` (batch interface) over the depth-stacked
      evidence.

  Sequential (multi-step, used by samplers):
      A non-exhaustive ``select`` and ``max_depth > 1``. ``traj_repr``
      uses the incremental ``init/step`` interface to accumulate per
      depth step. The loop terminates either at ``max_depth`` or when
      every branch is dead.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from torch import Tensor

from .protocols import AtomRepr, QueryRepr, ResolutionOp, Select, StateRepr, TrajRepr
from .repr import Repr
from .types import ProofEvidence, ProofState


def _atoms_from_evidence(evidence: ProofEvidence):
    """Split ``evidence.body`` into ``(preds, subjs, objs)`` index tensors."""
    body = evidence.body                                 # [B, C, D, M, 3] or [B, C, G, 3]
    preds = body[..., 0]
    subjs = body[..., 1]
    objs = body[..., 2]
    return preds, subjs, objs


def search_and_score(
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
    """Reference scorer composing the six framework primitives.

    Args:
        query: Initial query (consumed by ``resolve`` if ``initial_state``
            is None — the resolution op is responsible for converting
            queries into ProofStates).
        resolve: Resolution operator (typically a grounder).
        atom_repr / state_repr / traj_repr / query_repr / select: framework primitives.
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
        raise RuntimeError("search_and_score produced no accumulator (max_depth must be >= 1)")

    out = query_repr(accum, final_evidence)
    if not out.has_scores:
        raise RuntimeError("query_repr must return Repr with scores")
    return out.scores


ScorerFn = Callable[[Any], Dict[str, Tensor]]


def build_scorer(
    *,
    resolve: ResolutionOp,
    atom_repr: AtomRepr,
    state_repr: StateRepr,
    select: Select,
    traj_repr: TrajRepr,
    query_repr: QueryRepr,
    model: Any,
    max_depth: int = 1,
    name: str = "default",
) -> ScorerFn:
    """Factory: closes over the framework primitives and returns a scorer.

    The returned function takes a query batch and returns a dict mapping
    a label to a ``[B]`` score tensor — matching the calling convention
    expected by ``kge_kernels.eval.Evaluator``.
    """

    def scorer(queries: Any) -> Dict[str, Tensor]:
        scores = search_and_score(
            queries,
            resolve=resolve,
            atom_repr=atom_repr,
            state_repr=state_repr,
            select=select,
            traj_repr=traj_repr,
            query_repr=query_repr,
            model=model,
            max_depth=max_depth,
        )
        return {name: scores}

    return scorer


__all__ = ["ScorerFn", "build_scorer", "search_and_score"]
