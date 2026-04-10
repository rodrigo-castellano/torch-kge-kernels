"""Protocols for the six framework primitives.

These are ``typing.Protocol`` classes (NOT abstract base classes) so that
implementations in any repo can satisfy them via duck typing without
needing to import or subclass anything from tkk. The signatures match
``framework.tex`` §5–§7 exactly.

Six slots:
  ResolutionOp     — produces ProofEvidence from a ProofState
  AtomRepr         — atoms → Repr (per atom)
  StateRepr        — atom Repr + evidence → Repr (per state / per body)
  TrajRepr         — state Repr + evidence → Repr (per proof / trajectory)
  QueryRepr        — traj Repr + evidence → Repr (per query, scalar)
  Select           — evidence + state Repr → next ProofState (+ info)

Concrete implementations live in:
  atom_repr.py, state_repr.py, traj_repr.py, query_repr.py, select.py

The reference composition lives in scorer.py (``search_and_score``).
"""
from __future__ import annotations

from typing import Optional, Protocol, Tuple

from torch import Tensor

from .repr import Repr
from .types import ProofEvidence, ProofState, SelectInfo


class ResolutionOp(Protocol):
    """Produces successor states from a current proof state.

    Implementations live in ``grounder`` (e.g. ``grounder.bc.grounder``).
    Returned ``ProofEvidence`` carries the body atoms gathered along the
    resolution branches.
    """

    def __call__(self, state: ProofState) -> ProofEvidence: ...


class AtomRepr(Protocol):
    """Maps atom triples ``(pred, subj, obj)`` to per-atom ``Repr``.

    The leading shape of the output ``Repr`` matches the leading shape of
    the inputs (typically ``[B, C, D, M]`` or ``[B, C, G_body]``).

    ``model`` is whatever scoring backend the implementation needs (a
    ``KGEModel`` from ``kge_kernels.models`` or a ``KGEBackend``). It is
    passed positionally so the same primitive can be reused with different
    backends without subclassing.
    """

    def __call__(self, preds: Tensor, subjs: Tensor, objs: Tensor, model) -> Repr: ...


class StateRepr(Protocol):
    """Aggregates per-atom ``Repr`` into per-state (per-body) ``Repr``.

    Reads ``evidence.body_atom_mask_flat`` (or ``evidence.body_count``) to
    decide which atoms are valid in each body. Output leading shape is
    ``[B, C]`` (one Repr per candidate proof) or ``[B, C, D]`` (one Repr
    per depth step) depending on whether the implementation reduces over
    the depth dim.
    """

    def __call__(self, atom_repr: Repr, evidence: ProofEvidence) -> Repr: ...


class TrajRepr(Protocol):
    """Reduces per-state ``Repr`` into per-trajectory ``Repr``.

    Two interfaces are supported and MUST agree:

    1. Batch interface: ``forward(state_repr, evidence) -> Repr`` runs the
       reduction over the full evidence at once. Used by exhaustive
       methods (SBR/DCR/R2N) where the grounder produces all evidence
       in a single shot.

    2. Incremental interface: ``init(B, device)`` then repeated calls to
       ``step(accum, state_repr, info)``. Used by sequential methods
       (DPrL beam/greedy/sample) that accumulate one depth step at a time.

    Implementations should provide both. The ``forward`` ≡ ``init + step*D``
    parity is exercised by ``tests/framework/test_traj_repr.py``.
    """

    def __call__(self, state_repr: Repr, evidence: ProofEvidence) -> Repr: ...

    def init(self, B: int, device) -> Repr: ...

    def step(self, accum: Repr, state_repr: Repr, info: Optional[SelectInfo]) -> Repr: ...


class QueryRepr(Protocol):
    """Reduces per-trajectory ``Repr`` into a scalar score per query.

    Reads ``evidence.mask`` for proof-level validity (which candidate
    proofs are alive). Output ``Repr.scores`` has shape ``[B]``.
    """

    def __call__(self, traj_repr: Repr, evidence: ProofEvidence) -> Repr: ...


class Select(Protocol):
    """Picks next state(s) from the resolved successors.

    The exhaustive variant is the identity (``ExhaustiveSelect``); beam,
    greedy and sample variants prune to ``k`` paths. The returned
    ``SelectInfo`` is optional and carries log-probs / chosen scores for
    downstream consumers (PG loss, MCTS bookkeeping).
    """

    def __call__(
        self,
        evidence: ProofEvidence,
        s_repr: Repr,
    ) -> Tuple[ProofState, Optional[SelectInfo]]: ...


__all__ = [
    "AtomRepr",
    "QueryRepr",
    "ResolutionOp",
    "Select",
    "StateRepr",
    "TrajRepr",
]
