"""Protocols for the six framework primitives.

These are ``typing.Protocol`` classes (NOT abstract base classes) so that
implementations in any repo can satisfy them via duck typing without
needing to import or subclass anything from tkk. The signatures match
``framework.tex`` §5–§7 exactly.

Six slots:
  ResolutionOp     — produces ProofEvidence (proof path) or RuleGroundings
                     (rule path) from a ProofState
  AtomRepr         — atoms → Repr (per atom)
  StateRepr        — proof path: atom Repr + evidence → Repr (per state)
  RuleStateRepr    — rule path: per-firing operator (body_emb, rule_idx) → preds
  ProofTrajRepr    — proof path: per-trajectory accumulator (init/step/finalize)
  RuleTrajRepr     — rule path: K-iteration pool loop with shared atom pool
  QueryRepr        — traj Repr / pool + evidence → Repr (per query, scalar)
  Select           — evidence + state Repr → next ProofState (+ info)

Concrete implementations live in:
  repr_atom.py, repr_state.py, repr_state_rule.py,
  repr_traj.py, repr_traj_rule.py,
  repr_query.py, repr_query_pool.py, select.py

The reference compositions live in :class:`kge_kernels.search.ProofScorer`
(proof path) and :func:`kge_kernels.search.searcher._rule_loop`
(rule path).

Cross-repo Protocol satisfiers
------------------------------

tkk stays env- and policy-unaware. Concrete primitives that need a
policy network or a stateful environment live in DpRL and satisfy
these Protocols via duck typing:

  * ``DpRL.kge_experiments.ppo.policy_select.PolicySelect`` satisfies
    :class:`Select`. Reads ``evidence.obs`` (the optional Protocol
    field) to feed a policy net. Exposes ``set_gumbel_scale``.

  * ``DpRL.kge_experiments.env.stateful_resolve.StatefulEnvResolve``
    satisfies :class:`ResolutionOp`. Holds a stateful ``EnvVec`` plus
    alternated state buffers; populates ``evidence.obs`` so a paired
    ``PolicySelect`` can read it.

Dependency arrow stays one-way: ``DpRL → tkk → grounder``. tkk imports
neither DpRL nor any env/policy code; DpRL imports tkk's Protocols only
for type hints (no runtime dependency on tkk's class hierarchy).
"""
from __future__ import annotations

from typing import Optional, Protocol, Tuple

from torch import Tensor

from .repr import Repr
from .types import FiringsTensors, ProofEvidence, ProofState, SelectInfo


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
    the inputs (typically ``[B, P, D, M]`` or ``[B, P, G_body]``).

    ``model`` is the tkk-native KGE model (anything inheriting from
    ``kge_kernels.models.base.KGEBase``). It is passed positionally so
    the same primitive can be reused with different models without
    subclassing.
    """

    def __call__(self, preds: Tensor, subjs: Tensor, objs: Tensor, model) -> Repr: ...


class StateRepr(Protocol):
    """Aggregates per-atom ``Repr`` into per-state (per-body) ``Repr``.

    Reads ``evidence.body_atom_mask_flat`` (or ``evidence.body_count``) to
    decide which atoms are valid in each body. Output leading shape is
    ``[B, P]`` (one Repr per candidate proof) or ``[B, P, D]`` (one Repr
    per depth step) depending on whether the implementation reduces over
    the depth dim.
    """

    def __call__(self, atom_repr: Repr, evidence: ProofEvidence) -> Repr: ...


class ProofTrajRepr(Protocol):
    """Per-trajectory accumulator for the proof-based reasoning path.

    Used by sequential search methods (DpRL beam/greedy/sample/PPO) that
    accumulate one depth step at a time along a single trajectory. The
    accumulator type is implementation-defined (scalar, embedding, dict
    of named statistics, etc.) and is carried through ``init/step/finalize``.

    Contract::

        accum = traj.init(B, device)
        for d in range(max_depth):
            evidence = resolve(state)
            ...
            accum = traj.step(accum, s_repr, info)
            state, info = select(evidence, s_repr)
        traj_repr = traj.finalize(accum)
    """

    def init(self, B: int, device) -> Repr: ...

    def step(self, accum: Repr, state_repr: Repr, info: Optional[SelectInfo]) -> Repr: ...

    def finalize(self, accum: Repr) -> Repr: ...


class RuleStateRepr(Protocol):
    """Per-firing rule operator for the rule-based reasoning path.

    Takes flat per-firing tensors and returns the firing's predicted
    output atom embeddings/scores.

    Args:
        body_emb: ``[N_f, M*E]`` (or ``[N_f, M]`` for scalar pools) —
                  body atoms of each firing, concatenated along the body
                  dim. Caller is responsible for reshaping from
                  ``[N_f, M, E]``.
        rule_idx: ``[N_f]`` rule index per firing.

    Returns:
        ``[N_f, K_out, E]`` (or ``[N_f, K_out]`` for scalar pools)
        — predicted output for each output atom of each firing.
        ``K_out = 1`` for ``'head'`` prediction (head atom only) or
        ``K_out = M+1`` for ``'full'`` prediction (head + body atoms,
        the resnet pattern).
    """

    def __call__(self, body_emb: Tensor, rule_idx: Tensor) -> Tensor: ...


class RuleTrajRepr(Protocol):
    """K-iteration pool-iter loop for the rule-based reasoning path.

    The accumulator IS the deduplicated atom pool ``[N_pool, E]`` (or
    ``[N_pool]`` for scalar pools). Each iteration gathers body atom
    embeddings from the pool, applies a per-firing rule operator
    (``RuleStateRepr``), and scatters the predictions back into the pool
    via ``scatter_max``. After ``K`` iterations the pool encodes the
    K-step closure of the rules over the seed atom representations.

    Used by exhaustive methods (SBR / DCR / R2N) where the grounder
    produces all evidence in one shot via ``BCGrounder.run_bc``.

    Statically known ``K`` keeps the loop compile-unroll-friendly.

    Args:
        pool: ``[N_pool, E]`` initial pool (typically from atom_repr
              over the deduplicated atom_table).
        firings: flat per-firing tensors keyed by pool indices.
        state_repr_fn: per-firing rule operator (``RuleStateRepr``).

    Returns:
        ``(pool, ever_written)`` — updated pool ``[N_pool, E]`` and
        boolean ``ever_written [N_pool]`` mask of which slots were the
        target of any valid scatter during the K iterations. Query slots
        with ``ever_written=False`` should be tied at a constant score
        by the downstream :class:`QueryRepr` (matches keras-ns R2N's
        "no-grounding → tied last" semantic).
    """

    K: int

    def __call__(
        self,
        pool: Tensor,
        firings: FiringsTensors,
        state_repr_fn: RuleStateRepr,
    ) -> tuple[Tensor, Tensor]: ...


# ── Backwards-compat alias ────────────────────────────────────────────
# DpRL still imports ``TrajRepr`` (the proof-path Protocol). Step 4 of
# the cascade swaps DpRL's imports; until then this alias keeps things
# working without touching DpRL.
TrajRepr = ProofTrajRepr


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
    "ProofTrajRepr",
    "QueryRepr",
    "ResolutionOp",
    "RuleStateRepr",
    "RuleTrajRepr",
    "Select",
    "StateRepr",
    "TrajRepr",
]
