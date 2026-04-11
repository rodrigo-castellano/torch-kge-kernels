"""Framework-level types: ProofState / ProofEvidence Protocols + SelectInfo.

``ProofState`` and ``ProofEvidence`` are defined here as ``typing.Protocol``
classes so that ``torch-kge-kernels`` does NOT need to import any concrete
type from ``grounder``. ``grounder``'s existing dataclasses (see
``~/repos/grounder-swarm/main/types.py``) satisfy these protocols by duck
typing — no inheritance, no conversion, zero runtime cost.

This keeps the dependency graph clean: ``grounder`` depends on tkk, never
the other way around. Tests that need a fake evidence object can ship a
tiny dataclass that implements the protocol attributes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

from torch import Tensor

# ═══════════════════════════════════════════════════════════════════════
# Protocols satisfied by grounder.ProofState / grounder.ProofEvidence
# ═══════════════════════════════════════════════════════════════════════


@runtime_checkable
class ProofState(Protocol):
    """Snapshot of the proof search after the last resolution step.

    Field layout matches ``grounder.types.ProofState`` exactly so that the
    grounder dataclass satisfies this protocol via duck typing.
    """

    proof_goals: Tensor              # [B, S, G, 3]
    state_valid: Tensor              # [B, S]
    top_ridx: Tensor                 # [B, S]
    next_var_indices: Optional[Tensor]  # [B] or None


@runtime_checkable
class ProofEvidence(Protocol):
    """Accumulated proof trace from completed groundings.

    Field layout matches ``grounder.types.ProofEvidence`` exactly. The
    structured layout uses ``D > 0`` (depth dim populated) and the legacy
    flat layout uses ``D == 0``. Properties allow consumers to ignore the
    distinction.
    """

    body: Tensor          # [B, C, D, M, 3] or [B, C, G_body, 3]
    mask: Tensor          # [B, C]
    count: Tensor         # [B]
    rule_idx: Tensor      # [B, C, D] or [B, C]
    body_count: Tensor    # [B, C, D] or [B, C]
    D: int
    M: int
    head: Optional[Tensor]  # [B, C, D, 3] or None

    @property
    def body_flat(self) -> Tensor: ...

    @property
    def rule_idx_top(self) -> Tensor: ...

    @property
    def body_count_total(self) -> Tensor: ...

    @property
    def body_atom_mask_flat(self) -> Tensor: ...


# ═══════════════════════════════════════════════════════════════════════
# Concrete dataclass — owned by tkk (nothing else ships one)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class SelectInfo:
    """Optional metadata returned alongside a ``Select`` step.

    ``ExhaustiveSelect`` returns ``None``. Greedy/beam variants populate
    ``chosen_scores``. Sampling variants populate ``log_probs`` for policy
    gradient. The dataclass is intentionally extensible: MCTS / Q-learning
    subclasses can add fields without breaking existing consumers.
    """

    log_probs: Optional[Tensor] = None       # [B, S] policy gradient
    chosen_scores: Optional[Tensor] = None   # [B, k] beam/greedy
    chosen_indices: Optional[Tensor] = None  # [B, k] which action was picked
    extra: dict = field(default_factory=dict)  # MCTS visit counts etc.


__all__ = ["ProofState", "ProofEvidence", "SelectInfo"]
