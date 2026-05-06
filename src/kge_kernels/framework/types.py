"""Framework-level types: ProofState / ProofEvidence Protocols + SelectInfo.

``ProofState`` and ``ProofEvidence`` are defined here as ``typing.Protocol``
classes so that ``torch-kge-kernels`` does NOT need to import any concrete
type from ``grounder``. ``grounder``'s existing dataclasses (see
``~/repos/grounder-swarm/main/types.py``) satisfy these protocols by duck
typing — no inheritance, no conversion, zero runtime cost.

This keeps the dependency graph clean: ``grounder`` depends on tkk, never
the other way around. Tests that need a fake evidence object can ship a
tiny dataclass that implements the protocol attributes.

Rule-evidence types live here too — ``RuleGroundings`` (Protocol matching
grounder's ``run_bc`` output) and ``FiringsTensors`` (flat per-firing
layout consumed by the pool-iter loop). These are the data carriers for
the rule-based reasoning path (SBR / DCR / R2N exhaustive); the proof
path uses ``ProofEvidence`` instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, runtime_checkable

import torch
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

    proof_goals: Tensor              # [B, G, A, 3]
    state_valid: Tensor              # [B, G]
    top_ridx: Tensor                 # [B, G]
    next_var_indices: Optional[Tensor]  # [B] or None


@runtime_checkable
class ProofEvidence(Protocol):
    """Accumulated proof trace from completed groundings.

    Field layout matches ``grounder.types.ProofEvidence`` exactly. The
    structured layout uses ``D > 0`` (depth dim populated) and the legacy
    flat layout uses ``D == 0``. Properties allow consumers to ignore the
    distinction.

    Optional ``obs`` field carries an opaque per-step observation tensor
    that policy-driven Selects (e.g. DpRL's ``PolicySelect``) feed into a
    policy network. tkk's reference Selects ignore this field; grounder
    evidence has ``obs is None``. Policy-aware evidence producers (e.g.
    DpRL's ``StatefulEnvResolve``) populate it. Backward compatible:
    consumers should use ``getattr(evidence, "obs", None)``.
    """

    body: Tensor          # [B, P, D, M, 3] or [B, P, G_body, 3]
    mask: Tensor          # [B, P]
    count: Tensor         # [B]
    rule_idx: Tensor      # [B, P, D] or [B, P]
    body_count: Tensor    # [B, P, D] or [B, P]
    D: int
    M: int
    head: Optional[Tensor]  # [B, P, D, 3] or None
    obs: Optional[Any] = None  # opaque policy-input tensor; None on grounder evidence

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

    log_probs: Optional[Tensor] = None       # [B, G] policy gradient
    chosen_scores: Optional[Tensor] = None   # [B, k] beam/greedy
    chosen_indices: Optional[Tensor] = None  # [B, k] which action was picked
    extra: dict = field(default_factory=dict)  # MCTS visit counts etc.


# ═══════════════════════════════════════════════════════════════════════
# Rule-evidence types (rule-based reasoning path)
# ═══════════════════════════════════════════════════════════════════════


@runtime_checkable
class RuleGroundings(Protocol):
    """Per-rule (A_in, A_out) evidence keyed by a deduplicated atom pool.

    Field layout matches ``grounder.types.RuleGroundings`` exactly so that
    the grounder dataclass satisfies this protocol via duck typing. The
    grounder's :func:`BCGrounder.run_bc` returns a concrete instance with
    ``query_pool_idx`` populated.

    Used by the rule-based reasoning path (SBR / DCR / R2N exhaustive,
    pool-iter K iterations). The proof path uses ``ProofEvidence`` instead.

    Attributes:
        atom_table:     ``[N_pool, 3]`` deduplicated (pred, h, t) atoms.
        A_in:           ``Dict[rule_idx, [G_r, M_r]]`` body-atom indices.
        A_out:          ``Dict[rule_idx, [G_r, 1]]`` head-atom indices.
        num_atoms:      ``N_pool``.
        num_rules:      maximum rule index + 1.
        query_pool_idx: ``[B]`` indices into ``atom_table`` for each test
                        query atom; populated by ``run_bc``.
    """
    atom_table: Tensor
    A_in: Dict[int, Tensor]
    A_out: Dict[int, Tensor]
    num_atoms: int
    num_rules: int
    query_pool_idx: Optional[Tensor]


@dataclass
class FiringsTensors:
    """Flat per-firing index tensors keyed by an atom pool.

    A ``firing`` is one rule application — one row of ``A_in / A_out`` in
    keras-ns's terminology. The pool-iter loop consumes these flat
    tensors over ``N_f`` firings.

    Padding atoms (body slots beyond a firing's actual body length, or
    invalid firings) point at the pool's padding slot (``head_pool_idx``
    is set to a real slot but masked off via ``firing_valid``); the
    corresponding entries in ``body_atom_valid`` / ``firing_valid`` are
    False.

    Attributes:
        rule_idx:        ``[N_f]`` rule index per firing.
        body_pool_idx:   ``[N_f, M]`` indices into the atom pool of each
                         body atom of the firing.
        body_atom_valid: ``[N_f, M]`` validity mask per body slot.
        head_pool_idx:   ``[N_f]`` index into the atom pool of the
                         firing's head atom.
        firing_valid:    ``[N_f]`` per-firing validity mask.
    """
    rule_idx: Tensor
    body_pool_idx: Tensor
    body_atom_valid: Tensor
    head_pool_idx: Tensor
    firing_valid: Tensor


def build_firings_from_rule_groundings(
    rg: RuleGroundings,
    *,
    M_max: Optional[int] = None,
    pad_idx: int = 0,
) -> FiringsTensors:
    """Concatenate per-rule (A_in, A_out) into flat per-firing tensors.

    Each rule ``r`` contributes ``G_r = A_in[r].shape[0]`` firings.
    Total ``N_f = sum_r G_r``. Body widths are padded to ``M_max`` (the
    max over all rules' body lengths if not provided) with ``pad_idx``
    (which should index the pool's padding slot 0 for safe gathers).

    The output preserves the rule_idx ordering of ``sorted(rg.A_in.keys())``
    so the same (rg, M_max) input always produces byte-identical firings.

    Args:
        rg: ``RuleGroundings``-conforming object (e.g. grounder's
            ``RuleGroundings`` dataclass).
        M_max: pad target body width. If None, uses
            ``max_r M_r`` over present rules; falls back to 1 for empty rg.
        pad_idx: pool slot to use for invalid body atom positions.

    Returns:
        ``FiringsTensors`` with all five fields concatenated over rules.
    """
    rule_indices = sorted(rg.A_in.keys())
    device = rg.atom_table.device

    if not rule_indices:
        empty_M = M_max or 1
        return FiringsTensors(
            rule_idx=torch.zeros(0, dtype=torch.long, device=device),
            body_pool_idx=torch.zeros(0, empty_M, dtype=torch.long, device=device),
            body_atom_valid=torch.zeros(0, empty_M, dtype=torch.bool, device=device),
            head_pool_idx=torch.zeros(0, dtype=torch.long, device=device),
            firing_valid=torch.zeros(0, dtype=torch.bool, device=device),
        )

    if M_max is None:
        M_max = max(int(rg.A_in[r].shape[1]) for r in rule_indices)

    body_chunks = []
    valid_chunks = []
    head_chunks = []
    rule_chunks = []
    for r in rule_indices:
        A_in_r = rg.A_in[r]                          # [G_r, M_r]
        A_out_r = rg.A_out[r]                        # [G_r, 1]
        G_r, M_r = A_in_r.shape
        if M_r < M_max:
            pad = torch.full(
                (G_r, M_max - M_r), pad_idx, dtype=torch.long, device=device,
            )
            body_padded = torch.cat([A_in_r, pad], dim=1)
            valid = torch.cat(
                [
                    torch.ones(G_r, M_r, dtype=torch.bool, device=device),
                    torch.zeros(G_r, M_max - M_r, dtype=torch.bool, device=device),
                ],
                dim=1,
            )
        else:
            body_padded = A_in_r
            valid = torch.ones(G_r, M_max, dtype=torch.bool, device=device)
        body_chunks.append(body_padded)
        valid_chunks.append(valid)
        head_chunks.append(A_out_r.squeeze(-1))
        rule_chunks.append(
            torch.full((G_r,), r, dtype=torch.long, device=device)
        )

    rule_idx = torch.cat(rule_chunks, dim=0)
    return FiringsTensors(
        rule_idx=rule_idx,
        body_pool_idx=torch.cat(body_chunks, dim=0),
        body_atom_valid=torch.cat(valid_chunks, dim=0),
        head_pool_idx=torch.cat(head_chunks, dim=0),
        firing_valid=torch.ones(rule_idx.shape[0], dtype=torch.bool, device=device),
    )


__all__ = [
    "FiringsTensors",
    "ProofEvidence",
    "ProofState",
    "RuleGroundings",
    "SelectInfo",
    "build_firings_from_rule_groundings",
]
