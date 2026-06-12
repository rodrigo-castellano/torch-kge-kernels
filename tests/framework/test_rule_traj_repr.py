"""Algebraic parity tests for the rule-based reasoning path.

Locks the math worked out in
``~/repos/torch-ns-swarm/tkk-consolidation/docs/reasoners_framework_vs_keras.md``
sections 3-5: SBR / DCR / R2N evaluated under View C (pool-iter, K
iterations, scatter_max). Each test builds a tiny KB + rules + KGE-init
by hand, runs ``K`` iterations of the appropriate ``RuleTrajRepr``, and
asserts the gathered query score matches a hand-computed scalar.

Tests use the in-memory primitives directly (no grounder dependency)
— the grounder's role is to produce :class:`RuleGroundings`, but here
we synthesize equivalent flat firings via
:func:`build_firings_from_rule_groundings` over a hand-built RG.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pytest
import torch
from torch import Tensor

from kge_kernels.framework import (
    DCRPoolLoop,
    FilterSignRuleState,
    LookupAtPool,
    MinPoolLoop,
    MinRuleState,
    RuleMLPPoolLoop,
    RuleMLPState,
    build_firings_from_rule_groundings,
)


@dataclass
class FakeRuleGroundings:
    """Concrete dataclass satisfying the RuleGroundings Protocol."""
    atom_table: Tensor
    A_in: Dict[int, Tensor]
    A_out: Dict[int, Tensor]
    num_atoms: int
    num_rules: int
    query_pool_idx: Optional[Tensor] = None


def _run_scalar_pool(
    pool_size: int,
    init_scores: Dict[int, float],
    A_in: Dict[int, Tensor],
    A_out: Dict[int, Tensor],
    query_idx: int,
    *,
    K: int,
    state_repr_fn,
    loop_cls,
    M_max: int,
) -> float:
    """Helper: build pool + firings, run K iterations, return query score."""
    pool = torch.zeros(pool_size, dtype=torch.float32)
    for slot, score in init_scores.items():
        pool[slot] = score

    rg = FakeRuleGroundings(
        atom_table=torch.zeros(pool_size, 3, dtype=torch.long),
        A_in=A_in,
        A_out=A_out,
        num_atoms=pool_size,
        num_rules=max(A_in.keys()) + 1 if A_in else 1,
    )
    firings = build_firings_from_rule_groundings(rg, M_max=M_max, pad_idx=0)

    loop = loop_cls(K=K)
    new_pool, ever_written = loop(pool, firings, state_repr_fn)

    # Pin unwritten_score=0.0 so the algebraic tests are independent
    # of the pool's KGE-init value (each test sets pool slots
    # explicitly above; an unwritten query slot would otherwise leak
    # the seed init value into the assertion).
    qpr = LookupAtPool(unwritten_score=0.0)
    score = qpr(new_pool, torch.tensor([query_idx]), ever_written)
    return float(score.item())


# ═══════════════════════════════════════════════════════════════════════
# SBR — 3 tests
# ═══════════════════════════════════════════════════════════════════════


class TestSBR:

    def test_sbr_K1_one_rule(self):
        """KB={p(a,b), q(b,c)}, rule p,q→t, query t(a,c).
        KGE p=0.9, q=0.7. Expect K=1: min(0.9, 0.7) = 0.70."""
        # Pool slots: 0=pad, 1=p, 2=q, 3=t
        score = _run_scalar_pool(
            pool_size=4,
            init_scores={1: 0.9, 2: 0.7, 3: 0.5},
            A_in={0: torch.tensor([[1, 2]])},        # rule 0 body = [p, q]
            A_out={0: torch.tensor([[3]])},           # rule 0 head = t
            query_idx=3,
            K=1,
            state_repr_fn=MinRuleState(),
            loop_cls=MinPoolLoop,
            M_max=2,
        )
        assert abs(score - 0.7) < 1e-5, f"expected 0.70, got {score}"

    def test_sbr_K2_chain_through_shared_atom(self):
        """Doc's worked example. KB={q(b,c), r(a,b), s(a,b)}.
        Rules R1: p,q→t; R2: r→p; R3: s→p.
        KGE p=0.30, q=0.90, r=0.70, s=0.95, t=0.50.
        Expect K=2: 0.90 (R3 lifts pool[p] to 0.95 in iter 1,
        R1 reads min(0.95, 0.90)=0.90 in iter 2)."""
        # Pool slots: 0=pad, 1=p, 2=q, 3=r, 4=s, 5=t
        score = _run_scalar_pool(
            pool_size=6,
            init_scores={1: 0.30, 2: 0.90, 3: 0.70, 4: 0.95, 5: 0.50},
            A_in={
                0: torch.tensor([[1, 2]]),                      # R1: [p, q] (M=2)
                1: torch.tensor([[3]]),                         # R2: [r]    (M=1)
                2: torch.tensor([[4]]),                         # R3: [s]    (M=1)
            },
            A_out={
                0: torch.tensor([[5]]),                         # R1 head = t
                1: torch.tensor([[1]]),                         # R2 head = p
                2: torch.tensor([[1]]),                         # R3 head = p
            },
            query_idx=5,
            K=2,
            state_repr_fn=MinRuleState(),
            loop_cls=MinPoolLoop,
            M_max=2,
        )
        assert abs(score - 0.90) < 1e-5, f"expected 0.90, got {score}"

    def test_sbr_K3_idempotent(self):
        """Same setup as K=2; verify K=3 = K=2 (fixed point reached)."""
        kwargs = dict(
            pool_size=6,
            init_scores={1: 0.30, 2: 0.90, 3: 0.70, 4: 0.95, 5: 0.50},
            A_in={
                0: torch.tensor([[1, 2]]),
                1: torch.tensor([[3]]),
                2: torch.tensor([[4]]),
            },
            A_out={
                0: torch.tensor([[5]]),
                1: torch.tensor([[1]]),
                2: torch.tensor([[1]]),
            },
            query_idx=5,
            state_repr_fn=MinRuleState(),
            loop_cls=MinPoolLoop,
            M_max=2,
        )
        score_k2 = _run_scalar_pool(K=2, **kwargs)
        score_k3 = _run_scalar_pool(K=3, **kwargs)
        assert abs(score_k2 - score_k3) < 1e-7, (
            f"K=3 should equal K=2 at fixed point: K=2={score_k2}, K=3={score_k3}"
        )


# ═══════════════════════════════════════════════════════════════════════
# DCR — 2 tests
# ═══════════════════════════════════════════════════════════════════════


class TestDCR:

    def test_dcr_K1_filter_sign(self):
        """1 firing, 2 body atoms, hand-set (φ_pos=0.9, σ_pos=+1, φ_neg=0.1, σ_neg=-1).
        KGE preds (P=0.8, Q=0.3). Hand-compute Gödel filter+sign and assert."""
        # Set filter logits: sigmoid(2.197) ≈ 0.9, sigmoid(-2.197) ≈ 0.1
        # Set sign logits: tanh(10) ≈ +1.0, tanh(-10) ≈ -1.0
        state = FilterSignRuleState(num_rules=1, M=2)
        with torch.no_grad():
            state.filter_logits.copy_(torch.tensor([[2.197, -2.197]]))
            state.sign_logits.copy_(torch.tensor([[10.0, -10.0]]))

        # Hand expected:
        # P (σ≈+1, φ≈0.9): gated = P = 0.8.    weighted = 0.9*0.8 + 0.1*1 = 0.82
        # Q (σ≈-1, φ≈0.1): gated = 1-Q = 0.7.  weighted = 0.1*0.7 + 0.9*1 = 0.97
        # head = min(0.82, 0.97) = 0.82
        # Pool slots: 0=pad, 1=P, 2=Q, 3=head
        score = _run_scalar_pool(
            pool_size=4,
            init_scores={1: 0.8, 2: 0.3, 3: 0.0},
            A_in={0: torch.tensor([[1, 2]])},
            A_out={0: torch.tensor([[3]])},
            query_idx=3,
            K=1,
            state_repr_fn=state,
            loop_cls=DCRPoolLoop,
            M_max=2,
        )
        assert abs(score - 0.82) < 1e-3, f"expected ~0.82, got {score}"

    def test_dcr_K2_chain_with_sign_flip(self):
        """Chain r1: a→b (σ=+1), r2: b→c (σ=-1). Both with φ=0.9.
        KGE a=0.8, b=0.3, c=0.5.

        K=1: F2 reads b=0.3 (KGE-init). gated = 1 - 0.3 = 0.7.
              weighted = 0.9 * 0.7 + 0.1 * 1 = 0.73. → c=0.73.
        K=2: F2 reads b=0.82 (updated by F1 in iter 1). gated = 1 - 0.82 = 0.18.
              weighted = 0.9 * 0.18 + 0.1 * 1 = 0.262. → c=0.262.
        Confirms K=2 ≠ K=1 with non-monotone aggregation."""
        # Two rules need separate FilterSignRuleStates since
        # FilterSignRuleState's R parameters all share M.
        # Use a single state with R=2, M=1 — different signs per rule.
        state = FilterSignRuleState(num_rules=2, M=1)
        with torch.no_grad():
            # rule 0: φ=0.9, σ=+1
            # rule 1: φ=0.9, σ=-1
            state.filter_logits.copy_(torch.tensor([[2.197], [2.197]]))
            state.sign_logits.copy_(torch.tensor([[10.0], [-10.0]]))

        kwargs = dict(
            pool_size=4,    # 0=pad, 1=a, 2=b, 3=c
            init_scores={1: 0.8, 2: 0.3, 3: 0.5},
            A_in={
                0: torch.tensor([[1]]),    # rule 0: [a]
                1: torch.tensor([[2]]),    # rule 1: [b]
            },
            A_out={
                0: torch.tensor([[2]]),    # rule 0 head = b
                1: torch.tensor([[3]]),    # rule 1 head = c
            },
            query_idx=3,
            state_repr_fn=state,
            loop_cls=DCRPoolLoop,
            M_max=1,
        )
        score_k1 = _run_scalar_pool(K=1, **kwargs)
        score_k2 = _run_scalar_pool(K=2, **kwargs)
        # K=1: c gets 0.73 (reading KGE-init b=0.3)
        assert abs(score_k1 - 0.73) < 1e-3, f"K=1 expected ~0.73, got {score_k1}"
        # K=2: c gets ~0.262 (reading updated b≈0.82)
        assert abs(score_k2 - 0.262) < 1e-2, f"K=2 expected ~0.262, got {score_k2}"
        # And the two are visibly different.
        assert abs(score_k1 - score_k2) > 0.4, (
            f"K=1 and K=2 must differ by > 0.4: K=1={score_k1}, K=2={score_k2}"
        )


# ═══════════════════════════════════════════════════════════════════════
# R2N — 3 tests
# ═══════════════════════════════════════════════════════════════════════


def _identity_output_layer(emb: Tensor) -> Tensor:
    """Trivial output_layer that returns the embedding's first dim."""
    if emb.dim() == 1:
        return emb
    return emb[..., 0]


def _run_emb_pool(
    pool_size: int,
    embed_dim: int,
    init_embs: Dict[int, list],
    A_in: Dict[int, Tensor],
    A_out: Dict[int, Tensor],
    query_idx: int,
    *,
    K: int,
    mlp: RuleMLPState,
    prediction_type: str,
    M_max: int,
    aggregation: str = "max",
) -> Tensor:
    """Helper: build embedding pool + firings, run K iterations, return query embedding."""
    pool = torch.zeros(pool_size, embed_dim, dtype=torch.float32)
    for slot, emb in init_embs.items():
        pool[slot] = torch.tensor(emb, dtype=torch.float32)

    rg = FakeRuleGroundings(
        atom_table=torch.zeros(pool_size, 3, dtype=torch.long),
        A_in=A_in,
        A_out=A_out,
        num_atoms=pool_size,
        num_rules=max(A_in.keys()) + 1 if A_in else 1,
    )
    firings = build_firings_from_rule_groundings(rg, M_max=M_max, pad_idx=0)

    loop = RuleMLPPoolLoop(
        K=K, prediction_type=prediction_type, aggregation=aggregation)
    new_pool, ever_written = loop(pool, firings, mlp)
    return new_pool[query_idx]


class TestR2N:

    def test_r2n_K1_E1_scalar(self):
        """1 firing, MLP_R hand-set so that MLP_R([a, b]) = a + b.
        KGE r=0.6, s=0.4, t=arbitrary. K=1 → pool[t] = 0.6 + 0.4 = 1.0."""
        # Hand-set MLP weights so that MLP([a, b]) = a + b.
        # in_dim=2 (M=2 atoms × E=1), atom_emb_dim=1, num_atoms_out=1, hidden=1.
        mlp = RuleMLPState(
            num_rules=1, in_dim=2, atom_emb_dim=1,
            num_atoms_out=1, hidden_dim=1,
        )
        with torch.no_grad():
            mlp.l1.copy_(torch.tensor([[[1.0], [1.0]]]))   # [R=1, in=2, h=1]
            mlp.b1.zero_()
            mlp.l2.copy_(torch.tensor([[[1.0]]]))           # [R=1, h=1, out=1]
            mlp.b2.zero_()

        emb = _run_emb_pool(
            pool_size=4,                                       # 0=pad, 1=r, 2=s, 3=t
            embed_dim=1,
            init_embs={1: [0.6], 2: [0.4], 3: [0.0]},
            A_in={0: torch.tensor([[1, 2]])},
            A_out={0: torch.tensor([[3]])},
            query_idx=3,
            K=1,
            mlp=mlp,
            prediction_type="head",
            M_max=2,
        )
        assert abs(float(emb.item()) - 1.0) < 1e-5, (
            f"expected pool[t]=1.0, got {emb.item()}"
        )

    def test_r2n_K2_chain_reveals_chained_mlp(self):
        """Chain like SBR test 2 (R1: p,q→t; R2: r→p; R3: s→p) but R2N MLPs.
        All MLPs do sum. Head-only mode (body atoms not rewritten).

        K=1: pool[t] = MLP_R1([KGE_p=0.3, KGE_q=0.9]) = 1.2.
        K=2: pool[p] updated to max(MLP_R2(KGE_r=0.7), MLP_R3(KGE_s=0.95)) = 0.95.
              pool[t] = MLP_R1([0.95, 0.9]) = 1.85.
        Confirms K=2 ≠ K=1: pool reads chain through the head-only updates."""
        # 3 rules, M_max=2 (R1 has 2 body atoms; R2/R3 have 1 padded to 2).
        # MLP: in_dim=2*E=2, hidden=1, out=1. Sum behavior via [[1],[1]] / [[1]].
        mlp = RuleMLPState(
            num_rules=3, in_dim=2, atom_emb_dim=1,
            num_atoms_out=1, hidden_dim=1,
        )
        with torch.no_grad():
            mlp.l1.copy_(torch.tensor([
                [[1.0], [1.0]],   # R1: sum both body atoms
                [[1.0], [1.0]],   # R2: sum (second slot is padded → 0)
                [[1.0], [1.0]],   # R3: sum (second slot padded → 0)
            ]))
            mlp.b1.zero_()
            mlp.l2.copy_(torch.tensor([
                [[1.0]], [[1.0]], [[1.0]],
            ]))
            mlp.b2.zero_()

        kwargs = dict(
            pool_size=6,
            embed_dim=1,
            init_embs={1: [0.30], 2: [0.90], 3: [0.70], 4: [0.95], 5: [0.0]},
            A_in={
                0: torch.tensor([[1, 2]]),    # R1: [p, q]
                1: torch.tensor([[3]]),       # R2: [r]
                2: torch.tensor([[4]]),       # R3: [s]
            },
            A_out={
                0: torch.tensor([[5]]),       # R1 head = t
                1: torch.tensor([[1]]),       # R2 head = p
                2: torch.tensor([[1]]),       # R3 head = p
            },
            query_idx=5,
            mlp=mlp,
            prediction_type="head",
            M_max=2,
        )
        emb_k1 = _run_emb_pool(K=1, **kwargs)
        emb_k2 = _run_emb_pool(K=2, **kwargs)

        # K=1: pool[t] = 0.3 + 0.9 = 1.2
        assert abs(float(emb_k1.item()) - 1.2) < 1e-5, (
            f"K=1 expected pool[t]=1.2, got {emb_k1.item()}"
        )
        # K=2: pool[t] = 0.95 + 0.9 = 1.85 (pool[p] lifted to 0.95 via R3 in iter 1)
        assert abs(float(emb_k2.item()) - 1.85) < 1e-5, (
            f"K=2 expected pool[t]=1.85, got {emb_k2.item()}"
        )

    def test_r2n_E2_scatter_max_stitches_per_dimension(self):
        """E=2. Two firings target pool[p] with predictions r=[1,0] and s=[0,1].
        scatter_max elementwise: pool[p] = [max(1,0), max(0,1)] = [1, 1].
        Confirms the keras-faithful per-dim stitching that no single proof produces."""
        # 2 rules, identity MLP at hidden_dim=2 so each rule emits its body emb.
        # in_dim = M*E = 1*2 = 2. atom_emb_dim=2. num_atoms_out=1.
        # MLP_R(body_emb) = body_emb. Use l1=I, l2=I, no bias.
        mlp = RuleMLPState(
            num_rules=2, in_dim=2, atom_emb_dim=2,
            num_atoms_out=1, hidden_dim=2,
        )
        with torch.no_grad():
            # l1 [R=2, in=2, h=2]: identity per rule.
            mlp.l1.copy_(torch.stack([torch.eye(2), torch.eye(2)]))
            mlp.b1.zero_()
            # l2 [R=2, h=2, out=2]: identity per rule.
            mlp.l2.copy_(torch.stack([torch.eye(2), torch.eye(2)]))
            mlp.b2.zero_()

        # Pool: 0=pad, 1=r, 2=s, 3=p.
        # Both firings have M=1; head=p.
        emb = _run_emb_pool(
            pool_size=4,
            embed_dim=2,
            init_embs={1: [1.0, 0.0], 2: [0.0, 1.0], 3: [0.0, 0.0]},
            A_in={
                0: torch.tensor([[1]]),    # R1: [r]
                1: torch.tensor([[2]]),    # R2: [s]
            },
            A_out={
                0: torch.tensor([[3]]),    # R1 head = p
                1: torch.tensor([[3]]),    # R2 head = p
            },
            query_idx=3,
            K=1,
            mlp=mlp,
            prediction_type="head",
            M_max=1,
        )
        # pool[p] should be elementwise max of [1,0] and [0,1] = [1, 1].
        assert torch.allclose(emb, torch.tensor([1.0, 1.0]), atol=1e-5), (
            f"expected pool[p]=[1,1] (per-dim scatter_max stitching), got {emb}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Smoke tests on build_firings_from_rule_groundings
# ═══════════════════════════════════════════════════════════════════════


class TestBuildFirings:

    def test_pads_short_rule_to_M_max(self):
        rg = FakeRuleGroundings(
            atom_table=torch.zeros(5, 3, dtype=torch.long),
            A_in={0: torch.tensor([[1, 2]]), 1: torch.tensor([[3]])},
            A_out={0: torch.tensor([[4]]), 1: torch.tensor([[4]])},
            num_atoms=5, num_rules=2,
        )
        firings = build_firings_from_rule_groundings(rg, M_max=2, pad_idx=0)
        # Two firings, both width 2.
        assert firings.body_pool_idx.shape == (2, 2)
        # First firing (rule 0): body=[1,2], all valid.
        assert firings.body_pool_idx[0].tolist() == [1, 2]
        assert firings.body_atom_valid[0].tolist() == [True, True]
        # Second firing (rule 1): body=[3, pad], second slot invalid.
        assert firings.body_pool_idx[1].tolist() == [3, 0]
        assert firings.body_atom_valid[1].tolist() == [True, False]
        assert firings.head_pool_idx.tolist() == [4, 4]
        assert firings.rule_idx.tolist() == [0, 1]
        assert firings.firing_valid.tolist() == [True, True]

    def test_empty_rg_returns_zero_firings(self):
        rg = FakeRuleGroundings(
            atom_table=torch.zeros(0, 3, dtype=torch.long),
            A_in={}, A_out={}, num_atoms=0, num_rules=1,
        )
        firings = build_firings_from_rule_groundings(rg, M_max=2)
        assert firings.body_pool_idx.shape == (0, 2)
        assert firings.head_pool_idx.shape == (0,)
        assert firings.rule_idx.shape == (0,)

    def test_M_max_inferred_from_rules(self):
        rg = FakeRuleGroundings(
            atom_table=torch.zeros(5, 3, dtype=torch.long),
            A_in={0: torch.tensor([[1]]), 1: torch.tensor([[1, 2, 3]])},
            A_out={0: torch.tensor([[4]]), 1: torch.tensor([[4]])},
            num_atoms=5, num_rules=2,
        )
        firings = build_firings_from_rule_groundings(rg)   # M_max not passed
        # Inferred M_max = 3 (from rule 1).
        assert firings.body_pool_idx.shape == (2, 3)


class TestR2NAggregation:
    """The `aggregation` modes of RuleMLPPoolLoop (docs/MONOTONICITY.md in
    torch-ns: per-dim amax makes the score monotone in the firing set;
    `gated` selects ONE whole vector by body score and breaks monotonicity)."""

    @staticmethod
    def _two_rule_mlp(bias_rule1: float = 0.0) -> RuleMLPState:
        """2 rules, M=2, E=1; MLP_r([a, b]) = a + b (+ per-rule output bias)."""
        mlp = RuleMLPState(
            num_rules=2, in_dim=2, atom_emb_dim=1,
            num_atoms_out=1, hidden_dim=1,
        )
        with torch.no_grad():
            mlp.l1.copy_(torch.tensor([[[1.0], [1.0]]] * 2))
            mlp.b1.zero_()
            mlp.l2.copy_(torch.tensor([[[1.0]]] * 2))
            mlp.b2.zero_()
            mlp.b2[1] += bias_rule1
        return mlp

    # pool slots: 0=pad, 1/2 = strong bodies (sigma(sum)≈0.99),
    # 4/5 = weak bodies (sigma(sum)≈0.007), 3 = head.
    _COMMON = dict(
        pool_size=6,
        embed_dim=1,
        init_embs={1: [5.0], 2: [5.0], 4: [-5.0], 5: [-5.0], 3: [0.0]},
        query_idx=3,
        K=1,
        prediction_type="head",
        M_max=2,
    )

    def test_gated_equals_max_single_firing(self):
        """One firing per atom: gated and max agree exactly."""
        kw = dict(self._COMMON,
                  A_in={0: torch.tensor([[1, 2]])},
                  A_out={0: torch.tensor([[3]])})
        out_max = _run_emb_pool(mlp=self._two_rule_mlp(), aggregation="max", **kw)
        out_gated = _run_emb_pool(mlp=self._two_rule_mlp(), aggregation="gated", **kw)
        assert torch.equal(out_max, out_gated)
        assert abs(float(out_max.item()) - 10.0) < 1e-5

    def test_gated_factual_bodies_beat_loud_noise(self):
        """Firing A: strong bodies (pred relu(10)=10). Firing B: weak bodies
        but a +100 output bias (pred relu(-10)+100 = 100). max picks the loud
        noise (100); gated picks the factual-bodied firing (10)."""
        kw = dict(self._COMMON,
                  A_in={0: torch.tensor([[1, 2]]), 1: torch.tensor([[4, 5]])},
                  A_out={0: torch.tensor([[3]]), 1: torch.tensor([[3]])})
        out_max = _run_emb_pool(
            mlp=self._two_rule_mlp(bias_rule1=100.0), aggregation="max", **kw)
        out_gated = _run_emb_pool(
            mlp=self._two_rule_mlp(bias_rule1=100.0), aggregation="gated", **kw)
        assert abs(float(out_max.item()) - 100.0) < 1e-4   # noise wins the max
        assert abs(float(out_gated.item()) - 10.0) < 1e-4  # gate keeps the chain

    def test_gated_not_monotone_in_firing_count(self):
        """Adding a weak-bodied firing leaves the gated output UNCHANGED
        (score not monotone in the firing set), while max only grows."""
        solo = dict(self._COMMON,
                    A_in={0: torch.tensor([[1, 2]])},
                    A_out={0: torch.tensor([[3]])})
        both = dict(self._COMMON,
                    A_in={0: torch.tensor([[1, 2]]), 1: torch.tensor([[4, 5]])},
                    A_out={0: torch.tensor([[3]]), 1: torch.tensor([[3]])})
        g_solo = _run_emb_pool(
            mlp=self._two_rule_mlp(bias_rule1=100.0), aggregation="gated", **solo)
        g_both = _run_emb_pool(
            mlp=self._two_rule_mlp(bias_rule1=100.0), aggregation="gated", **both)
        m_solo = _run_emb_pool(
            mlp=self._two_rule_mlp(bias_rule1=100.0), aggregation="max", **solo)
        m_both = _run_emb_pool(
            mlp=self._two_rule_mlp(bias_rule1=100.0), aggregation="max", **both)
        assert torch.equal(g_solo, g_both)                 # gated: invariant
        assert float(m_both.item()) > float(m_solo.item())  # max: inflates

    def test_mean_sum_modes_run(self):
        """mean/sum (the other keras options) produce the expected algebra."""
        kw = dict(self._COMMON,
                  A_in={0: torch.tensor([[1, 2]]), 1: torch.tensor([[4, 5]])},
                  A_out={0: torch.tensor([[3]]), 1: torch.tensor([[3]])})
        out_sum = _run_emb_pool(mlp=self._two_rule_mlp(), aggregation="sum", **kw)
        out_mean = _run_emb_pool(mlp=self._two_rule_mlp(), aggregation="mean", **kw)
        assert abs(float(out_sum.item()) - 10.0) < 1e-4    # 10 + relu(-10)=0
        assert abs(float(out_mean.item()) - 5.0) < 1e-4    # (10 + 0) / 2
