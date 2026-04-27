"""End-to-end method-tuple tests (framework.pdf §11).

Each test instantiates one canonical method as a 6-tuple of framework
primitives, runs it on fake structured ProofEvidence, and asserts the
output is sane (correct shape, finite values).

This is the "framework integration is provable, not just possible"
checkpoint — adding a new method should be one row in the table, not
new infrastructure.

Method tuples covered:
  - SBR:  Enum + KGEScore   + TNorm(min) + Exhaustive + TNorm(min)   + Max
  - DCR:  Enum + KGEBoth    + PhiPsi     + Exhaustive + TNorm(min)   + Max
  - R2N:  Enum + KGEEmbed   + Concat     + Exhaustive + RuleMLP      + MLPSum
  - DPrL: SLD  + MLP        + Sum        + Beam       + PolicyProduct+ Sum
"""
from __future__ import annotations

import torch
import torch.nn as nn

from kge_kernels.framework import (
    ConcatStateRepr,
    KGEBothAtom,
    KGEEmbedAtom,
    KGEScoreAtom,
    MaxQueryRepr,
    MLPAtom,
    MLPSumQueryRepr,
    PhiPsiStateRepr,
    PolicyProductTrajRepr,
    Repr,
    RuleMLPTrajRepr,
    SelectInfo,
    SumQueryRepr,
    SumStateRepr,
    TNormStateRepr,
    TNormTrajRepr,
)
from kge_kernels.framework.select import BeamSelect
from kge_kernels.search import ProofScorer, SearchSpec, make_searcher
from kge_kernels.models import TransE

from tests.framework.conftest import make_structured_evidence


# ───────────────────────────────────────────────────────────────────────
# SBR
# ───────────────────────────────────────────────────────────────────────


def test_sbr_method_tuple():
    """SBR = Enum + KGEScore + TNorm(min) + Exhaustive + TNorm(min) + Max."""
    ev = make_structured_evidence(B=2, P=3, D=2, M=2)
    model = TransE(num_entities=10, num_relations=5, dim=8)

    sbr = make_searcher(
        "exhaustive",
        resolve=lambda state: ev,
        atom_repr=KGEScoreAtom(),
        state_repr=TNormStateRepr("min"),
        traj_repr=TNormTrajRepr("min"),
        query_repr=MaxQueryRepr(),
        model=model,
        name="sbr",
    )
    queries = torch.randint(1, 10, (2, 3))
    out = sbr(queries)
    assert "sbr" in out
    assert out["sbr"].shape == (2,)
    assert torch.isfinite(out["sbr"]).all()
    # T-norm(min) over [0, 1] sigmoid scores → output in [0, 1].
    assert ((out["sbr"] >= 0) & (out["sbr"] <= 1)).all()


# ───────────────────────────────────────────────────────────────────────
# DCR
# ───────────────────────────────────────────────────────────────────────


def test_dcr_method_tuple():
    """DCR = Enum + KGEBoth + PhiPsi + Exhaustive + TNorm(min) + Max."""
    ev = make_structured_evidence(B=2, P=3, D=2, M=2)
    model = TransE(num_entities=10, num_relations=5, dim=8)
    num_rules = int(ev.rule_idx.max().item()) + 1

    dcr = make_searcher(
        "exhaustive",
        resolve=lambda state: ev,
        atom_repr=KGEBothAtom(),
        state_repr=PhiPsiStateRepr(num_rules=num_rules, embed_dim=8, tnorm="product"),
        traj_repr=TNormTrajRepr("min"),
        query_repr=MaxQueryRepr(),
        model=model,
        name="dcr",
    )
    queries = torch.randint(1, 10, (2, 3))
    out = dcr(queries)
    assert "dcr" in out
    assert out["dcr"].shape == (2,)
    assert torch.isfinite(out["dcr"]).all()


# ───────────────────────────────────────────────────────────────────────
# R2N
# ───────────────────────────────────────────────────────────────────────


def test_r2n_method_tuple():
    """R2N = Enum + KGEEmbed + Concat + Exhaustive + RuleMLP + MLPSum."""
    ev = make_structured_evidence(B=2, P=3, D=2, M=2)
    embed_dim = 8
    model = TransE(num_entities=10, num_relations=5, dim=embed_dim)
    num_rules = int(ev.rule_idx.max().item()) + 1
    M = 2

    r2n = make_searcher(
        "exhaustive",
        resolve=lambda state: ev,
        atom_repr=KGEEmbedAtom(),
        state_repr=ConcatStateRepr(max_atoms=M),
        traj_repr=RuleMLPTrajRepr(num_rules=num_rules, in_dim=M * embed_dim, out_dim=embed_dim),
        query_repr=MLPSumQueryRepr(embed_dim=embed_dim),
        model=model,
        name="r2n",
    )
    queries = torch.randint(1, 10, (2, 3))
    out = r2n(queries)
    assert "r2n" in out
    assert out["r2n"].shape == (2,)
    assert torch.isfinite(out["r2n"]).all()


# ───────────────────────────────────────────────────────────────────────
# DPrL
# ───────────────────────────────────────────────────────────────────────


class _SumWithPolicyScore(nn.Module):
    """SumStateRepr + an L2-norm-as-scalar-score "policy head".

    DPrL canonically uses Sum state_repr (produces embeddings) + a
    learned policy that projects embeddings to per-state action
    scores. The framework PDF lists Sum at the state_repr slot but
    leaves the policy projection implicit (it's part of select in
    framework.pdf §13). For the integration test we wire a trivial
    L2-norm projection so BeamSelect has scalar scores to rank.
    """

    def __init__(self) -> None:
        super().__init__()
        self.sum = SumStateRepr()

    def forward(self, atom_repr, evidence):
        s = self.sum(atom_repr, evidence)
        # Add a scalar score so BeamSelect/GreedySelect work.
        return Repr(embeddings=s.embeddings, scores=s.embeddings.norm(dim=-1))


class _BeamWithLogProbs(nn.Module):
    """BeamSelect adapter that synthesizes uniform log_probs into SelectInfo.

    Real DPrL gets log-probs from its policy network. For the framework
    integration test we just need PolicyProductTrajRepr to receive a
    valid ``info.log_probs`` at each step.
    """

    def __init__(self, k: int) -> None:
        super().__init__()
        self.beam = BeamSelect(k=k)

    def forward(self, evidence, s_repr):
        state, info = self.beam(evidence, s_repr)
        B = (
            s_repr.scores.shape[0] if s_repr.has_scores
            else s_repr.embeddings.shape[0]
        )
        return state, SelectInfo(
            chosen_indices=info.chosen_indices,
            chosen_scores=info.chosen_scores,
            log_probs=torch.full((B,), -2.0),
        )


class _SumOverBatch(nn.Module):
    """QueryRepr that sums per-query along ``B`` (no candidate-C dim).

    DPrL's canonical sequential loop produces a single trajectory per
    query (the chosen path through the tree). The canonical
    :class:`ProofScorer` loop doesn't expand the C dim during
    sequential search, so the PDF's ``Sum`` query_repr is implicitly
    "identity over the single-trajectory output" rather than the
    candidate-pool reduction in :class:`SumQueryRepr`.

    This stub captures that semantic: identity on [B] → [B].
    """

    def forward(self, traj_repr, evidence):
        if not traj_repr.has_scores:
            raise ValueError("_SumOverBatch requires traj_repr.scores")
        # traj_repr.scores already shape [B] from PolicyProductTrajRepr.step.
        return Repr(scores=traj_repr.scores)


def test_dprl_method_tuple():
    """DPrL = SLD + MLP + Sum + Beam + PolicyProduct + Sum.

    Sequential search; PolicyProductTrajRepr accumulates per-step
    log-probs into a per-query scalar. The query_repr is identity on
    the [B] accumulator (sequential search produces one trajectory
    per query, not C candidates — see the docstring on
    ``_SumOverBatch`` above).
    """
    embed_dim = 8
    model = TransE(num_entities=10, num_relations=5, dim=embed_dim)
    ev = make_structured_evidence(B=2, P=3, D=1, M=2)

    scorer = ProofScorer(
        resolve=lambda state: ev,
        atom_repr=MLPAtom(embed_dim=embed_dim),
        state_repr=_SumWithPolicyScore(),
        select=_BeamWithLogProbs(k=2),
        traj_repr=PolicyProductTrajRepr(),
        query_repr=_SumOverBatch(),
        spec=SearchSpec(batch_size=2, max_depth=2),
        model=model,
        capture="dynamic",
        name="dprl",
    )
    scores = scorer(torch.zeros(2, 3, dtype=torch.long))["dprl"]
    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()


# ───────────────────────────────────────────────────────────────────────
# Single-axis ablation (PDF §15)
# ───────────────────────────────────────────────────────────────────────


def test_sbr_query_aggregation_ablation():
    """Swap MaxQueryRepr → SumQueryRepr keeping the rest of SBR fixed."""
    ev = make_structured_evidence(B=2, P=3, D=2, M=2)
    model = TransE(num_entities=10, num_relations=5, dim=8)

    base_kw = dict(
        resolve=lambda state: ev,
        atom_repr=KGEScoreAtom(),
        state_repr=TNormStateRepr("min"),
        traj_repr=TNormTrajRepr("min"),
        model=model,
    )
    sbr_max = make_searcher("exhaustive", **base_kw, query_repr=MaxQueryRepr(), name="sbr_max")
    sbr_sum = make_searcher("exhaustive", **base_kw, query_repr=SumQueryRepr(), name="sbr_sum")

    queries = torch.randint(1, 10, (2, 3))
    out_max = sbr_max(queries)["sbr_max"]
    out_sum = sbr_sum(queries)["sbr_sum"]
    assert out_max.shape == out_sum.shape == (2,)
    # Sum >= Max when there are ≥ 1 valid proofs (sum over non-negative scores).
    assert (out_sum >= out_max - 1e-6).all()
