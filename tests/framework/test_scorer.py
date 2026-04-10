"""End-to-end search_and_score tests with a fake resolve operator."""
from __future__ import annotations

import torch

from kge_kernels.framework import (
    ExhaustiveSelect,
    KGEScoreAtom,
    MaxQueryRepr,
    TNormStateRepr,
    TNormTrajRepr,
    build_scorer,
    search_and_score,
)
from kge_kernels.models import TransE

from .conftest import make_structured_evidence


def test_exhaustive_search_and_score_shape():
    ev = make_structured_evidence(B=2, C=3, D=2, M=2)

    def fake_resolve(state):
        return ev

    model = TransE(num_entities=7, num_relations=5, dim=8)

    scores = search_and_score(
        query=None,
        resolve=fake_resolve,
        atom_repr=KGEScoreAtom(),
        state_repr=TNormStateRepr("min"),
        select=ExhaustiveSelect(),
        traj_repr=TNormTrajRepr("min"),
        query_repr=MaxQueryRepr(),
        model=model,
        max_depth=1,
    )
    assert scores.shape == (2,)
    assert ((scores >= 0) & (scores <= 1)).all()


def test_build_scorer_returns_dict():
    ev = make_structured_evidence(B=2, C=3, D=2, M=2)

    def fake_resolve(state):
        return ev

    model = TransE(num_entities=7, num_relations=5, dim=8)

    scorer = build_scorer(
        resolve=fake_resolve,
        atom_repr=KGEScoreAtom(),
        state_repr=TNormStateRepr("min"),
        select=ExhaustiveSelect(),
        traj_repr=TNormTrajRepr("min"),
        query_repr=MaxQueryRepr(),
        model=model,
        max_depth=1,
        name="sbr_min",
    )
    out = scorer(queries=None)
    assert "sbr_min" in out
    assert out["sbr_min"].shape == (2,)


def test_q_select_can_read_evidence_for_future_compat():
    """A custom Select subclass should be able to compute Q = H + V from
    ``evidence.rule_idx`` and ``s_repr.scores``. This is the future
    Q-learning hook from framework.tex §13 — we just verify the signature
    is expressive enough, not that any learning happens."""
    import torch.nn as nn

    from kge_kernels.framework import Repr, SelectInfo

    class QSelect(nn.Module):
        def __init__(self):
            super().__init__()
            self.lambdas = nn.Parameter(torch.zeros(2))  # 2 rules

        def forward(self, evidence, s_repr):
            v = s_repr.scores                       # [B, C, D] (per-state per-depth)
            r_idx = evidence.rule_idx               # [B, C, D] structured
            h = self.lambdas[r_idx]                 # [B, C, D]
            q = h + v
            # Reduce D to get per-candidate Q, then argmax over C
            q_per_c = q.mean(dim=-1)                # [B, C]
            chosen = q_per_c.argmax(dim=-1, keepdim=True)
            return None, SelectInfo(
                chosen_indices=chosen,
                chosen_scores=torch.gather(q_per_c, -1, chosen),
            )

    ev = make_structured_evidence(B=2, C=3, D=2, M=2)
    model = TransE(num_entities=7, num_relations=5, dim=8)
    select = QSelect()

    # Just check the wiring runs end-to-end without error.
    scores = search_and_score(
        query=None,
        resolve=lambda s: ev,
        atom_repr=KGEScoreAtom(),
        state_repr=TNormStateRepr("min"),
        select=select,
        traj_repr=TNormTrajRepr("min"),
        query_repr=MaxQueryRepr(),
        model=model,
        max_depth=1,
    )
    assert scores.shape == (2,)
