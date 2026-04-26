"""Searcher protocol + make_scorer_from_searcher adapter tests."""
from __future__ import annotations

import torch

from kge_kernels.search import (
    DirectSearcher,
    ExhaustiveSearcher,
    GreedySearcher,
    MultiRolloutSearcher,
    Searcher,
    make_scorer_from_searcher,
    make_searcher,
)
from kge_kernels.framework import (
    KGEScoreAtom,
    MaxQueryRepr,
    TNormStateRepr,
    TNormTrajRepr,
)
from kge_kernels.models import TransE

from tests.framework.conftest import make_structured_evidence


def test_searcher_protocol_recognizes_concrete_classes():
    """All concrete Searcher classes should satisfy the Protocol."""
    model = TransE(num_entities=7, num_relations=5, dim=8)
    direct = DirectSearcher(atom_repr=KGEScoreAtom(), model=model)
    assert isinstance(direct, Searcher)


def test_make_scorer_from_searcher_reshape():
    """The adapter must reshape K-major flat scores back to [B, P]."""
    class _FakeSearcher:
        def __call__(self, queries: torch.Tensor):
            # queries: [N, 3]; return one score per query.
            return {"score": queries.sum(dim=-1).float()}

        def set_gumbel_scale(self, scale: float):
            pass

    searcher = _FakeSearcher()
    scorer = make_scorer_from_searcher(searcher, "score")

    B, P = 4, 5
    q_buf = torch.randint(1, 10, (B, 3))
    pool_buf = torch.randint(1, 10, (B, P))

    scores = scorer(q_buf, pool_buf, "tail")
    assert scores.shape == (B, P)

    # Verify the K-major flat → [B, P] reshape: scores[b, k] equals
    # the sum of the triple constructed from (q_buf[b], pool_buf[b, k]).
    for b in range(B):
        for k in range(P):
            expected = q_buf[b, 0] + q_buf[b, 1] + pool_buf[b, k]   # tail mode replaces col 2
            assert scores[b, k].item() == expected.item()


def test_make_scorer_head_mode_replaces_col1():
    class _FakeSearcher:
        def __call__(self, queries):
            return {"score": queries.float().sum(dim=-1)}

        def set_gumbel_scale(self, scale): pass

    scorer = make_scorer_from_searcher(_FakeSearcher(), "score")
    B, P = 2, 3
    q_buf = torch.tensor([[10, 20, 30], [40, 50, 60]])
    pool_buf = torch.tensor([[1, 2, 3], [4, 5, 6]])
    scores = scorer(q_buf, pool_buf, "head")
    # head mode replaces col 1 (subj). For (b=0, k=0): pred=10, subj=1, obj=30 → sum=41.
    assert scores[0, 0].item() == 10 + 1 + 30
    assert scores[1, 2].item() == 40 + 6 + 60


def test_exhaustive_searcher_sbr_tuple():
    """SBR per framework.pdf §11: Enum + KGEScore + TNorm(min) + Exhaustive + TNorm(min) + Max."""
    ev = make_structured_evidence(B=2, C=3, D=2, M=2)
    model = TransE(num_entities=7, num_relations=5, dim=8)

    sbr = ExhaustiveSearcher(
        resolve=lambda state: ev,
        atom_repr=KGEScoreAtom(),
        state_repr=TNormStateRepr("min"),
        traj_repr=TNormTrajRepr("min"),
        query_repr=MaxQueryRepr(),
        model=model,
        name="sbr_min",
    )
    queries = torch.randint(1, 7, (2, 3))
    out = sbr(queries)
    assert "sbr_min" in out
    assert out["sbr_min"].shape == (2,)
    assert ((out["sbr_min"] >= 0) & (out["sbr_min"] <= 1)).all()


def test_make_searcher_factory():
    """make_searcher dispatches by string name."""
    ev = make_structured_evidence(B=2, C=3, D=1, M=2)
    model = TransE(num_entities=7, num_relations=5, dim=8)

    sbr = make_searcher(
        "exhaustive",
        resolve=lambda s: ev,
        atom_repr=KGEScoreAtom(),
        state_repr=TNormStateRepr("min"),
        traj_repr=TNormTrajRepr("min"),
        query_repr=MaxQueryRepr(),
        model=model,
    )
    queries = torch.randint(1, 7, (2, 3))
    assert "exhaustive" in sbr(queries)


def test_make_searcher_unknown_strategy_raises():
    import pytest
    with pytest.raises(ValueError, match="Unknown strategy"):
        make_searcher("not_a_real_strategy")


def test_multirollout_requires_set_gumbel_scale():
    """MultiRolloutSearcher should reject base Searchers without set_gumbel_scale."""
    import pytest

    class _NoGumbel:
        def __call__(self, queries):
            return {"score": torch.zeros(queries.shape[0])}

    with pytest.raises(TypeError, match="doesn't support set_gumbel_scale"):
        MultiRolloutSearcher(_NoGumbel(), scales=[0.0, 0.3])


def test_multirollout_takes_max_per_mode():
    """MultiRolloutSearcher should keep the elementwise-max score across rollouts."""
    class _ScalarBase:
        def __init__(self):
            self._scale = 0.0

        def __call__(self, queries):
            # Score = scale * 1 (so higher scale → higher score).
            return {"score": torch.full((queries.shape[0],), self._scale)}

        def set_gumbel_scale(self, scale):
            self._scale = scale

    base = _ScalarBase()
    multi = MultiRolloutSearcher(base, scales=[0.1, 0.5, 0.3])
    queries = torch.randint(1, 7, (4, 3))
    out = multi(queries)
    # Max across [0.1, 0.5, 0.3] = 0.5.
    assert (out["score"] == 0.5).all()


def test_direct_searcher_kge_score():
    model = TransE(num_entities=7, num_relations=5, dim=8)
    direct = DirectSearcher(atom_repr=KGEScoreAtom(), model=model, name="kge")
    queries = torch.tensor([[1, 2, 3], [2, 3, 4]])
    out = direct(queries)
    assert "kge" in out
    assert out["kge"].shape == (2,)
    # KGEScoreAtom returns sigmoid scores in [0, 1].
    assert ((out["kge"] >= 0) & (out["kge"] <= 1)).all()


def test_searcher_to_scorer_end_to_end():
    """ExhaustiveSearcher → make_scorer_from_searcher → ScoreFn for RankingEvaluator."""
    ev = make_structured_evidence(B=10, C=2, D=1, M=2)  # B = total flat pool size
    model = TransE(num_entities=20, num_relations=5, dim=8)

    sbr = ExhaustiveSearcher(
        resolve=lambda state: ev,
        atom_repr=KGEScoreAtom(),
        state_repr=TNormStateRepr("min"),
        traj_repr=TNormTrajRepr("min"),
        query_repr=MaxQueryRepr(),
        model=model,
        name="sbr",
    )
    score_fn = make_scorer_from_searcher(sbr, "sbr")
    B, P = 5, 2  # B*P = 10 matches the FakeProofEvidence batch dim.
    q_buf = torch.randint(1, 20, (B, 3))
    pool_buf = torch.randint(1, 20, (B, P))
    scores = score_fn(q_buf, pool_buf, "tail")
    assert scores.shape == (B, P)
