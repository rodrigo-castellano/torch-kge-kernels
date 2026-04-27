"""Searcher protocol + make_scorer_from_searcher adapter tests."""
from __future__ import annotations

import torch

from kge_kernels.framework.select import BeamSelect, ExhaustiveSelect
from kge_kernels.search import (
    DirectSearcher,
    MultiRolloutSearcher,
    SearchSpec,
    Searcher,
    UnifiedSearcher,
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
    """The adapter must reshape K-major flat scores back to [B, C]."""
    class _FakeSearcher:
        def __call__(self, queries: torch.Tensor):
            # queries: [N, 3]; return one score per query.
            return {"score": queries.sum(dim=-1).float()}

        def set_gumbel_scale(self, scale: float):
            pass

    searcher = _FakeSearcher()
    scorer = make_scorer_from_searcher(searcher, "score")

    B, C = 4, 5
    q_buf = torch.randint(1, 10, (B, 3))
    pool_buf = torch.randint(1, 10, (B, C))

    scores = scorer(q_buf, pool_buf, "tail")
    assert scores.shape == (B, C)

    # Verify the K-major flat → [B, C] reshape: scores[b, k] equals
    # the sum of the triple constructed from (q_buf[b], pool_buf[b, k]).
    for b in range(B):
        for k in range(C):
            expected = q_buf[b, 0] + q_buf[b, 1] + pool_buf[b, k]   # tail mode replaces col 2
            assert scores[b, k].item() == expected.item()


def test_make_scorer_head_mode_replaces_col1():
    class _FakeSearcher:
        def __call__(self, queries):
            return {"score": queries.float().sum(dim=-1)}

        def set_gumbel_scale(self, scale): pass

    scorer = make_scorer_from_searcher(_FakeSearcher(), "score")
    B, C = 2, 3
    q_buf = torch.tensor([[10, 20, 30], [40, 50, 60]])
    pool_buf = torch.tensor([[1, 2, 3], [4, 5, 6]])
    scores = scorer(q_buf, pool_buf, "head")
    # head mode replaces col 1 (subj). For (b=0, k=0): pred=10, subj=1, obj=30 → sum=41.
    assert scores[0, 0].item() == 10 + 1 + 30
    assert scores[1, 2].item() == 40 + 6 + 60


def test_unified_exhaustive_sbr_tuple():
    """SBR per framework.pdf §11: Enum + KGEScore + TNorm(min) + Exhaustive + TNorm(min) + Max."""
    ev = make_structured_evidence(B=2, P=3, D=2, M=2)
    model = TransE(num_entities=7, num_relations=5, dim=8)

    sbr = make_searcher(
        "exhaustive",
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
    ev = make_structured_evidence(B=2, P=3, D=1, M=2)
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


def test_beam_unified_gumbel_scale_zero_bit_exact():
    """UnifiedSearcher+BeamSelect with gumbel_scale_buf zero must match the no-buf baseline."""
    ev = make_structured_evidence(B=4, P=4, D=1, M=2, seed=11)
    model = TransE(num_entities=7, num_relations=5, dim=8)
    queries = torch.randint(1, 7, (4, 3), generator=torch.Generator().manual_seed(0))

    common_kwargs = dict(
        resolve=lambda state: ev,
        atom_repr=KGEScoreAtom(),
        state_repr=TNormStateRepr("min"),
        traj_repr=TNormTrajRepr("min"),
        query_repr=MaxQueryRepr(),
        spec=SearchSpec(batch_size=4, max_depth=1, beam_width=2),
        model=model,
        name="beam",
        capture="dynamic",
    )

    base_no_buf = UnifiedSearcher(select=BeamSelect(k=2), **common_kwargs)

    buf = torch.zeros(())
    base_with_buf = UnifiedSearcher(
        select=BeamSelect(k=2, gumbel_scale_buf=buf), **common_kwargs,
    )

    out_ref = base_no_buf(queries)
    out_zero = base_with_buf(queries)
    assert torch.equal(out_ref["beam"], out_zero["beam"])


def test_unified_beam_gumbel_buf_is_shared_with_select():
    """UnifiedSearcher.set_gumbel_scale forwards to BeamSelect's buf."""
    buf = torch.zeros(())
    model = TransE(num_entities=7, num_relations=5, dim=8)
    searcher = UnifiedSearcher(
        resolve=lambda s: make_structured_evidence(B=2, P=3, D=1, M=2),
        atom_repr=KGEScoreAtom(),
        state_repr=TNormStateRepr("min"),
        select=BeamSelect(k=2, gumbel_scale_buf=buf),
        traj_repr=TNormTrajRepr("min"),
        query_repr=MaxQueryRepr(),
        spec=SearchSpec(batch_size=2, max_depth=1, beam_width=2),
        model=model,
        capture="dynamic",
    )
    import pytest

    assert searcher.select.gumbel_scale_buf is buf
    searcher.set_gumbel_scale(0.42)
    assert buf.item() == pytest.approx(0.42)


def test_multirollout_beam_gumbel_changes_select_output():
    """End-to-end gate: MultiRolloutSearcher cycling scales mutates the
    same buf that BeamSelect reads; chosen indices change at scale > 0.

    Verified through direct ``select`` invocation rather than
    ``__call__`` because ``search_and_score`` with ``max_depth=1`` does
    not propagate ``select.info`` (the exhaustive shortcut). The wire
    fix is what enables future sequential Searchers to use gumbel.
    """
    import pytest

    torch.manual_seed(0)
    buf = torch.zeros(())
    model = TransE(num_entities=7, num_relations=5, dim=8)
    base = UnifiedSearcher(
        resolve=lambda s: make_structured_evidence(B=4, P=4, D=1, M=2),
        atom_repr=KGEScoreAtom(),
        state_repr=TNormStateRepr("min"),
        select=BeamSelect(k=2, gumbel_scale_buf=buf),
        traj_repr=TNormTrajRepr("min"),
        query_repr=MaxQueryRepr(),
        spec=SearchSpec(batch_size=4, max_depth=1, beam_width=2),
        model=model,
        capture="dynamic",
    )
    multi = MultiRolloutSearcher(base, scales=[0.0, 5.0])

    # Drive the buf through MultiRolloutSearcher to confirm the wire.
    multi.set_gumbel_scale(0.0)
    assert buf.item() == 0.0
    multi.set_gumbel_scale(5.0)
    assert buf.item() == pytest.approx(5.0)

    # With buf=5.0, BeamSelect's chosen indices should differ from buf=0.
    from kge_kernels.framework import Repr

    raw_scores = torch.zeros(8, 4)   # ties → noise dominates
    base.set_gumbel_scale(0.0)
    _, info_zero = base.select(evidence=None, s_repr=Repr(scores=raw_scores))
    base.set_gumbel_scale(5.0)
    _, info_noisy = base.select(evidence=None, s_repr=Repr(scores=raw_scores))
    assert not torch.equal(info_zero.chosen_indices, info_noisy.chosen_indices)


def test_searcher_to_scorer_end_to_end():
    """UnifiedSearcher (exhaustive) → make_scorer_from_searcher → ScoreFn for RankingEvaluator."""
    ev = make_structured_evidence(B=10, P=2, D=1, M=2)  # B = total flat pool size
    model = TransE(num_entities=20, num_relations=5, dim=8)

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
    score_fn = make_scorer_from_searcher(sbr, "sbr")
    B, C = 5, 2  # B*C = 10 matches the FakeProofEvidence batch dim.
    q_buf = torch.randint(1, 20, (B, 3))
    pool_buf = torch.randint(1, 20, (B, C))
    scores = score_fn(q_buf, pool_buf, "tail")
    assert scores.shape == (B, C)
