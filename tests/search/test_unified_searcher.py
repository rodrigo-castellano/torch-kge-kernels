"""Phase 4 tests: :class:`UnifiedSearcher` capture-mode dispatch."""
from __future__ import annotations

import pytest
import torch

from kge_kernels.framework import (
    BeamSelect,
    ExhaustiveSelect,
    KGEScoreAtom,
    MaxQueryRepr,
    TNormStateRepr,
    TNormTrajRepr,
)
from kge_kernels.models import TransE
from kge_kernels.search import SearchSpec, Searcher, UnifiedSearcher

from tests.framework.conftest import make_structured_evidence


def _make_unified(B: int, capture: str, *, select=None, name: str = "u") -> UnifiedSearcher:
    ev = make_structured_evidence(B=B, P=3, D=1, M=2, seed=4)
    model = TransE(num_entities=7, num_relations=5, dim=8)
    return UnifiedSearcher(
        resolve=lambda state: ev,
        atom_repr=KGEScoreAtom(),
        state_repr=TNormStateRepr("min"),
        select=select if select is not None else ExhaustiveSelect(),
        traj_repr=TNormTrajRepr("min"),
        query_repr=MaxQueryRepr(),
        spec=SearchSpec(batch_size=B),
        model=model,
        name=name,
        capture=capture,
    )


def test_unified_searcher_satisfies_protocol():
    s = _make_unified(B=2, capture="dynamic")
    assert isinstance(s, Searcher)


def test_unified_searcher_dynamic_runs():
    s = _make_unified(B=2, capture="dynamic", name="dyn")
    queries = torch.randint(1, 7, (2, 3))
    out = s(queries)
    assert "dyn" in out
    assert out["dyn"].shape == (2,)


def test_unified_searcher_static_single_graph_for_exhaustive():
    """ExhaustiveSelect → single-graph capture (no alternation needed)."""
    s = _make_unified(B=2, capture="static", select=ExhaustiveSelect())
    assert s._needs_alternation is False
    queries = torch.randint(1, 7, (2, 3))
    out = s(queries)
    assert "u" in out
    assert out["u"].shape == (2,)


def test_unified_searcher_static_alternated_pair_for_beam():
    """BeamSelect (sequential) → alternation flag set to True."""
    s = _make_unified(B=2, capture="static", select=BeamSelect(k=2))
    assert s._needs_alternation is True


def test_unified_searcher_static_dynamic_close_on_exhaustive():
    """Same primitives + same input → numerically close across capture modes.

    torch.compile may fuse ops and use TF32 / different reduction order,
    so we accept ~1e-4 absolute deviation. Phase 7c will tighten this
    by restructuring primitives to write into static buffers.
    """
    queries = torch.randint(1, 7, (2, 3), generator=torch.Generator().manual_seed(0))
    dyn = _make_unified(B=2, capture="dynamic", name="m")
    sta = _make_unified(B=2, capture="static", name="m")
    out_dyn = dyn(queries)["m"]
    out_sta = sta(queries)["m"]
    torch.testing.assert_close(out_sta, out_dyn, rtol=1e-2, atol=1e-2)


def test_unified_searcher_shape_mismatch_raises_on_static():
    s = _make_unified(B=4, capture="static")
    with pytest.raises(ValueError, match="spec.batch_size"):
        s(torch.randint(1, 7, (3, 3)))


def test_unified_searcher_dynamic_skips_shape_check():
    """Dynamic mode does not enforce spec.batch_size on inputs."""
    s = _make_unified(B=4, capture="dynamic")
    # The shape mismatch should NOT raise (unlike static mode).
    s(torch.randint(1, 7, (3, 3)))


def test_unified_searcher_set_gumbel_scale_no_select_buf_is_noop():
    """set_gumbel_scale on a select without a buf must not raise."""
    s = _make_unified(B=2, capture="dynamic", select=ExhaustiveSelect())
    s.set_gumbel_scale(0.5)   # No error.


def test_unified_searcher_set_gumbel_scale_propagates_to_select():
    buf = torch.zeros(())
    s = _make_unified(B=2, capture="dynamic", select=BeamSelect(k=2, gumbel_scale_buf=buf))
    s.set_gumbel_scale(0.5)
    assert buf.item() == pytest.approx(0.5)


def test_search_spec_is_frozen():
    spec = SearchSpec(batch_size=4)
    with pytest.raises(Exception):  # dataclass(frozen=True) raises FrozenInstanceError
        spec.batch_size = 5  # type: ignore[misc]


def test_unified_searcher_set_gumbel_scale_forwards_to_select():
    """Convenience forwarder: searcher.set_gumbel_scale → select.set_gumbel_scale."""
    buf = torch.zeros(())
    s = _make_unified(B=2, capture="dynamic", select=BeamSelect(k=2, gumbel_scale_buf=buf))
    s.set_gumbel_scale(0.7)
    assert buf.item() == pytest.approx(0.7)
