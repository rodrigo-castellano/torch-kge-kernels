"""Compile-safety smoke tests for the framework primitives.

The ``framework.tex`` spec and ``ethereal-launching-hollerith.md`` plan
both describe fullgraph compile-safety as a design goal for the framework
primitives: SBR-style exhaustive scoring should compose cleanly into a
single ``torch.compile(fullgraph=True)`` region without graph breaks,
so that DPrL's compiled CUDA-graph closures can import the primitives
wherever they are free.

This module wraps ``search_and_score`` + the most common primitive
compositions in ``torch.compile(fullgraph=True, dynamic=False)`` and
asserts that:

  1. The compiled function runs successfully and agrees with the eager
     version (numerical parity, modulo float32 noise).
  2. ``torch._dynamo.explain`` reports zero graph breaks.

The tests are parameterized over the main exhaustive SBR composition
(``KGEScoreAtom`` + ``TNormStateRepr`` + ``TNormTrajRepr`` +
``MaxQueryRepr`` + ``ExhaustiveSelect``) because that is the
composition torch-ns actually consumes. The sequential variants
(``GreedySelect`` / ``BeamSelect`` / ``SampleSelect``) are NOT tested
here because their Python-side tree / loop orchestration is
intentionally outside the compile boundary.

Failure modes that would indicate a regression:

- A new primitive introduces ``.item()``, ``.tolist()``, ``.cpu()``, or
  a ``bool(tensor)`` call, which triggers a graph break.
- A mask built with ``torch.nonzero(...)`` introduces dynamic shapes.
- Any primitive tries to pickle a closure or capture a Python-side
  counter inside the forward.
"""
from __future__ import annotations

import pytest
import torch

from kge_kernels.framework import (
    ExhaustiveSelect,
    KGEScoreAtom,
    MaxQueryRepr,
    TNormStateRepr,
    TNormTrajRepr,
    search_and_score,
)
from kge_kernels.models import TransE

from .conftest import make_structured_evidence

_HAS_DYNAMO = hasattr(torch, "_dynamo")


def _build_composition():
    """Build the canonical SBR-exhaustive composition used in the tests."""
    return {
        "atom_repr": KGEScoreAtom(),
        "state_repr": TNormStateRepr("min"),
        "traj_repr": TNormTrajRepr("min"),
        "query_repr": MaxQueryRepr(),
        "select": ExhaustiveSelect(),
    }


def _run_scorer(model, evidence, comp):
    def fake_resolve(state):
        return evidence

    return search_and_score(
        query=None,
        resolve=fake_resolve,
        model=model,
        max_depth=1,
        **comp,
    )


@pytest.mark.skipif(not _HAS_DYNAMO, reason="torch._dynamo not available")
def test_search_and_score_compile_fullgraph_no_graph_breaks():
    """Wrap the canonical SBR composition in ``torch.compile(fullgraph=True)``
    and assert it produces a single graph with no breaks."""
    import torch._dynamo as dynamo  # type: ignore[attr-defined]

    ev = make_structured_evidence(B=2, C=3, D=2, M=2)
    model = TransE(num_entities=7, num_relations=5, dim=8)
    comp = _build_composition()

    # Eager baseline
    eager_scores = _run_scorer(model, ev, comp)

    # Reset dynamo state so the test is deterministic across runs.
    dynamo.reset()

    # Compile a thin wrapper that hides the resolve closure so dynamo
    # sees a standard function boundary.
    def run(query):
        return _run_scorer(model, ev, comp)

    # Use explain to get a structured report on graph-break reasons.
    explanation = dynamo.explain(run)(None)
    graph_break_count = getattr(
        explanation, "graph_break_count", len(getattr(explanation, "break_reasons", []))
    )
    assert graph_break_count == 0, (
        f"search_and_score produced {graph_break_count} graph break(s); "
        f"reasons: {getattr(explanation, 'break_reasons', '?')}"
    )

    # Run the compiled version and check numerical parity.
    compiled = torch.compile(run, fullgraph=True, dynamic=False)
    compiled_scores = compiled(None)
    assert torch.allclose(compiled_scores, eager_scores, atol=1e-5), (
        f"Compiled vs eager mismatch: "
        f"compiled={compiled_scores} eager={eager_scores}"
    )


@pytest.mark.skipif(not _HAS_DYNAMO, reason="torch._dynamo not available")
def test_tnorm_state_repr_is_fullgraph_compilable():
    """A smaller, more targeted compile smoke test on the state_repr primitive."""
    import torch._dynamo as dynamo  # type: ignore[attr-defined]

    from kge_kernels.framework import Repr

    ev = make_structured_evidence(B=2, C=3, D=2, M=2)
    state_repr = TNormStateRepr("min")

    scores = torch.rand(2, 3, 2, 2)

    def run(x):
        return state_repr(Repr(scores=x), ev).scores

    dynamo.reset()
    explanation = dynamo.explain(run)(scores)
    breaks = getattr(
        explanation, "graph_break_count", len(getattr(explanation, "break_reasons", []))
    )
    assert breaks == 0, (
        f"TNormStateRepr produced {breaks} graph break(s); "
        f"reasons: {getattr(explanation, 'break_reasons', '?')}"
    )


@pytest.mark.skipif(not _HAS_DYNAMO, reason="torch._dynamo not available")
def test_max_query_repr_is_fullgraph_compilable():
    """Targeted compile smoke test on the query_repr primitive."""
    import torch._dynamo as dynamo  # type: ignore[attr-defined]

    from kge_kernels.framework import Repr

    from .conftest import FakeProofEvidence

    mask = torch.tensor([[True, True, False], [True, False, False]])
    ev = FakeProofEvidence(
        body=torch.zeros(2, 3, 1, 1, 3, dtype=torch.long),
        mask=mask,
        count=mask.sum(dim=-1),
        rule_idx=torch.zeros(2, 3, 1, dtype=torch.long),
        body_count=torch.zeros(2, 3, 1, dtype=torch.long),
        D=1,
        M=1,
    )
    q = MaxQueryRepr()

    def run(s):
        return q(Repr(scores=s), ev).scores

    dynamo.reset()
    explanation = dynamo.explain(run)(torch.rand(2, 3))
    breaks = getattr(
        explanation, "graph_break_count", len(getattr(explanation, "break_reasons", []))
    )
    assert breaks == 0, (
        f"MaxQueryRepr produced {breaks} graph break(s); "
        f"reasons: {getattr(explanation, 'break_reasons', '?')}"
    )
