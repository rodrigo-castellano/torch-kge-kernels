"""QueryRepr reduction tests."""
from __future__ import annotations

import math

import torch

from kge_kernels.framework import (
    LogSumExpQueryRepr,
    MLPSumQueryRepr,
    MaxQueryRepr,
    MeanQueryRepr,
    Repr,
    SumQueryRepr,
)

from .conftest import FakeProofEvidence


def _evidence_with_mask(mask: torch.Tensor) -> FakeProofEvidence:
    """Build a minimal evidence object that only needs ``mask``."""
    B, C = mask.shape
    return FakeProofEvidence(
        body=torch.zeros(B, C, 1, 1, 3, dtype=torch.long),
        mask=mask.to(torch.bool),
        count=mask.sum(dim=-1),
        rule_idx=torch.zeros(B, C, 1, dtype=torch.long),
        body_count=torch.zeros(B, C, 1, dtype=torch.long),
        D=1, M=1,
    )


def test_max_query_repr_respects_mask():
    ev = _evidence_with_mask(torch.tensor([[1, 1, 0]]))
    scores = torch.tensor([[0.1, 0.7, 0.99]])  # last is masked out
    out = MaxQueryRepr()(Repr(scores=scores), ev)
    assert torch.allclose(out.scores[0], torch.tensor(0.7))


def test_sum_query_repr_respects_mask():
    ev = _evidence_with_mask(torch.tensor([[1, 1, 0]]))
    scores = torch.tensor([[0.1, 0.2, 99.0]])
    out = SumQueryRepr()(Repr(scores=scores), ev)
    assert torch.allclose(out.scores[0], torch.tensor(0.3))


def test_mean_query_repr_respects_mask():
    ev = _evidence_with_mask(torch.tensor([[1, 1, 0]]))
    scores = torch.tensor([[0.2, 0.4, 99.0]])
    out = MeanQueryRepr()(Repr(scores=scores), ev)
    assert torch.allclose(out.scores[0], torch.tensor(0.3))


def test_logsumexp_query_repr():
    ev = _evidence_with_mask(torch.tensor([[1, 1]]))
    scores = torch.tensor([[0.0, 0.0]])
    out = LogSumExpQueryRepr()(Repr(scores=scores), ev)
    assert torch.allclose(out.scores[0], torch.tensor(math.log(2.0)), atol=1e-5)


def test_mlp_sum_query_repr_shape():
    ev = _evidence_with_mask(torch.tensor([[1, 1, 1]]))
    emb = torch.randn(1, 3, 4)
    out = MLPSumQueryRepr(embed_dim=4)(Repr(embeddings=emb), ev)
    assert out.scores.shape == (1,)
