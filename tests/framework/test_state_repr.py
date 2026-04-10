"""StateRepr aggregation + masking tests."""
from __future__ import annotations

import torch

from kge_kernels.framework import (
    ConcatStateRepr,
    MaxStateRepr,
    MeanStateRepr,
    Repr,
    SumStateRepr,
    TNormStateRepr,
)

from .conftest import FakeProofEvidence, make_legacy_evidence, make_structured_evidence


def test_tnorm_min_structured():
    ev = make_structured_evidence(B=2, C=3, D=2, M=2)
    scores = torch.tensor(
        [
            [[[0.4, 0.6], [0.2, 0.8]], [[0.7, 0.7], [0.5, 0.5]], [[0.9, 0.9], [0.1, 0.1]]],
            [[[0.3, 0.3], [0.3, 0.3]], [[0.4, 0.4], [0.4, 0.4]], [[0.5, 0.5], [0.5, 0.5]]],
        ],
        dtype=torch.float32,
    )
    out = TNormStateRepr("min")(Repr(scores=scores), ev)
    assert out.scores.shape == (2, 3, 2)
    assert torch.allclose(out.scores[0, 0, 0], torch.tensor(0.4))
    assert torch.allclose(out.scores[0, 2, 1], torch.tensor(0.1))


def test_tnorm_product_structured():
    ev = make_structured_evidence(B=1, C=1, D=1, M=2)
    scores = torch.tensor([[[[0.5, 0.4]]]])
    out = TNormStateRepr("product")(Repr(scores=scores), ev)
    assert torch.allclose(out.scores[0, 0, 0], torch.tensor(0.2))


def test_tnorm_min_legacy():
    ev = make_legacy_evidence(B=1, C=1, G_body=3)
    scores = torch.tensor([[[0.7, 0.2, 0.9]]])
    out = TNormStateRepr("min")(Repr(scores=scores), ev)
    assert out.scores.shape == (1, 1)
    assert torch.allclose(out.scores[0, 0], torch.tensor(0.2))


def test_sum_state_repr_structured_with_mask():
    ev = make_structured_evidence(B=1, C=1, D=1, M=3)
    # Make only first 2 atoms valid
    ev.body_count = torch.tensor([[[2]]], dtype=torch.long)
    emb = torch.tensor([[[[[1.0, 1.0], [2.0, 2.0], [99.0, 99.0]]]]])
    out = SumStateRepr()(Repr(embeddings=emb), ev)
    assert out.embeddings.shape == (1, 1, 1, 2)
    assert torch.allclose(out.embeddings[0, 0, 0], torch.tensor([3.0, 3.0]))


def test_mean_state_repr():
    ev = make_structured_evidence(B=1, C=1, D=1, M=2)
    emb = torch.tensor([[[[[2.0, 4.0], [4.0, 8.0]]]]])
    out = MeanStateRepr()(Repr(embeddings=emb), ev)
    assert torch.allclose(out.embeddings[0, 0, 0], torch.tensor([3.0, 6.0]))


def test_max_state_repr():
    ev = make_structured_evidence(B=1, C=1, D=1, M=3)
    emb = torch.tensor([[[[[1.0, -1.0], [3.0, -3.0], [2.0, 0.0]]]]])
    out = MaxStateRepr()(Repr(embeddings=emb), ev)
    assert torch.allclose(out.embeddings[0, 0, 0], torch.tensor([3.0, 0.0]))


def test_concat_state_repr_pads():
    ev = make_structured_evidence(B=1, C=1, D=1, M=2)
    emb = torch.tensor([[[[[1.0, 2.0], [3.0, 4.0]]]]])
    out = ConcatStateRepr(max_atoms=3)(Repr(embeddings=emb), ev)
    # max_atoms=3, embed_dim=2 → 6
    assert out.embeddings.shape == (1, 1, 1, 6)
    # Atoms 0 and 1 contribute, atom 2 is zero-padded
    assert torch.allclose(
        out.embeddings[0, 0, 0],
        torch.tensor([1.0, 2.0, 3.0, 4.0, 0.0, 0.0]),
    )
