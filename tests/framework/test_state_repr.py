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

from .conftest import make_legacy_evidence, make_structured_evidence


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


def test_sum_state_repr_scores_polymorphic():
    """Score-path: SumStateRepr should reduce ``scores`` when no embeddings."""
    ev = make_legacy_evidence(B=1, C=1, G_body=3)
    ev.body_count = torch.tensor([[2]], dtype=torch.long)  # only 2 atoms valid
    scores = torch.tensor([[[0.4, 0.6, 99.0]]])  # 99.0 is masked
    out = SumStateRepr()(Repr(scores=scores), ev)
    assert out.scores.shape == (1, 1)
    assert out.embeddings is None
    assert torch.allclose(out.scores[0, 0], torch.tensor(1.0))


def test_mean_state_repr_scores_polymorphic():
    """Score-path: MeanStateRepr should reduce ``scores`` when no embeddings."""
    ev = make_legacy_evidence(B=1, C=1, G_body=3)
    ev.body_count = torch.tensor([[2]], dtype=torch.long)
    scores = torch.tensor([[[0.4, 0.6, 99.0]]])
    out = MeanStateRepr()(Repr(scores=scores), ev)
    assert out.scores.shape == (1, 1)
    assert out.embeddings is None
    assert torch.allclose(out.scores[0, 0], torch.tensor(0.5))


def test_mean_state_repr_scores_empty_body():
    """Empty body (body_count=0): mean should be 0, not NaN."""
    ev = make_legacy_evidence(B=1, C=1, G_body=2)
    ev.body_count = torch.tensor([[0]], dtype=torch.long)
    scores = torch.tensor([[[1.0, 2.0]]])
    out = MeanStateRepr()(Repr(scores=scores), ev)
    assert torch.allclose(out.scores[0, 0], torch.tensor(0.0))


def test_body_atom_mask_flat_override_supports_interspersed_padding():
    """Per-atom validity mask honors body_atom_mask_flat for non-prefix layouts.

    DpRL's derived_states pack atoms with padding in arbitrary positions
    (not always trailing). When ``body_atom_mask_flat`` matches the atom
    leading shape, ``_per_atom_validity_mask`` uses it directly instead
    of deriving a prefix mask from ``body_count``.
    """
    from dataclasses import dataclass

    @dataclass
    class _CustomEvidence:
        body_count: torch.Tensor      # [B, C]
        body_atom_mask_flat: torch.Tensor  # [B, C, G_body]

    # Atoms at positions 0, 2 are valid; position 1 is padded; position 3 is valid.
    mask = torch.tensor([[[True, False, True, True]]])
    body_count = mask.sum(dim=-1).long()  # [B, C] = [[3]]
    ev = _CustomEvidence(body_count=body_count, body_atom_mask_flat=mask)
    scores = torch.tensor([[[0.5, 99.0, 0.7, 0.3]]])

    out = SumStateRepr()(Repr(scores=scores), ev)
    # Sum over valid atoms: 0.5 + 0.7 + 0.3 = 1.5
    assert torch.allclose(out.scores[0, 0], torch.tensor(1.5))
    out_mean = MeanStateRepr()(Repr(scores=scores), ev)
    # Mean over 3 valid atoms.
    assert torch.allclose(out_mean.scores[0, 0], torch.tensor(0.5))
