"""Repr container shape validation tests."""
from __future__ import annotations

import pytest
import torch

from kge_kernels.framework import Repr


def test_repr_embeddings_only():
    emb = torch.randn(2, 3, 4)
    r = Repr(embeddings=emb)
    assert r.has_embeddings and not r.has_scores
    assert r.leading_shape == (2, 3)


def test_repr_scores_only():
    sc = torch.randn(2, 3)
    r = Repr(scores=sc)
    assert r.has_scores and not r.has_embeddings
    assert r.leading_shape == (2, 3)


def test_repr_both_aligned():
    r = Repr(embeddings=torch.randn(2, 3, 4), scores=torch.randn(2, 3))
    assert r.leading_shape == (2, 3)


def test_repr_misaligned_raises():
    with pytest.raises(ValueError, match="leading shape mismatch"):
        Repr(embeddings=torch.randn(2, 3, 4), scores=torch.randn(2, 4))


def test_repr_empty_raises():
    with pytest.raises(ValueError, match="at least one"):
        Repr()
