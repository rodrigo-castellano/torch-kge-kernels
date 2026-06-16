"""Smoke tests for the KGE models in kge_kernels.models."""
from __future__ import annotations

import pytest
import torch

from kge_kernels.models import ComplEx, ConvE, DistMult, KGEModel, ModE, RotatE, TransE, TuckER

MODELS_2D = [
    ("transe", lambda: TransE(num_entities=7, num_relations=5, dim=8)),
    ("distmult", lambda: DistMult(num_entities=7, num_relations=5, dim=8)),
    ("mode", lambda: ModE(num_entities=7, num_relations=5, dim=8)),
    ("complex", lambda: ComplEx(num_entities=7, num_relations=5, dim=8)),
    ("rotate", lambda: RotatE(num_entities=7, num_relations=5, dim=8)),
    ("tucker", lambda: TuckER(num_entities=7, num_relations=5, entity_dim=8)),
    (
        "conve",
        lambda: ConvE(
            num_entities=7, num_relations=5, dim=200,
            embedding_height=10, embedding_width=20,
        ),
    ),
]


@pytest.mark.parametrize("name,builder", MODELS_2D)
def test_score_triples_shape(name, builder):
    torch.manual_seed(0)
    model: KGEModel = builder()
    h = torch.tensor([0, 1, 2])
    r = torch.tensor([0, 1, 2])
    t = torch.tensor([3, 4, 5])
    out = model.score(h, r, t)
    assert out.shape == (3,)


@pytest.mark.parametrize("name,builder", MODELS_2D)
def test_score_all_tails_shape(name, builder):
    torch.manual_seed(0)
    model: KGEModel = builder()
    h = torch.tensor([0, 1, 2])
    r = torch.tensor([0, 1, 2])
    out = model.score(h, r, None)
    assert out.shape == (3, model.num_entities)


@pytest.mark.parametrize("name,builder", MODELS_2D)
def test_score_all_heads_shape(name, builder):
    torch.manual_seed(0)
    model: KGEModel = builder()
    r = torch.tensor([0, 1, 2])
    t = torch.tensor([3, 4, 5])
    out = model.score(None, r, t)
    assert out.shape == (3, model.num_entities)


@pytest.mark.parametrize("name,builder", MODELS_2D)
def test_compose_returns_embedding(name, builder):
    torch.manual_seed(0)
    model: KGEModel = builder()
    h = torch.tensor([0, 1, 2])
    r = torch.tensor([0, 1, 2])
    t = torch.tensor([3, 4, 5])
    emb = model.compose(h, r, t)
    assert emb.dim() == 2
    assert emb.shape[0] == 3
    assert emb.shape[1] >= 1


@pytest.mark.parametrize("cls_name", ["rotate", "rotate_ns"])
@pytest.mark.parametrize("p_norm", [1, 2])
def test_rotate_pool_feature(cls_name, p_norm):
    """RotatE/RotatENS pool_feature: per-component modulus, shape [B, half_dim],
    non-negative (the keras-ns RotatE.call feature). The R2N consumer negates
    it, so sigmoid(sum(-feature)) is a proximity score: discriminative over
    distinct triples and <= 0.5 (close → ~0.5, far → ~0), with a zeroed pool
    → sigmoid(0) = 0.5. Contrast the signed compose pool, whose components
    cancel under the sum → ~0.5 everywhere (degenerate)."""
    from kge_kernels.models import build_model
    torch.manual_seed(0)
    model = build_model(name=cls_name, num_entities=12, num_relations=5,
                        dim=8, gamma=6.0, p_norm=p_norm)
    h = torch.tensor([0, 1, 2, 3]); r = torch.tensor([0, 1, 2, 3]); t = torch.tensor([4, 5, 6, 7])
    feat = model.pool_feature(h, r, t)
    assert feat.shape == (4, model.half_dim)
    assert (feat >= 0).all()
    score = torch.sigmoid((-feat).sum(-1))          # the R2N head sees -feature
    assert ((score >= 0) & (score <= 0.5 + 1e-6)).all()
    assert score.std() > 1e-4                       # discriminative, not degenerate
    # zeroed pool (no-firing, zero_unwritten) → neutral 0.5
    z = torch.sigmoid((-torch.zeros_like(feat)).sum(-1))
    assert torch.allclose(z, torch.full_like(z, 0.5))


def test_transe_score_triples_value():
    """Sanity check: TransE score is -|| h+r-t || (≤ 0)."""
    torch.manual_seed(0)
    model = TransE(num_entities=4, num_relations=2, dim=4, p_norm=2)
    h = torch.tensor([0])
    r = torch.tensor([0])
    t = torch.tensor([0])
    out = model.score(h, r, t)
    assert out.item() <= 1e-6


def test_score_all_tails_consistent_with_triples():
    """score(h, r, None)[t] should equal score(h, r, t)."""
    torch.manual_seed(0)
    model = TransE(num_entities=5, num_relations=3, dim=4)
    h = torch.tensor([1, 2])
    r = torch.tensor([0, 1])
    t = torch.tensor([3, 4])
    triple = model.score(h, r, t)
    all_tails = model.score(h, r, None)
    gathered = all_tails.gather(1, t.unsqueeze(1)).squeeze(1)
    assert torch.allclose(triple, gathered, atol=1e-5)


def test_score_dispatch_both_none_raises():
    torch.manual_seed(0)
    model = TransE(num_entities=5, num_relations=3, dim=4)
    # Both None hits the entity_embeddings(None) path which raises TypeError.
    with pytest.raises((TypeError, AttributeError)):
        model.score(None, torch.tensor([0]), None)


@pytest.mark.parametrize("p_norm,dim,d_chunk", [(1, 16, 4), (1, 16, 5), (2, 16, 4), (1, 32, 7)])
def test_rotate_dchunked_matches_one_pass(p_norm, dim, d_chunk):
    """RotatE: score(d_chunk=int) must match score(d_chunk=None) (one-pass)."""
    torch.manual_seed(0)
    model = RotatE(num_entities=11, num_relations=5, dim=dim, p_norm=p_norm)
    h = torch.tensor([0, 3, 7, 10])
    r = torch.tensor([0, 2, 4, 1])
    t = torch.tensor([1, 5, 9, 0])

    direct_tails = model.score(h, r, None)
    chunked_tails = model.score(h, r, None, d_chunk=d_chunk)
    assert chunked_tails.shape == direct_tails.shape
    assert torch.allclose(chunked_tails, direct_tails, atol=1e-5, rtol=1e-5)

    direct_heads = model.score(None, r, t)
    chunked_heads = model.score(None, r, t, d_chunk=d_chunk)
    assert chunked_heads.shape == direct_heads.shape
    assert torch.allclose(chunked_heads, direct_heads, atol=1e-5, rtol=1e-5)


def test_forward_delegates_to_score():
    torch.manual_seed(0)
    model = TransE(num_entities=5, num_relations=3, dim=4)
    h = torch.tensor([1])
    r = torch.tensor([0])
    assert torch.allclose(model(h, r, None), model.score(h, r, None))
