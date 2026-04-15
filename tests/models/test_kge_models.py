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
    out = model.score_triples(h, r, t)
    assert out.shape == (3,)


@pytest.mark.parametrize("name,builder", MODELS_2D)
def test_score_all_tails_shape(name, builder):
    torch.manual_seed(0)
    model: KGEModel = builder()
    h = torch.tensor([0, 1, 2])
    r = torch.tensor([0, 1, 2])
    out = model.score_all_tails(h, r)
    assert out.shape == (3, model.num_entities)


@pytest.mark.parametrize("name,builder", MODELS_2D)
def test_score_all_heads_shape(name, builder):
    torch.manual_seed(0)
    model: KGEModel = builder()
    r = torch.tensor([0, 1, 2])
    t = torch.tensor([3, 4, 5])
    out = model.score_all_heads(r, t)
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
    # Embed dim is at least 1 (model-specific exact width is checked elsewhere)
    assert emb.shape[1] >= 1


def test_transe_score_triples_value():
    """Sanity check: TransE score is -|| h+r-t || (≤ 0)."""
    torch.manual_seed(0)
    model = TransE(num_entities=4, num_relations=2, dim=4, p_norm=2)
    h = torch.tensor([0])
    r = torch.tensor([0])
    t = torch.tensor([0])
    out = model.score_triples(h, r, t)
    assert out.item() <= 1e-6


def test_score_all_tails_consistent_with_score_triples():
    """score_all_tails(h, r)[t] should equal score_triples(h, r, t)."""
    torch.manual_seed(0)
    model = TransE(num_entities=5, num_relations=3, dim=4)
    h = torch.tensor([1, 2])
    r = torch.tensor([0, 1])
    t = torch.tensor([3, 4])
    triple = model.score_triples(h, r, t)
    all_tails = model.score_all_tails(h, r)
    gathered = all_tails.gather(1, t.unsqueeze(1)).squeeze(1)
    assert torch.allclose(triple, gathered, atol=1e-5)


def test_score_dispatch_specific_triples():
    torch.manual_seed(0)
    model = TransE(num_entities=5, num_relations=3, dim=4)
    h = torch.tensor([1, 2])
    r = torch.tensor([0, 1])
    t = torch.tensor([3, 4])
    via_score = model.score(h, r, t)
    via_triples = model.score_triples(h, r, t)
    assert torch.allclose(via_score, via_triples)


def test_score_dispatch_all_tails_when_t_none():
    torch.manual_seed(0)
    model = TransE(num_entities=5, num_relations=3, dim=4)
    h = torch.tensor([1, 2])
    r = torch.tensor([0, 1])
    via_score = model.score(h, r, None)
    via_all = model.score_all_tails(h, r)
    assert torch.allclose(via_score, via_all)


def test_score_dispatch_all_heads_when_h_none():
    torch.manual_seed(0)
    model = TransE(num_entities=5, num_relations=3, dim=4)
    r = torch.tensor([0, 1])
    t = torch.tensor([3, 4])
    via_score = model.score(None, r, t)
    via_all = model.score_all_heads(r, t)
    assert torch.allclose(via_score, via_all)


def test_score_dispatch_both_none_raises():
    torch.manual_seed(0)
    model = TransE(num_entities=5, num_relations=3, dim=4)
    with pytest.raises(ValueError, match="at least one"):
        model.score(None, torch.tensor([0]), None)


@pytest.mark.parametrize("p_norm,dim,d_chunk", [(1, 16, 4), (1, 16, 5), (2, 16, 4), (1, 32, 7)])
def test_rotate_dchunked_matches_direct(p_norm, dim, d_chunk):
    torch.manual_seed(0)
    model = RotatE(num_entities=11, num_relations=5, dim=dim, p_norm=p_norm)
    h = torch.tensor([0, 3, 7, 10])
    r = torch.tensor([0, 2, 4, 1])
    t = torch.tensor([1, 5, 9, 0])

    direct_tails = model.score_all_tails(h, r)
    chunked_tails = model.score_all_tails_dchunked(h, r, d_chunk=d_chunk)
    assert chunked_tails.shape == direct_tails.shape
    assert torch.allclose(chunked_tails, direct_tails, atol=1e-5, rtol=1e-5)

    direct_heads = model.score_all_heads(r, t)
    chunked_heads = model.score_all_heads_dchunked(r, t, d_chunk=d_chunk)
    assert chunked_heads.shape == direct_heads.shape
    assert torch.allclose(chunked_heads, direct_heads, atol=1e-5, rtol=1e-5)


def test_forward_delegates_to_score():
    torch.manual_seed(0)
    model = TransE(num_entities=5, num_relations=3, dim=4)
    h = torch.tensor([1])
    r = torch.tensor([0])
    assert torch.allclose(model(h, r, None), model.score_all_tails(h, r))
