"""Smoke tests for the KGE models in kge_kernels.models."""
from __future__ import annotations

import pytest
import torch

from kge_kernels.models import ComplEx, DistMult, KGEModel, ModE, RotatE, TransE, TuckER

MODELS_2D = [
    ("transe", lambda: TransE(num_entities=7, num_relations=5, dim=8)),
    ("distmult", lambda: DistMult(num_entities=7, num_relations=5, dim=8)),
    ("mode", lambda: ModE(num_entities=7, num_relations=5, dim=8)),
    ("complex", lambda: ComplEx(num_entities=7, num_relations=5, dim=8)),
    ("rotate", lambda: RotatE(num_entities=7, num_relations=5, dim=8)),
    ("tucker", lambda: TuckER(num_entities=7, num_relations=5, entity_dim=8)),
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
