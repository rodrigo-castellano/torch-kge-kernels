"""Tests for kge_kernels.eval.Evaluator (exhaustive filtered ranking mode)."""
from __future__ import annotations

import torch

from kge_kernels.data import build_filter_maps
from kge_kernels.eval import Evaluator
from kge_kernels.models import TransE


def _toy_kg():
    """Tiny KG with 5 entities and 2 relations."""
    num_entities = 5
    num_relations = 2
    # Train triples (r, h, t)
    train = [(0, 0, 1), (0, 1, 2), (0, 2, 3), (1, 0, 4), (1, 2, 4)]
    # Test triples (disjoint from train)
    test = [(0, 3, 4), (1, 1, 3)]
    return num_entities, num_relations, train, test


def _trained_model(num_entities: int, num_relations: int, seed: int = 0) -> TransE:
    torch.manual_seed(seed)
    return TransE(num_entities=num_entities, num_relations=num_relations, dim=4)


def test_evaluate_exhaustive_empty_returns_zeros():
    num_entities, num_relations, _, _ = _toy_kg()
    model = _trained_model(num_entities, num_relations)
    evaluator = Evaluator(model, num_entities, device=torch.device("cpu"))
    out = evaluator.evaluate(torch.zeros(0, 3, dtype=torch.long))
    assert out["MRR"] == 0.0


def test_evaluate_exhaustive_returns_dict():
    num_entities, num_relations, train, test = _toy_kg()
    head_filter, tail_filter = build_filter_maps(train, test)
    model = _trained_model(num_entities, num_relations)
    evaluator = Evaluator(
        model, num_entities,
        head_filter=head_filter, tail_filter=tail_filter,
        device=torch.device("cpu"),
    )
    test_t = torch.tensor(test, dtype=torch.long)
    out = evaluator.evaluate(test_t)
    assert set(out.keys()) == {"MRR", "Hits@1", "Hits@3", "Hits@10"}
    assert 0.0 <= out["MRR"] <= 1.0


def test_evaluate_preserves_training_mode():
    num_entities, num_relations, train, test = _toy_kg()
    head_filter, tail_filter = build_filter_maps(train)
    model = _trained_model(num_entities, num_relations)
    model.train()
    evaluator = Evaluator(
        model, num_entities,
        head_filter=head_filter, tail_filter=tail_filter,
        device=torch.device("cpu"),
    )
    evaluator.evaluate(torch.tensor(test, dtype=torch.long))
    assert model.training


def test_evaluate_with_domain_constraint():
    num_entities, num_relations, train, test = _toy_kg()
    head_filter, tail_filter = build_filter_maps(train)
    model = _trained_model(num_entities, num_relations)
    evaluator = Evaluator(
        model, num_entities,
        head_filter=head_filter, tail_filter=tail_filter,
        head_domain={0: {0, 1, 2, 3, 4}, 1: {0, 1, 2, 3, 4}},
        tail_domain={0: {0, 1, 2, 3, 4}, 1: {0, 1, 2, 3, 4}},
        device=torch.device("cpu"),
    )
    out = evaluator.evaluate(torch.tensor(test, dtype=torch.long))
    assert out["MRR"] >= 0.0
