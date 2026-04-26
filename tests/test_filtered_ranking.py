"""Tests for kge_kernels.eval.evaluate (filtered + domain-restricted)."""
from __future__ import annotations

import torch

from kge_kernels.eval import CandidateProvider, evaluate
from kge_kernels.models import TransE
from kge_kernels.scoring import Sampler


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


def _make_sampler(num_entities: int, num_relations: int, known_triples) -> Sampler:
    return Sampler.from_data(
        all_known_triples_idx=torch.tensor(known_triples, dtype=torch.long),
        num_entities=num_entities, num_relations=num_relations,
        device=torch.device("cpu"), min_entity_idx=0,
    )


def test_evaluate_exhaustive_empty_returns_zeros():
    num_entities, num_relations, train, _ = _toy_kg()
    model = _trained_model(num_entities, num_relations)
    sampler = _make_sampler(num_entities, num_relations, train)
    provider = CandidateProvider(sampler, num_entities, k=None)
    out = evaluate(
        model, torch.zeros(0, 3, dtype=torch.long), provider,
        device=torch.device("cpu"), compile=False,
    )
    assert out["MRR"] == 0.0


def test_evaluate_exhaustive_returns_dict():
    num_entities, num_relations, train, test = _toy_kg()
    model = _trained_model(num_entities, num_relations)
    sampler = _make_sampler(num_entities, num_relations, train + test)
    provider = CandidateProvider(sampler, num_entities, k=None)
    out = evaluate(
        model, torch.tensor(test, dtype=torch.long), provider,
        scheme="both", batch_size=4,
        device=torch.device("cpu"), compile=False,
    )
    assert set(out.keys()) == {"MRR", "Hits@1", "Hits@3", "Hits@10"}
    assert 0.0 <= out["MRR"] <= 1.0


def test_evaluate_preserves_training_mode():
    num_entities, num_relations, train, test = _toy_kg()
    model = _trained_model(num_entities, num_relations)
    model.train()
    sampler = _make_sampler(num_entities, num_relations, train)
    provider = CandidateProvider(sampler, num_entities, k=None)
    evaluate(
        model, torch.tensor(test, dtype=torch.long), provider,
        device=torch.device("cpu"), compile=False,
    )
    assert model.training


def test_evaluate_with_domain_constraint():
    num_entities, num_relations, train, test = _toy_kg()
    model = _trained_model(num_entities, num_relations)
    sampler = _make_sampler(num_entities, num_relations, train)
    # Permissive domain mask: every entity is valid for both relations.
    full_domain = {0: {0, 1, 2, 3, 4}, 1: {0, 1, 2, 3, 4}}
    provider = CandidateProvider(
        sampler, num_entities, k=None,
        head_domain=full_domain, tail_domain=full_domain,
    )
    out = evaluate(
        model, torch.tensor(test, dtype=torch.long), provider,
        scheme="both", batch_size=4,
        device=torch.device("cpu"), compile=False,
    )
    assert out["MRR"] >= 0.0
