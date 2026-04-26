"""Tests for the filtered ranking flow via RankingEvaluator + SamplerCandidates."""
from __future__ import annotations

import torch

from kge_kernels.eval import RankingEvaluator, SamplerCandidates
from kge_kernels.models import kge_default_scorer
from kge_kernels.models import TransE
from kge_kernels.scoring import Sampler


def _toy_kg():
    num_entities = 5
    num_relations = 2
    train = [(0, 0, 1), (0, 1, 2), (0, 2, 3), (1, 0, 4), (1, 2, 4)]
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


def _build(model, sampler, num_entities, *, k=None, batch_size=4, **kw):
    candidates = SamplerCandidates(sampler, k=k, **kw)
    return RankingEvaluator(
        scorer=lambda q, p, m: kge_default_scorer(model, q, p, m),
        candidates=candidates,
        batch_size=batch_size, device=torch.device("cpu"), compile=False,
    )


def test_evaluate_exhaustive_empty_returns_zeros():
    num_entities, num_relations, train, _ = _toy_kg()
    model = _trained_model(num_entities, num_relations)
    sampler = _make_sampler(num_entities, num_relations, train)
    ev = _build(model, sampler, num_entities)
    result = ev.evaluate(torch.zeros(0, 3, dtype=torch.long))
    assert result.metrics()["MRR"] == 0.0


def test_evaluate_exhaustive_returns_dict():
    num_entities, num_relations, train, test = _toy_kg()
    model = _trained_model(num_entities, num_relations)
    sampler = _make_sampler(num_entities, num_relations, train + test)
    ev = _build(model, sampler, num_entities)
    out = ev.evaluate(torch.tensor(test, dtype=torch.long)).metrics()
    assert set(out.keys()) == {"MRR", "Hits@1", "Hits@3", "Hits@10"}
    assert 0.0 <= out["MRR"] <= 1.0


def test_evaluate_with_domain_constraint():
    num_entities, num_relations, train, test = _toy_kg()
    model = _trained_model(num_entities, num_relations)
    sampler = _make_sampler(num_entities, num_relations, train)
    full_domain = {0: {0, 1, 2, 3, 4}, 1: {0, 1, 2, 3, 4}}
    ev = _build(model, sampler, num_entities,
                head_domain=full_domain, tail_domain=full_domain)
    out = ev.evaluate(torch.tensor(test, dtype=torch.long)).metrics()
    assert out["MRR"] >= 0.0
