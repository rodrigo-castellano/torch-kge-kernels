"""Tests for kge_kernels.scoring.helpers (corrupt_with_topup, corrupt_to_lists)."""
from __future__ import annotations

import torch

from kge_kernels.scoring import Sampler, corrupt_to_lists, corrupt_with_topup


def _make_sampler(known=None, num_entities=10, num_relations=2, **kwargs) -> Sampler:
    if known is None:
        known = torch.empty((0, 3), dtype=torch.long)
    return Sampler.from_data(
        all_known_triples_idx=known,
        num_entities=num_entities,
        num_relations=num_relations,
        device=torch.device("cpu"),
        default_mode="tail",
        seed=0,
        min_entity_idx=0,
        **kwargs,
    )


# ═══════════════════════════════════════════════════════════════════════
# corrupt_to_lists
# ═══════════════════════════════════════════════════════════════════════


def test_corrupt_to_lists_drops_invalid_rows():
    neg = torch.tensor([
        [[0, 0, 1], [0, 0, 2], [0, 0, 3]],
        [[1, 0, 0], [1, 0, 1], [1, 0, 2]],
    ], dtype=torch.long)
    valid = torch.tensor([
        [True, False, True],
        [True, True, False],
    ])
    out = corrupt_to_lists(neg, valid)
    assert out == [
        [(0, 0, 1), (0, 0, 3)],
        [(1, 0, 0), (1, 0, 1)],
    ]


def test_corrupt_to_lists_empty_batch():
    neg = torch.empty((0, 3, 3), dtype=torch.long)
    valid = torch.empty((0, 3), dtype=torch.bool)
    assert corrupt_to_lists(neg, valid) == []


# ═══════════════════════════════════════════════════════════════════════
# corrupt_with_topup
# ═══════════════════════════════════════════════════════════════════════


def test_corrupt_with_topup_returns_exactly_k_per_query():
    sampler = _make_sampler()
    queries = [(0, 1, 2), (1, 3, 4)]
    out = corrupt_with_topup(sampler, queries, num_negatives=5, mode="tail")
    assert len(out) == 2
    for row in out:
        assert len(row) == 5


def test_corrupt_with_topup_exhaustive_when_k_is_none():
    """num_negatives=None falls through to a single corrupt_with_mask call
    and returns variable-length valid rows (no padding/topup)."""
    sampler = _make_sampler()
    queries = [(0, 1, 2)]
    out = corrupt_with_topup(sampler, queries, num_negatives=None, mode="tail")
    assert len(out) == 1
    # Exhaustive on 10 entities with 1 known positive removed → 9 candidates max.
    assert 0 < len(out[0]) <= 9
    for triple in out[0]:
        assert isinstance(triple, tuple)
        assert len(triple) == 3


def test_corrupt_with_topup_handles_small_domain_pool():
    """Small per-domain pool shouldn't leave rows short of K — topup fills them."""
    # 4 entities split into two 2-entity domains; relation 0 head/tail both in 'a'.
    known = torch.tensor([[0, 0, 2]], dtype=torch.long)
    sampler = _make_sampler(
        known=known, num_entities=4,
        domain2idx={"a": [0, 2], "b": [1, 3]},
        entity2domain={0: "a", 2: "a", 1: "b", 3: "b"},
    )
    queries = [(0, 0, 2)]
    out = corrupt_with_topup(sampler, queries, num_negatives=10, mode="tail")
    # Domain 'a' has only entity 0 left as a non-positive tail (ent 2 is filtered);
    # topup fills the 10 slots with duplicates of whatever remains.
    assert len(out[0]) == 10


def test_corrupt_with_topup_empty_queries_returns_empty():
    sampler = _make_sampler()
    assert corrupt_with_topup(sampler, [], num_negatives=3, mode="tail") == []
