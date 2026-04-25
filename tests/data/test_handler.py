"""Tests for KGEDatasetHandler — base loader + convenience methods."""
from __future__ import annotations

import torch

from kge_kernels.data import KGEDatasetHandler
from kge_kernels.scoring import Sampler


def _write_tiny_dataset(tmp_path):
    """Build a minimal dataset folder with train/valid/test/facts."""
    base = tmp_path / "tinykg"
    base.mkdir()
    (base / "train.txt").write_text("alice\tknows\tbob\nbob\tknows\tcarol\n")
    (base / "valid.txt").write_text("alice\tknows\tcarol\n")
    (base / "test.txt").write_text("carol\tknows\talice\n")
    (base / "facts.txt").write_text("dave\tknows\teve\n")
    return str(tmp_path), "tinykg"


def test_handler_loads_canonical_pipeline(tmp_path):
    base_path, name = _write_tiny_dataset(tmp_path)
    h = KGEDatasetHandler(name, base_path)
    # Alphabetical id assignment: alice=0, bob=1, carol=2, dave=3, eve=4
    assert h.entity2id == {"alice": 0, "bob": 1, "carol": 2, "dave": 3, "eve": 4}
    assert h.relation2id == {"knows": 0}
    assert h.num_entities == 5
    assert h.num_relations == 1
    assert len(h.train_idx) == 2
    assert len(h.valid_idx) == 1
    assert len(h.test_idx) == 1
    assert len(h.known_idx) == 1


def test_handler_split_idx_returns_indexed_triples(tmp_path):
    base_path, name = _write_tiny_dataset(tmp_path)
    h = KGEDatasetHandler(name, base_path)
    assert h.split_idx("train") == h.train_idx
    assert h.split_idx("valid") == h.valid_idx
    assert h.split_idx("test") == h.test_idx
    assert h.split_idx("known") == h.known_idx


def test_build_sampler_uses_loaded_id_space(tmp_path):
    base_path, name = _write_tiny_dataset(tmp_path)
    h = KGEDatasetHandler(name, base_path)
    sampler = h.build_sampler()
    assert isinstance(sampler, Sampler)
    assert sampler.num_entities == h.num_entities
    assert sampler.num_relations == h.num_relations
    # Filter index covers ALL ground triples (train + valid + test + facts)
    pos = torch.tensor([[0, 0, 1]], dtype=torch.long)  # knows(alice, bob)
    neg, valid = sampler.corrupt_with_mask(pos, num_negatives=10, mode="tail")
    rows = [tuple(t.tolist()) for t, k in zip(neg[0], valid[0]) if k]
    # Filter should remove every known (knows, alice, *) triple
    for r, h_idx, t_idx in rows:
        assert (r, h_idx, t_idx) not in h.ground_facts_idx_set


def test_build_sampler_picks_up_domain_info(tmp_path):
    base_path, name = _write_tiny_dataset(tmp_path)
    base = tmp_path / name
    (base / "domain.txt").write_text("people alice bob carol\nothers dave eve\n")
    h = KGEDatasetHandler(name, base_path, domain_file="domain.txt")
    sampler = h.build_sampler()
    # Domain-restricted corruption: tail of (knows, alice, bob) stays in 'people'.
    pos = torch.tensor([[0, 0, 1]], dtype=torch.long)
    neg, valid = sampler.corrupt_with_mask(pos, num_negatives=10, mode="tail")
    rows = [tuple(t.tolist()) for t, k in zip(neg[0], valid[0]) if k]
    people_ids = {h.entity2id[n] for n in ("alice", "bob", "carol")}
    for _r, _h, t in rows:
        assert t in people_ids
