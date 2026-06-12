"""Smoke tests for kge_kernels.inference: build a tiny KGE checkpoint
on disk and reload it through KGEInference.
"""
from __future__ import annotations

import json
import os

import pytest
import torch

from kge_kernels.inference import (
    KGEInference,
    _Atom,
    find_latest_run,
    normalize_backend,
)


def _write_fake_checkpoint(model_dir: str, num_entities: int = 5, num_relations: int = 3, dim: int = 8) -> None:
    """Write a config.json + entity/relation maps + weights.pth that
    KGEInference can load."""
    os.makedirs(model_dir, exist_ok=True)
    config = {
        "model": "TransE",
        "num_entities": num_entities,
        "num_relations": num_relations,
        "dim": dim,
        "gamma": 12.0,
        "p": 1,
    }
    with open(os.path.join(model_dir, "config.json"), "w") as h:
        json.dump(config, h)
    entity2id = {f"e{i}": i for i in range(num_entities)}
    relation2id = {f"r{i}": i for i in range(num_relations)}
    with open(os.path.join(model_dir, "entity2id.json"), "w") as h:
        json.dump(entity2id, h)
    with open(os.path.join(model_dir, "relation2id.json"), "w") as h:
        json.dump(relation2id, h)

    from kge_kernels.models import build_model
    model = build_model("TransE", num_entities, num_relations, dim=dim, gamma=12.0, p_norm=1)
    torch.save(model.state_dict(), os.path.join(model_dir, "weights.pth"))


# ═══════════════════════════════════════════════════════════════════════
# Atom parser
# ═══════════════════════════════════════════════════════════════════════


def test_atom_parses_binary_predicate():
    atom = _Atom("father(alice,bob)")
    assert atom.r == "father"
    assert atom.args == ["alice", "bob"]
    assert atom.to_tuple() == ("father", "alice", "bob")


def test_atom_strips_trailing_dot():
    atom = _Atom("father(alice,bob).")
    assert atom.to_tuple() == ("father", "alice", "bob")


# ═══════════════════════════════════════════════════════════════════════
# normalize_backend
# ═══════════════════════════════════════════════════════════════════════


def test_normalize_backend_aliases():
    assert normalize_backend("torch") == "pytorch"
    assert normalize_backend("pytorch") == "pytorch"
    assert normalize_backend("pykeen") == "pykeen"
    assert normalize_backend(None) == "pytorch"


def test_normalize_backend_rejects_unknown():
    with pytest.raises(ValueError):
        normalize_backend("tensorflow")


# ═══════════════════════════════════════════════════════════════════════
# find_latest_run
# ═══════════════════════════════════════════════════════════════════════


def test_find_latest_run_returns_none_when_dir_missing(tmp_path):
    assert find_latest_run(str(tmp_path / "missing")) is None


def test_find_latest_run_picks_newest(tmp_path):
    (tmp_path / "old").mkdir()
    (tmp_path / "new").mkdir()
    # bump mtime on new
    os.utime(tmp_path / "new", (1_700_000_000, 1_700_000_000))
    os.utime(tmp_path / "old", (1_600_000_000, 1_600_000_000))
    assert find_latest_run(str(tmp_path)) == "new"


def test_find_latest_run_filters_by_prefix(tmp_path):
    (tmp_path / "torch_run_a").mkdir()
    (tmp_path / "pykeen_run_b").mkdir()
    assert find_latest_run(str(tmp_path), prefix="torch_") == "torch_run_a"


# ═══════════════════════════════════════════════════════════════════════
# KGEInference round-trip
# ═══════════════════════════════════════════════════════════════════════


def test_kge_inference_loads_fake_checkpoint(tmp_path):
    run_signature = "torch_test_TransE_8"
    model_dir = tmp_path / run_signature
    _write_fake_checkpoint(str(model_dir), num_entities=5, num_relations=3, dim=8)

    engine = KGEInference(
        dataset_name="test",
        base_path=str(tmp_path),
        checkpoint_dir=str(tmp_path),
        run_signature=run_signature,
        device="cpu",
    )
    # Trigger lazy model build
    engine.model = engine._build_and_load_model()

    assert engine.entity2id == {f"e{i}": i for i in range(5)}
    assert engine.relation2id == {f"r{i}": i for i in range(3)}
    assert engine.model is not None


def test_kge_inference_predict_batch(tmp_path):
    run_signature = "torch_test_TransE_8"
    model_dir = tmp_path / run_signature
    _write_fake_checkpoint(str(model_dir), num_entities=5, num_relations=3, dim=8)

    engine = KGEInference(
        dataset_name="test",
        base_path=str(tmp_path),
        checkpoint_dir=str(tmp_path),
        run_signature=run_signature,
        device="cpu",
    )
    scores = engine.predict_batch(["r0(e0,e1)", "r1(e2,e3)"])
    assert len(scores) == 2
    assert all(0.0 <= s <= 1.0 for s in scores)
