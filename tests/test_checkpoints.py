"""Tests for kge_kernels.checkpoints."""
from __future__ import annotations

import json
import os

import torch
from torch import nn

from kge_kernels.checkpoints import (
    load_checkpoint,
    model_state_dict,
    normalize_loaded_state_dict,
    save_checkpoint,
    save_state_dict,
    unwrap_model,
    write_json_payload,
)

# ═══════════════════════════════════════════════════════════════════════
# normalize_loaded_state_dict
# ═══════════════════════════════════════════════════════════════════════


def test_normalize_strips_orig_mod_prefix():
    raw = {
        "_orig_mod.weight": torch.ones(3),
        "_orig_mod.bias": torch.zeros(3),
    }
    out = normalize_loaded_state_dict(raw)
    assert set(out.keys()) == {"weight", "bias"}
    assert torch.equal(out["weight"], torch.ones(3))


def test_normalize_noop_without_prefix():
    raw = {"weight": torch.ones(3)}
    out = normalize_loaded_state_dict(raw)
    assert out is raw or out == raw


def test_normalize_mixed_prefixes():
    raw = {
        "_orig_mod.weight": torch.ones(2),
        "plain_key": torch.zeros(2),
    }
    out = normalize_loaded_state_dict(raw)
    assert "weight" in out
    assert "plain_key" in out


# ═══════════════════════════════════════════════════════════════════════
# unwrap_model
# ═══════════════════════════════════════════════════════════════════════


def test_unwrap_plain_module_is_identity():
    m = nn.Linear(3, 3)
    assert unwrap_model(m) is m


def test_unwrap_dataparallel():
    inner = nn.Linear(3, 3)
    dp = nn.DataParallel(inner)
    assert unwrap_model(dp) is inner


def test_unwrap_handles_orig_mod_attribute():
    inner = nn.Linear(3, 3)

    class Wrapper(nn.Module):
        def __init__(self, mod):
            super().__init__()
            self._orig_mod = mod

    w = Wrapper(inner)
    assert unwrap_model(w) is inner


# ═══════════════════════════════════════════════════════════════════════
# save / load round-trip
# ═══════════════════════════════════════════════════════════════════════


def test_save_state_dict_roundtrip(tmp_path):
    path = str(tmp_path / "weights.pth")
    sd = {"w": torch.tensor([1.0, 2.0, 3.0])}
    out = save_state_dict(path, sd)
    assert os.path.isfile(out)
    loaded = torch.load(out, weights_only=False)
    assert torch.equal(loaded["w"], sd["w"])


def test_write_json_payload(tmp_path):
    path = str(tmp_path / "cfg.json")
    payload = {"lr": 1e-3, "model": "transe", "dim": 64}
    write_json_payload(path, payload)
    with open(path) as handle:
        loaded = json.load(handle)
    assert loaded == payload


def test_save_checkpoint_and_load_roundtrip(tmp_path):
    save_dir = str(tmp_path / "ckpt")
    sd = {"layer.weight": torch.ones(2, 2)}
    payload = {"model": "transe", "dim": 8, "epochs_completed": 10}
    wp, pp = save_checkpoint(save_dir, state_dict=sd, payload=payload)
    assert os.path.isfile(wp)
    assert os.path.isfile(pp)

    loaded_sd, loaded_payload = load_checkpoint(save_dir)
    assert torch.equal(loaded_sd["layer.weight"], sd["layer.weight"])
    assert loaded_payload == payload


def test_load_checkpoint_picks_best_over_final(tmp_path):
    save_dir = str(tmp_path / "ckpt")
    os.makedirs(save_dir)
    save_state_dict(os.path.join(save_dir, "weights.pth"), {"w": torch.zeros(1)})
    write_json_payload(os.path.join(save_dir, "config.json"), {"marker": "final"})
    write_json_payload(os.path.join(save_dir, "best_config.json"), {"marker": "best"})

    _, payload = load_checkpoint(save_dir)
    assert payload == {"marker": "best"}


def test_model_state_dict_unwraps():
    inner = nn.Linear(3, 3)
    dp = nn.DataParallel(inner)
    sd = model_state_dict(dp)
    # Keys should NOT have "module." prefix
    assert all(not k.startswith("module.") for k in sd.keys())
