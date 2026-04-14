"""KGE checkpoint helpers.

Generic save/load utilities plus higher-level checkpoint management
for the full ``train_model`` pipeline (best, latest, final checkpoints
with config payloads).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from .config import TrainConfig

# ═══════════════════════════════════════════════════════════════════════
# Low-level helpers
# ═══════════════════════════════════════════════════════════════════════


def normalize_loaded_state_dict(
    state_dict: Dict[str, Tensor],
) -> Dict[str, Tensor]:
    """Strip the ``_orig_mod.`` prefix that ``torch.compile`` adds."""
    if any(key.startswith("_orig_mod.") for key in state_dict):
        prefix = "_orig_mod."
        return {
            (key[len(prefix):] if key.startswith(prefix) else key): value
            for key, value in state_dict.items()
        }
    return state_dict


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the inner module from ``DataParallel`` and/or ``torch.compile``."""
    inner: nn.Module = model.module if isinstance(model, nn.DataParallel) else model
    if hasattr(inner, "_orig_mod"):
        inner = inner._orig_mod  # type: ignore[assignment]
    return inner


def model_state_dict(model: nn.Module) -> Dict[str, Tensor]:
    """Return ``unwrap_model(model).state_dict()``."""
    return unwrap_model(model).state_dict()


def save_state_dict(path: str, state_dict: Dict[str, Tensor]) -> str:
    """Save a state_dict with ``torch.save``. Returns the written path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    torch.save(state_dict, path)
    return path


def write_json_payload(path: str, payload: Dict[str, Any]) -> str:
    """Write a JSON payload with ``indent=2``. Returns the written path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def save_checkpoint(
    save_dir: str,
    *,
    state_dict: Dict[str, Tensor],
    payload: Dict[str, Any],
    weights_name: str = "weights.pth",
    payload_name: str = "config.json",
) -> Tuple[str, str]:
    """Save a ``(state_dict, payload)`` pair to ``save_dir``."""
    os.makedirs(save_dir, exist_ok=True)
    weights_path = os.path.join(save_dir, weights_name)
    payload_path = os.path.join(save_dir, payload_name)
    save_state_dict(weights_path, state_dict)
    write_json_payload(payload_path, payload)
    return weights_path, payload_path


def load_checkpoint(
    save_dir: str,
    *,
    weights_name: str = "weights.pth",
    payload_names: Tuple[str, ...] = ("best_config.json", "config.json"),
    map_location: Any = "cpu",
) -> Tuple[Dict[str, Tensor], Dict[str, Any]]:
    """Load a ``(state_dict, payload)`` pair saved via :func:`save_checkpoint`."""
    weights_path = os.path.join(save_dir, weights_name)
    raw = torch.load(weights_path, map_location=map_location, weights_only=False)
    state_dict = normalize_loaded_state_dict(raw)

    payload: Dict[str, Any] = {}
    for name in payload_names:
        path = os.path.join(save_dir, name)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            break
    return state_dict, payload


# ═══════════════════════════════════════════════════════════════════════
# Pipeline-level checkpoint management
# ═══════════════════════════════════════════════════════════════════════


def build_config_payload(
    cfg: TrainConfig,
    *,
    num_entities: int,
    num_relations: int,
    metrics_payload: Optional[Dict[str, float]] = None,
    validation_history_payload: Optional[List[Dict[str, float]]] = None,
    best_valid_mrr_payload: Optional[float] = None,
    best_valid_epoch_payload: int = 0,
    stopped_early_payload: bool = False,
    epochs_completed_payload: int = 0,
) -> Dict[str, object]:
    """Serialize ``TrainConfig`` + training metadata into a JSON payload."""
    payload = asdict(cfg)
    payload.update({
        "num_entities": num_entities,
        "num_relations": num_relations,
    })
    mrr = None
    if metrics_payload:
        mrr = metrics_payload.get("test_mrr") or metrics_payload.get("valid_mrr")
    if mrr:
        payload["mrr"] = mrr
    payload["metrics"] = metrics_payload or {}
    payload["validation_history"] = validation_history_payload or []
    payload["best_valid_mrr"] = best_valid_mrr_payload
    payload["best_valid_epoch"] = best_valid_epoch_payload
    payload["stopped_early"] = stopped_early_payload
    payload["epochs_completed"] = epochs_completed_payload
    if cfg.run_signature:
        payload["run_signature"] = cfg.run_signature
    model_name = cfg.model.lower()
    if model_name == "rotate":
        payload.update({"gamma": cfg.gamma, "p": cfg.p})
    elif model_name == "tucker":
        payload.update({
            "entity_dim": cfg.dim,
            "relation_dim": cfg.relation_dim or cfg.dim,
            "dropout": cfg.dropout,
        })
    return payload


def load_checkpoint_payload(checkpoint_dir: str) -> Dict[str, object]:
    """Load a config payload dict from ``best_config.json`` or ``config.json``."""
    for filename in ("best_config.json", "config.json"):
        path = os.path.join(checkpoint_dir, filename)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
    raise FileNotFoundError(
        f"No best_config.json or config.json found in {checkpoint_dir}"
    )


def config_from_payload(payload: Dict[str, object], save_dir: str) -> TrainConfig:
    """Reconstruct a ``TrainConfig`` from a payload dict."""
    kwargs = {"save_dir": save_dir}
    for field_name in TrainConfig.__dataclass_fields__:
        if field_name != "save_dir" and field_name in payload:
            kwargs[field_name] = payload[field_name]
    return TrainConfig(**kwargs)


def save_latest_weights(save_dir: str, state_dict: Dict[str, object]) -> str:
    """Save to ``<save_dir>/latest_weights.pth``."""
    return save_state_dict(os.path.join(save_dir, "latest_weights.pth"), state_dict)


def save_best_checkpoint(
    save_dir: str,
    *,
    state_dict: Dict[str, object],
    config_payload: Dict[str, object],
) -> tuple[str, str]:
    """Save ``best_weights.pth`` + ``best_config.json``."""
    os.makedirs(save_dir, exist_ok=True)
    weights_path = os.path.join(save_dir, "best_weights.pth")
    config_path = os.path.join(save_dir, "best_config.json")
    save_state_dict(weights_path, state_dict)
    write_json_payload(config_path, config_payload)
    return weights_path, config_path


def save_final_checkpoint(
    save_dir: str,
    *,
    state_dict: Dict[str, object],
    config_payload: Dict[str, object],
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
) -> tuple[str, str]:
    """Save ``weights.pth`` + ``config.json`` + ``entity2id.json`` + ``relation2id.json``."""
    os.makedirs(save_dir, exist_ok=True)
    weights_path = os.path.join(save_dir, "weights.pth")
    config_path = os.path.join(save_dir, "config.json")
    save_state_dict(weights_path, state_dict)
    save_latest_weights(save_dir, state_dict)
    write_json_payload(config_path, config_payload)
    write_json_payload(os.path.join(save_dir, "entity2id.json"), entity2id)
    write_json_payload(os.path.join(save_dir, "relation2id.json"), relation2id)
    return weights_path, config_path


__all__ = [
    "build_config_payload",
    "config_from_payload",
    "load_checkpoint",
    "load_checkpoint_payload",
    "model_state_dict",
    "normalize_loaded_state_dict",
    "save_best_checkpoint",
    "save_checkpoint",
    "save_final_checkpoint",
    "save_latest_weights",
    "save_state_dict",
    "unwrap_model",
    "write_json_payload",
]
