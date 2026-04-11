"""Generic KGE checkpoint helpers.

Save-to-disk / load-from-disk utilities for KGE training. These are
intentionally decoupled from any specific config dataclass: a
checkpoint is a plain ``state_dict`` plus a plain JSON-serializable
``payload`` dict. Consumers attach their own config serialization on
top (see ``kge_experiments.kge_module.core.checkpoints`` in DpRL for
an example that adds ``TrainConfig`` persistence).

Shared utilities:

  - :func:`normalize_loaded_state_dict` strips the ``_orig_mod.`` prefix
    that ``torch.compile`` prepends, so checkpoints saved from a
    compiled model load cleanly into an un-compiled one.
  - :func:`unwrap_model` returns the real ``nn.Module`` inside a
    ``DataParallel`` + ``torch.compile`` chain.
  - :func:`save_state_dict`, :func:`save_checkpoint`,
    :func:`load_checkpoint` are thin wrappers around ``torch.save`` /
    ``torch.load`` with sensible defaults (``weights_only=False`` on
    load for payload dicts that include Python objects).
  - :func:`write_json_payload` writes a JSON-serializable dict to disk
    with pretty-printing.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple

import torch
from torch import Tensor, nn


def normalize_loaded_state_dict(
    state_dict: Dict[str, Tensor],
) -> Dict[str, Tensor]:
    """Strip the ``_orig_mod.`` prefix that ``torch.compile`` adds.

    A model saved after ``torch.compile`` has every parameter key
    prefixed with ``_orig_mod.``. Loading those keys into an
    un-compiled instance fails; this helper makes the state_dict
    portable across compiled / un-compiled loads.
    """
    if any(key.startswith("_orig_mod.") for key in state_dict):
        prefix = "_orig_mod."
        return {
            (key[len(prefix):] if key.startswith(prefix) else key): value
            for key, value in state_dict.items()
        }
    return state_dict


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the inner module from ``DataParallel`` and/or ``torch.compile``.

    Strips a single ``DataParallel`` layer, then a single ``_orig_mod``
    attribute (set by ``torch.compile``) if present. The original
    training wrapper is left intact — only the returned reference is
    unwrapped.
    """
    inner: nn.Module = model.module if isinstance(model, nn.DataParallel) else model
    if hasattr(inner, "_orig_mod"):
        inner = inner._orig_mod  # type: ignore[assignment]
    return inner


def model_state_dict(model: nn.Module) -> Dict[str, Tensor]:
    """Return ``unwrap_model(model).state_dict()``.

    Shortcut used by training loops that save checkpoints from a
    compiled + DataParallel-wrapped model: the saved tensors are free
    of both wrappers and load cleanly anywhere.
    """
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
    """Save a ``(state_dict, payload)`` pair to ``save_dir``.

    The weights go to ``<save_dir>/<weights_name>`` and the payload to
    ``<save_dir>/<payload_name>``. Both paths are returned. The caller
    is responsible for deciding what to put in the payload (config,
    metrics, vocabulary sizes, etc.).
    """
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
    """Load a ``(state_dict, payload)`` pair saved via :func:`save_checkpoint`.

    Returns the (normalized) state_dict and the payload. The state_dict
    is run through :func:`normalize_loaded_state_dict` so ``_orig_mod.``
    prefixes from compile-time saves are stripped.

    The payload is read from the first existing filename in
    ``payload_names`` so callers can support both a "best" and a
    "final" layout without duplicate calls.
    """
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


__all__ = [
    "load_checkpoint",
    "model_state_dict",
    "normalize_loaded_state_dict",
    "save_checkpoint",
    "save_state_dict",
    "unwrap_model",
    "write_json_payload",
]
