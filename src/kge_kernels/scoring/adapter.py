"""Model adapter: unwrapping, sigmoid normalization, and backend construction.

Consumers pass raw ``nn.Module`` model objects; this module handles
DataParallel/compile unwrapping, dispatches to the appropriate scoring method,
applies sigmoid, and constructs ``KGEBackend`` instances for the lower-level
scoring kernels (used by partial-atom scoring and grounder).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .partial import precompute_partial_scores as _precompute_partial_scores
from .partial import score_partial_atoms
from .types import KGEBackend


def _unwrap_model(model: "nn.Module") -> "nn.Module":
    """Unwrap DataParallel and torch.compile wrappers.

    Canonical definition lives in ``training.checkpoints.unwrap_model``.
    Inlined here to avoid eagerly importing ``training/`` at package load time.
    """
    actual = model.module if isinstance(model, nn.DataParallel) else model
    if hasattr(actual, "_orig_mod"):
        actual = actual._orig_mod
    return actual


def _num_entities(model: nn.Module) -> int:
    actual = _unwrap_model(model)
    if hasattr(actual, "num_constants"):
        return int(actual.num_constants)
    if hasattr(actual, "num_entities"):
        return int(actual.num_entities)
    raise AttributeError("Model adapter requires num_constants or num_entities for fallback batched scoring")


def _score_triples_sigmoid(model: nn.Module, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
    actual = _unwrap_model(model)
    if hasattr(actual, "score"):
        raw = actual.score(h, r, t)
    elif hasattr(actual, "score_triples"):
        raw = actual.score_triples(h, r, t)
    elif hasattr(actual, "score_atoms"):
        raw = actual.score_atoms(r, h, t)
    else:
        raise AttributeError("Model adapter requires score(), score_triples(), or score_atoms()")
    return torch.sigmoid(raw)


def _score_all_tails_sigmoid(model: nn.Module, h: Tensor, r: Tensor) -> Tensor:
    actual = _unwrap_model(model)
    if hasattr(actual, "score"):
        return torch.sigmoid(actual.score(h, r, None))
    if hasattr(actual, "score_all_tails_batch"):
        return torch.sigmoid(actual.score_all_tails_batch(h, r))

    batch_size = h.shape[0]
    num_entities = _num_entities(actual)
    device = h.device
    all_tails = torch.arange(num_entities, device=device).unsqueeze(0).expand(batch_size, -1)
    h_exp = h.unsqueeze(1).expand_as(all_tails).reshape(-1)
    if r.dim() == 0:
        r_exp = r.expand(batch_size * num_entities)
    else:
        r_exp = r.unsqueeze(1).expand_as(all_tails).reshape(-1)
    t_exp = all_tails.reshape(-1)
    return _score_triples_sigmoid(actual, h_exp, r_exp, t_exp).view(batch_size, num_entities)


def _score_all_heads_sigmoid(model: nn.Module, r: Tensor, t: Tensor) -> Tensor:
    actual = _unwrap_model(model)
    if hasattr(actual, "score"):
        return torch.sigmoid(actual.score(None, r, t))
    if hasattr(actual, "score_all_heads_batch"):
        return torch.sigmoid(actual.score_all_heads_batch(r, t))

    batch_size = t.shape[0]
    num_entities = _num_entities(actual)
    device = t.device
    all_heads = torch.arange(num_entities, device=device).unsqueeze(0).expand(batch_size, -1)
    h_exp = all_heads.reshape(-1)
    if r.dim() == 0:
        r_exp = r.expand(batch_size * num_entities)
    else:
        r_exp = r.unsqueeze(1).expand_as(all_heads).reshape(-1)
    t_exp = t.unsqueeze(1).expand_as(all_heads).reshape(-1)
    return _score_triples_sigmoid(actual, h_exp, r_exp, t_exp).view(batch_size, num_entities)


def build_backend(model: nn.Module) -> KGEBackend:
    """Construct a ``KGEBackend`` from a model with sigmoid normalization."""
    return KGEBackend(
        score_triples=lambda h, r, t: _score_triples_sigmoid(model, h, r, t),
        score_all_tails=lambda h, r: _score_all_tails_sigmoid(model, h, r),
        score_all_heads=lambda r, t: _score_all_heads_sigmoid(model, r, t),
    )


def kge_score_triples(model: nn.Module, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
    """Score specific triples with sigmoid normalization."""
    return _score_triples_sigmoid(model, h, r, t)


def kge_score_all_tails(model: nn.Module, h: Tensor, r: Tensor) -> Tensor:
    """Score all entities as tails with sigmoid normalization."""
    return _score_all_tails_sigmoid(model, h, r)


def kge_score_all_heads(model: nn.Module, r: Tensor, t: Tensor) -> Tensor:
    """Score all entities as heads with sigmoid normalization."""
    return _score_all_heads_sigmoid(model, r, t)


def kge_score_all_tails_dchunked(
    model: nn.Module, h: Tensor, r: Tensor, d_chunk: int = 64
) -> Tensor:
    """Chunk-over-D exhaustive tail scoring with sigmoid normalization.

    Falls back to :func:`kge_score_all_tails` for models without a native
    ``score_all_tails_dchunked`` implementation.
    """
    actual = _unwrap_model(model)
    if hasattr(actual, "score_all_tails_dchunked"):
        return torch.sigmoid(actual.score_all_tails_dchunked(h, r, d_chunk=d_chunk))
    return _score_all_tails_sigmoid(actual, h, r)


def kge_score_all_heads_dchunked(
    model: nn.Module, r: Tensor, t: Tensor, d_chunk: int = 64
) -> Tensor:
    """Chunk-over-D exhaustive head scoring with sigmoid normalization.

    Falls back to :func:`kge_score_all_heads` for models without a native
    ``score_all_heads_dchunked`` implementation.
    """
    actual = _unwrap_model(model)
    if hasattr(actual, "score_all_heads_dchunked"):
        return torch.sigmoid(actual.score_all_heads_dchunked(r, t, d_chunk=d_chunk))
    return _score_all_heads_sigmoid(actual, r, t)


def precompute_partial_scores(
    kge_model: nn.Module,
    pred_remap: Tensor,
    const_remap: Tensor,
    batch_chunk: int = 64,
) -> Tuple[Tensor, Tensor]:
    """Precompute partial-score tables via the model adapter."""
    return _precompute_partial_scores(build_backend(kge_model), pred_remap, const_remap, batch_chunk=batch_chunk)


__all__ = [
    "build_backend",
    "precompute_partial_scores",
    "score_partial_atoms",
]
