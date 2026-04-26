"""KGE scoring entry points.

Two layers:

1. **Model adapter** (``build_backend``, ``kge_score_*``) — accept raw
   ``nn.Module`` objects, unwrap ``DataParallel`` / ``torch.compile``,
   dispatch to the model's scoring method, and apply sigmoid
   normalization. The hot public API used by ns / DpRL.
2. **Composite helpers** (``classify_atoms``, ``kge_score_triples_remapped``)
   — pure-tensor atom-type classification (used by partial-atom scoring
   and proof-state evaluators) and remap-aware triple scoring with a
   neutral fallback for unmapped atoms.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .types import KGEBackend


# ============================================================================
# Model adapter — unwrap DataParallel / torch.compile, sigmoid normalize
# ============================================================================


def _unwrap_model(model: nn.Module) -> nn.Module:
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
    raise AttributeError(
        "Model adapter requires num_constants or num_entities for fallback batched scoring"
    )


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


# ============================================================================
# Atom-type classification (proof-state-shape tensors)
# ============================================================================


def classify_atoms(
    preds: Tensor,
    args1: Tensor,
    args2: Tensor,
    constant_no: int,
    padding_idx: int,
    true_pred_idx: int,
    false_pred_idx: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Compute atom-type masks for proof-state scoring. Pure tensor ops.

    Atoms are classified by their predicate id (padding sentinel / True /
    False / regular) and the bind-status of their arguments (constant id in
    ``[1..constant_no]`` versus runtime-variable id ``> constant_no``).

    Returns:
        ``(is_padding, is_true, is_false, is_ground, is_partial)`` —
        broadcast-compatible boolean masks. ``is_ground`` covers atoms with
        both arguments bound to constants; ``is_partial`` covers atoms with
        exactly one argument bound (one variable, one constant).
    """
    is_padding = preds == padding_idx
    is_true = preds == true_pred_idx
    is_false = preds == false_pred_idx
    is_terminal = is_true | is_false
    a1_const = (args1 > 0) & (args1 <= constant_no)
    a2_const = (args2 > 0) & (args2 <= constant_no)
    is_ground = ~is_padding & ~is_terminal & a1_const & a2_const
    is_partial = ~is_padding & ~is_terminal & ~is_ground & (a1_const | a2_const)
    return is_padding, is_true, is_false, is_ground, is_partial


# ============================================================================
# Remap-aware scoring (KGE id space != caller id space)
# ============================================================================


@torch.no_grad()
def kge_score_triples_remapped(
    model: nn.Module,
    queries: Tensor,
    pred_remap: Tensor,
    const_remap: Tensor,
    *,
    fallback_score: float = 0.5,
    log: bool = False,
    log_eps: float = 1e-8,
) -> Tensor:
    """Score remapped ``[B, 3]`` triples; unmapped rows fall back to a neutral score.

    The caller's id space is mapped into the KGE model's id space via
    ``pred_remap`` and ``const_remap``: each table holds the KGE id at the
    caller's index, or ``-1`` for unmapped slots. Rows with any unmapped
    component (or padding ``preds == 0``) get ``fallback_score`` (neutral
    0.5 by default — produces ``-log(0.5)`` if ``log=True``, avoiding
    ``-inf`` / ``NaN`` in downstream accumulators).

    Branchless: scores every row through the model (with clamps to keep
    indices in range) and overwrites unmapped rows post-hoc, so the
    function is safe inside compiled / CUDA-graph paths.

    Args:
        model: KGE model (sigmoid-normalized scores).
        queries: ``[B, 3]`` triples in caller's id space, ``(pred, head, tail)``.
        pred_remap: ``[P_caller]`` int tensor; ``pred_remap[caller_id] =
            kge_id`` (``-1`` = unmapped).
        const_remap: ``[E_caller]`` int tensor; same shape contract.
        fallback_score: probability assigned to unmapped rows. Default 0.5.
        log: If True, return ``log(scores.clamp(min=log_eps))``.
        log_eps: Lower clamp before ``log`` to avoid ``log(0)``.

    Returns:
        ``[B]`` scores (probabilities by default; log-probs when ``log=True``).
    """
    preds = queries[:, 0]
    heads = queries[:, 1]
    tails = queries[:, 2]
    r = pred_remap[preds.clamp(min=0)]
    h = const_remap[heads.clamp(min=0)]
    t = const_remap[tails.clamp(min=0)]
    valid = (r >= 0) & (h >= 0) & (t >= 0) & (preds > 0)
    safe_h = h.clamp(min=0)
    safe_r = r.clamp(min=0)
    safe_t = t.clamp(min=0)
    raw = kge_score_triples(model, safe_h, safe_r, safe_t)
    scores = torch.where(valid, raw, torch.full_like(raw, fallback_score))
    if log:
        return torch.log(scores.clamp(min=log_eps))
    return scores


__all__ = [
    "build_backend",
    "classify_atoms",
    "kge_score_all_heads",
    "kge_score_all_heads_dchunked",
    "kge_score_all_tails",
    "kge_score_all_tails_dchunked",
    "kge_score_triples",
    "kge_score_triples_remapped",
]
