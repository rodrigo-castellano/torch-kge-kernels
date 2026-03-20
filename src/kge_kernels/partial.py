"""Partial-atom scoring utilities built on top of the shared scoring kernels."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

from .scoring import _score_all_heads, _score_all_tails
from .types import KGEBackend


@torch.no_grad()
def precompute_partial_scores(
    backend: KGEBackend,
    pred_remap: Tensor,
    const_remap: Tensor,
    batch_chunk: int = 64,
) -> Tuple[Tensor, Tensor]:
    """Precompute partial-atom lookup tables for grounding-style use cases.

    For each mapped predicate/entity pair, this computes:

    - the best score over all possible tails for ``pred(entity, ?)``
    - the best score over all possible heads for ``pred(?, entity)``

    Args:
        backend: Explicit scoring backend.
        pred_remap: Predicate remap tensor of shape ``[P_im]`` with ``-1`` for
            unmapped predicates.
        const_remap: Entity remap tensor of shape ``[E_im]`` with ``-1`` for
            unmapped entities.
        batch_chunk: Number of entities scored per chunk in the batched
            exhaustive backend calls.

    Returns:
        ``(max_tail_score, max_head_score)``, both of shape ``[P_im, E_im]``.
    """

    device = const_remap.device
    p_im = pred_remap.shape[0]
    e_im = const_remap.shape[0]

    max_tail_score = torch.zeros(p_im, e_im, dtype=torch.float32, device=device)
    max_head_score = torch.zeros(p_im, e_im, dtype=torch.float32, device=device)

    valid_preds = (pred_remap >= 0).nonzero(as_tuple=True)[0]
    valid_ents = (const_remap >= 0).nonzero(as_tuple=True)[0]
    if valid_preds.numel() == 0 or valid_ents.numel() == 0:
        return max_tail_score, max_head_score

    if batch_chunk <= 0:
        batch_chunk = 64

    kge_ents = const_remap[valid_ents]
    for im_pred in valid_preds:
        kge_rel = pred_remap[im_pred]
        tail_scores = _partial_score_chunked(backend, kge_ents, kge_rel, role=0, batch_chunk=batch_chunk)
        head_scores = _partial_score_chunked(backend, kge_ents, kge_rel, role=1, batch_chunk=batch_chunk)
        max_tail_score[im_pred, valid_ents] = tail_scores
        max_head_score[im_pred, valid_ents] = head_scores

    return max_tail_score, max_head_score


def _partial_score_chunked(
    backend: KGEBackend,
    kge_ents: Tensor,
    kge_rel: Tensor,
    role: int,
    batch_chunk: int,
) -> Tensor:
    """Compute best head/tail completions for a chunk of entities."""

    num_entities = kge_ents.shape[0]
    device = kge_ents.device
    result = torch.empty(num_entities, dtype=torch.float32, device=device)

    for start in range(0, num_entities, batch_chunk):
        end = min(start + batch_chunk, num_entities)
        chunk = kge_ents[start:end]
        rel_exp = kge_rel.expand(chunk.shape[0])
        if role == 0:
            raw = _score_all_tails(backend, chunk, rel_exp)
        else:
            raw = _score_all_heads(backend, rel_exp, chunk)
        result[start:end] = raw.max(dim=1).values

    return result


def score_partial_atoms(
    preds: Tensor,
    args1: Tensor,
    args2: Tensor,
    constant_no: int,
    max_tail_score: Tensor,
    max_head_score: Tensor,
) -> Tensor:
    """Score partially grounded binary atoms by table lookup.

    The function assumes the shared grounding convention:

    - ``pred(const, ?)`` uses ``max_tail_score``
    - ``pred(?, const)`` uses ``max_head_score``
    - fully unbound atoms receive score ``0``

    Args:
        preds: Predicate ids ``[N]``.
        args1: First argument ids ``[N]``.
        args2: Second argument ids ``[N]``.
        constant_no: Highest constant id in the caller's index space.
        max_tail_score: Precomputed tail lookup table ``[P, E]``.
        max_head_score: Precomputed head lookup table ``[P, E]``.

    Returns:
        Scores of shape ``[N]``.
    """

    n = preds.shape[0]
    device = preds.device
    scores = torch.zeros(n, dtype=torch.float32, device=device)
    if n == 0:
        return scores

    a1 = args1.long()
    a2 = args2.long()
    p = preds.long()
    p_max, e_max = max_tail_score.shape

    safe_p = p.clamp(0, p_max - 1)
    safe_a1 = a1.clamp(0, e_max - 1)
    safe_a2 = a2.clamp(0, e_max - 1)

    tail_var = (a1 > 0) & (a1 <= constant_no) & (a2 > constant_no)
    scores = torch.where(tail_var, max_tail_score[safe_p, safe_a1], scores)

    head_var = (a1 > constant_no) & (a2 > 0) & (a2 <= constant_no)
    scores = torch.where(head_var, max_head_score[safe_p, safe_a2], scores)

    return scores


__all__ = ["precompute_partial_scores", "score_partial_atoms"]
