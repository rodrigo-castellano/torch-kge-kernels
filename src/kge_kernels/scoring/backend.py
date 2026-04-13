"""Low-level KGE scoring kernels.

This module is intentionally minimal. It does not inspect model objects,
normalize outputs, or emulate missing batched methods. Consumers must provide
an explicit backend with tensor callables for triple scoring and exhaustive
head/tail scoring.
"""

from __future__ import annotations

from typing import Literal, Tuple

import torch
from torch import Tensor

from .sampler import corrupt as generate_corruptions
from .types import KGEBackend, ScoreOutput, SupportsCorruptWithMask


def _score_triples(backend: KGEBackend, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
    """Score a batch of fully ground triples in backend-native score space."""
    return backend.score_triples(h, r, t)


def _score_all_tails(backend: KGEBackend, h: Tensor, r: Tensor) -> Tensor:
    """Score all candidate tails for each ``(h, r)`` pair."""
    return backend.score_all_tails(h, r)


def _score_all_heads(backend: KGEBackend, r: Tensor, t: Tensor) -> Tensor:
    """Score all candidate heads for each ``(r, t)`` pair."""
    return backend.score_all_heads(r, t)


def _score_k_tails(
    backend: KGEBackend,
    h: Tensor,
    r: Tensor,
    t: Tensor,
    sampler: SupportsCorruptWithMask,
    num_corruptions: int,
) -> Tuple[Tensor, Tensor]:
    """Score a positive tail query against sampled tail corruptions."""
    queries = torch.stack([r, h, t], dim=1)
    corruption = generate_corruptions(
        sampler,
        queries,
        num_corruptions=num_corruptions,
        mode="tail",
    )
    neg = corruption.negatives
    valid = corruption.valid_mask
    k = neg.shape[1]
    pos_scores = _score_triples(backend, h, r, t)
    neg_h = neg[:, :, 1].reshape(-1)
    neg_r = neg[:, :, 0].reshape(-1)
    neg_t = neg[:, :, 2].reshape(-1)
    neg_scores = _score_triples(backend, neg_h, neg_r, neg_t).view(h.shape[0], k)
    neg_scores[~valid] = float("-inf")
    return torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1), valid


def _score_k_heads(
    backend: KGEBackend,
    h: Tensor,
    r: Tensor,
    t: Tensor,
    sampler: SupportsCorruptWithMask,
    num_corruptions: int,
) -> Tuple[Tensor, Tensor]:
    """Score a positive head query against sampled head corruptions."""
    queries = torch.stack([r, h, t], dim=1)
    corruption = generate_corruptions(
        sampler,
        queries,
        num_corruptions=num_corruptions,
        mode="head",
    )
    neg = corruption.negatives
    valid = corruption.valid_mask
    k = neg.shape[1]
    pos_scores = _score_triples(backend, h, r, t)
    neg_h = neg[:, :, 1].reshape(-1)
    neg_r = neg[:, :, 0].reshape(-1)
    neg_t = neg[:, :, 2].reshape(-1)
    neg_scores = _score_triples(backend, neg_h, neg_r, neg_t).view(h.shape[0], k)
    neg_scores[~valid] = float("-inf")
    return torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1), valid


def score(
    backend: KGEBackend,
    triples: Tensor,
    *,
    mode: Literal["triples", "head", "tail"] = "triples",
    num_corruptions: int | None = None,
    sampler: SupportsCorruptWithMask | None = None,
) -> ScoreOutput:
    """Public scoring entry point for triple, sampled, or exhaustive scoring.

    Args:
        backend: Explicit scoring backend.
        triples: Query triples in ``(r, h, t)`` format, shape ``[B, 3]``.
        mode: ``triples`` for direct triple scores, ``head`` for head ranking,
            or ``tail`` for tail ranking.
        num_corruptions: ``None`` means exhaustive head/tail scoring. A
            positive integer means sampled scoring against ``K`` corruptions.
        sampler: Required when ``num_corruptions`` is not ``None``.

    Returns:
        ``ScoreOutput``. Direct triple scoring returns ``[B]``. Exhaustive head
        or tail scoring returns ``[B, E]``. Sampled scoring returns ``[B, 1+K]``
        with the positive score in column 0 and a ``valid_mask`` for the
        sampled negatives.
    """

    r = triples[:, 0]
    h = triples[:, 1]
    t = triples[:, 2]

    if mode == "triples":
        return ScoreOutput(scores=_score_triples(backend, h, r, t))

    if num_corruptions is None:
        if mode == "tail":
            return ScoreOutput(scores=_score_all_tails(backend, h, r))
        if mode == "head":
            return ScoreOutput(scores=_score_all_heads(backend, r, t))
        raise ValueError(f"Unsupported score mode: {mode}")

    if sampler is None:
        raise ValueError("Sampled scoring requires a sampler")

    if mode == "tail":
        scores, valid_mask = _score_k_tails(backend, h, r, t, sampler, num_corruptions)
        return ScoreOutput(scores=scores, valid_mask=valid_mask)
    if mode == "head":
        scores, valid_mask = _score_k_heads(backend, h, r, t, sampler, num_corruptions)
        return ScoreOutput(scores=scores, valid_mask=valid_mask)
    raise ValueError(f"Unsupported score mode: {mode}")



__all__ = [
    "score",
]
