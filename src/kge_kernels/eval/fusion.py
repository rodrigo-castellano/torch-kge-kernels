"""Rank fusion methods: RRF and z-score fusion.

All functions are CUDA-graph safe (static shapes, no .item(), seeded RNG).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
from torch import Tensor


def rrf(
    scores_dict: Dict[str, Tensor],
    pool_k: int,
    n_queries: int,
    device: torch.device,
    k: float = 60.0,
    seed: int = 42,
    modes: Optional[Sequence[str]] = None,
    mode_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Tensor]:
    """Reciprocal Rank Fusion across scoring modes.

    For each candidate, computes ``RRF = sum(w_i / (k + rank_i))`` where
    ``rank_i`` is the candidate's 1-based rank under mode *i* within its
    query group.

    Tie-breaking uses seeded noise for reproducibility and fairness.

    Args:
        scores_dict: ``{mode: [pool_size]}`` flat scores per mode.
        pool_k: Candidates per query (``1 + n_corruptions``).
        n_queries: Number of queries.
        device: Computation device.
        k: RRF constant (default 60, standard in IR).
        seed: Seed for tie-breaking noise.
        modes: Subset of modes to fuse. ``None`` means all.
        mode_weights: Optional per-mode weights (default 1.0 each).

    Returns:
        ``{"rrf": [pool_size]}`` flat fused scores.
    """
    K = pool_k
    CQ = n_queries
    gen = torch.Generator(device=device).manual_seed(seed)

    rrf_scores = torch.zeros(CQ, K, dtype=torch.float32, device=device)
    arange_k = torch.arange(K, device=device).unsqueeze(0).expand(CQ, K)

    items = scores_dict.items() if modes is None else ((m, scores_dict[m]) for m in modes if m in scores_dict)

    for mode_name, scores_flat in items:
        w = 1.0 if mode_weights is None else mode_weights.get(mode_name, 1.0)
        # Reshape flat [CQ*K] → [CQ, K] (column-major pool layout)
        scores_2d = scores_flat.view(K, CQ).t()

        # Rank with seeded noise for fair tie-breaking (1e-10 matches v1 behavior)
        noise = torch.rand(CQ, K, generator=gen, device=device, dtype=scores_2d.dtype) * 1e-10
        sorted_idx = (scores_2d + noise).argsort(dim=1, descending=True)
        ranks = torch.zeros_like(sorted_idx, dtype=torch.long)
        ranks.scatter_(1, sorted_idx, arange_k + 1)

        rrf_scores += w / (k + ranks.float())

    # Flatten back to [CQ*K] in column-major order
    return {"rrf": rrf_scores.t().reshape(-1)}


def zscore_fusion(
    scores_dict: Dict[str, Tensor],
    pool_k: int,
    n_queries: int,
    device: torch.device,
    modes: Optional[Sequence[str]] = None,
) -> Dict[str, Tensor]:
    """Z-score normalization + mean across scoring modes.

    Each mode's scores are normalized to zero mean and unit variance within
    each query group, then averaged.

    Args:
        scores_dict: ``{mode: [pool_size]}`` flat scores per mode.
        pool_k: Candidates per query.
        n_queries: Number of queries.
        device: Computation device.
        modes: Subset of modes to fuse. ``None`` means all.

    Returns:
        ``{"zscore": [pool_size]}`` flat fused scores.
    """
    K = pool_k
    CQ = n_queries
    fused = torch.zeros(CQ, K, dtype=torch.float32, device=device)
    n_modes = 0

    items = scores_dict.items() if modes is None else ((m, scores_dict[m]) for m in modes if m in scores_dict)

    for _, scores_flat in items:
        scores_2d = scores_flat.view(K, CQ).t()
        mu = scores_2d.mean(dim=1, keepdim=True)
        std = scores_2d.std(dim=1, keepdim=True).clamp(min=1e-8)
        fused += (scores_2d - mu) / std
        n_modes += 1

    if n_modes > 0:
        fused /= n_modes

    return {"zscore": fused.t().reshape(-1)}


__all__ = ["rrf", "zscore_fusion"]
