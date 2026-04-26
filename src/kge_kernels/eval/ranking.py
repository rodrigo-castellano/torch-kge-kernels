"""Ranking and metrics kernels for KGE evaluation."""

from __future__ import annotations

from typing import Dict, Literal, Optional, Sequence, Tuple

import torch
from torch import Tensor


def compute_ranks(
    scores: Tensor,
    true_idx: Tensor,
    valid_mask: Optional[torch.BoolTensor] = None,
    tie_handling: Literal["average", "random"] = "average",
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Compute 1-based rank of the true item among candidates.

    This is the single rank-computation function used by all evaluation
    paths: exhaustive filtered ranking, sampled corruption ranking, and
    streaming mini-batch ranking.

    Args:
        scores: ``[B, N]`` score matrix — all candidate scores per query.
        true_idx: ``[B]`` index of the true item in each row.
        valid_mask: ``[B, N]`` boolean mask — ``False`` entries are treated
            as ``-inf`` (ignored). ``None`` means all entries are valid.
        tie_handling: ``"average"`` gives ``1 + #better + 0.5 * #tied``
            (excluding the true item from the tie count);
            ``"random"`` breaks ties with a seeded coin flip.
        generator: RNG for ``"random"`` tie handling.

    Returns:
        ``[B]`` float tensor of 1-based ranks.
    """
    B, N = scores.shape
    device = scores.device

    if valid_mask is not None:
        scores = scores.masked_fill(~valid_mask, float("-inf"))

    batch_idx = torch.arange(B, device=device)
    target = scores[batch_idx, true_idx]  # [B]

    greater = (scores > target.unsqueeze(1)).sum(dim=1, dtype=torch.float32)

    if tie_handling == "average":
        equal = (scores == target.unsqueeze(1)).sum(dim=1, dtype=torch.float32)
        # Subtract 1 for the true item itself (it ties with itself)
        return greater + 1.0 + 0.5 * (equal - 1.0).clamp(min=0)
    else:
        tied = scores == target.unsqueeze(1)
        tied[batch_idx, true_idx] = False  # exclude true item from tie-breaking
        rnd = torch.rand(scores.shape, generator=generator, device=device)
        rnd_target = rnd[batch_idx, true_idx]
        coin = rnd > rnd_target.unsqueeze(1)
        tied_won = (tied & coin).sum(dim=1, dtype=torch.float32)
        return greater + 1.0 + tied_won


def metrics_from_ranks(
    ranks: Tensor,
    ks: Tuple[int, ...] = (1, 3, 10),
) -> Dict[str, float]:
    """Summarize a rank tensor into MRR and Hits@k.

    Args:
        ranks: Float tensor of 1-based ranks (shape ``[N]``).
        ks: Cutoffs for Hits@k. Default ``(1, 3, 10)`` produces
            ``Hits@1``, ``Hits@3``, ``Hits@10``.

    Returns:
        Dict with ``MRR`` and one ``Hits@{k}`` entry per ``k``.
    """
    if ranks.numel() == 0:
        out: Dict[str, float] = {"MRR": 0.0}
        for k in ks:
            out[f"Hits@{k}"] = 0.0
        return out
    inv = 1.0 / ranks.double()
    results: Dict[str, float] = {"MRR": inv.mean().item()}
    for k in ks:
        results[f"Hits@{k}"] = (ranks <= k).double().mean().item()
    return results




# ---------------------------------------------------------------------------
# Rank fusion (moved from fusion.py)
# ---------------------------------------------------------------------------


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
    query group. Tie-breaking uses seeded float64 noise so the perturbation
    stays observable (float32 precision ≈1.19e-7 would swallow a 1e-10 tie
    break).
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

        noise = torch.rand(CQ, K, generator=gen, device=device, dtype=torch.float64) * 1e-10
        sorted_idx = (scores_2d.to(torch.float64) + noise).argsort(dim=1, descending=True)
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


__all__ = [
    "compute_ranks",
    "metrics_from_ranks",
    "rrf",
    "zscore_fusion",
]
