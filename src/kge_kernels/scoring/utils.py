"""Internal tensor utilities for torch-kge-kernels."""

from __future__ import annotations

import torch
from torch import Tensor

from .types import LongTensor


def _mix_hash(triples: LongTensor, b_e: int, b_r: int) -> LongTensor:
    """Pack ``(r, h, t)`` triples into sortable 64-bit hash values.

    The arithmetic bases are provided by the caller so the mapping stays unique
    for that caller's entity and relation ranges without relying on fixed bit
    widths.
    """

    h = triples[..., 1].to(dtype=triples.dtype)
    r = triples[..., 0].to(dtype=triples.dtype)
    t = triples[..., 2].to(dtype=triples.dtype)
    return ((h * b_r) + r) * b_e + t


def compute_bernoulli_probs(triples_rht: LongTensor, num_relations: int) -> Tensor:
    """Per-relation head-corruption probabilities for Bernoulli negative sampling.

    Args:
        triples_rht: ``[N, 3]`` triples in ``(r, h, t)`` format.
        num_relations: Total number of relations.

    Returns:
        ``[num_relations]`` float tensor of head-corruption probabilities,
        clamped to ``[0.05, 0.95]``.
    """
    t = triples_rht.cpu().long()
    rels = t[:, 0]
    heads = t[:, 1]
    tails = t[:, 2]
    N = t.shape[0]

    ones = torch.ones(N, dtype=torch.float32)
    triple_counts = torch.zeros(num_relations, dtype=torch.float32).scatter_add_(0, rels, ones)

    max_ent = max(heads.max().item(), tails.max().item()) + 1

    rh_rels = torch.unique(rels * max_ent + heads) // max_ent
    unique_head_counts = torch.zeros(num_relations, dtype=torch.float32).scatter_add_(
        0, rh_rels, torch.ones(rh_rels.shape[0], dtype=torch.float32)
    )

    rt_rels = torch.unique(rels * max_ent + tails) // max_ent
    unique_tail_counts = torch.zeros(num_relations, dtype=torch.float32).scatter_add_(
        0, rt_rels, torch.ones(rt_rels.shape[0], dtype=torch.float32)
    )

    # tph = tails per head, hpt = heads per tail
    tph = triple_counts / unique_head_counts.clamp(min=1)
    hpt = triple_counts / unique_tail_counts.clamp(min=1)

    denom = tph + hpt
    probs = torch.where(denom > 0, tph / denom, torch.full_like(denom, 0.5))
    return probs.clamp(0.05, 0.95)
