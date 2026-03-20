"""Training-time negative sampling helpers shared across projects."""

from __future__ import annotations

import random
from typing import Sequence, Tuple

import torch
from torch import Tensor


def compute_bernoulli_probs(
    triples: Sequence[Tuple[int, int, int]],
    num_relations: int,
) -> Tensor:
    counts_head = [dict() for _ in range(num_relations)]
    counts_tail = [dict() for _ in range(num_relations)]
    for h, r, t in triples:
        counts_head[r][h] = counts_head[r].get(h, 0) + 1
        counts_tail[r][t] = counts_tail[r].get(t, 0) + 1
    tph = torch.zeros(num_relations, dtype=torch.float)
    hpt = torch.zeros(num_relations, dtype=torch.float)
    for ridx in range(num_relations):
        if counts_head[ridx]:
            tph[ridx] = sum(counts_head[ridx].values()) / max(1, len(counts_head[ridx]))
        if counts_tail[ridx]:
            hpt[ridx] = sum(counts_tail[ridx].values()) / max(1, len(counts_tail[ridx]))
    denom = tph + hpt
    probs = torch.where(denom > 0, tph / denom, torch.full_like(denom, 0.5))
    return probs.clamp(0.05, 0.95)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_known_triple_hash_tensor(
    triples: Sequence[Tuple[int, int, int]],
    *,
    num_entities: int,
    num_relations: int,
    device: torch.device,
) -> Tensor:
    if not triples:
        return torch.empty(0, dtype=torch.long, device=device)
    hashes = [((h * num_relations) + r) * num_entities + t for h, r, t in triples]
    return torch.unique(torch.tensor(hashes, dtype=torch.long, device=device))


def sorted_membership(sorted_values: Tensor, query: Tensor) -> Tensor:
    if sorted_values.numel() == 0:
        return torch.zeros_like(query, dtype=torch.bool)
    idx = torch.searchsorted(sorted_values, query)
    valid = idx < sorted_values.numel()
    clamped = idx.clamp(max=sorted_values.numel() - 1)
    return valid & (sorted_values[clamped] == query)


def sample_batch_negatives(
    *,
    batch: Tensor,
    neg_ratio: int,
    bern_probs: Tensor,
    known_triple_hashes: Tensor,
    num_entities: int,
    num_relations: int,
) -> Tensor:
    batch_size = batch.size(0)
    if batch_size < 2:
        raise RuntimeError("Batch redistribution requires batch_size >= 2")

    rels = batch[:, 1]
    probs = bern_probs.to(rels.device)[rels]
    corrupt_head = torch.bernoulli(probs).bool().unsqueeze(0).expand(neg_ratio, -1).clone()
    negatives = batch.unsqueeze(0).expand(neg_ratio, -1, -1).clone()
    unresolved = torch.ones((neg_ratio, batch_size), dtype=torch.bool, device=batch.device)

    batch_heads = batch[:, 0]
    batch_tails = batch[:, 2]
    base_idx = torch.arange(batch_size, device=batch.device)
    start_shifts = torch.randint(1, batch_size, size=(neg_ratio,), device=batch.device)

    for attempt in range(batch_size - 1):
        shifts = ((start_shifts + attempt - 1) % (batch_size - 1)) + 1
        source_idx = (base_idx.unsqueeze(0) + shifts.unsqueeze(1)) % batch_size
        candidate_heads = batch_heads[source_idx]
        candidate_tails = batch_tails[source_idx]

        proposal_heads = torch.where(unresolved & corrupt_head, candidate_heads, negatives[:, :, 0])
        proposal_tails = torch.where(unresolved & ~corrupt_head, candidate_tails, negatives[:, :, 2])
        triple_hashes = ((proposal_heads * num_relations) + negatives[:, :, 1]) * num_entities + proposal_tails
        valid = ~sorted_membership(known_triple_hashes, triple_hashes)
        accepted = unresolved & valid

        negatives[:, :, 0] = torch.where(accepted & corrupt_head, candidate_heads, negatives[:, :, 0])
        negatives[:, :, 2] = torch.where(accepted & ~corrupt_head, candidate_tails, negatives[:, :, 2])
        unresolved &= ~accepted
        if not unresolved.any():
            return negatives.reshape(-1, 3)

    raise RuntimeError("Batch redistribution could not produce filtered negatives")


def sample_random_negatives(
    *,
    batch: Tensor,
    neg_ratio: int,
    bern_probs: Tensor,
    num_entities: int,
) -> Tensor:
    """Legacy unfiltered random corruption used by current DpRL KGE training."""
    rels = batch[:, 1]
    probs = bern_probs.to(rels.device)[rels]
    corrupt_head = torch.bernoulli(probs).bool()
    expanded = batch.repeat_interleave(neg_ratio, dim=0)
    corrupt_head = corrupt_head.repeat_interleave(neg_ratio).to(expanded.device)
    rand_entities = torch.randint(0, num_entities, size=(expanded.size(0),), device=expanded.device)
    expanded[corrupt_head, 0] = rand_entities[corrupt_head]
    expanded[~corrupt_head, 2] = rand_entities[~corrupt_head]
    return expanded
