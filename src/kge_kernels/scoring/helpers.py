"""On-the-fly corruption helpers built on top of :class:`Sampler`.

The Sampler primitive (``corrupt_with_mask``) does one shot of vectorized
corruption per call. When callers need exactly ``K`` valid negatives per
query — typical of per-batch on-the-fly corruption inside a PyTorch
``Dataset.__getitem__`` — small domain pools can leave some rows short
of ``K`` after the filter+unique pass. ``corrupt_with_topup`` retries a
bounded number of times with doubled draw counts and tops up any
remaining slots by drawing duplicates from each query's unfiltered pool,
so the returned shape is dense and consumers don't have to handle holes.

This is the on-the-fly path; the bulk SL training pipeline keeps using
``Sampler.corrupt_with_mask`` directly (one big call per epoch) and does
not need topup.
"""
from __future__ import annotations

import random
from typing import List, Optional, Tuple

import torch

from .sampler import Sampler


def corrupt_to_lists(
    neg: torch.Tensor, valid: torch.Tensor,
) -> List[List[Tuple[int, int, int]]]:
    """Convert ``(neg [B, K, 3], valid [B, K])`` to a Python list of valid rows.

    Drops every entry whose ``valid`` flag is False; preserves per-query
    grouping. Returns rows in ``(r, h, t)`` order matching the sampler's
    public tensor format.
    """
    out: List[List[Tuple[int, int, int]]] = []
    for triples, mask in zip(neg.tolist(), valid.tolist()):
        out.append([
            tuple(map(int, triple))
            for triple, keep in zip(triples, mask) if keep
        ])
    return out


def corrupt_with_topup(
    sampler: Sampler,
    queries: List[Tuple[int, int, int]],
    *,
    num_negatives: Optional[int],
    mode: str,
    max_retries: int = 4,
) -> List[List[Tuple[int, int, int]]]:
    """Generate ``num_negatives`` valid negatives per query, with retry + duplicate fallback.

    Wraps :meth:`Sampler.corrupt_with_mask` (with ``filter=True, unique=False``)
    and retries up to ``max_retries`` times, doubling the draw count each
    pass to maximize the chance of filling every row. If after the retries
    some rows are still short, falls back to an unfiltered exhaustive draw
    and tops up via :func:`random.choices` from each query's own pool.

    ``num_negatives=None`` skips topup entirely and returns the variable-length
    valid rows from a single exhaustive draw — used by ranking eval where
    "all candidates" is the desired output, not a fixed-K bundle.

    Returns a list of length ``len(queries)`` where each row has exactly
    ``num_negatives`` entries (or as close as the sampler can deliver in
    extreme small-domain edge cases).
    """
    if not queries:
        return []
    queries_tensor = torch.tensor(queries, dtype=torch.long)

    if num_negatives is None:
        neg, valid = sampler.corrupt_with_mask(
            queries_tensor, num_negatives=None, mode=mode,
            device=torch.device("cpu"), filter=True, unique=False,
        )
        return corrupt_to_lists(neg, valid)

    target = num_negatives
    gathered: List[List[Tuple[int, int, int]]] = [[] for _ in queries]
    pending = list(range(len(queries)))

    for _ in range(max_retries):
        if not pending:
            break
        sub = queries_tensor[pending]
        remaining = [target - len(gathered[idx]) for idx in pending]
        draw_k = max(1, max(remaining)) * 2
        neg, valid = sampler.corrupt_with_mask(
            sub, num_negatives=draw_k, mode=mode,
            device=torch.device("cpu"), filter=True, unique=False,
        )
        rows = corrupt_to_lists(neg, valid)
        next_pending: List[int] = []
        for idx, row, need in zip(pending, rows, remaining):
            if row:
                gathered[idx].extend(row[:need])
            if len(gathered[idx]) < target:
                next_pending.append(idx)
        pending = next_pending

    if pending:
        fallback = corrupt_with_topup(
            sampler, [queries[idx] for idx in pending],
            num_negatives=None, mode=mode,
        )
        for idx, row in zip(pending, fallback):
            if not row:
                continue
            need = target - len(gathered[idx])
            gathered[idx].extend(random.choices(row, k=need))

    return [row[:target] for row in gathered]


__all__ = ["corrupt_to_lists", "corrupt_with_topup"]
