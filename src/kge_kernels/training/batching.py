"""Shared per-epoch training batch iterator.

Used by tkk's BCE training pipeline and by ns's training loop. Both
consume the same per-epoch schedule: sample all negatives once, shuffle,
iterate in fixed-size batches. The iterator yields positive and negative
tensors in their structured shape so callers (KGE-only, reasoner) can
flatten or reshape as they need.
"""
from __future__ import annotations

from typing import Iterator, List, Optional, Tuple

import torch
from torch import Tensor

from ..scoring import Sampler


def iterate_epoch_batches(
    train_triples: Tensor,
    sampler: Sampler,
    *,
    batch_size: int,
    num_negatives: int,
    corrupt_modes: List[str],
    generator: Optional[torch.Generator] = None,
    filter: bool = True,
    unique: bool = False,
) -> Iterator[Tuple[Tensor, Tensor, Tensor]]:
    """Yield (pos, neg, valid) batches for one training epoch.

    Samples fresh negatives once at the start of the epoch (one call to
    the sampler per mode, concatenated along the negatives axis). Then
    shuffles via ``torch.randperm`` and yields fixed-size batches.

    Args:
        train_triples: ``[N, 3]`` long tensor of positives in ``(r, h, t)``.
        sampler: Configured tkk ``Sampler``.
        batch_size: Positive batch size ``B``.
        num_negatives: Negatives per positive, per corruption mode.
        corrupt_modes: One or more sampler modes, e.g. ``["head", "tail"]``
            for head-and-tail training, or ``["bernoulli"]`` / ``["tail"]``.
        generator: Optional ``torch.Generator`` for reproducible shuffling.
        filter: Pass through to ``Sampler.corrupt_with_mask``.
        unique: Pass through to ``Sampler.corrupt_with_mask``.

    Yields:
        ``(pos [B, 3], neg [B, K_total, 3], valid [B, K_total])``, where
        ``K_total = len(corrupt_modes) * num_negatives``. The last batch
        may have fewer than ``B`` rows.
    """
    N = train_triples.shape[0]
    device = train_triples.device

    # Per-epoch negative sampling (once, not per batch).
    all_negs: List[Tensor] = []
    all_valid: List[Tensor] = []
    for mode in corrupt_modes:
        neg, valid = sampler.corrupt_with_mask(
            train_triples, num_negatives=num_negatives, mode=mode,
            filter=filter, unique=unique,
        )
        all_negs.append(neg)
        all_valid.append(valid)
    neg_epoch = torch.cat(all_negs, dim=1)       # [N, K_total, 3]
    valid_epoch = torch.cat(all_valid, dim=1)    # [N, K_total]

    perm = torch.randperm(N, device=device, generator=generator)

    for start in range(0, N, batch_size):
        idx = perm[start:start + batch_size]
        yield train_triples[idx], neg_epoch[idx], valid_epoch[idx]


__all__ = ["iterate_epoch_batches"]
