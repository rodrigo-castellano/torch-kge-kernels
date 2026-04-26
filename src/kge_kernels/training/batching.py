"""Shared batch acquisition primitives for training and env loops.

Two complementary access patterns:

- ``iterate_epoch_batches`` — SL pattern. Pre-samples all negatives once,
  shuffles via ``torch.randperm``, yields ``(pos, neg, valid)`` per batch
  for full epoch coverage. Used by tkk's BCE training pipeline and by ns's
  training loop.

- ``pick_query_batch`` — RL/episodic pattern. Picks ``B`` queries from a
  pool in one shot (weighted multinomial / round-robin / uniform). No
  corruption; callers (e.g. DpRL's ``EnvVec.reset``) own the curriculum
  decision and call ``Sampler.corrupt`` themselves for negatives.

Raw triple corruption itself stays in :class:`kge_kernels.scoring.Sampler`
— this module never produces negatives directly.
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
        filter: Pass through to ``Sampler.corrupt``.
        unique: Pass through to ``Sampler.corrupt``.

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
        neg, valid = sampler.corrupt(
            train_triples, num_negatives=num_negatives, mode=mode,
            filter=filter, unique=unique, return_mask=True,
        )
        all_negs.append(neg)
        all_valid.append(valid)
    neg_epoch = torch.cat(all_negs, dim=1)       # [N, K_total, 3]
    valid_epoch = torch.cat(all_valid, dim=1)    # [N, K_total]

    perm = torch.randperm(N, device=device, generator=generator)

    for start in range(0, N, batch_size):
        idx = perm[start:start + batch_size]
        yield train_triples[idx], neg_epoch[idx], valid_epoch[idx]


def pick_query_batch(
    queries: Tensor,
    batch_size: int,
    *,
    sampling_weights: Optional[Tensor] = None,
    ptrs: Optional[Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Pick a batch of queries from a pool of size ``N``.

    Selection mode (priority):
        1. Weighted multinomial — if ``sampling_weights`` is given.
        2. Round-robin — else if ``ptrs`` is given (``indices = ptrs % N``).
        3. Uniform random — fallback.

    Pointer advance: when ``ptrs`` is given, the returned ``new_ptrs`` is
    ``(ptrs + 1) % N`` regardless of which mode produced the indices. The
    advance is always by 1; callers needing custom advance semantics
    (slotted curriculum, masked done envs) should manage ptrs themselves.

    Args:
        queries: ``[N, *]`` pool tensor (any per-row shape).
        batch_size: number of queries to pick.
        sampling_weights: optional ``[N]`` weights for weighted sampling.
        ptrs: optional ``[B]`` long tensor of per-slot pointers.
        generator: optional ``torch.Generator`` for randint / multinomial.

    Returns:
        ``(batch [B, *], indices [B], new_ptrs [B] | None)``. ``new_ptrs``
        is ``None`` when ``ptrs`` was not supplied.
    """
    N = queries.shape[0]
    device = queries.device

    if sampling_weights is not None:
        indices = torch.multinomial(
            sampling_weights, batch_size, replacement=True, generator=generator
        )
    elif ptrs is not None:
        indices = ptrs % N
    else:
        indices = torch.randint(0, N, (batch_size,), device=device, generator=generator)

    new_ptrs = (ptrs + 1) % N if ptrs is not None else None
    return queries[indices], indices, new_ptrs


__all__ = ["iterate_epoch_batches", "pick_query_batch"]
