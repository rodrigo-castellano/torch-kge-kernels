"""Candidate pool construction for corruption-based evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import torch
from torch import Tensor

from ..types import SupportsCorruptWithMask


@dataclass
class CandidatePool:
    """Interleaved positive + negative candidate pool.

    Layout: column-major ``pool.view(K, CQ, 3)`` gives ``[slot, query, triple]``.
    Positives are at slot 0, negatives at slots 1..K-1.

    Attributes:
        pool: ``[CQ * K, 3]`` interleaved triples in ``(r, h, t)`` format.
        valid_mask: ``[CQ, K]`` boolean mask (``False`` for zero-padded slots).
        K: Candidates per query (``1 + n_corruptions``).
        CQ: Number of queries in this chunk.
    """

    pool: Tensor
    valid_mask: Tensor
    K: int
    CQ: int

    @property
    def pool_size(self) -> int:
        return self.CQ * self.K

    @staticmethod
    def build(
        queries: Tensor,
        sampler: SupportsCorruptWithMask,
        n_corruptions: int,
        mode: Literal["head", "tail"] = "head",
        device: Optional[torch.device] = None,
    ) -> "CandidatePool":
        """Build a candidate pool from queries using the sampler.

        Args:
            queries: ``[CQ, 3]`` positive triples in ``(r, h, t)`` format.
            sampler: Sampler with ``corrupt_with_mask`` method.
            n_corruptions: Number of negatives per positive.
            mode: Corruption side (``"head"`` or ``"tail"``).
            device: Device for the pool tensors.

        Returns:
            A ``CandidatePool`` with interleaved positives and negatives.
        """
        device = device or queries.device
        CQ = queries.shape[0]
        neg, neg_valid = sampler.corrupt_with_mask(
            queries, num_negatives=n_corruptions, mode=mode, device=device,
        )
        neg_count = neg.shape[1]
        K = 1 + neg_count

        # Build pool in column-major layout: [K, CQ, 3]
        pool = torch.zeros(K, CQ, 3, dtype=torch.long, device=device)
        pool[0] = queries.to(device)
        if neg_count > 0:
            pool[1:] = neg.transpose(0, 1)

        # Valid mask: [CQ, K] — positives always valid, negatives from sampler
        valid = torch.ones(CQ, K, dtype=torch.bool, device=device)
        if neg_count > 0:
            valid[:, 1:] = neg_valid

        return CandidatePool(pool=pool.reshape(-1, 3), valid_mask=valid, K=K, CQ=CQ)

    @staticmethod
    def build_into(
        buffer: Tensor,
        offset: int,
        queries: Tensor,
        sampler: SupportsCorruptWithMask,
        n_corruptions: int,
        mode: Literal["head", "tail"] = "head",
        device: Optional[torch.device] = None,
    ) -> "CandidatePool":
        """Build pool into a pre-allocated buffer. Zero allocation.

        Writes into ``buffer[offset : offset + CQ * K]``. The buffer must
        be large enough. For ``reduce-overhead`` / CUDA graph compatibility.

        Args:
            buffer: Pre-allocated ``[max_pool, 3]`` tensor.
            offset: Write position in the buffer.
            queries: ``[CQ, 3]`` positive triples.
            sampler: Sampler with ``corrupt_with_mask``.
            n_corruptions: Negatives per positive.
            mode: Corruption side.
            device: Device override.

        Returns:
            ``CandidatePool`` whose ``.pool`` is a view into the buffer.
        """
        device = device or queries.device
        CQ = queries.shape[0]
        neg, neg_valid = sampler.corrupt_with_mask(
            queries, num_negatives=n_corruptions, mode=mode, device=device,
        )
        neg_count = neg.shape[1]
        K = 1 + neg_count
        P = CQ * K

        # Write into buffer (column-major layout)
        buf_view = buffer[offset:offset + P].view(K, CQ, 3)
        buf_view[0].copy_(queries.to(device))
        if neg_count > 0:
            buf_view[1:].copy_(neg.transpose(0, 1))

        valid = torch.ones(CQ, K, dtype=torch.bool, device=device)
        if neg_count > 0:
            valid[:, 1:] = neg_valid

        return CandidatePool(pool=buffer[offset:offset + P], valid_mask=valid, K=K, CQ=CQ)

    @staticmethod
    def build_batched(
        queries: Tensor,
        sampler: SupportsCorruptWithMask,
        n_corruptions: int,
        modes: Sequence[str],
        device: Optional[torch.device] = None,
        buffer: Optional[Tensor] = None,
    ) -> tuple[Tensor, list["CandidatePool"]]:
        """Build pools for multiple corruption modes into one buffer.

        Returns the combined flat pool and a list of ``CandidatePool`` objects
        (one per mode). If ``buffer`` is provided, writes into it (zero alloc);
        otherwise allocates.

        Args:
            queries: ``[CQ, 3]`` positive triples.
            sampler: Sampler with ``corrupt_with_mask``.
            n_corruptions: Negatives per positive.
            modes: Corruption modes (e.g. ``["head", "tail"]``).
            device: Device.
            buffer: Optional pre-allocated ``[max_pool, 3]`` tensor.

        Returns:
            ``(combined_pool, pools)`` — flat ``[total, 3]`` tensor and per-mode
            pool metadata.
        """
        pools = []
        offset = 0
        for mode in modes:
            if buffer is not None:
                pool_obj = CandidatePool.build_into(
                    buffer, offset, queries, sampler, n_corruptions, mode=mode, device=device,
                )
            else:
                pool_obj = CandidatePool.build(
                    queries, sampler, n_corruptions, mode=mode, device=device,
                )
            pools.append(pool_obj)
            offset += pool_obj.pool_size

        if buffer is not None:
            combined = buffer[:offset]
        else:
            combined = torch.cat([p.pool for p in pools])

        return combined, pools


__all__ = ["CandidatePool"]
