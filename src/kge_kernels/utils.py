"""Internal tensor utilities for torch-kge-kernels."""

from __future__ import annotations

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
