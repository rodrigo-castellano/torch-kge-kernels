"""StateRepr implementations: per-atom Repr → per-state Repr.

A "state" here means the body of one resolution step at one depth (or, in
legacy flat layout, the whole accumulated body of one candidate proof).
The StateRepr reduces the M (atoms-per-body) dimension and respects
``evidence.body_atom_mask_flat`` for masking.

Output leading shape:
  - structured (evidence.D > 0): ``[B, C, D]`` — one state Repr per depth
  - legacy flat (evidence.D == 0): ``[B, C]``  — one state Repr per body
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from .repr import Repr
from .types import ProofEvidence


def _per_atom_validity_mask(atom_lead_shape: tuple, evidence: ProofEvidence, device) -> Tensor:
    """Build a per-atom boolean validity mask matching the atom leading shape.

    Structured: ``[B, C, D, M]`` from ``body_count[B, C, D]``.
    Legacy:     ``[B, C, G_body]`` from ``body_count[B, C]``.
    """
    body_count = evidence.body_count
    if body_count.dim() == 3:
        # structured: [B, C, D] body_count, atoms-per-depth = M
        B, C, D = body_count.shape
        if len(atom_lead_shape) != 4:
            raise ValueError(
                f"StateRepr expected atom shape [B,C,D,M] for structured evidence; got {atom_lead_shape}"
            )
        M = atom_lead_shape[3]
        m_idx = torch.arange(M, device=device)
        return m_idx < body_count.unsqueeze(-1)         # [B, C, D, M]
    # legacy flat
    if len(atom_lead_shape) != 3:
        raise ValueError(
            f"StateRepr expected atom shape [B,C,G_body] for legacy evidence; got {atom_lead_shape}"
        )
    G = atom_lead_shape[2]
    g_idx = torch.arange(G, device=device)
    return g_idx < body_count.unsqueeze(-1)             # [B, C, G_body]


# ═══════════════════════════════════════════════════════════════════════
# Score-path aggregators (T-norm)
# ═══════════════════════════════════════════════════════════════════════


class TNormStateRepr(nn.Module):
    """Aggregate scores via a t-norm (min or product) over atoms per body.

    Reduction is along the last leading dim (M for structured, G for legacy).
    Padded atoms contribute the t-norm identity (1.0) so they do not change
    the result.
    """

    def __init__(self, tnorm: Literal["min", "product"] = "min") -> None:
        super().__init__()
        if tnorm not in ("min", "product"):
            raise ValueError(f"Unknown t-norm: {tnorm}")
        self.tnorm = tnorm

    def forward(self, atom_repr: Repr, evidence: ProofEvidence) -> Repr:
        if not atom_repr.has_scores:
            raise ValueError("TNormStateRepr requires atom_repr.scores")
        scores = atom_repr.scores
        mask = _per_atom_validity_mask(tuple(scores.shape), evidence, scores.device)
        if self.tnorm == "min":
            # Pad invalid atoms with +inf so they cannot lower the min.
            big = torch.finfo(scores.dtype).max
            masked = torch.where(mask, scores, torch.full_like(scores, big))
            reduced = masked.min(dim=-1).values
            # If a whole body is empty, the min collapses to +inf — replace with 0
            any_valid = mask.any(dim=-1)
            reduced = torch.where(any_valid, reduced, torch.zeros_like(reduced))
        else:
            ones = torch.ones_like(scores)
            masked = torch.where(mask, scores, ones)
            reduced = masked.prod(dim=-1)
        return Repr(scores=reduced)


# ═══════════════════════════════════════════════════════════════════════
# Embedding-path aggregators (Sum / Mean / Max / Concat)
# ═══════════════════════════════════════════════════════════════════════


class SumStateRepr(nn.Module):
    """Sum atom embeddings per body (masking padded atoms to zero)."""

    def forward(self, atom_repr: Repr, evidence: ProofEvidence) -> Repr:
        if not atom_repr.has_embeddings:
            raise ValueError("SumStateRepr requires atom_repr.embeddings")
        emb = atom_repr.embeddings
        mask = _per_atom_validity_mask(tuple(emb.shape[:-1]), evidence, emb.device)
        masked = emb * mask.unsqueeze(-1).to(emb.dtype)
        return Repr(embeddings=masked.sum(dim=-2))


class MeanStateRepr(nn.Module):
    """Mean of atom embeddings per body, ignoring padded atoms."""

    def forward(self, atom_repr: Repr, evidence: ProofEvidence) -> Repr:
        if not atom_repr.has_embeddings:
            raise ValueError("MeanStateRepr requires atom_repr.embeddings")
        emb = atom_repr.embeddings
        mask = _per_atom_validity_mask(tuple(emb.shape[:-1]), evidence, emb.device)
        m = mask.unsqueeze(-1).to(emb.dtype)
        masked_sum = (emb * m).sum(dim=-2)
        denom = m.sum(dim=-2).clamp(min=1.0)
        return Repr(embeddings=masked_sum / denom)


class MaxStateRepr(nn.Module):
    """Element-wise max over atom embeddings per body (padded → -inf)."""

    def forward(self, atom_repr: Repr, evidence: ProofEvidence) -> Repr:
        if not atom_repr.has_embeddings:
            raise ValueError("MaxStateRepr requires atom_repr.embeddings")
        emb = atom_repr.embeddings
        mask = _per_atom_validity_mask(tuple(emb.shape[:-1]), evidence, emb.device)
        neg_inf = torch.finfo(emb.dtype).min
        masked = emb.masked_fill(~mask.unsqueeze(-1), neg_inf)
        reduced = masked.max(dim=-2).values
        any_valid = mask.any(dim=-1, keepdim=True)
        reduced = torch.where(any_valid, reduced, torch.zeros_like(reduced))
        return Repr(embeddings=reduced)


class ConcatStateRepr(nn.Module):
    """Concatenate atom embeddings per body, padding/truncating to ``max_atoms``.

    Useful for fixed-width MLP downstream consumers (R2N-style).
    """

    def __init__(self, max_atoms: int) -> None:
        super().__init__()
        self.max_atoms = max_atoms

    def forward(self, atom_repr: Repr, evidence: ProofEvidence) -> Repr:
        if not atom_repr.has_embeddings:
            raise ValueError("ConcatStateRepr requires atom_repr.embeddings")
        emb = atom_repr.embeddings                          # [..., A, E]
        A = emb.shape[-2]
        E = emb.shape[-1]
        if A < self.max_atoms:
            pad_shape = list(emb.shape)
            pad_shape[-2] = self.max_atoms - A
            pad = torch.zeros(pad_shape, dtype=emb.dtype, device=emb.device)
            emb = torch.cat([emb, pad], dim=-2)
        elif A > self.max_atoms:
            emb = emb[..., : self.max_atoms, :]
        flat = emb.reshape(*emb.shape[:-2], self.max_atoms * E)
        return Repr(embeddings=flat)


__all__ = [
    "ConcatStateRepr",
    "MaxStateRepr",
    "MeanStateRepr",
    "SumStateRepr",
    "TNormStateRepr",
]
