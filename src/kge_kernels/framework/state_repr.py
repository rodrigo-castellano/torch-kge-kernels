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

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .repr import Repr
from .types import ProofEvidence


def _per_atom_validity_mask(atom_lead_shape: tuple, evidence: ProofEvidence, device) -> Tensor:
    """Build a per-atom boolean validity mask matching the atom leading shape.

    First tries ``evidence.body_atom_mask_flat`` (per-atom mask that
    supports interspersed-padding layouts). Falls back to ``body_count``-
    derived prefix masks when ``body_atom_mask_flat`` is unavailable or
    its shape doesn't match.

    Structured fallback: ``[B, C, D, M]`` from ``body_count[B, C, D]``.
    Legacy fallback:     ``[B, C, G_body]`` from ``body_count[B, C]``.
    """
    custom = getattr(evidence, "body_atom_mask_flat", None)
    if isinstance(custom, Tensor) and tuple(custom.shape) == tuple(atom_lead_shape):
        return custom

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
    """Sum atom values per body (masking padded atoms to zero).

    Polymorphic on ``atom_repr``: sums embeddings if present, otherwise
    sums scores. Output ``Repr`` carries the same field as the input.
    """

    def forward(self, atom_repr: Repr, evidence: ProofEvidence) -> Repr:
        if atom_repr.has_embeddings:
            emb = atom_repr.embeddings
            mask = _per_atom_validity_mask(tuple(emb.shape[:-1]), evidence, emb.device)
            masked = emb * mask.unsqueeze(-1).to(emb.dtype)
            return Repr(embeddings=masked.sum(dim=-2))
        if atom_repr.has_scores:
            sc = atom_repr.scores
            mask = _per_atom_validity_mask(tuple(sc.shape), evidence, sc.device)
            masked = sc * mask.to(sc.dtype)
            return Repr(scores=masked.sum(dim=-1))
        raise ValueError("SumStateRepr requires atom_repr.embeddings or atom_repr.scores")


class MeanStateRepr(nn.Module):
    """Mean of atom values per body, ignoring padded atoms.

    Polymorphic on ``atom_repr``: averages embeddings if present,
    otherwise averages scores. Output ``Repr`` carries the same field
    as the input.
    """

    def forward(self, atom_repr: Repr, evidence: ProofEvidence) -> Repr:
        if atom_repr.has_embeddings:
            emb = atom_repr.embeddings
            mask = _per_atom_validity_mask(tuple(emb.shape[:-1]), evidence, emb.device)
            m = mask.unsqueeze(-1).to(emb.dtype)
            masked_sum = (emb * m).sum(dim=-2)
            denom = m.sum(dim=-2).clamp(min=1.0)
            return Repr(embeddings=masked_sum / denom)
        if atom_repr.has_scores:
            sc = atom_repr.scores
            mask = _per_atom_validity_mask(tuple(sc.shape), evidence, sc.device)
            m = mask.to(sc.dtype)
            masked_sum = (sc * m).sum(dim=-1)
            denom = m.sum(dim=-1).clamp(min=1.0)
            return Repr(scores=masked_sum / denom)
        raise ValueError("MeanStateRepr requires atom_repr.embeddings or atom_repr.scores")


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


# ═══════════════════════════════════════════════════════════════════════
# Hybrid Φ/Ψ aggregator (DCR)
# ═══════════════════════════════════════════════════════════════════════


class PhiPsiStateRepr(nn.Module):
    """DCR's per-rule Φ/Ψ message aggregator (framework.pdf §6.2).

    For each body atom ``b`` at depth ``d`` in proof ``p`` with rule
    index ``r = rule_idx[B, P, d]``::

        e' = Φ_r(e(b), o(b))             # embedding × score → embedding'
        m  = Ψ_r(e(b), e')               # embedding × embedding' → score'

    Then aggregate ``m`` over atoms in the body via t-norm (min or product).

    Per-rule MLPs are stored as ``[R, ...]`` parameter tensors and
    gathered by ``rule_idx``. Memory cost scales with ``num_rules``;
    for typical KG datasets (R ≤ 10) this is fine.

    Requires ``atom_repr`` to have BOTH ``embeddings`` and ``scores``
    — DCR is the hybrid-regime row of framework.pdf §11. Pair with
    :class:`KGEBothAtom` upstream.
    """

    def __init__(
        self,
        num_rules: int,
        embed_dim: int,
        *,
        hidden_dim: Optional[int] = None,
        tnorm: Literal["min", "product"] = "product",
    ) -> None:
        super().__init__()
        if num_rules < 1:
            raise ValueError("num_rules must be >= 1")
        if tnorm not in ("min", "product"):
            raise ValueError(f"Unknown t-norm: {tnorm}")
        h = hidden_dim or embed_dim
        self.num_rules = num_rules
        self.embed_dim = embed_dim
        self.tnorm = tnorm
        # Φ_r: concat([E], [1]) → E. Stored as grouped linear params.
        self.phi_l1 = nn.Parameter(torch.empty(num_rules, embed_dim + 1, h))
        self.phi_b1 = nn.Parameter(torch.zeros(num_rules, h))
        self.phi_l2 = nn.Parameter(torch.empty(num_rules, h, embed_dim))
        self.phi_b2 = nn.Parameter(torch.zeros(num_rules, embed_dim))
        # Ψ_r: concat([E], [E]) → 1.
        self.psi_l1 = nn.Parameter(torch.empty(num_rules, 2 * embed_dim, h))
        self.psi_b1 = nn.Parameter(torch.zeros(num_rules, h))
        self.psi_l2 = nn.Parameter(torch.empty(num_rules, h, 1))
        self.psi_b2 = nn.Parameter(torch.zeros(num_rules, 1))
        for p in (self.phi_l1, self.phi_l2, self.psi_l1, self.psi_l2):
            nn.init.kaiming_uniform_(p, a=5 ** 0.5)

    def forward(self, atom_repr: Repr, evidence: ProofEvidence) -> Repr:
        if not atom_repr.has_embeddings or not atom_repr.has_scores:
            raise ValueError(
                "PhiPsiStateRepr requires atom_repr with both embeddings and scores"
            )
        emb = atom_repr.embeddings           # [B, C, D, M, E] or [B, C, G, E]
        sc = atom_repr.scores                # [B, C, D, M]    or [B, C, G]
        rule_idx = evidence.rule_idx         # [B, C, D]       or [B, C]

        # Broadcast rule_idx to atom leading shape (add the M/G dim).
        rule_atom = rule_idx.unsqueeze(-1).expand(emb.shape[:-1])

        # Φ_r(e, s) → e'
        es = torch.cat([emb, sc.unsqueeze(-1)], dim=-1)              # [..., E+1]
        phi_l1_g = self.phi_l1[rule_atom]                            # [..., E+1, h]
        phi_b1_g = self.phi_b1[rule_atom]                            # [..., h]
        h1 = F.relu(torch.einsum("...e,...eh->...h", es, phi_l1_g) + phi_b1_g)
        phi_l2_g = self.phi_l2[rule_atom]                            # [..., h, E]
        phi_b2_g = self.phi_b2[rule_atom]                            # [..., E]
        e_prime = torch.einsum("...h,...he->...e", h1, phi_l2_g) + phi_b2_g

        # Ψ_r(e, e') → m (scalar score per atom)
        ee = torch.cat([emb, e_prime], dim=-1)                        # [..., 2E]
        psi_l1_g = self.psi_l1[rule_atom]                            # [..., 2E, h]
        psi_b1_g = self.psi_b1[rule_atom]
        h2 = F.relu(torch.einsum("...e,...eh->...h", ee, psi_l1_g) + psi_b1_g)
        psi_l2_g = self.psi_l2[rule_atom]                            # [..., h, 1]
        psi_b2_g = self.psi_b2[rule_atom]                            # [..., 1]
        msg = torch.einsum("...h,...hk->...k", h2, psi_l2_g) + psi_b2_g
        msg = torch.sigmoid(msg.squeeze(-1))                         # [..., M] in [0,1]

        # T-norm over atoms (last leading dim).
        mask = _per_atom_validity_mask(tuple(msg.shape), evidence, msg.device)
        if self.tnorm == "min":
            big = torch.finfo(msg.dtype).max
            masked = torch.where(mask, msg, torch.full_like(msg, big))
            reduced = masked.min(dim=-1).values
            any_valid = mask.any(dim=-1)
            reduced = torch.where(any_valid, reduced, torch.zeros_like(reduced))
        else:  # product
            ones = torch.ones_like(msg)
            masked = torch.where(mask, msg, ones)
            reduced = masked.prod(dim=-1)
        return Repr(scores=reduced)


__all__ = [
    "ConcatStateRepr",
    "MaxStateRepr",
    "MeanStateRepr",
    "PhiPsiStateRepr",
    "SumStateRepr",
    "TNormStateRepr",
]
