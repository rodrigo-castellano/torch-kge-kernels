"""StateRepr implementations: per-atom Repr → per-state Repr.

A "state" here means the body of one resolution step at one depth (or, in
legacy flat layout, the whole accumulated body of one candidate proof).
The StateRepr reduces the M (atoms-per-body) dimension and respects
``evidence.body_atom_mask_flat`` for masking.

Output leading shape:
  - structured (evidence.D > 0): ``[B, P, D]`` — one state Repr per depth
  - legacy flat (evidence.D == 0): ``[B, P]``  — one state Repr per body
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

    Structured fallback: ``[B, P, D, M]`` from ``body_count[B, P, D]``.
    Legacy fallback:     ``[B, P, G_body]`` from ``body_count[B, P]``.
    """
    custom = getattr(evidence, "body_atom_mask_flat", None)
    if isinstance(custom, Tensor) and tuple(custom.shape) == tuple(atom_lead_shape):
        return custom

    body_count = evidence.body_count
    if body_count.dim() == 3:
        # structured: [B, P, D] body_count, atoms-per-depth = M
        B, P, D = body_count.shape
        if len(atom_lead_shape) != 4:
            raise ValueError(
                f"StateRepr expected atom shape [B,P,D,M] for structured evidence; got {atom_lead_shape}"
            )
        M = atom_lead_shape[3]
        m_idx = torch.arange(M, device=device)
        return m_idx < body_count.unsqueeze(-1)         # [B, P, D, M]
    # legacy flat
    if len(atom_lead_shape) != 3:
        raise ValueError(
            f"StateRepr expected atom shape [B,P,G_body] for legacy evidence; got {atom_lead_shape}"
        )
    G = atom_lead_shape[2]
    g_idx = torch.arange(G, device=device)
    return g_idx < body_count.unsqueeze(-1)             # [B, P, G_body]


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
        # Both "min" and "product" t-norms over [0,1] scores share the same
        # identity element (1.0): an empty conjunction is vacuously true.
        # ns's SBR matches this via ``masked_fill(~mask, 1.0)``.
        ones = torch.ones_like(scores)
        masked = torch.where(mask, scores, ones)
        if self.tnorm == "min":
            reduced = masked.min(dim=-1).values
        else:
            reduced = masked.prod(dim=-1)
        return Repr(scores=reduced)


class RuleWeightedStateRepr(nn.Module):
    """T-norm aggregator scaled by per-rule learnable weights.

    Implements the RNM (sigmoid) and DeepStocklog (softmax) state-repr
    rows of framework.pdf §11. After the t-norm reduction over body
    atoms, multiplies each grounding by a per-rule weight gathered from
    the rule index of that grounding::

        score(g) = activation(w_{rule(g)}) * tnorm(body_atom_scores)

    ``weight_mode`` chooses how the per-rule scalar is normalized:

    - ``"sigmoid"`` — element-wise σ(w_r), independent per rule (RNM).
      ``w`` initializes to 0 → σ(0) = 0.5 (uniform mid-prior).
    - ``"softmax"`` — softmax over rules, competitive (DeepStocklog).
      ``w`` initializes to 1 → softmax(1) = 1/R (uniform).

    The structured-vs-legacy layout decision matches
    :class:`TNormStateRepr`: the activation is broadcast against the
    rule-index tensor, whose shape lines up with the t-norm output —
    ``[B, P]`` for legacy flat evidence (``rule_idx``), ``[B, P, D]``
    for structured evidence (``rule_idx`` per depth). Invalid groundings
    typically carry ``rule_idx = -1``; PyTorch's negative tensor
    indexing wraps to ``rule_weights[R-1]``, which downstream
    ``MaxQueryRepr`` masks out via ``evidence.mask`` — so the wrong
    weight cannot leak into the final score.

    Pair upstream with :class:`KGEScoreAtom` (RNM/DeepStocklog ship as
    score-path reasoners; no body embeddings).
    """

    def __init__(
        self,
        num_rules: int,
        *,
        weight_mode: Literal["sigmoid", "softmax"] = "sigmoid",
        tnorm: Literal["min", "product"] = "min",
    ) -> None:
        super().__init__()
        if num_rules < 1:
            raise ValueError("num_rules must be >= 1")
        if weight_mode not in ("sigmoid", "softmax"):
            raise ValueError(f"Unknown weight_mode: {weight_mode}")
        self.num_rules = num_rules
        self.weight_mode = weight_mode
        # RNM convention: zeros → σ(0) = 0.5.
        # DeepStocklog convention: ones → softmax(1) = 1/R.
        init = torch.zeros(num_rules) if weight_mode == "sigmoid" else torch.ones(num_rules)
        self.rule_weights = nn.Parameter(init)
        self._tnorm_repr = TNormStateRepr(tnorm)

    def forward(self, atom_repr: Repr, evidence: ProofEvidence) -> Repr:
        s_repr = self._tnorm_repr(atom_repr, evidence)
        scores = s_repr.scores
        if self.weight_mode == "sigmoid":
            weights = torch.sigmoid(self.rule_weights)
        else:
            weights = F.softmax(self.rule_weights, dim=0)
        # rule_idx aligns with state_repr's leading shape: structured
        # evidence has [B, P, D] and the t-norm output is [B, P, D];
        # legacy flat evidence has [B, P] and the output is [B, P].
        rule_idx = evidence.rule_idx
        per_grounding_weight = weights[rule_idx.long()]
        return Repr(scores=per_grounding_weight * scores)


class GatedTNormStateRepr(nn.Module):
    """Gated SBR's gate-blended t-norm state aggregator (framework.pdf §11).

    Implements the GatedSBR row: a learned per-grounding gate decides how
    much to trust the t-norm body-min vs. defaulting to a vacuous-true 1::

        gate_g = sigmoid(gate_net(flatten(masked_atom_embeddings_g)))
        head_g = gate_g * tnorm_min(body_atom_scores_g)
                 + (1 - gate_g) * 1.0

    The residual to ``1.0`` prevents gate collapse: even when the network
    drives ``gate → 0``, valid groundings still score 1, so a query
    that has any valid grounding cannot collapse to 0 from the gate
    alone (it can still fail through ``MaxQueryRepr``'s mask handling).

    Pair upstream with an atom_repr that produces both scores AND
    per-atom embeddings — :class:`KGEPairAtom` (entity-pair concat,
    matches ns's pre-migration GatedSBRReasoning) or
    :class:`KGEBothAtom` (full triple compose). The gate input is the
    flattened embeddings, so the constructor needs to know the per-atom
    embedding dim and the maximum body atoms per grounding.

    Gate regularization (``L_gate = E[(gate - 0.5)²]`` over valid
    groundings) is exposed via the :attr:`gate_regularization` attribute
    after each forward call. The caller adds it to the training loss
    with a configurable λ — this primitive itself does not multiply by λ.
    """

    def __init__(
        self,
        embed_dim: int,
        max_body_atoms: int,
        *,
        gate_type: Literal["linear", "mlp"] = "linear",
        hidden_dim: int = 32,
        dropout: float = 0.3,
        tnorm: Literal["min", "product"] = "min",
    ) -> None:
        super().__init__()
        if gate_type not in ("linear", "mlp"):
            raise ValueError(f"Unknown gate_type: {gate_type}")
        self.embed_dim = embed_dim
        self.max_body_atoms = max_body_atoms
        self.gate_type = gate_type
        self._tnorm_repr = TNormStateRepr(tnorm)

        gate_input_dim = max_body_atoms * embed_dim
        if gate_type == "mlp":
            layer1 = nn.Linear(gate_input_dim, hidden_dim)
            layer2 = nn.Linear(hidden_dim, 1)
            # Init final layer at 0 → sigmoid(0)=0.5 (neutral blend at start).
            nn.init.zeros_(layer2.weight)
            nn.init.constant_(layer2.bias, 0.0)
            self.gate = nn.Sequential(
                layer1, nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                layer2, nn.Sigmoid(),
            )
        else:
            gate_linear = nn.Linear(gate_input_dim, 1)
            nn.init.zeros_(gate_linear.weight)
            nn.init.constant_(gate_linear.bias, 0.0)
            self.gate = nn.Sequential(gate_linear, nn.Sigmoid())

        # Updated by every forward; read by the caller to add to loss.
        self.gate_regularization: Optional[Tensor] = None

    def forward(self, atom_repr: Repr, evidence: ProofEvidence) -> Repr:
        if not atom_repr.has_scores or not atom_repr.has_embeddings:
            raise ValueError(
                "GatedTNormStateRepr requires atom_repr with both scores "
                "and embeddings (use KGEPairAtom or KGEBothAtom upstream)."
            )

        # 1. T-norm body-min (probabilistic head) over the body atoms.
        s_repr = self._tnorm_repr(atom_repr, evidence)
        prob_head = s_repr.scores

        # 2. Gate from masked embeddings. Zero out invalid atoms before
        # flattening so padding can't bias the gate.
        emb = atom_repr.embeddings
        atom_mask = _per_atom_validity_mask(tuple(emb.shape[:-1]), evidence, emb.device)
        masked_emb = emb * atom_mask.unsqueeze(-1).to(emb.dtype)
        leading_shape = masked_emb.shape[:-2]            # [B, P] or [B, P, D]
        M = masked_emb.shape[-2]
        E = masked_emb.shape[-1]
        gate_input = masked_emb.reshape(-1, M * E)
        gate = self.gate(gate_input).reshape(leading_shape)

        # 3. Residual blend.
        blended = gate * prob_head + (1.0 - gate)

        # 4. Stash the regularization signal for the caller.
        # Mask the centered-deviation by valid groundings so padded slots
        # don't dilute the mean toward 0.25.
        proof_mask = evidence.mask.to(gate.dtype)
        # ``proof_mask`` is [B, P]; ``gate`` matches the state-repr leading
        # shape. For structured evidence the gate is [B, P, D] and the
        # mask broadcasts naturally.
        if gate.dim() > proof_mask.dim():
            proof_mask = proof_mask.unsqueeze(-1).expand_as(gate)
        centered = (gate - 0.5) ** 2 * proof_mask
        self.gate_regularization = centered.sum() / proof_mask.sum().clamp(min=1.0)

        return Repr(scores=blended)


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
        emb = atom_repr.embeddings           # [B, P, D, M, E] or [B, P, G, E]
        sc = atom_repr.scores                # [B, P, D, M]    or [B, P, G]
        rule_idx = evidence.rule_idx         # [B, P, D]       or [B, P]

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


def _godel_iff(a: Tensor, b: Tensor) -> Tensor:
    """Godel biconditional: ``min(max(1-a, b), max(a, 1-b))``."""
    return torch.minimum(
        torch.maximum(1.0 - a, b),
        torch.maximum(a, 1.0 - b),
    )


class FilterSignStateRepr(nn.Module):
    """DCR's filter+sign Godel-attention state aggregator (framework.pdf §11).

    Implements the Barbiero/Marra DCR variant exactly:

      sign_r(e)   = sigmoid(sign_mlp_r(e))         # per-atom polarity
      filter_r(e) = sigmoid(filter_mlp_r(e))       # per-atom relevance
      sign_term   = godel_iff(sign_r(e), score(b)) # match polarity to score
      term        = max(1 - filter_r(e), sign_term)
      head_p      = tnorm_min(term over body atoms)

    Two per-rule MLPs (filter and sign), each ``E → hidden → 1`` with
    LeakyReLU between layers. ``sign_mlp`` returns logits; ``filter_mlp``
    returns sigmoid output (init Sigmoid in the final layer matches
    ns.DCRReasoning's pre-migration behavior).

    Pair upstream with :class:`KGEPairAtom` (entity-pair embeddings +
    KGE scores; matches ns's entity-pair convention) — the per-atom
    embedding goes into both MLPs and the per-atom score is the logical
    "value" the iff matches against.

    Aggregation over the body axis uses min by default; pass
    ``tnorm="product"`` for product semantics.
    """

    def __init__(
        self,
        num_rules: int,
        embed_dim: int,
        *,
        hidden_dim: int = 64,
        tnorm: Literal["min", "product"] = "min",
    ) -> None:
        super().__init__()
        if num_rules < 1:
            raise ValueError("num_rules must be >= 1")
        if tnorm not in ("min", "product"):
            raise ValueError(f"Unknown t-norm: {tnorm}")
        self.num_rules = num_rules
        self.embed_dim = embed_dim
        self.tnorm = tnorm
        # Per-rule filter MLP: E → hidden → 1, sigmoid.
        self.filter_l1 = nn.Parameter(torch.empty(num_rules, embed_dim, hidden_dim))
        self.filter_b1 = nn.Parameter(torch.zeros(num_rules, hidden_dim))
        self.filter_l2 = nn.Parameter(torch.empty(num_rules, hidden_dim, 1))
        self.filter_b2 = nn.Parameter(torch.zeros(num_rules, 1))
        # Per-rule sign MLP: E → hidden → 1 (logits; sigmoid applied in forward).
        self.sign_l1 = nn.Parameter(torch.empty(num_rules, embed_dim, hidden_dim))
        self.sign_b1 = nn.Parameter(torch.zeros(num_rules, hidden_dim))
        self.sign_l2 = nn.Parameter(torch.empty(num_rules, hidden_dim, 1))
        self.sign_b2 = nn.Parameter(torch.zeros(num_rules, 1))
        for p in (self.filter_l1, self.filter_l2, self.sign_l1, self.sign_l2):
            nn.init.kaiming_uniform_(p, a=5 ** 0.5)

    def _per_atom_filter_sign(self, emb: Tensor, rule_idx_atom: Tensor):
        """Apply per-rule filter+sign MLPs to per-atom embeddings.

        Args:
            emb: ``[..., E]`` per-atom embeddings.
            rule_idx_atom: ``[...]`` matching leading shape; per-atom rule.

        Returns ``(filter_attn, sign_attn)`` each ``[...]`` in [0, 1].

        Computes the filter/sign outputs for ALL rules in a single
        ``einsum`` and gathers the per-position rule's result. This
        replaces a per-position weight gather (``self.l1[rule_idx]``
        materialises ``[..., E, h]`` for every atom — OOMs on eval
        chunks) and a Python loop over rules (incompatible with
        ``torch.compile(fullgraph=True, mode="reduce-overhead")``).
        Compute is ``num_rules``× redundant per atom, memory is
        ``[..., num_rules, h]`` intermediate — strictly smaller than
        the gather pattern when ``E > num_rules`` (the typical regime).
        """
        # all_filter_h1[..., r, h] = emb[..., e] @ filter_l1[r, e, h]
        all_filter_h1 = F.leaky_relu(
            torch.einsum("...e,reh->...rh", emb, self.filter_l1) + self.filter_b1
        )
        # ... → [..., num_rules]
        all_filter = torch.sigmoid(
            (
                torch.einsum("...rh,rhk->...rk", all_filter_h1, self.filter_l2)
                + self.filter_b2
            ).squeeze(-1)
        )
        all_sign_h1 = F.leaky_relu(
            torch.einsum("...e,reh->...rh", emb, self.sign_l1) + self.sign_b1
        )
        all_sign = torch.sigmoid(
            (
                torch.einsum("...rh,rhk->...rk", all_sign_h1, self.sign_l2)
                + self.sign_b2
            ).squeeze(-1)
        )
        # Gather per-atom rule's result.
        gather_idx = rule_idx_atom.unsqueeze(-1)            # [..., 1]
        filter_attn = torch.gather(all_filter, dim=-1, index=gather_idx).squeeze(-1)
        sign_attn = torch.gather(all_sign, dim=-1, index=gather_idx).squeeze(-1)
        return filter_attn, sign_attn

    def forward(self, atom_repr: Repr, evidence: ProofEvidence) -> Repr:
        if not atom_repr.has_embeddings or not atom_repr.has_scores:
            raise ValueError(
                "FilterSignStateRepr requires atom_repr with both embeddings "
                "and scores (use KGEPairAtom or KGEBothAtom upstream)."
            )
        emb = atom_repr.embeddings           # [B, P, D, M, E] or [B, P, M, E]
        sc = atom_repr.scores                # leading shape matches emb without E

        # Per-atom rule index broadcast to atom leading shape. Padded
        # groundings carry ``rule_idx=-1``; clamp before the gather (the
        # downstream ``MaxQueryRepr.mask`` zeroes those slots out so the
        # mis-routed rule's score never leaks into the final output).
        rule_idx = evidence.rule_idx.clamp(min=0)        # [B, P, D] or [B, P]
        rule_atom = rule_idx.unsqueeze(-1).expand(emb.shape[:-1])

        filter_attn, sign_attn = self._per_atom_filter_sign(emb, rule_atom)
        # filter_attn / sign_attn already collapse the trailing scalar dim.
        sign_term = _godel_iff(sign_attn, sc)
        term = torch.maximum(1.0 - filter_attn, sign_term)        # [..., M]

        # T-norm over atoms with the standard empty-conjunction identity (1.0).
        mask = _per_atom_validity_mask(tuple(term.shape), evidence, term.device)
        ones = torch.ones_like(term)
        masked = torch.where(mask, term, ones)
        if self.tnorm == "min":
            reduced = masked.min(dim=-1).values
        else:
            reduced = masked.prod(dim=-1)
        return Repr(scores=reduced)


class ClusteredFilterSignStateRepr(nn.Module):
    """Clustered DCR: Gumbel formula clustering + filter/sign Godel attention.

    Adds a learned per-rule formula clustering layer in front of
    :class:`FilterSignStateRepr`'s per-atom filter+sign MLPs. For each
    grounding, body-atom embeddings are flattened, projected into ``K``
    formula embeddings of size ``H``, and a single formula is selected
    via Gumbel-logistic + straight-through argmax. The selected formula
    embedding is broadcast over atom positions (with a per-rule
    positional embedding added) before filter/sign attention.

    Bit-by-bit reproduces ns's pre-migration ``ClusteredDCRReasoning``
    math, including the exact init pattern (separate per-rule MLPs in
    ``ModuleList`` flavour, gathered here as stacked parameters).

    Inputs (via ``atom_repr``):
      * embeddings ``[B, P, D, M, E]`` or ``[B, P, M, E]``
      * scores ``[..., M]`` — concept "values" for the iff
    Output: ``Repr(scores=...)`` of shape matching the body-leading dim
    minus ``M`` (i.e. ``[B, P, D]`` or ``[B, P]``).
    """

    def __init__(
        self,
        num_rules: int,
        embed_dim: int,
        max_body_atoms: int,
        *,
        num_formulas: int = 4,
        formula_hidden: int = 64,
        temperature: float = 1.0,
        tnorm: Literal["min", "product"] = "min",
    ) -> None:
        super().__init__()
        if num_rules < 1:
            raise ValueError("num_rules must be >= 1")
        self.num_rules = num_rules
        self.embed_dim = embed_dim
        self.max_body_atoms = max_body_atoms
        self.num_formulas = num_formulas
        self.formula_hidden = formula_hidden
        self.temperature = temperature
        self.tnorm = tnorm

        flat_in = max_body_atoms * embed_dim
        formula_emb_dim = num_formulas * formula_hidden
        # Per-rule formula embedder: flat → K*H, two-layer LeakyReLU.
        self.fe_l1 = nn.Parameter(torch.empty(num_rules, flat_in, formula_emb_dim))
        self.fe_b1 = nn.Parameter(torch.zeros(num_rules, formula_emb_dim))
        self.fe_l2 = nn.Parameter(torch.empty(num_rules, formula_emb_dim, formula_emb_dim))
        self.fe_b2 = nn.Parameter(torch.zeros(num_rules, formula_emb_dim))
        # Per-rule formula scorer: H → 1.
        self.fs_w = nn.Parameter(torch.empty(num_rules, formula_hidden, 1))
        self.fs_b = nn.Parameter(torch.zeros(num_rules, 1))
        # Per-rule positional embedding: [num_rules, M, H].
        self.pos_emb = nn.Parameter(torch.randn(num_rules, max_body_atoms, formula_hidden))
        # Filter / sign MLPs over formula embeddings (input dim H, not E).
        self._fs = FilterSignStateRepr(
            num_rules=num_rules, embed_dim=formula_hidden,
            hidden_dim=formula_hidden, tnorm=tnorm,
        )
        for p in (self.fe_l1, self.fe_l2, self.fs_w):
            nn.init.kaiming_uniform_(p, a=5 ** 0.5)

    @staticmethod
    def _gumbel_logistic(logits: Tensor, temperature: float) -> Tensor:
        coolness = 1.0 / temperature
        u = torch.rand_like(logits).clamp(1e-8, 1 - 1e-8)
        logistic_noise = torch.log(u) - torch.log(1 - u)
        return F.softmax(logits * coolness + logistic_noise * coolness, dim=-1)

    def forward(self, atom_repr: Repr, evidence: ProofEvidence) -> Repr:
        if not atom_repr.has_embeddings or not atom_repr.has_scores:
            raise ValueError(
                "ClusteredFilterSignStateRepr requires both embeddings "
                "and scores in atom_repr."
            )
        emb = atom_repr.embeddings           # [..., M, E]
        sc = atom_repr.scores                # [..., M]
        leading = emb.shape[:-2]             # [B, P] or [B, P, D]
        M = emb.shape[-2]
        E = emb.shape[-1]
        K = self.num_formulas
        H = self.formula_hidden

        # Mask invalid atom embeddings (matches ns: zero out pre-flatten).
        atom_mask_full = _per_atom_validity_mask(emb.shape[:-1], evidence, emb.device)
        masked_emb = emb * atom_mask_full.unsqueeze(-1).to(emb.dtype)

        # Per-rule formula embedder + formula scorer + positional embedding,
        # all vectorized via einsum + gather. Computes ALL rules at every
        # grounding in one fused pass, then picks the per-grounding rule's
        # outputs along the rule axis. No Python loops, no per-position
        # weight gather — fullgraph + reduce-overhead torch.compile-safe.
        # Padded groundings carry ``rule_idx=-1``; clamp to 0 before the
        # gather (those slots are masked out by ``MaxQueryRepr.mask``).
        rule_idx = evidence.rule_idx.clamp(min=0)          # [B, P] or [B, P, D]
        flat_emb = masked_emb.reshape(*leading, M * E)     # [..., M*E]
        # Two-layer formula embedder for ALL rules: [..., R, K*H].
        all_h1 = F.leaky_relu(
            torch.einsum("...i,rij->...rj", flat_emb, self.fe_l1) + self.fe_b1
        )                                                                    # [..., R, K*H]
        all_formula = (
            torch.einsum("...rj,rjk->...rk", all_h1, self.fe_l2) + self.fe_b2
        )                                                                    # [..., R, K*H]
        all_formula = all_formula.reshape(*leading, self.num_rules, K, H)    # [..., R, K, H]
        # Per-formula score for ALL rules: [..., R, K, H] @ [R, H, 1] → [..., R, K]
        all_logits = (
            torch.einsum("...rkh,rhj->...rkj", all_formula, self.fs_w)
            + self.fs_b.unsqueeze(-2)
        ).squeeze(-1)
        # Gather per-grounding rule's outputs.
        rule_idx_kh = rule_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
            *rule_idx.shape, 1, K, H,
        )                                                                    # [..., 1, K, H]
        formula_emb = torch.gather(all_formula, dim=-3, index=rule_idx_kh).squeeze(-3)
        rule_idx_k = rule_idx.unsqueeze(-1).unsqueeze(-1).expand(
            *rule_idx.shape, 1, K,
        )                                                                    # [..., 1, K]
        formula_logits = torch.gather(all_logits, dim=-2, index=rule_idx_k).squeeze(-2)
        # Positional embeddings: pos_emb is [num_rules, M, H], keyed by
        # rule index along axis 0; ``self.pos_emb[rule_idx]`` produces
        # the per-grounding [M, H] slice directly without a separate
        # gather call.
        atom_formula_emb_pos = self.pos_emb[rule_idx]                        # [..., M, H]

        if self.temperature > 0.0:
            # ``_gumbel_logistic`` returns soft probs that ns ignores
            # (it picks via argmax of raw logits); calling it still
            # consumes the RNG draw so seed-equivalent runs match.
            _ = self._gumbel_logistic(formula_logits, self.temperature)
        selected = formula_logits.argmax(dim=-1, keepdim=True)        # [..., 1]
        # Gather the selected formula embedding per grounding: [..., H].
        selected_emb = torch.gather(
            formula_emb, dim=-2,
            index=selected.unsqueeze(-1).expand(*selected.shape, H),
        ).squeeze(-2)

        # Add per-rule positional embeddings broadcast across atoms.
        atom_formula_emb = selected_emb.unsqueeze(-2) + atom_formula_emb_pos  # [..., M, H]

        # Hand off to filter/sign with formula-augmented embeddings.
        return self._fs(
            Repr(embeddings=atom_formula_emb, scores=sc),
            evidence,
        )


__all__ = [
    "ClusteredFilterSignStateRepr",
    "ConcatStateRepr",
    "FilterSignStateRepr",
    "GatedTNormStateRepr",
    "MaxStateRepr",
    "MeanStateRepr",
    "PhiPsiStateRepr",
    "RuleWeightedStateRepr",
    "SumStateRepr",
    "TNormStateRepr",
]
