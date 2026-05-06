"""Per-firing rule operators for the rule-based reasoning path.

These primitives are the per-firing flavor of state aggregation: instead
of operating on per-grounding-tree ``[B, P, D, M, ...]`` layouts (the
proof path's :mod:`repr_state`), they operate on flat per-firing
``[N_f, M, ...]`` layouts produced by the pool-iter loop.

Three implementations:

* :class:`MinRuleState` — SBR T-norm-min over body atom scores. Scalar
  pool ``[N_pool]`` → scalar prediction ``[N_f, 1]``.
* :class:`FilterSignRuleState` — DCR Gödel filter+sign aggregation. Scalar
  pool ``[N_pool]`` → scalar prediction ``[N_f, 1]``.
* :class:`RuleMLPState` — R2N per-rule MLP over body atom embeddings.
  Embedding pool ``[N_pool, E]`` → embedding prediction
  ``[N_f, K_out, E]`` (``K_out=1`` for ``'head'``, ``M+1`` for ``'full'``).

All conform to the :class:`RuleStateRepr` Protocol — they take
``(body_emb, rule_idx)`` and return ``[N_f, K_out, E]``.

Per-rule parameters (filter/sign for DCR, MLP weights for R2N) are stored
as grouped tensors ``[num_rules, ...]`` and gathered by ``rule_idx`` via
vectorized einsum (no Python loop, ``torch.compile``-safe).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MinRuleState(nn.Module):
    """SBR per-firing min-over-body operator.

    Reads ``body_emb [N_f, M]`` (body atom scores) and ``body_atom_valid
    [N_f, M]`` (passed via the closure-captured firings) — but since this
    primitive sees only ``body_emb``, the caller masks invalid body slots
    by writing them to ``+1`` (T-norm-min identity) before passing them
    in. The pool-iter loop's gather automatically does this when
    ``body_atom_valid=False``.

    Returns ``[N_f, 1]`` scalar head score per firing — shape matches
    ``RuleStateRepr`` contract for ``K_out=1`` over an empty ``E`` dim.

    rule_idx is unused (T-norm is rule-agnostic) but kept in the
    signature for Protocol conformance.
    """

    def forward(self, body_emb: Tensor, rule_idx: Tensor) -> Tensor:
        # body_emb: [N_f, M] — body atom scores after invalid slots masked to +1
        head_score = body_emb.min(dim=-1).values         # [N_f]
        return head_score.unsqueeze(-1)                  # [N_f, 1]


class FilterSignRuleState(nn.Module):
    """DCR Gödel filter+sign aggregation.

    Each rule has ``M`` filter weights ``φ_r ∈ [0, 1]`` and ``M`` sign
    weights ``σ_r ∈ {-1, +1}`` (continuous via tanh). For each firing of
    rule ``r``::

        head_score = min_m( φ_{r,m} * gate(σ_{r,m}, body_score_m) )

    where ``gate(+1, s) = s`` and ``gate(-1, s) = 1 - s``. Filter close
    to 0 means the body atom is irrelevant (term ≈ 1, doesn't constrain
    the min). Sign flips the polarity.

    Per-rule parameters stored as ``[R, M]`` and gathered by ``rule_idx``.
    """

    def __init__(self, num_rules: int, M: int) -> None:
        super().__init__()
        if num_rules < 1:
            raise ValueError("num_rules must be >= 1")
        if M < 1:
            raise ValueError("M must be >= 1")
        self.num_rules = num_rules
        self.M = M
        # Init filter ~0.5 (mild contribution) and sign ~+1 (positive).
        self.filter_logits = nn.Parameter(torch.zeros(num_rules, M))      # sigmoid → 0.5
        self.sign_logits = nn.Parameter(torch.full((num_rules, M), 2.0))  # tanh → ~+0.96

    def forward(self, body_emb: Tensor, rule_idx: Tensor) -> Tensor:
        # body_emb: [N_f, M] body atom scores
        # rule_idx: [N_f]    per-firing rule
        rule_idx_clamped = rule_idx.clamp(min=0)
        phi = torch.sigmoid(self.filter_logits)[rule_idx_clamped]      # [N_f, M]
        sigma = torch.tanh(self.sign_logits)[rule_idx_clamped]         # [N_f, M], in [-1, 1]
        # Polarity gate: for sigma=+1 use s, for sigma=-1 use 1-s; linear
        # interpolation keeps the operator differentiable through sign.
        # gate = (1 + σ)/2 * s + (1 - σ)/2 * (1 - s)
        s_pos = body_emb
        s_neg = 1.0 - body_emb
        gated = 0.5 * (1.0 + sigma) * s_pos + 0.5 * (1.0 - sigma) * s_neg  # [N_f, M]
        # Filter weighting + min (T-norm).
        # When φ → 0: phi * gated → 0; we need 1 (identity) instead so
        # min ignores it. Use linear blend: phi * gated + (1 - phi) * 1.
        weighted = phi * gated + (1.0 - phi) * 1.0                     # [N_f, M]
        head_score = weighted.min(dim=-1).values                       # [N_f]
        return head_score.unsqueeze(-1)                                # [N_f, 1]


class RuleMLPState(nn.Module):
    """R2N per-firing rule MLP (replaces torch_ns/reasoning/state_repr.RuleMLPStateRepr).

    For each firing with body atom embeddings ``body_emb [N_f, M*E]``
    and rule index ``rule_idx [N_f]``, applies the firing's per-rule
    two-layer MLP::

        head_emb = MLP_r(body_emb)

    Per-rule MLPs are stored as grouped parameters ``[num_rules, in_dim,
    hidden]`` / ``[num_rules, hidden, out_total]`` where
    ``out_total = num_atoms_out * atom_emb_dim``. Gathered by ``rule_idx``
    via vectorized einsum + gather (no Python loop).

    ``num_atoms_out`` selects the prediction layout:

    * ``'head'`` (``num_atoms_out=1``) — predict only the head atom's
      embedding. Pair with ``RulePoolLoop(prediction_type='head')``.
    * ``'full'`` (``num_atoms_out=M+1``) — predict embeddings for the
      head AND every body atom of the rule. Pair with
      ``RulePoolLoop(prediction_type='full')``. Matches keras-ns's R2N
      ``prediction_type='full'`` resnet semantics.
    """

    def __init__(
        self,
        num_rules: int,
        in_dim: int,
        atom_emb_dim: int,
        num_atoms_out: int,
        *,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if num_rules < 1:
            raise ValueError("num_rules must be >= 1")
        if num_atoms_out < 1:
            raise ValueError("num_atoms_out must be >= 1")
        out_total = num_atoms_out * atom_emb_dim
        h = hidden_dim or atom_emb_dim
        self.num_rules = num_rules
        self.in_dim = in_dim
        self.atom_emb_dim = atom_emb_dim
        self.num_atoms_out = num_atoms_out
        self.l1 = nn.Parameter(torch.empty(num_rules, in_dim, h))
        self.b1 = nn.Parameter(torch.zeros(num_rules, h))
        self.l2 = nn.Parameter(torch.empty(num_rules, h, out_total))
        self.b2 = nn.Parameter(torch.zeros(num_rules, out_total))
        nn.init.kaiming_uniform_(self.l1, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.l2, a=5 ** 0.5)

    def forward(self, body_emb: Tensor, rule_idx: Tensor) -> Tensor:
        out_total = self.num_atoms_out * self.atom_emb_dim
        all_h1 = F.relu(
            torch.einsum("ne,reh->nrh", body_emb, self.l1) + self.b1
        )                                                       # [N_f, R, h]
        all_e = (
            torch.einsum("nrh,rho->nro", all_h1, self.l2) + self.b2
        )                                                       # [N_f, R, out_total]
        gather_idx = rule_idx.clamp(min=0).unsqueeze(-1).unsqueeze(-1).expand(
            -1, 1, out_total,
        )                                                       # [N_f, 1, out_total]
        e_flat = torch.gather(all_e, dim=-2, index=gather_idx).squeeze(-2)
        return e_flat.reshape(-1, self.num_atoms_out, self.atom_emb_dim)


__all__ = ["FilterSignRuleState", "MinRuleState", "RuleMLPState"]
