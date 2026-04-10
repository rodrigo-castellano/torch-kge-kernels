"""Shared test fixtures: a tiny FakeProofEvidence dataclass.

This dataclass implements the ``ProofEvidence`` Protocol via duck typing
without depending on grounder. Tests build it with hand-rolled tensors so
the framework primitives are exercised end-to-end without requiring the
actual resolution operator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass
class FakeProofEvidence:
    """Concrete dataclass that satisfies the ProofEvidence Protocol.

    Mirrors ``grounder.types.ProofEvidence`` field-for-field. Tests build
    these directly to avoid pulling grounder into tkk's test suite.
    """

    body: Tensor
    mask: Tensor
    count: Tensor
    rule_idx: Tensor
    body_count: Tensor
    D: int = 0
    M: int = 0
    head: Optional[Tensor] = None

    @property
    def body_flat(self) -> Tensor:
        if self.body.dim() == 5:
            B, C, D, M, _ = self.body.shape
            return self.body.reshape(B, C, D * M, 3)
        return self.body

    @property
    def rule_idx_top(self) -> Tensor:
        if self.rule_idx.dim() == 3:
            return self.rule_idx[:, :, 0]
        return self.rule_idx

    @property
    def body_count_total(self) -> Tensor:
        if self.body_count.dim() == 3:
            return self.body_count.sum(dim=-1)
        return self.body_count

    @property
    def body_atom_mask_flat(self) -> Tensor:
        if self.body_count.dim() == 3:
            B, C, D = self.body_count.shape
            M = self.body.shape[3] if self.body.dim() == 5 else 1
            m_idx = torch.arange(M, device=self.body_count.device)
            per_depth = m_idx < self.body_count.unsqueeze(-1)
            return per_depth.reshape(B, C, D * M)
        G = self.body.shape[2]
        idx = torch.arange(G, device=self.body_count.device)
        return idx < self.body_count.unsqueeze(-1)


def make_structured_evidence(
    B: int = 2,
    C: int = 3,
    D: int = 2,
    M: int = 2,
    num_preds: int = 5,
    num_ents: int = 7,
    seed: int = 0,
) -> FakeProofEvidence:
    """Build a small structured-layout evidence object with random indices."""
    g = torch.Generator().manual_seed(seed)
    body = torch.stack(
        [
            torch.randint(num_preds, (B, C, D, M), generator=g),
            torch.randint(num_ents, (B, C, D, M), generator=g),
            torch.randint(num_ents, (B, C, D, M), generator=g),
        ],
        dim=-1,
    )
    mask = torch.ones(B, C, dtype=torch.bool)
    count = torch.full((B,), C, dtype=torch.long)
    rule_idx = torch.randint(2, (B, C, D), generator=g)
    body_count = torch.full((B, C, D), M, dtype=torch.long)
    return FakeProofEvidence(
        body=body, mask=mask, count=count, rule_idx=rule_idx,
        body_count=body_count, D=D, M=M,
    )


def make_legacy_evidence(
    B: int = 2,
    C: int = 3,
    G_body: int = 4,
    num_preds: int = 5,
    num_ents: int = 7,
    seed: int = 0,
) -> FakeProofEvidence:
    """Build a small legacy-flat-layout evidence object."""
    g = torch.Generator().manual_seed(seed)
    body = torch.stack(
        [
            torch.randint(num_preds, (B, C, G_body), generator=g),
            torch.randint(num_ents, (B, C, G_body), generator=g),
            torch.randint(num_ents, (B, C, G_body), generator=g),
        ],
        dim=-1,
    )
    mask = torch.ones(B, C, dtype=torch.bool)
    count = torch.full((B,), C, dtype=torch.long)
    rule_idx = torch.zeros(B, C, dtype=torch.long)
    body_count = torch.full((B, C), G_body, dtype=torch.long)
    return FakeProofEvidence(
        body=body, mask=mask, count=count, rule_idx=rule_idx,
        body_count=body_count, D=0, M=0,
    )
