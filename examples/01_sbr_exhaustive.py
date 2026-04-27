"""Example 1 — SBR-style exhaustive scoring via framework primitives.

Reproduces the canonical SBR baseline from ``framework.tex`` §14
(SBR-Greedy / SBR-Exhaustive) by composing:

    atom_repr   = KGEScoreAtom          (σ(KGE(h, r, t)) per atom)
    state_repr  = TNormStateRepr("min") (Gödel conjunction over body atoms)
    traj_repr   = TNormTrajRepr("min")  (min across depths — identity at D=1)
    query_repr  = MaxQueryRepr          (Gödel disjunction over proofs)
    select      = ExhaustiveSelect      (one-shot; max_depth=1)

The `resolve` operator is mocked here with a hand-built `ProofEvidence`
dataclass so the example is self-contained. In real usage the grounder
(``grounder.bc.grounder``) produces a compatible ``ProofEvidence``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from kge_kernels.framework import (
    ExhaustiveSelect,
    KGEScoreAtom,
    MaxQueryRepr,
    TNormStateRepr,
    TNormTrajRepr,
)
from kge_kernels.models import TransE
from kge_kernels.search import ProofScorer, SearchSpec


@dataclass
class ToyEvidence:
    """Minimal ``ProofEvidence``-satisfying dataclass for this example."""

    body: Tensor           # [B, P, D, M, 3]
    mask: Tensor           # [B, P]
    count: Tensor          # [B]
    rule_idx: Tensor       # [B, P, D]
    body_count: Tensor     # [B, P, D]
    D: int = 0
    M: int = 0
    head: Optional[Tensor] = None

    @property
    def body_flat(self) -> Tensor:
        B, P, D, M, _ = self.body.shape
        return self.body.reshape(B, P, D * M, 3)

    @property
    def rule_idx_top(self) -> Tensor:
        return self.rule_idx[:, :, 0]

    @property
    def body_count_total(self) -> Tensor:
        return self.body_count.sum(dim=-1)

    @property
    def body_atom_mask_flat(self) -> Tensor:
        B, P, D = self.body_count.shape
        M = self.body.shape[3]
        m_idx = torch.arange(M, device=self.body_count.device)
        per_depth = m_idx < self.body_count.unsqueeze(-1)
        return per_depth.reshape(B, P, D * M)


def main() -> None:
    torch.manual_seed(0)

    num_entities = 10
    num_relations = 3

    # Pretrained KGE model — here we just initialize one for demonstration.
    model = TransE(
        num_entities=num_entities, num_relations=num_relations, dim=8
    )

    # Hand-build a toy evidence object representing 2 queries, each with
    # 3 candidate proofs, each proof of depth 1 with 2 body atoms.
    B, P, D, M = 2, 3, 1, 2
    body = torch.stack(
        [
            torch.randint(num_relations, (B, P, D, M)),
            torch.randint(num_entities, (B, P, D, M)),
            torch.randint(num_entities, (B, P, D, M)),
        ],
        dim=-1,
    )
    ev = ToyEvidence(
        body=body,
        mask=torch.ones(B, P, dtype=torch.bool),
        count=torch.full((B,), P, dtype=torch.long),
        rule_idx=torch.zeros(B, P, D, dtype=torch.long),
        body_count=torch.full((B, P, D), M, dtype=torch.long),
        D=D,
        M=M,
    )

    def resolve(_state):
        # In real usage this would call the grounder.
        return ev

    scorer = ProofScorer(
        resolve=resolve,
        atom_repr=KGEScoreAtom(),
        state_repr=TNormStateRepr("min"),
        traj_repr=TNormTrajRepr("min"),
        query_repr=MaxQueryRepr(),
        select=ExhaustiveSelect(),
        model=model,
        spec=SearchSpec(batch_size=B, max_depth=1),
        capture="dynamic",
        name="sbr_exhaustive",
    )
    scores = scorer(torch.zeros(B, 3, dtype=torch.long))["sbr_exhaustive"]

    print("SBR-exhaustive scores:", scores.tolist())
    print("  shape:", list(scores.shape))
    print("  in [0, 1]:", bool(((scores >= 0) & (scores <= 1)).all()))


if __name__ == "__main__":
    main()
