"""Select implementations: choose next path(s) from successors.

The exhaustive variant is the identity — used by SBR/DCR/R2N where the
grounder produces all evidence in one shot and the canonical loop runs
for a single iteration.

The non-exhaustive variants (greedy/beam/sample) record the chosen
indices in ``SelectInfo`` so a caller-supplied state factory can
construct the narrowed next state. ``tkk`` does NOT own ``ProofState``
as a concrete type (the grounder owns that), so Select implementations
either pass the input state through unchanged (when narrowing is left
to the caller) or use a ``state_factory`` callable provided at init.

All implementations are branchless: no ``.item()``, no Python control
flow on tensor values, ``torch.compile(fullgraph=True)`` safe.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .repr import Repr
from .types import ProofEvidence, ProofState, SelectInfo

StateFactory = Callable[[ProofState, Tensor, Tensor], ProofState]


class ExhaustiveSelect(nn.Module):
    """Identity select: pass the input state through unchanged.

    Used by exhaustive scorers where the grounder produces all evidence
    in a single shot (SBR/DCR/R2N). The canonical loop terminates after
    one iteration when paired with this select.
    """

    def forward(
        self, evidence: ProofEvidence, s_repr: Repr
    ) -> Tuple[ProofState, Optional[SelectInfo]]:
        # The "state" the caller provides is conceptually the same after
        # exhaustive evaluation; the loop terminates immediately.
        return None, None  # type: ignore[return-value]


class GreedySelect(nn.Module):
    """Argmax over per-state scores; returns chosen indices in SelectInfo."""

    def __init__(self, state_factory: Optional[StateFactory] = None) -> None:
        super().__init__()
        self.state_factory = state_factory

    def forward(
        self, evidence: ProofEvidence, s_repr: Repr
    ) -> Tuple[ProofState, Optional[SelectInfo]]:
        if not s_repr.has_scores:
            raise ValueError("GreedySelect requires s_repr.scores")
        scores = s_repr.scores                          # [B, S]
        chosen = scores.argmax(dim=-1, keepdim=True)    # [B, 1]
        chosen_scores = torch.gather(scores, dim=-1, index=chosen)
        info = SelectInfo(
            chosen_indices=chosen,
            chosen_scores=chosen_scores,
            log_probs=None,
        )
        next_state = self.state_factory(None, chosen, chosen_scores) if self.state_factory else None  # type: ignore[arg-type]
        return next_state, info


class BeamSelect(nn.Module):
    """Top-k over per-state scores."""

    def __init__(self, k: int, state_factory: Optional[StateFactory] = None) -> None:
        super().__init__()
        self.k = k
        self.state_factory = state_factory

    def forward(
        self, evidence: ProofEvidence, s_repr: Repr
    ) -> Tuple[ProofState, Optional[SelectInfo]]:
        if not s_repr.has_scores:
            raise ValueError("BeamSelect requires s_repr.scores")
        scores = s_repr.scores
        k = min(self.k, scores.shape[-1])
        topk = torch.topk(scores, k=k, dim=-1)
        info = SelectInfo(
            chosen_indices=topk.indices,
            chosen_scores=topk.values,
            log_probs=None,
        )
        next_state = (
            self.state_factory(None, topk.indices, topk.values)  # type: ignore[arg-type]
            if self.state_factory
            else None
        )
        return next_state, info


class SampleSelect(nn.Module):
    """Categorical sampling over per-state logits.

    ``s_repr.scores`` is treated as logits (softmax inside). ``log_probs``
    of the chosen actions is recorded in ``SelectInfo`` for policy gradient.
    """

    def __init__(
        self,
        n: int = 1,
        state_factory: Optional[StateFactory] = None,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__()
        self.n = n
        self.state_factory = state_factory
        self.generator = generator

    def forward(
        self, evidence: ProofEvidence, s_repr: Repr
    ) -> Tuple[ProofState, Optional[SelectInfo]]:
        if not s_repr.has_scores:
            raise ValueError("SampleSelect requires s_repr.scores")
        logits = s_repr.scores                             # [B, S]
        probs = torch.softmax(logits, dim=-1)
        # multinomial supports replacement and a generator; this stays
        # branchless and does not call .item() / .tolist().
        chosen = torch.multinomial(
            probs.reshape(-1, probs.shape[-1]),
            num_samples=self.n,
            replacement=True,
            generator=self.generator,
        ).reshape(*probs.shape[:-1], self.n)
        log_p = torch.log(probs.clamp(min=1e-12))
        chosen_log_p = torch.gather(log_p, dim=-1, index=chosen)
        chosen_scores = torch.gather(logits, dim=-1, index=chosen)
        info = SelectInfo(
            chosen_indices=chosen,
            chosen_scores=chosen_scores,
            log_probs=chosen_log_p,
        )
        next_state = (
            self.state_factory(None, chosen, chosen_scores)  # type: ignore[arg-type]
            if self.state_factory
            else None
        )
        return next_state, info


__all__ = [
    "BeamSelect",
    "ExhaustiveSelect",
    "GreedySelect",
    "SampleSelect",
    "StateFactory",
]
