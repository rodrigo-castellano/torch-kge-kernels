"""Abstract base class for KGE models in tkk.

A ``KGEModel`` is a torch ``nn.Module`` that knows how to:
  - score specific triples       → ``score_triples(h, r, t) -> [N]``
  - score all tails for ``(h, r)`` → ``score_all_tails(h, r) -> [N, E]``
  - score all heads for ``(r, t)`` → ``score_all_heads(r, t) -> [N, E]``
  - compose a fused embedding    → ``compose(h, r, t) -> [N, E]``

The first three methods plug directly into ``KGEBackend`` (the existing
``kge_kernels.scoring`` protocol). The ``compose`` method is what
``KGEEmbedAtom`` consumes — it's the fused per-atom embedding (TransE's
``h+r-t``, ComplEx's bilinear form, etc.) that used to be duplicated in
``DpRL.kge_experiments.nn.atom_embedders`` and ``torch-ns.ns_lib.nn.kge_layers``.

``score(h, r, t=None)`` is a convenience dispatch provided by the base:
``t is None`` → all tails, ``h is None`` → all heads, otherwise specific
triples. This matches the call convention used by DpRL's inference and
evaluation paths.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Optional

from torch import Tensor, nn


class KGEModel(nn.Module):
    """Base class for KGE models with the four required scoring methods."""

    num_entities: int
    num_relations: int
    dim: int

    @abstractmethod
    def score_triples(self, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
        """Score a batch of fully ground triples → ``[N]``."""

    @abstractmethod
    def score_all_tails(self, h: Tensor, r: Tensor) -> Tensor:
        """Score all entities as tail for each ``(h, r)`` pair → ``[N, E]``."""

    @abstractmethod
    def score_all_heads(self, r: Tensor, t: Tensor) -> Tensor:
        """Score all entities as head for each ``(r, t)`` pair → ``[N, E]``."""

    @abstractmethod
    def compose(self, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
        """Fused per-atom embedding ``[N, E]`` consumed by KGEEmbedAtom."""

    def score(
        self,
        h: Optional[Tensor],
        r: Tensor,
        t: Optional[Tensor] = None,
    ) -> Tensor:
        """Unified dispatch: specific triples, all tails, or all heads.

        - ``h`` and ``t`` both non-None → ``score_triples`` → ``[N]``
        - ``t is None`` and ``h`` non-None → ``score_all_tails`` → ``[N, E]``
        - ``h is None`` and ``t`` non-None → ``score_all_heads`` → ``[N, E]``
        """
        if h is not None and t is not None:
            return self.score_triples(h, r, t)
        if t is None:
            if h is None:
                raise ValueError("score() requires at least one of h or t to be non-None")
            return self.score_all_tails(h, r)
        return self.score_all_heads(r, t)

    def forward(
        self,
        h: Optional[Tensor],
        r: Tensor,
        t: Optional[Tensor] = None,
    ) -> Tensor:  # noqa: D401
        """Default forward delegates to ``score`` for nn.Module compatibility."""
        return self.score(h, r, t)


__all__ = ["KGEModel"]
