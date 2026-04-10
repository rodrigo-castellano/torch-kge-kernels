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
"""
from __future__ import annotations

from abc import abstractmethod

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

    def forward(self, h: Tensor, r: Tensor, t: Tensor) -> Tensor:  # noqa: D401
        """Default forward returns triple scores."""
        return self.score_triples(h, r, t)


__all__ = ["KGEModel"]
