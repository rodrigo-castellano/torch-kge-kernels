"""RotatE: complex rotation, score = gamma - || (h . r) - t ||_p.

Embeddings are stored as single interleaved ``[re | im]`` tensors
(matching torch-ns's proven layout) so that Adam's per-parameter
adaptive statistics cover the full complex vector.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .base import KGEModel


class RotatE(KGEModel):
    """RotatE knowledge graph embedding (complex rotation)."""

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dim: int,
        gamma: float = 12.0,
        p_norm: int = 1,
    ) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RotatE requires even dim (real + imaginary halves)")
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.half_dim = dim // 2
        self.gamma = nn.Parameter(
            torch.tensor(gamma, dtype=torch.get_default_dtype()), requires_grad=False
        )
        self.p = p_norm
        self.entity_embeddings = nn.Embedding(num_entities, dim)
        self.rel_phase = nn.Embedding(num_relations, self.half_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 6 / math.sqrt(self.half_dim)
        nn.init.uniform_(self.entity_embeddings.weight, -bound, bound)
        nn.init.uniform_(self.rel_phase.weight, -math.pi, math.pi)
        self._clamp_entity_modulus()

    @torch.no_grad()
    def _clamp_entity_modulus(self) -> None:
        """Clamp entity embeddings to the unit disk in complex space.

        Called once at initialisation to match torch-ns's init.
        """
        w = self.entity_embeddings.weight.data
        re, im = w[:, :self.half_dim], w[:, self.half_dim:]
        mod = torch.clamp(torch.sqrt(re * re + im * im), min=1e-6)
        factor = torch.clamp(1.0 / mod, max=1.0)
        re.mul_(factor)
        im.mul_(factor)

    def _split_ent(self, idx: Tensor):
        emb = self.entity_embeddings(idx)
        return emb[..., :self.half_dim], emb[..., self.half_dim:]

    def _hr(self, h: Tensor, r: Tensor):
        h_re, h_im = self._split_ent(h)
        phase = torch.remainder(self.rel_phase(r), 2 * math.pi)
        c, s = torch.cos(phase), torch.sin(phase)
        return h_re * c - h_im * s, h_re * s + h_im * c

    def _dist(self, a_re: Tensor, a_im: Tensor, b_re: Tensor, b_im: Tensor) -> Tensor:
        if self.p == 1:
            return ((a_re - b_re).abs() + (a_im - b_im).abs()).sum(dim=-1)
        return torch.sqrt(((a_re - b_re) ** 2 + (a_im - b_im) ** 2) + 1e-9).sum(dim=-1)

    def score_triples(self, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
        hr_re, hr_im = self._hr(h, r)
        t_re, t_im = self._split_ent(t)
        return self.gamma - self._dist(hr_re, hr_im, t_re, t_im)

    def score_all_tails(self, h: Tensor, r: Tensor) -> Tensor:
        hr_re, hr_im = self._hr(h, r)
        all_re = self.entity_embeddings.weight[:, :self.half_dim]
        all_im = self.entity_embeddings.weight[:, self.half_dim:]
        return self.gamma - self._dist(
            hr_re.unsqueeze(1), hr_im.unsqueeze(1),
            all_re.unsqueeze(0), all_im.unsqueeze(0),
        )

    def score_all_heads(self, r: Tensor, t: Tensor) -> Tensor:
        t_re, t_im = self._split_ent(t)
        phase = torch.remainder(self.rel_phase(r), 2 * math.pi)
        c, s = torch.cos(phase).unsqueeze(1), torch.sin(phase).unsqueeze(1)
        all_re = self.entity_embeddings.weight[:, :self.half_dim].unsqueeze(0)
        all_im = self.entity_embeddings.weight[:, self.half_dim:].unsqueeze(0)
        all_hr_re = all_re * c - all_im * s
        all_hr_im = all_re * s + all_im * c
        return self.gamma - self._dist(
            all_hr_re, all_hr_im, t_re.unsqueeze(1), t_im.unsqueeze(1)
        )

    def compose(self, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
        """Fused RotatE feature: concatenated (rotated h - t) real/imag."""
        hr_re, hr_im = self._hr(h, r)
        t_re, t_im = self._split_ent(t)
        diff_re = hr_re - t_re
        diff_im = hr_im - t_im
        return torch.cat([diff_re, diff_im], dim=-1)


__all__ = ["RotatE"]
