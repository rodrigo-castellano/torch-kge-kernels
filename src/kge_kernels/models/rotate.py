"""RotatE: complex rotation, score = gamma - || (h ∘ r) - t ||_p."""
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
        self.ent_re = nn.Embedding(num_entities, self.half_dim)
        self.ent_im = nn.Embedding(num_entities, self.half_dim)
        self.rel_phase = nn.Embedding(num_relations, self.half_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 6 / math.sqrt(self.half_dim)
        nn.init.uniform_(self.ent_re.weight, -bound, bound)
        nn.init.uniform_(self.ent_im.weight, -bound, bound)
        nn.init.uniform_(self.rel_phase.weight, -math.pi, math.pi)
        self.project_entity_modulus_()

    @torch.no_grad()
    def project_entity_modulus_(self) -> None:
        """Clamp entity embeddings to the unit disk in complex space.

        Called once at initialisation and optionally after each optimiser
        step by ``train_kge`` (which checks ``hasattr(model, 'project_entity_modulus_')``).
        """
        re = self.ent_re.weight.data
        im = self.ent_im.weight.data
        mod = torch.clamp(torch.sqrt(re * re + im * im), min=1e-6)
        factor = torch.clamp(1.0 / mod, max=1.0)
        self.ent_re.weight.data = re * factor
        self.ent_im.weight.data = im * factor

    def _hr(self, h: Tensor, r: Tensor):
        h_re = self.ent_re(h)
        h_im = self.ent_im(h)
        phase = torch.remainder(self.rel_phase(r), 2 * math.pi)
        c, s = torch.cos(phase), torch.sin(phase)
        return h_re * c - h_im * s, h_re * s + h_im * c

    def _dist(self, a_re: Tensor, a_im: Tensor, b_re: Tensor, b_im: Tensor) -> Tensor:
        if self.p == 1:
            return ((a_re - b_re).abs() + (a_im - b_im).abs()).sum(dim=-1)
        return torch.sqrt(((a_re - b_re) ** 2 + (a_im - b_im) ** 2) + 1e-9).sum(dim=-1)

    def score_triples(self, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
        hr_re, hr_im = self._hr(h, r)
        return self.gamma - self._dist(hr_re, hr_im, self.ent_re(t), self.ent_im(t))

    def score_all_tails(self, h: Tensor, r: Tensor) -> Tensor:
        hr_re, hr_im = self._hr(h, r)
        return self.gamma - self._dist(
            hr_re.unsqueeze(1),
            hr_im.unsqueeze(1),
            self.ent_re.weight.unsqueeze(0),
            self.ent_im.weight.unsqueeze(0),
        )

    def score_all_heads(self, r: Tensor, t: Tensor) -> Tensor:
        t_re = self.ent_re(t)
        t_im = self.ent_im(t)
        phase = torch.remainder(self.rel_phase(r), 2 * math.pi)
        c, s = torch.cos(phase).unsqueeze(1), torch.sin(phase).unsqueeze(1)
        ent_re_w = self.ent_re.weight.unsqueeze(0)
        ent_im_w = self.ent_im.weight.unsqueeze(0)
        all_hr_re = ent_re_w * c - ent_im_w * s
        all_hr_im = ent_re_w * s + ent_im_w * c
        return self.gamma - self._dist(
            all_hr_re, all_hr_im, t_re.unsqueeze(1), t_im.unsqueeze(1)
        )

    def compose(self, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
        """Fused RotatE feature: concatenated (rotated h - t) real/imag."""
        hr_re, hr_im = self._hr(h, r)
        diff_re = hr_re - self.ent_re(t)
        diff_im = hr_im - self.ent_im(t)
        return torch.cat([diff_re, diff_im], dim=-1)


__all__ = ["RotatE"]
