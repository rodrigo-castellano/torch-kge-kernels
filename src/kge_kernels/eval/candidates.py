"""Candidate sources for ranking evaluation.

A :class:`CandidateSource` is a protocol: anything that, given a query
batch and a mode, returns the per-query candidate entity pool plus a
validity mask. The evaluator does not care how candidates are picked —
sampler-based, exhaustive, fixed pre-defined pools — only that the
shape stays static (``K_fixed`` is invariant for the source's lifetime
so downstream buffers can be preallocated once).

:class:`SamplerCandidates` is the standard concrete implementation,
wrapping a tkk :class:`~kge_kernels.scoring.Sampler`. The sampler owns
filter and (entity-level) domain knowledge; this class adds:

- ``K_fixed`` sizing for the exhaustive-vs-sampled split.
- Padding to ``K_fixed`` with a sentinel + valid mask.
- Per-relation domain restriction via ``head_domain`` / ``tail_domain``
  dicts (``{rel: {entity_ids}}``) — the standard countries / ablation
  eval protocol.
"""
from __future__ import annotations

from typing import Dict, Literal, Optional, Protocol

import torch
from torch import Tensor

Mode = Literal["head", "tail"]


class CandidateSource(Protocol):
    """Generates fixed-shape candidate pools per query batch.

    The evaluator allocates static buffers off ``K_fixed``; once an
    instance is constructed, ``K_fixed`` MUST NOT change.
    """

    K_fixed: int

    def candidates(self, queries: Tensor, mode: Mode) -> tuple[Tensor, Tensor]:
        """Return ``(cand_entities [B, K_fixed], valid [B, K_fixed])``.

        ``cand_entities`` may pad invalid slots with any in-range index;
        ``valid`` is ``False`` there.
        """
        ...


class SamplerCandidates:
    """Sampler-backed candidate source — the standard implementation.

    Two modalities (set via ``k``):

    - ``k=None``: exhaustive over the valid pool for each query's
      relation. ``K_fixed = max_pool_len - 1`` if the sampler has a
      domain (else ``num_entities - 1``) — i.e. "every entity in the
      domain, minus the true entity slot". Rows with shorter valid
      pools get ``valid=False`` on the tail.
    - ``k=N`` (int): sample N entities per query; ``K_fixed = N``.
      ``valid`` is ``False`` where the sampler couldn't find enough
      filtered-unique candidates in the available pool.

    The sampler's ``filter=True`` removes known positives (train ∪
    valid ∪ test facts installed at sampler construction) from the
    candidate pool. The true entity for each query is therefore NOT
    in the returned candidates — the evaluator scores it separately.

    Per-relation domain restriction: pass ``head_domain`` /
    ``tail_domain`` (``{relation_id: {valid_entity_ids}}``) to mask out
    candidates that aren't in the relation's domain. This is the
    standard countries / ablation eval protocol; applied as a post-mask
    on ``valid`` after sampling.
    """

    def __init__(
        self,
        sampler,
        *,
        k: Optional[int] = None,
        head_domain: Optional[Dict[int, set]] = None,
        tail_domain: Optional[Dict[int, set]] = None,
        unique: bool = False,
    ) -> None:
        """``unique``: forward to ``sampler.corrupt(unique=...)``. The
        sampler's pairwise-dedup is ``O(B * k^2)`` GPU memory; for
        exhaustive eval (k = num_entities, ~40k on wn18rr) that's
        50+ GiB and OOMs. Exhaustive enumeration produces each entity
        at most once by construction, so dedup is redundant. Default
        ``False`` is safe for any exhaustive / unique-by-construction
        caller. DpRL's rollout tests need dedup of sampled draws and
        pass ``unique=True`` explicitly.
        """
        self.sampler = sampler
        self.num_entities = sampler.num_entities
        self.k = k
        self._unique = unique

        has_domain = getattr(sampler, "_has_domain_info", lambda: False)()
        if k is None:
            pool_size = (
                (getattr(sampler, "max_pool_len", self.num_entities) - 1)
                if has_domain
                else (self.num_entities - 1)
            )
            self.K_fixed = int(pool_size)
            self._is_exhaustive = True
        else:
            self.K_fixed = int(k)
            self._is_exhaustive = False

        # Per-relation domain masks: [num_relations, num_entities] booleans.
        # Built once at construction so the per-call lookup is a single
        # gather. Eager — outside the evaluator's compile region.
        self._head_domain_mask: Optional[Tensor] = None
        self._tail_domain_mask: Optional[Tensor] = None
        if head_domain is not None or tail_domain is not None:
            num_relations = sampler.num_relations
            device = sampler.device
            if head_domain is not None:
                self._head_domain_mask = self._build_domain_mask(
                    head_domain, num_relations, self.num_entities, device
                )
            if tail_domain is not None:
                self._tail_domain_mask = self._build_domain_mask(
                    tail_domain, num_relations, self.num_entities, device
                )

    @staticmethod
    def _build_domain_mask(
        domain_dict: Dict[int, set],
        num_relations: int,
        num_entities: int,
        device: torch.device,
    ) -> Tensor:
        """Convert ``{rel: {ents}}`` dict to ``[R, E]`` boolean mask."""
        mask = torch.zeros(num_relations, num_entities, dtype=torch.bool, device=device)
        for r, ents in domain_dict.items():
            if not ents:
                continue
            ent_t = torch.tensor(sorted(ents), dtype=torch.long, device=device)
            mask[int(r), ent_t] = True
        return mask

    def candidates(
        self,
        queries: Tensor,   # [B, 3] int64, (r, h, t)
        mode: Mode,
    ) -> tuple[Tensor, Tensor]:
        # Sampler draws `num_negatives=None` for exhaustive, int for sampled.
        num_neg = None if self._is_exhaustive else self.K_fixed
        neg, mask = self.sampler.corrupt(
            queries,
            num_negatives=num_neg,
            mode=mode,
            filter=True,
            unique=self._unique,
            return_mask=True,
        )
        # neg: [B, K, 3] in (r, h, t) format; extract the column that was
        # actually corrupted — that's our candidate entity for each slot.
        col = 1 if mode == "head" else 2
        cand_entities = neg[:, :, col]   # [B, K]

        # Sampler marks shortfalls with the sentinel ``num_entities``
        # (out-of-range). Clamp to 0 so downstream embedding lookups
        # never fault; the validity mask still gates contribution.
        cand_entities = cand_entities.clamp_(0, self.num_entities - 1)

        # Pad to K_fixed so downstream compile shapes stay static.
        B, K = cand_entities.shape
        if K < self.K_fixed:
            pad = self.K_fixed - K
            cand_entities = torch.cat(
                [cand_entities, torch.zeros(B, pad, dtype=cand_entities.dtype, device=cand_entities.device)],
                dim=1,
            )
            mask = torch.cat(
                [mask, torch.zeros(B, pad, dtype=torch.bool, device=mask.device)],
                dim=1,
            )
        elif K > self.K_fixed:
            raise RuntimeError(
                f"Sampler returned K={K} candidates but source expected "
                f"K_fixed={self.K_fixed}. The sampler's domain / pool size "
                f"is misconfigured — fix the upstream sampler instead of "
                f"clipping."
            )

        # Per-relation domain restriction (e.g. countries: head of relation
        # r must be in head_domain[r]). Applied after sampling so we mask
        # out-of-domain candidates regardless of sampler's domain config.
        domain_mask = self._head_domain_mask if mode == "head" else self._tail_domain_mask
        if domain_mask is not None:
            rels = queries[:, 0]                            # [B]
            row_mask = domain_mask[rels]                    # [B, num_entities]
            cand_in_domain = row_mask.gather(1, cand_entities)  # [B, K_fixed]
            mask = mask & cand_in_domain
        return cand_entities, mask


__all__ = ["Mode", "CandidateSource", "SamplerCandidates"]
