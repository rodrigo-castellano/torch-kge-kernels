"""Fully vectorized corruption sampler shared across projects."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

import torch

from .types import CorruptionOutput, LongTensor, SamplerConfig, SupportsCorruptWithMask, Tensor
from .utils import _mix_hash


class Sampler:
    """Fully vectorized corruption sampler.

    Public tensors are always in ``(relation, head, tail)`` format.
    """

    def __init__(self, cfg: SamplerConfig) -> None:
        self.cfg = cfg
        self.device = cfg.device
        self.num_entities = cfg.num_entities
        self.num_relations = cfg.num_relations
        self.default_mode = cfg.default_mode
        self.order_negatives = cfg.order_negatives
        self.min_entity_idx = cfg.min_entity_idx
        self._filter_hashes_sorted: Optional[LongTensor] = None
        self.b_e = max(2 * self.num_entities + 1, 1024)
        self.b_r = max(2 * self.num_relations + 1, 128)
        self.domain_padded: Optional[Tensor] = None
        self.domain_len: Optional[Tensor] = None
        self.ent2dom: Optional[LongTensor] = None
        self.pos_in_dom: Optional[LongTensor] = None
        self.num_domains = 0
        self.max_pool_len = 0

    @classmethod
    def from_data(
        cls,
        all_known_triples_idx: Optional[LongTensor],
        num_entities: int,
        num_relations: int,
        device: torch.device,
        default_mode: Literal["head", "tail", "both"] = "both",
        seed: int = 0,
        domain2idx: Optional[Dict[str, List[int]]] = None,
        entity2domain: Optional[Dict[int, str]] = None,
        order_negatives: bool = False,
        min_entity_idx: int = 1,
    ) -> "Sampler":
        """Construct a sampler from known triples and optional domain pools.

        Args:
            all_known_triples_idx: Known positive triples in ``(r, h, t)``
                format. Used for filtered corruption generation.
            num_entities: Number of entity ids available to corrupt with.
            num_relations: Number of relation ids.
            device: Device where the internal lookup tensors will live.
            default_mode: Default corruption side used by ``corrupt`` methods.
            seed: Stored seed for downstream callers; runtime sampling remains
                purely tensor-based.
            domain2idx: Optional domain-to-entity mapping for typed corruption.
            entity2domain: Optional inverse mapping used with ``domain2idx``.
            order_negatives: Whether to sort valid negatives deterministically.
            min_entity_idx: Smallest valid entity id. Use ``0`` when entity 0
                is real data rather than padding.
        """
        cfg = SamplerConfig(num_entities, num_relations, device, default_mode, seed, order_negatives, min_entity_idx)
        self = cls(cfg)
        if all_known_triples_idx is not None and all_known_triples_idx.numel() > 0:
            hashes = _mix_hash(all_known_triples_idx.detach().to(device=self.device, dtype=torch.long), self.b_e, self.b_r)
            self._filter_hashes_sorted = torch.sort(torch.unique(hashes)).values
        else:
            self._filter_hashes_sorted = torch.empty((0,), dtype=torch.long, device=self.device)
        if domain2idx and entity2domain:
            self._build_domain_structures(domain2idx, entity2domain, device)
        return self

    @property
    def hashes_sorted(self) -> LongTensor:
        """Return the sorted known-triple hashes used by the filter."""
        if self._filter_hashes_sorted is None:
            return torch.empty((0,), dtype=torch.long, device=self.device)
        return self._filter_hashes_sorted

    def _build_domain_structures(
        self,
        domain2idx: Dict[str, List[int]],
        entity2domain: Dict[int, str],
        device: torch.device,
    ) -> None:
        """Materialize padded domain pools for compiled-friendly sampling."""
        domain_names = sorted(domain2idx.keys())
        domain_lists = [torch.tensor(domain2idx[n], dtype=torch.int32, device=device) for n in domain_names]
        self.num_domains = len(domain_lists)
        self.max_pool_len = max((t.numel() for t in domain_lists), default=0)
        self.domain_padded = torch.zeros((self.num_domains, self.max_pool_len), dtype=torch.int32, device=device)
        self.domain_len = torch.zeros((self.num_domains,), dtype=torch.int32, device=device)
        for i, tensor in enumerate(domain_lists):
            self.domain_padded[i, : tensor.numel()] = tensor
            self.domain_len[i] = tensor.numel()
        max_ent = max(entity2domain.keys(), default=0)
        self.ent2dom = torch.full((max_ent + 1,), -1, dtype=torch.int32, device=device)
        self.pos_in_dom = torch.zeros((max_ent + 1,), dtype=torch.int32, device=device)
        for dom_idx, name in enumerate(domain_names):
            row = self.domain_padded[dom_idx, : self.domain_len[dom_idx]]
            if row.numel() > 0:
                self.ent2dom[row.long()] = dom_idx
                self.pos_in_dom[row.long()] = torch.arange(row.numel(), device=device, dtype=torch.int32)

    def _has_domain_info(self) -> bool:
        """Return ``True`` when typed/domain-aware corruption is available."""
        return self.domain_padded is not None and self.num_domains > 0

    def _get_corruption_indices(self, mode: str) -> List[int]:
        """Map corruption mode to columns in internal ``(h, r, t)`` layout."""
        return [0] if mode == "head" else [2] if mode == "tail" else [0, 2]

    def _corrupt_relation_col(
        self,
        pos_hrt: LongTensor,
        count: int,
        is_exhaustive: bool,
        device: torch.device,
    ) -> LongTensor:
        """Generate relation corruptions in internal ``(h, r, t)`` layout."""
        batch_size = pos_hrt.shape[0]
        if is_exhaustive:
            all_rels = torch.arange(self.num_relations, device=device, dtype=pos_hrt.dtype)
            all_rels = all_rels.unsqueeze(0).expand(batch_size, -1)
            orig_rel = pos_hrt[:, 1:2]
            mask = all_rels != orig_rel
            indices = mask.long().cumsum(dim=1) - 1
            indices = indices.clamp(min=0, max=count - 1)
            return torch.gather(all_rels, 1, indices[:, :count])
        orig = pos_hrt[:, 1:2].expand(-1, count)
        rnd = torch.randint(0, self.num_relations - 1, (batch_size, count), device=device)
        return rnd + (rnd >= orig)

    def _corrupt_entity_global(
        self,
        pos_hrt: LongTensor,
        col: int,
        count: int,
        is_exhaustive: bool,
        device: torch.device,
    ) -> LongTensor:
        """Generate entity corruptions from the global entity pool."""
        batch_size = pos_hrt.shape[0]
        pool_size = self.num_entities
        lo = self.min_entity_idx
        orig_ent = pos_hrt[:, col : col + 1]

        if is_exhaustive:
            all_ents = torch.arange(lo, lo + pool_size, device=device, dtype=pos_hrt.dtype)
            all_ents = all_ents.unsqueeze(0).expand(batch_size, -1)
            mask = all_ents != orig_ent
            sort_keys = (~mask).long()
            _, perm = torch.sort(sort_keys, dim=1, stable=True)
            sorted_ents = torch.gather(all_ents, 1, perm)
            return sorted_ents[:, :count]

        rnd = torch.randint(lo, lo + pool_size - 1, (batch_size, count), device=device)
        return rnd + ((rnd >= orig_ent) & (orig_ent >= lo)).long()

    def _corrupt_entity_domain(
        self,
        pos_hrt: LongTensor,
        col: int,
        count: int,
        is_exhaustive: bool,
        orig_slice: LongTensor,
        device: torch.device,
    ) -> LongTensor:
        """Generate entity corruptions restricted to each example's domain."""
        orig_ent = pos_hrt[:, col]
        d_ids = self.ent2dom[orig_ent].long()
        pools = self.domain_padded[d_ids]
        lo = self.min_entity_idx
        lengths = self.domain_len[d_ids].long()
        positions = torch.arange(pools.shape[1], device=device).unsqueeze(0)
        in_domain = positions < lengths.unsqueeze(1)
        valid_mask = in_domain & (pools != orig_ent.unsqueeze(1)) & (pools >= lo)

        if is_exhaustive:
            sort_keys = (~valid_mask).long()
            _, perm = torch.sort(sort_keys, dim=1, stable=True)
            sorted_pools = torch.gather(pools, 1, perm)
            sorted_valid = torch.gather(valid_mask, 1, perm)
            result = sorted_pools[:, :count]
            result.masked_fill_(~sorted_valid[:, :count], -1)
            return result

        orig_flat = orig_slice.reshape(-1)
        valid = orig_flat >= lo
        result_flat = orig_flat.clone()
        safe_orig = torch.where(valid, orig_flat, torch.full_like(orig_flat, lo))
        d_flat = self.ent2dom[safe_orig].long()
        lengths = self.domain_len[d_flat]
        positions = self.pos_in_dom[safe_orig].long()
        has_alternatives = lengths > 1

        pool_len_m1 = (lengths - 1).float().clamp(min=0)
        rnd = torch.floor(torch.rand(orig_flat.shape, device=device) * pool_len_m1).long()
        shifted_idx = rnd + (rnd >= positions)

        sampled_ents = self.domain_padded[d_flat, shifted_idx].long()
        mask = valid & has_alternatives
        result_flat = torch.where(mask, sampled_ents, result_flat)
        return result_flat.reshape(orig_slice.shape)

    def _filter_mask_batched(self, triples: LongTensor) -> torch.BoolTensor:
        """Return a keep-mask for batched triples against known positives."""
        if self._filter_hashes_sorted is None or self._filter_hashes_sorted.numel() == 0:
            return torch.ones(triples.shape[:-1], dtype=torch.bool, device=triples.device)
        batch_size, k = triples.shape[:2]
        flat = triples.reshape(-1, 3)
        hashes = _mix_hash(flat, self.b_e, self.b_r)

        target = self._filter_hashes_sorted
        pos = torch.searchsorted(target, hashes)
        in_range = pos < target.numel()
        safe_pos = pos.clamp(min=0, max=target.numel() - 1)
        eq = in_range & (target[safe_pos] == hashes)
        return (~eq).reshape(batch_size, k)

    def _filter_keep_mask(self, triples: LongTensor) -> torch.BoolTensor:
        """Return a keep-mask for flat triples against known positives."""
        if self._filter_hashes_sorted is None or self._filter_hashes_sorted.numel() == 0:
            return torch.ones((triples.shape[0],), dtype=torch.bool, device=triples.device)
        hashes = _mix_hash(triples, self.b_e, self.b_r)
        target = self._filter_hashes_sorted

        pos = torch.searchsorted(target, hashes)
        in_range = pos < target.numel()
        safe_pos = pos.clamp(min=0, max=target.numel() - 1)
        eq = in_range & (target[safe_pos] == hashes)
        return ~eq

    def _postprocess(
        self,
        neg: LongTensor,
        batch_size: int,
        k: int,
        do_filter: bool,
        do_unique: bool,
        device: torch.device,
    ) -> tuple[LongTensor, torch.BoolTensor]:
        """Convert internal triples to public ``(r, h, t)`` and compact them."""
        neg_rht = torch.stack([neg[:, :, 1], neg[:, :, 0], neg[:, :, 2]], dim=-1)
        lo = self.min_entity_idx
        valid = (neg_rht[:, :, 1] >= lo) & (neg_rht[:, :, 2] >= lo)

        if do_filter:
            valid = valid & self._filter_mask_batched(neg_rht)

        if do_unique:
            hashes = _mix_hash(neg_rht, self.b_e, self.b_r)
            hashes = torch.where(valid, hashes, torch.full_like(hashes, -1))
            for idx in range(1, k):
                is_dup = (hashes[:, :idx] == hashes[:, idx : idx + 1]).any(dim=1)
                valid[:, idx] = valid[:, idx] & ~is_dup

        neg_rht = neg_rht * valid.unsqueeze(-1)

        sort_key = (~valid).long()
        _, perm = torch.sort(sort_key, dim=1, stable=True)
        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, k)
        neg_rht = neg_rht[batch_idx, perm]
        valid = valid[batch_idx, perm]

        if self.order_negatives and k > 0:
            keys = neg_rht[:, :, 0] * 10_000_000 + neg_rht[:, :, 1] * 10_000 + neg_rht[:, :, 2]
            keys = keys.masked_fill(neg_rht.sum(-1) == 0, 2**62)
            _, idx = torch.sort(keys, dim=1)
            neg_rht = neg_rht[batch_idx, idx]
            valid = valid[batch_idx, idx]

        return neg_rht, valid

    def corrupt_with_mask(
        self,
        positives: LongTensor,
        *,
        num_negatives: Optional[int] = None,
        mode: Optional[Literal["head", "tail", "both"]] = None,
        device: Optional[torch.device] = None,
        filter: bool = True,
        unique: bool = True,
    ) -> tuple[LongTensor, torch.BoolTensor]:
        """Generate corruptions and an explicit validity mask.

        Args:
            positives: Positive triples in public ``(r, h, t)`` format.
            num_negatives: Number of negatives per positive. ``None`` means
                exhaustive corruption over the active candidate pool.
            mode: Which side to corrupt: ``head``, ``tail``, or ``both``.
            device: Optional override for the runtime device.
            filter: Remove known positives using the indexed fact set.
            unique: Deduplicate negatives within each row after filtering.

        Returns:
            ``(negatives, valid_mask)`` where ``negatives`` has shape
            ``[B, K, 3]`` in ``(r, h, t)`` format and ``valid_mask`` has shape
            ``[B, K]``.
        """
        device = device or self.device
        mode = mode or self.default_mode
        pos = positives.to(device=device)
        batch_size = pos.shape[0]

        if batch_size == 0:
            empty_neg = torch.zeros((0, num_negatives or 0, 3), dtype=pos.dtype, device=device)
            empty_valid = torch.zeros((0, num_negatives or 0), dtype=torch.bool, device=device)
            return empty_neg, empty_valid

        pos_hrt = torch.stack([pos[:, 1], pos[:, 0], pos[:, 2]], dim=1)
        cols = self._get_corruption_indices(mode)
        is_exhaustive = num_negatives is None

        pool_size = (self.max_pool_len - 1) if self._has_domain_info() else (self.num_entities - 1)
        k = pool_size * len(cols) if is_exhaustive else num_negatives
        neg = pos_hrt.unsqueeze(1).expand(batch_size, k, 3).clone()

        per_col = k // len(cols)
        remainder = k % len(cols)
        start = 0
        for idx, col in enumerate(cols):
            count = per_col + (1 if idx < remainder else 0)
            if count == 0:
                continue
            if col == 1:
                result = self._corrupt_relation_col(pos_hrt, count, is_exhaustive, device)
            elif not self._has_domain_info():
                result = self._corrupt_entity_global(pos_hrt, col, count, is_exhaustive, device)
            else:
                orig_slice = neg[:, start : start + count, col]
                result = self._corrupt_entity_domain(pos_hrt, col, count, is_exhaustive, orig_slice, device)
            neg[:, start : start + count, col] = result
            start += count

        return self._postprocess(neg, batch_size, k, do_filter=filter, do_unique=unique, device=device)

    def corrupt(
        self,
        positives: LongTensor,
        *,
        num_negatives: Optional[int] = None,
        mode: Optional[Literal["head", "tail", "both"]] = None,
        device: Optional[torch.device] = None,
        filter: bool = True,
        unique: bool = True,
    ) -> LongTensor:
        """Generate corruptions only.

        This is a convenience wrapper over :meth:`corrupt_with_mask` for callers
        that rely on padded invalid rows instead of an explicit mask.
        """
        neg, _ = self.corrupt_with_mask(
            positives,
            num_negatives=num_negatives,
            mode=mode,
            device=device,
            filter=filter,
            unique=unique,
        )
        return neg


def corrupt(
    sampler: SupportsCorruptWithMask,
    positives: LongTensor,
    *,
    num_corruptions: Optional[int] = None,
    mode: Optional[Literal["head", "tail", "both"]] = None,
    device: Optional[torch.device] = None,
    filter: bool = True,
    unique: bool = True,
) -> CorruptionOutput:
    """Public corruption entry point for sampled or exhaustive generation.

    Args:
        sampler: Any sampler exposing ``corrupt_with_mask``.
        positives: Positive triples in ``(r, h, t)`` format, shape ``[B, 3]``.
        num_corruptions: Number of corruptions per query. ``None`` means
            exhaustive corruption over the active candidate pool.
        mode: Which side to corrupt: ``head``, ``tail``, or ``both``.
        device: Optional runtime device override.
        filter: Remove known positives using the sampler index.
        unique: Deduplicate negatives within each query after filtering.

    Returns:
        ``CorruptionOutput`` with padded negatives ``[B, K, 3]`` in ``(r, h, t)``
        format and a ``valid_mask`` of shape ``[B, K]``.
    """

    negatives, valid_mask = sampler.corrupt_with_mask(
        positives,
        num_negatives=num_corruptions,
        mode=mode,
        device=device,
        filter=filter,
        unique=unique,
    )
    return CorruptionOutput(negatives=negatives, valid_mask=valid_mask)


__all__ = ["Sampler", "SamplerConfig", "corrupt"]
