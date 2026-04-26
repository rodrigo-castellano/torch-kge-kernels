"""Partial-atom scoring utilities built on top of :func:`kge_score`."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .kge import kge_score


@torch.no_grad()
def precompute_partial_scores(
    model: nn.Module,
    pred_remap: Tensor,
    const_remap: Tensor,
    batch_chunk: int = 64,
    *,
    sigmoid: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Precompute partial-atom lookup tables for grounding-style use cases.

    For each mapped predicate/entity pair, this computes:

    - the best score over all possible tails for ``pred(entity, ?)``
    - the best score over all possible heads for ``pred(?, entity)``

    Args:
        model: tkk-native KGE model.
        pred_remap: Predicate remap tensor of shape ``[P_im]`` with ``-1`` for
            unmapped predicates.
        const_remap: Entity remap tensor of shape ``[E_im]`` with ``-1`` for
            unmapped entities.
        batch_chunk: Number of entities scored per chunk in the batched
            exhaustive calls.
        sigmoid: forwarded to :func:`kge_score`.

    Returns:
        ``(max_tail_score, max_head_score)``, both of shape ``[P_im, E_im]``.
    """

    device = const_remap.device
    p_im = pred_remap.shape[0]
    e_im = const_remap.shape[0]

    max_tail_score = torch.zeros(p_im, e_im, dtype=torch.float32, device=device)
    max_head_score = torch.zeros(p_im, e_im, dtype=torch.float32, device=device)

    valid_preds = (pred_remap >= 0).nonzero(as_tuple=True)[0]
    valid_ents = (const_remap >= 0).nonzero(as_tuple=True)[0]
    if valid_preds.numel() == 0 or valid_ents.numel() == 0:
        return max_tail_score, max_head_score

    if batch_chunk <= 0:
        batch_chunk = 64

    kge_ents = const_remap[valid_ents]
    for im_pred in valid_preds:
        kge_rel = pred_remap[im_pred]
        tail_scores = _partial_score_chunked(
            model, kge_ents, kge_rel, role=0, batch_chunk=batch_chunk, sigmoid=sigmoid,
        )
        head_scores = _partial_score_chunked(
            model, kge_ents, kge_rel, role=1, batch_chunk=batch_chunk, sigmoid=sigmoid,
        )
        max_tail_score[im_pred, valid_ents] = tail_scores
        max_head_score[im_pred, valid_ents] = head_scores

    return max_tail_score, max_head_score


def _partial_score_chunked(
    model: nn.Module,
    kge_ents: Tensor,
    kge_rel: Tensor,
    role: int,
    batch_chunk: int,
    sigmoid: bool,
) -> Tensor:
    """Compute best head/tail completions for a chunk of entities."""

    num_entities = kge_ents.shape[0]
    device = kge_ents.device
    result = torch.empty(num_entities, dtype=torch.float32, device=device)

    for start in range(0, num_entities, batch_chunk):
        end = min(start + batch_chunk, num_entities)
        chunk = kge_ents[start:end]
        rel_exp = kge_rel.expand(chunk.shape[0])
        if role == 0:
            raw = kge_score(model, chunk, rel_exp, None, sigmoid=sigmoid)
        else:
            raw = kge_score(model, None, rel_exp, chunk, sigmoid=sigmoid)
        result[start:end] = raw.max(dim=1).values

    return result


def score_partial_atoms(
    preds: Tensor,
    args1: Tensor,
    args2: Tensor,
    constant_no: int,
    max_tail_score: Tensor,
    max_head_score: Tensor,
) -> Tensor:
    """Score partially grounded binary atoms by table lookup.

    The function assumes the shared grounding convention:

    - ``pred(const, ?)`` uses ``max_tail_score``
    - ``pred(?, const)`` uses ``max_head_score``
    - fully unbound atoms receive score ``0``

    Args:
        preds: Predicate ids ``[N]``.
        args1: First argument ids ``[N]``.
        args2: Second argument ids ``[N]``.
        constant_no: Highest constant id in the caller's index space.
        max_tail_score: Precomputed tail lookup table ``[P, E]``.
        max_head_score: Precomputed head lookup table ``[P, E]``.

    Returns:
        Scores of shape ``[N]``.
    """

    n = preds.shape[0]
    device = preds.device
    scores = torch.zeros(n, dtype=torch.float32, device=device)
    if n == 0:
        return scores

    a1 = args1.long()
    a2 = args2.long()
    p = preds.long()
    p_max, e_max = max_tail_score.shape

    safe_p = p.clamp(0, p_max - 1)
    safe_a1 = a1.clamp(0, e_max - 1)
    safe_a2 = a2.clamp(0, e_max - 1)

    tail_var = (a1 > 0) & (a1 <= constant_no) & (a2 > constant_no)
    scores = torch.where(tail_var, max_tail_score[safe_p, safe_a1], scores)

    head_var = (a1 > constant_no) & (a2 > 0) & (a2 <= constant_no)
    scores = torch.where(head_var, max_head_score[safe_p, safe_a2], scores)

    return scores


class LazyPartialScorer:
    """Lazy per-(predicate, entity) partial score computation with caching.

    Computes ``max_tail_score[p, e]`` and ``max_head_score[p, e]`` on-the-fly
    as partial atoms are encountered during proof search, then caches them in
    lookup tables for subsequent accesses. No upfront precompute — scales to
    any dataset size.

    Call ``ensure_for_derived_states(derived_states, derived_counts)`` before
    each scoring step. The scorer then uses ``max_tail_score`` / ``max_head_score``
    as pure lookup tables (CUDA-graph safe).
    """

    def __init__(
        self,
        model: nn.Module,
        pred_remap: Tensor,
        const_remap: Tensor,
        constant_no: int,
        padding_idx: int,
        true_pred_idx: int = -1,
        false_pred_idx: int = -1,
        *,
        sigmoid: bool = True,
    ) -> None:
        device = const_remap.device
        P_im = pred_remap.shape[0]
        E_im = const_remap.shape[0]

        self.model = model
        self.pred_remap = pred_remap
        self.const_remap = const_remap
        self.constant_no = constant_no
        self.padding_idx = padding_idx
        self.true_pred_idx = true_pred_idx
        self.false_pred_idx = false_pred_idx
        self.sigmoid = sigmoid
        self.max_tail_score = torch.zeros(P_im, E_im, dtype=torch.float32, device=device)
        self.max_head_score = torch.zeros(P_im, E_im, dtype=torch.float32, device=device)

        self._cached_tail = torch.zeros(P_im, E_im, dtype=torch.bool, device=device)
        self._cached_head = torch.zeros(P_im, E_im, dtype=torch.bool, device=device)
        self._computed: set = set()
        self._n_computed = 0

        E_kge = (const_remap >= 0).sum().item()
        self._batch_chunk = max(1, min(512, int(2e9 / (max(E_kge, 1) * 4 * 1024))))

    @torch.no_grad()
    def ensure(self, preds: Tensor) -> None:
        """Pre-compute ALL entities for given predicates (eager, per-predicate)."""
        unique = preds.unique().tolist()
        new = [int(p) for p in unique if p > 0 and int(p) not in self._computed]
        if not new:
            return
        valid_mask = self.const_remap >= 0
        valid_indices = valid_mask.nonzero(as_tuple=True)[0]
        valid_kge = self.const_remap[valid_indices]
        E = valid_kge.shape[0]
        if E == 0:
            return
        for im_pred in new:
            kge_rel = self.pred_remap[im_pred]
            if kge_rel < 0:
                self.max_tail_score[im_pred, valid_indices] = 0.5
                self.max_head_score[im_pred, valid_indices] = 0.5
                self._cached_tail[im_pred, valid_indices] = True
                self._cached_head[im_pred, valid_indices] = True
                self._computed.add(im_pred)
                continue
            for start in range(0, E, self._batch_chunk):
                end = min(start + self._batch_chunk, E)
                ents = valid_kge[start:end]
                idx = valid_indices[start:end]
                rel = kge_rel.expand(ents.shape[0])
                self.max_tail_score[im_pred, idx] = kge_score(
                    self.model, ents, rel, None, sigmoid=self.sigmoid,
                ).max(dim=1).values
                self.max_head_score[im_pred, idx] = kge_score(
                    self.model, None, rel, ents, sigmoid=self.sigmoid,
                ).max(dim=1).values
            self._cached_tail[im_pred, valid_indices] = True
            self._cached_head[im_pred, valid_indices] = True
            self._computed.add(im_pred)
            self._n_computed += E

    @torch.no_grad()
    def ensure_for_derived_states(
        self,
        derived_states: Tensor,
        derived_counts: Tensor,
    ) -> None:
        """Compute and cache partial scores for atoms in derived states.

        Only processes valid (non-padding) atoms. Single GPU→CPU sync.
        """
        B, S, A, _ = derived_states.shape
        C = self.constant_no
        pad = self.padding_idx

        valid_s = torch.arange(S, device=derived_states.device).unsqueeze(0) < derived_counts.unsqueeze(1)
        valid_atoms = valid_s.unsqueeze(2).expand(B, S, A).reshape(-1)
        flat = derived_states.reshape(-1, 3)[valid_atoms]

        if flat.shape[0] == 0:
            return

        preds = flat[:, 0]
        args1 = flat[:, 1]
        args2 = flat[:, 2]

        a1_const = (args1 > 0) & (args1 <= C)
        a2_const = (args2 > 0) & (args2 <= C)
        is_partial_tail = a1_const & ~a2_const & (preds != pad) & (preds != self.true_pred_idx)
        is_partial_head = ~a1_const & a2_const & (preds != pad) & (preds != self.true_pred_idx)

        tp = preds[is_partial_tail]
        te = args1[is_partial_tail]
        hp = preds[is_partial_head]
        he = args2[is_partial_head]

        tail_uncached = ~self._cached_tail[tp, te] if tp.numel() > 0 else tp
        head_uncached = ~self._cached_head[hp, he] if hp.numel() > 0 else hp
        n = (tail_uncached.sum() + head_uncached.sum()).item()
        if n == 0:
            return

        if tail_uncached.numel() > 0 and tail_uncached.any():
            self._compute_entries(tp[tail_uncached], te[tail_uncached], role="tail")
        if head_uncached.numel() > 0 and head_uncached.any():
            self._compute_entries(hp[head_uncached], he[head_uncached], role="head")

    def _compute_entries(self, preds: Tensor, entities: Tensor, role: str) -> None:
        """Batch-compute missing partial scores, grouped by predicate."""
        keys = preds.long() * (self.const_remap.shape[0] + 1) + entities.long()
        unique_keys = keys.unique()
        unique_preds = unique_keys // (self.const_remap.shape[0] + 1)
        unique_ents = unique_keys % (self.const_remap.shape[0] + 1)

        for p_im in unique_preds.unique().tolist():
            p_im = int(p_im)
            kge_rel = self.pred_remap[p_im]
            mask = unique_preds == p_im
            im_ents = unique_ents[mask]

            if kge_rel < 0:
                table = self.max_tail_score if role == "tail" else self.max_head_score
                cache = self._cached_tail if role == "tail" else self._cached_head
                table[p_im, im_ents] = 0.5
                cache[p_im, im_ents] = True
                continue

            kge_ents = self.const_remap[im_ents]
            valid = kge_ents >= 0
            if not valid.any():
                continue
            kge_ents_valid = kge_ents[valid]
            im_ents_valid = im_ents[valid]

            table = self.max_tail_score if role == "tail" else self.max_head_score
            cache = self._cached_tail if role == "tail" else self._cached_head
            bc = self._batch_chunk

            for start in range(0, kge_ents_valid.shape[0], bc):
                end = min(start + bc, kge_ents_valid.shape[0])
                chunk_kge = kge_ents_valid[start:end]
                chunk_im = im_ents_valid[start:end]
                rel = kge_rel.expand(chunk_kge.shape[0])
                if role == "tail":
                    s = kge_score(self.model, chunk_kge, rel, None, sigmoid=self.sigmoid)
                else:
                    s = kge_score(self.model, None, rel, chunk_kge, sigmoid=self.sigmoid)
                table[p_im, chunk_im] = s.max(dim=1).values
                cache[p_im, chunk_im] = True

            self._n_computed += kge_ents_valid.shape[0]

    @property
    def n_computed(self) -> int:
        return self._n_computed


__all__ = ["LazyPartialScorer", "precompute_partial_scores", "score_partial_atoms"]
