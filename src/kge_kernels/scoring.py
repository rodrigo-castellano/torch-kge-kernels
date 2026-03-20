"""Unified KGE scoring primitives."""

from __future__ import annotations

from typing import Dict, Optional, Set, Tuple

import torch
import torch.nn as nn
from torch import Tensor


def _normalize_scores(model: nn.Module, raw: Tensor) -> Tensor:
    mode = getattr(model, "kge_output_mode", "raw")
    if mode == "raw":
        return torch.sigmoid(raw)
    if mode == "sigmoid":
        return raw
    raise ValueError(f"Unsupported kge_output_mode={mode!r}; expected 'raw' or 'sigmoid'")


def kge_score_triples(model: nn.Module, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
    if hasattr(model, "score_triples"):
        raw = model.score_triples(h, r, t)
    else:
        raw = model.score_atoms(r, h, t)
    return _normalize_scores(model, raw)


def kge_score_all_tails(
    model: nn.Module,
    h: Tensor,
    r: Tensor,
    *,
    filter_map: Optional[Dict[Tuple[int, int], Set[int]]] = None,
    true_tails: Optional[Tensor] = None,
    domain: Optional[Set[int]] = None,
) -> Tensor:
    if hasattr(model, "score_all_tails_batch"):
        raw = model.score_all_tails_batch(h, r)
        scores = _normalize_scores(model, raw)
    else:
        batch_size = h.shape[0]
        num_entities: int = model.num_constants
        dev = h.device
        h_exp = h.unsqueeze(1).expand(batch_size, num_entities).reshape(-1)
        r_exp = r.unsqueeze(1).expand(batch_size, num_entities).reshape(-1) if r.dim() > 0 else r.expand(batch_size * num_entities)
        t_all = torch.arange(num_entities, device=dev).unsqueeze(0).expand(batch_size, num_entities).reshape(-1)
        scores = kge_score_triples(model, h_exp, r_exp, t_all).view(batch_size, num_entities)

    _apply_masks(scores, h, r, filter_map, true_tails, domain)
    return scores


def kge_score_all_heads(
    model: nn.Module,
    r: Tensor,
    t: Tensor,
    *,
    filter_map: Optional[Dict[Tuple[int, int], Set[int]]] = None,
    true_heads: Optional[Tensor] = None,
    domain: Optional[Set[int]] = None,
) -> Tensor:
    if hasattr(model, "score_all_heads_batch"):
        raw = model.score_all_heads_batch(r, t)
        scores = _normalize_scores(model, raw)
    else:
        batch_size = t.shape[0]
        num_entities: int = model.num_constants
        dev = t.device
        t_exp = t.unsqueeze(1).expand(batch_size, num_entities).reshape(-1)
        r_exp = r.unsqueeze(1).expand(batch_size, num_entities).reshape(-1) if r.dim() > 0 else r.expand(batch_size * num_entities)
        h_all = torch.arange(num_entities, device=dev).unsqueeze(0).expand(batch_size, num_entities).reshape(-1)
        scores = kge_score_triples(model, h_all, r_exp, t_exp).view(batch_size, num_entities)

    _apply_masks(scores, r, t, filter_map, true_heads, domain)
    return scores


def _apply_masks(
    scores: Tensor,
    idx1: Tensor,
    idx2: Tensor,
    filter_map: Optional[Dict[Tuple[int, int], Set[int]]],
    true_entities: Optional[Tensor],
    domain: Optional[Set[int]],
) -> None:
    batch_size, num_entities = scores.shape
    device = scores.device

    if domain is not None:
        domain_mask = torch.zeros(num_entities, dtype=torch.bool, device=device)
        domain_mask[torch.tensor(sorted(domain), dtype=torch.long, device=device)] = True
        scores[:, ~domain_mask] = float("-inf")

    if filter_map is not None and true_entities is not None:
        idx1_list = idx1.tolist() if idx1.dim() > 0 else [idx1.item()] * batch_size
        idx2_list = idx2.tolist() if idx2.dim() > 0 else [idx2.item()] * batch_size
        for row in range(batch_size):
            key = (int(idx1_list[row]), int(idx2_list[row]))
            known = filter_map.get(key)
            if known:
                true_ent = true_entities[row].item()
                indices = torch.tensor([ent for ent in known if ent != true_ent], dtype=torch.long, device=device)
                if indices.numel() > 0:
                    scores[row, indices] = float("-inf")


def kge_score_k_tails(model: nn.Module, h: Tensor, r: Tensor, t: Tensor, sampler: object, num_corruptions: int) -> Tuple[Tensor, Tensor]:
    queries = torch.stack([r, h, t], dim=1)
    neg = sampler.corrupt(queries, num_negatives=num_corruptions, mode="tail")
    k = neg.shape[1]
    pos_scores = kge_score_triples(model, h, r, t)
    neg_h = neg[:, :, 1].reshape(-1)
    neg_r = neg[:, :, 0].reshape(-1)
    neg_t = neg[:, :, 2].reshape(-1)
    neg_scores = kge_score_triples(model, neg_h, neg_r, neg_t).view(h.shape[0], k)
    is_valid = neg.sum(dim=-1) > 0
    neg_scores[~is_valid] = float("-inf")
    return torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1), is_valid


def kge_score_k_heads(model: nn.Module, h: Tensor, r: Tensor, t: Tensor, sampler: object, num_corruptions: int) -> Tuple[Tensor, Tensor]:
    queries = torch.stack([r, h, t], dim=1)
    neg = sampler.corrupt(queries, num_negatives=num_corruptions, mode="head")
    k = neg.shape[1]
    pos_scores = kge_score_triples(model, h, r, t)
    neg_h = neg[:, :, 1].reshape(-1)
    neg_r = neg[:, :, 0].reshape(-1)
    neg_t = neg[:, :, 2].reshape(-1)
    neg_scores = kge_score_triples(model, neg_h, neg_r, neg_t).view(h.shape[0], k)
    is_valid = neg.sum(dim=-1) > 0
    neg_scores[~is_valid] = float("-inf")
    return torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1), is_valid


@torch.no_grad()
def precompute_partial_scores(
    kge_model: nn.Module,
    pred_remap: Tensor,
    const_remap: Tensor,
    batch_chunk: int = 0,
    entity_chunk: int = 2048,
) -> Tuple[Tensor, Tensor]:
    del entity_chunk
    device = const_remap.device
    p_im = pred_remap.shape[0]
    e_im = const_remap.shape[0]

    max_tail_score = torch.zeros(p_im, e_im, dtype=torch.float32, device=device)
    max_head_score = torch.zeros(p_im, e_im, dtype=torch.float32, device=device)

    valid_preds = (pred_remap >= 0).nonzero(as_tuple=True)[0]
    valid_ents = (const_remap >= 0).nonzero(as_tuple=True)[0]
    n_ents = valid_ents.shape[0]

    if n_ents == 0 or valid_preds.shape[0] == 0:
        return max_tail_score, max_head_score

    if batch_chunk <= 0:
        dim = 512
        if hasattr(kge_model, "ent_re"):
            dim = kge_model.ent_re.weight.shape[1]
        elif hasattr(kge_model, "embedding_dim"):
            dim = kge_model.embedding_dim
        bytes_per_elem = max(1, dim) * 4 * 2
        batch_chunk = max(8, min(512, int(2e9 / bytes_per_elem)))

    kge_ents = const_remap[valid_ents]
    for im_pred in valid_preds:
        kge_rel = pred_remap[im_pred]
        tail_scores = _partial_score_chunked(kge_model, kge_ents, kge_rel, role=0, batch_chunk=batch_chunk)
        max_tail_score[im_pred, valid_ents] = tail_scores
        head_scores = _partial_score_chunked(kge_model, kge_ents, kge_rel, role=1, batch_chunk=batch_chunk)
        max_head_score[im_pred, valid_ents] = head_scores

    return max_tail_score, max_head_score


def _partial_score_chunked(kge_model: nn.Module, kge_ents: Tensor, kge_rel: Tensor, role: int, batch_chunk: int = 64) -> Tensor:
    num_entities = kge_ents.shape[0]
    device = kge_ents.device
    result = torch.empty(num_entities, dtype=torch.float32, device=device)

    for start in range(0, num_entities, batch_chunk):
        end = min(start + batch_chunk, num_entities)
        chunk = kge_ents[start:end]
        batch = chunk.shape[0]
        rel_exp = kge_rel.expand(batch)
        if role == 0:
            raw = kge_score_all_tails(kge_model, chunk, rel_exp)
        else:
            raw = kge_score_all_heads(kge_model, rel_exp, chunk)
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


__all__ = [
    "kge_score_all_heads",
    "kge_score_all_tails",
    "kge_score_k_heads",
    "kge_score_k_tails",
    "kge_score_triples",
    "precompute_partial_scores",
    "score_partial_atoms",
]
