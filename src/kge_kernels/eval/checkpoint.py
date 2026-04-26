"""Convenience evaluation entrypoints.

``evaluate_ranking``    — filtered ranking helper for callers that have
                          ``head_filter`` / ``tail_filter`` dicts on hand.
                          Self-contained matmul-based exhaustive or
                          sampled ranking against a tkk model.
``evaluate_checkpoint`` — load a saved checkpoint and evaluate a split.

For the canonical scorer-pluggable evaluator (used by torch-ns, DpRL,
and any custom scorer with an ``eval_scores`` method), use
:func:`kge_kernels.eval.evaluate` directly.
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
from torch import Tensor, nn

from .ranking import compute_ranks, ranking_metrics


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Unwrap DataParallel and torch.compile wrappers."""
    inner: nn.Module = model.module if isinstance(model, nn.DataParallel) else model
    if hasattr(inner, "_orig_mod"):
        inner = inner._orig_mod  # type: ignore[assignment]
    return inner


def _apply_masks(
    scores: Tensor,
    idx1: Tensor,
    idx2: Tensor,
    filter_map: Optional[Dict[Tuple[int, int], Set[int]]],
    true_entities: Optional[Tensor],
    domain: Optional[Set[int]],
) -> None:
    """Apply domain and known-fact filter masks to a score matrix in-place."""
    batch_size, num_entities = scores.shape
    device = scores.device

    if domain is not None:
        domain_mask = torch.zeros(num_entities, dtype=torch.bool, device=device)
        domain_mask[torch.tensor(sorted(domain), dtype=torch.long, device=device)] = True
        scores[:, ~domain_mask] = float("-inf")

    if filter_map is not None and true_entities is not None:
        idx1_list = idx1.tolist() if idx1.dim() > 0 else [int(idx1.item())] * batch_size
        idx2_list = idx2.tolist() if idx2.dim() > 0 else [int(idx2.item())] * batch_size
        for row in range(batch_size):
            key = (int(idx1_list[row]), int(idx2_list[row]))
            known = filter_map.get(key)
            if known:
                true_ent = int(true_entities[row].item())
                filtered = [ent for ent in known if ent != true_ent]
                if filtered:
                    scores[row, torch.tensor(filtered, dtype=torch.long, device=device)] = float("-inf")


@torch.no_grad()
def evaluate_ranking(
    model: nn.Module,
    triples: Sequence[Tuple[int, int, int]],
    num_entities: int,
    head_filter: Dict[Tuple[int, int], Set[int]],
    tail_filter: Dict[Tuple[int, int], Set[int]],
    device: torch.device,
    chunk_size: int = 0,
    head_domain: Optional[Dict[int, Set[int]]] = None,
    tail_domain: Optional[Dict[int, Set[int]]] = None,
    eval_num_corruptions: int = 0,
    seed: int = 0,
    eval_batch_size: int = 0,
    show_progress: bool = False,
    progress_label: str = "Evaluation",
    corruption_scheme: str = "both",
    sampler=None,
) -> Dict[str, float]:
    """Filtered ranking on a tkk KGE model.

    ``eval_num_corruptions`` controls sampling: ``0`` = exhaustive (rank
    against every entity), ``>0`` = sampled ranking against that many
    corruptions per query (requires ``sampler``). Filtered protocol
    masks known facts from the candidate pool. ``head_domain`` /
    ``tail_domain`` further restrict candidates per relation.

    For scorer-pluggable evaluation (custom scorer objects with
    ``eval_scores``), use :func:`kge_kernels.eval.evaluate` instead.
    """
    del eval_batch_size, progress_label

    if eval_num_corruptions > 0 and sampler is None:
        raise ValueError("Sampled evaluation (eval_num_corruptions > 0) requires a sampler")

    if not triples:
        return {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0}

    triples_list: List[Tuple[int, int, int]] = [(int(r), int(h), int(t)) for r, h, t in triples]

    training_mode = model.training
    model.eval()
    actual_model = _unwrap_model(model)

    dim = getattr(actual_model, "dim", 512)
    max_chunk = max(1, min(512, (4 * 1024**3) // max(1, num_entities * dim * 4)))

    by_relation: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
    for r, h, t in triples_list:
        by_relation[r].append((r, h, t))

    modes: Sequence[str] = ("head", "tail") if corruption_scheme == "both" else (corruption_scheme,)
    K = eval_num_corruptions
    sampled = K > 0
    tie_gen = torch.Generator(device=device).manual_seed(seed) if sampled else None

    all_ranks_list: List[Tensor] = []
    processed = 0
    report_every = max(1, len(triples_list) // 20)
    last_report = 0

    for r, rel_triples in by_relation.items():
        all_heads = torch.tensor([h for _, h, _ in rel_triples], dtype=torch.long, device=device)
        all_tails = torch.tensor([t for _, _, t in rel_triples], dtype=torch.long, device=device)
        r_tensor = torch.tensor(r, dtype=torch.long, device=device)
        t_dom = tail_domain.get(r) if tail_domain else None
        h_dom = head_domain.get(r) if head_domain else None

        for chunk_start in range(0, len(rel_triples), max_chunk):
            chunk_end = min(chunk_start + max_chunk, len(rel_triples))
            heads = all_heads[chunk_start:chunk_end]
            tails = all_tails[chunk_start:chunk_end]
            B = heads.shape[0]

            for mode in modes:
                if mode == "tail":
                    r_b = r_tensor.expand(B)
                    full_scores = actual_model.score(heads, r_b, None)
                    true_ents = tails
                    fmap, idx1, idx2, dom = tail_filter, heads, r_b, t_dom
                else:
                    r_b = r_tensor.expand(B)
                    full_scores = actual_model.score(None, r_b, tails)
                    true_ents = heads
                    fmap, idx1, idx2, dom = head_filter, r_b, tails, h_dom

                _apply_masks(full_scores, idx1, idx2, fmap, true_ents, dom)

                if sampled:
                    pos_triples = torch.stack([r_b, heads, tails], dim=1)
                    neg, valid_mask = sampler.corrupt(
                        pos_triples, num_negatives=K, mode=mode,
                        device=device, filter=True, unique=False, return_mask=True,
                    )
                    corrupt_col = 1 if mode == "head" else 2
                    sampled_ents = neg[:, :, corrupt_col].clamp(0, num_entities - 1)
                    true_scores = full_scores[torch.arange(B, device=device), true_ents].unsqueeze(1)
                    corr_scores = full_scores.gather(1, sampled_ents)
                    pool_scores = torch.cat([true_scores, corr_scores], dim=1)
                    valid = torch.ones(B, 1 + K, dtype=torch.bool, device=device)
                    valid[:, 1:] = valid_mask
                    true_idx = torch.zeros(B, dtype=torch.long, device=device)
                    ranks = compute_ranks(
                        pool_scores, true_idx, valid_mask=valid,
                        tie_handling="random", generator=tie_gen,
                    )
                else:
                    ranks = compute_ranks(full_scores, true_ents)
                all_ranks_list.append(ranks)

        processed += len(rel_triples)
        if show_progress and (processed >= len(triples_list) or processed - last_report >= report_every):
            running = torch.cat(all_ranks_list)
            running_mrr = (1.0 / running.double()).mean().item()
            print(f"Evaluation {processed}/{len(triples_list)} triples | rolling_mrr={running_mrr:.4f}")
            last_report = processed

    if training_mode:
        model.train()

    if not all_ranks_list:
        return {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0}
    return ranking_metrics(torch.cat(all_ranks_list))


def evaluate_checkpoint(
    checkpoint_dir: str,
    *,
    weights_name: str = "best_weights.pth",
    split: str = "test",
    cpu: bool = False,
    compile_model: bool = False,
    compile_mode: str = "reduce-overhead",
    compile_fullgraph: bool = True,
    compile_warmup_steps: int = 0,
    show_progress: bool = True,
    eval_limit: int = 0,
) -> Dict[str, float]:
    """Load a saved checkpoint and evaluate on the requested split."""
    from ..data import (
        add_reciprocal_triples,
        build_filter_maps,
        build_relation_domains,
        encode_split_triples,
        load_triples_with_mappings,
    )
    from ..data import resolve_split_path, resolve_train_path
    from ..models.factory import build_training_model
    from ..training.checkpoints import (
        config_from_payload,
        load_checkpoint_payload,
        normalize_loaded_state_dict,
        unwrap_model,
    )

    payload = load_checkpoint_payload(checkpoint_dir)
    cfg = config_from_payload(payload, checkpoint_dir)
    cfg.cpu = cpu
    cfg.multi_gpu = False
    cfg.compile = compile_model
    cfg.compile_mode = compile_mode
    cfg.compile_fullgraph = compile_fullgraph

    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.cpu else "cpu")
    train_file = resolve_train_path(cfg.train_path, cfg.dataset, cfg.data_root, cfg.train_split)
    print(f"Loading triples from {train_file} ...")
    # tkk standalone uses dense 0-based ids; see ``training/pipeline.py``
    # for the full rationale.
    triples, e2id, r2id = load_triples_with_mappings(train_file, padding_idx=None)
    if not triples:
        raise ValueError("No triples found for evaluation")

    valid_triples = []
    valid_path = resolve_split_path(
        split_name="valid",
        explicit_path=cfg.valid_path,
        dataset=cfg.dataset,
        data_root=cfg.data_root,
        split_filename=cfg.valid_split,
    )
    if valid_path:
        print(f"Loading validation triples from {valid_path} ...")
        valid_triples = encode_split_triples(valid_path, e2id, r2id, "validation")
        print(f"#valid triples={len(valid_triples):,}")

    test_triples = []
    test_path = resolve_split_path(
        split_name="test",
        explicit_path=cfg.test_path,
        dataset=cfg.dataset,
        data_root=cfg.data_root,
        split_filename=cfg.test_split,
    )
    if test_path:
        print(f"Loading test triples from {test_path} ...")
        test_triples = encode_split_triples(test_path, e2id, r2id, "test")
        print(f"#test triples={len(test_triples):,}")

    num_entities = len(e2id)
    num_relations = len(r2id)
    head_filter, tail_filter = build_filter_maps(triples, valid_triples, test_triples)
    use_domain_eval = bool(cfg.dataset and "countries" in cfg.dataset.lower())
    head_domain, tail_domain = (build_relation_domains(triples + valid_triples + test_triples) if use_domain_eval else (None, None))

    if cfg.use_reciprocal:
        triples, r2id, num_relations = add_reciprocal_triples(triples, r2id, inv_suffix="__inv")

    model = build_training_model(cfg, num_entities, num_relations, device)
    weights_path = os.path.join(checkpoint_dir, weights_name)
    state_dict = normalize_loaded_state_dict(torch.load(weights_path, map_location=device))
    unwrap_model(model).load_state_dict(state_dict, strict=True)

    split_name = split.lower()
    if split_name == "train":
        eval_triples = triples if not cfg.use_reciprocal else [t for t in triples if t[0] < num_relations // 2]
    elif split_name == "valid":
        eval_triples = valid_triples
    elif split_name == "test":
        eval_triples = test_triples
    else:
        raise ValueError(f"Unknown split '{split}'. Use train|valid|test.")
    if not eval_triples:
        raise ValueError(f"No triples available for split '{split_name}'")
    if eval_limit > 0 and len(eval_triples) > eval_limit:
        print(f"Profiling/evaluation limit: using first {eval_limit} of {len(eval_triples)} {split_name} triples")
        eval_triples = eval_triples[:eval_limit]

    if cfg.compile and compile_warmup_steps > 0:
        warmup_queries = min(len(eval_triples), max(1, compile_warmup_steps))
        print(f"Eval compile warmup: running {warmup_queries} query step(s) before timed evaluation")
        _ = evaluate_ranking(
            model,
            eval_triples[:warmup_queries],
            num_entities,
            head_filter,
            tail_filter,
            device,
            cfg.eval_chunk_size,
            head_domain=head_domain,
            tail_domain=tail_domain,
            eval_num_corruptions=cfg.eval_num_corruptions,
            seed=cfg.seed,
            show_progress=False,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()

    print(f"Evaluating {split_name} split from {weights_path} ...")
    metrics = evaluate_ranking(
        model,
        eval_triples,
        num_entities,
        head_filter,
        tail_filter,
        device,
        cfg.eval_chunk_size,
        head_domain=head_domain,
        tail_domain=tail_domain,
        eval_num_corruptions=cfg.eval_num_corruptions,
        seed=cfg.seed,
        show_progress=show_progress,
        progress_label=f"{split_name.capitalize()} eval",
    )
    print(
        f"{split_name.capitalize()} metrics | "
        f"mrr={metrics['MRR']:.4f} "
        f"h1={metrics['Hits@1']:.4f} "
        f"h3={metrics['Hits@3']:.4f} "
        f"h10={metrics['Hits@10']:.4f}"
    )
    return metrics


__all__ = ["evaluate_checkpoint", "evaluate_ranking"]
