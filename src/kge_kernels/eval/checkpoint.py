"""Convenience evaluation entrypoints.

``evaluate_ranking``    — evaluate a model on triples using the unified Evaluator
``evaluate_checkpoint`` — load a saved checkpoint and evaluate a split
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Sequence, Set, Tuple

import torch

from .evaluator import Evaluator


def evaluate_ranking(
    model: torch.nn.Module,
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
) -> Dict[str, float]:
    """Thin wrapper around :class:`Evaluator` for exhaustive filtered ranking.

    ``eval_batch_size`` and ``progress_label`` are accepted for backward
    compatibility but not used.
    """
    del eval_batch_size, progress_label
    evaluator = Evaluator(
        model,
        num_entities,
        num_corruptions=0,
        head_filter=head_filter,
        tail_filter=tail_filter,
        head_domain=head_domain,
        tail_domain=tail_domain,
        batch_size=chunk_size,
        seed=seed,
        device=device,
    )
    triples_tensor = torch.tensor(triples, dtype=torch.long, device=device)
    return evaluator.evaluate(triples_tensor, show_progress=show_progress)


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
    from ..data.paths import resolve_split_path, resolve_train_path
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
    triples, e2id, r2id = load_triples_with_mappings(train_file)
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
