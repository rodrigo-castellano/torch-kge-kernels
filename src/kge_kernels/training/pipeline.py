"""Full KGE training pipeline.

``train_model`` orchestrates the complete lifecycle:
  data loading → model construction → training (via ``train_kge``) →
  validation / early stopping → final evaluation → checkpoint saving.

This is the function that both DpRL and standalone scripts should call
for KGE experiments. The lean ``train_kge`` inner loop remains available
for consumers that need custom orchestration.
"""
from __future__ import annotations

import math
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from ..data import (
    add_reciprocal_triples,
    build_filter_maps,
    build_relation_domains,
    build_relation_domains_from_file,
    encode_split_triples,
    load_domain_file,
    load_triples_with_mappings,
)
from ..data.paths import resolve_split_path, resolve_train_path
from ..eval.checkpoint import evaluate_ranking
from ..losses import NSSALoss
from ..models.factory import build_training_model
from ..scoring import Sampler as _KGESampler
from ..scoring import compute_bernoulli_probs
from .batching import iterate_epoch_batches
from .checkpoints import (
    build_config_payload,
    save_best_checkpoint,
    save_final_checkpoint,
    save_latest_weights,
)
from .checkpoints import (
    unwrap_model as _unwrap,
)
from .config import TrainArtifacts, TrainConfig
from .loop import (
    TripleDataset,
    make_cosine_warmup_scheduler,
    set_seed,
    train_kge,
)


def _limit_eval_triples(
    name: str,
    triples: List[Tuple[int, int, int]],
    limit: Optional[int],
) -> List[Tuple[int, int, int]]:
    if not triples:
        return []
    if limit and len(triples) > limit:
        print(f"Evaluation on {name}: limiting to first {limit} of {len(triples)} triples")
        return triples[:limit]
    return triples


def _current_state_dict(model: torch.nn.Module) -> Dict[str, Tensor]:
    return _unwrap(model).state_dict()


def _build_known_sampling_triples(
    *,
    train_triples: List[Tuple[int, int, int]],
    valid_triples: List[Tuple[int, int, int]],
    test_triples: List[Tuple[int, int, int]],
    use_reciprocal: bool,
    num_relations_orig: int,
) -> List[Tuple[int, int, int]]:
    if not use_reciprocal:
        return list(train_triples) + list(valid_triples) + list(test_triples)

    known = list(train_triples)
    if valid_triples:
        known.extend(valid_triples)
        known.extend((r + num_relations_orig, t, h) for r, h, t in valid_triples)
    if test_triples:
        known.extend(test_triples)
        known.extend((r + num_relations_orig, t, h) for r, h, t in test_triples)
    return known


def _build_interleaved_pool(
    train_triples: List[Tuple[int, int, int]],
    num_entities: int,
    neg_ratio: int,
    corrupt_mode: str,
    sampler: "_KGESampler",
) -> Tuple[Tensor, Tensor]:
    """Build a fixed pool of (positive, filtered corruption) pairs.

    Uses the sampler with ``filter=True`` to exclude known positives.

    Returns:
        items: ``[~N * (1 + neg_ratio), 3]`` tensor of (r, h, t) triples.
        labels: ``[~N * (1 + neg_ratio)]`` tensor of 0/1 labels.
    """
    pos = torch.tensor(train_triples, dtype=torch.long, device=sampler.device)
    neg, valid_mask = sampler.corrupt_with_mask(
        pos, num_negatives=neg_ratio, mode=corrupt_mode,
        filter=True, unique=False,
    )
    # Interleave: [pos0, neg0_0, ..., neg0_K, pos1, neg1_0, ...]
    N = pos.shape[0]
    K = neg.shape[1]  # neg_ratio
    items_list: List[Tensor] = []
    labels_list: List[float] = []
    for i in range(N):
        items_list.append(pos[i])
        labels_list.append(1.0)
        for k in range(K):
            if valid_mask[i, k]:
                items_list.append(neg[i, k])
                labels_list.append(0.0)
    return torch.stack(items_list), torch.tensor(labels_list, device=sampler.device)


def _build_metric_summary(prefix: str, rank_metrics: Dict[str, float], metrics: Dict[str, float], logs: List[str]) -> None:
    if any(math.isnan(value) for value in rank_metrics.values()):
        return
    metrics[f"{prefix}_mrr"] = float(rank_metrics["MRR"])
    metrics[f"{prefix}_hits1"] = float(rank_metrics["Hits@1"])
    metrics[f"{prefix}_hits3"] = float(rank_metrics["Hits@3"])
    metrics[f"{prefix}_hits10"] = float(rank_metrics["Hits@10"])
    logs.append(
        f"{prefix} mrr={rank_metrics['MRR']:.4f} "
        f"h1={rank_metrics['Hits@1']:.4f} "
        f"h3={rank_metrics['Hits@3']:.4f} "
        f"h10={rank_metrics['Hits@10']:.4f}"
    )


def train_model(cfg: TrainConfig) -> TrainArtifacts:
    """Full KGE training pipeline.

    Loads data, builds model, trains via ``train_kge`` with validation /
    early stopping / checkpoint management in the ``on_epoch_end``
    callback, then runs final evaluation and saves the checkpoint.

    Args:
        cfg: Complete training configuration.

    Returns:
        ``TrainArtifacts`` with entity/relation mappings, checkpoint
        paths, and final metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.cpu else "cpu")

    # ── Data loading ──────────────────────────────────────────────────
    train_file = resolve_train_path(cfg.train_path, cfg.dataset, cfg.data_root, cfg.train_split)
    # Resolve valid/test paths upfront so entity/relation IDs are sorted
    # alphabetically across all splits (matching torch-ns's read_ontology).
    _valid_p = resolve_split_path(
        split_name="valid", explicit_path=cfg.valid_path,
        dataset=cfg.dataset, data_root=cfg.data_root,
        split_filename=cfg.valid_split,
    )
    _test_p = resolve_split_path(
        split_name="test", explicit_path=cfg.test_path,
        dataset=cfg.dataset, data_root=cfg.data_root,
        split_filename=cfg.test_split,
    )
    print(f"Loading triples from {train_file} ...")
    train_triples, e2id, r2id = load_triples_with_mappings(
        train_file, extra_paths=[p for p in (_valid_p, _test_p) if p],
    )
    if not train_triples:
        raise ValueError("No triples found for training")
    print(f"#entities={len(e2id):,}, #relations={len(r2id):,}, #train triples={len(train_triples):,}")

    valid_triples: List[Tuple[int, int, int]] = []
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

    test_triples: List[Tuple[int, int, int]] = []
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
    num_relations_orig = len(r2id)
    head_filter, tail_filter = build_filter_maps(train_triples, valid_triples, test_triples)

    # Domain-restricted evaluation and corruption
    domain_path = cfg.domain_file
    if domain_path is None and cfg.dataset:
        candidate = os.path.join(cfg.data_root, cfg.dataset, "domain2constants.txt")
        if os.path.isfile(candidate):
            domain_path = candidate
    domain2idx, entity2domain = (
        load_domain_file(domain_path, e2id) if domain_path else ({}, {})
    )
    use_domain_eval = bool(domain2idx)
    if use_domain_eval:
        # Use the full domain-file memberships so that eval ranks against
        # ALL entities declared in domain2constants.txt, not just those
        # observed in the data.  This matches ns-old's _IndexedCorruptionAdapter
        # which samples from the full domain pool.
        head_domain, tail_domain = build_relation_domains_from_file(
            train_triples + valid_triples + test_triples,
            entity2domain, domain2idx,
        )
    else:
        head_domain, tail_domain = None, None

    if cfg.use_reciprocal:
        train_triples, r2id, num_relations = add_reciprocal_triples(train_triples, r2id, inv_suffix="__inv")
        print(f"Reciprocal relations enabled: new #relations={num_relations:,}, #train triples={len(train_triples):,}")
    else:
        num_relations = len(r2id)

    # corruption_scheme names the PREDICTION direction (which entity to rank
    # at eval time). Training must corrupt that SAME column so the model
    # learns to distinguish the true entity from corrupted ones.
    # The Sampler's mode names which column to REPLACE, so they match 1:1.
    _scheme_to_mode = {"tail": "head", "head": "tail", "both": "bernoulli"}
    corrupt_mode = _scheme_to_mode.get(cfg.corruption_scheme, "bernoulli")

    # ── Deterministic initialization ────────────────────────────────
    set_seed(cfg.seed)

    # ── Model, optimizer, scheduler ───────────────────────────────────
    model = build_training_model(cfg, num_entities, num_relations, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, eps=1e-7)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)

    plateau_scheduler = None
    if cfg.scheduler == "plateau":
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-4,
        )

    os.makedirs(cfg.save_dir, exist_ok=True)

    # ── Validation state ──────────────────────────────────────────────
    eval_limit = cfg.eval_limit if cfg.eval_limit > 0 else None
    valid_eval_limit = cfg.valid_eval_queries if cfg.valid_eval_queries > 0 else eval_limit
    valid_eval_triples = _limit_eval_triples("validation", valid_triples, valid_eval_limit)
    valid_eval_every = cfg.valid_eval_every

    epoch_durations: List[float] = []
    validation_history: List[Dict[str, float]] = []
    best_valid_mrr = float("-inf")
    best_valid_epoch = 0
    best_state_dict: Optional[Dict[str, Tensor]] = None
    no_improve_evals = 0
    stopped_early = False
    epochs_completed = 0

    def make_payload(metrics_payload: Optional[Dict[str, float]] = None) -> Dict[str, object]:
        return build_config_payload(
            cfg,
            num_entities=num_entities,
            num_relations=num_relations,
            metrics_payload=metrics_payload,
            validation_history_payload=validation_history,
            best_valid_mrr_payload=None if best_valid_epoch == 0 else best_valid_mrr,
            best_valid_epoch_payload=best_valid_epoch,
            stopped_early_payload=stopped_early,
            epochs_completed_payload=epochs_completed,
        )

    def _exhaustive_valid_mrr(mdl: torch.nn.Module, valid_t: List[Tuple[int, int, int]]) -> float:
        """Exhaustive filtered validation MRR."""
        valid_metrics = evaluate_ranking(
            mdl, valid_t, num_entities,
            head_filter, tail_filter, device, cfg.eval_chunk_size,
            head_domain=head_domain, tail_domain=tail_domain,
            corruption_scheme=cfg.corruption_scheme,
        )
        return float(valid_metrics["MRR"])

    def _run_validation(epoch: int, mdl: torch.nn.Module) -> bool:
        """Run validation, update best model, check early stopping. Returns True to stop."""
        nonlocal best_valid_mrr, best_valid_epoch, best_state_dict, no_improve_evals, stopped_early
        should_run = (
            valid_eval_every > 0 and valid_eval_triples
            and (epoch % valid_eval_every == 0 or epoch == cfg.epochs)
        )
        if not should_run:
            return False

        if cfg.loss == "bce" and use_domain_eval:
            current_valid_mrr = _exhaustive_valid_mrr(mdl, valid_eval_triples)
        else:
            # Exhaustive filtered ranking
            valid_metrics = evaluate_ranking(
                mdl, valid_eval_triples, num_entities,
                head_filter, tail_filter, device, cfg.eval_chunk_size,
                head_domain=head_domain, tail_domain=tail_domain,
                corruption_scheme=cfg.corruption_scheme,
            )
            current_valid_mrr = float(valid_metrics["MRR"])

        if current_valid_mrr > best_valid_mrr:
            best_valid_mrr = current_valid_mrr
            best_valid_epoch = epoch
            best_state_dict = {k: v.detach().cpu().clone() for k, v in _current_state_dict(mdl).items()}
            no_improve_evals = 0
            save_best_checkpoint(cfg.save_dir, state_dict=_current_state_dict(mdl),
                                 config_payload=make_payload({"valid_mrr": current_valid_mrr}))
        else:
            no_improve_evals += 1
            if cfg.use_early_stopping and no_improve_evals >= cfg.patience:
                stopped_early = True
                return True
        return False

    # ── Train (BCE interleaved or NSSA) ───────────────────────────────
    train_start = time.perf_counter()

    if cfg.loss == "bce":
        # BCE training with complete query groups.
        known_sampling_triples = _build_known_sampling_triples(
            train_triples=train_triples, valid_triples=valid_triples,
            test_triples=test_triples, use_reciprocal=cfg.use_reciprocal,
            num_relations_orig=num_relations_orig,
        )
        # Training sampler: NO domain restriction.  ns-old's
        # _IndexedCorruptionAdapter *does* domain-restrict training
        # negatives, but tkk uses BCE-with-logits whose gradient signal
        # stays strong even for extreme logits.  Combined with a tiny
        # domain pool (e.g. 5 entities for ablation/countries "regions")
        # this collapses training — the model overfits the handful of
        # same-domain negatives but never learns to rank unseen entities.
        # Domain restriction is applied only at eval time (head_domain /
        # tail_domain in the Evaluator).
        sampler = _KGESampler.from_data(
            all_known_triples_idx=torch.tensor(known_sampling_triples, dtype=torch.long),
            num_entities=num_entities, num_relations=num_relations,
            device=device, default_mode=corrupt_mode,
            bern_probs=compute_bernoulli_probs(
                torch.tensor(train_triples, dtype=torch.long), num_relations,
            ),
            min_entity_idx=0,
        )

        # Pre-allocate training triples on device for bare-tensor loop
        # (no DataLoader / Python iteration overhead).
        train_pos = torch.tensor(train_triples, dtype=torch.long, device=device)
        # Corruption modes for training
        if cfg.corruption_scheme == "both":
            _train_corrupt_modes = ["head", "tail"]
        else:
            _train_corrupt_modes = [corrupt_mode]

        # RotatE needs sigmoid+BCE (matching ns-old's output_layer which
        # applies sigmoid inside the model).  BCE_with_logits is numerically
        # equivalent but the different gradient scale prevents RotatE from
        # learning on large KGs (wn18rr).  ComplEx works with either.
        _use_sigmoid = cfg.model.lower() == "rotate"
        def _bce_fn(scores: Tensor, labels: Tensor) -> Tensor:
            if _use_sigmoid:
                return F.binary_cross_entropy(torch.sigmoid(scores), labels)
            return F.binary_cross_entropy_with_logits(scores, labels)

        model.train()
        for epoch in range(1, cfg.epochs + 1):
            epoch_start = time.perf_counter()

            running = 0.0
            n_batches = 0
            for pos, neg_batch, mask_batch in iterate_epoch_batches(
                train_pos, sampler,
                batch_size=cfg.batch_size,
                num_negatives=cfg.neg_ratio,
                corrupt_modes=_train_corrupt_modes,
                filter=True, unique=False,
            ):
                B = pos.shape[0]
                neg_valid = neg_batch[mask_batch]      # [M, 3]

                all_items = torch.cat([pos, neg_valid], dim=0)
                labels = torch.cat([
                    torch.ones(B, device=device),
                    torch.zeros(neg_valid.shape[0], device=device),
                ])
                optimizer.zero_grad(set_to_none=True)
                scores = model(all_items[:, 1], all_items[:, 0], all_items[:, 2])
                loss = _bce_fn(scores, labels)
                loss.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                running += loss.item()
                n_batches += 1

            epochs_completed = epoch
            epoch_time = time.perf_counter() - epoch_start
            epoch_durations.append(epoch_time)
            avg_loss = running / max(1, n_batches)
            print(f"Epoch {epoch:03d} | loss={avg_loss:.4f} | time={epoch_time:.2f}s", end="\r")
            save_latest_weights(cfg.save_dir, _current_state_dict(model))

            if plateau_scheduler is not None:
                plateau_scheduler.step(avg_loss)

            if _run_validation(epoch, model):
                break
    else:
        # NSSA training via train_kge
        known_sampling_triples = _build_known_sampling_triples(
            train_triples=train_triples, valid_triples=valid_triples,
            test_triples=test_triples, use_reciprocal=cfg.use_reciprocal,
            num_relations_orig=num_relations_orig,
        )
        sampler = _KGESampler.from_data(
            all_known_triples_idx=torch.tensor(known_sampling_triples, dtype=torch.long),
            num_entities=num_entities, num_relations=num_relations,
            device=device, default_mode="bernoulli",
            bern_probs=compute_bernoulli_probs(
                torch.tensor(train_triples, dtype=torch.long), num_relations,
            ),
            min_entity_idx=0,
            domain2idx=domain2idx or None,
            entity2domain=entity2domain or None,
        )
        dataloader = DataLoader(
            TripleDataset(train_triples), batch_size=cfg.batch_size,
            shuffle=True, num_workers=cfg.num_workers,
            pin_memory=(device.type == "cuda"), drop_last=False,
        )
        scheduler = None
        if cfg.scheduler == "cosine":
            total_steps = max(1, cfg.epochs * len(dataloader))
            scheduler = make_cosine_warmup_scheduler(
                optimizer, total_steps=total_steps, warmup_ratio=cfg.warmup_ratio,
            )
        loss_fn = NSSALoss(adv_temp=cfg.adv_temp, neg_ratio=cfg.neg_ratio)

        @torch.no_grad()
        def sample_negatives(batch: Tensor) -> Tensor:
            neg_rht, _ = sampler.corrupt_with_mask(
                batch, num_negatives=cfg.neg_ratio,
                mode=corrupt_mode, filter=False, unique=False,
            )
            return neg_rht.reshape(-1, 3)

        def on_epoch_end(epoch: int, avg_loss: float, mdl: torch.nn.Module, epoch_time: float) -> bool:
            nonlocal epochs_completed
            epochs_completed = epoch
            epoch_durations.append(epoch_time)
            print(f"Epoch {epoch:03d} | loss={avg_loss:.4f} | time={epoch_time:.2f}s", end="\r")
            save_latest_weights(cfg.save_dir, _current_state_dict(mdl))
            if plateau_scheduler is not None:
                plateau_scheduler.step(avg_loss)
            return _run_validation(epoch, mdl)

        train_kge(
            cfg, model, dataloader,
            optimizer=optimizer, loss_fn=loss_fn,
            sample_negatives=sample_negatives,
            scheduler=scheduler, scaler=scaler,
            device=device, on_epoch_end=on_epoch_end,
        )

    total_train_time = time.perf_counter() - train_start
    if epoch_durations:
        print(f"\nTraining time | epochs={len(epoch_durations)} | total={total_train_time:.2f}s | avg_per_epoch={sum(epoch_durations) / len(epoch_durations):.2f}s")

    # ── Restore best model ────────────────────────────────────────────
    if best_state_dict is not None:
        print(f"Restoring best validation model from epoch {best_valid_epoch:03d} (MRR={best_valid_mrr:.4f})")
        _unwrap(model).load_state_dict(best_state_dict)

    # ── Final evaluation ──────────────────────────────────────────────
    metrics: Dict[str, float] = {}
    metric_logs: List[str] = []

    if cfg.report_train_mrr:
        train_eval_triples = _limit_eval_triples(
            "train",
            train_triples if not cfg.use_reciprocal else [t for t in train_triples if t[0] < num_relations // 2],
            eval_limit,
        )
        if train_eval_triples:
            _build_metric_summary(
                "train",
                evaluate_ranking(
                    model, train_eval_triples, num_entities,
                    head_filter, tail_filter, device, cfg.eval_chunk_size,
                    head_domain=head_domain, tail_domain=tail_domain,
                    eval_num_corruptions=cfg.eval_num_corruptions, seed=cfg.seed,
                    corruption_scheme=cfg.corruption_scheme,
                    sampler=sampler if cfg.eval_num_corruptions > 0 else None,
                ),
                metrics, metric_logs,
            )

    if valid_triples and valid_eval_triples:
        print("Computing validation metrics ...")
        _build_metric_summary(
            "valid",
            evaluate_ranking(
                model, valid_eval_triples, num_entities,
                head_filter, tail_filter, device, cfg.eval_chunk_size,
                head_domain=head_domain, tail_domain=tail_domain,
                eval_num_corruptions=cfg.eval_num_corruptions, seed=cfg.seed,
                corruption_scheme=cfg.corruption_scheme,
                sampler=sampler if cfg.eval_num_corruptions > 0 else None,
            ),
            metrics, metric_logs,
        )

    if test_triples:
        print("Computing test metrics ...")
        _build_metric_summary(
            "test",
            evaluate_ranking(
                model, test_triples, num_entities,
                head_filter, tail_filter, device, cfg.eval_chunk_size,
                head_domain=head_domain, tail_domain=tail_domain,
                eval_num_corruptions=cfg.eval_num_corruptions, seed=cfg.seed,
                show_progress=True, progress_label="Test eval",
                corruption_scheme=cfg.corruption_scheme,
                sampler=sampler if cfg.eval_num_corruptions > 0 else None,
            ),
            metrics, metric_logs,
        )

    if metric_logs:
        print("Evaluation | " + " | ".join(metric_logs))

    # ── Save final checkpoint ─────────────────────────────────────────
    weights_path, config_path = save_final_checkpoint(
        cfg.save_dir,
        state_dict=_current_state_dict(model),
        config_payload=make_payload(metrics),
        entity2id=e2id,
        relation2id=r2id,
    )
    print(f"Saved model to {cfg.save_dir}")
    return TrainArtifacts(
        entity2id=e2id,
        relation2id=r2id,
        config_path=config_path,
        weights_path=weights_path,
        metrics=metrics or None,
    )


__all__ = ["train_model"]
