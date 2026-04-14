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
    encode_split_triples,
    load_triples_with_mappings,
)
from ..data.paths import resolve_split_path, resolve_train_path
from ..eval.checkpoint import evaluate_ranking
from ..losses import NSSALoss
from ..models.factory import build_training_model
from ..scoring import Sampler as _KGESampler
from ..scoring import compute_bernoulli_probs
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
    seed: int,
) -> Tuple[Tensor, Tensor]:
    """Build a fixed pool of (positive, corruption) pairs like torch-ns.

    Returns:
        items: ``[N * (1 + neg_ratio), 3]`` tensor of (r, h, t) triples.
        labels: ``[N * (1 + neg_ratio)]`` tensor of 0/1 labels.
    """
    gen = torch.Generator().manual_seed(seed)
    pos = torch.tensor(train_triples, dtype=torch.long)
    N = pos.shape[0]
    items_list = []
    labels_list = []
    for i in range(N):
        items_list.append(pos[i])
        labels_list.append(1.0)
        for _ in range(neg_ratio):
            neg = pos[i].clone()
            if corrupt_mode == "tail":
                neg[2] = torch.randint(num_entities, (1,), generator=gen).item()
            elif corrupt_mode == "head":
                neg[1] = torch.randint(num_entities, (1,), generator=gen).item()
            else:  # both
                if torch.rand(1, generator=gen).item() < 0.5:
                    neg[2] = torch.randint(num_entities, (1,), generator=gen).item()
                else:
                    neg[1] = torch.randint(num_entities, (1,), generator=gen).item()
            items_list.append(neg)
            labels_list.append(0.0)
    return torch.stack(items_list), torch.tensor(labels_list)


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
    print(f"Loading triples from {train_file} ...")
    train_triples, e2id, r2id = load_triples_with_mappings(train_file)
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
    use_domain_eval = bool(cfg.dataset and "countries" in cfg.dataset.lower())
    head_domain, tail_domain = (build_relation_domains(train_triples + valid_triples + test_triples) if use_domain_eval else (None, None))

    if cfg.use_reciprocal:
        train_triples, r2id, num_relations = add_reciprocal_triples(train_triples, r2id, inv_suffix="__inv")
        print(f"Reciprocal relations enabled: new #relations={num_relations:,}, #train triples={len(train_triples):,}")
    else:
        num_relations = len(r2id)

    corrupt_mode = cfg.corruption_scheme if cfg.corruption_scheme != "both" else "bernoulli"

    # ── Model, optimizer, scheduler ───────────────────────────────────
    model = build_training_model(cfg, num_entities, num_relations, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)

    plateau_scheduler = None
    if cfg.scheduler == "plateau":
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-4,
        )

    set_seed(cfg.seed)
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

    def _run_validation(epoch: int, mdl: torch.nn.Module) -> bool:
        """Run validation, update best model, check early stopping. Returns True to stop."""
        nonlocal best_valid_mrr, best_valid_epoch, best_state_dict, no_improve_evals, stopped_early
        should_run = (
            valid_eval_every > 0 and valid_eval_triples
            and (epoch % valid_eval_every == 0 or epoch == cfg.epochs)
        )
        if not should_run:
            return False

        # Within-batch validation: score each positive against 1 corruption
        # Matches torch-ns validation protocol for stable epoch selection
        mdl.eval()
        with torch.no_grad():
            val_pos = torch.tensor(valid_eval_triples, dtype=torch.long, device=device)
            val_neg = val_pos.clone()
            if cfg.corruption_scheme == "head":
                val_neg[:, 1] = torch.randint(num_entities, (val_neg.shape[0],), device=device)
            else:
                val_neg[:, 2] = torch.randint(num_entities, (val_neg.shape[0],), device=device)
            pos_s = torch.sigmoid(mdl(val_pos[:, 1], val_pos[:, 0], val_pos[:, 2]))
            neg_s = torch.sigmoid(mdl(val_neg[:, 1], val_neg[:, 0], val_neg[:, 2]))
            current_valid_mrr = float((pos_s > neg_s).float().mean())
        mdl.train()

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
        # BCE training with complete query groups:
        # each batch = positives + their fresh negatives scored together.
        known_sampling_triples = _build_known_sampling_triples(
            train_triples=train_triples, valid_triples=valid_triples,
            test_triples=test_triples, use_reciprocal=cfg.use_reciprocal,
            num_relations_orig=num_relations_orig,
        )
        sampler = _KGESampler.from_data(
            all_known_triples_idx=torch.tensor(known_sampling_triples, dtype=torch.long),
            num_entities=num_entities, num_relations=num_relations,
            device=device, default_mode=corrupt_mode,
            bern_probs=compute_bernoulli_probs(
                torch.tensor(train_triples, dtype=torch.long), num_relations,
            ),
            min_entity_idx=0,
        )
        dataloader = DataLoader(
            TripleDataset(train_triples), batch_size=cfg.batch_size,
            shuffle=True, num_workers=cfg.num_workers,
            pin_memory=(device.type == "cuda"), drop_last=False,
        )

        model.train()
        for epoch in range(1, cfg.epochs + 1):
            epoch_start = time.perf_counter()
            running = 0.0
            n_batches = 0
            for batch in dataloader:
                batch = batch.to(device, non_blocking=True)
                # Fresh negatives each batch, filtered
                neg, _ = sampler.corrupt_with_mask(
                    batch, num_negatives=cfg.neg_ratio,
                    mode=corrupt_mode, filter=True, unique=False,
                )
                neg = neg.reshape(-1, 3)
                # Complete query group: score pos + neg together
                all_items = torch.cat([batch, neg], dim=0)
                labels = torch.cat([
                    torch.ones(batch.shape[0], device=device),
                    torch.zeros(neg.shape[0], device=device),
                ])
                optimizer.zero_grad(set_to_none=True)
                scores = model(all_items[:, 1], all_items[:, 0], all_items[:, 2])
                loss = F.binary_cross_entropy_with_logits(scores, labels)
                loss.backward()
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
        train_rht = torch.tensor(train_triples, dtype=torch.long)
        bern_probs = compute_bernoulli_probs(train_rht, num_relations)
        sampler = _KGESampler.from_data(
            all_known_triples_idx=torch.tensor(known_sampling_triples, dtype=torch.long),
            num_entities=num_entities, num_relations=num_relations,
            device=device, default_mode="bernoulli",
            bern_probs=bern_probs, min_entity_idx=0,
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
                mode=corrupt_mode, filter=True, unique=False,
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
