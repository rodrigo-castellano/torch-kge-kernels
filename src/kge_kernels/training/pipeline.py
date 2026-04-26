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
from ..data import resolve_split_path, resolve_train_path
from ..losses import NSSALoss
from ..models.factory import build_training_model
from ..scoring import BernoulliSampler as _BernoulliKGESampler
from ..scoring import Sampler as _KGESampler
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


def _build_ranking_evaluator(
    model: torch.nn.Module,
    sampler,
    cfg: TrainConfig,
    num_entities: int,
    *,
    head_domain,
    tail_domain,
    use_domain_eval: bool,
    device: torch.device,
):
    """Build the cached RankingEvaluator. Called once after the
    training sampler is built; reused for every validation pass and
    final eval. Compile cache lives on this instance."""
    from ..eval.candidates import SamplerCandidates
    from ..eval.eval_hooks import kge_default_scorer, recommended_eval_batch_size
    from ..eval.ranking_evaluator import RankingEvaluator

    eval_negs = getattr(cfg, "eval_num_corruptions", 0) or None
    candidates = SamplerCandidates(
        sampler, k=eval_negs,
        head_domain=head_domain if use_domain_eval else None,
        tail_domain=tail_domain if use_domain_eval else None,
    )
    scheme = cfg.corruption_scheme if cfg.corruption_scheme in ("head", "tail", "both") else "both"
    modes = ("head", "tail") if scheme == "both" else (scheme,)
    batch_size = recommended_eval_batch_size(model, num_entities)
    scorer = lambda q, p, m: kge_default_scorer(model, q, p, m)
    return RankingEvaluator(
        scorer=scorer,
        candidates=candidates,
        batch_size=batch_size,
        modes=modes,
        device=device,
        compile=True,
        tie_handling="average",
        seed=0,
    )


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
    # tkk standalone uses a dense 0-based id space (no padding sentinel) —
    # its models, sampler, and evaluator size everything to exactly
    # ``num_entities`` / ``num_relations``. ns / DpRL consume the same
    # data via ``KnowledgeBase`` with ``padding_idx=0`` (id 0 reserved)
    # so the shift convention lives in one parameter at the loader.
    train_triples, e2id, r2id = load_triples_with_mappings(
        train_file, extra_paths=[p for p in (_valid_p, _test_p) if p],
        padding_idx=None,
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

    # Disk persistence is opt-in. The shared logging contract
    # (kge_kernels.logging.run_experiment) wraps train_model with a
    # RunContext that handles model saving via output/runs/<exp>/<run>/.
    # Standalone train_model() calls (e.g. from sweep scripts) default to
    # save_dir=None which skips all disk writes.
    if cfg.save_dir is not None:
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

    # The RankingEvaluator is instantiated once after the sampler is
    # built (see the loss-branch below) and reused across every
    # validation pass + the final eval — its compiled scorers and
    # static buffers persist for the training run's lifetime.
    _evaluator = [None]  # boxed so inner functions can see the assignment

    def _unified_valid_mrr(mdl: torch.nn.Module,
                           valid_t: List[Tuple[int, int, int]]) -> float:
        """Validation MRR via the cached RankingEvaluator."""
        ev = _evaluator[0]
        assert ev is not None, "RankingEvaluator must be built before validation"
        triples_tensor = torch.tensor(valid_t, dtype=torch.long, device=device)
        result = ev.evaluate(triples_tensor)
        return float(result.metrics()["MRR"])

    def _run_validation(epoch: int, mdl: torch.nn.Module) -> bool:
        """Run validation, update best model, check early stopping. Returns True to stop."""
        nonlocal best_valid_mrr, best_valid_epoch, best_state_dict, no_improve_evals, stopped_early
        should_run = (
            valid_eval_every > 0 and valid_eval_triples
            and (epoch % valid_eval_every == 0 or epoch == cfg.epochs)
        )
        if not should_run:
            return False

        current_valid_mrr = _unified_valid_mrr(mdl, valid_eval_triples)

        if current_valid_mrr > best_valid_mrr:
            best_valid_mrr = current_valid_mrr
            best_valid_epoch = epoch
            best_state_dict = {k: v.detach().cpu().clone() for k, v in _current_state_dict(mdl).items()}
            no_improve_evals = 0
            if cfg.save_dir is not None:
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
        if corrupt_mode == "bernoulli":
            sampler = _BernoulliKGESampler.from_data(
                all_known_triples_idx=torch.tensor(known_sampling_triples, dtype=torch.long),
                num_entities=num_entities, num_relations=num_relations,
                device=device,
                bern_probs=_BernoulliKGESampler.compute_probs(
                    torch.tensor(train_triples, dtype=torch.long), num_relations,
                ),
                min_entity_idx=0,
            )
        else:
            sampler = _KGESampler.from_data(
                all_known_triples_idx=torch.tensor(known_sampling_triples, dtype=torch.long),
                num_entities=num_entities, num_relations=num_relations,
                device=device, default_mode=corrupt_mode,
                min_entity_idx=0,
            )
        # Build the cached RankingEvaluator now that the sampler is in
        # hand. The sampler's filter hashes were built from train ∪
        # valid ∪ test positives, so known positives are excluded
        # correctly from the candidate pool.
        _evaluator[0] = _build_ranking_evaluator(
            model, sampler, cfg, num_entities,
            head_domain=head_domain, tail_domain=tail_domain,
            use_domain_eval=use_domain_eval, device=device,
        )

        # Pre-allocate training triples on device for bare-tensor loop
        # (no DataLoader / Python iteration overhead).
        train_pos = torch.tensor(train_triples, dtype=torch.long, device=device)
        # Corruption modes for training
        if cfg.corruption_scheme == "both":
            _train_corrupt_modes = ["head", "tail"]
        else:
            _train_corrupt_modes = [corrupt_mode]

        # Sigmoid-vs-logits is now baked into each model's
        # ``_train_loss_is_from_logits`` class flag (RotatE sets it False
        # because its raw scores saturate BCE_with_logits on large KGs;
        # ComplEx / DistMult / TransE / ... all use from_logits).
        # ``model.train_step`` consults this flag.
        from .epoch import train_epoch as _train_epoch

        model.train()
        for epoch in range(1, cfg.epochs + 1):
            epoch_start = time.perf_counter()
            losses = _train_epoch(
                model, sampler, optimizer, train_pos,
                batch_size=cfg.batch_size,
                num_negatives=cfg.neg_ratio,
                corrupt_modes=_train_corrupt_modes,
                grad_clip=cfg.grad_clip,
                filter_negatives=True, unique_negatives=False,
                compile=getattr(cfg, "compile", False),
            )
            avg_loss = losses["loss"]
            epochs_completed = epoch
            epoch_time = time.perf_counter() - epoch_start
            epoch_durations.append(epoch_time)
            print(f"Epoch {epoch:03d} | loss={avg_loss:.4f} | time={epoch_time:.2f}s", end="\r")
            if cfg.save_dir is not None:
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
        sampler = _BernoulliKGESampler.from_data(
            all_known_triples_idx=torch.tensor(known_sampling_triples, dtype=torch.long),
            num_entities=num_entities, num_relations=num_relations,
            device=device,
            bern_probs=_BernoulliKGESampler.compute_probs(
                torch.tensor(train_triples, dtype=torch.long), num_relations,
            ),
            min_entity_idx=0,
            domain2idx=domain2idx or None,
            entity2domain=entity2domain or None,
        )
        # Build the cached RankingEvaluator (same as BCE branch).
        _evaluator[0] = _build_ranking_evaluator(
            model, sampler, cfg, num_entities,
            head_domain=head_domain, tail_domain=tail_domain,
            use_domain_eval=use_domain_eval, device=device,
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
            neg_rht = sampler.corrupt(
                batch, num_negatives=cfg.neg_ratio,
                mode=corrupt_mode, filter=False, unique=False,
            )
            return neg_rht.reshape(-1, 3)

        def on_epoch_end(epoch: int, avg_loss: float, mdl: torch.nn.Module, epoch_time: float) -> bool:
            nonlocal epochs_completed
            epochs_completed = epoch
            epoch_durations.append(epoch_time)
            print(f"Epoch {epoch:03d} | loss={avg_loss:.4f} | time={epoch_time:.2f}s", end="\r")
            if cfg.save_dir is not None:
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

    # Final train / valid / test eval reuses the same RankingEvaluator
    # built during training — same compile cache, same static buffers.
    def _final_eval(split_label, triples):
        ev = _evaluator[0]
        assert ev is not None, "RankingEvaluator must be built before final eval"
        triples_t = torch.tensor(triples, dtype=torch.long, device=device)
        result = ev.evaluate(triples_t)
        return result.metrics()

    if cfg.report_train_mrr:
        train_eval_triples = _limit_eval_triples(
            "train",
            train_triples if not cfg.use_reciprocal else [t for t in train_triples if t[0] < num_relations // 2],
            eval_limit,
        )
        if train_eval_triples:
            _build_metric_summary("train", _final_eval("train", train_eval_triples),
                                  metrics, metric_logs)

    if valid_triples and valid_eval_triples:
        print("Computing validation metrics ...")
        _build_metric_summary("valid", _final_eval("valid", valid_eval_triples),
                              metrics, metric_logs)

    if test_triples:
        print("Computing test metrics ...")
        _build_metric_summary("test", _final_eval("test", test_triples),
                              metrics, metric_logs)

    if metric_logs:
        print("Evaluation | " + " | ".join(metric_logs))

    # ── Save final checkpoint (only when save_dir is set) ────────────
    if cfg.save_dir is not None:
        weights_path, config_path = save_final_checkpoint(
            cfg.save_dir,
            state_dict=_current_state_dict(model),
            config_payload=make_payload(metrics),
            entity2id=e2id,
            relation2id=r2id,
        )
        print(f"Saved model to {cfg.save_dir}")
    else:
        weights_path, config_path = "", ""
    return TrainArtifacts(
        entity2id=e2id,
        relation2id=r2id,
        config_path=config_path,
        weights_path=weights_path,
        metrics=metrics or None,
    )


__all__ = ["train_model"]
