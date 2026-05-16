"""tkk training builder — factory functions for KGE components.

Mirrors DpRL's ``kge_experiments/builder.py`` structure
(``build_env`` / ``build_policy`` / ``build_algorithm`` /
``build_callbacks`` / ``run_evaluation``) using KGE-natural names:
``build_data`` / ``build_model`` / ``build_optimizer`` /
``build_callbacks`` / ``build_evaluator`` / ``run_evaluation``.

The long-running training-loop body lives in
:mod:`kge_kernels.training.train`. The short pipeline glue lives in
:mod:`kge_kernels.training.experiment`.

Usage (from ``experiment.pipeline``)::

    set_seed(cfg.seed)
    data       = build_data(cfg)
    model      = build_model(cfg, data)
    optim      = build_optimizer(cfg, model)
    evaluator  = build_evaluator(cfg, model, data)
    callbacks  = build_callbacks(cfg, evaluator, data)

    train_m = train(model, data, optim, callbacks, evaluator, cfg) if cfg.epochs > 0 else {}
    eval_m  = run_evaluation(model, evaluator, data, callbacks, cfg)
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from ..data import KnowledgeBase
from ..models.factory import build_training_model
from .checkpoints import (
    build_config_payload,
    save_best_checkpoint,
    save_final_checkpoint,
    unwrap_model,
)
from .config import TrainConfig


# ─── DataBundle — holds everything the trainer needs from the data side ──

@dataclass
class DataBundle:
    """Single source of truth for "the world" — DpRL's ``EnvVec`` equivalent."""
    train_triples: List[Tuple[int, int, int]]
    valid_triples: List[Tuple[int, int, int]]
    test_triples: List[Tuple[int, int, int]]
    entity2id: Dict[str, int]
    relation2id: Dict[str, int]
    num_entities: int
    num_relations: int
    num_relations_orig: int
    sampler: Any                                  # Sampler | BernoulliSampler
    train_pos: Tensor                             # [N, 3] long, on GPU
    corrupt_mode: str                             # "head" | "tail" | "bernoulli"
    train_corrupt_modes: List[str]
    valid_eval_triples: List[Tuple[int, int, int]]
    head_domain: Optional[Any] = None
    tail_domain: Optional[Any] = None
    use_domain_eval: bool = False
    device: torch.device = torch.device("cpu")


# ─── OptimBundle — optimizer + scaler + plateau scheduler ───────────────

@dataclass
class OptimBundle:
    optimizer: torch.optim.Optimizer
    scaler: torch.amp.GradScaler
    plateau_scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = None


# ─── Callbacks — validation hook + early-stop + best-checkpoint state ───

@dataclass
class Callbacks:
    """Per-epoch validation, best-model bookkeeping, early stopping.

    Owns the run-level training state (best_*, validation_history,
    stopped_early, …) that ``train()`` mutates and ``run_evaluation``
    reads. Mirrors the role of DpRL's ``CallbackManager`` but bundles
    the state directly for the simpler tkk loop.
    """
    cfg: TrainConfig
    evaluator: Any                                # RankingEvaluator
    data: DataBundle
    # State mutated during training
    epochs_completed: int = 0
    epoch_durations: List[float] = field(default_factory=list)
    validation_history: List[Dict[str, float]] = field(default_factory=list)
    best_valid_mrr: float = float("-inf")
    best_valid_epoch: int = 0
    best_state_dict: Optional[Dict[str, Tensor]] = None
    no_improve_evals: int = 0
    stopped_early: bool = False

    def make_payload(self, metrics_payload: Optional[Dict[str, float]] = None) -> Dict[str, object]:
        """Serialize cfg + run-level state into a checkpoint payload dict."""
        return build_config_payload(
            self.cfg,
            num_entities=self.data.num_entities,
            num_relations=self.data.num_relations,
            metrics_payload=metrics_payload,
            validation_history_payload=self.validation_history,
            best_valid_mrr_payload=None if self.best_valid_epoch == 0 else self.best_valid_mrr,
            best_valid_epoch_payload=self.best_valid_epoch,
            stopped_early_payload=self.stopped_early,
            epochs_completed_payload=self.epochs_completed,
        )

    def on_epoch_end(self, epoch: int, model: torch.nn.Module, train_loss: float) -> bool:
        """Run validation, update best model, check early stop. Returns True to stop."""
        cfg = self.cfg
        valid_eval_every = cfg.valid_eval_every
        should_run = (
            valid_eval_every > 0 and self.data.valid_eval_triples
            and (epoch % valid_eval_every == 0 or epoch == cfg.epochs)
        )
        if not should_run:
            return False

        triples_tensor = torch.tensor(
            self.data.valid_eval_triples, dtype=torch.long, device=self.data.device,
        )
        result = self.evaluator.evaluate(triples_tensor)
        current_valid_mrr = float(result.metrics()["MRR"])

        if current_valid_mrr > self.best_valid_mrr:
            self.best_valid_mrr = current_valid_mrr
            self.best_valid_epoch = epoch
            sd = unwrap_model(model).state_dict()
            self.best_state_dict = {k: v.detach().cpu().clone() for k, v in sd.items()}
            self.no_improve_evals = 0
            if cfg.save_dir is not None:
                save_best_checkpoint(
                    cfg.save_dir,
                    state_dict=sd,
                    config_payload=self.make_payload({"valid_mrr": current_valid_mrr}),
                )
        else:
            self.no_improve_evals += 1
            if cfg.use_early_stopping and self.no_improve_evals >= cfg.patience:
                self.stopped_early = True
                return True
        return False


# ─── build_data — equivalent to DpRL.build_env ──────────────────────────

def build_data(cfg: TrainConfig) -> DataBundle:
    """Load triples, build id maps, instantiate sampler.

    Single ``KnowledgeBase(...)`` call covers file resolution, integer
    triple loading, alphabetical id maps, domain parsing, filter maps,
    and the ground-fact set. ``kb.build_sampler(...)`` then produces
    the corruption sampler (regular or Bernoulli, domain-restricted
    or not) from the same loaded state.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.cpu else "cpu")

    # Auto-discover domain file under ``<data_root>/<dataset>/domain2constants.txt``
    # when not explicitly set — preserves the legacy convention.
    domain_file = cfg.domain_file
    if domain_file is None and cfg.dataset:
        candidate = os.path.join(cfg.data_root, cfg.dataset, "domain2constants.txt")
        if os.path.isfile(candidate):
            domain_file = candidate

    print(f"Loading dataset {cfg.dataset!r} from {cfg.data_root} ...")
    kb = KnowledgeBase(
        dataset_name=cfg.dataset, base_path=cfg.data_root,
        train_file=cfg.train_split, valid_file=cfg.valid_split, test_file=cfg.test_split,
        fact_file=None,                       # tkk doesn't use a separate facts file
        domain_file=domain_file,
        padding_idx=None,                     # dense 0-based ids — tkk standalone convention
        valid_size=None,
        use_reciprocal=cfg.use_reciprocal,
    )
    if not kb.train_idx:
        raise ValueError("No triples found for training")
    print(f"#entities={kb.num_entities:,}, #relations={kb.num_relations:,}, "
          f"#train triples={len(kb.train_idx):,}")
    if cfg.use_reciprocal:
        print(f"Reciprocal relations enabled: orig_relations={kb.num_relations_orig:,}")

    # corruption_scheme names the PREDICTION direction (which entity to rank
    # at eval). Training corrupts that SAME column. Sampler mode names which
    # column to REPLACE, so they match 1:1.
    scheme_to_mode = {"tail": "head", "head": "tail", "both": "bernoulli"}
    corrupt_mode = scheme_to_mode.get(cfg.corruption_scheme, "bernoulli")
    train_corrupt_modes = (
        ["head", "tail"] if cfg.corruption_scheme == "both" else [corrupt_mode]
    )

    # Training sampler: NO domain restriction (BCE-with-logits keeps gradient
    # signal strong on extreme logits; tiny domain pools collapse training).
    # Domain restriction is applied only at eval time inside the evaluator.
    sampler = kb.build_sampler(
        default_mode=corrupt_mode, seed=cfg.seed, device=device,
        kind=("bernoulli" if corrupt_mode == "bernoulli" else "regular"),
        train_triples_for_bern=torch.tensor(kb.train_idx, dtype=torch.long),
        with_domain=False,
    )

    eval_limit = cfg.eval_limit if cfg.eval_limit > 0 else None
    valid_eval_limit = cfg.valid_eval_queries if cfg.valid_eval_queries > 0 else eval_limit
    valid_eval_triples = _limit_triples("validation", kb.valid_idx, valid_eval_limit)

    return DataBundle(
        train_triples=kb.train_idx, valid_triples=kb.valid_idx, test_triples=kb.test_idx,
        entity2id=kb.entity2id, relation2id=kb.relation2id,
        num_entities=kb.num_entities, num_relations=kb.num_relations,
        num_relations_orig=(kb.num_relations_orig or kb.num_relations),
        sampler=sampler,
        train_pos=torch.tensor(kb.train_idx, dtype=torch.long, device=device),
        corrupt_mode=corrupt_mode, train_corrupt_modes=train_corrupt_modes,
        valid_eval_triples=valid_eval_triples,
        head_domain=kb.head_domain, tail_domain=kb.tail_domain,
        use_domain_eval=kb.use_domain_eval, device=device,
    )


# ─── build_model — equivalent to DpRL.build_policy ──────────────────────

def build_model(cfg: TrainConfig, data: DataBundle) -> torch.nn.Module:
    """Instantiate the KGE model, sized to ``data.num_entities/relations``."""
    return build_training_model(cfg, data.num_entities, data.num_relations, data.device)


# ─── build_optimizer — optimizer + scheduler + AMP scaler ───────────────

def build_optimizer(cfg: TrainConfig, model: torch.nn.Module) -> OptimBundle:
    """Adam + optional ReduceLROnPlateau + GradScaler. Also makes save_dir."""
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, eps=1e-7,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)
    plateau_scheduler = None
    if cfg.scheduler == "plateau":
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-4,
        )

    # Disk persistence is opt-in. The shared logging contract
    # (kge_kernels.runs.run_cli) wraps pipeline with a RunContext that
    # handles model saving via output/runs/<exp>/<run>/. Standalone
    # pipeline() calls (e.g. from sweep scripts) default to save_dir=None.
    if cfg.save_dir is not None:
        os.makedirs(cfg.save_dir, exist_ok=True)

    return OptimBundle(optimizer=optimizer, scaler=scaler, plateau_scheduler=plateau_scheduler)


# ─── build_callbacks — validation/early-stop/best-checkpoint bundle ─────

def build_callbacks(cfg: TrainConfig, evaluator: Any, data: DataBundle) -> Callbacks:
    """Initialise the training-time callback bundle (state starts empty)."""
    return Callbacks(cfg=cfg, evaluator=evaluator, data=data)


# ─── build_evaluator — cached RankingEvaluator (val + final) ────────────

def build_evaluator(cfg: TrainConfig, model: torch.nn.Module, data: DataBundle):
    """Build the cached RankingEvaluator. Reused for periodic val + final test.

    The sampler's filter hashes were built from train ∪ valid ∪ test
    positives, so known positives are excluded correctly from the
    candidate pool.
    """
    from ..eval.candidates import SamplerCandidates
    from ..eval.ranking_evaluator import RankingEvaluator
    from ..models.scorer import kge_default_scorer, recommended_eval_batch_size

    eval_negs = getattr(cfg, "eval_num_corruptions", 0) or None
    candidates = SamplerCandidates(
        data.sampler, k=eval_negs,
        head_domain=data.head_domain if data.use_domain_eval else None,
        tail_domain=data.tail_domain if data.use_domain_eval else None,
        unique=eval_negs is not None,
    )
    scheme = cfg.corruption_scheme if cfg.corruption_scheme in ("head", "tail", "both") else "both"
    modes = ("head", "tail") if scheme == "both" else (scheme,)
    return RankingEvaluator(
        scorer=lambda q, p, m: kge_default_scorer(model, q, p, m),
        candidates=candidates,
        batch_size=recommended_eval_batch_size(model, data.num_entities),
        modes=modes, device=data.device, compile=True,
        tie_handling="average", seed=0,
    )


# ─── run_evaluation — final test eval + checkpoint save ─────────────────

def run_evaluation(
    model: torch.nn.Module,
    evaluator: Any,
    data: DataBundle,
    callbacks: Callbacks,
    cfg: TrainConfig,
) -> Dict[str, Any]:
    """Restore best model, compute final train/valid/test metrics, save checkpoint.

    Returns a flat dict with ``train_*`` / ``valid_*`` / ``test_*`` metric
    keys, plus reserved ``_config_path`` / ``_weights_path`` entries the
    caller pops into ``TrainArtifacts``.
    """
    if callbacks.best_state_dict is not None:
        print(f"Restoring best validation model from epoch {callbacks.best_valid_epoch:03d} "
              f"(MRR={callbacks.best_valid_mrr:.4f})")
        unwrap_model(model).load_state_dict(callbacks.best_state_dict)

    metrics: Dict[str, float] = {}
    metric_logs: List[str] = []

    def _final_eval(triples: List[Tuple[int, int, int]]) -> Dict[str, float]:
        triples_t = torch.tensor(triples, dtype=torch.long, device=data.device)
        return evaluator.evaluate(triples_t).metrics()

    eval_limit = cfg.eval_limit if cfg.eval_limit > 0 else None

    if cfg.report_train_mrr:
        train_eval_triples = _limit_triples(
            "train",
            (data.train_triples if not cfg.use_reciprocal
             else [t for t in data.train_triples if t[0] < data.num_relations // 2]),
            eval_limit,
        )
        if train_eval_triples:
            _build_metric_summary("train", _final_eval(train_eval_triples), metrics, metric_logs)

    if data.valid_triples and data.valid_eval_triples:
        print("Computing validation metrics ...")
        _build_metric_summary("valid", _final_eval(data.valid_eval_triples), metrics, metric_logs)

    if data.test_triples:
        print("Computing test metrics ...")
        _build_metric_summary("test", _final_eval(data.test_triples), metrics, metric_logs)

    if metric_logs:
        print("Evaluation | " + " | ".join(metric_logs))

    weights_path: str = ""
    config_path: str = ""
    if cfg.save_dir is not None:
        weights_path, config_path = save_final_checkpoint(
            cfg.save_dir,
            state_dict=unwrap_model(model).state_dict(),
            config_payload=callbacks.make_payload(metrics),
            entity2id=data.entity2id,
            relation2id=data.relation2id,
        )
        print(f"Saved model to {cfg.save_dir}")

    return {**metrics, "_config_path": config_path, "_weights_path": weights_path}


# ─── helpers ────────────────────────────────────────────────────────────

def _limit_triples(name: str, triples: List, limit: Optional[int]) -> List:
    """Cap eval-set size for fast iteration; print a heads-up when cropping."""
    if not triples:
        return []
    if limit and len(triples) > limit:
        print(f"Evaluation on {name}: limiting to first {limit} of {len(triples)} triples")
        return triples[:limit]
    return triples


def _build_metric_summary(prefix: str, rank_metrics: Dict[str, float],
                          metrics: Dict[str, float], logs: List[str]) -> None:
    """Flatten a RankingEvaluator metrics dict into the run's flat metrics dict."""
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


__all__ = [
    "Callbacks",
    "DataBundle",
    "OptimBundle",
    "build_callbacks",
    "build_data",
    "build_evaluator",
    "build_model",
    "build_optimizer",
    "run_evaluation",
]
