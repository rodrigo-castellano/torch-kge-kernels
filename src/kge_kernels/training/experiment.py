"""Full KGE training pipeline.

:func:`pipeline` orchestrates the complete lifecycle:
  data loading → model construction → training (via ``train_epoch``) →
  validation / early stopping → final evaluation → checkpoint saving.

Pure config-in / artifacts-out — no :class:`RunContext`, no I/O
lifecycle. The run-bundle layer lives in
:mod:`kge_kernels.training.cli`.

Single training path. Loss and sampler are one-line opt-ins from
``TrainConfig``: ``cfg.loss == "nssa"`` swaps the per-step loss to
:func:`kge_kernels.training.loss.nssa_train_step`; ``cfg.corruption_scheme
== "both"`` (the default) routes through ``corrupt_mode == "bernoulli"``
which picks ``BernoulliSampler`` instead of the plain ``Sampler``.

The unified path always goes through ``train_epoch``.
"""
from __future__ import annotations

from .builder import (
    build_callbacks,
    build_data,
    build_evaluator,
    build_model,
    build_optimizer,
    run_evaluation,
)
from .config import TrainArtifacts, TrainConfig
from .epoch import set_seed
from .train import train


def pipeline(cfg: TrainConfig) -> TrainArtifacts:
    """Full KGE training pipeline.

    Loads data, builds model, runs the unified ``train_epoch`` loop
    (BCE by default; ``cfg.loss == "nssa"`` swaps the per-step loss),
    handles validation / early stopping / checkpoint management, then
    runs final evaluation and saves the checkpoint.

    Args:
        cfg: Complete training configuration.

    Returns:
        :class:`TrainArtifacts` with entity/relation mappings,
        checkpoint paths, and final metrics.
    """
    set_seed(cfg.seed)

    data       = build_data(cfg)
    model      = build_model(cfg, data)
    optim      = build_optimizer(cfg, model)
    evaluator  = build_evaluator(cfg, model, data)
    callbacks  = build_callbacks(cfg, evaluator, data)

    train_m: dict = {}
    if cfg.epochs > 0:
        train_m = train(model, data, optim, callbacks, evaluator, cfg)

    eval_m = run_evaluation(model, evaluator, data, callbacks, cfg)
    config_path = eval_m.pop("_config_path", "")
    weights_path = eval_m.pop("_weights_path", "")

    return TrainArtifacts(
        entity2id=data.entity2id,
        relation2id=data.relation2id,
        config_path=config_path,
        weights_path=weights_path,
        metrics={**train_m, **eval_m} or None,
    )


__all__ = ["pipeline"]
