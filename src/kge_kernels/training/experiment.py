"""KGE pretraining run-experiment hook.

Wraps the existing :func:`kge_kernels.training.pipeline.train_model`
function as a :func:`kge_kernels.runs.run_cli`-compatible
``run_experiment(ctx, cfg)`` callable. Use this when you want a
standalone KGE backbone pretrain via the shared run-bundle machinery.

Layered the same way the consumer repos (torch-ns, DpRL) do it:

  * ``run_experiment(ctx, cfg)`` — the public hook called by ``run_cli``.
  * ``train(cfg)`` — the inner trainer (re-export of ``train_model``).
"""
from __future__ import annotations

from typing import Mapping

from .config import TrainConfig
from .pipeline import train_model
from ..runs import RunContext


# Re-export under the uniform "train" name (matches torch-ns / DpRL convention).
train = train_model


def run_experiment(ctx: RunContext, cfg: TrainConfig) -> Mapping[str, dict]:
    """Run a single KGE pretraining experiment.

    Wires the run-bundle context onto ``cfg`` (saves go through
    ``ctx.paths.root``), invokes :func:`train_model`, then forwards
    the flat metrics dict (``train_mrr`` / ``valid_mrr`` / ``test_mrr``
    / etc.) into ``ctx.log_metrics`` split by prefix.
    """
    cfg.save_dir = str(ctx.paths.root)

    artifacts = train(cfg)
    metrics = artifacts.metrics or {}

    # Split flat keys by prefix (train_mrr → split="train", etc.).
    by_split: dict[str, dict] = {"train": {}, "val": {}, "test": {}}
    for key, value in metrics.items():
        for prefix, split in (("train_", "train"), ("valid_", "val"), ("test_", "test")):
            if key.startswith(prefix):
                by_split[split][key[len(prefix):]] = value
                break
        else:
            by_split.setdefault("other", {})[key] = value

    for split, md in by_split.items():
        if md:
            ctx.log_metrics(md, split=split)

    return by_split


__all__ = ["run_experiment", "train"]
