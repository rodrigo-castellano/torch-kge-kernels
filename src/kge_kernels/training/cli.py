"""CLI entrypoint for standalone KGE pretraining via the shared run_cli driver.

Owns the run-bundle lifecycle: parses the standard ``--set`` /
``--grid`` CLI surface (via :func:`kge_kernels.runs.run_cli`), then
for each (combo × seed) bridges the :class:`RunContext` (paths /
metrics persistence / model save policy) with the pure trainer in
:mod:`kge_kernels.training.experiment`.

Run via ``python -m kge_kernels.training.cli --set dataset=family ...``.
"""
from __future__ import annotations

from typing import Mapping

from .config import TrainConfig
from .experiment import pipeline
from ..runs import RunContext, run_cli


def run_experiment(ctx: RunContext, cfg: TrainConfig) -> Mapping[str, dict]:
    """Run-experiment hook for ``run_cli``.

    Wires the run bundle's :class:`RunContext` onto ``cfg`` (saves go
    through ``ctx.paths.root``), invokes :func:`pipeline`, then
    forwards the flat metrics dict (``train_mrr`` / ``valid_mrr`` /
    ``test_mrr`` / etc.) into ``ctx.log_metrics`` split by prefix.
    """
    cfg.save_dir = str(ctx.paths.root)

    artifacts = pipeline(cfg)
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


def main() -> None:
    run_cli(
        config_cls=TrainConfig,
        run_experiment=run_experiment,
        description="tkk: standalone KGE pretraining",
    )


if __name__ == "__main__":
    main()
