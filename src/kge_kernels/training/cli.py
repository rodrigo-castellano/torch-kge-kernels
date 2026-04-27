"""CLI entrypoint for standalone KGE pretraining via the shared run_cli driver.

Single thin shim. The actual work happens in
:func:`kge_kernels.training.experiment.run_experiment` (called per
(combo × seed) by ``run_cli``); CLI parsing + ``--set`` / ``--grid``
overrides + grid expansion + per-seed loop come from
:func:`kge_kernels.runs.run_cli`.

Run via ``python -m kge_kernels.training.cli --set dataset=family ...``.

Run-bundle metadata (family / signature / logging_config) is provided
by methods on :class:`TrainConfig` (duck-typed by ``run_cli``).
"""
from __future__ import annotations

from .config import TrainConfig
from .experiment import run_experiment
from ..runs import run_cli


def main() -> None:
    run_cli(
        config_cls=TrainConfig,
        run_experiment=run_experiment,
        description="tkk: standalone KGE pretraining",
    )


if __name__ == "__main__":
    main()
