"""tkk-shared CLI helpers — generic runner driver around ``run_experiment``.

The :func:`run_cli` driver lets each consumer repo expose a uniform CLI
(``--set KEY=VALUE`` / ``--grid KEY=V1,V2``) over its own typed config.
The repo provides:

  * a ``@dataclass`` config type (e.g. ``DpRLTrainConfig`` /
    ``NSTrainConfig`` / ``KGETrainConfig``),
  * an :class:`kge_kernels.logging.ExperimentSpec` adapter that knows
    how to run one experiment from a resolved config.

…and gets a fully-functional CLI shim with ~5 lines of glue.
"""
from .runner import run_cli

__all__ = ["run_cli"]
