"""Shared run-bundle machinery: paths + lifecycle + CLI driver.

Layered policy → paths → runtime, then a CLI driver that ties them
together and calls the consumer's ``run_experiment(ctx, cfg)`` function.

Public API:

  * :class:`LoggingConfig` (and sub-configs) — run-bundle policy
    (where to write, what to save, when).
  * :class:`RunPaths` + :func:`build_run_paths` — canonical
    ``output/runs/<family>/<run_id>/...`` layout.
  * :class:`RunContext` — runtime object handed to ``run_experiment``;
    owns ``log_metrics`` / ``log_event`` / ``save_model`` /
    ``stdout_capture`` / ``write_report`` / ``promote_model``.
  * :func:`run_cli` — the CLI entry point. Takes a typed config class
    and a ``run_experiment(ctx, cfg) -> dict | None`` callable.

There is no longer an ``ExperimentSpec`` Protocol — consumers write a
plain function instead of a callback-bag class. Run-bundle metadata
(family / signature / logging_config) comes from methods on the config
dataclass (duck-typed), or from explicit overrides passed to
``run_cli``.
"""

from .cli import run_cli, run_one
from .config import LoggingConfig, ModelConfig, OutputConfig, RegistryConfig, ReportConfig
from .context import RunContext
from .layout import RunPaths, build_run_id, build_run_paths

__all__ = [
    "LoggingConfig",
    "ModelConfig",
    "OutputConfig",
    "RegistryConfig",
    "ReportConfig",
    "RunContext",
    "RunPaths",
    "build_run_id",
    "build_run_paths",
    "run_cli",
    "run_one",
]
