"""Shared run orchestrator for repository-specific experiment specs."""

from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any, Mapping, Protocol
import traceback

from .config import LoggingConfig
from .context import RunContext
from .layout import build_run_id, build_run_paths


class ExperimentSpec(Protocol):
    """Repository adapter for the shared run lifecycle."""

    def resolve_config(self, raw_config: Any) -> Any:
        """Convert raw config input into the repo's resolved config object."""

    def logging_config(self, config: Any) -> LoggingConfig:
        """Return the shared logging config for this run."""

    def family(self, config: Any) -> str:
        """Return the run family, for example 'train' or 'eval'."""

    def signature(self, config: Any) -> str:
        """Return a stable run signature."""

    def run(self, ctx: RunContext, config: Any) -> Mapping[str, Any]:
        """Execute the repository-specific work and return a final summary."""


def _seed_from_config(config: Any) -> int:
    if isinstance(config, Mapping):
        return int(config.get("seed", 0))
    return int(getattr(config, "seed", 0))


def run_experiment(raw_config: Any, spec: ExperimentSpec) -> Mapping[str, Any]:
    """Run a repository-specific experiment inside the shared lifecycle."""
    config = spec.resolve_config(raw_config)
    logging_cfg = spec.logging_config(config)
    family = spec.family(config)
    signature = spec.signature(config)
    seed = _seed_from_config(config)
    run_id, day, started_at = build_run_id(signature=signature, seed=seed)
    paths = build_run_paths(logging_cfg.output.output_root, family=family, run_id=run_id, day=day)
    ctx = RunContext(
        logging=logging_cfg,
        family=family,
        signature=signature,
        seed=seed,
        run_id=run_id,
        started_at=started_at,
        paths=paths,
        resolved_config=config,
    )

    try:
        ctx.log_event("run_started")
        with ctx.stdout_capture():
            summary = dict(spec.run(ctx, config))
        if logging_cfg.registry.enabled and logging_cfg.registry.promote_on_success:
            promoted = ctx.promote_model()
            if promoted is not None:
                ctx.log_event("model_promoted", registry_path=str(promoted))
                summary.setdefault("registry_version_path", str(promoted))
        ctx.log_event("run_completed")
        ctx.finish(status="completed", final_metrics=summary)
        return summary
    except Exception as exc:
        error_summary = {
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        ctx.log_event("run_failed", error=str(exc))
        ctx.finish(status="failed", final_metrics=error_summary, error=str(exc))
        raise


__all__ = [
    "ExperimentSpec",
    "run_experiment",
]
