"""Path helpers for run-oriented experiment logging."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    """Concrete filesystem layout for a single run."""

    experiment_root: Path
    root: Path
    artifacts_dir: Path
    registry_root: Path
    model_filename: str = "model.safetensors"
    report_filename: str = "report.md"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def config_path(self) -> Path:
        return self.root / "config.json"

    @property
    def stdout_path(self) -> Path:
        return self.root / "stdout.log"

    @property
    def events_path(self) -> Path:
        return self.root / "events.jsonl"

    @property
    def metrics_path(self) -> Path:
        return self.root / "metrics.json"

    @property
    def model_path(self) -> Path:
        return self.root / self.model_filename

    @property
    def model_info_path(self) -> Path:
        return self.root / "model_info.json"

    @property
    def campaign_path(self) -> Path:
        return self.experiment_root / "campaign.json"

    @property
    def report_path(self) -> Path:
        return self.experiment_root / self.report_filename


def sanitize_slug(value: str) -> str:
    """Normalize user-provided identifiers into stable path components."""
    collapsed = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    return collapsed.strip("-._") or "run"


def build_run_id(signature: str, seed: int) -> tuple[str, str]:
    """Return a stable run id and ISO start time."""
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%d-%H%M%S")
    run_id = f"{ts}_{sanitize_slug(signature)}_s{seed}"
    return run_id, now.isoformat()


def build_run_paths(
    output_root: str,
    family: str,
    run_id: str,
    *,
    model_filename: str = "model.safetensors",
    report_filename: str = "report.md",
) -> RunPaths:
    """Create the canonical path layout for a run."""
    output_root_path = Path(output_root).expanduser()
    family_slug = sanitize_slug(family)
    experiment_root = output_root_path / "runs" / family_slug
    root = experiment_root / run_id
    return RunPaths(
        experiment_root=experiment_root,
        root=root,
        artifacts_dir=root / "artifacts",
        registry_root=output_root_path / "registry",
        model_filename=sanitize_slug(model_filename),
        report_filename=sanitize_slug(report_filename),
    )


__all__ = [
    "RunPaths",
    "build_run_id",
    "build_run_paths",
    "sanitize_slug",
]
