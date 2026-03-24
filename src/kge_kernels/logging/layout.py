"""Path helpers for run-oriented experiment logging."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re


@dataclass(frozen=True)
class RunPaths:
    """Concrete filesystem layout for a single run."""

    root: Path
    config_dir: Path
    logs_dir: Path
    model_dir: Path
    artifacts_dir: Path
    family_report_dir: Path
    registry_root: Path

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"


def sanitize_slug(value: str) -> str:
    """Normalize user-provided identifiers into stable path components."""
    collapsed = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    return collapsed.strip("-._") or "run"


def build_run_id(signature: str, seed: int) -> tuple[str, str, str]:
    """Return a stable run id, date component, and ISO start time."""
    now = datetime.now(timezone.utc)
    day = now.strftime("%Y-%m-%d")
    ts = now.strftime("%Y%m%d-%H%M%S")
    run_id = f"{ts}_{sanitize_slug(signature)}_s{seed}"
    return run_id, day, now.isoformat()


def build_run_paths(output_root: str, family: str, run_id: str, day: str) -> RunPaths:
    """Create the canonical path layout for a run."""
    output_root_path = Path(output_root).expanduser()
    family_slug = sanitize_slug(family)
    root = output_root_path / "runs" / family_slug / day / run_id
    return RunPaths(
        root=root,
        config_dir=root / "config",
        logs_dir=root / "logs",
        model_dir=root / "model",
        artifacts_dir=root / "artifacts",
        family_report_dir=output_root_path / "reports" / family_slug / day,
        registry_root=output_root_path / "registry",
    )


__all__ = [
    "RunPaths",
    "build_run_id",
    "build_run_paths",
    "sanitize_slug",
]
