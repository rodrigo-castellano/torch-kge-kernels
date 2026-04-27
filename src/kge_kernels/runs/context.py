"""Runtime logging context for a single experiment run."""

from __future__ import annotations

import json
import shutil
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from .config import LoggingConfig
from .layout import RunPaths, sanitize_slug


def _normalize_json(value: Any) -> Any:
    """Convert common Python and PyTorch values into JSON-safe structures."""
    if is_dataclass(value):
        return _normalize_json(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _normalize_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json(v) for v in value]
    if isinstance(value, set):
        return sorted(_normalize_json(v) for v in value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            return repr(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return {
            k: _normalize_json(v)
            for k, v in vars(value).items()
            if not k.startswith("_")
        }
    return repr(value)


class _TeeStream:
    """Mirror writes to the original stream and a file."""

    def __init__(self, original: Any, file_handle: Any) -> None:
        self._original = original
        self._file_handle = file_handle

    def write(self, data: str) -> int:
        self._file_handle.write(data)
        self._original.write(data)
        return len(data)

    def flush(self) -> None:
        self._file_handle.flush()
        self._original.flush()


class RunContext:
    """Run-scoped logging, artifact, and promotion helpers."""

    def __init__(
        self,
        *,
        logging: LoggingConfig,
        family: str,
        signature: str,
        seed: int,
        run_id: str,
        started_at: str,
        paths: RunPaths,
        resolved_config: Any,
    ) -> None:
        self.logging = logging
        self.family = family
        self.signature = signature
        self.seed = int(seed)
        self.run_id = run_id
        self.started_at = started_at
        self.paths = paths
        self.resolved_config = resolved_config
        self._saved_model_path: Optional[Path] = None
        self._saved_model_info: dict[str, Any] | None = None

        for directory in (
            self.paths.experiment_root,
            self.paths.root,
            self.paths.artifacts_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        self._update_campaign_metadata()
        self._write_json(self.paths.config_path, _normalize_json(resolved_config))
        self._write_manifest(status="running")

    def _write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_normalize_json(payload), indent=2, sort_keys=True) + "\n")

    def _append_jsonl(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_normalize_json(dict(payload)), sort_keys=True) + "\n")

    def _append_grouped_metric_row(
        self,
        path: Path,
        payload: Mapping[str, Any],
        *,
        split: Optional[str],
    ) -> None:
        """Append a split-grouped metric snapshot row."""
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
        else:
            existing = {}

        if not isinstance(existing, dict):
            existing = {}

        canonical_split = self._canonical_metric_split(split)
        for bucket in ("train", "val", "test"):
            existing.setdefault(bucket, [])
        existing.setdefault(canonical_split, [])
        existing[canonical_split].append(_normalize_json(dict(payload)))
        path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _write_manifest(self, *, status: str, error: Optional[str] = None, final_metrics: Any = None) -> None:
        manifest = {
            "run": {
                "id": self.run_id,
                "experiment_name": self.family,
                "family": self.family,
                "signature": self.signature,
                "seed": self.seed,
                "status": status,
                "started_at": self.started_at,
                "ended_at": datetime.now(timezone.utc).isoformat() if status != "running" else None,
                "error": error,
            },
            "paths": {
                "experiment_root": str(self.paths.experiment_root),
                "run_root": str(self.paths.root),
                "model_path": str(self._saved_model_path) if self._saved_model_path else None,
            },
            "model": _normalize_json(self._saved_model_info) if self._saved_model_info is not None else None,
            "final_metrics": _normalize_json(final_metrics) if final_metrics is not None else None,
        }
        self._write_json(self.paths.manifest_path, manifest)

    def _update_campaign_metadata(self, *, writer: Optional[str] = None) -> None:
        created_at = datetime.now(timezone.utc).isoformat()
        path = self.paths.campaign_path
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
        else:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        payload.setdefault("name", self.family)
        payload.setdefault("created_at", created_at)
        run_ids = payload.get("run_ids")
        if not isinstance(run_ids, list):
            run_ids = []
        if self.run_id not in run_ids:
            run_ids.append(self.run_id)
        payload["run_ids"] = run_ids
        if writer is not None:
            payload["writer"] = writer
        self._write_json(path, payload)

    def _canonical_metric_split(self, split: Optional[str]) -> str:
        if split is None:
            return "other"
        normalized = sanitize_slug(split).lower()
        if normalized in {"eval", "valid", "validation", "val"}:
            return "val"
        if normalized in {"train", "test"}:
            return normalized
        return normalized or "other"

    @contextmanager
    def stdout_capture(self):
        """Mirror stdout/stderr into the run log."""
        if not self.logging.output.save_stdout:
            yield
            return
        with self.paths.stdout_path.open("a", encoding="utf-8", buffering=1) as handle:
            tee_out = _TeeStream(sys.stdout, handle)
            tee_err = _TeeStream(sys.stderr, handle)
            with redirect_stdout(tee_out), redirect_stderr(tee_err):
                yield

    def log_event(self, name: str, **payload: Any) -> None:
        """Append a structured event line if event logging is enabled."""
        if not self.logging.output.save_events:
            return
        self._append_jsonl(
            self.paths.events_path,
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": name,
                **payload,
            },
        )

    def log_metrics(self, metrics: Mapping[str, Any], *, step: Optional[int] = None, split: Optional[str] = None) -> None:
        """Append one readable metric snapshot row for the given split."""
        if not self.logging.output.save_metrics:
            return
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **dict(metrics),
        }
        if step is not None:
            entry["step"] = int(step)
        self._append_grouped_metric_row(self.paths.metrics_path, entry, split=split)

    def save_model(
        self,
        serializer: Any,
        *,
        saved_as: Optional[str] = None,
        metric_name: Optional[str] = None,
        metric_value: Optional[float] = None,
        global_step: Optional[int] = None,
        extra_metadata: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Path]:
        """Persist the single model artifact for this run."""
        if self.logging.model.mode == "none":
            return None
        model_path = self.paths.model_path
        serializer(model_path)
        self._saved_model_path = model_path
        model_info = {
            "save_mode": saved_as or self.logging.model.mode,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "global_step": int(global_step) if global_step is not None else None,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        if extra_metadata:
            model_info.update(_normalize_json(dict(extra_metadata)))
        self._saved_model_info = model_info
        self._write_json(self.paths.model_info_path, self._saved_model_info)
        return model_path

    def write_report(self, content: str, *, metadata: Optional[Mapping[str, Any]] = None) -> Optional[Path]:
        """Create the optional experiment-level human or agent report."""
        if not self.logging.report.enabled:
            return None
        report_path = self.paths.report_path
        report_path.write_text(content.rstrip() + "\n", encoding="utf-8")
        writer = self.logging.report.writer
        if metadata and metadata.get("writer"):
            writer = str(metadata["writer"])
        self._update_campaign_metadata(writer=writer)
        return report_path

    def promote_model(self, *, model_name: Optional[str] = None) -> Optional[Path]:
        """Copy the current run bundle into the mirrored registry."""
        if not self.logging.registry.enabled or self._saved_model_path is None:
            return None
        experiment_name = sanitize_slug(self.family)
        version_dir = self.paths.registry_root / experiment_name / self.run_id
        version_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.paths.root, version_dir, dirs_exist_ok=True)
        provenance = {
            "registry_name": model_name or self.logging.registry.model_name,
            "source_run_path": str(self.paths.root),
            "source_run_id": self.run_id,
            "source_signature": self.signature,
            "promoted_at": datetime.now(timezone.utc).isoformat(),
        }
        self._write_json(version_dir / "promotion.json", provenance)
        return version_dir

    def finish(self, *, status: str, final_metrics: Mapping[str, Any], error: Optional[str] = None) -> None:
        """Finalize the single run manifest."""
        self._write_manifest(status=status, error=error, final_metrics=final_metrics)


__all__ = [
    "RunContext",
]
