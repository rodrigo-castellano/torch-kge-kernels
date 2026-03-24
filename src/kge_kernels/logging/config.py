"""Configuration types for the shared experiment logging runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class OutputConfig:
    """Filesystem output settings for a run."""

    output_root: str = "./output"
    save_stdout: bool = True
    save_events: bool = True
    save_metrics: bool = True


@dataclass
class ModelConfig:
    """Single-model save policy."""

    mode: Literal["none", "last", "best"] = "best"
    metric: str = "valid/mrr"
    maximize: bool = True
    filename: str = "model.safetensors"


@dataclass
class RegistryConfig:
    """Promotion settings for the minimal model registry."""

    enabled: bool = False
    promote_on_success: bool = False
    model_name: Optional[str] = None


@dataclass
class ReportConfig:
    """Optional human or agent-authored run report."""

    enabled: bool = False
    writer: Literal["none", "human", "agent"] = "none"
    filename: str = "report.md"


@dataclass
class LoggingConfig:
    """Top-level logging configuration shared across repositories."""

    family: Optional[str] = None
    output: OutputConfig = field(default_factory=OutputConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    registry: RegistryConfig = field(default_factory=RegistryConfig)
    report: ReportConfig = field(default_factory=ReportConfig)


__all__ = [
    "LoggingConfig",
    "ModelConfig",
    "OutputConfig",
    "RegistryConfig",
    "ReportConfig",
]
