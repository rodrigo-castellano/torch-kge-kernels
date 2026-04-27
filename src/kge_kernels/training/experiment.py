"""KGE pretraining experiment.

Pure config-in / artifacts-out trainer. No I/O lifecycle, no
:class:`RunContext` — that layer lives in :mod:`kge_kernels.training.cli`.

Exposes a single :func:`pipeline` function (a re-export of the
existing :func:`kge_kernels.training.pipeline.train_model`) so the
trainer entry point has a uniform name across all consumer repos.
"""
from __future__ import annotations

from .pipeline import train_model

# Uniform "pipeline" name across consumers (matches torch_ns.experiment.pipeline,
# kge_experiments.experiment.pipeline). Underlying body is train_model.
pipeline = train_model

__all__ = ["pipeline"]
