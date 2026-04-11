"""Smoke test: run each example script end-to-end.

This keeps the ``examples/`` directory honest — any API change that
breaks the quickstart scripts fails CI. Each example is imported
via ``runpy.run_path`` so the test exercises the ``if __name__ ==
"__main__":`` block, not just the module import.
"""
from __future__ import annotations

import runpy
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
EXAMPLE_SCRIPTS = [
    "01_sbr_exhaustive.py",
    "02_train_kge.py",
    "03_filtered_eval.py",
]


@pytest.mark.parametrize("script", EXAMPLE_SCRIPTS)
def test_example_runs(script: str) -> None:
    """Each example script should run to completion without error."""
    path = EXAMPLES_DIR / script
    assert path.exists(), f"missing example: {path}"
    runpy.run_path(str(path), run_name="__main__")
