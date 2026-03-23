"""Evaluation results container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class EvalResults:
    """Container for evaluation metrics and statistics.

    Attributes:
        metrics: ``{mode: {MRR, Hits@1, Hits@3, Hits@10}}`` per scoring mode.
        stats: Aggregate stats (proved_pos, proved_neg, etc.).
        config: Evaluation configuration for reproducibility.
    """

    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    stats: Dict[str, float] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    def print(self) -> None:
        """Print results in a formatted table."""
        if not self.metrics:
            print("No results.")
            return

        col_w = max(24, max(len(m) for m in self.metrics) + 2)
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)

        if self.stats:
            print("\n--- Statistics ---")
            for k, v in self.stats.items():
                print(f"  {k}: {v:.4f}")

        print(f"\n{'Mode':<{col_w}} {'MRR':>8} {'Hits@1':>8} {'Hits@3':>8} {'Hits@10':>8}")
        print("-" * (col_w + 36))
        for mode, m in self.metrics.items():
            print(
                f"{mode:<{col_w}} {m.get('MRR', 0):>8.4f} "
                f"{m.get('Hits@1', 0):>8.4f} {m.get('Hits@3', 0):>8.4f} "
                f"{m.get('Hits@10', 0):>8.4f}"
            )
        print("=" * 80)

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict."""
        return {"metrics": self.metrics, "stats": self.stats, "config": self.config}


__all__ = ["EvalResults"]
