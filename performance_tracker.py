"""Performance tracker — records model evaluation metrics over time."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


class PerformanceTracker:
    """Tracks model metrics with timestamps, supports drift detection via PSI/KS."""

    def __init__(self, results_dir: Path | str = "./ML_Results") -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._history_file = self.results_dir / "performance_history.jsonl"

    def record(
        self,
        model_name: str,
        metrics: dict[str, float | int],
        tags: dict[str, str] | None = None,
    ) -> None:
        """Append a performance record.

        Args:
            model_name: Identifier for the model.
            metrics: Dict of metric_name -> numeric value.
            tags: Optional metadata (e.g. {"dataset_version": "2024-06"}).
        """
        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model_name,
            "metrics": metrics,
            "tags": tags or {},
        }

        with open(self._history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def history(self, model: str | None = None) -> list[dict]:
        """Load all records, optionally filtered by model name."""
        if not self._history_file.exists():
            return []

        records: list[dict] = []
        with open(self._history_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if model is None or rec["model"] == model:
                    records.append(rec)
        return records

    @staticmethod
    def population_stability_index(
        expected: np.ndarray, actual: np.ndarray, bins: int = 10
    ) -> float:
        """Compute PSI between two distributions.

        PSI < 0.1  → no significant drift
        PSI 0.1–0.25 → moderate drift
        PSI > 0.25 → significant drift
        """
        hist_exp, bin_edges = np.histogram(expected, bins=bins, density=True)
        hist_act, _ = np.histogram(actual, bins=bin_edges, density=True)

        # Avoid division by zero / log(0)
        hist_exp = np.clip(hist_exp, 1e-10, None)
        hist_act = np.clip(hist_act, 1e-10, None)

        psi = np.sum((hist_act - hist_exp) * np.log(hist_act / hist_exp))
        return float(round(psi, 6))

    @staticmethod
    def ks_test(
        expected: np.ndarray, actual: np.ndarray
    ) -> tuple[float, float]:
        """Two-sample Kolmogorov-Smirnov test returning (statistic, p-value)."""
        from scipy.stats import ks_2samp  # lazy import — scipy is optional

        return ks_2samp(expected, actual)

    def drift_report(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str = "feature",
    ) -> dict[str, Any]:
        """Generate a drift report for a single feature."""
        psi = self.population_stability_index(reference, current)
        ks_stat, ks_pval = self.ks_test(reference, current)

        return {
            "feature": feature_name,
            "psi": psi,
            "ks_statistic": round(ks_stat, 6),
            "ks_p_value": round(ks_pval, 6),
            "drift_detected": psi > 0.25 or ks_pval < 0.05,
            "interpretation": (
                "significant drift"
                if psi > 0.25
                else "moderate drift"
                if psi > 0.1
                else "no significant drift"
            ),
        }