"""DataService — data loading, cohort analysis, heatmap, and drift detection."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from app.services.model_service import ModelService

try:
    from logging_config import logger
except ImportError:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("intentsight")

try:
    from performance_tracker import PerformanceTracker  # type: ignore[no-redef]
    _tracker = PerformanceTracker()
except Exception:
    _tracker = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[2]
DATASET_CANDIDATES = (
    ROOT / "Behaviour_Extended_Dataset.csv",
    ROOT / "Behaviour_Dataset.csv",
)


NUMERIC_METRICS = {
    "app_usage_time_min": {
        "label": "App usage",
        "bins": [-np.inf, 60, 120, 240, np.inf],
        "labels": ["0-60m", "61-120m", "121-240m", "240m+"],
    },
    "swipe_right_ratio": {
        "label": "Swipe right ratio",
        "bins": [-np.inf, 0.25, 0.5, 0.75, np.inf],
        "labels": ["0-.25", ".26-.50", ".51-.75", ".76-1.00"],
    },
    "match_rate": {
        "label": "Match rate",
        "bins": [-np.inf, 0.1, 0.25, 0.5, np.inf],
        "labels": ["0-.10", ".11-.25", ".26-.50", ".50+"],
    },
    "bmi": {
        "label": "BMI",
        "bins": [-np.inf, 18.5, 25, 30, np.inf],
        "labels": ["Under", "Healthy", "Elevated", "High"],
    },
}

CATEGORY_FIELDS = {
    "gender": "Gender",
    "income_bracket": "Income",
    "education_level": "Education",
    "sexual_orientation": "Orientation",
    "location_type": "Location",
    "swipe_time_of_day": "Swipe time",
}


class DataService:
    """Data loading, engineered features, dashboard data, and drift detection."""

    def __init__(self, model_service: ModelService):
        self.model_service = model_service
        self._dataset_source: str | None = None

    def dataset_path(self) -> Path:
        """Resolve dataset path — respects INTENTSIGHT_DATASET_PATH env var."""
        override_value = os.getenv("INTENTSIGHT_DATASET_PATH")
        override = Path(override_value) if override_value else None
        candidates = [override] if override else list(DATASET_CANDIDATES)

        for candidate in candidates:
            if candidate is None:
                continue
            if not candidate.is_absolute():
                candidate = ROOT / candidate
            if candidate.exists():
                self._dataset_source = str(candidate)
                return candidate

        searched = ", ".join(str(p) for p in candidates if p is not None)
        raise FileNotFoundError(f"No behavior dataset CSV found. Searched: {searched}")

    def raw_frame(self) -> pd.DataFrame:
        """Load raw CSV — no caching to avoid stale data if env var changes."""
        path = self.dataset_path()
        logger.info("Loading raw dataset: %s", path)
        return pd.read_csv(path)

    def dashboard_frame(self) -> pd.DataFrame:
        raw = self.raw_frame()
        raw = self._add_engineered_fields(raw)

        # Construct engagement_level target (same logic as preprocess.py)
        from sklearn.preprocessing import StandardScaler as _SS
        behav_cols = ["app_usage_time_min", "swipe_right_ratio", "message_sent_count",
                      "likes_received", "emoji_usage_rate"]
        behav_available = [c for c in behav_cols if c in raw.columns]
        if behav_available:
            scaled = _SS().fit_transform(raw[behav_available])
            raw["engagement_score"] = scaled.sum(axis=1)
            raw["engagement_level"] = pd.qcut(
                raw["engagement_score"], q=3, labels=[0, 1, 2]
            ).astype(int)
        else:
            raw["engagement_level"] = 1  # fallback: Medium

        y_encoded = raw["engagement_level"].values
        _, test_index = train_test_split(
            raw.index.to_numpy(),
            test_size=0.2,
            random_state=42,
            stratify=y_encoded,
        )
        test_frame = raw.loc[test_index].reset_index(drop=True)

        try:
            predictions = self.model_service.test_predictions().reset_index(drop=True)
            if len(predictions) == len(test_frame):
                test_frame = pd.concat([test_frame, predictions], axis=1)
            else:
                test_frame = self._fallback_predictions(test_frame)
        except Exception:
            test_frame = self._fallback_predictions(test_frame)

        for metric in self._available_numeric_metrics(test_frame):
            self._add_band(test_frame, metric)
        return test_frame

    def _fallback_predictions(self, frame: pd.DataFrame) -> pd.DataFrame:
        fallback = frame.copy()
        level_map = {0: "Low", 1: "Medium", 2: "High"}
        fallback["prediction_label"] = fallback["engagement_level"].map(level_map)
        fallback["prediction_encoded"] = fallback["engagement_level"]
        fallback["confidence"] = np.nan
        return fallback

    @staticmethod
    def _add_engineered_fields(frame: pd.DataFrame) -> pd.DataFrame:
        engineered = frame.copy()
        if {"mutual_matches", "likes_received"}.issubset(engineered.columns):
            engineered["match_rate"] = engineered["mutual_matches"] / (
                engineered["likes_received"] + 1
            )
        if {"message_sent_count", "mutual_matches"}.issubset(engineered.columns):
            engineered["msg_per_match"] = engineered["message_sent_count"] / (
                engineered["mutual_matches"] + 1
            )
        if {"weight_kg", "height_cm"}.issubset(engineered.columns):
            engineered["bmi"] = engineered["weight_kg"] / ((engineered["height_cm"] / 100) ** 2)
        return engineered

    def _available_numeric_metrics(
        self, frame: pd.DataFrame | None = None
    ) -> dict[str, dict[str, Any]]:
        source = frame if frame is not None else self._add_engineered_fields(self.raw_frame())
        return {key: value for key, value in NUMERIC_METRICS.items() if key in source.columns}

    @staticmethod
    def _add_band(frame: pd.DataFrame, metric: str) -> None:
        config = NUMERIC_METRICS[metric]
        frame[f"{metric}_band"] = pd.cut(
            frame[metric],
            bins=config["bins"],
            labels=config["labels"],
            include_lowest=True,
        ).astype(str)

    def options(self) -> dict[str, Any]:
        numeric_metrics = self._available_numeric_metrics()
        categories = {
            key: value for key, value in CATEGORY_FIELDS.items()
            if key in self.raw_frame().columns
        }
        return {
            "metrics": [
                {"value": key, "label": value["label"]}
                for key, value in numeric_metrics.items()
            ],
            "categories": [
                {"value": key, "label": value}
                for key, value in categories.items()
            ],
        }

    def cohorts(
        self,
        view: str = "aggregated",
        metric: str = "app_usage_time_min",
        category: str = "gender",
    ) -> dict[str, Any]:
        frame = self.dashboard_frame()
        available_metrics = self._available_numeric_metrics(frame)
        available_categories = {
            key: value for key, value in CATEGORY_FIELDS.items()
            if key in frame.columns
        }

        if metric not in available_metrics:
            raise ValueError(f"Unsupported metric: {metric}")
        if category not in available_categories:
            raise ValueError(f"Unsupported category: {category}")
        if view not in {"aggregated", "category", "individual"}:
            raise ValueError(f"Unsupported view: {view}")

        band = f"{metric}_band"
        order = available_metrics[metric]["labels"]

        if view == "aggregated":
            grouped = self._summarize(frame, [band])
            grouped["x"] = grouped[band]
            grouped["series"] = "All users"
        elif view == "category":
            grouped = self._summarize(frame, [category, band])
            grouped["x"] = grouped[band]
            grouped["series"] = grouped[category].astype(str)
        else:
            individual = frame.copy()
            individual["cohort"] = (
                individual["gender"].astype(str)
                + " / "
                + individual["income_bracket"].astype(str)
                + " / "
                + individual[band].astype(str)
            )
            top = individual["cohort"].value_counts().head(12).index
            grouped = self._summarize(individual[individual["cohort"].isin(top)], ["cohort"])
            grouped["x"] = grouped["cohort"]
            grouped["series"] = "Top cohorts"

        grouped["sort_key"] = grouped["x"].apply(
            lambda value: order.index(value) if value in order else 999
        )
        grouped = grouped.sort_values(["series", "sort_key", "x"])
        return {
            "view": view,
            "metric": metric,
            "metric_label": available_metrics[metric]["label"],
            "category": category,
            "points": grouped[
                ["x", "series", "count", "high_count", "medium_count", "low_count",
                 "high_share", "avg_confidence"]
            ].to_dict(orient="records"),
        }

    @staticmethod
    def _summarize(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        grouped = (
            frame.groupby(columns, observed=True)
            .agg(
                count=("prediction_label", "size"),
                high_count=("prediction_label", lambda s: int((s == "High").sum())),
                medium_count=("prediction_label", lambda s: int((s == "Medium").sum())),
                low_count=("prediction_label", lambda s: int((s == "Low").sum())),
                avg_confidence=("confidence", "mean"),
            )
            .reset_index()
        )
        grouped["high_share"] = (grouped["high_count"] / grouped["count"]).round(4)
        grouped["avg_confidence"] = grouped["avg_confidence"].fillna(0).round(4)
        return grouped

    def heatmap(self, x: str = "gender", y: str = "app_usage_time_min") -> dict[str, Any]:
        frame = self.dashboard_frame()
        available_metrics = self._available_numeric_metrics(frame)
        x_column = self._resolve_heatmap_column(x, frame)
        y_column = self._resolve_heatmap_column(y, frame)

        grouped = self._summarize(frame, [y_column, x_column])
        rows = sorted(grouped[y_column].astype(str).unique().tolist())
        cols = sorted(grouped[x_column].astype(str).unique().tolist())
        max_count = int(grouped["count"].max()) if len(grouped) else 0

        cells = []
        for row in rows:
            for col in cols:
                match = grouped[
                    (grouped[y_column].astype(str) == row)
                    & (grouped[x_column].astype(str) == col)
                ]
                if match.empty:
                    cells.append({
                        "row": row, "column": col,
                        "count": 0, "high_share": 0, "avg_confidence": 0,
                    })
                else:
                    record = match.iloc[0]
                    cells.append({
                        "row": row, "column": col,
                        "count": int(record["count"]),
                        "high_share": float(record["high_share"]),
                        "avg_confidence": float(record["avg_confidence"]),
                    })

        return {
            "x": x, "y": y,
            "x_label": CATEGORY_FIELDS.get(x, available_metrics.get(x, {}).get("label", x)),
            "y_label": CATEGORY_FIELDS.get(y, available_metrics.get(y, {}).get("label", y)),
            "rows": rows, "columns": cols,
            "max_count": max_count, "cells": cells,
        }

    def _resolve_heatmap_column(
        self, field: str, frame: pd.DataFrame | None = None
    ) -> str:
        available_metrics = self._available_numeric_metrics(frame)
        if field in available_metrics:
            return f"{field}_band"
        if field in CATEGORY_FIELDS and (frame is None or field in frame.columns):
            return field
        raise ValueError(f"Unsupported heatmap field: {field}")

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------

    def check_drift(
        self, current_data: pd.DataFrame, reference_data: pd.DataFrame | None = None
    ) -> dict[str, Any]:
        """Detect data drift using PSI and KS test per numeric column.

        Args:
            current_data: DataFrame of new/batch data.
            reference_data: Reference DataFrame (defaults to X_test_selected).

        Returns:
            Dict with per-feature drift metrics and overall status.
        """
        from performance_tracker import PerformanceTracker

        if reference_data is None:
            reference_data = self.model_service.x_test_selected()

        numeric_current = current_data.select_dtypes(include=[np.number])
        numeric_reference = reference_data.select_dtypes(include=[np.number])

        common_cols = numeric_current.columns.intersection(numeric_reference.columns)
        alerts = {}
        results = {}

        tracker = PerformanceTracker()
        for col in common_cols:
            try:
                drift = tracker.drift_report(
                    reference=numeric_reference[col].dropna().to_numpy(),
                    current=numeric_current[col].dropna().to_numpy(),
                    feature_name=col,
                )
                results[col] = drift
                if drift["drift_detected"]:
                    alerts[col] = drift
            except Exception:
                continue

        overall_status = "drift_detected" if alerts else "stable"
        return {
            "status": overall_status,
            "features_checked": len(common_cols),
            "features_drifted": len(alerts),
            "alerts": alerts,
            "details": results,
        }