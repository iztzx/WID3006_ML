"""ModelService — calibrated predictions, audit logging, caching, and OOD detection."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from connection_scoring import add_connection_features

try:
    from logging_config import logger
except ImportError:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logger = logging.getLogger("intentsight")


ROOT = Path(__file__).resolve().parents[2]
PREPROCESSED_DIR = ROOT / "Preprocessed_Data_V2"
RESULTS_DIR = ROOT / "ML_Results"


@dataclass(frozen=True)
class ArtifactStatus:
    name: str
    path: str
    exists: bool


class ModelService:
    """Loads calibrated model artifacts, provides predictions with OOD detection and audit trail."""

    required_artifacts = {
        "best_tuned_model": RESULTS_DIR / "best_tuned_model.pkl",
        "final_comparison": RESULTS_DIR / "final_comparison.csv",
        "x_test_selected": PREPROCESSED_DIR / "X_test_selected.csv",
        "y_test": PREPROCESSED_DIR / "y_test.csv",
        "target_encoder": PREPROCESSED_DIR / "target_encoder.pkl",
        "selected_features": PREPROCESSED_DIR / "selected_features.pkl",
        "scaler": PREPROCESSED_DIR / "scaler.pkl",
    }

    def __init__(self) -> None:
        """Initialise model service with training-data ranges for OOD detection."""
        self._train_ranges: dict[str, dict[str, float]] = {}
        self._model: Any | None = None
        self._init_train_ranges()

    def _init_train_ranges(self) -> None:
        """Cache training-data ranges for OOD detection."""
        try:
            x_test = self.x_test_selected()
            for col in x_test.columns:
                self._train_ranges[col] = {
                    "min": float(x_test[col].min()),
                    "max": float(x_test[col].max()),
                    "mean": float(x_test[col].mean()),
                    "std": float(x_test[col].std()),
                }
        except Exception:
            logger.warning("Could not compute training-data ranges for OOD")

    def artifact_status(self) -> list[ArtifactStatus]:
        return [
            ArtifactStatus(name=name, path=str(path), exists=path.exists())
            for name, path in self.required_artifacts.items()
        ]

    def health(self) -> dict[str, Any]:
        artifacts = self.artifact_status()
        missing = [a.name for a in artifacts if not a.exists]
        return {
            "status": "ok" if not missing else "degraded",
            "missing_artifacts": missing,
            "model_loaded": bool(getattr(self, "_model", None) is not None),
            "artifacts": [a.__dict__ for a in artifacts],
        }

    @lru_cache(maxsize=1)
    def final_comparison(self) -> pd.DataFrame:
        return pd.read_csv(RESULTS_DIR / "final_comparison.csv")

    @lru_cache(maxsize=1)
    def target_encoder(self):
        return joblib.load(PREPROCESSED_DIR / "target_encoder.pkl")

    @lru_cache(maxsize=1)
    def selected_features(self) -> list[str]:
        return list(joblib.load(PREPROCESSED_DIR / "selected_features.pkl"))

    @lru_cache(maxsize=1)
    def full_columns(self) -> list[str]:
        """Get full column names from the scaler's feature names."""
        scaler = self.scaler()
        if hasattr(scaler, "feature_names_in_"):
            return list(scaler.feature_names_in_)
        # Fallback: use selected features
        return self.selected_features()

    @lru_cache(maxsize=1)
    def scaler(self):
        return joblib.load(PREPROCESSED_DIR / "scaler.pkl")

    @lru_cache(maxsize=1)
    def y_test(self) -> pd.Series:
        frame = pd.read_csv(PREPROCESSED_DIR / "y_test.csv")
        return frame.iloc[:, 0]

    @lru_cache(maxsize=1)
    def x_test_selected(self) -> pd.DataFrame:
        return pd.read_csv(PREPROCESSED_DIR / "X_test_selected.csv")

    def _load_model(self):
        model = getattr(self, "_model", None)
        if model is None:
            model = joblib.load(RESULTS_DIR / "best_tuned_model.pkl")
            self._model = model
            logger.info("Model loaded: %s", type(model).__name__)
        return model

    def metrics(self) -> dict[str, Any]:
        comparison = self.final_comparison().copy()
        numeric_cols = [
            "CV Accuracy (mean)",
            "CV Accuracy (std)",
            "Test Accuracy",
            "Test F1 (weighted)",
            "Train Time (s)",
        ]
        for col in numeric_cols:
            if col in comparison:
                comparison[col] = pd.to_numeric(
                    comparison[col], errors="coerce"
                ).fillna(0)

        y_test = self.y_test()
        distribution = y_test.value_counts().sort_index()
        encoder = self.target_encoder()
        class_distribution = [
            {
                "encoded": int(enc),
                "label": str(encoder.inverse_transform([int(enc)])[0]),
                "count": int(cnt),
                "share": round(float(cnt / len(y_test)), 4),
            }
            for enc, cnt in distribution.items()
        ]

        majority_count = int(distribution.max())
        majority_encoded = int(distribution.idxmax())
        baseline = majority_count / len(y_test)
        best = comparison.sort_values("Test Accuracy", ascending=False).iloc[0]

        return {
            "best_model": {
                "name": str(best["Model"]),
                "test_accuracy": round(float(best["Test Accuracy"]), 4),
                "weighted_f1": round(float(best["Test F1 (weighted)"]), 4),
            },
            "majority_baseline": {
                "label": str(encoder.inverse_transform([majority_encoded])[0]),
                "accuracy": round(float(baseline), 4),
                "count": majority_count,
                "total": int(len(y_test)),
            },
            "nested_cv": {
                "model": str(best["Model"]),
                "cv_mean": round(float(best["CV Accuracy (mean)"]), 4),
                "cv_std": round(float(best["CV Accuracy (std)"]), 4),
            },
            "class_distribution": class_distribution,
            "comparison": comparison.to_dict(orient="records"),
            "defensibility_note": (
                "Model predicts a connection-readiness stage from match efficiency, "
                "conversation depth, profile completeness, swipe balance, and "
                "activity signals. The labels are product-oriented weak labels, "
                "not claims about private user intent."
            ),
        }

    @lru_cache(maxsize=128)
    def test_predictions(self) -> pd.DataFrame:
        model = self._load_model()
        x_test = self.x_test_selected()
        encoder = self.target_encoder()
        encoded = model.predict(x_test).astype(int)
        labels = encoder.inverse_transform(encoded)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(x_test)
            confidence = np.max(proba, axis=1)
        else:
            confidence = np.full(len(encoded), np.nan)

        return pd.DataFrame(
            {
                "prediction_encoded": encoded,
                "prediction_label": labels,
                "confidence": confidence,
            }
        )

    def _check_ood(self, scaled_df: pd.DataFrame) -> dict[str, Any]:
        """Flag out-of-distribution inputs based on training ranges."""
        ood_flags: dict[str, Any] = {}
        for col in scaled_df.columns:
            if col not in self._train_ranges:
                continue
            rng = self._train_ranges[col]
            val = float(scaled_df[col].iloc[0])
            # Flag if value is outside [mean ± 3*std]
            z = abs(val - rng["mean"]) / (rng["std"] + 1e-10)
            if z > 3.0:
                ood_flags[col] = {
                    "value": val,
                    "train_range": [round(rng["min"], 3), round(rng["max"], 3)],
                    "z_score": round(z, 2),
                }
        return ood_flags

    def _init_predictions_log(self) -> None:
        """Ensure the audit log file exists."""
        log_path = ROOT / "predictions_log.jsonl"
        if not log_path.exists():
            log_path.touch()

    def _log_prediction(self, payload: dict, result: dict) -> None:
        """Append prediction to audit log (thread-safe append)."""
        log_path = ROOT / "predictions_log.jsonl"
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
            "result": result,
        }
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            logger.exception("Audit log write failed for payload: %s", payload)

    def predict_scenario(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return prediction with calibration, OOD flag, and audit log."""
        model = self._load_model()
        full_columns = self.full_columns()
        selected_features = self.selected_features()

        input_df = pd.DataFrame(np.zeros((1, len(full_columns))), columns=full_columns)

        values = dict(payload)

        for col, val in values.items():
            if col in input_df.columns:
                input_df.at[0, col] = val

        engineered = add_connection_features(input_df)
        for col in full_columns:
            if col in engineered.columns:
                input_df[col] = engineered[col]

        # Scale first, then run OOD detection on scaled features
        scaled = pd.DataFrame(self.scaler().transform(input_df), columns=full_columns)
        ood_flags = self._check_ood(scaled[selected_features])

        selected = scaled[selected_features]
        encoded = model.predict(selected).astype(int)
        label = self.target_encoder().inverse_transform(encoded)[0]

        confidence = None
        calibrated_proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(selected)
            confidence = float(np.max(proba, axis=1)[0])
            calibrated_proba = {
                str(self.target_encoder().inverse_transform([i])[0]): round(float(p), 4)
                for i, p in enumerate(proba[0])
            }

        result = {
            "prediction": str(label),
            "encoded": int(encoded[0].item()),
            "confidence": round(float(confidence), 4)
            if confidence is not None
            else None,
            "calibrated_probabilities": calibrated_proba,
            "ood_flags": ood_flags if ood_flags else None,
            "note": (
                "Exploratory connection-readiness prediction. Fields not supplied "
                "default to zero, so interaction features may be incomplete."
            ),
        }

        # Audit log
        self._log_prediction(payload, result)
        return result
