"""Pre-compute SHAP values from the best CatBoost model for interactive dashboard.

Run once: python compute_shap_values.py
Produces: ML_Results/shap_values.csv, ML_Results/shap_feature_importance.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "ML_Results"
PREP_DIR = ROOT / "Preprocessed_Data_V2"
N_SAMPLES = 1000


def main() -> None:
    logger.info("Loading model and data...")
    pipeline = joblib.load(RESULTS_DIR / "best_tuned_model.pkl")
    X_train = pd.read_csv(PREP_DIR / "X_train_selected_unresampled.csv")
    feature_names = list(X_train.columns)

    # Extract CatBoost from Pipeline(SMOTE -> CalibratedClassifierCV(CatBoost))
    calibrated = pipeline.named_steps["model"]
    catboost_model = calibrated.calibrated_classifiers_[0].estimator

    # Sample rows for SHAP computation
    sample = X_train.sample(n=min(N_SAMPLES, len(X_train)), random_state=42)
    logger.info("Computing SHAP values on %d samples...", len(sample))

    # CatBoost native SHAP — fast, no shap library needed at runtime
    from catboost import Pool

    pool = Pool(sample, feature_names=feature_names)
    shap_values = catboost_model.get_feature_importance(type="ShapValues", data=pool)

    # shap_values shape: (n_samples, n_features + 1) — last col is base value
    # For multiclass: shape is (n_samples, n_classes * (n_features + 1))
    shap_arr = np.array(shap_values)

    if shap_arr.ndim == 2 and shap_arr.shape[1] == len(feature_names) + 1:
        # Binary or regression — single set of SHAP values
        shap_df = pd.DataFrame(shap_arr[:, :-1], columns=feature_names)
        shap_df.to_csv(RESULTS_DIR / "shap_values.csv", index=False)
        mean_abs = np.abs(shap_arr[:, :-1]).mean(axis=0)
    elif shap_arr.ndim == 3:
        # Multiclass: (n_samples, n_classes, n_features + 1)
        n_classes = shap_arr.shape[1]
        # Save per-class SHAP values
        for cls_idx in range(n_classes):
            cls_shap = shap_arr[cls_idx, :, :-1]
            cls_df = pd.DataFrame(cls_shap, columns=feature_names)
            cls_df.to_csv(RESULTS_DIR / f"shap_values_class{cls_idx}.csv", index=False)
        # Overall importance = mean across classes of mean |SHAP|
        mean_abs = np.abs(shap_arr[:, :, :-1]).mean(axis=(0, 1))
        # Also save a combined file (average across classes)
        combined = np.abs(shap_arr[:, :, :-1]).mean(axis=1)
        pd.DataFrame(combined, columns=feature_names).to_csv(
            RESULTS_DIR / "shap_values.csv", index=False
        )
    else:
        # Flat 2D: (n_samples, n_features) — no base value column
        shap_df = pd.DataFrame(shap_arr, columns=feature_names)
        shap_df.to_csv(RESULTS_DIR / "shap_values.csv", index=False)
        mean_abs = np.abs(shap_arr).mean(axis=0)

    # Feature importance ranking
    importance_df = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": mean_abs}
    ).sort_values("mean_abs_shap", ascending=False)
    importance_df.to_csv(RESULTS_DIR / "shap_feature_importance.csv", index=False)

    logger.info("Saved shap_values.csv and shap_feature_importance.csv")
    logger.info("Top 10 features:\n%s", importance_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
