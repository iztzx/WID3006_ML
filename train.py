# =============================================================================
# ML TRAINING — Engagement Level Pipeline
# WID3006 ML Group Assignment: "Tying the (Data) Knot"
# =============================================================================
# 6 models, 5-fold CV, RandomizedSearchCV on top 3, SHAP interpretability,
# calibrated best model. Target: 3-class engagement level.
# =============================================================================

import logging
import os
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import seaborn as sns
from scipy.stats import randint, uniform
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
import shap
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

try:
    from logging_config import logger
except ImportError:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logger = logging.getLogger("intentsight")

warnings.filterwarnings("ignore", category=FutureWarning)

# --- CONFIGURATION ---
ROOT = Path(__file__).resolve().parent
PREPROCESSED_DIR = ROOT / "Preprocessed_Data_V2"
OUTPUT_PATH = ROOT / "ML_Results"
os.makedirs(OUTPUT_PATH, exist_ok=True)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# =============================================================================
# STEP 1: LOAD PREPROCESSED ARTIFACTS
# =============================================================================
logger.info("STEP 1: Loading preprocessed data...")

X_train = pd.read_csv(PREPROCESSED_DIR / "X_train_selected_unresampled.csv")
X_test = pd.read_csv(PREPROCESSED_DIR / "X_test_selected.csv")
y_train = pd.read_csv(PREPROCESSED_DIR / "y_train_original.csv").values.ravel()
y_test = pd.read_csv(PREPROCESSED_DIR / "y_test.csv").values.ravel()

target_encoder = joblib.load(PREPROCESSED_DIR / "target_encoder.pkl")
selected_features = joblib.load(PREPROCESSED_DIR / "selected_features.pkl")
scaler = joblib.load(PREPROCESSED_DIR / "scaler.pkl")

n_classes = len(target_encoder.classes_)
logger.info(
    "  Train: %s | Test: %s | Classes: %s",
    X_train.shape,
    X_test.shape,
    list(target_encoder.classes_),
)


# =============================================================================
# STEP 2: TRAIN 6 BASE MODELS WITH 5-FOLD CV
# =============================================================================
logger.info("STEP 2: Training 6 base models...")


def make_pipeline(model):
    """Wrap model in SMOTE-in-pipeline (leakage-free CV)."""
    return Pipeline([("smote", SMOTE(random_state=42)), ("model", model)])


models = {
    "Logistic Regression": make_pipeline(
        LogisticRegression(max_iter=2000, random_state=42)
    ),
    "Random Forest": make_pipeline(
        RandomForestClassifier(
            n_estimators=300, max_depth=20, random_state=42, n_jobs=-1
        )
    ),
    "Gradient Boosting": make_pipeline(
        GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        )
    ),
    "XGBoost": make_pipeline(
        XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            eval_metric="mlogloss",
        )
    ),
    "LightGBM": make_pipeline(
        LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    ),
    "CatBoost": make_pipeline(
        CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.1, random_state=42, verbose=0
        )
    ),
}

base_results = []

for name, pipeline in models.items():
    logger.info("  Training %s...", name)
    t0 = pd.Timestamp.now()

    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average="weighted")
    elapsed = (pd.Timestamp.now() - t0).total_seconds()

    base_results.append(
        {
            "Model": name,
            "CV Accuracy (mean)": round(cv_scores.mean(), 4),
            "CV Accuracy (std)": round(cv_scores.std(), 4),
            "Test Accuracy": round(test_acc, 4),
            "Test F1 (weighted)": round(test_f1, 4),
            "Train Time (s)": round(elapsed, 1),
        }
    )
    logger.info(
        "    CV: %.4f +/- %.4f | Test: %.4f | F1: %.4f | %.1fs",
        cv_scores.mean(),
        cv_scores.std(),
        test_acc,
        test_f1,
        elapsed,
    )

base_results_df = pd.DataFrame(base_results).sort_values(
    "Test Accuracy", ascending=False
)
print("\n--- Base Model Comparison ---")
print(base_results_df.to_string(index=False))


# =============================================================================
# STEP 3: HYPERPARAMETER TUNING (top 3 via RandomizedSearchCV)
# =============================================================================
logger.info("STEP 3: Tuning top 3 models with RandomizedSearchCV...")

top3_names = base_results_df.head(3)["Model"].tolist()
logger.info("  Top 3: %s", top3_names)

param_distributions = {
    "Random Forest": {
        "model__n_estimators": randint(200, 600),
        "model__max_depth": randint(10, 40),
        "model__min_samples_split": randint(2, 15),
        "model__min_samples_leaf": randint(1, 10),
    },
    "Gradient Boosting": {
        "model__n_estimators": randint(100, 400),
        "model__max_depth": randint(3, 10),
        "model__learning_rate": uniform(0.01, 0.29),
        "model__subsample": uniform(0.6, 0.4),
    },
    "XGBoost": {
        "model__n_estimators": randint(200, 600),
        "model__max_depth": randint(3, 10),
        "model__learning_rate": uniform(0.01, 0.29),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.6, 0.4),
        "model__reg_alpha": uniform(0.001, 9.999),
    },
    "LightGBM": {
        "model__n_estimators": randint(200, 600),
        "model__max_depth": randint(4, 12),
        "model__learning_rate": uniform(0.01, 0.29),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.6, 0.4),
    },
    "CatBoost": {
        "model__iterations": randint(200, 600),
        "model__depth": randint(4, 10),
        "model__learning_rate": uniform(0.01, 0.29),
        "model__l2_leaf_reg": uniform(0.001, 9.999),
    },
    "Logistic Regression": {
        "model__C": uniform(0.01, 99.99),
        "model__penalty": ["l2"],
        "model__solver": ["lbfgs"],
    },
}

tuned_results = {}
tuned_pipelines = {}

for name in top3_names:
    logger.info("  Tuning %s...", name)
    t0 = pd.Timestamp.now()

    search = RandomizedSearchCV(
        models[name],
        param_distributions=param_distributions[name],
        n_iter=20,
        cv=3,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)

    best_pipe = search.best_estimator_
    y_pred = best_pipe.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average="weighted")
    elapsed = (pd.Timestamp.now() - t0).total_seconds()

    tuned_results[name] = {
        "best_params": search.best_params_,
        "cv_score": round(search.best_score_, 4),
        "test_acc": round(test_acc, 4),
        "test_f1": round(test_f1, 4),
    }
    tuned_pipelines[name] = best_pipe

    logger.info(
        "    Best CV: %.4f | Test: %.4f | F1: %.4f | %.1fs | %s",
        search.best_score_,
        test_acc,
        test_f1,
        elapsed,
        search.best_params_,
    )


# =============================================================================
# STEP 4: SELECT BEST MODEL & CALIBRATE
# =============================================================================
logger.info("STEP 4: Selecting best model and calibrating...")

all_candidates = []
for r in base_results:
    all_candidates.append(
        {"name": r["Model"], "test_acc": r["Test Accuracy"], "source": "base"}
    )
for name, r in tuned_results.items():
    all_candidates.append({"name": name, "test_acc": r["test_acc"], "source": "tuned"})

best_candidate = max(all_candidates, key=lambda x: x["test_acc"])
best_name = best_candidate["name"]
logger.info(
    "  Best: %s (%s) — Test Acc: %.4f",
    best_name,
    best_candidate["source"],
    best_candidate["test_acc"],
)

if best_name in tuned_pipelines:
    best_pipeline = tuned_pipelines[best_name]
else:
    best_pipeline = models[best_name]

# Calibrate
base_estimator = best_pipeline.named_steps["model"]
calibrated_pipeline = Pipeline(
    [
        ("smote", SMOTE(random_state=42)),
        ("model", CalibratedClassifierCV(base_estimator, method="sigmoid", cv=3)),
    ]
)
calibrated_pipeline.fit(X_train, y_train)

y_pred_cal = calibrated_pipeline.predict(X_test)
y_proba_cal = calibrated_pipeline.predict_proba(X_test)
cal_acc = accuracy_score(y_test, y_pred_cal)
cal_f1 = f1_score(y_test, y_pred_cal, average="weighted")

logger.info("  Calibrated — Accuracy: %.4f, F1: %.4f", cal_acc, cal_f1)

# Reliability diagram (for first class)
fraction_of_positives, mean_predicted_value = calibration_curve(
    (y_test == 0).astype(int), y_proba_cal[:, 0], n_bins=10, strategy="uniform"
)
plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly calibrated")
plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=best_name)
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.title(f"Calibration Plot — {best_name}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_PATH / "calibration_plot.png", dpi=150)
plt.close()


# =============================================================================
# STEP 5: SHAP INTERPRETABILITY
# =============================================================================
logger.info("STEP 5: SHAP interpretability...")

shap_model = RandomForestClassifier(
    n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
)
shap_model.fit(X_train, y_train)

SHAP_SAMPLE_SIZE = 1000
shap_sample = X_train.sample(n=min(SHAP_SAMPLE_SIZE, len(X_train)), random_state=42)
explainer = shap.TreeExplainer(shap_model)
shap_values = explainer.shap_values(shap_sample)

# For multi-class, use the first class or mean absolute
if isinstance(shap_values, list):
    sv = shap_values[0]
else:
    sv = shap_values

for plot_type, suffix, title in [
    (None, "shap_summary", "Beeswarm"),
    ("bar", "shap_bar", "Bar"),
]:
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, shap_sample, plot_type=plot_type, show=False, max_display=20)
    plt.title(f"SHAP Feature Importance ({title})", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / f"{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close()

del shap_values, explainer
logger.info("  SHAP plots saved.")


# =============================================================================
# STEP 6: FEATURE IMPORTANCE
# =============================================================================
logger.info("STEP 6: Feature importance...")

importances = shap_model.feature_importances_
importance_df = pd.DataFrame(
    {
        "Feature": X_train.columns,
        "Importance": importances,
    }
).sort_values("Importance", ascending=False)

importance_df["Cumulative"] = importance_df["Importance"].cumsum()
threshold = 0.95 * importance_df["Importance"].sum()
selected_unbiased = importance_df[importance_df["Cumulative"] <= threshold][
    "Feature"
].tolist()
if len(selected_unbiased) < 20:
    selected_unbiased = importance_df.head(20)["Feature"].tolist()

plt.figure(figsize=(10, 8))
sns.barplot(
    x="Importance",
    y="Feature",
    data=importance_df.head(20),
    palette="magma",
    hue="Feature",
    legend=False,
)
plt.title("Top 20 Feature Importances", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_PATH / "feature_importance_unbiased.png", dpi=150)
plt.close()

logger.info("  Selected %d / %d features.", len(selected_unbiased), X_train.shape[1])


# =============================================================================
# STEP 7: FINAL COMPARISON TABLE
# =============================================================================
logger.info("STEP 7: Building final comparison table...")

final_results = base_results.copy()

for name, r in tuned_results.items():
    final_results.append(
        {
            "Model": f"{name} (tuned)",
            "CV Accuracy (mean)": r["cv_score"],
            "CV Accuracy (std)": 0,
            "Test Accuracy": r["test_acc"],
            "Test F1 (weighted)": r["test_f1"],
            "Train Time (s)": 0,
        }
    )

final_results.append(
    {
        "Model": f"{best_name} (calibrated)",
        "CV Accuracy (mean)": 0,
        "CV Accuracy (std)": 0,
        "Test Accuracy": round(cal_acc, 4),
        "Test F1 (weighted)": round(cal_f1, 4),
        "Train Time (s)": 0,
    }
)

# Majority baseline
final_results.append(
    {
        "Model": "Majority Baseline",
        "CV Accuracy (mean)": round(max(np.bincount(y_train)) / len(y_train), 4),
        "CV Accuracy (std)": 0,
        "Test Accuracy": round(max(np.bincount(y_test)) / len(y_test), 4),
        "Test F1 (weighted)": 0,
        "Train Time (s)": 0,
    }
)

final_df = (
    pd.DataFrame(final_results)
    .sort_values("Test Accuracy", ascending=False)
    .reset_index(drop=True)
)

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)
print(final_df.to_string(index=False))

final_df.to_csv(OUTPUT_PATH / "final_comparison.csv", index=False)


# =============================================================================
# STEP 8: FINAL VISUALIZATION
# =============================================================================
logger.info("STEP 8: Generating visualizations...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.barplot(
    x="Test Accuracy",
    y="Model",
    data=final_df,
    palette="viridis",
    ax=axes[0],
    hue="Model",
    legend=False,
)
axes[0].set_title("Test Accuracy by Model", fontsize=14)
axes[0].set_xlim(final_df["Test Accuracy"].min() - 0.02, 1.0)

nonzero_f1 = final_df[final_df["Test F1 (weighted)"] > 0].copy()
if len(nonzero_f1) > 0:
    sns.barplot(
        x="Test F1 (weighted)",
        y="Model",
        data=nonzero_f1,
        palette="magma",
        ax=axes[1],
        hue="Model",
        legend=False,
    )
    axes[1].set_title("Test F1 (Weighted) by Model", fontsize=14)
    axes[1].set_xlim(nonzero_f1["Test F1 (weighted)"].min() - 0.02, 1.0)

plt.tight_layout()
plt.savefig(OUTPUT_PATH / "final_comparison.png", dpi=150)
plt.close()


# =============================================================================
# STEP 9: CLASSIFICATION REPORT
# =============================================================================
logger.info("STEP 9: Classification report...")

y_pred_best = best_pipeline.predict(X_test)
class_names = [str(c) for c in target_encoder.classes_]
report = classification_report(y_test, y_pred_best, target_names=class_names)
print("\n--- Classification Report (Best Model) ---")
print(report)

with open(OUTPUT_PATH / "classification_report.txt", "w") as f:
    f.write(f"Best Model: {best_name}\n\n")
    f.write(report)


# =============================================================================
# STEP 10: SAVE ARTIFACTS
# =============================================================================
logger.info("STEP 10: Saving artifacts...")

joblib.dump(calibrated_pipeline, OUTPUT_PATH / "best_tuned_model.pkl")
joblib.dump(target_encoder, OUTPUT_PATH / "target_encoder.pkl")
joblib.dump(selected_unbiased, OUTPUT_PATH / "selected_features.pkl")
joblib.dump(scaler, OUTPUT_PATH / "scaler.pkl")

logger.info("  Artifacts saved to %s", OUTPUT_PATH)
for f in sorted(OUTPUT_PATH.iterdir()):
    logger.info("    %s", f.name)


# =============================================================================
# DONE
# =============================================================================
logger.info("=" * 60)
logger.info(
    "ML Pipeline Complete! Best: %s (calibrated) — Acc: %.4f", best_name, cal_acc
)
logger.info("=" * 60)
