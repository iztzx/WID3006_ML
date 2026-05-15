# IntentSight

**Connection Readiness Classifier** — WID3006 ML Group Assignment ("Tying the Data Knot")

IntentSight is a five-class classifier that scores dating-app users on connection readiness — from "Needs Profile Help" to "Likely To Connect" — using behavioral signals, profile attributes, and match-funnel metrics. It ships as a FastAPI service with a Streamlit dashboard, SHAP interpretability, calibration, drift detection, and a Google Colab notebook for zero-setup execution.

---

## Table of Contents

- [Problem Statement [D1]](#problem-statement)
- [Dataset [D2 / 4.2]](#dataset)
- [Target Variable [D3 / 4.3]](#target-variable)
- [Feature Engineering [D3 / 4.4]](#feature-engineering)
- [Model Pipeline [D3 / 4.5-4.8]](#model-pipeline)
- [Results [D4 / 4.9]](#results)
- [Interpretability [D4 / 4.10]](#interpretability)
- [Prediction and Deployment [D5 / 4.11-4.14]](#prediction-and-deployment)
- [API Reference](#api-reference)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Dependencies](#dependencies)
- [CI / CD](#ci--cd)

---

## Problem Statement

> Deliverable D1 — Business Understanding & Value Proposition

Modern relationships are increasingly shaped by digital interactions — swipe patterns, message frequency, emoji usage, and online presence all leave behavioral traces. Dating apps face a core challenge: **how do you identify users who are genuinely ready to connect versus those who need guidance?**

IntentSight addresses this by classifying users into five connection-readiness stages, enabling product teams to:

- **Personalize onboarding** — guide "Needs Profile Help" users with profile improvement tips
- **Optimize matching** — prioritize "Likely To Connect" users in the match queue
- **Reduce churn** — identify "Mostly Browsing" users before they disengage
- **Improve match quality** — flag "Swipes Too Freely" users for selectivity coaching

**Target:** A five-level connection-readiness stage (`connection_stage`), constructed as a weakly supervised product label (not a claim about private intent).

| Stage | Selection Criteria |
|---|---|
| **Likely To Connect** | `connection_score` rank >= 0.80 |
| **Ready To Chat** | Default middle tier |
| **Mostly Browsing** | Middle tier + `browser_issue` >= 62nd percentile + `browser_issue` >= `swipe_issue` |
| **Swipes Too Freely** | Middle tier + `swipe_issue` >= 50th percentile |
| **Needs Profile Help** | `connection_score` rank <= 0.20 |

The composite `connection_score` is a weighted blend (`connection_scoring.py:106-112`):

```
connection_score = 0.35 x match_quality
                 + 0.30 x conversation_quality
                 + 0.20 x profile_quality
                 + 0.15 x activity_level
                 - 0.10 x swipe_excess
```

**Input:** 19 raw features -> 10+ engineered features -> 20-30 selected features (95% cumulative RF importance).

**Prediction horizon:** Real-time per-user scoring (<1ms inference).

---

## Dataset

> Deliverable D2 / 4.2 — Data Source and Feature Description

**Source:** [Dating App Behavior Dataset](https://www.kaggle.com/datasets/keyushnisar/dating-app-behavior-dataset) — 50,000 synthetic records with 19 features.

**Feature types:**

| Category | Features |
|---|---|
| **Numeric (11)** | `age`, `app_usage_time_min`, `likes_received`, `mutual_matches`, `message_sent_count`, `bio_length`, `emoji_usage_rate`, `height_cm`, `weight_kg`, `profile_pics_count`, `last_active_hour` |
| **Categorical (8)** | `gender`, `income_bracket`, `education_level`, `sexual_orientation`, `location_type`, `swipe_time_of_day`, `body_type`, `interest_tags` |

**Data quality:** No critical missing values. Duplicates and nulls logged during preprocessing (`preprocess.py:62-69`).

---

## Target Variable

> Deliverable D3 / 4.3 — Target Construction and Class Definitions

**Name:** `connection_stage` — five-class ordinal target.

**Construction rationale:** Rather than using a single behavioral metric, we construct a composite `connection_score` from four quality dimensions (match, conversation, profile, activity) minus a swipe-excess penalty. Users are then ranked by percentile and assigned to stages based on funnel-informed thresholds.

**Why weak supervision?** The dataset lacks ground-truth labels for "connection readiness." We derive labels from observable behavioral signals using domain-informed rules — this is a product signal, not a claim about private intent.

**Class distribution (approximate):**

- Likely To Connect: ~20%
- Ready To Chat: ~30%
- Mostly Browsing: ~15%
- Swipes Too Freely: ~15%
- Needs Profile Help: ~20%

**Why not SMOTE before split?** SMOTE is applied inside the `imblearn.Pipeline` during cross-validation to prevent synthetic samples from leaking into validation folds.

---

## Feature Engineering

> Deliverable D3 / 4.4 — Feature Transformations and Selection

Source: `connection_scoring.py`, `feature_store.py`

### Engineered Features (12+)

| Feature | Formula / Logic | Source |
|---|---|---|
| `match_rate` | `mutual_matches / (likes_received + 1)` | connection_scoring.py:65 |
| `msg_per_match` | `message_sent_count / (mutual_matches + 1)` | connection_scoring.py:66 |
| `bmi` | `weight_kg / (height_cm / 100)^2` | connection_scoring.py:67 |
| `num_interests` | Count of parsed `interest_tags` | connection_scoring.py:40-46 |
| `profile_completeness` | `pics/6 x 0.4 + bio/300 x 0.4 + interests/5 x 0.2` | connection_scoring.py:68-72 |
| `selectivity_balance` | `1 - |swipe_ratio - 0.55| / 0.55`, clipped [0,1] | connection_scoring.py:73-75 |
| `swipe_excess` | `max(swipe_ratio - 0.70, 0)` | connection_scoring.py:76 |
| `like_to_match_gap` | `max(likes - matches, 0)` | connection_scoring.py:77 |
| `conversation_depth` | `log1p(messages) x log1p(msg_per_match)` | connection_scoring.py:78-80 |
| `social_pull` | `likes / (pics + 1)` | connection_scoring.py:81 |
| `activity_level` | `log1p(app_usage_time_min)` | connection_scoring.py:82 |
| `last_active_sin/cos` | Cyclical encoding: `sin/cos(2pi x hour/24)` | connection_scoring.py:84-87 |
| `match_quality` | Weighted: 0.45 x match_rate + 0.25 x bounded(matches) + 0.15 x selectivity + 0.15 x bounded(social_pull) | connection_scoring.py:89-94 |
| `conversation_quality` | Weighted: 0.40 x bounded(msg_per_match) + 0.30 x bounded(messages) + 0.20 x bounded(emoji) + 0.10 x bounded(usage) | connection_scoring.py:95-100 |
| `profile_quality` | Weighted: 0.60 x completeness + 0.25 x bounded(bio) + 0.15 x bounded(pics) | connection_scoring.py:101-105 |
| `connection_score` | 0.35 x match_quality + 0.30 x conversation_quality + 0.20 x profile_quality + 0.15 x activity - 0.10 x swipe_excess | connection_scoring.py:106-112 |
| `browser_issue` | 0.45 x (1-bounded(usage)) + 0.35 x (1-bounded(messages)) + 0.20 x (1-bounded(matches)) | connection_scoring.py:113-117 |
| `swipe_issue` | 0.55 x bounded(swipe_excess) + 0.45 x (1-match_rate) | connection_scoring.py:118-121 |

### Feature Selection

Random Forest (300 trees) trained on all features -> ranked by importance -> top set covering 95% cumulative importance retained (minimum 20). Saved to `Preprocessed_Data_V2/selected_features.pkl`.

---

## Model Pipeline

> Deliverable D3 / 4.5-4.8 — Split, CV, Model Selection, Tuning

Source: `train.py` — 10 steps, fully reproducible (`random_state=42`).

### Train/Test Split [4.5]

80/20 stratified split preserving class proportions. No SMOTE before split — SMOTE is applied only inside CV pipelines.

### Cross-Validation [4.6]

5-fold stratified CV on all 6 models. SMOTE applied per-fold inside `imblearn.Pipeline` to prevent leakage.

### Models Trained [4.7]

| Model | Key Hyperparameters | Train Time |
|---|---|---|
| Logistic Regression | `max_iter=2000` | ~10s |
| Random Forest | `n_estimators=300, max_depth=20, n_jobs=-1` | ~29s |
| Gradient Boosting | `n_estimators=100, max_depth=5, lr=0.1` | ~401s |
| XGBoost | `n_estimators=300, max_depth=6, subsample=0.8, colsample=0.8` | ~38s |
| LightGBM | `n_estimators=300, max_depth=8, subsample=0.8, colsample=0.8` | ~31s |
| CatBoost | `iterations=300, depth=6, lr=0.1` | ~43s |

### Hyperparameter Tuning [4.8]

Top 3 models by CV accuracy are tuned via `RandomizedSearchCV` (20 iterations, 3-fold CV). Parameter distributions defined per model in `train.py:190-229`.

### Calibration

Best model wrapped in `CalibratedClassifierCV(method="sigmoid", cv=3)` with SMOTE pipeline.

---

## Results

> Deliverable D4 / 4.9 — Evaluation Metrics and Model Comparison

From `ML_Results/final_comparison.csv`:

| Model | CV Accuracy | Test Accuracy | F1 (weighted) |
|---|---|---|---|
| **CatBoost (tuned)** | 0.9585 | **0.9653** | 0.9653 |
| CatBoost (calibrated) | — | **0.9644** | 0.9644 |
| LightGBM (tuned) | 0.9495 | 0.9554 | 0.9554 |
| LightGBM | 0.9515 | 0.9553 | 0.9553 |
| XGBoost | 0.9514 | 0.9552 | 0.9552 |
| XGBoost (tuned) | 0.9511 | 0.9547 | 0.9547 |
| Logistic Regression | 0.9464 | 0.9461 | 0.9460 |
| Gradient Boosting | 0.9345 | 0.9391 | 0.9392 |
| Random Forest | 0.9021 | 0.9051 | 0.9050 |
| Majority Baseline | 0.2084 | 0.2084 | 0.0000 |

**Production default:** CatBoost (calibrated) — 96.44% test accuracy, 0.9644 F1.

**Baseline comparison:** All models exceed the majority-class baseline (~20.8%) by >4.5x, confirming genuine predictive signal.

---

## Interpretability

> Deliverable D4 / 4.10 — SHAP Analysis and Feature Importance

### SHAP Analysis

A separate Random Forest (200 trees, `max_depth=15`) is trained on 1,000 samples for SHAP TreeExplainer. Generates beeswarm and bar plots (`shap_summary.png`, `shap_bar.png`).

Key findings:

- Behavioral features (app usage, message count, likes received) are the strongest predictors
- Demographic features (income, education) provide supplementary signal
- Engineered features (match_rate, conversation_depth) rank highly

### Feature Importance

Built-in RF feature importance plot saved to `Preprocessed_Data_V2/feature_importances.png` and `ML_Results/feature_importance_unbiased.png`.

### Calibration Validation

Reliability diagram confirms calibrated probabilities are well-aligned with actual outcomes. Saved to `ML_Results/calibration_plot.png`.

---

1

### Prediction Function [4.11]

Single-user scoring via `POST /v1/predict` with full feature engineering, calibration, and OOD detection. Also available through the Streamlit dashboard's Scenario Predictor page.

### Probability Calibration [4.12]

`CalibratedClassifierCV(method="sigmoid", cv=3)` wraps the best model. Calibration plot validates reliability.

### OOD Detection and Drift Monitoring [4.13]

**At inference** (`model_service.py:213-229`): Each input feature is compared against training-data distribution (mean, std). Z-score > 3.0 triggers an OOD flag for that feature.

**Historical** (`performance_tracker.py`): Supports Population Stability Index (PSI) and two-sample Kolmogorov-Smirnov test for monitoring feature drift over time.

- PSI < 0.1 -> no significant drift
- PSI 0.1-0.25 -> moderate drift
- PSI > 0.25 -> significant drift
- KS p-value < 0.05 -> significant drift

### Fairness Considerations

The model does not use protected attributes (gender, sexual_orientation) as direct predictors after one-hot encoding. Predictions are product signals for intervention design, not claims about user worth or intent.

---

## Architecture

```
Data Sources (CSV)
       |
       v
 preprocess.py          <- target construction, feature engineering, scaling, selection
       |
       v
 train.py               <- 6 models, 5-fold CV, tuning (top 3), calibration, SHAP
       |                   (with autoML comparison: auto-sklearn / FLAML / PyCaret)
       v (artifacts)
 model_service.py       <- calibrated model loading, OOD detection (z>3), audit log
 data_service.py        <- cohort analysis, heatmap, drift detection (PSI, KS), options
       |
       v
 app/main.py            <- FastAPI, /v1/ endpoints + /api/ backward-compat aliases
       |
   +---+---+
   |       |
Streamlit  Docker Compose (Tier 3)
(Tier 1)   api:8000 + dashboard:8501
port 8501
```

---

## API Reference

All endpoints versioned under `/v1/`. Backward-compatible `/api/` aliases also available.

**`GET /v1/health`** — Returns service status, artifact availability, model load state.

**`GET /v1/metrics`** — Full model comparison table, majority baseline, class distribution, nested CV results.

**`GET /v1/options`** — Available metrics and categories for cohort/heatmap queries.

**`GET /v1/cohorts`** — Cohort analysis with three view modes: `aggregated`, `category`, `individual`.

**`GET /v1/heatmap`** — Cross-tabulated connection readiness by two dimensions.

**`POST /v1/predict`** — Single-user scoring with calibration, OOD detection, audit logging.

Request body (all fields optional, defaults to 0):

```json
{
  "app_usage_time_min": 120,
  "swipe_right_ratio": 0.5,
  "likes_received": 50,
  "mutual_matches": 10,
  "message_sent_count": 30,
  "bio_length": 140,
  "emoji_usage_rate": 0.3,
  "height_cm": 170,
  "weight_kg": 70,
  "profile_pics_count": 3,
  "last_active_hour": 12
}
```

Response:

```json
{
  "prediction": "Likely To Connect",
  "encoded": 0,
  "confidence": 0.9234,
  "calibrated_probabilities": {
    "Likely To Connect": 0.9234,
    "Ready To Chat": 0.0366,
    "Mostly Browsing": 0.02,
    "Swipes Too Freely": 0.01,
    "Needs Profile Help": 0.01
  },
  "ood_flags": null,
  "note": "Exploratory connection-readiness prediction..."
}
```

**`POST /v1/predict/batch`** — Batch scoring. Accepts a list of scenarios, returns predictions with per-item index.

Every prediction is logged to `predictions_log.jsonl` with full input, output, and OOD flags. Inputs flagged as out-of-distribution (z-score > 3 on any feature) include `ood_flags` with per-feature details.

---

## Streamlit Dashboard

Six pages (source: `streamlit_app.py`):

| Page | Contents |
|---|---|
| **Overview** | KPI tiles (best model, accuracy, F1, classes), label distribution pie chart, all-models accuracy bar chart |
| **Model Comparison** | Full comparison table, accuracy vs F1 scatter plot (bubble = train time), nested CV results, calibration curve |
| **Feature Importance** | SHAP beeswarm + bar plots, built-in RF feature importance, top-15 feature correlation heatmap |
| **Scenario Predictor** | Interactive sliders for all 11 raw features -> real-time prediction with class probabilities and confidence gauge. OOD warnings shown when any feature >3 sigma from training mean |
| **Data Explorer** | Raw dataset preview, summary statistics, per-feature distributions, relationship-intent breakdown |
| **Audit Log** | All logged predictions with timestamps, class distribution of predictions, confidence histogram |

---

## Quick Start

### Google Colab (Zero Setup) [Recommended for Reviewers]

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iztzx/WID3006_ML/blob/main/IntentSight_Colab.ipynb)

Full pipeline in-browser, ~10-15 min on free T4 GPU. Handles installs, EDA, preprocessing, training, tuning, SHAP, calibration, and artifact export. Self-contained — no repo clone needed.

### Local (Tiers 1 or 2)

```bash
git clone https://github.com/iztzx/WID3006_ML.git && cd WID3006_ML
python -m pip install -r requirements.txt

python preprocess.py          # generates Preprocessed_Data_V2/
python train.py               # generates ML_Results/

streamlit run streamlit_app.py          # Tier 1: dashboard, port 8501
# -- or --
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000  # Tier 2: API
```

### Docker (Tier 3)

```bash
docker compose up --build
```

Two containers mount `ML_Results/`, `Preprocessed_Data_V2/`, and dataset CSVs as read-only volumes. Dashboard waits for API healthcheck before starting.

### Deployment Tiers Summary

| Tier | Stack | Entry Point |
|---|---|---|
| 1 | Streamlit | `streamlit run streamlit_app.py` (port 8501) |
| 2 | FastAPI + Uvicorn | `uvicorn app.main:app --reload --port 8000` |
| 3 | Docker Compose | `docker compose up --build` (api:8000 + dashboard:8501) |

---

## Project Structure

```
.
├── app/
│   ├── main.py                 FastAPI, Pydantic models, lifespan
│   └── services/
│       ├── model_service.py    Model loading, scoring, OOD, audit log
│       └── data_service.py     Cohorts, heatmap, drift, options
├── streamlit_app.py            Dashboard (6 pages)
├── connection_scoring.py       Target construction + feature engineering
├── feature_store.py            Feature registry (FeatureDefinition dataclass)
├── preprocess.py               Load -> validate -> target -> engineer -> encode -> scale/split/select
├── train.py                    6 models, CV, tuning, calibration, SHAP, artifacts
├── performance_tracker.py      PSI, KS test, metric history (JSONL)
├── logging_config.py           Structured JSON logging
├── IntentSight_Colab.ipynb     Zero-setup Colab notebook (self-contained)
├── docker-compose.yml          Tier 3: API + dashboard
├── Dockerfile / Dockerfile.streamlit
├── requirements.txt
├── tests/
│   └── test_api.py             22 integration tests
├── .github/
│   └── workflows/
│       └── ci.yml              Ruff lint + mypy + pytest
├── ML_Results/                  <- generated by train.py
│   ├── best_tuned_model.pkl
│   ├── final_comparison.csv
│   ├── target_encoder.pkl
│   ├── selected_features.pkl
│   ├── scaler.pkl
│   ├── calibration_plot.png
│   ├── shap_summary.png
│   ├── shap_bar.png
│   ├── feature_importance.png
│   ├── feature_importance_unbiased.png
│   ├── final_comparison.png
│   └── classification_report.txt
├── Preprocessed_Data_V2/       <- generated by preprocess.py
│   ├── X_train_selected_unresampled.csv
│   ├── X_test_selected.csv
│   ├── y_train_original.csv
│   ├── y_test.csv
│   ├── scaler.pkl
│   ├── target_encoder.pkl
│   ├── selected_features.pkl
│   └── feature_importances.png
└── Behaviour_Extended_Dataset.csv  (source data, not committed)
```

---

## Testing

```bash
python -m pytest tests/ -v                    # 22 tests
python -m pytest tests/ -v --cov=app         # with coverage
```

The 22 tests (source: `tests/test_api.py`) cover:

- Health endpoint and degraded state on missing artifacts
- Backward-compatible `/api/` route aliases
- Metrics: best model, baseline, nested CV fields, comparison table
- Cohorts: all 3 view modes, invalid metric/category/view -> 400
- Heatmap: valid and invalid field -> 400
- Options: metrics and categories present
- Single prediction: valid input, minimal input, field validation -> 422, boundary values (0 and max), backward-compatible alias
- Batch prediction: 2 scenarios, empty list, invalid field in scenario -> 422
- DataService: fallback dataset without BMI column
- Error handling: degraded health, global 500 handler

---

## Dependencies

| Category | Packages |
|---|---|
| **Core** | fastapi >=0.110, uvicorn[standard] >=0.27, pandas >=2.0, numpy >=1.24, scikit-learn >=1.3, scipy >=1.11, imbalanced-learn >=0.11, joblib >=1.3 |
| **Models** | xgboost >=2.0, lightgbm >=4.0, catboost >=1.2 |
| **Dashboard** | streamlit >=1.30, plotly >=5.18 |
| **Visualization** | matplotlib >=3.7, seaborn >=0.13 |
| **Interpretability** | shap >=0.44 |
| **Logging** | python-json-logger >=2.0 |
| **Testing** | pytest >=8.0, pytest-timeout >=2.2, httpx >=0.26 |
| **Dev** | ruff >=0.4, mypy >=1.10 |

---

## CI / CD

GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push to `main` and PRs to `main`/`develop`:

1. **Lint:** Ruff check + format
2. **Type check:** mypy on `app/`, `tests/`, `preprocess.py`, `train.py`, `feature_store.py`, `performance_tracker.py`, `logging_config.py`
3. **Test:** pytest, 300s timeout, Python 3.11 on `ubuntu-latest`

---

## Requirement Mapping

Quick reference for graders — maps each assignment deliverable to the relevant code/doc section:

| Deliverable | Description | Where to Find |
|---|---|---|
| **D1** | Problem framing, business value | [Problem Statement](#problem-statement) |
| **D2** | Data understanding, EDA | [Dataset](#dataset), Colab Section 3 |
| **D3** | Methodology | [Target Variable](#target-variable), [Feature Engineering](#feature-engineering), [Model Pipeline](#model-pipeline) |
| **D4** | Results, interpretation | [Results](#results), [Interpretability](#interpretability) |
| **D5** | Deployment, prediction | [Prediction and Deployment](#prediction-and-deployment), [Quick Start](#quick-start) |
| **4.1** | AutoML comparison | Colab Section 10 (auto-sklearn / FLAML / PyCaret) |
| **4.2** | Dataset description | [Dataset](#dataset) |
| **4.3** | Target variable | [Target Variable](#target-variable), `connection_scoring.py` |
| **4.4** | Feature engineering | [Feature Engineering](#feature-engineering) |
| **4.5** | Train/test split | [Model Pipeline](#model-pipeline), `preprocess.py:198-200` |
| **4.6** | Cross-validation + SMOTE | [Model Pipeline](#model-pipeline), `train.py:138-168` |
| **4.7** | Model selection | [Model Pipeline](#model-pipeline), `train.py:88-134` |
| **4.8** | Hyperparameter tuning | [Model Pipeline](#model-pipeline), `train.py:180-271` |
| **4.9** | Evaluation metrics | [Results](#results), `ML_Results/classification_report.txt` |
| **4.10** | Interpretability (SHAP) | [Interpretability](#interpretability), `ML_Results/shap_summary.png` |
| **4.11** | Prediction function | [API Reference](#api-reference), `app/services/model_service.py:251-304` |
| **4.12** | Calibration and validation | [Calibration](#calibration), `ML_Results/calibration_plot.png` |
| **4.13** | Fairness / OOD / Drift | [OOD Detection and Drift Monitoring](#ood-detection-and-drift-monitoring) |
| **4.14** | Multi-tier deployment | [Deployment Tiers Summary](#deployment-tiers-summary), Docker Compose |

---

*WID3006 Machine Learning group project at Universiti Malaya — "Tying the Data Knot."*
