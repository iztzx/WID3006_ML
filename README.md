# ConnectionLens

**Connection Readiness Classifier** — WID3006 ML Group Assignment ("Tying the Data Knot")

ConnectionLens is a five-class classifier that scores dating-app users on connection readiness — from "Needs Profile Help" to "Likely To Connect" — using behavioral signals, profile attributes, and match-funnel metrics. The entire pipeline runs in a single self-contained Google Colab notebook with SHAP interpretability, probability calibration, and an interactive Streamlit dashboard.

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
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Requirement Mapping](#requirement-mapping)

---

## Problem Statement

> Deliverable D1 — Business Understanding & Value Proposition

Modern relationships are increasingly shaped by digital interactions — swipe patterns, message frequency, emoji usage, and online presence all leave behavioral traces. Dating apps face a core challenge: **how do you identify users who are genuinely ready to connect versus those who need guidance?**

ConnectionLens addresses this by classifying users into five connection-readiness stages, enabling product teams to:

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

The composite `connection_score` is a weighted blend:

```
connection_score = 0.35 x match_quality
                 + 0.30 x conversation_quality
                 + 0.20 x profile_quality
                 + 0.15 x activity_level
                 - 0.10 x swipe_excess
```

**Input:** 25 raw features -> 16+ engineered features -> 20-30 selected features (95% cumulative RF importance).

---

## Dataset

> Deliverable D2 / 4.2 — Data Source and Feature Description

**Source:** [Dating App Behavior Dataset](https://www.kaggle.com/datasets/keyushnisar/dating-app-behavior-dataset) — 50,000 synthetic records with 25 features.

**Feature types:**

| Category | Features |
|---|---|
| **Numeric (11)** | `age`, `app_usage_time_min`, `likes_received`, `mutual_matches`, `message_sent_count`, `bio_length`, `emoji_usage_rate`, `height_cm`, `weight_kg`, `profile_pics_count`, `last_active_hour` |
| **Categorical (8)** | `gender`, `income_bracket`, `education_level`, `sexual_orientation`, `location_type`, `swipe_time_of_day`, `body_type`, `interest_tags` |
| **Derived (6)** | `relationship_intent`, `match_outcome`, `swipe_right_ratio`, `engagement_score`, `zodiac`, `app_usage_time_label` |

**Data quality:** No critical missing values. Duplicates and nulls handled during preprocessing.

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

### Engineered Features (16+)

| Feature | Formula / Logic |
|---|---|
| `match_rate` | `mutual_matches / (likes_received + 1)` |
| `msg_per_match` | `message_sent_count / (mutual_matches + 1)` |
| `bmi` | `weight_kg / (height_cm / 100)^2` |
| `num_interests` | Count of parsed `interest_tags` |
| `profile_completeness` | `pics/6 x 0.4 + bio/300 x 0.4 + interests/5 x 0.2` |
| `selectivity_balance` | `1 - |swipe_ratio - 0.55| / 0.55`, clipped [0,1] |
| `swipe_excess` | `max(swipe_ratio - 0.70, 0)` |
| `like_to_match_gap` | `max(likes - matches, 0)` |
| `conversation_depth` | `log1p(messages) x log1p(msg_per_match)` |
| `social_pull` | `likes / (pics + 1)` |
| `activity_level` | `log1p(app_usage_time_min)` |
| `last_active_sin/cos` | Cyclical encoding: `sin/cos(2pi x hour/24)` |
| `match_quality` | Weighted: 0.45 x match_rate + 0.25 x bounded(matches) + 0.15 x selectivity + 0.15 x bounded(social_pull) |
| `conversation_quality` | Weighted: 0.40 x bounded(msg_per_match) + 0.30 x bounded(messages) + 0.20 x bounded(emoji) + 0.10 x bounded(usage) |
| `profile_quality` | Weighted: 0.60 x completeness + 0.25 x bounded(bio) + 0.15 x bounded(pics) |
| `connection_score` | 0.35 x match_quality + 0.30 x conversation_quality + 0.20 x profile_quality + 0.15 x activity - 0.10 x swipe_excess |
| `browser_issue` | 0.45 x (1-bounded(usage)) + 0.35 x (1-bounded(messages)) + 0.20 x (1-bounded(matches)) |
| `swipe_issue` | 0.55 x bounded(swipe_excess) + 0.45 x (1-match_rate) |

### Feature Selection

Random Forest (300 trees) trained on all features -> ranked by importance -> top set covering 95% cumulative importance retained (minimum 20).

---

## Model Pipeline

> Deliverable D3 / 4.5-4.8 — Split, CV, Model Selection, Tuning

Fully reproducible (`random_state=42`). All parameters centralized in the `CONFIG` dict.

### Train/Test Split [4.5]

80/20 stratified split preserving class proportions. No SMOTE before split — SMOTE is applied only inside CV pipelines.

### Cross-Validation [4.6]

5-fold stratified CV on all 6 models. SMOTE applied per-fold inside `imblearn.Pipeline` to prevent leakage.

### Models Trained [4.7]

| Model | Key Hyperparameters |
|---|---|
| Logistic Regression | `max_iter=2000` |
| Random Forest | `n_estimators=300, max_depth=20` |
| Gradient Boosting | `n_estimators=100, max_depth=5, lr=0.1` |
| XGBoost | `n_estimators=300, max_depth=6, subsample=0.8, colsample=0.8` |
| LightGBM | `n_estimators=300, max_depth=8, subsample=0.8, colsample=0.8` |
| CatBoost | `iterations=300, depth=6, lr=0.1` |

### Hyperparameter Tuning [4.8]

Top 3 models by CV accuracy are tuned via `RandomizedSearchCV` (20 iterations, 3-fold CV).

### Calibration

Best model wrapped in `CalibratedClassifierCV(method="sigmoid", cv=3)` with SMOTE pipeline.

---

## Results

> Deliverable D4 / 4.9 — Evaluation Metrics and Model Comparison

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

A separate Random Forest (200 trees, `max_depth=15`) is trained on 1,000 samples for SHAP TreeExplainer. Generates beeswarm and bar plots.

Key findings:

- Behavioral features (app usage, message count, likes received) are the strongest predictors
- Demographic features (income, education) provide supplementary signal
- Engineered features (match_rate, conversation_depth) rank highly

### Feature Importance

Built-in RF feature importance plot generated during preprocessing.

### Calibration Validation

Reliability diagram confirms calibrated probabilities are well-aligned with actual outcomes.

---

## Prediction and Deployment

> Deliverable D5 / 4.11-4.14

### Prediction Function [4.11]

Single-user scoring via the Streamlit dashboard's Scenario Predictor page. Adjust 11 profile sliders and get real-time connection-stage predictions with class probabilities.

### Probability Calibration [4.12]

`CalibratedClassifierCV(method="sigmoid", cv=3)` wraps the best model. Calibration plot validates reliability.

### OOD Detection [4.13]

Sensitivity analysis in the dashboard perturbs each input feature by +/-10% and shows which changes would most affect the prediction.

### Fairness Considerations

The model does not use protected attributes (gender, sexual_orientation) as direct predictors after one-hot encoding. Predictions are product signals for intervention design, not claims about user worth or intent.

---

## Quick Start

### Google Colab (Zero Setup) — Recommended

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iztzx/WID3006_ML/blob/main/ConnectionLens_Colab.ipynb)

Full pipeline in-browser, ~10-15 min on free T4 GPU. Handles installs, EDA, preprocessing, training, tuning, SHAP, calibration, and artifact export. Self-contained — no repo clone needed.

The notebook is divided into 16 sections:

1. Install dependencies
2. Configuration (centralized `CONFIG` dict)
3. Load dataset (via Kaggle API)
4. Exploratory Data Analysis
5. Connection scoring and feature engineering
6. Preprocessing and target construction
7. Train 6 models with 5-fold CV
8. Hyperparameter tuning (top 3)
9. Best model selection, calibration, validation
10. SHAP interpretability
11. Classification report
12. AutoML baseline comparison (FLAML)
13. Final comparison table
14. Save artifacts
15. Download artifacts (optional)
16. Launch Streamlit dashboard (optional)

---

## Project Structure

```
.
├── ConnectionLens_Colab.ipynb      Self-contained Colab notebook (primary deliverable)
├── README.md                       This file
├── dating_app_behavior_dataset.csv             Original dataset (19 columns)
├── dating_app_behavior_dataset_extended1.csv   Extended dataset (25 columns, used in notebook)
├── WIA1006_WID3006_Group Assignment_2526.pdf   Assignment brief
└── .gitignore
```

**Generated at runtime (not committed):**

- `Preprocessed_Data_V2/` — train/test splits, scaler, encoder, selected features
- `ML_Results/` — trained model, comparison table, classification report

---

## Dependencies

All dependencies are installed within the Colab notebook (Section 1). No `requirements.txt` needed.

| Category | Packages |
|---|---|
| **Core** | pandas, numpy, scikit-learn, scipy, imbalanced-learn, joblib |
| **Models** | xgboost, lightgbm, catboost |
| **Visualization** | matplotlib, seaborn, plotly |
| **Interpretability** | shap |
| **Dashboard** | streamlit |
| **AutoML** | flaml (fallback if auto-sklearn unavailable) |
| **Data** | kagglehub |

---

## Requirement Mapping

Quick reference for graders — maps each assignment deliverable to the relevant notebook section:

| Deliverable | Description | Where to Find |
|---|---|---|
| **D1** | Problem framing, business value | [Problem Statement](#problem-statement), Notebook Section 0 |
| **D2** | Data understanding, EDA | [Dataset](#dataset), Notebook Sections 3-4 |
| **D3** | Methodology | [Target Variable](#target-variable), [Feature Engineering](#feature-engineering), [Model Pipeline](#model-pipeline), Notebook Sections 5-8 |
| **D4** | Results, interpretation | [Results](#results), [Interpretability](#interpretability), Notebook Sections 9-11 |
| **D5** | Deployment, prediction | [Prediction and Deployment](#prediction-and-deployment), Notebook Section 16 |
| **4.1** | AutoML comparison | Notebook Section 12 (FLAML) |
| **4.2** | Dataset description | [Dataset](#dataset), Notebook Section 3 |
| **4.3** | Target variable | [Target Variable](#target-variable), Notebook Section 5 |
| **4.4** | Feature engineering | [Feature Engineering](#feature-engineering), Notebook Section 5 |
| **4.5** | Train/test split | Notebook Section 6 |
| **4.6** | Cross-validation + SMOTE | Notebook Section 7 |
| **4.7** | Model selection | Notebook Section 7 |
| **4.8** | Hyperparameter tuning | Notebook Section 8 |
| **4.9** | Evaluation metrics | Notebook Sections 9, 11, 13 |
| **4.10** | Interpretability (SHAP) | Notebook Section 10 |
| **4.11** | Prediction function | Notebook Section 16 (Streamlit dashboard) |
| **4.12** | Calibration and validation | Notebook Section 9 |
| **4.13** | Fairness / OOD / Drift | Notebook Section 16 (sensitivity analysis) |
| **4.14** | Deployment | Notebook Section 16 (Streamlit dashboard) |

---

*WID3006 Machine Learning group project at Universiti Malaya — "Tying the Data Knot."*
