# ConnectionLens

**Connection Readiness Classifier** — WID3006 ML Group Assignment ("Tying the Data Knot")

ConnectionLens is a five-class classifier that scores dating-app users on connection readiness — from "Needs Profile Help" to "Likely To Connect" — using behavioral signals, profile attributes, and match-funnel metrics. The pipeline trains six base models, tunes them with Optuna Bayesian optimization, and builds a Stacking Ensemble that achieves **98.26% test accuracy**. The entire pipeline runs in a single self-contained Google Colab notebook with SHAP interpretability, isotonic probability calibration, and an interactive 6-page Streamlit dashboard.

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
| --- | --- |
| **Likely To Connect** | `connection_score` rank >= 0.80 |
| **Ready To Chat** | Default middle tier |
| **Mostly Browsing** | Middle tier + `browser_issue` >= 62nd percentile + `browser_issue` >= `swipe_issue` |
| **Swipes Too Freely** | Middle tier + `swipe_issue` >= 50th percentile |
| **Needs Profile Help** | `connection_score` rank <= 0.20 |

The composite `connection_score` is a weighted blend:

```text
connection_score = 0.35 x match_quality
                 + 0.30 x conversation_quality
                 + 0.20 x profile_quality
                 + 0.15 x activity_level
                 - 0.10 x swipe_excess
```

**Input:** 25 raw features -> 18 engineered features -> 66 selected features (95% cumulative RF importance, minimum 20).

---

## Dataset

> Deliverable D2 / 4.2 — Data Source and Feature Description

**Source:** [Dating App Behavior Dataset](https://www.kaggle.com/datasets/keyushnisar/dating-app-behavior-dataset) — 50,000 synthetic records with 25 features.

**Feature types:**

| Category | Features |
| --- | --- |
| **Numeric (11)** | `age`, `app_usage_time_min`, `likes_received`, `mutual_matches`, `message_sent_count`, `bio_length`, `emoji_usage_rate`, `height_cm`, `weight_kg`, `profile_pics_count`, `last_active_hour` |
| **Categorical (12)** | `gender`, `income_bracket`, `education_level`, `sexual_orientation`, `location_type`, `swipe_time_of_day`, `body_type`, `zodiac_sign`, `interest_tags`, `app_usage_time_label`, `swipe_right_label`, `relationship_intent` |
| **Derived (2)** | `swipe_right_ratio`, `engagement_score` |

`match_outcome` is also present. `bmi`, `num_interests`, and 16 other features are engineered during preprocessing (see Feature Engineering). `app_usage_time_label` and `swipe_right_label` are dropped after encoding as redundant with their numeric counterparts.

**Data quality:** No critical missing values. Duplicates and nulls handled during preprocessing.

---

## Target Variable

> Deliverable D3 / 4.3 — Target Construction and Class Definitions

**Name:** `connection_stage` — five-class ordinal target.

**Construction rationale:** Rather than using a single behavioral metric, we construct a composite `connection_score` from four quality dimensions (match, conversation, profile, activity) minus a swipe-excess penalty. Users are then ranked by percentile and assigned to stages based on funnel-informed thresholds.

**Why weak supervision?** The dataset lacks ground-truth labels for "connection readiness." We derive labels from observable behavioral signals using domain-informed rules — this is a product signal, not a claim about private intent.

**Class distribution (from training set — roughly balanced):**

| Stage | Count | Proportion |
| --- | --- | --- |
| Likely To Connect | 7,855 | 19.6% |
| Ready To Chat | 8,335 | 20.8% |
| Mostly Browsing | 7,809 | 19.5% |
| Swipes Too Freely | 8,000 | 20.0% |
| Needs Profile Help | 8,001 | 20.0% |

**Why not SMOTE before split?** SMOTE is applied inside the `imblearn.Pipeline` during cross-validation to prevent synthetic samples from leaking into validation folds.

---

## Feature Engineering

> Deliverable D3 / 4.4 — Feature Transformations and Selection

### Engineered Features (18)

| Feature | Formula / Logic |
| --- | --- |
| `match_rate` | `mutual_matches / (likes_received + 1)` |
| `msg_per_match` | `message_sent_count / (mutual_matches + 1)` |
| `bmi` | `weight_kg / (height_cm / 100)^2` |
| `num_interests` | Count of parsed `interest_tags` |
| `profile_completeness` | `pics/6 x 0.4 + bio/300 x 0.4 + interests/5 x 0.2` |
| `selectivity_balance` | `1 - abs(swipe_ratio - 0.55) / 0.55`, clipped [0,1] |
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
| --- | --- |
| Logistic Regression | `max_iter=2000` |
| Random Forest | `n_estimators=300, max_depth=20` |
| Extra Trees | `n_estimators=300, max_depth=20` |
| XGBoost | `n_estimators=300, max_depth=6, subsample=0.8, colsample_bytree=0.8` |
| LightGBM | `n_estimators=300, max_depth=8, subsample=0.8, colsample_bytree=0.8` |
| CatBoost | `iterations=300, depth=6, learning_rate=0.1` |

### Hyperparameter Tuning [4.8]

All 6 models are tuned via **Optuna** Bayesian optimization (TPE sampler, MedianPruner) — 60 trials per model with a 120-second timeout. Search spaces cover all key hyperparameters (e.g., `n_estimators`, `max_depth`, `learning_rate`, regularization terms).

### Ensemble Methods

After tuning, two ensemble methods are built:

- **Stacking Classifier** — all 6 tuned models as base estimators, Logistic Regression as meta-learner (`passthrough=True`, `stack_method="predict_proba"`, same StratifiedKFold for meta-features to prevent leakage)
- **Soft-Voting Classifier** — top 3 tuned models by CV accuracy, soft voting

The Stacking Ensemble is selected if it beats the best single tuned model; otherwise the best single model is used.

### Calibration

Best model wrapped in `CalibratedClassifierCV(method="isotonic", cv=3)` with SMOTE pipeline. The calibrated version is used only if its F1 ≥ the uncalibrated F1.

---

## Results

> Deliverable D4 / 4.9 — Evaluation Metrics and Model Comparison

| Model | CV Accuracy | Test Accuracy | F1 (weighted) |
| --- | --- | --- | --- |
| **Stacking Ensemble** | 0.9819 | **0.9826** | 0.9826 |
| Logistic Regression | 0.9646 | 0.9600 | 0.9599 |
| CatBoost | 0.9556 | 0.9588 | 0.9588 |
| LightGBM | 0.9528 | 0.9559 | 0.9559 |
| XGBoost | 0.9496 | 0.9546 | 0.9546 |
| Extra Trees | 0.8938 | 0.8967 | 0.8968 |
| Random Forest | 0.8894 | 0.8906 | 0.8905 |

**Production default:** Stacking Ensemble — 98.26% test accuracy, 0.9826 F1.

**Per-class performance (Stacking Ensemble):**

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Likely To Connect | 0.99 | 0.99 | 0.99 | 2,000 |
| Mostly Browsing | 0.98 | 0.97 | 0.97 | 1,952 |
| Needs Profile Help | 0.98 | 0.99 | 0.99 | 2,000 |
| Ready To Chat | 0.98 | 0.98 | 0.98 | 2,084 |
| Swipes Too Freely | 0.98 | 0.98 | 0.98 | 1,964 |

**Baseline comparison:** All models exceed random-chance accuracy (20%) by >4.4x, confirming genuine predictive signal.

---

## Interpretability

> Deliverable D4 / 4.10 — SHAP Analysis and Feature Importance

### SHAP Analysis

SHAP values are computed on the **best single tuned model** using `shap.Explainer` (auto-selects Tree/Linear backend) on 1,000 training samples. Generates beeswarm, bar, and feature-importance plots. For CatBoost multi-class output (3D array), class 0 is shown for beeswarm and values are averaged across classes for the bar plot.

The Streamlit dashboard additionally uses `shap.KernelExplainer` (model-agnostic) on 200 test samples for the full Stacking Ensemble, with per-prediction waterfall explanations.

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

Single-user scoring via the Streamlit dashboard's **Prediction Playground** page. Adjust 11 profile sliders (or use 5 quick presets: "The Connector", "The Swiper", "The Lurker", "The Newcomer", "The Chatty One") and get real-time connection-stage predictions with class probabilities, a radar chart, and a confidence gauge.

### Probability Calibration [4.12]

`CalibratedClassifierCV(method="isotonic", cv=3)` wraps the best model. Calibration plot validates reliability.

### OOD Detection [4.13]

Sensitivity analysis in the dashboard perturbs each input feature by +/-10% and shows which changes would most affect the prediction.

### Fairness Considerations

The model does not use protected attributes (gender, sexual_orientation) as direct predictors after one-hot encoding. Predictions are product signals for intervention design, not claims about user worth or intent.

---

## Quick Start

### Google Colab (Zero Setup) — Recommended

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iztzx/WID3006_ML/blob/main/ConnectionLens_Colab.ipynb)

Full pipeline in-browser, ~10-15 min on free T4 GPU. Handles installs, EDA, preprocessing, training, tuning, SHAP, calibration, and artifact export. Self-contained — no repo clone needed.

The notebook contains 33 cells (16 markdown, 17 code) organized as follows:

1. Install dependencies
2. Configuration (centralized `CONFIG` dict with scoring weights, pipeline settings, model hyperparameters, Optuna search spaces)
3. Load dataset (via `kagglehub` from Kaggle)
4. Exploratory Data Analysis (distributions, correlations)
5. Connection scoring and feature engineering (18 engineered features)
6. Preprocessing and target construction (encoding, scaling, RF-based feature selection → 66 features)
7. Train 6 models with 5-fold CV (SMOTE in pipeline)
8. Optuna hyperparameter tuning (60 trials per model, TPE sampler)
9. Stacking Ensemble + Soft-Voting construction, isotonic calibration
10. SHAP interpretability (beeswarm, bar, feature importance)
11. Classification report and comparison table
12. AutoML baseline comparison (FLAML)
13. Save artifacts (model, scaler, encoder, features, comparison CSV)
14. Download artifacts (optional)
15. Launch Streamlit dashboard (6 pages, optional)

---

## Project Structure

```text
.
├── ConnectionLens_Colab.ipynb                    Self-contained Colab notebook (primary deliverable)
├── README.md                                     This file
├── requirements.txt                              Python dependencies for local execution
├── dating_app_behavior_dataset.csv               Original dataset (19 columns, 50K rows)
├── dating_app_behavior_dataset_extended1.csv     Extended dataset (25 columns, used by notebook)
├── WIA1006_WID3006_Group Assignment_2526.pdf     Assignment brief
└── .gitignore
```

**Generated at runtime (gitignored except where noted):**

- `Preprocessed_Data_V2/` — train/test splits (`X_train_selected_unresampled.csv`, `X_test_selected.csv`, `y_train_original.csv`, `y_test.csv`), scaler, encoder, selected features
- `ML_Results/` — trained model (`best_tuned_model.pkl`), comparison table (`final_comparison.csv`), classification report (`classification_report.txt` — committed), test features, scaler, encoder

---

## Dependencies

All dependencies are installed within the Colab notebook (Section 1). For local execution, install with `pip install -r requirements.txt`.

| Category | Packages |
| --- | --- |
| **Core** | pandas, numpy, scikit-learn, scipy, imbalanced-learn, joblib |
| **Models** | xgboost, lightgbm, catboost |
| **Tuning** | optuna (TPE sampler, MedianPruner) |
| **Visualization** | matplotlib, seaborn, plotly |
| **Interpretability** | shap |
| **Dashboard** | streamlit |
| **AutoML** | flaml (fallback if auto-sklearn unavailable) |
| **Data** | kagglehub |

---

## Requirement Mapping

Quick reference for graders — maps each assignment deliverable to the relevant notebook section:

| Deliverable | Description | Where to Find |
| --- | --- | --- |
| **D1** | Problem framing, business value | [Problem Statement](#problem-statement), Notebook Cell 0 |
| **D2** | Data understanding, EDA | [Dataset](#dataset), Notebook Cells 5-9 |
| **D3** | Methodology | [Target Variable](#target-variable), [Feature Engineering](#feature-engineering), [Model Pipeline](#model-pipeline), Notebook Cells 10-17 |
| **D4** | Results, interpretation | [Results](#results), [Interpretability](#interpretability), Notebook Cells 20-25 |
| **D5** | Deployment, prediction | [Prediction and Deployment](#prediction-and-deployment), Notebook Cell 32 |
| **4.1** | AutoML comparison | Notebook Cell 25 (FLAML) |
| **4.2** | Dataset description | [Dataset](#dataset), Notebook Cell 5 |
| **4.3** | Target variable | [Target Variable](#target-variable), Notebook Cell 11 |
| **4.4** | Feature engineering | [Feature Engineering](#feature-engineering), Notebook Cell 11 |
| **4.5** | Train/test split | Notebook Cell 13 |
| **4.6** | Cross-validation + SMOTE | Notebook Cell 15 |
| **4.7** | Model selection | Notebook Cells 15, 17 (base + Optuna-tuned + ensembles) |
| **4.8** | Hyperparameter tuning | Notebook Cell 17 (Optuna, 60 trials per model) |
| **4.9** | Evaluation metrics | Notebook Cells 23, 25 |
| **4.10** | Interpretability (SHAP) | Notebook Cell 21 |
| **4.11** | Prediction function | Notebook Cell 32 (Streamlit Prediction Playground) |
| **4.12** | Calibration and validation | Notebook Cell 17 (isotonic calibration, Step 3) |
| **4.13** | Fairness / OOD / Drift | Notebook Cell 32 (sensitivity analysis in dashboard) |
| **4.14** | Deployment | Notebook Cell 32 (Streamlit dashboard) |

---

### WID3006 Machine Learning group project at Universiti Malaya — "Tying the Data Knot."
