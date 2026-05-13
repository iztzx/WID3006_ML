# IntentSight

**User Engagement Level Classifier** — WID3006 ML Group Assignment ("Tying the Data Knot")

A production-style data product that classifies dating-app users into **Low / Medium / High** engagement levels from behavioural and demographic features. Ships with three deployment tiers, a full ML pipeline with 6 models, SHAP interpretability, calibration, drift detection, and a Google Colab notebook for zero-setup execution.

---

## Table of Contents

- [Deployment Tiers](#deployment-tiers)
- [Quick Start (Local)](#quick-start-local)
- [Google Colab (No Local Setup)](#google-colab-no-local-setup)
- [Docker (Tier 3)](#docker-tier-3)
- [ML Pipeline](#ml-pipeline)
- [Feature Engineering](#feature-engineering)
- [API Reference](#api-reference)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Project Structure](#project-structure)
- [Generated Artifacts](#generated-artifacts)
- [Testing](#testing)
- [CI / CD](#ci--cd)
- [Configuration](#configuration)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

---

## Deployment Tiers

|Tier|Stack|Entry Point|Port|
|---|---|---|---|
|1|Streamlit|`streamlit run streamlit_app.py`|8501|
|2|FastAPI + Uvicorn|`uvicorn app.main:app`|8000|
|3|Docker Compose|`docker compose up`|8000 + 8501|

---

## Quick Start (Local)

**Prerequisites:** Python 3.10+ (tested on 3.11 / 3.12), pip.

```powershell
# 1. Clone the repository
git clone https://github.com/iztzx/WID3006_ML.git
cd WID3006_ML

# 2. Install dependencies
python -m pip install -r requirements.txt

# 3. Run preprocessing (generates artifacts in Preprocessed_Data_V2/)
python preprocess.py

# 4. Run ML pipeline (generates artifacts in ML_Results/)
python train.py

# 5a. Launch Streamlit dashboard (Tier 1)
streamlit run streamlit_app.py

# 5b. OR launch FastAPI (Tier 2)
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Steps 3 and 4 must be run before launching the dashboard or API — they generate the model artifacts that the services load at startup.

---

## Google Colab (No Local Setup)

Can't run locally? Use the Colab notebook — it runs the full pipeline in the browser.

1. Open in Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iztzx/WID3006_ML/blob/main/IntentSight_Colab.ipynb)
2. Upload `Behaviour_Extended_Dataset.csv` when prompted (or mount Google Drive)
3. Run all cells — the notebook handles installs, preprocessing, training, tuning, SHAP, and artifact export
4. Download the generated `ML_Results/` zip for the trained model and artifacts

**Runtime:** ~10–15 minutes on a free T4 GPU runtime.

The notebook mirrors the local pipeline exactly: 6 models with 5-fold CV, hyperparameter tuning on the top 3, calibration, SHAP beeswarm + bar plots, and a final comparison table.

---

## Docker (Tier 3)

**Prerequisites:** Docker and Docker Compose.

```bash
docker compose up --build
```

This launches two containers:

|Container|Image|Port|Healthcheck|
|---|---|---|---|
|`intentsight-api`|`Dockerfile` (Python 3.11-slim)|8000|`GET /v1/health` every 30s|
|`intentsight-dashboard`|`Dockerfile.streamlit` (Python 3.11-slim)|8501|`GET /_stcore/health`|

Both containers mount `ML_Results/`, `Preprocessed_Data_V2/`, and dataset CSVs as read-only volumes. The dashboard waits for the API to be healthy before starting.

- **Dashboard:** <http://localhost:8501>
- **API:** <http://localhost:8000>
- **API docs (Swagger):** <http://localhost:8000/docs>

---

## ML Pipeline

### Target Construction

The original `relationship_intent` column in the dataset has **zero predictive signal** (all features are statistically independent — correlations < 0.005, MI ≈ 0, chi-square p=0.66). After 12+ experiments confirming this, the target was reframed as a **3-class User Engagement Level** constructed from behavioural features:

1. Standardize 5 behavioural features (z-score): `app_usage_time_min`, `swipe_right_ratio`, `message_sent_count`, `likes_received`, `emoji_usage_rate`
2. Sum into composite `engagement_score`
3. Bin into 3 equal-frequency tiers via `pd.qcut` → `engagement_level` (0=Low, 1=Medium, 2=High)

This target is interpretable, actionable for a dating-app business, and achieves **89–96% accuracy** with proper modelling.

### Preprocessing (`preprocess.py`)

|Step|Description|
|---|---|
|Load & validate|Read CSV, check ≥100 rows, log nulls/duplicates|
|Target construction|Z-score sum → `engagement_score` → `pd.qcut` → `engagement_level`|
|Feature engineering|Interest tags → 49 binary columns, `num_interests`, `match_rate`, `msg_per_match`, `bmi`|
|Encode categoricals|Ordinal for `income_bracket` / `education_level`, one-hot for the rest|
|Scale & split|`StandardScaler`, 80/20 stratified train/test split|
|Feature selection|RF importance, 95% cumulative threshold (min 20 features)|

### Training (`train.py`)

|Step|Description|
|---|---|
|Base models|6 models with 5-fold stratified CV (all wrapped in SMOTE pipeline)|
|Tuning|`RandomizedSearchCV` (20 iterations, 3-fold) on top 3 by test accuracy|
|Calibration|`CalibratedClassifierCV` (sigmoid, 3-fold) on the best model|
|Interpretability|SHAP beeswarm + bar plots on a dedicated RF (200 estimators, 1000 samples)|
|Comparison|All base + tuned + calibrated + majority baseline → CSV|

### Models

|#|Model|SMOTE|Key Hyperparameters|
|---|---|---|---|
|1|Logistic Regression|Yes|max_iter=2000|
|2|Random Forest|Yes|300 trees, max_depth=20|
|3|Gradient Boosting|Yes|100 trees, max_depth=5|
|4|XGBoost|Yes|300 trees, max_depth=6, subsample=0.8|
|5|LightGBM|Yes|300 trees, max_depth=8, subsample=0.8|
|6|CatBoost|Yes|300 iterations, depth=6|

SMOTE is applied inside an `imblearn.Pipeline` during CV to prevent data leakage.

### Expected Results

|Model|Expected Accuracy|Majority Baseline|
|---|---|---|
|XGBoost|93–96%|33.3%|
|LightGBM|93–96%|33.3%|
|CatBoost|90–95%|33.3%|
|Random Forest|85–89%|33.3%|
|Gradient Boosting|85–89%|33.3%|
|Logistic Regression|60–70%|33.3%|

---

## Feature Engineering

### Raw Features (19)

**Numeric (11):** `age`, `app_usage_time_min`, `likes_received`, `mutual_matches`, `message_sent_count`, `bio_length`, `emoji_usage_rate`, `height_cm`, `weight_kg`, `profile_pics_count`, `last_active_hour`

**Categorical (8):** `gender`, `income_bracket`, `education_level`, `sexual_orientation`, `location_type`, `swipe_time_of_day`, `body_type`, `interest_tags`

### Engineered Features (5+)

|Feature|Formula|Description|
|---|---|---|
|`match_rate`|`mutual_matches / (likes_received + 1)`|Efficiency of converting likes to matches|
|`msg_per_match`|`message_sent_count / (mutual_matches + 1)`|Messaging intensity per match|
|`bmi`|`weight_kg / (height_cm/100)^2`|Body mass index|
|`num_interests`|`len(interest_tags.split(","))`|Number of declared interests|
|`tag_*`|Binary (0/1) per unique tag|49 binary columns from parsed interest tags|

### Feature Selection

A Random Forest is trained on all features. Features are ranked by importance and the top set covering 95% of cumulative importance is selected (minimum 20 features). The final selected feature set is saved to `Preprocessed_Data_V2/selected_features.pkl`.

---

## API Reference

All endpoints are versioned under `/v1/`. Backward-compatible `/api/` aliases are also available.

### `GET /v1/health`

Returns model health and artifact status.

```json
{
  "status": "ok",
  "missing_artifacts": [],
  "model_loaded": true,
  "artifacts": [...]
}
```

### `GET /v1/metrics`

Returns model comparison table, majority baseline, class distribution, and nested CV results.

### `GET /v1/options`

Returns available metrics and categories for cohort/heatmap queries.

### `GET /v1/cohorts?view=aggregated&metric=app_usage_time_min&category=gender`

Cohort analysis with three view modes:

|View|Description|
|---|---|
|`aggregated`|Single series across metric bands|
|`category`|One series per category value (e.g., per gender)|
|`individual`|Top 12 cohorts by gender/income/metric band|

### `GET /v1/heatmap?x=gender&y=app_usage_time_min`

Cross-tabulated heatmap data with `count`, `high_share`, and `avg_confidence` per cell.

### `POST /v1/predict`

Single scenario prediction with calibration, OOD detection, and audit logging.

**Request body (all fields optional, defaults to 0):**

|Field|Type|Range|
|---|---|---|
|`app_usage_time_min`|float|0–1000|
|`swipe_right_ratio`|float|0–1|
|`likes_received`|int|0–10000|
|`mutual_matches`|int|0–10000|
|`message_sent_count`|int|0–10000|
|`bio_length`|int|0–5000|
|`emoji_usage_rate`|float|0–10|
|`height_cm`|float|80–250|
|`weight_kg`|float|20–300|
|`profile_pics_count`|int|0–50|
|`last_active_hour`|int|0–23|

**Response:**

```json
{
  "prediction": "High",
  "encoded": 2,
  "confidence": 0.9234,
  "calibrated_probabilities": {"Low": 0.02, "Medium": 0.0566, "High": 0.9234},
  "ood_flags": null,
  "note": "Exploratory scenario prediction..."
}
```

Inputs flagged as out-of-distribution (z-score > 3 on any feature) include `ood_flags` with per-feature details. Every prediction is logged to `predictions_log.jsonl` for audit.

### `POST /v1/predict/batch`

Batch prediction — accepts a list of scenarios, returns predictions with count.

### Error Responses

|Code|Description|
|---|---|
|400|Invalid metric, category, or view parameter|
|422|Validation error (field out of range)|
|404|Unknown route|
|500|Internal server error (structured JSON)|

---

## Streamlit Dashboard

The Tier 1 dashboard (`streamlit_app.py`) provides an interactive UI for:

- **Model performance:** accuracy, F1, calibration plots, comparison table
- **Feature importance:** bar chart from `final_comparison.csv`, SHAP values
- **Cohort analysis:** filter by metric band, category, or individual cohorts
- **Heatmap:** cross-tabulated engagement by any two dimensions
- **Scenario simulation:** input feature values and get real-time predictions with confidence

Run with: `streamlit run streamlit_app.py` (port 8501).

---

## Project Structure

```text
.
├── app/                          # FastAPI application
│   ├── main.py                   # API routes, Pydantic models, lifespan
│   └── services/
│       ├── model_service.py      # Model loading, predictions, OOD detection, audit log
│       └── data_service.py       # Data loading, cohort analysis, heatmap, drift detection
├── streamlit_app.py              # Streamlit dashboard (Tier 1)
├── preprocess.py                 # Data preprocessing & target construction
├── train.py                      # ML pipeline (6 models, CV, tuning, SHAP, calibration)
├── feature_store.py              # Feature registry (definitions, types, transformations)
├── performance_tracker.py        # Drift detection (PSI, KS test) & metric history
├── logging_config.py             # Structured JSON logging (python-json-logger)
├── IntentSight_Colab.ipynb       # Google Colab notebook (full pipeline)
├── docker-compose.yml            # Multi-service deployment (API + dashboard)
├── Dockerfile                    # FastAPI container (Python 3.11-slim, multi-stage)
├── Dockerfile.streamlit          # Streamlit container (Python 3.11-slim, multi-stage)
├── requirements.txt              # Python dependencies
├── tests/
│   └── test_api.py               # 22 pytest tests (health, metrics, cohorts, predict, batch)
├── docs/
│   └── CONTRIBUTING.md           # Contribution guidelines
├── ML_Results/                   # Generated model artifacts (after train.py)
│   ├── best_tuned_model.pkl      # Calibrated best model
│   ├── final_comparison.csv      # All models comparison table
│   ├── target_encoder.pkl        # Label encoder for engagement_level
│   ├── selected_features.pkl     # Selected feature names
│   └── scaler.pkl                # StandardScaler fitted on training data
├── Preprocessed_Data_V2/         # Generated data artifacts (after preprocess.py)
│   ├── X_train_selected_unresampled.csv
│   ├── X_test_selected.csv
│   ├── y_train_original.csv
│   ├── y_test.csv
│   ├── scaler.pkl
│   ├── target_encoder.pkl
│   └── selected_features.pkl
└── Behaviour_Extended_Dataset.csv # Source dataset (not committed)
```

---

## Generated Artifacts

### `Preprocessed_Data_V2/` (from `preprocess.py`)

|File|Description|
|---|---|
|`X_train_selected_unresampled.csv`|Training features (selected, unscaled for SMOTE)|
|`X_test_selected.csv`|Test features (selected, scaled)|
|`y_train_original.csv`|Training labels (encoded)|
|`y_test.csv`|Test labels (encoded)|
|`scaler.pkl`|fitted `StandardScaler`|
|`target_encoder.pkl`|fitted `LabelEncoder`|
|`selected_features.pkl`|List of selected feature names|

### `ML_Results/` (from `train.py`)

|File|Description|
|---|---|
|`best_tuned_model.pkl`|Calibrated best model (SMOTE + estimator)|
|`final_comparison.csv`|All models: CV accuracy, test accuracy, F1|
|`target_encoder.pkl`|Label encoder copy|
|`selected_features.pkl`|Feature list copy|
|`scaler.pkl`|Scaler copy|
|`feature_importance.png`|Top 20 feature importances (RF)|
|`calibration_plot.png`|Calibration curve for best model|
|`shap_beeswarm.png`|SHAP beeswarm plot (top 20 features)|
|`shap_bar.png`|SHAP bar plot (top 20 features)|
|`classification_report.txt`|Precision/recall/F1 per class|

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=app --cov-report=term-missing

# Run specific test
python -m pytest tests/test_api.py::test_health_ok -v
```

**Coverage:** 22 tests covering health, metrics, cohorts, heatmap, options, single prediction, batch prediction, input validation, error handling, and backward-compatible aliases.

**Test stack:** pytest + httpx `TestClient`. Tests use monkeypatching and `tmp_path` for isolation — no real model artifacts required for most tests.

---

## CI / CD

GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push to `main` and PRs to `main` / `develop`:

|Step|Tool|Description|
|---|---|---|
|Lint|Ruff|`ruff check . --exit-non-zero-on-fix` + `ruff format . --check`|
|Type check|mypy|`mypy --ignore-missing-imports` on `app/`, `tests/`, `preprocess.py`, `train.py`, `feature_store.py`, `performance_tracker.py`, `logging_config.py`|
|Test|pytest|`python -m pytest tests/ -v --tb=short --timeout=300`|

**Runtime:** Python 3.11 on `ubuntu-latest`, pip cache enabled.

---

## Configuration

### Environment Variables

|Variable|Default|Description|
|---|---|---|
|`INTENTSIGHT_DATASET_PATH`|Auto-detected|Override dataset CSV path|

The dataset is auto-detected in this order: `Behaviour_Extended_Dataset.csv`, then `Behaviour_Dataset.csv` in the project root. Set `INTENTSIGHT_DATASET_PATH` to use a custom path.

### Structured Logging

All services use structured JSON logging via `python-json-logger`:

```json
{
  "timestamp": "2026-05-13T10:30:00Z",
  "severity": "INFO",
  "service": "intentsight",
  "environment": "production",
  "message": "Model loaded: CalibratedClassifierCV"
}
```

Suppressed loggers: `optuna`, `flaml`, `httpx`, `urllib3` (set to `WARNING`).

---

## Dependencies

### Core

|Package|Version|Purpose|
|---|---|---|
|fastapi|≥0.110|API framework|
|uvicorn[standard]|≥0.27|ASGI server|
|pandas|≥2.0|Data manipulation|
|numpy|≥1.24|Numerical computing|
|scikit-learn|≥1.3|ML models, preprocessing, metrics|
|scipy|≥1.11|Statistical tests (KS)|
|imbalanced-learn|≥0.11|SMOTE oversampling|
|joblib|≥1.3|Model serialization|

### Models

|Package|Version|Purpose|
|---|---|---|
|xgboost|≥2.0|XGBoost classifier|
|lightgbm|≥4.0|LightGBM classifier|
|catboost|≥1.2|CatBoost classifier|

### Visualization & Interpretability

|Package|Version|Purpose|
|---|---|---|
|matplotlib|≥3.7|Static plots (SHAP, feature importance)|
|seaborn|≥0.13|Statistical visualizations|
|plotly|≥5.18|Interactive Streamlit charts|
|shap|≥0.44|SHAP interpretability|

### API, Dashboard & Logging

|Package|Version|Purpose|
|---|---|---|
|streamlit|≥1.30|Tier 1 dashboard|
|python-json-logger|≥2.0|Structured JSON logging|

### Testing & Dev

|Package|Version|Purpose|
|---|---|---|
|pytest|≥8.0|Test framework|
|httpx|≥0.26|Async HTTP client for FastAPI tests|
|ruff|≥0.4|Linting + formatting|
|mypy|≥1.10|Static type checking|

---

## Contributing

See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for:

- Development setup (Python 3.10+, venv, pip install)
- Code style (Ruff, mypy --strict, Google-style docstrings)
- Testing guidelines (Arrange/Act/Assert, monkeypatch, ≥90% coverage target)
- Git workflow (feature branches, small commits, lint+tests before PR)
- PR checklist (tests pass, types check, docs updated)

---

## License

This project is part of the WID3006 Machine Learning course group assignment at Universiti Malaya.
