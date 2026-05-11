# IntentSight

**User Intent Signal Explorer** — WID3006 ML Group Assignment ("Tying the Data Knot")

A production-style data product that classifies dating-app user intent from
behavioural features. Ships with three deployment tiers:

| Tier | Stack | Entry Point |
| --- | --- | --- |
| 1 | Streamlit | `streamlit run streamlit_app.py` |
| 2 | FastAPI + Uvicorn | `uvicorn app.main:app` |
| 3 | Docker Compose | `docker compose up` |

## Quick Start (Local)

```powershell
# 1. Install dependencies
python -m pip install -r requirements.txt

# 2. Run the ML pipeline (generates artifacts in ML_Results/)
python Machine_Learning_V2.py

# 3a. Launch Streamlit dashboard (Tier 1)
streamlit run streamlit_app.py

# 3b. OR launch FastAPI (Tier 2)
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## Docker (Tier 3)

```bash
docker compose up --build
```

- **Dashboard:** <http://localhost:8501>
- **API:** <http://localhost:8000>
- **API docs:** <http://localhost:8000/docs>

## API Endpoints

| Method | Route | Description |
| --- | --- | --- |
| GET | `/v1/health` | Artifact status & model health |
| GET | `/v1/metrics` | Model comparison, baseline, class distribution |
| GET | `/v1/options` | Available metrics and categories |
| GET | `/v1/cohorts` | Cohort analysis (aggregated / category / individual) |
| GET | `/v1/heatmap` | Cross-tabulated heatmap data |
| POST | `/v1/predict` | Single scenario prediction |
| POST | `/v1/predict/batch` | Batch prediction |

Backward-compatible `/api/` aliases are also available.

## Project Structure

```text
.
├── app/                     # FastAPI application
│   ├── main.py              # API routes and lifespan
│   └── services/            # Model + data services
├── streamlit_app.py         # Streamlit dashboard (Tier 1)
├── Machine_Learning_V2.py   # ML pipeline (8 models, nested CV, calibration)
├── Data_Preprocessing_V2.py # Data preprocessing pipeline
├── feature_store.py         # Feature registry
├── performance_tracker.py   # Drift detection (PSI, KS)
├── logging_config.py        # Structured JSON logging
├── docker-compose.yml       # Multi-service deployment
├── Dockerfile               # FastAPI container
├── Dockerfile.streamlit     # Streamlit container
├── tests/                   # pytest test suite
├── ML_Results/              # Generated model artifacts
├── Preprocessed_Data_V2/    # Preprocessed data artifacts
└── requirements.txt         # Python dependencies
```

## Documentation

- [RUN_INTENTSIGHT.md](RUN_INTENTSIGHT.md) — Full Windows/VS Code runbook
- [MODEL_CARD.md](MODEL_CARD.md) — Model card with intended use, limitations, and fairness notes
- [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) — Contribution guide
