# Contributing to IntentSight

Thank you for your interest in contributing to IntentSight! This document covers setup, code style, testing conventions, and workflows.

## Prerequisites

- Python 3.10+ (tested with 3.11/3.12)
- pip / uv
- Git

## Quick Start

```powershell
# 1. Clone the repository
git clone <repo-url>
cd WID3006_ML

# 2. (Recommended) Create a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate     # macOS/Linux

# 3. Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 4. Run unit tests
python -m pytest tests/ -v --tb=short

# 5. Start the dashboard
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 in your browser.

## Project Structure

```
.
├── app/
│   ├── main.py                  # FastAPI application entry point
│   └── services/
│       ├── model_service.py     # Model loading, prediction, calibration, OOD
│       └── data_service.py      # Data loading, cohorts, heatmap, drift
├── Preprocessed_Data_V2/        # Preprocessed artifacts (CSV, PKL)
├── ML_Results/                  # Model outputs, comparison tables, plots
├── tests/                       # Unit / integration tests
├── Machine_Learning_V2.py       # Full ML pipeline (train, tune, calibrate)
├── Data_Preprocessing_V2.py     # Preprocessing and feature engineering
├── feature_store.py             # Feature registry and metadata
├── performance_tracker.py       # Drift detection (PSI, KS) and metric history
├── logging_config.py            # Structured JSON logging setup
├── requirements.txt             # Python dependencies
├── MODEL_CARD.md                # Model documentation
├── Dockerfile                   # Container build
└── .github/workflows/ci.yml     # CI: lint + test on PR
```

## Code Style

- **Format**: Use [Ruff](https://docs.astral.sh/ruff/) with default settings. Run `ruff check . && ruff format .` before committing.
- **Type hints**: All public functions and methods must have type annotations. Use `mypy` to check: `mypy --strict app/ tests/`.
- **Imports**: Group in this order: stdlib, third-party, local. Use `from __future__ import annotations` at the top of every file.
- **Docstrings**: Google-style docstrings on all public modules, classes, and functions.
- **Logging**: Use `from logging_config import logger` throughout. Replace all `print()` calls with `logger.info()`, `logger.warning()`, etc.

## Testing Conventions

- **Framework**: `pytest` with `httpx` for FastAPI TestClient.
- **File naming**: `tests/test_<module>.py` mirrors `app/<module>.py`.
- **Test structure**: Arrange / Act / Assert. One assertion concept per test.
- **Fixtures**: Use `monkeypatch` and `tmp_path` for env var and filesystem isolation.
- **Coverage goal**: ≥ 90% on critical API paths (health, metrics, predict, cohorts, heatmap, batch).
- Run tests: `python -m pytest tests/ -v --cov=app --cov-report=term-missing`

## Git Workflow

1. Create a feature branch from `main`: `git checkout -b feat/your-feature-name`
2. Make small, focused commits with clear messages.
3. Run lint and tests locally before pushing.
4. Open a Pull Request against `main` with a description of what changed and why.
5. CI will run `ruff check`, `mypy`, and `pytest`. All checks must pass.

## Adding a New Feature

1. Update `MODEL_CARD.md` if the change affects model behaviour.
2. Add corresponding tests in `tests/`.
3. Update `requirements.txt` if new dependencies are needed.
4. Document any API changes (new endpoints, schema changes).
5. Update this `CONTRIBUTING.md` if the workflow changes.

## Questions?

Open an issue or reach out to the project maintainers. We're happy to help you get started!