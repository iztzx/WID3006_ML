"""IntentSight API — v3 with versioned routes, batch predict, and error handling."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.services.data_service import DataService
from app.services.model_service import ModelService

try:
    from logging_config import logger
except ImportError:
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logger = logging.getLogger("intentsight")


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "app" / "static"

# ---------------------------------------------------------------------------
# Shared services (initialised at module level for broad compatibility,
# and again inside the lifespan for async-aware setups)
# ---------------------------------------------------------------------------
_model_service = ModelService()
_data_service = DataService(_model_service)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle — also sets services on app.state."""
    logger.info("IntentSight API starting — initialising services")
    app.state.model_service = _model_service
    app.state.data_service = _data_service
    yield
    logger.info("IntentSight API shut down")


app = FastAPI(
    title="IntentSight",
    description="Defensible ML intent-signal explorer for dating-app behaviour data.",
    version="3.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — allow the dashboard to be embedded or served from another origin
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Mount static files — serve index.html for SPA-style navigation
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# Global error handler
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Return structured JSON for any unhandled exception."""
    logger.exception("Unhandled error on %s %s", request.method, request.url)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class ScenarioPayload(BaseModel):
    """Single profile for point prediction."""

    app_usage_time_min: float = Field(120, ge=0, le=1000)
    swipe_right_ratio: float = Field(0.5, ge=0, le=1)
    likes_received: float = Field(50, ge=0, le=10000)
    mutual_matches: float = Field(10, ge=0, le=10000)
    message_sent_count: float = Field(30, ge=0, le=10000)
    bio_length: float = Field(140, ge=0, le=5000)
    emoji_usage_rate: float = Field(0.3, ge=0, le=10)
    height_cm: float = Field(170, ge=80, le=250)
    weight_kg: float = Field(70, ge=20, le=300)
    profile_pics_count: int = Field(3, ge=0, le=50)
    last_active_hour: int = Field(12, ge=0, le=23)


class BatchPayload(BaseModel):
    """Batch of scenarios for bulk prediction."""

    scenarios: list[ScenarioPayload]


class BatchResponse(BaseModel):
    """Batch prediction result."""

    predictions: list[dict]
    count: int


# ---------------------------------------------------------------------------
# Route helpers — access services from app.state (set by lifespan) with
# a fallback to the module-level singletons for compatibility.
# ---------------------------------------------------------------------------


def get_model_service() -> ModelService:
    try:
        return app.state.model_service
    except (AttributeError, RuntimeError):
        return _model_service


def get_data_service() -> DataService:
    try:
        return app.state.data_service
    except (AttributeError, RuntimeError):
        return _data_service


# ---------------------------------------------------------------------------
# Routes — all versioned under /v1/
# ---------------------------------------------------------------------------


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/v1/health")
async def health():
    return get_model_service().health()


@app.get("/v1/metrics")
async def metrics():
    try:
        return get_model_service().metrics()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/v1/options")
async def options():
    try:
        return get_data_service().options()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/v1/cohorts")
async def cohorts(
    view: Annotated[str, "aggregated|category|individual"] = "aggregated",
    metric: str = "app_usage_time_min",
    category: str = "gender",
):
    try:
        return get_data_service().cohorts(view=view, metric=metric, category=category)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/v1/heatmap")
async def heatmap(x: str = "gender", y: str = "app_usage_time_min"):
    try:
        return get_data_service().heatmap(x=x, y=y)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/predict")
async def predict(payload: ScenarioPayload):
    try:
        logger.info("Single prediction requested: %s", payload.model_dump())
        result = get_model_service().predict_scenario(payload.model_dump())
        logger.info("Prediction result: %s", result)
        return result
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/predict/batch", response_model=BatchResponse)
async def predict_batch(payload: BatchPayload):
    """Batch prediction endpoint — validates each item individually."""
    results = []
    for idx, scenario in enumerate(payload.scenarios):
        try:
            data = scenario.model_dump()
            pred = get_model_service().predict_scenario(data)
            pred["index"] = idx
            results.append(pred)
        except Exception as exc:
            results.append({"index": idx, "error": str(exc)})
    logger.info("Batch prediction: %d scenarios processed", len(results))
    return BatchResponse(predictions=results, count=len(results))


# Backward-compatible aliases (map to v1)
@app.get("/api/health")
async def health_alias():
    return await health()


@app.get("/api/metrics")
async def metrics_alias():
    return await metrics()


@app.get("/api/options")
async def options_alias():
    return await options()


@app.get("/api/cohorts")
async def cohorts_alias(
    view: Annotated[str, "aggregated|category|individual"] = "aggregated",
    metric: str = "app_usage_time_min",
    category: str = "gender",
):
    return await cohorts(view=view, metric=metric, category=category)


@app.get("/api/heatmap")
async def heatmap_alias(x: str = "gender", y: str = "app_usage_time_min"):
    return await heatmap(x=x, y=y)


@app.post("/api/predict")
async def predict_alias(payload: ScenarioPayload):
    return await predict(payload)


# Expose for external import (e.g., tests)
MODEL_SERVICE = _model_service
DATA_SERVICE = _data_service
