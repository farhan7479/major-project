"""FastAPI service serving the trained load-forecasting models.

Loads every checkpoint from ml/models/checkpoints/ at startup via the shared
ml.serve.model_registry helper, so the API output is byte-identical to what
the training scripts saw on the test set.

Endpoints:
  GET  /health
  GET  /models      — list available models + their kind (tabular / sequence / arima)
  GET  /buildings   — list available buildings + the supported prediction window
  POST /forecast    — single (model, building, target) → kWh prediction
  POST /batch-forecast — same input, multiple models in one call
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Put the project's ml/ directory on the path so we can import serve.model_registry
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "ml"))

import pandas as pd  # noqa: E402

from serve.model_registry import ModelRegistry  # noqa: E402

ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "serve" / "artifacts"
CHECKPOINTS_DIR = PROJECT_ROOT / "ml" / "models" / "checkpoints"


# ----- request / response schemas -----

class WeatherOverride(BaseModel):
    airTemperature: Optional[float] = None
    dewTemperature: Optional[float] = None
    cloudCoverage: Optional[float] = None
    windSpeed: Optional[float] = None
    windDirection: Optional[float] = None
    seaLvlPressure: Optional[float] = None
    precipDepth1HR: Optional[float] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class ForecastRequest(BaseModel):
    model: str = Field(..., description="Model name, e.g. 'random_forest', 'xgboost', 'lstm', 'transformer', 'arima'")
    building_id: str = Field(..., description="One of the buildings returned by /buildings")
    target_datetime: str = Field(..., description="ISO datetime of the hour to predict")
    weather_override: Optional[WeatherOverride] = None


class BatchForecastRequest(BaseModel):
    models: list[str] = Field(..., description="List of model names to compare in this request")
    building_id: str
    target_datetime: str
    weather_override: Optional[WeatherOverride] = None


class ForecastResponse(BaseModel):
    model: str
    building_id: str
    target_datetime: str
    prediction_kwh: float
    actual_kwh: Optional[float] = None


class BatchForecastResponse(BaseModel):
    building_id: str
    target_datetime: str
    actual_kwh: Optional[float] = None
    predictions: dict[str, float]


# ----- lifespan: load registry once at startup -----

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"loading models from {CHECKPOINTS_DIR}")
    app.state.registry = ModelRegistry(ARTIFACTS_DIR, CHECKPOINTS_DIR)
    print(f"loaded models: {[m['name'] for m in app.state.registry.list_models()]}")
    yield


app = FastAPI(title="Energy Forecasting API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://127.0.0.1:5173", "http://127.0.0.1:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": len(app.state.registry.list_models())}


@app.get("/models")
async def models():
    return {"models": app.state.registry.list_models()}


@app.get("/buildings")
async def buildings():
    reg = app.state.registry
    return {
        "buildings": reg.list_buildings(),
        "test_period": reg.test_period(),
    }


@app.get("/metrics")
async def metrics():
    """Test-set leaderboard for the Dashboard page."""
    leaderboard_path = PROJECT_ROOT / "ml" / "models" / "results" / "leaderboard.csv"
    if not leaderboard_path.exists():
        raise HTTPException(status_code=404, detail="leaderboard.csv missing — run aggregate_results.py first")
    df = pd.read_csv(leaderboard_path)
    return {"leaderboard": df.to_dict(orient="records")}


def _parse_ts(value: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(value)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"invalid datetime '{value}': {e}")


@app.post("/forecast", response_model=ForecastResponse)
async def forecast(req: ForecastRequest):
    target_ts = _parse_ts(req.target_datetime)
    override = req.weather_override.to_dict() if req.weather_override else None
    try:
        kwh = app.state.registry.predict(req.model, req.building_id, target_ts, override)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    actual = app.state.registry.actual_consumption(req.building_id, target_ts)
    return ForecastResponse(
        model=req.model,
        building_id=req.building_id,
        target_datetime=target_ts.isoformat(),
        prediction_kwh=round(kwh, 2),
        actual_kwh=round(actual, 2) if actual is not None else None,
    )


@app.post("/batch-forecast", response_model=BatchForecastResponse)
async def batch_forecast(req: BatchForecastRequest):
    target_ts = _parse_ts(req.target_datetime)
    override = req.weather_override.to_dict() if req.weather_override else None
    preds = {}
    for model_name in req.models:
        try:
            preds[model_name] = round(
                app.state.registry.predict(model_name, req.building_id, target_ts, override), 2
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"{model_name}: {e}")
    actual = app.state.registry.actual_consumption(req.building_id, target_ts)
    return BatchForecastResponse(
        building_id=req.building_id,
        target_datetime=target_ts.isoformat(),
        actual_kwh=round(actual, 2) if actual is not None else None,
        predictions=preds,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
