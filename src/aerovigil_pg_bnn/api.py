"""
FastAPI server for Aerovigil PG-BNN inference.

This module provides a production-ready REST API for wind turbine RUL prediction.
"""

import asyncio
import json
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field

from .model import PhysicsGuidedBNN

# ─── CONFIGURATION ─────────────────────────────────────────────
DEFAULT_REPO_ID = "AerovigilAI/wind-turbine-pg-bnn"

# Input bounds for validation (physical constraints)
BOUNDS = {
    "vibration_rms": (0.0, 50.0),  # mm/s
    "bearing_temp": (20.0, 150.0),  # °C
    "generator_temp": (20.0, 200.0),  # °C
    "power_output": (0.0, 5000.0),  # kW
    "wind_speed": (0.0, 30.0),  # m/s
    "operating_hours": (0.0, 100000.0),  # hours
}

# SSE event terminator (two LF, per the Server-Sent Events spec).
_SSE_END = chr(10) + chr(10)


# ─── PYDANTIC MODELS ───────────────────────────────────────────
class TelemetryInput(BaseModel):
    """Single SCADA telemetry reading for RUL prediction."""

    vibration_rms: float = Field(
        ...,
        ge=BOUNDS["vibration_rms"][0],
        le=BOUNDS["vibration_rms"][1],
        description="Drive-train vibration RMS (mm/s)",
    )
    bearing_temp: float = Field(
        ...,
        ge=BOUNDS["bearing_temp"][0],
        le=BOUNDS["bearing_temp"][1],
        description="Main bearing temperature (°C)",
    )
    generator_temp: float = Field(
        ...,
        ge=BOUNDS["generator_temp"][0],
        le=BOUNDS["generator_temp"][1],
        description="Generator winding temperature (°C)",
    )
    power_output: float = Field(
        ...,
        ge=BOUNDS["power_output"][0],
        le=BOUNDS["power_output"][1],
        description="Active power generation (kW)",
    )
    wind_speed: float = Field(
        ...,
        ge=BOUNDS["wind_speed"][0],
        le=BOUNDS["wind_speed"][1],
        description="Nacelle wind speed (m/s)",
    )
    operating_hours: float = Field(
        ...,
        ge=BOUNDS["operating_hours"][0],
        le=BOUNDS["operating_hours"][1],
        description="Cumulative operating time (hours)",
    )


class BatchInput(BaseModel):
    """Batch of telemetry readings."""

    samples: list[TelemetryInput] = Field(
        ..., min_length=1, max_length=1000, description="List of telemetry readings"
    )
    n_mcmc_samples: int = Field(
        default=100, ge=10, le=500, description="Number of MCVI samples per prediction"
    )


class PredictionOutput(BaseModel):
    """Single RUL prediction with uncertainty."""

    predicted_rul_days: float = Field(..., description="Mean predicted RUL")
    uncertainty_days: float = Field(..., description="Standard deviation")
    confidence_interval_95: list[float] = Field(..., description="[lower, upper] 95% CI")
    risk_level: str = Field(..., description="LOW, MODERATE, HIGH, or CRITICAL")
    maintenance_recommended: bool = Field(..., description="Whether RUL < 45 days")
    inference_time_ms: float = Field(..., description="Server-side inference time")


class BatchOutput(BaseModel):
    """Batch prediction results."""

    predictions: list[PredictionOutput]
    total_time_ms: float
    model_version: str


class TrendInput(BaseModel):
    """Telemetry sequence/trend for trajectory analysis."""

    samples: list[TelemetryInput] = Field(
        ..., min_length=1, max_length=1000, description="Chronological telemetry sequence"
    )
    n_mcmc_samples: int = Field(
        default=50, ge=10, le=500, description="Number of MCVI samples per point in trend"
    )


class TrendPointOutput(BaseModel):
    """Prediction for a point in the trend sequence."""

    step: int = Field(..., description="1-based step index")
    predicted_rul_days: float = Field(..., description="Mean predicted RUL")
    uncertainty_days: float = Field(..., description="Standard deviation")
    confidence_interval_95: list[float] = Field(..., description="[lower, upper] 95% CI")
    risk_level: str = Field(..., description="LOW, MODERATE, HIGH, or CRITICAL")


class TrendOutput(BaseModel):
    """Trend trajectory and degradation analysis."""

    trend: list[TrendPointOutput]
    initial_rul_days: float = Field(..., description="Predicted RUL at step 1")
    latest_rul_days: float = Field(..., description="Predicted RUL at final step")
    total_rul_delta_days: float = Field(..., description="Change in RUL across sequence")
    degradation_trend: str = Field(..., description="DEGRADING, STABLE, or IMPROVING")
    total_time_ms: float = Field(..., description="Server-side inference time")
    model_version: str


class HealthStatus(BaseModel):
    """System health status."""

    status: str
    model_loaded: bool
    scaler_loaded: bool = False
    model_version: str
    device: str
    cuda_available: bool


class ModelInfo(BaseModel):
    """Model metadata."""

    name: str
    version: str
    architecture: str
    physics_constraints: bool
    input_features: list[str]
    output_format: str
    performance_metrics: dict[str, float]


# ─── MODEL LOADING (LIFESPAN) ──────────────────────────────────
_model: Optional[PhysicsGuidedBNN] = None
_config: Optional[dict] = None
_scaler_mean: Optional[np.ndarray] = None
_scaler_std: Optional[np.ndarray] = None
_model_version: str = "unknown"


def _find_local_artifacts() -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """Look for local model, config, and scaler files."""
    repo_root = Path(__file__).resolve().parents[2]

    # Check explicit env vars first
    model_env = os.environ.get("MODEL_PATH")
    config_env = os.environ.get("CONFIG_PATH")
    scaler_env = os.environ.get("SCALER_PATH")

    if model_env and Path(model_env).exists():
        m_path = Path(model_env)
        c_path = (
            Path(config_env)
            if config_env and Path(config_env).exists()
            else m_path.parent / "config.json"
        )
        s_path = (
            Path(scaler_env)
            if scaler_env and Path(scaler_env).exists()
            else m_path.parent / "scaler.npz"
        )
        return m_path, c_path if c_path.exists() else None, s_path if s_path.exists() else None

    candidate_dirs = [
        Path(os.environ.get("AEROVIGIL_WEIGHTS_DIR", "")),
        Path("/app/artifacts/pg_bnn_demo"),
        repo_root / "artifacts" / "pg_bnn_demo",
        Path("artifacts/pg_bnn_demo"),
        Path("/app/models"),
    ]

    for d in candidate_dirs:
        if str(d) and d.exists():
            m_path = d / "bnn_demo.pt"
            c_path = d / "config.json"
            s_path = d / "scaler.npz"
            if m_path.exists():
                return (
                    m_path,
                    c_path if c_path.exists() else None,
                    s_path if s_path.exists() else None,
                )

    return None, None, None


def _load_model() -> tuple[PhysicsGuidedBNN, dict, Optional[np.ndarray], Optional[np.ndarray]]:
    """Load model, config, and scaler from local path or Hugging Face Hub."""
    global _model_version, _scaler_mean, _scaler_std

    model_path, config_path, scaler_path = _find_local_artifacts()

    if model_path and config_path:
        with open(config_path) as f:
            config = json.load(f)
        model = PhysicsGuidedBNN(config)
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))

        mean, std = None, None
        if scaler_path and scaler_path.exists():
            try:
                scaler_data = np.load(scaler_path)
                mean = scaler_data["mean"].astype(np.float32)
                std = scaler_data["std"].astype(np.float32)
                std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
            except Exception as e:
                print(f"⚠️ Warning: Could not load scaler from {scaler_path}: {e}")
    else:
        # Download from Hugging Face
        config_p = hf_hub_download(repo_id=DEFAULT_REPO_ID, filename="config.json")
        model_p = hf_hub_download(repo_id=DEFAULT_REPO_ID, filename="bnn_demo.pt")

        with open(config_p) as f:
            config = json.load(f)

        model = PhysicsGuidedBNN(config)
        model.load_state_dict(torch.load(model_p, map_location="cpu", weights_only=True))

        mean, std = None, None
        try:
            scaler_p = hf_hub_download(repo_id=DEFAULT_REPO_ID, filename="scaler.npz")
            scaler_data = np.load(scaler_p)
            mean = scaler_data["mean"].astype(np.float32)
            std = scaler_data["std"].astype(np.float32)
            std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        except Exception:
            pass

    model.eval()
    _model_version = config.get("model_name", "unknown")
    _scaler_mean = mean
    _scaler_std = std

    return model, config, mean, std


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage model lifecycle."""
    global _model, _config, _scaler_mean, _scaler_std

    print("🔄 Loading model...")
    start = time.time()
    _model, _config, _scaler_mean, _scaler_std = _load_model()
    elapsed = time.time() - start
    scaler_status = "loaded" if _scaler_mean is not None else "not present"
    print(f"✅ Model loaded in {elapsed:.2f}s (scaler: {scaler_status})")

    yield

    print("🧹 Shutting down...")
    _model = None
    _config = None
    _scaler_mean = None
    _scaler_std = None


# ─── FASTAPI APP ───────────────────────────────────────────────
app = FastAPI(
    title="Aerovigil PG-BNN API",
    description="Physics-Guided Bayesian Neural Network for Wind Turbine RUL Prediction",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://aerovigil.abacusai.app", "https://huggingface.co"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─── UTILITY FUNCTIONS ─────────────────────────────────────────
def _run_inference(features: torch.Tensor, n_samples: int = 100) -> dict:
    """Run MCVI inference."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    _model.train()  # Enable dropout for MCVI

    pred_values: list[float] = []
    start = time.time()

    with torch.no_grad():
        for _ in range(n_samples):
            rul_mean, _ = _model(features)
            pred_values.append(rul_mean.item())

    inference_time = (time.time() - start) * 1000  # ms

    predictions = torch.tensor(pred_values)
    mean_rul = float(predictions.mean())
    std_rul = float(predictions.std())
    ci_lower = float(predictions.quantile(0.025))
    ci_upper = float(predictions.quantile(0.975))

    # Risk assessment
    if mean_rul < 14:
        risk = "CRITICAL"
    elif mean_rul < 30:
        risk = "HIGH"
    elif mean_rul < 45:
        risk = "MODERATE"
    else:
        risk = "LOW"

    return {
        "predicted_rul_days": round(mean_rul, 1),
        "uncertainty_days": round(std_rul, 1),
        "confidence_interval_95": [round(ci_lower, 1), round(ci_upper, 1)],
        "risk_level": risk,
        "maintenance_recommended": mean_rul < 45,
        "inference_time_ms": round(inference_time, 2),
    }


def _telemetry_to_tensor(telemetry: TelemetryInput) -> torch.Tensor:
    """Convert Pydantic model to tensor, applying training normalization scaler if present."""
    raw = np.array(
        [
            [
                telemetry.vibration_rms,
                telemetry.bearing_temp,
                telemetry.generator_temp,
                telemetry.power_output,
                telemetry.wind_speed,
                telemetry.operating_hours,
            ]
        ],
        dtype=np.float32,
    )
    if _scaler_mean is not None and _scaler_std is not None:
        raw = (raw - _scaler_mean) / _scaler_std
    return torch.tensor(raw, dtype=torch.float32)


# ─── API ENDPOINTS ─────────────────────────────────────────────
@app.get("/", response_model=dict)
async def root() -> dict:
    """API root with links."""
    return {
        "name": "Aerovigil PG-BNN API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "predict_batch": "/predict/batch",
        "predict_stream": "/predict/stream",
        "trend": "/trend",
        "predict_trend": "/predict/trend",
    }


@app.get("/health", response_model=HealthStatus)
async def health() -> HealthStatus:
    """Health check endpoint."""
    return HealthStatus(
        status="healthy" if _model is not None else "unhealthy",
        model_loaded=_model is not None,
        scaler_loaded=_scaler_mean is not None,
        model_version=_model_version,
        device=str(next(_model.parameters()).device) if _model else "none",
        cuda_available=torch.cuda.is_available(),
    )


@app.get("/model/info", response_model=ModelInfo)
async def model_info() -> ModelInfo:
    """Get model metadata."""
    if _config is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    perf = _config.get("performance", {})
    return ModelInfo(
        name=_config.get("model_name", "unknown"),
        version="0.1.0",
        architecture="Physics-Guided Bayesian Neural Network",
        physics_constraints=_config.get("physics", {}).get("enabled", False),
        input_features=list(_config.get("input_features", {}).keys()),
        output_format="gaussian (mean + log_variance)",
        performance_metrics={
            "mae_days": perf.get("mae_days", 0),
            "accuracy": perf.get("accuracy_45_day", 0),
            "recall": perf.get("recall", 0),
        },
    )


@app.post("/predict", response_model=PredictionOutput)
async def predict(
    request: Request,
    input_data: TelemetryInput,
    n_mcmc_samples: int = 100,
) -> PredictionOutput:
    """
    Predict RUL for a single telemetry reading.

    - **n_mcmc_samples**: Number of Monte Carlo samples (10-500)
    """
    features = _telemetry_to_tensor(input_data)
    result = _run_inference(features, n_mcmc_samples)
    return PredictionOutput(**result)


@app.post("/predict/batch", response_model=BatchOutput)
async def predict_batch(
    request: Request,
    batch_input: BatchInput,
) -> BatchOutput:
    """
    Predict RUL for multiple telemetry readings.

    Maximum 1000 samples per request.
    """
    if len(batch_input.samples) > 1000:
        raise HTTPException(status_code=422, detail="Batch size exceeds maximum of 1000")

    start = time.time()
    predictions: list[PredictionOutput] = []

    for sample in batch_input.samples:
        features = _telemetry_to_tensor(sample)
        result = _run_inference(features, batch_input.n_mcmc_samples)
        predictions.append(PredictionOutput(**result))

    total_time = (time.time() - start) * 1000

    return BatchOutput(
        predictions=predictions,
        total_time_ms=round(total_time, 2),
        model_version=_model_version,
    )


@app.post("/predict/stream")
async def predict_stream(
    request: Request,
    input_data: TelemetryInput,
    n_mcmc_samples: int = 100,
) -> StreamingResponse:
    """Stream prediction results (Server-Sent Events)."""

    async def event_generator() -> AsyncIterator[str]:
        if _model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        features = _telemetry_to_tensor(input_data)

        _model.train()
        for i in range(n_mcmc_samples):
            with torch.no_grad():
                rul_mean, _ = _model(features)
                payload = {"sample": i + 1, "rul": round(rul_mean.item(), 2)}
                yield "data: " + json.dumps(payload) + _SSE_END
            await asyncio.sleep(0.01)  # Small delay for streaming effect

        yield "data: [DONE]" + _SSE_END

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


def _calculate_trend(trend_input: TrendInput) -> TrendOutput:
    if len(trend_input.samples) > 1000:
        raise HTTPException(status_code=422, detail="Trend size exceeds maximum of 1000")

    start = time.time()
    points: list[TrendPointOutput] = []

    for idx, sample in enumerate(trend_input.samples, start=1):
        features = _telemetry_to_tensor(sample)
        res = _run_inference(features, trend_input.n_mcmc_samples)
        points.append(
            TrendPointOutput(
                step=idx,
                predicted_rul_days=res["predicted_rul_days"],
                uncertainty_days=res["uncertainty_days"],
                confidence_interval_95=res["confidence_interval_95"],
                risk_level=res["risk_level"],
            )
        )

    total_time = (time.time() - start) * 1000
    initial_rul = points[0].predicted_rul_days
    latest_rul = points[-1].predicted_rul_days
    delta = round(latest_rul - initial_rul, 2)

    if delta < -1.0:
        deg = "DEGRADING"
    elif delta > 1.0:
        deg = "IMPROVING"
    else:
        deg = "STABLE"

    return TrendOutput(
        trend=points,
        initial_rul_days=initial_rul,
        latest_rul_days=latest_rul,
        total_rul_delta_days=delta,
        degradation_trend=deg,
        total_time_ms=round(total_time, 2),
        model_version=_model_version,
    )


@app.post("/predict/trend", response_model=TrendOutput)
async def predict_trend(
    request: Request,
    trend_input: TrendInput,
) -> TrendOutput:
    """Predict RUL trend across a sequence of telemetry readings."""
    return _calculate_trend(trend_input)


@app.post("/trend", response_model=TrendOutput)
async def trend_endpoint(
    request: Request,
    trend_input: TrendInput,
) -> TrendOutput:
    """Alias for /predict/trend."""
    return _calculate_trend(trend_input)


# ─── ERROR HANDLERS ────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all exception handler."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "path": str(request.url),
        },
    )


# ─── MAIN ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
