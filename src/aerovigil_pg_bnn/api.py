"""
FastAPI server for Aerovigil PG-BNN inference.

This module provides a production-ready REST API for wind turbine RUL prediction.
"""

import json
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from huggingface_hub import hf_hub_download


# ─── CONFIGURATION ─────────────────────────────────────────────
DEFAULT_REPO_ID = "AerovigilAI/wind-turbine-pg-bnn"
MODEL_PATH = "/app/models/bnn_demo.pt"
CONFIG_PATH = "/app/config.json"

# Input bounds for validation (physical constraints)
BOUNDS = {
    "vibration_rms": (0.0, 50.0),      # mm/s
    "bearing_temp": (20.0, 150.0),     # °C
    "generator_temp": (20.0, 200.0),   # °C
    "power_output": (0.0, 5000.0),     # kW
    "wind_speed": (0.0, 30.0),         # m/s
    "operating_hours": (0.0, 100000.0),# hours
}


# ─── PYDANTIC MODELS ───────────────────────────────────────────
class TelemetryInput(BaseModel):
    """Single SCADA telemetry reading for RUL prediction."""

    vibration_rms: float = Field(
        ..., ge=BOUNDS["vibration_rms"][0], le=BOUNDS["vibration_rms"][1],
        description="Drive-train vibration RMS (mm/s)"
    )
    bearing_temp: float = Field(
        ..., ge=BOUNDS["bearing_temp"][0], le=BOUNDS["bearing_temp"][1],
        description="Main bearing temperature (°C)"
    )
    generator_temp: float = Field(
        ..., ge=BOUNDS["generator_temp"][0], le=BOUNDS["generator_temp"][1],
        description="Generator winding temperature (°C)"
    )
    power_output: float = Field(
        ..., ge=BOUNDS["power_output"][0], le=BOUNDS["power_output"][1],
        description="Active power generation (kW)"
    )
    wind_speed: float = Field(
        ..., ge=BOUNDS["wind_speed"][0], le=BOUNDS["wind_speed"][1],
        description="Nacelle wind speed (m/s)"
    )
    operating_hours: float = Field(
        ..., ge=BOUNDS["operating_hours"][0], le=BOUNDS["operating_hours"][1],
        description="Cumulative operating time (hours)"
    )


class BatchInput(BaseModel):
    """Batch of telemetry readings."""

    samples: List[TelemetryInput] = Field(
        ..., min_length=1, max_length=1000,
        description="List of telemetry readings"
    )
    n_mcmc_samples: int = Field(
        default=100, ge=10, le=500,
        description="Number of MCVI samples per prediction"
    )


class PredictionOutput(BaseModel):
    """Single RUL prediction with uncertainty."""

    predicted_rul_days: float = Field(..., description="Mean predicted RUL")
    uncertainty_days: float = Field(..., description="Standard deviation")
    confidence_interval_95: List[float] = Field(
        ..., description="[lower, upper] 95% CI"
    )
    risk_level: str = Field(..., description="LOW, MODERATE, HIGH, or CRITICAL")
    maintenance_recommended: bool = Field(
        ..., description="Whether RUL < 45 days"
    )
    inference_time_ms: float = Field(..., description="Server-side inference time")


class BatchOutput(BaseModel):
    """Batch prediction results."""

    predictions: List[PredictionOutput]
    total_time_ms: float
    model_version: str


class HealthStatus(BaseModel):
    """System health status."""

    status: str
    model_loaded: bool
    model_version: str
    device: str
    cuda_available: bool


class ModelInfo(BaseModel):
    """Model metadata."""

    name: str
    version: str
    architecture: str
    physics_constraints: bool
    input_features: List[str]
    output_format: str
    performance_metrics: Dict[str, float]


# ─── MODEL LOADING (LIFESPAN) ──────────────────────────────────
_model: Optional[PhysicsGuidedBNN] = None
_config: Optional[Dict] = None
_model_version: str = "unknown"


def _load_model() -> tuple:
    """Load model from Hugging Face Hub or local path."""
    global _model_version

    # Try local first, then download from HF
    if Path(MODEL_PATH).exists() and Path(CONFIG_PATH).exists():
        with open(CONFIG_PATH) as f:
            config = json.load(f)
        model = PhysicsGuidedBNN(config)
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        )
    else:
        # Download from Hugging Face
        config_path = hf_hub_download(
            repo_id=DEFAULT_REPO_ID, filename="config.json"
        )
        model_path = hf_hub_download(
            repo_id=DEFAULT_REPO_ID, filename="bnn_demo.pt"
        )

        with open(config_path) as f:
            config = json.load(f)

        model = PhysicsGuidedBNN(config)
        model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )

    model.eval()
    _model_version = config.get("model_name", "unknown")

    return model, config


from pathlib import Path
from .model import PhysicsGuidedBNN


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage model lifecycle."""
    global _model, _config

    print("🔄 Loading model...")
    start = time.time()
    _model, _config = _load_model()
    elapsed = time.time() - start
    print(f"✅ Model loaded in {elapsed:.2f}s")

    yield

    print("🧹 Shutting down...")
    _model = None
    _config = None


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
def _run_inference(features: torch.Tensor, n_samples: int = 100) -> Dict:
    """Run MCVI inference."""
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    _model.train()  # Enable dropout for MCVI

    predictions = []
    start = time.time()

    with torch.no_grad():
        for _ in range(n_samples):
            rul_mean, _ = _model(features)
            predictions.append(rul_mean.item())

    inference_time = (time.time() - start) * 1000  # ms

    predictions = torch.tensor(predictions)
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
    """Convert Pydantic model to tensor."""
    return torch.tensor([[
        telemetry.vibration_rms,
        telemetry.bearing_temp,
        telemetry.generator_temp,
        telemetry.power_output,
        telemetry.wind_speed,
        telemetry.operating_hours,
    ]], dtype=torch.float32)


# ─── API ENDPOINTS ─────────────────────────────────────────────
@app.get("/", response_model=Dict)
async def root():
    """API root with links."""
    return {
        "name": "Aerovigil PG-BNN API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health", response_model=HealthStatus)
async def health():
    """Health check endpoint."""
    return HealthStatus(
        status="healthy" if _model is not None else "unhealthy",
        model_loaded=_model is not None,
        model_version=_model_version,
        device=str(next(_model.parameters()).device) if _model else "none",
        cuda_available=torch.cuda.is_available(),
    )


@app.get("/model/info", response_model=ModelInfo)
async def model_info():
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
):
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
):
    """
    Predict RUL for multiple telemetry readings.

    Maximum 1000 samples per request.
    """
    if len(batch_input.samples) > 1000:
        raise HTTPException(
            status_code=422,
            detail="Batch size exceeds maximum of 1000"
        )

    start = time.time()
    predictions = []

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
):
    """Stream prediction results (Server-Sent Events)."""
    from fastapi.responses import StreamingResponse

    async def event_generator():
        features = _telemetry_to_tensor(input_data)

        _model.train()
        for i in range(n_mcmc_samples):
            with torch.no_grad():
                rul_mean, _ = _model(features)
                yield f"data: {{\"sample\": {i+1}, \"rul\": {rul_mean.item():.2f}}}\n\n"
            await asyncio.sleep(0.01)  # Small delay for streaming effect

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


# ─── ERROR HANDLERS ────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
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
