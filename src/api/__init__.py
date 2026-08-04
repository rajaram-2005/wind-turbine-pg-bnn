"""FastAPI advisory service package for wind-turbine-pg-bnn."""

from src.api.app import app, create_app
from src.api.schemas import (
    AdvisoryResponse,
    FleetRequest,
    FleetResponse,
    FleetSummary,
    HealthResponse,
)

__all__ = [
    "AdvisoryResponse",
    "FleetRequest",
    "FleetResponse",
    "FleetSummary",
    "HealthResponse",
    "app",
    "create_app",
]
