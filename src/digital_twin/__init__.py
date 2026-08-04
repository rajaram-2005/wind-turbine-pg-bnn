"""Digital Twin subpackage for wind turbines."""

from __future__ import annotations

from src.digital_twin.prompts import generate_engineering_prompt
from src.digital_twin.specs import SPECS_LIBRARY, TurbineSpec, get_spec, list_specs
from src.digital_twin.twin import WindTurbineDigitalTwin

__all__ = [
    "SPECS_LIBRARY",
    "TurbineSpec",
    "WindTurbineDigitalTwin",
    "generate_engineering_prompt",
    "get_spec",
    "list_specs",
]
