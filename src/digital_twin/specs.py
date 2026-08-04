"""Specs Library for Wind Turbine models and their physical constraints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TurbineSpec(BaseModel):
    """Configuration specification for a specific wind turbine model."""

    model_name: str = Field(..., description="Model name of the turbine")
    manufacturer: str = Field(..., description="Manufacturer of the turbine")
    rated_power_mw: float = Field(..., ge=0.1, description="Rated electrical power output in MW")
    rotor_diameter_m: float = Field(..., ge=10.0, description="Rotor diameter in meters")
    hub_height_m: float = Field(..., ge=10.0, description="Hub height in meters")
    gearbox_ratio: float = Field(..., ge=1.0, description="Gearbox speed-up ratio")

    # Physical limits for the digital twin's gearbox & generator
    vibration_limit_mms: float = Field(4.5, ge=0.0, description="Vibration limit in mm/s RMS")
    temperature_limit_c: float = Field(80.0, ge=0.0, description="Gearbox oil temperature limit in °C")
    rpm_limit_hss: float = Field(1800.0, ge=0.0, description="High-speed shaft RPM limit")
    viscosity_min_cst: float = Field(10.0, ge=0.0, description="Minimum allowed oil viscosity in cSt")
    viscosity_max_cst: float = Field(50.0, ge=0.0, description="Maximum allowed oil viscosity in cSt")

    # Bearing mechanical specifications for ISO 281 life calculations
    bearing_dynamic_load_c_kn: float = Field(1200.0, ge=1.0, description="Basic dynamic load rating C in kN")
    bearing_equivalent_load_p_kn: float = Field(180.0, ge=1.0, description="Reference equivalent load P in kN")


# Predefined specs library of standard commercial and reference wind turbines
SPECS_LIBRARY: dict[str, TurbineSpec] = {
    "GE-1.5": TurbineSpec(
        model_name="GE 1.5 SLE",
        manufacturer="GE Renewable Energy",
        rated_power_mw=1.5,
        rotor_diameter_m=77.0,
        hub_height_m=80.0,
        gearbox_ratio=72.0,
        vibration_limit_mms=4.5,
        temperature_limit_c=80.0,
        rpm_limit_hss=1800.0,
        viscosity_min_cst=10.0,
        viscosity_max_cst=50.0,
        bearing_dynamic_load_c_kn=1200.0,
        bearing_equivalent_load_p_kn=180.0,
    ),
    "Vestas-V90": TurbineSpec(
        model_name="V90-2.0 MW",
        manufacturer="Vestas",
        rated_power_mw=2.0,
        rotor_diameter_m=90.0,
        hub_height_m=80.0,
        gearbox_ratio=113.5,
        vibration_limit_mms=4.2,
        temperature_limit_c=78.0,
        rpm_limit_hss=1650.0,
        viscosity_min_cst=12.0,
        viscosity_max_cst=45.0,
        bearing_dynamic_load_c_kn=1600.0,
        bearing_equivalent_load_p_kn=220.0,
    ),
    "Siemens-SWT-2.3": TurbineSpec(
        model_name="SWT-2.3-101",
        manufacturer="Siemens Wind Power",
        rated_power_mw=2.3,
        rotor_diameter_m=101.0,
        hub_height_m=99.5,
        gearbox_ratio=91.0,
        vibration_limit_mms=4.8,
        temperature_limit_c=82.0,
        rpm_limit_hss=1500.0,
        viscosity_min_cst=11.0,
        viscosity_max_cst=48.0,
        bearing_dynamic_load_c_kn=1900.0,
        bearing_equivalent_load_p_kn=260.0,
    ),
    "NREL-5MW": TurbineSpec(
        model_name="NREL 5MW Reference Turbine",
        manufacturer="NREL",
        rated_power_mw=5.0,
        rotor_diameter_m=126.0,
        hub_height_m=90.0,
        gearbox_ratio=97.0,
        vibration_limit_mms=5.0,
        temperature_limit_c=85.0,
        rpm_limit_hss=1173.7,
        viscosity_min_cst=8.0,
        viscosity_max_cst=60.0,
        bearing_dynamic_load_c_kn=3200.0,
        bearing_equivalent_load_p_kn=450.0,
    ),
}


def get_spec(model_key: str) -> TurbineSpec:
    """Retrieve turbine specification from the library, falling back to GE-1.5 if not found."""
    if model_key in SPECS_LIBRARY:
        return SPECS_LIBRARY[model_key]
    # Allow loose matching or default
    cleaned_key = model_key.strip().replace(" ", "-")
    for k, spec in SPECS_LIBRARY.items():
        if k.lower() == cleaned_key.lower() or spec.model_name.lower() == model_key.lower():
            return spec
    raise KeyError(f"Turbine model '{model_key}' not found in specifications library. Available models: {list(SPECS_LIBRARY.keys())}")


def list_specs() -> dict[str, dict[str, str | float]]:
    """Return a summary of all predefined specifications."""
    return {
        key: {
            "model_name": spec.model_name,
            "manufacturer": spec.manufacturer,
            "rated_power_mw": spec.rated_power_mw,
            "rotor_diameter_m": spec.rotor_diameter_m,
            "hub_height_m": spec.hub_height_m,
        }
        for key, spec in SPECS_LIBRARY.items()
    }
