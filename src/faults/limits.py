"""Per-asset detection-limit overrides.

Operators tune detection thresholds per turbine at runtime (e.g. raise the
vibration alarm level on a machine with a known high baseline).  Overrides
apply on top of the :class:`src.digital_twin.specs.TurbineSpec` limits and
persist in the durable store.

Supported override keys (all optional):

* ``vibration_limit_mms`` — ISO 10816-style vibration alarm level
* ``temperature_limit_c`` — gearbox oil temperature limit
* ``rpm_limit_hss`` — high-speed shaft RPM limit
* ``viscosity_min_cst`` / ``viscosity_max_cst`` — oil viscosity window
"""

from __future__ import annotations

from typing import Any

OVERRIDE_KEYS = (
    "vibration_limit_mms",
    "temperature_limit_c",
    "rpm_limit_hss",
    "viscosity_min_cst",
    "viscosity_max_cst",
)


def validate_overrides(overrides: dict[str, Any]) -> dict[str, float]:
    """Coerce and validate an override dict; raises ValueError on bad keys."""
    cleaned: dict[str, float] = {}
    for key, value in overrides.items():
        if key not in OVERRIDE_KEYS:
            raise ValueError(
                f"unknown limit override '{key}'; supported: {', '.join(OVERRIDE_KEYS)}"
            )
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"limit override '{key}' must be numeric, got {value!r}") from exc
        if not number > 0:
            raise ValueError(f"limit override '{key}' must be positive, got {number}")
        cleaned[key] = number
    return cleaned


def apply_overrides(limits, overrides: dict[str, float] | None) -> None:
    """Mutate a ``_Limits``-like object in place with validated overrides."""
    if not overrides:
        return
    for key, value in overrides.items():
        if hasattr(limits, key):
            setattr(limits, key, float(value))
