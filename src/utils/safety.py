"""
Safety gate for wind-turbine-pg-bnn.

All outputs of the RUL predictor must pass through `AdvisoryRecommendation`,
which enforces the ADVISORY_ONLY contract documented in docs/SAFETY.md:

* No direct actuation commands (throttles, RPM setpoints, torque, breaker
  trips, pitch commands).
* No fabricated Lockout/Tagout (LOTO) procedures.
* No part / tool / SKU numbers presented as authoritative maintenance
  instructions.

Attempting to attach such fields raises SafetyBoundaryError. This is a
deliberate fail-closed guard so that accidental wiring of the predictor
into a control path cannot silently emit dangerous payloads.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass, field
from typing import Any

# Fields that are NEVER permitted in a recommendation payload.
# Matched case-insensitively against both top-level keys and nested keys.
_BLOCKED_KEY_PATTERNS: list[re.Pattern] = [
    re.compile(r"throttle", re.IGNORECASE),
    re.compile(r"torque_demand", re.IGNORECASE),
    re.compile(r"pitch_command", re.IGNORECASE),
    re.compile(r"rpm_setpoint", re.IGNORECASE),
    re.compile(r"breaker", re.IGNORECASE),
    re.compile(r"loto", re.IGNORECASE),
    re.compile(r"lockout", re.IGNORECASE),
    re.compile(r"tagout", re.IGNORECASE),
    re.compile(r"sku", re.IGNORECASE),
    re.compile(r"part_number", re.IGNORECASE),
    re.compile(r"tool_part", re.IGNORECASE),
    re.compile(r"actuat(e|ion)", re.IGNORECASE),
]


class SafetyBoundaryError(RuntimeError):
    """Raised when code attempts to produce a direct-actuation output."""


@dataclass(frozen=True)
class AdvisoryRecommendation:
    """
    A decision-support payload for a reliability engineer.

    All fields are informational. There are intentionally no setpoint,
    throttle, torque, pitch, breaker, LOTO, or part-number fields.
    """

    asset_id: str
    predicted_rul_days: float
    epistemic_std: float
    aleatoric_std: float
    physics_violations: list[str] = field(default_factory=list)
    suggested_inspection_window_days: float = 7.0
    early_warning_triggered: bool = False
    warning_horizon_days: float = 45.0
    rationale: str = ""
    advisory_only: bool = True
    generated_at: str = field(
        default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        # Freeze-by-contract: explicitly list allowed keys, do NOT pass through
        # arbitrary kwargs that consumers might try to stuff with commands.
        return {
            "asset_id": self.asset_id,
            "predicted_rul_days": float(self.predicted_rul_days),
            "epistemic_std": float(self.epistemic_std),
            "aleatoric_std": float(self.aleatoric_std),
            "physics_violations": list(self.physics_violations),
            "suggested_inspection_window_days": float(self.suggested_inspection_window_days),
            "early_warning_triggered": bool(self.early_warning_triggered),
            "warning_horizon_days": float(self.warning_horizon_days),
            "rationale": self.rationale,
            "advisory_only": True,
            "generated_at": self.generated_at,
            "disclaimer": (
                "Decision-support only. Not a direct actuation command. "
                "Review by a qualified operator and cross-check against OEM "
                "documentation is required before any maintenance action."
            ),
        }


def enforce_safety_contract(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Walk a candidate payload dict and raise SafetyBoundaryError if any
    forbidden direct-actuation field is present.

    Used as a defensive boundary at the edges of the system (e.g. before
    serializing to JSON for a UI, before writing to a message bus).
    """

    def _walk(obj: Any, path: str = "") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                keypath = f"{path}.{k}" if path else str(k)
                for pat in _BLOCKED_KEY_PATTERNS:
                    if pat.search(str(k)):
                        raise SafetyBoundaryError(
                            f"Blocked key '{keypath}' matches forbidden "
                            f"pattern '{pat.pattern}'. Direct-actuation fields "
                            f"are not permitted by the advisory-only contract."
                        )
                _walk(v, keypath)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _walk(v, f"{path}[{i}]")

    _walk(payload)
    return payload
