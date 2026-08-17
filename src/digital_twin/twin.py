"""Wind Turbine Digital Twin class.

Runtime hardening notes
-----------------------
* **Deterministic simulation** — scenario fluctuations come from a
  per-asset seeded RNG (CRC32 of the asset id), never from Python's
  process-randomized ``hash()``, so the same twin + profile + duration
  always reproduces the same trajectory.
* **Bounded memory** — retained state history is capped at
  ``max_history`` records and the advisory feature buffer at
  ``_ADVISORY_BUFFER_MAX`` snapshots, so a long-running twin cannot grow
  without bound.
* **Input validation** — non-finite telemetry/BNN values and invalid
  simulation durations are rejected with a clear ``ValueError`` instead of
  silently corrupting wear physics.
* **Advisory failover** — if the attached serving model raises during an
  update, the twin falls back to the ``bnn_state`` path (or records the
  error) instead of failing the whole state ingestion.
"""

from __future__ import annotations

import datetime
import logging
import math
import random
import zlib
from collections import deque
from typing import Any

import pandas as pd

from src.agents.cyber_team import build_cyber_team_brief
from src.data.ingest import CHANNELS
from src.digital_twin.specs import TurbineSpec
from src.faults.detector import FaultDetector, FaultReport
from src.physics.constraints import (
    GearboxPhysicsConstraints,
    check_violations,
    iso_281_l10_hours,
)
from src.utils.schema import BNNState, Telemetry, TurbinePayload

logger = logging.getLogger(__name__)

# Rolling telemetry buffer size used to build model features for advisories.
_ADVISORY_BUFFER_MAX = 512
# Default cap on retained state records per twin (memory bound).
DEFAULT_MAX_HISTORY = 10_000
# Longest scenario simulation allowed, in hours (one year of hourly steps).
MAX_SIMULATION_HOURS = 24.0 * 365.0


class WindTurbineDigitalTwin:
    """
    Virtual representation (Digital Twin) of a physical wind turbine asset.

    Maintains physical specifications, manages current state history,
    computes advanced engineering health metrics (like bearing L10 life),
    models cumulative wear, and supports operator scenario simulation.
    """

    def __init__(
        self,
        asset_id: str,
        spec: TurbineSpec,
        serving_model=None,
        *,
        max_history: int = DEFAULT_MAX_HISTORY,
        notifier=None,
    ):
        if max_history < 1:
            raise ValueError(f"max_history must be >= 1, got {max_history}")
        self.asset_id = asset_id
        self.spec = spec
        self.state_history: list[dict[str, Any]] = []
        self.max_history: int = max_history
        self.cumulative_wear: float = 0.0  # Normalized wear index (0.0 = brand new, 1.0 = failure)
        self.last_updated: datetime.datetime = datetime.datetime.now(datetime.timezone.utc)
        # Raw snapshot buffer feeding the advisory feature pipeline.
        self._telemetry_buffer: deque[dict[str, float]] = deque(maxlen=_ADVISORY_BUFFER_MAX)
        self.serving_model = None
        # Whole-turbine fault detection (taxonomy + oil analysis + rules).
        self.fault_detector = FaultDetector(spec)
        self.last_fault_report: FaultReport | None = None
        # Optional email notifier: raises CRITICAL/HIGH alerts on new findings.
        self.notifier = notifier
        self.last_notifications: list[dict] = []
        if serving_model is not None:
            self.attach_serving_model(serving_model)

    def attach_serving_model(self, serving_model) -> None:
        """Attach a :class:`src.models.serving.ServingModel` so every state
        update computes its advisory from the trained PG-BNN (rather than the
        incoming bnn_state block). Fails fast on feature-dim mismatch."""
        channels = tuple(serving_model.features_config.channels)
        if tuple(sorted(channels)) != tuple(sorted(CHANNELS)):
            raise ValueError(
                f"serving model channels {channels} do not match the twin's "
                f"telemetry channels {CHANNELS}"
            )
        expected = len(CHANNELS) * len(serving_model.features_config.stats)
        if serving_model.expected_feature_dim != expected:
            raise ValueError(
                f"feature-dim mismatch: serving model expects "
                f"{serving_model.expected_feature_dim} features, twin provides {expected}"
            )
        self.serving_model = serving_model

    @staticmethod
    def _reject_non_finite(values: dict[str, Any], what: str) -> None:
        """Fail fast on NaN/inf values that would silently poison physics
        calculations (wear, ISO 281 life) or feature extraction."""
        for key, value in values.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(
                    f"non-finite {what} value for '{key}': {value!r}. "
                    "Rejecting the snapshot to protect the twin runtime."
                )

    def _compute_advisory(
        self, telemetry: Telemetry, bnn_state: BNNState | None
    ) -> tuple[dict[str, Any] | None, str | None, str | None]:
        """Compute the advisory for this snapshot.

        Model path (when a serving model is attached) uses the rolling
        telemetry buffer to build window features; otherwise the incoming
        bnn_state block drives the advisory (previous behavior). Both paths
        flow through run_advisory → enforce_safety_contract.

        Returns ``(advisory, source, error)``. A serving-model failure does
        NOT fail the update: the twin falls back to the ``bnn_state`` path
        and records ``error`` for the caller to surface.
        """
        payload = TurbinePayload(asset_id=self.asset_id, telemetry=telemetry, bnn_state=bnn_state)
        if self.serving_model is not None:
            try:
                df = pd.DataFrame(list(self._telemetry_buffer), columns=list(CHANNELS))
                return self.serving_model.advisory(payload, df), "model", None
            except Exception as exc:  # model hiccups must not kill state ingestion
                error = f"serving model advisory failed, fell back to bnn_state path: {exc}"
                logger.warning("%s: %s", self.asset_id, error)
                if bnn_state is None:
                    return None, "model", error
                try:
                    from src.models.predictor import run_advisory

                    return run_advisory(payload), "bnn_state", error
                except Exception as exc2:  # pragma: no cover - defensive
                    return None, "bnn_state", f"{error}; bnn_state fallback failed: {exc2}"
        if bnn_state is not None:
            try:
                from src.models.predictor import run_advisory

                return run_advisory(payload), "bnn_state", None
            except Exception as exc:  # pragma: no cover - defensive
                return None, "bnn_state", f"bnn_state advisory failed: {exc}"
        return None, None, None

    def update_state(
        self,
        telemetry: Telemetry,
        bnn_state: BNNState | None = None,
        timestamp: datetime.datetime | None = None,
    ) -> dict[str, Any]:
        """
        Ingest a new telemetry snapshot, compute physical violations,
        calculate bearing life under current operating conditions,
        increment cumulative wear, and store the updated state in history.
        """
        if timestamp is None:
            timestamp = datetime.datetime.now(datetime.timezone.utc)

        # Fail fast on non-finite sensor values (defense in depth — the
        # pydantic schema already bounds every field).
        telemetry_dict = telemetry.model_dump()
        self._reject_non_finite(telemetry_dict, "telemetry")
        if bnn_state is not None:
            self._reject_non_finite(bnn_state.model_dump(), "bnn_state")

        # Map our TurbineSpec constraints to a GearboxPhysicsConstraints object
        gb_constraints = GearboxPhysicsConstraints(
            vibration_limit_mms=self.spec.vibration_limit_mms,
            temperature_limit_c=self.spec.temperature_limit_c,
            rpm_limit_hss=self.spec.rpm_limit_hss,
            viscosity_min_cst=self.spec.viscosity_min_cst,
            viscosity_max_cst=self.spec.viscosity_max_cst,
        )

        # Check physical violations using spec-specific constraints
        violations = check_violations(telemetry_dict, gb_constraints)

        # Calculate bearing L10 hours under current conditions
        # P_scaled = equivalent load scaled by load_pct
        load_multiplier = telemetry.load_pct / 100.0
        p_current = self.spec.bearing_equivalent_load_p_kn * max(0.1, load_multiplier)

        # RPM is the high speed shaft RPM
        l10_hours = iso_281_l10_hours(
            C=self.spec.bearing_dynamic_load_c_kn,
            P=p_current,
            p=10.0 / 3.0,  # roller bearings exponent
            rpm=max(1.0, telemetry.rpm),
        )

        # Dynamic cumulative wear model
        # Base wear increment per hour (scaled to sample interval, e.g. 1 hour = 0.0001)
        base_wear_rate = 1e-4

        # Stress factors accelerate wear
        vibration_stress = max(1.0, (telemetry.vibration_mms / self.spec.vibration_limit_mms) ** 2)
        temp_stress = max(1.0, (telemetry.temperature_c / self.spec.temperature_limit_c) ** 1.5)
        load_stress = max(1.0, (telemetry.load_pct / 100.0) ** 2)

        wear_increment = base_wear_rate * vibration_stress * temp_stress * load_stress
        if violations:
            # Multiplier for active physical limits violation
            wear_increment *= 1.5 + 0.5 * len(violations)

        self.cumulative_wear = min(1.0, self.cumulative_wear + wear_increment)
        self.last_updated = timestamp
        self._telemetry_buffer.append(telemetry_dict)

        # Whole-turbine fault detection: every part, every fault type. The
        # confirmation pass uses the previous buffered windows so persistent
        # faults gain confidence and first sightings are flagged as new.
        self.last_fault_report = self.fault_detector.detect(
            telemetry_dict,
            history=list(self._telemetry_buffer)[:-1],
            asset_id=self.asset_id,
            timestamp=timestamp.isoformat(),
        )
        if self.notifier is not None:
            # Severe (CRITICAL/HIGH) findings page the recipient by email;
            # dedupe/cooldown is handled inside the notifier.
            self.last_notifications = [
                n.to_dict() for n in self.notifier.process_report(self.last_fault_report)
            ]
        else:
            self.last_notifications = []

        # Bridge to the advisory engine: model path when a serving model is
        # attached, else the incoming bnn_state block (previous behavior).
        advisory, advisory_source, advisory_error = self._compute_advisory(telemetry, bnn_state)
        team_rul = advisory.get("predicted_rul_days") if advisory else None
        team_uncertainty = advisory.get("epistemic_std", 0.0) if advisory else 0.0
        if team_rul is None and bnn_state is not None:
            team_rul = bnn_state.predicted_rul_days
            team_uncertainty = bnn_state.epistemic_uncertainty
        agent_team = build_cyber_team_brief(
            asset_id=self.asset_id,
            predicted_rul_days=team_rul,
            epistemic_std=team_uncertainty,
            physics_violations=violations,
            cumulative_wear=self.cumulative_wear,
            bearing_l10_hours=l10_hours,
            telemetry=telemetry_dict,
        )

        state_record = {
            "timestamp": timestamp.isoformat(),
            "telemetry": telemetry_dict,
            "bnn_state": bnn_state.model_dump() if bnn_state else None,
            "physics_violations": violations,
            "bearing_l10_hours": l10_hours,
            "cumulative_wear": self.cumulative_wear,
            "advisory": advisory,
            "advisory_source": advisory_source,
            "advisory_error": advisory_error,
            "agent_team": agent_team,
            "fault_report": self.last_fault_report.to_dict(),
            "notifications": self.last_notifications,
        }
        self.state_history.append(state_record)
        if len(self.state_history) > self.max_history:
            # Memory bound: keep only the most recent records.
            del self.state_history[: len(self.state_history) - self.max_history]
        return state_record

    def simulate_scenario(
        self,
        profile: str = "nominal",
        hours: float = 24.0,
    ) -> list[dict[str, Any]]:
        """
        Simulate the progression of the digital twin over a given period (in hours)
        under a specified operating profile.

        Supported profiles:
          - "nominal": Normal operating conditions, nominal wear rate.
          - "overload": Turbine is pushed hard, elevated load, temperature, vibration. Accelerates wear.
          - "derated": Defensive mode, lower RPM, load, and temperature. Minimizes wear.
          - "viscosity_loss": Loss of oil viscosity scenario.

        ``hours`` must be a positive finite number no larger than
        :data:`MAX_SIMULATION_HOURS` (one year); anything else is rejected
        with :class:`ValueError`. The trajectory is deterministic for a
        given asset id (seeded RNG), regardless of process.

        Returns a list of simulated state records.
        """
        if not math.isfinite(hours) or hours <= 0:
            raise ValueError(f"hours must be a positive finite number of hours, got {hours!r}")
        if hours > MAX_SIMULATION_HOURS:
            raise ValueError(
                f"hours ({hours}) exceeds the maximum simulation horizon "
                f"of {MAX_SIMULATION_HOURS:.0f} hours (1 year)"
            )

        simulated_records = []
        current_time = self.last_updated

        # Deterministic per-asset fluctuation RNG (stable across processes,
        # unlike Python's salted hash()).
        rng = random.Random(zlib.crc32(self.asset_id.encode("utf-8")))

        # Define profile telemetry characteristics
        if profile == "nominal":
            vib = self.spec.vibration_limit_mms * 0.6
            temp = self.spec.temperature_limit_c * 0.75
            rpm = self.spec.rpm_limit_hss * 0.85
            visc = (self.spec.viscosity_min_cst + self.spec.viscosity_max_cst) / 2.0
            load = 75.0
        elif profile == "overload":
            vib = self.spec.vibration_limit_mms * 1.2
            temp = self.spec.temperature_limit_c * 1.1
            rpm = self.spec.rpm_limit_hss * 1.05
            visc = self.spec.viscosity_min_cst * 0.95
            load = 110.0
        elif profile == "derated":
            vib = self.spec.vibration_limit_mms * 0.4
            temp = self.spec.temperature_limit_c * 0.6
            rpm = self.spec.rpm_limit_hss * 0.5
            visc = (self.spec.viscosity_min_cst + self.spec.viscosity_max_cst) / 2.0
            load = 40.0
        elif profile == "viscosity_loss":
            vib = self.spec.vibration_limit_mms * 0.9
            temp = self.spec.temperature_limit_c * 1.05
            rpm = self.spec.rpm_limit_hss * 0.85
            visc = self.spec.viscosity_min_cst * 0.6
            load = 75.0
        else:
            raise ValueError(f"Unknown simulation profile: {profile}")

        # Simulate step-by-step (fractional hours round up to whole hourly steps)
        step_hours = 1.0
        steps = max(1, int(math.ceil(hours)))
        for _ in range(steps):
            current_time += datetime.timedelta(hours=step_hours)
            # Add small fluctuations to make it realistic
            factor = 1.0 + (0.05 * (rng.randrange(100) - 50) / 100.0)

            tel = Telemetry(
                vibration_mms=min(50.0, max(0.0, vib * factor)),
                temperature_c=min(200.0, max(-40.0, temp * factor)),
                rpm=min(3000.0, max(0.0, rpm * factor)),
                oil_viscosity_cst=min(500.0, max(1.0, visc * factor)),
                load_pct=min(120.0, max(0.0, load * factor)),
            )

            # Estimate mock BNN state based on wear
            # More wear -> lower predicted RUL
            mock_rul_days = max(1.0, (1.0 - self.cumulative_wear) * 365.0)
            bnn_s = BNNState(
                predicted_rul_days=mock_rul_days,
                epistemic_uncertainty=0.05 + 0.1 * self.cumulative_wear,
                aleatoric_uncertainty=0.08 + 0.05 * (vib / self.spec.vibration_limit_mms),
            )

            rec = self.update_state(tel, bnn_s, timestamp=current_time)
            simulated_records.append(rec)

        return simulated_records
