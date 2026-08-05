"""Wind Turbine Digital Twin class."""

from __future__ import annotations

import datetime
from collections import deque
from typing import Any

import pandas as pd

from src.data.ingest import CHANNELS
from src.digital_twin.specs import TurbineSpec
from src.physics.constraints import (
    GearboxPhysicsConstraints,
    check_violations,
    iso_281_l10_hours,
)
from src.utils.schema import BNNState, Telemetry, TurbinePayload

# Rolling telemetry buffer size used to build model features for advisories.
_ADVISORY_BUFFER_MAX = 512


class WindTurbineDigitalTwin:
    """
    Virtual representation (Digital Twin) of a physical wind turbine asset.

    Maintains physical specifications, manages current state history,
    computes advanced engineering health metrics (like bearing L10 life),
    models cumulative wear, and supports operator scenario simulation.
    """

    def __init__(self, asset_id: str, spec: TurbineSpec, serving_model=None):
        self.asset_id = asset_id
        self.spec = spec
        self.state_history: list[dict[str, Any]] = []
        self.cumulative_wear: float = 0.0  # Normalized wear index (0.0 = brand new, 1.0 = failure)
        self.last_updated: datetime.datetime = datetime.datetime.now(datetime.timezone.utc)
        # Raw snapshot buffer feeding the advisory feature pipeline.
        self._telemetry_buffer: deque[dict[str, float]] = deque(maxlen=_ADVISORY_BUFFER_MAX)
        self.serving_model = None
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

    def _compute_advisory(
        self, telemetry: Telemetry, bnn_state: BNNState | None
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Compute the advisory for this snapshot.

        Model path (when a serving model is attached) uses the rolling
        telemetry buffer to build window features; otherwise the incoming
        bnn_state block drives the advisory (previous behavior). Both paths
        flow through run_advisory → enforce_safety_contract.
        """
        payload = TurbinePayload(
            asset_id=self.asset_id, telemetry=telemetry, bnn_state=bnn_state
        )
        if self.serving_model is not None:
            df = pd.DataFrame(list(self._telemetry_buffer), columns=list(CHANNELS))
            return self.serving_model.advisory(payload, df), "model"
        if bnn_state is not None:
            from src.models.predictor import run_advisory

            return run_advisory(payload), "bnn_state"
        return None, None

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

        # Map our TurbineSpec constraints to a GearboxPhysicsConstraints object
        gb_constraints = GearboxPhysicsConstraints(
            vibration_limit_mms=self.spec.vibration_limit_mms,
            temperature_limit_c=self.spec.temperature_limit_c,
            rpm_limit_hss=self.spec.rpm_limit_hss,
            viscosity_min_cst=self.spec.viscosity_min_cst,
            viscosity_max_cst=self.spec.viscosity_max_cst,
        )

        # Check physical violations using spec-specific constraints
        telemetry_dict = telemetry.model_dump()
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
            wear_increment *= (1.5 + 0.5 * len(violations))

        self.cumulative_wear = min(1.0, self.cumulative_wear + wear_increment)
        self.last_updated = timestamp
        self._telemetry_buffer.append(telemetry_dict)

        # Bridge to the advisory engine: model path when a serving model is
        # attached, else the incoming bnn_state block (previous behavior).
        advisory, advisory_source = self._compute_advisory(telemetry, bnn_state)

        state_record = {
            "timestamp": timestamp.isoformat(),
            "telemetry": telemetry_dict,
            "bnn_state": bnn_state.model_dump() if bnn_state else None,
            "physics_violations": violations,
            "bearing_l10_hours": l10_hours,
            "cumulative_wear": self.cumulative_wear,
            "advisory": advisory,
            "advisory_source": advisory_source,
        }
        self.state_history.append(state_record)
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

        Returns a list of simulated state records.
        """
        simulated_records = []
        current_time = self.last_updated

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

        # Simulate step-by-step
        step_hours = 1.0
        steps = int(hours)
        for i in range(steps):
            current_time += datetime.timedelta(hours=step_hours)
            # Add small fluctuations to make it realistic
            factor = 1.0 + (0.05 * (hash(f"{self.asset_id}-{i}") % 100 - 50) / 100.0)

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
