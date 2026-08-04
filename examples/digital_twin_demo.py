#!/usr/bin/env python3
"""
Example demonstration of the WindTurbineDigitalTwin API.

Shows how to:
1. Load a turbine specification from the specs library.
2. Initialize the WindTurbineDigitalTwin virtual asset.
3. Ingest SCADA telemetry to calculate ISO 281 bearing life and active violations.
4. Run operational simulations under different scenario profiles.
5. Generate an AI reliability engineer prompt context.
"""

import os
import sys

# Adjust path to import src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.digital_twin import (  # noqa: E402
    WindTurbineDigitalTwin,
    generate_engineering_prompt,
    get_spec,
)
from src.utils.schema import BNNState, Telemetry  # noqa: E402


def main():
    print("=====================================================================")
    # 1. Load a turbine specification
    print("Step 1: Loading Turbine Specification from Library...")
    spec = get_spec("Vestas-V90")
    print(f"Loaded: {spec.model_name} (Rotor: {spec.rotor_diameter_m}m, Hub: {spec.hub_height_m}m)")
    print(f"Gearbox Temp Limit: {spec.temperature_limit_c}°C, Vibration Limit: {spec.vibration_limit_mms}mm/s")

    print("\n=====================================================================")
    # 2. Initialize the Digital Twin
    print("Step 2: Initializing WindTurbineDigitalTwin virtual asset...")
    twin = WindTurbineDigitalTwin(asset_id="WTG-VESTAS-042", spec=spec)
    print(f"Digital Twin active for asset: {twin.asset_id}")
    print(f"Initial Wear Index: {twin.cumulative_wear:.5f}")

    print("\n=====================================================================")
    # 3. Ingest SCADA Telemetry & update state
    print("Step 3: Ingesting Telemetry snapshots...")

    # Ingest a healthy snapshot
    healthy_tel = Telemetry(
        vibration_mms=1.4,
        temperature_c=58.0,
        rpm=1450.0,
        oil_viscosity_cst=32.0,
        load_pct=72.0,
    )
    bnn = BNNState(
        predicted_rul_days=310.0,
        epistemic_uncertainty=0.02,
        aleatoric_uncertainty=0.06,
    )

    state_rec = twin.update_state(healthy_tel, bnn)
    print("Healthy State Updated:")
    print(f" - Active Violations: {state_rec['physics_violations']}")
    print(f" - Bearing L10 rated life: {state_rec['bearing_l10_hours']:.1f} hours")
    print(f" - Cumulative wear index: {state_rec['cumulative_wear']:.5f}")

    # Ingest an anomalous/high-stress snapshot
    print("\nIngesting Anomalous high-vibration high-temperature telemetry...")
    anomalous_tel = Telemetry(
        vibration_mms=4.8,  # Limit is 4.2
        temperature_c=81.5,  # Limit is 78.0
        rpm=1600.0,
        oil_viscosity_cst=30.0,
        load_pct=105.0,
    )
    state_rec_anom = twin.update_state(anomalous_tel, bnn)
    print("Anomalous State Updated:")
    print(f" - Active Violations: {state_rec_anom['physics_violations']}")
    print(f" - Bearing L10 rated life: {state_rec_anom['bearing_l10_hours']:.1f} hours (accelerated damage)")
    print(f" - Cumulative wear index: {state_rec_anom['cumulative_wear']:.5f} (increased rate)")

    print("\n=====================================================================")
    # 4. Simulate a scenario profile
    print("Step 4: Running 12-hour 'overload' simulation scenario...")
    sim_records = twin.simulate_scenario(profile="overload", hours=12)
    print(f"Simulation completed over {len(sim_records)} steps.")
    print(f"Final simulated wear index: {twin.cumulative_wear:.5f}")
    print(f"Final bearing L10 life: {sim_records[-1]['bearing_l10_hours']:.1f} hours")

    print("\n=====================================================================")
    # 5. Generate copilot prompt context
    print("Step 5: Generating AI Reliability Engineer Prompt block...")
    prompt = generate_engineering_prompt(twin)
    print("\n--- BEGIN AI PROMPT CONTEXT ---")
    # Print first 25 lines of prompt
    print("\n".join(prompt.splitlines()[:30]))
    print("... [remainder of prompt truncated for display] ...")
    print("--- END AI PROMPT CONTEXT ---")
    print("=====================================================================")


if __name__ == "__main__":
    main()
