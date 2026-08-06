"""Prompt generation utilities for LLM-based engineering assistants and copilots."""

from __future__ import annotations

from typing import Any

from src.digital_twin.twin import WindTurbineDigitalTwin


def generate_engineering_prompt(twin: WindTurbineDigitalTwin) -> str:
    """
    Generate a comprehensive contextual prompt for an AI reliability engineer.

    Provides complete specifications, telemetry state history, safety constraints,
    and structured task prompts while strictly maintaining advisory boundaries.
    """
    if not twin.state_history:
        current_state: dict[str, Any] = {}
        history_summary = "No telemetry ingested yet."
    else:
        current_state = twin.state_history[-1]
        history_summary = f"Last ingested telemetry at {current_state['timestamp']}."

    spec = twin.spec
    violations = current_state.get("physics_violations", [])
    l10_hours = current_state.get("bearing_l10_hours", float("inf"))
    wear = current_state.get("cumulative_wear", 0.0)

    # Compile the prompt text
    prompt = f"""SYSTEM INSTRUCTIONS:
You are an expert Wind Turbine Reliability Engineer & Reliability Copilot.
Your goal is to analyze the telemetry, physical state, and specifications of the Wind Turbine Digital Twin, diagnose potential health anomalies, and formulate an engineering report.

--------------------------------------------------------------------------------
⚠️ SAFETY NOTICE — ADVISORY / DECISION-SUPPORT ONLY
Your advice is strict DECISION-SUPPORT ONLY.
* Do NOT recommend specific control or direct actuation commands (such as manual throttle percentages, RPM setpoints, generator breaker trips, or pitch angles).
* Do NOT emit Lockout/Tagout (LOTO) step-by-step procedures.
* Do NOT specify exact manufacturer SKU or tooling part numbers as authoritative instructions.
* Every recommendation must be presented as informational only, to be reviewed by a certified on-site technician and cross-checked with OEM documentation.
--------------------------------------------------------------------------------

TURBINE SPECIFICATIONS & ASSET IDENTITY:
- Asset ID: {twin.asset_id}
- Turbine Model: {spec.model_name}
- Manufacturer: {spec.manufacturer}
- Rated Power: {spec.rated_power_mw} MW
- Rotor Diameter: {spec.rotor_diameter_m} m
- Hub Height: {spec.hub_height_m} m
- Gearbox Speed-up Ratio: 1:{spec.gearbox_ratio}
- Bearing Dynamic Load Rating (C): {spec.bearing_dynamic_load_c_kn} kN
- Bearing Reference Equivalent Load (P): {spec.bearing_equivalent_load_p_kn} kN

DESIGN LIMITS (Gearbox):
- Vibration limit: {spec.vibration_limit_mms} mm/s RMS
- Temperature limit: {spec.temperature_limit_c} °C
- RPM limit: {spec.rpm_limit_hss} RPM
- Oil viscosity nominal range: [{spec.viscosity_min_cst}, {spec.viscosity_max_cst}] cSt

CURRENT TWIN HEALTH METRICS:
- Cumulative Wear index: {wear:.4f} (0.0 = New, 1.0 = Failure)
- ISO 281 Bearing L10 Rated Life (current conditions): {f"{l10_hours:.1f} hours" if l10_hours != float("inf") else "Infinite"}
- Active Physics Violations: {", ".join(violations) if violations else "None"}
- Last Updated: {twin.last_updated.isoformat()}
- Status: {history_summary}

CURRENT TELEMETRY SNAPSHOT:
"""
    if current_state:
        tel = current_state["telemetry"]
        prompt += f"""- Vibration: {tel["vibration_mms"]:.2f} mm/s
- Temperature: {tel["temperature_c"]:.1f} °C
- Speed: {tel["rpm"]:.1f} RPM
- Oil Viscosity: {tel["oil_viscosity_cst"]:.1f} cSt
- Generator Load: {tel["load_pct"]:.1f}%
"""
        if current_state.get("bnn_state"):
            bnn = current_state["bnn_state"]
            prompt += f"""
PROBABILISTIC BNN PREDICTIONS:
- Predicted Remaining Useful Life (RUL): {bnn["predicted_rul_days"]:.1f} days
- Epistemic Uncertainty (Model Ignorance): {bnn["epistemic_uncertainty"]:.3f}
- Aleatoric Uncertainty (Sensor Noise): {bnn["aleatoric_uncertainty"]:.3f}
"""
        advisory = current_state.get("advisory")
        if advisory:
            prompt += f"""
ADVISORY ENGINE OUTPUT (source: {current_state.get("advisory_source", "unknown")}):
- Predicted RUL: {advisory["predicted_rul_days"]:.1f} days
- Epistemic σ: {advisory["epistemic_std"]:.3f} | Aleatoric σ: {advisory["aleatoric_std"]:.3f}
- Suggested inspection window: {advisory["suggested_inspection_window_days"]:.1f} days
- Early warning (45-day horizon): {"TRIGGERED" if advisory["early_warning_triggered"] else "not triggered"}
"""
    else:
        prompt += "(No telemetry records available)\n"

    prompt += """
EXPERT ANALYSIS TASK:
Provide a structured, advisory-only engineering assessment covering:
1. **Asset Summary**: A high-level assessment of the turbine model and its operating state.
2. **Anomaly & Violation Assessment**: Explanation of any active physical violations, high vibration/temperature trends, or viscosity out-of-spec indices.
3. **Probabilistic RUL & Bearing Life Interpretation**: Compare the BNN's predicted RUL with the ISO 281 bearing L10 rated hours. Point out any high epistemic/aleatoric uncertainty.
4. **Maintenance Planning Advice**: Formulate a proposed inspection timeframe based on the RUL and safety criteria, explicitly stating that all findings must be verified on-site.
"""
    return prompt
