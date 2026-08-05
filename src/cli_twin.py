"""CLI commands for wind turbine Digital Twin operations (twin-*)."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from src.digital_twin.prompts import generate_engineering_prompt
from src.digital_twin.specs import get_spec
from src.digital_twin.twin import WindTurbineDigitalTwin
from src.utils.encoding import configure_utf8_stdio
from src.utils.schema import BNNState, Telemetry


def load_optional_telemetry(payload_path: str | None) -> tuple[Telemetry | None, BNNState | None]:
    """Parse telemetry and optional BNNState from a JSON payload path."""
    if not payload_path:
        return None, None
    try:
        data = json.loads(Path(payload_path).read_text(encoding="utf-8"))
        tel_data = data.get("telemetry", data)
        telemetry = Telemetry(
            vibration_mms=tel_data["vibration_mms"],
            temperature_c=tel_data["temperature_c"],
            rpm=tel_data["rpm"],
            oil_viscosity_cst=tel_data["oil_viscosity_cst"],
            load_pct=tel_data["load_pct"],
        )
        bnn_data = data.get("bnn_state")
        bnn_state = None
        if bnn_data:
            bnn_state = BNNState(
                predicted_rul_days=bnn_data["predicted_rul_days"],
                epistemic_uncertainty=bnn_data["epistemic_uncertainty"],
                aleatoric_uncertainty=bnn_data["aleatoric_uncertainty"],
            )
        return telemetry, bnn_state
    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error parsing payload JSON: {e}", file=sys.stderr)
        sys.exit(1)


# --------------------------------------------------------------------------- #
# CLI Entrypoints                                                             #
# --------------------------------------------------------------------------- #

def status_main(argv: Sequence[str] | None = None) -> int:
    """Entrypoint for `twin-status` CLI."""
    parser = argparse.ArgumentParser(
        prog="twin-status",
        description="Fetch or print the current status of a Wind Turbine Digital Twin.",
    )
    parser.add_argument("--asset-id", default="WTG-001", help="Turbine Asset identifier.")
    parser.add_argument("--model", default="GE-1.5", help="Turbine model from specs library.")
    parser.add_argument("--payload", help="Path to optional telemetry payload JSON.")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format.")
    parser.add_argument(
        "--advisory",
        action="store_true",
        help="Compute and print the advisory engine output for this state "
        "(trained-model path when --model-path is given, else the bnn_state block).",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        metavar="CHECKPOINT",
        help="Optional trained PG-BNN bundle to attach to the twin "
        "(advisories then come from the model, not bnn_state).",
    )

    args = parser.parse_args(argv)

    configure_utf8_stdio()

    try:
        spec = get_spec(args.model)
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    serving = None
    if args.model_path:
        from src.models.serving import load_serving_model

        serving = load_serving_model(args.model_path)

    twin = WindTurbineDigitalTwin(args.asset_id, spec, serving_model=serving)

    telemetry, bnn_state = load_optional_telemetry(args.payload)
    if telemetry:
        twin.update_state(telemetry, bnn_state)
    else:
        # Load a default healthy state
        default_tel = Telemetry(
            vibration_mms=1.5,
            temperature_c=55.0,
            rpm=1440.0,
            oil_viscosity_cst=32.0,
            load_pct=75.0,
        )
        default_bnn = BNNState(
            predicted_rul_days=320.0,
            epistemic_uncertainty=0.02,
            aleatoric_uncertainty=0.07,
        )
        twin.update_state(default_tel, default_bnn)

    if args.format == "json":
        output = {
            "asset_id": twin.asset_id,
            "specification": spec.model_dump(),
            "health_state": twin.state_history[-1] if twin.state_history else None,
        }
        print(json.dumps(output, indent=2))
    else:
        last_rec = twin.state_history[-1]
        print("============================================================")
        print(f"Digital Twin Status for Asset: {twin.asset_id}")
        print("============================================================")
        print(f"Model: {spec.model_name} (by {spec.manufacturer})")
        print(f"Rated Power: {spec.rated_power_mw} MW")
        print(f"Gearbox Ratio: 1:{spec.gearbox_ratio}")
        print("------------------------------------------------------------")
        print(f"Vibration Level: {last_rec['telemetry']['vibration_mms']:.2f} mm/s (Limit: {spec.vibration_limit_mms} mm/s)")
        print(f"Oil Temperature: {last_rec['telemetry']['temperature_c']:.1f} °C (Limit: {spec.temperature_limit_c} °C)")
        print(f"HSS Shaft Speed: {last_rec['telemetry']['rpm']:.1f} RPM (Limit: {spec.rpm_limit_hss} RPM)")
        print("------------------------------------------------------------")
        print(f"Calculated ISO 281 Bearing L10 Life: {last_rec['bearing_l10_hours']:.1f} hours")
        print(f"Cumulative Physical Wear Index: {last_rec['cumulative_wear']:.5f}")
        print(f"Active Physical Violations: {', '.join(last_rec['physics_violations']) if last_rec['physics_violations'] else 'None'}")
        if last_rec["bnn_state"]:
            print(f"Probabilistic predicted RUL: {last_rec['bnn_state']['predicted_rul_days']:.1f} days")
        if args.advisory:
            adv = last_rec.get("advisory")
            print("------------------------------------------------------------")
            if adv:
                print(f"Advisory (source: {last_rec.get('advisory_source')}):")
                print(f"  Predicted RUL: {adv['predicted_rul_days']:.1f} days "
                      f"(epistemic σ={adv['epistemic_std']:.3f}, aleatoric σ={adv['aleatoric_std']:.3f})")
                print(f"  Suggested inspection window: {adv['suggested_inspection_window_days']:.1f} days")
                print("  Early warning (45d): "
                      f"{'TRIGGERED' if adv['early_warning_triggered'] else 'not triggered'}")
            else:
                print("Advisory: none available (no serving model attached and no bnn_state)")
        print("============================================================")

    return 0


def simulate_main(argv: Sequence[str] | None = None) -> int:
    """Entrypoint for `twin-simulate` CLI."""
    parser = argparse.ArgumentParser(
        prog="twin-simulate",
        description="Simulate wind turbine operations and wear profile progression.",
    )
    parser.add_argument("--asset-id", default="WTG-SIM", help="Turbine Asset identifier.")
    parser.add_argument("--model", default="GE-1.5", help="Turbine model from specs library.")
    parser.add_argument(
        "--profile",
        choices=["nominal", "overload", "derated", "viscosity_loss"],
        default="nominal",
        help="Operating profile to simulate.",
    )
    parser.add_argument("--hours", type=float, default=24.0, help="Simulation duration in hours.")
    parser.add_argument("-o", "--output", help="Path to save simulation history JSON.")

    args = parser.parse_args(argv)

    configure_utf8_stdio()

    try:
        spec = get_spec(args.model)
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    twin = WindTurbineDigitalTwin(args.asset_id, spec)

    print(f"Starting simulation for {args.asset_id} [{spec.model_name}]...")
    print(f"Profile: '{args.profile}', Duration: {args.hours} hours...")

    records = twin.simulate_scenario(profile=args.profile, hours=args.hours)

    print(f"Simulation completed. Cumulative wear index: {twin.cumulative_wear:.5f}")
    print(f"Final simulated state bearing L10 life: {records[-1]['bearing_l10_hours']:.1f} hours")

    if args.output:
        Path(args.output).write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"Saved simulation history of {len(records)} steps to {args.output}")
    else:
        # Print summary table of last few steps
        print("\nLast 5 simulation steps:")
        print(f"{'Timestamp':<25} | {'Vib (mm/s)':<10} | {'Temp (°C)':<10} | {'Wear Index':<12} | {'Violations':<15}")
        print("-" * 85)
        for r in records[-5:]:
            v_str = ", ".join(r["physics_violations"]) if r["physics_violations"] else "None"
            print(
                f"{r['timestamp'][:19]:<25} | "
                f"{r['telemetry']['vibration_mms']:<10.2f} | "
                f"{r['telemetry']['temperature_c']:<10.1f} | "
                f"{r['cumulative_wear']:<12.5f} | "
                f"{v_str:<15}"
            )

    return 0


def prompt_main(argv: Sequence[str] | None = None) -> int:
    """Entrypoint for `twin-prompt` CLI."""
    parser = argparse.ArgumentParser(
        prog="twin-prompt",
        description="Generate contextual AI/LLM reliability advisor prompts from Digital Twin state.",
    )
    parser.add_argument("--asset-id", default="WTG-001", help="Turbine Asset identifier.")
    parser.add_argument("--model", default="GE-1.5", help="Turbine model from specs library.")
    parser.add_argument("--payload", help="Path to optional telemetry payload JSON.")
    parser.add_argument("-o", "--output", help="Path to write prompt text (default: stdout).")

    args = parser.parse_args(argv)

    configure_utf8_stdio()

    try:
        spec = get_spec(args.model)
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    twin = WindTurbineDigitalTwin(args.asset_id, spec)

    telemetry, bnn_state = load_optional_telemetry(args.payload)
    if telemetry:
        twin.update_state(telemetry, bnn_state)
    else:
        # Load default state with a minor vibration issue to make prompt interesting
        default_tel = Telemetry(
            vibration_mms=4.8,  # slightly above 4.5
            temperature_c=78.2,
            rpm=1720.0,
            oil_viscosity_cst=11.5,
            load_pct=98.0,
        )
        default_bnn = BNNState(
            predicted_rul_days=25.4,
            epistemic_uncertainty=0.08,
            aleatoric_uncertainty=0.15,
        )
        twin.update_state(default_tel, default_bnn)

    prompt = generate_engineering_prompt(twin)

    if args.output:
        Path(args.output).write_text(prompt, encoding="utf-8")
        print(f"Contextual prompt saved to {args.output}")
    else:
        sys.stdout.write(prompt if prompt.endswith("\n") else prompt + "\n")

    return 0
