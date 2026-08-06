"""Streamlit advisory UI for AeroVigil (wind-turbine-pg-bnn engine).

AeroVigil v1.0.0 — https://aerovigil.abacusai.app

Run::

    streamlit run src/ui/app.py

Decision-support only — the UI deliberately exposes no actuation controls
(no throttle, pitch, RPM-setpoint, breaker, or LOTO widgets). Every computed
recommendation is screened by ``enforce_safety_contract`` before display.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.models.predictor import run_advisory
from src.reporting.reports import (
    advisories_from_dataframe,
    build_fleet_report,
    format_advisory_markdown,
)
from src.ui.defaults import default_snapshot
from src.utils.safety import enforce_safety_contract
from src.utils.schema import BNNState, Telemetry, TurbinePayload

SAFETY_BANNER = (
    "⚠️ **ADVISORY / DECISION-SUPPORT ONLY.** This tool does not issue PLC/SCADA "
    "setpoints, torque throttles, speed commands, or Lockout/Tagout (LOTO) "
    "procedures. Review every output with a qualified operator and cross-check "
    "against OEM documentation before any maintenance action. See `docs/SAFETY.md`."
)

# Sensible defaults that double as field documentation — threaded from
# configs/default.yaml (ui.default_snapshot); values unchanged.
_DEFAULTS = default_snapshot()


@st.cache_resource(show_spinner=False)
def _load_serving(bundle_path: str):
    """Load a trained PG-BNN serving bundle (cached across reruns)."""
    from src.models.serving import load_serving_model

    return load_serving_model(bundle_path)


def _model_based_advisory(payload, model_path: str, window_csv, tel: dict) -> dict:
    """Serving path: model computes RUL/uncertainties from a telemetry window.

    If no window CSV is uploaded, the current snapshot is repeated over the
    model's expected window size so a single reading still yields an
    advisory (documented demo behavior).
    """
    serving = _load_serving(model_path)
    channels = list(serving.features_config.channels)
    if window_csv is not None:
        df = pd.read_csv(window_csv)
        if "timestamp" in df.columns:
            df = df.drop(columns=["timestamp"])
    else:
        n = max(int(serving.features_config.window_size), 1)
        df = pd.DataFrame({ch: [float(tel[ch])] * n for ch in channels})
    return serving.advisory(payload, df)


st.set_page_config(page_title="AeroVigil advisory", page_icon="🌀", layout="wide")
st.warning(SAFETY_BANNER)
st.title("🌀 AeroVigil — RUL advisory")
st.caption(
    "**AeroVigil v1.0.0** — Physics-Guided Bayesian Neural Network for drivetrain "
    "remaining-useful-life prediction · [aerovigil.abacusai.app](https://aerovigil.abacusai.app)"
)

tab_single, tab_fleet, tab_twin, tab_telemetry = st.tabs(
    ["Single asset", "Fleet", "Digital Twin", "Telemetry (AeroZip)"]
)


# --------------------------------------------------------------------------- #
# Single asset                                                                #
# --------------------------------------------------------------------------- #
with tab_single:
    st.subheader("Single-asset advisory")
    col_meta, col_out = st.columns([1, 1])

    with col_meta:
        asset_id = st.text_input("Asset ID", value="WTG-001")
        st.markdown("**Telemetry snapshot** (10 s window means)")
        c1, c2 = st.columns(2)
        tel = {
            "vibration_mms": c1.number_input(
                "Vibration (mm/s)", 0.0, 50.0, _DEFAULTS["vibration_mms"], step=0.1
            ),
            "temperature_c": c2.number_input(
                "Temperature (°C)", -40.0, 200.0, _DEFAULTS["temperature_c"], step=0.5
            ),
            "rpm": c1.number_input("HSS RPM", 0.0, 3000.0, _DEFAULTS["rpm"], step=10.0),
            "oil_viscosity_cst": c2.number_input(
                "Oil viscosity (cSt)", 1.0, 500.0, _DEFAULTS["oil_viscosity_cst"], step=0.5
            ),
            "load_pct": c1.number_input(
                "Generator load (% rated)", 0.0, 120.0, _DEFAULTS["load_pct"], step=1.0
            ),
        }
        st.markdown("**BNN state** (pre-computed RUL + uncertainties)")
        use_trained = st.toggle(
            "Use trained model",
            value=False,
            help="Load a trained PG-BNN bundle (artifact registry) and compute "
            "RUL/uncertainties from a telemetry window; otherwise the values "
            "below are used unchanged.",
        )
        model_path = st.text_input(
            "Model bundle path",
            value="artifacts/bnn_demo.pt",
            disabled=not use_trained,
        )
        window_csv = st.file_uploader(
            "Telemetry window CSV (timestamp + 5 channels)",
            type=["csv"],
            disabled=not use_trained,
            help="Window used to build model features. If omitted, the "
            "single snapshot above is used as a degenerate one-sample window.",
        )
        rul = st.number_input(
            "Predicted RUL (days)", 0.0, 3650.0, 120.0, step=1.0, disabled=use_trained
        )
        epi = st.number_input(
            "Epistemic uncertainty σ", 0.0, 100.0, 0.05, step=0.01, disabled=use_trained
        )
        ale = st.number_input(
            "Aleatoric uncertainty σ", 0.0, 100.0, 0.10, step=0.01, disabled=use_trained
        )

        compute = st.button("Compute advisory", type="primary")

    with col_out:
        if compute:
            try:
                payload = TurbinePayload(
                    asset_id=asset_id,
                    telemetry=Telemetry(**tel),
                    bnn_state=BNNState(
                        predicted_rul_days=float(rul),
                        epistemic_uncertainty=float(epi),
                        aleatoric_uncertainty=float(ale),
                    ),
                )
                if use_trained:
                    rec = _model_based_advisory(payload, model_path, window_csv, tel)
                else:
                    rec = run_advisory(payload)
                enforce_safety_contract(rec)
            except Exception as exc:  # noqa: BLE001 - surface validation errors to the user
                st.error(f"Could not compute advisory: {exc}")
            else:
                st.success("Advisory computed.")
                m1, m2, m3 = st.columns(3)
                m1.metric("Predicted RUL", f"{rec['predicted_rul_days']:.1f} days")
                m2.metric(
                    "Inspection window", f"{rec['suggested_inspection_window_days']:.1f} days"
                )
                m3.metric("Physics violations", len(rec["physics_violations"]))
                st.markdown(format_advisory_markdown(rec))
        else:
            st.info("Enter telemetry and BNN state, then click **Compute advisory**.")


# --------------------------------------------------------------------------- #
# Fleet                                                                       #
# --------------------------------------------------------------------------- #
with tab_fleet:
    st.subheader("Fleet advisory")
    st.caption(
        "Upload a CSV with columns: `asset_id`, `vibration_mms`, `temperature_c`, "
        "`rpm`, `oil_viscosity_cst`, `load_pct`, `predicted_rul_days`, "
        "`epistemic_uncertainty`, `aleatoric_uncertainty`."
    )
    uploaded = st.file_uploader("Fleet CSV", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            records = advisories_from_dataframe(df)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not process fleet CSV: {exc}")
        else:
            summary_df = pd.DataFrame(
                [
                    {
                        "asset_id": r["asset_id"],
                        "predicted_rul_days": round(r["predicted_rul_days"], 2),
                        "epistemic_std": round(r["epistemic_std"], 4),
                        "aleatoric_std": round(r["aleatoric_std"], 4),
                        "inspection_window_days": round(r["suggested_inspection_window_days"], 2),
                        "violations": len(r["physics_violations"]),
                    }
                    for r in records
                ]
            )
            mean_rul = summary_df["predicted_rul_days"].mean()
            at_risk = int(
                (summary_df["predicted_rul_days"] < 104.0).sum()
            )  # 90d horizon + 14d buffer

            k1, k2, k3 = st.columns(3)
            k1.metric("Assets assessed", len(records))
            k2.metric("Mean predicted RUL", f"{mean_rul:.1f} days")
            k3.metric("Assets at risk (<104d)", at_risk)

            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            report_md = build_fleet_report(records)
            st.download_button(
                "⬇️ Download fleet report (markdown)",
                data=report_md,
                file_name="fleet_advisory_report.md",
                mime="text/markdown",
            )


# --------------------------------------------------------------------------- #
# Digital Twin                                                                #
# --------------------------------------------------------------------------- #
with tab_twin:
    from src.digital_twin.prompts import generate_engineering_prompt
    from src.digital_twin.specs import SPECS_LIBRARY, get_spec
    from src.digital_twin.twin import WindTurbineDigitalTwin

    st.subheader("Digital twin ↔ advisory bridge")
    st.caption(
        "Each twin update flows through the advisory engine: with a serving "
        "model attached the PG-BNN computes RUL/uncertainties from the twin's "
        "rolling telemetry buffer; otherwise the incoming bnn_state block is "
        "used. Advisory-only, as everywhere else."
    )

    t1, t2 = st.columns(2)
    twin_model_key = t1.selectbox("Turbine spec", list(SPECS_LIBRARY.keys()), index=0)
    twin_asset = t2.text_input("Twin asset ID", value="WTG-TWIN-1")
    twin_model_path = st.text_input(
        "Serving model bundle (optional)",
        value="",
        help="artifacts/bnn_demo.pt or a Hermes export. Attached to the twin "
        "so advisories come from the trained model.",
    )

    if "twins" not in st.session_state:
        st.session_state["twins"] = {}

    def _get_or_create_twin():
        if twin_asset not in st.session_state["twins"]:
            serving = _load_serving(twin_model_path) if twin_model_path else None
            st.session_state["twins"][twin_asset] = WindTurbineDigitalTwin(
                twin_asset, get_spec(twin_model_key), serving_model=serving
            )
        return st.session_state["twins"][twin_asset]

    tc1, tc2 = st.columns(2)
    with tc1:
        st.markdown("**Ingest a snapshot** (defaults from configs/default.yaml)")
        twin_bnn = st.checkbox(
            "Attach demo bnn_state (else model/none)",
            value=True,
            help="Without a serving model the bnn_state path is used.",
        )
        if st.button("Update twin state"):
            try:
                twin = _get_or_create_twin()
                bnn = (
                    BNNState(
                        predicted_rul_days=180.0,
                        epistemic_uncertainty=0.05,
                        aleatoric_uncertainty=0.1,
                    )
                    if twin_bnn
                    else None
                )
                twin.update_state(Telemetry(**_DEFAULTS), bnn)
                rec = twin.state_history[-1]
                st.session_state["twin_last"] = rec
            except Exception as exc:  # noqa: BLE001
                st.error(f"Twin update failed: {exc}")
    with tc2:
        st.markdown("**Simulate a scenario**")
        profile = st.selectbox("Profile", ["nominal", "overload", "derated", "viscosity_loss"])
        hours = st.slider("Duration (hours)", 1, 72, 12)
        if st.button("Run simulation"):
            try:
                twin = _get_or_create_twin()
                records = twin.simulate_scenario(profile=profile, hours=float(hours))
                st.session_state["twin_last"] = records[-1]
                st.session_state["twin_sim_wear"] = twin.cumulative_wear
            except Exception as exc:  # noqa: BLE001
                st.error(f"Simulation failed: {exc}")

    last = st.session_state.get("twin_last")
    if last:
        m1, m2, m3 = st.columns(3)
        m1.metric("Cumulative wear", f"{last['cumulative_wear']:.4f}")
        m2.metric("Bearing L10 life", f"{last['bearing_l10_hours']:.0f} h")
        m3.metric("Violations", len(last["physics_violations"]))
        adv = last.get("advisory")
        if adv:
            st.markdown(f"**Advisory** (source: `{last.get('advisory_source')}`)")
            st.markdown(format_advisory_markdown(adv))
        else:
            st.info("No advisory on this state (attach a serving model or a bnn_state).")
        with st.expander("Reliability-copilot prompt"):
            twin = st.session_state["twins"].get(twin_asset)
            if twin is not None:
                st.code(generate_engineering_prompt(twin), language="text")
    else:
        st.info("Update the twin state or run a simulation to see the bridged advisory.")


# --------------------------------------------------------------------------- #
# Telemetry (AeroZip)                                                         #
# --------------------------------------------------------------------------- #
with tab_telemetry:
    st.subheader("AeroZip telemetry compression")
    st.caption(
        "Delta + deadband + quantize compression with anomaly bypass. "
        "Lossy in normal mode (bounded by quantum + deadband), LOSSLESS "
        "when the anomaly gate bypasses compression (diagnostic fidelity "
        "during faults)."
    )
    aero_csv = st.file_uploader(
        "Telemetry window CSV (columns = the 5 channels, optional timestamp)",
        type=["csv"],
        key="aerozip_csv",
    )
    aero_df = None
    if aero_csv is not None:
        try:
            aero_df = pd.read_csv(aero_csv)
            if "timestamp" in aero_df.columns:
                aero_df = aero_df.drop(columns=["timestamp"])
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read CSV: {exc}")
            aero_df = None
    else:
        n_demo = st.slider("Demo window length (samples)", 60, 600, 120, step=20)
        x = np.arange(n_demo)
        rng_np = np.random.default_rng(0)
        aero_df = pd.DataFrame(
            {
                "vibration_mms": 2.5 + 0.4 * np.sin(x / 9.0) + rng_np.normal(0, 0.05, n_demo),
                "temperature_c": 62.0 + 2.0 * np.sin(x / 15.0) + rng_np.normal(0, 0.2, n_demo),
                "rpm": 1500.0 + 40.0 * np.sin(x / 11.0) + rng_np.normal(0, 4.0, n_demo),
                "oil_viscosity_cst": 32.0 - 1.0 * np.sin(x / 13.0) + rng_np.normal(0, 0.2, n_demo),
                "load_pct": 80.0 + 5.0 * np.sin(x / 8.0) + rng_np.normal(0, 0.8, n_demo),
            }
        )

    if aero_df is not None:
        from src.models.telemetry.pipeline import compress_window, restore_window

        channels = [
            c
            for c in ("vibration_mms", "temperature_c", "rpm", "oil_viscosity_cst", "load_pct")
            if c in aero_df.columns
        ]
        if len(channels) != 5:
            st.error("CSV must contain the 5 canonical channels.")
        else:
            window = {c: aero_df[c].to_numpy() for c in channels}
            comp = compress_window(window)
            rest = restore_window(comp)

            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Compressed size", f"{comp.compressed_bytes} B")
            a2.metric("Raw size", f"{comp.raw_bytes} B")
            a3.metric("Ratio", f"{comp.ratio:.3f}")
            a4.metric(
                "Anomaly score",
                f"{comp.anomaly_score:.3f}",
                help="Above the bypass threshold the window ships lossless raw float64.",
            )
            st.write(f"Anomaly bypass (lossless): **{'yes' if comp.bypass else 'no'}**")

            errs = rest.max_abs_error(window)
            err_df = pd.DataFrame(
                {"channel": list(errs), "max abs error": [round(v, 6) for v in errs.values()]}
            )
            st.dataframe(err_df, use_container_width=True, hide_index=True)
