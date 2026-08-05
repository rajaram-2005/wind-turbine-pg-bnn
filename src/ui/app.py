"""Streamlit advisory UI for AeroVigil (wind-turbine-pg-bnn engine).

AeroVigil v1.0.0 — https://aerovigil.abacusai.app

Run::

    streamlit run src/ui/app.py

Decision-support only — the UI deliberately exposes no actuation controls
(no throttle, pitch, RPM-setpoint, breaker, or LOTO widgets). Every computed
recommendation is screened by ``enforce_safety_contract`` before display.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.models.predictor import run_advisory
from src.ui.defaults import default_snapshot
from src.reporting.reports import (
    advisories_from_dataframe,
    build_fleet_report,
    format_advisory_markdown,
)
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


st.set_page_config(page_title="AeroVigil advisory", page_icon="🌀", layout="wide")
st.warning(SAFETY_BANNER)
st.title("🌀 AeroVigil — RUL advisory")
st.caption(
    "**AeroVigil v1.0.0** — Physics-Guided Bayesian Neural Network for drivetrain "
    "remaining-useful-life prediction · [aerovigil.abacusai.app](https://aerovigil.abacusai.app)"
)

tab_single, tab_fleet = st.tabs(["Single asset", "Fleet"])


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
        rul = st.number_input("Predicted RUL (days)", 0.0, 3650.0, 120.0, step=1.0)
        epi = st.number_input("Epistemic uncertainty σ", 0.0, 100.0, 0.05, step=0.01)
        ale = st.number_input("Aleatoric uncertainty σ", 0.0, 100.0, 0.10, step=0.01)

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
                rec = run_advisory(payload)
                enforce_safety_contract(rec)
            except Exception as exc:  # noqa: BLE001 - surface validation errors to the user
                st.error(f"Could not compute advisory: {exc}")
            else:
                st.success("Advisory computed.")
                m1, m2, m3 = st.columns(3)
                m1.metric("Predicted RUL", f"{rec['predicted_rul_days']:.1f} days")
                m2.metric("Inspection window", f"{rec['suggested_inspection_window_days']:.1f} days")
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
                        "inspection_window_days": round(
                            r["suggested_inspection_window_days"], 2
                        ),
                        "violations": len(r["physics_violations"]),
                    }
                    for r in records
                ]
            )
            mean_rul = summary_df["predicted_rul_days"].mean()
            at_risk = int((summary_df["predicted_rul_days"] < 104.0).sum())  # 90d horizon + 14d buffer

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
