"""Deprecated Gradio dashboard for AeroVigil.

The rich interactive Gradio dashboard has been superseded by the native
AeroVigilAI browser console served at ``http://localhost:8080/``. This module
builds a minimal Blocks app whose *visible* layout is only a deprecation notice
and a redirect button.

The headless prediction function is still registered (with an ``api_name``) so
external scripts that call the Gradio API route (e.g. ``/api/predict``) keep
working. It is intentionally **not** surfaced in the visual interface.
"""

from __future__ import annotations

from typing import Any

import gradio as gr

DEPRECATED_CSS = """
.aerovigil-deprecated { max-width: 720px; margin: 8vh auto; text-align: center; }
.aerovigil-deprecated .cta a {
    display: inline-block; margin-top: 1.5rem; padding: 0.9rem 1.8rem;
    background: linear-gradient(135deg,#0d9488,#06b6d4); color: #fff;
    font-weight: 600; border-radius: 10px; text-decoration: none;
    box-shadow: 0 8px 24px rgba(13,148,136,.35);
}
"""

NEW_CONSOLE_URL = "http://localhost:8080/"

_NOTICE_MD = f"""
# ⚠️ This dashboard has moved

The legacy AeroVigil Gradio dashboard is **deprecated**.

All operator tooling – digital twin, SCADA monitoring, fleet reports, AeroZip
telemetry, and low-level inference – now lives in the new **AeroVigilAI browser
console**, served from the single canonical deployment on port **8080**.

Please use the new console at: **{NEW_CONSOLE_URL}**

> Existing headless API integrations (e.g. `/api/predict`) continue to work
> unchanged; only the interactive UI has been retired.
"""

_REDIRECT_HTML = f"""
<div class="cta">
  <a href="{NEW_CONSOLE_URL}" target="_top" rel="noopener">Open the AeroVigilAI console →</a>
</div>
"""


def _headless_predict(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Backwards-compatible headless prediction entry point.

    Delegates to the canonical PG-BNN inference implementation so legacy
    Gradio-API callers receive the same results as ``POST /api/model``.
    """
    payload = payload or {}
    try:
        from src.aerovigil_pg_bnn.api import TelemetryInput, _run_inference, _telemetry_to_tensor

        telemetry = TelemetryInput(**payload)
        features = _telemetry_to_tensor(telemetry)
        n_samples = int(payload.get("n_mcmc_samples", 100))
        return _run_inference(features, n_samples)
    except Exception as exc:  # noqa: BLE001 - report cleanly to API callers
        return {"error": str(exc)}


def build_deprecated_interface() -> gr.Blocks:
    """Build the slim deprecation Blocks app (UI notice + hidden headless API)."""
    with gr.Blocks(title="AeroVigil (deprecated dashboard)") as demo:
        with gr.Column(elem_classes=["aerovigil-deprecated"]):
            gr.Markdown(_NOTICE_MD)
            gr.HTML(_REDIRECT_HTML)

        # Hidden, headless-only components backing the retained API route.
        with gr.Row(visible=False):
            _in = gr.JSON(label="payload")
            _out = gr.JSON(label="prediction")
            _btn = gr.Button("predict", visible=False)
            _btn.click(
                _headless_predict,
                inputs=_in,
                outputs=_out,
                api_name="predict",
            )
    return demo
