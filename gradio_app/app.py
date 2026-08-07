"""AeroVigil demo-day Gradio app.

This UI is intentionally presentation-friendly: plain language, scenario presets,
uncertainty visuals, and an offline-first local-weights path. It does not alter
model code, the safety contract, or the advisory-only runtime posture.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import plotly.graph_objects as go
import torch
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aerovigil_pg_bnn import PhysicsGuidedBNN  # noqa: E402

FEATURE_NAMES = [
    "vibration_rms",
    "bearing_temp",
    "generator_temp",
    "power_output",
    "wind_speed",
    "operating_hours",
]

SCENARIOS: dict[str, tuple[float, float, float, float, float, float]] = {
    "Healthy turbine": (12.5, 65.0, 80.0, 2000.0, 9.0, 1000.0),
    "Warning: degradation building": (20.0, 88.0, 115.0, 2100.0, 11.0, 52000.0),
    "Critical: act now": (34.0, 118.0, 150.0, 2400.0, 12.0, 78000.0),
}

RISK_STYLES = {
    "CRITICAL": {"color": "#ff5c70", "bg": "rgba(255,92,112,0.16)", "line": "#ff5c70"},
    "HIGH": {"color": "#ff9b5f", "bg": "rgba(255,155,95,0.16)", "line": "#ff9b5f"},
    "MODERATE": {"color": "#ffd15c", "bg": "rgba(255,209,92,0.16)", "line": "#ffd15c"},
    "LOW": {"color": "#3dd9b4", "bg": "rgba(61,217,180,0.18)", "line": "#3dd9b4"},
}

CPU_THREADS = max(1, min(os.cpu_count() or 1, 4))
torch.set_num_threads(CPU_THREADS)

APP_CSS = """
:root {
  --bg: #0a1628;
  --panel: #10233f;
  --panel-2: #122b4d;
  --card: rgba(16, 35, 63, 0.88);
  --border: rgba(124, 211, 255, 0.18);
  --text: #edf6ff;
  --muted: #9fb6cd;
  --teal: #20d3c2;
  --cyan: #8be9ff;
}
body, .gradio-container {
  background:
    radial-gradient(circle at top right, rgba(32, 211, 194, 0.10), transparent 26%),
    radial-gradient(circle at top left, rgba(139, 233, 255, 0.10), transparent 24%),
    linear-gradient(180deg, #08111f 0%, #0a1628 100%);
  color: var(--text);
}
.gradio-container { max-width: 1240px !important; }
.block, .panel, .gr-box, .gr-form, .gr-group, .gradio-container .contained {
  border-color: var(--border) !important;
}
#app-shell { gap: 18px; }
.hero-card, .info-card, .read-card, .rec-card, .stats-card, .footer-card {
  background: linear-gradient(180deg, rgba(12, 28, 48, 0.94), rgba(12, 24, 44, 0.94));
  border: 1px solid var(--border);
  border-radius: 22px;
  box-shadow: 0 12px 36px rgba(0, 0, 0, 0.24);
}
.hero-card { padding: 20px 24px 12px; margin-bottom: 8px; }
.info-card, .stats-card, .footer-card { padding: 18px 20px; }
.rec-card { padding: 18px 20px; margin-top: 12px; }
.metric-chip-row { display:flex; flex-wrap:wrap; gap:10px; margin:12px 0 8px; }
.metric-chip {
  padding: 8px 12px;
  border-radius: 999px;
  background: rgba(32, 211, 194, 0.10);
  border: 1px solid rgba(32, 211, 194, 0.26);
  color: var(--cyan);
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0.02em;
}
.kicker { color: var(--cyan); text-transform: uppercase; letter-spacing: 0.16em; font-size: 12px; }
.lead { color: var(--muted); font-size: 16px; line-height: 1.6; }
.hero-title { font-size: 34px; font-weight: 800; margin: 8px 0; }
.hero-image img {
  width: 100%; border-radius: 20px; border: 1px solid var(--border); object-fit: cover;
}
.gauge-shell {
  display:flex; flex-direction:column; align-items:center; justify-content:center; gap:10px;
  min-height: 330px; padding: 18px 12px 8px;
}
.gauge-ring {
  width: 220px; height: 220px; border-radius: 50%; position: relative;
  display:flex; align-items:center; justify-content:center;
  box-shadow: inset 0 0 28px rgba(0,0,0,0.35), 0 0 30px rgba(0,0,0,0.18);
}
.gauge-ring::after {
  content:""; width: 162px; height: 162px; border-radius: 50%; background: #091425;
  position:absolute; border: 1px solid rgba(255,255,255,0.05);
}
.gauge-core { position:relative; z-index:1; text-align:center; }
.gauge-number { font-size: 56px; font-weight: 800; line-height: 1; }
.gauge-label { color: var(--muted); font-size: 15px; }
.gauge-sub { color: var(--cyan); font-size: 14px; font-weight: 600; }
.risk-badge {
  display:inline-flex; align-items:center; gap:10px; padding: 10px 14px; border-radius: 999px;
  font-size: 14px; font-weight: 800; letter-spacing: 0.03em; border: 1px solid transparent;
}
.risk-low { background: rgba(61,217,180,0.16); color: #6cf2d0; border-color: rgba(61,217,180,0.34); }
.risk-moderate { background: rgba(255,209,92,0.12); color: #ffe58e; border-color: rgba(255,209,92,0.30); }
.risk-high { background: rgba(255,155,95,0.14); color: #ffbf92; border-color: rgba(255,155,95,0.28); }
.risk-critical { background: rgba(255,92,112,0.14); color: #ff9bab; border-color: rgba(255,92,112,0.28); }
.rec-title { font-size: 18px; font-weight: 800; margin-bottom: 6px; }
.rec-copy { color: var(--muted); line-height: 1.55; }
.stats-card h3 { margin-top: 0; }
.footer-note { color: var(--muted); font-size: 13px; line-height: 1.5; }
.plot-wrap { margin-top: 8px; }
.accordion-note { color: var(--muted); font-size: 15px; line-height: 1.6; }
button.primary, .primary { background: linear-gradient(90deg, #12d5c8, #4db5ff) !important; color: #07111f !important; }
"""


@dataclass(frozen=True)
class LoadedBundle:
    model: PhysicsGuidedBNN
    config: dict[str, Any]
    mean: np.ndarray | None
    std: np.ndarray | None
    source: str
    preprocess: bool


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def load_bundle() -> LoadedBundle:
    env_dir = os.getenv("AEROVIGIL_WEIGHTS_DIR")
    candidates: list[tuple[str, Path]] = []
    if env_dir:
        candidates.append((f"env AEROVIGIL_WEIGHTS_DIR ({env_dir})", Path(env_dir)))
    candidates.append(("local artifacts/pg_bnn_demo", ROOT / "artifacts" / "pg_bnn_demo"))

    for label, folder in candidates:
        weights = folder / "bnn_demo.pt"
        config_path = folder / "config.json"
        scaler_path = folder / "scaler.npz"
        if weights.exists() and config_path.exists():
            config = _load_json(config_path)
            model = PhysicsGuidedBNN(config)
            model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
            model.eval()
            mean = std = None
            preprocess = False
            if scaler_path.exists():
                scaler = np.load(scaler_path)
                mean = scaler["mean"].astype(np.float32)
                std = scaler["std"].astype(np.float32)
                std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
                preprocess = True
            return LoadedBundle(model, config, mean, std, label, preprocess)

    repo_id = "AerovigilAI/wind-turbine-pg-bnn"
    config_path = Path(hf_hub_download(repo_id=repo_id, filename="config.json"))
    weights_path = Path(hf_hub_download(repo_id=repo_id, filename="bnn_demo.pt"))
    config = _load_json(config_path)
    model = PhysicsGuidedBNN(config)
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()
    return LoadedBundle(model, config, None, None, "Hugging Face Hub (raw inputs)", False)


def apply_scenario(name: str) -> tuple[float, float, float, float, float, float]:
    return SCENARIOS[name]


def classify_risk(mean_rul: float) -> tuple[str, str, str]:
    if mean_rul < 14.0:
        return (
            "CRITICAL",
            "Urgent action: stage crane + crew now",
            "Failure risk is inside two weeks. Treat this as a rescue-avoidance window: "
            "lock a crane slot, pre-stage spares, and schedule the intervention immediately.",
        )
    if mean_rul < 30.0:
        return (
            "HIGH",
            "Plan the repair in the next 2–4 weeks",
            "The model still sees time to organize the work, but not to wait. "
            "Confirm parts, watch the trend daily, and reserve your field team.",
        )
    if mean_rul < 45.0:
        return (
            "MODERATE",
            "Use the 45-day planning window",
            "This is your proactive-maintenance zone. Put the turbine on the next maintenance "
            "schedule so you avoid a surprise breakdown.",
        )
    return (
        "LOW",
        "Healthy outlook: keep standard monitoring",
        "Telemetry looks nominal. Keep trend monitoring on the normal cadence and use this as a baseline "
        "for future comparisons.",
    )


def gauge_html(mean_rul: float, std_rul: float, risk: str) -> str:
    color = RISK_STYLES[risk]["color"]
    progress = min(max(mean_rul / 365.0, 0.0), 1.0)
    degrees = max(8.0, progress * 360.0)
    return f"""
    <div class="info-card gauge-shell">
      <div class="gauge-ring" style="background: conic-gradient({color} 0deg {degrees:.1f}deg, rgba(255,255,255,0.08) {degrees:.1f}deg 360deg);">
        <div class="gauge-core">
          <div class="gauge-number">{mean_rul:.0f}</div>
          <div class="gauge-label">days of healthy life</div>
          <div class="gauge-sub">±{std_rul:.1f} days uncertainty</div>
        </div>
      </div>
      <div class="gauge-label">AeroVigil estimates how much runway remains before the turbine enters the caution window.</div>
    </div>
    """


def badge_html(risk: str) -> str:
    css_class = {
        "LOW": "risk-low",
        "MODERATE": "risk-moderate",
        "HIGH": "risk-high",
        "CRITICAL": "risk-critical",
    }[risk]
    labels = {
        "LOW": "LOW RISK · healthy trend",
        "MODERATE": "MODERATE RISK · inside 45-day plan window",
        "HIGH": "HIGH RISK · intervention soon",
        "CRITICAL": "CRITICAL RISK · act now",
    }
    return f'<div class="risk-badge {css_class}">{labels[risk]}</div>'


def recommendation_html(risk: str, title: str, body: str) -> str:
    style = RISK_STYLES[risk]
    return f"""
    <div class="rec-card" style="border-color:{style["line"]}55; box-shadow: inset 0 0 0 1px {style["line"]}22;">
      <div class="rec-title" style="color:{style["color"]};">{title}</div>
      <div class="rec-copy">{body}</div>
    </div>
    """


def make_histogram(
    predictions: np.ndarray, mean_rul: float, ci_lower: float, ci_upper: float, risk: str
) -> go.Figure:
    style = RISK_STYLES[risk]
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=predictions,
            nbinsx=28,
            marker={"color": style["color"], "line": {"color": "#0c1627", "width": 1.2}},
            opacity=0.88,
            hovertemplate="%{x:.1f} days<extra></extra>",
        )
    )
    fig.add_vline(
        x=mean_rul,
        line_color="#8be9ff",
        line_width=3,
        annotation_text=f"mean {mean_rul:.1f} d",
        annotation_position="top right",
    )
    fig.add_vline(
        x=ci_lower,
        line_color="#cfd8e3",
        line_width=2,
        line_dash="dash",
        annotation_text="95% CI",
        annotation_position="top left",
    )
    fig.add_vline(x=ci_upper, line_color="#cfd8e3", line_width=2, line_dash="dash")
    fig.add_vline(
        x=45.0,
        line_color="#ffd15c",
        line_width=3,
        line_dash="dot",
        annotation_text="45-day line",
        annotation_position="bottom right",
    )
    fig.add_vline(
        x=14.0,
        line_color="#ff5c70",
        line_width=3,
        line_dash="dot",
        annotation_text="critical line",
        annotation_position="bottom left",
    )
    fig.update_layout(
        template="plotly_dark",
        height=360,
        margin={"l": 18, "r": 18, "t": 44, "b": 18},
        paper_bgcolor="#0d1b31",
        plot_bgcolor="#0d1b31",
        title="Distribution from 100 stochastic runs",
        xaxis_title="Predicted healthy life remaining (days)",
        yaxis_title="How often that value appeared",
        bargap=0.05,
        showlegend=False,
        font={"color": "#edf6ff"},
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    return fig


def preprocess_input(raw_values: list[float], bundle: LoadedBundle) -> np.ndarray:
    data = np.array(raw_values, dtype=np.float32)
    if bundle.preprocess and bundle.mean is not None and bundle.std is not None:
        data = (data - bundle.mean) / bundle.std
    return data.astype(np.float32)


def run_inference(bundle: LoadedBundle, x: np.ndarray, n_samples: int) -> np.ndarray:
    tensor = torch.tensor([x.tolist()], dtype=torch.float32)
    predictions: list[float] = []
    bundle.model.train()  # keep stochasticity enabled for MCVI
    with torch.no_grad():
        for _ in range(int(n_samples)):
            mean, _ = bundle.model(tensor)
            predictions.append(float(mean.squeeze().item()))
    return np.array(predictions, dtype=np.float32)


def predict_rul(
    vibration_rms: float,
    bearing_temp: float,
    generator_temp: float,
    power_output: float,
    wind_speed: float,
    operating_hours: float,
    n_samples: int,
) -> tuple[str, str, str, str, go.Figure, dict[str, Any]]:
    started = time.perf_counter()
    bundle = load_bundle()
    raw_values = [
        float(vibration_rms),
        float(bearing_temp),
        float(generator_temp),
        float(power_output),
        float(wind_speed),
        float(operating_hours),
    ]
    model_input = preprocess_input(raw_values, bundle)
    predictions = run_inference(bundle, model_input, int(n_samples))

    mean_rul = float(np.mean(predictions))
    std_rul = float(np.std(predictions))
    ci_lower = float(np.percentile(predictions, 2.5))
    ci_upper = float(np.percentile(predictions, 97.5))
    risk, rec_title, rec_copy = classify_risk(mean_rul)
    latency_ms = (time.perf_counter() - started) * 1000.0

    stats = (
        "### Prediction stats\n"
        f"- **95% interval:** {ci_lower:.1f} to {ci_upper:.1f} days\n"
        f"- **Uncertainty (σ):** {std_rul:.2f} days\n"
        f"- **Monte Carlo runs:** {int(n_samples)}\n"
        f"- **Weights source:** {bundle.source}\n"
        f"- **CPU latency:** {latency_ms:.0f} ms on {CPU_THREADS} thread(s)\n"
    )
    payload = {
        "mean_rul_days": round(mean_rul, 2),
        "uncertainty_days": round(std_rul, 2),
        "ci_95": [round(ci_lower, 2), round(ci_upper, 2)],
        "risk_level": risk,
        "recommendation_title": rec_title,
        "recommendation": rec_copy,
        "weights_source": bundle.source,
        "mc_samples": int(n_samples),
        "latency_ms": round(latency_ms, 1),
        "preprocessed": bundle.preprocess,
        "raw_input": dict(zip(FEATURE_NAMES, raw_values)),
    }

    return (
        gauge_html(mean_rul, std_rul, risk),
        badge_html(risk),
        recommendation_html(risk, rec_title, rec_copy),
        stats,
        make_histogram(predictions, mean_rul, ci_lower, ci_upper, risk),
        payload,
    )


def create_header_html() -> str:
    hero_path = ROOT / "docs" / "assets" / "social-card.png"
    hero_image = ""
    if hero_path.exists():
        hero_image = (
            f'<div class="hero-image"><img alt="AeroVigil social card" '
            f'src="/gradio_api/file={hero_path.resolve()}" /></div>'
        )
    return f"""
    <div class="hero-card">
      <div class="kicker">AeroVigil demo-day build</div>
      <div class="hero-title">AeroVigil — wind turbine health, predicted 45 days out</div>
      <div class="lead">
        A plain-language, advisory-only view of a physics-guided Bayesian model.
        Pick a scenario, adjust the turbine's vital signs, and see both the
        predicted runway and the model's honesty about uncertainty.
      </div>
      <div class="metric-chip-row">
        <div class="metric-chip">physics-guided</div>
        <div class="metric-chip">uncertainty shown</div>
        <div class="metric-chip">advisory-only</div>
      </div>
      {hero_image}
    </div>
    """


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="AeroVigil — wind turbine health, predicted 45 days out") as demo:
        gr.HTML(create_header_html())

        with gr.Row(equal_height=False, elem_id="app-shell"):
            with gr.Column(scale=5, min_width=360):
                gr.Markdown(
                    """
                    ### Set the turbine state
                    Use the presets for the stage demo, or fine-tune the six SCADA signals manually.
                    """,
                    elem_classes=["info-card"],
                )
                scenario = gr.Dropdown(
                    choices=list(SCENARIOS.keys()),
                    value="Healthy turbine",
                    label="Scenario preset",
                    info="Pick a canonical demo case and the sliders will update automatically.",
                )
                vibration_rms = gr.Slider(
                    0.0, 40.0, value=12.5, step=0.1, label="🫨 How much is it shaking? (mm/s)"
                )
                bearing_temp = gr.Slider(
                    30.0, 130.0, value=65.0, step=0.5, label="🌡️ Main bearing temperature (°C)"
                )
                generator_temp = gr.Slider(
                    40.0, 170.0, value=80.0, step=0.5, label="🔥 Generator temperature (°C)"
                )
                power_output = gr.Slider(
                    0.0, 3000.0, value=2000.0, step=10.0, label="⚡ Power output (kW)"
                )
                wind_speed = gr.Slider(0.0, 20.0, value=9.0, step=0.1, label="💨 Wind speed (m/s)")
                operating_hours = gr.Slider(
                    0.0, 90000.0, value=1000.0, step=100.0, label="⏱️ Lifetime run hours"
                )
                n_samples = gr.Slider(20, 200, value=100, step=10, label="Monte Carlo samples")
                predict_button = gr.Button(
                    "Predict the 45-day outlook", variant="primary", size="lg"
                )

            with gr.Column(scale=6, min_width=420):
                gauge = gr.HTML(value=gauge_html(280.0, 5.1, "LOW"))
                risk_badge = gr.HTML(value=badge_html("LOW"))
                recommendation = gr.HTML(
                    value=recommendation_html(
                        "LOW",
                        "Healthy outlook: keep standard monitoring",
                        "Telemetry looks nominal. Press Predict to run 100 stochastic passes and see the true uncertainty spread.",
                    )
                )
                stats = gr.Markdown(
                    "### Prediction stats\n"
                    "- **95% interval:** run a prediction\n"
                    "- **Uncertainty (σ):** run a prediction\n"
                    "- **Monte Carlo runs:** 100\n"
                    "- **Weights source:** waiting for load\n"
                    "- **CPU latency:** —\n",
                    elem_classes=["stats-card"],
                )
                plot = gr.Plot(label="Prediction distribution")
                api_payload = gr.JSON(value={}, visible=False)
                with gr.Accordion("🧑‍🏫 How do I read this?", open=False):
                    gr.Markdown(
                        """
                        - **Big number:** the average healthy-life runway left, in days.
                        - **Spread / histogram width:** how much the stochastic model disagrees with itself. Wider = less certain.
                        - **Yellow 45-day line:** left of this line means you are in the proactive-maintenance window.
                        - **Red 14-day line:** left of this line means the issue is now urgent.
                        - **AeroVigil advises only:** it helps a human schedule maintenance; it never controls the turbine.
                        """,
                        elem_classes=["accordion-note"],
                    )
                gr.HTML(
                    """
                    <div class="footer-card footer-note">
                      Advisory only. This demo shows decision-support output for maintenance planning.
                      It does not send turbine control commands, work instructions, or actuation signals.
                      Read more in <a href="https://github.com/rajaram-2005/wind-turbine-pg-bnn/blob/main/docs/SAFETY.md" target="_blank">docs/SAFETY.md</a>
                      and the <a href="https://github.com/rajaram-2005/wind-turbine-pg-bnn/blob/main/docs/DEMO_RUNBOOK.md" target="_blank">demo runbook</a>.
                    </div>
                    """
                )

        scenario.change(
            fn=apply_scenario,
            inputs=scenario,
            outputs=[
                vibration_rms,
                bearing_temp,
                generator_temp,
                power_output,
                wind_speed,
                operating_hours,
            ],
        )

        predict_button.click(
            fn=predict_rul,
            inputs=[
                vibration_rms,
                bearing_temp,
                generator_temp,
                power_output,
                wind_speed,
                operating_hours,
                n_samples,
            ],
            outputs=[gauge, risk_badge, recommendation, stats, plot, api_payload],
            api_name="predict_rul",
        )

    return demo


if __name__ == "__main__":
    theme = gr.themes.Soft(primary_hue="cyan", secondary_hue="slate", neutral_hue="slate")
    demo = build_interface()
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=theme,
        css=APP_CSS,
        allowed_paths=[str(ROOT)],
        strict_cors=False,
        show_error=True,
    )
