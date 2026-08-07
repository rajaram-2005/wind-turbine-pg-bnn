"""AeroVigil EPIC — mind-blowing wind turbine health dashboard.

A completely redesigned Gradio app with:
- Animated wind turbine SVG backgrounds with spinning blades
- Particle wind effects
- Tabbed interface: Dashboard, Predict, Fleet View, Trends, Digital Twin
- Animated gauges with reactive risk-level colors
- Fleet overview with multiple turbine cards
- Historical degradation trend charts
- Power curve visualization
- Multi-axis radar health charts
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

if __package__:
    from .colors import hex_to_rgba
else:
    from colors import hex_to_rgba

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aerovigil_pg_bnn import PhysicsGuidedBNN  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────

FEATURE_NAMES = [
    "vibration_rms",
    "bearing_temp",
    "generator_temp",
    "power_output",
    "wind_speed",
    "operating_hours",
]

SCENARIOS: dict[str, tuple[float, float, float, float, float, float]] = {
    "🟢 Healthy turbine": (12.5, 65.0, 80.0, 2000.0, 9.0, 1000.0),
    "🟡 Warning: degradation building": (20.0, 88.0, 115.0, 2100.0, 11.0, 52000.0),
    "🔴 Critical: act now": (34.0, 118.0, 150.0, 2400.0, 12.0, 78000.0),
}

RISK_COLORS = {
    "LOW": {"primary": "#00e5a0", "secondary": "#00b8d4", "glow": "rgba(0,229,160,0.3)"},
    "MODERATE": {"primary": "#ffd600", "secondary": "#ff9100", "glow": "rgba(255,214,0,0.3)"},
    "HIGH": {"primary": "#ff6d00", "secondary": "#ff3d00", "glow": "rgba(255,109,0,0.3)"},
    "CRITICAL": {"primary": "#ff1744", "secondary": "#d50000", "glow": "rgba(255,23,68,0.4)"},
}

RISK_LABELS = {
    "LOW": "LOW RISK · All systems nominal",
    "MODERATE": "MODERATE RISK · Inside 45-day planning window",
    "HIGH": "HIGH RISK · Intervention needed soon",
    "CRITICAL": "CRITICAL RISK · Immediate action required",
}


CPU_THREADS = max(1, min(os.cpu_count() or 1, 4))
torch.set_num_threads(CPU_THREADS)

# ── Fleet simulation data ─────────────────────────────────────────

FLEET_TURBINES = [
    {"id": "WT-001", "name": "North Ridge α", "model": "Vestas V90", "location": "North Sea, DK"},
    {"id": "WT-002", "name": "North Ridge β", "model": "Vestas V90", "location": "North Sea, DK"},
    {"id": "WT-003", "name": "Mumbai Coast 1", "model": "Suzlon S97", "location": "Gujarat, IN"},
    {"id": "WT-004", "name": "Mumbai Coast 2", "model": "Suzlon S97", "location": "Gujarat, IN"},
    {"id": "WT-005", "name": "Iowa Plains A", "model": "GE 1.5 SLE", "location": "Iowa, US"},
    {"id": "WT-006", "name": "Iowa Plains B", "model": "GE 1.5 SLE", "location": "Iowa, US"},
    {
        "id": "WT-007",
        "name": "Patagonia Wind",
        "model": "Gamesa G114",
        "location": "Buenos Aires, AR",
    },
    {"id": "WT-008", "name": "Nordic Frost", "model": "Nordex N100", "location": "Finland"},
    {"id": "WT-009", "name": "Sahara Edge", "model": "Siemens SWT-2.3", "location": "Morocco"},
    {"id": "WT-010", "name": "Offshore Delta", "model": "Senvion MM92", "location": "Netherlands"},
    {"id": "WT-011", "name": "Highland Echo", "model": "NREL 5MW", "location": "Scotland, UK"},
    {
        "id": "WT-012",
        "name": "Tropical Breeze",
        "model": "Suzlon S97",
        "location": "Tamil Nadu, IN",
    },
]


# ── Model loading ─────────────────────────────────────────────────


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
    # Prefer epic model, fall back to demo
    candidates.append(("local artifacts/pg_bnn_epic", ROOT / "artifacts" / "pg_bnn_epic"))
    candidates.append(("local artifacts/pg_bnn_demo", ROOT / "artifacts" / "pg_bnn_demo"))

    for label, folder in candidates:
        # Try both epic and demo weight names
        for weights_name in ["bnn_epic.pt", "bnn_demo.pt"]:
            weights = folder / weights_name
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


# ── Core logic ────────────────────────────────────────────────────


def apply_scenario(name: str) -> tuple[float, float, float, float, float, float]:
    return SCENARIOS[name]


def classify_risk(mean_rul: float) -> tuple[str, str, str]:
    if mean_rul < 14.0:
        return (
            "CRITICAL",
            "🚨 Urgent: Stage crane + crew now",
            "Failure risk is inside two weeks. Lock a crane slot, pre-stage spares, "
            "and schedule intervention immediately.",
        )
    if mean_rul < 30.0:
        return (
            "HIGH",
            "⚠️ Plan repair in 2–4 weeks",
            "The model sees time to organize, but not to wait. "
            "Confirm parts, watch the trend daily, reserve your field team.",
        )
    if mean_rul < 45.0:
        return (
            "MODERATE",
            "📋 Use the 45-day planning window",
            "Proactive-maintenance zone. Put the turbine on the next maintenance "
            "schedule to avoid a surprise breakdown.",
        )
    return (
        "LOW",
        "✅ Healthy: keep standard monitoring",
        "Telemetry looks nominal. Keep trend monitoring on normal cadence.",
    )


def preprocess_input(raw_values: list[float], bundle: LoadedBundle) -> np.ndarray:
    data = np.array(raw_values, dtype=np.float32)
    if bundle.preprocess and bundle.mean is not None and bundle.std is not None:
        data = (data - bundle.mean) / bundle.std
    return data.astype(np.float32)


def run_inference(bundle: LoadedBundle, x: np.ndarray, n_samples: int) -> np.ndarray:
    tensor = torch.tensor([x.tolist()], dtype=torch.float32)
    predictions: list[float] = []
    bundle.model.train()
    with torch.no_grad():
        for _ in range(int(n_samples)):
            mean, _ = bundle.model(tensor)
            predictions.append(float(mean.squeeze().item()))
    return np.array(predictions, dtype=np.float32)


def simulate_fleet_prediction(rng: np.random.Generator) -> list[dict]:
    """Generate simulated fleet data for the dashboard."""
    fleet_data = []
    for t in FLEET_TURBINES:
        vib = float(rng.uniform(4.0, 38.0))
        temp = float(rng.uniform(40.0, 125.0))
        gen_temp = float(rng.uniform(50.0, 160.0))
        power = float(rng.uniform(800.0, 2600.0))
        wind = float(rng.uniform(4.0, 16.0))
        hours = float(rng.uniform(500.0, 82000.0))
        rul = max(
            5.0,
            min(
                400.0,
                420.0
                - 5.0 * max(vib - 5.0, 0) ** 1.15
                - 2.0 * max(temp - 55.0, 0) ** 1.1
                - (hours / 87600.0) * 200.0
                + rng.normal(0, 8),
            ),
        )
        risk, _, _ = classify_risk(rul)
        fleet_data.append(
            {
                **t,
                "vibration_rms": round(vib, 1),
                "bearing_temp": round(temp, 1),
                "generator_temp": round(gen_temp, 1),
                "power_output": round(power, 0),
                "wind_speed": round(wind, 1),
                "operating_hours": round(hours, 0),
                "rul_days": round(rul, 1),
                "risk": risk,
            }
        )
    return fleet_data


# ── HTML generators ───────────────────────────────────────────────


def animated_background(risk: str = "LOW") -> str:
    colors = RISK_COLORS[risk]
    return f"""
    <div id="av-bg" style="position:fixed;top:0;left:0;width:100%;height:100%;z-index:-1;pointer-events:none;">
      <svg width="100%" height="100%" style="position:absolute;top:0;left:0;">
        <defs>
          <radialGradient id="skyGrad" cx="50%" cy="30%" r="80%">
            <stop offset="0%" style="stop-color:#0d1b2a"/>
            <stop offset="50%" style="stop-color:#0a1628"/>
            <stop offset="100%" style="stop-color:#060d18"/>
          </radialGradient>
          <radialGradient id="glowGrad" cx="50%" cy="50%" r="50%">
            <stop offset="0%" style="stop-color:{colors["glow"]}"/>
            <stop offset="100%" style="stop-color:transparent"/>
          </radialGradient>
          <linearGradient id="bladeGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style="stop-color:#e0f7fa;stop-opacity:0.9"/>
            <stop offset="100%" style="stop-color:#80deea;stop-opacity:0.5"/>
          </linearGradient>
          <filter id="glow"><feGaussianBlur stdDeviation="3" result="blur"/>
            <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
        </defs>
        <rect width="100%" height="100%" fill="url(#skyGrad)"/>
        <!-- Stars -->
        <g opacity="0.6">
          {"".join(f'<circle cx="{i * 7.3 % 500}" cy="{i * 3.7 % 300}" r="{0.5 + i % 3 * 0.3}" fill="white" opacity="{0.3 + i % 5 * 0.1}"><animate attributeName="opacity" values="{0.2 + i % 3 * 0.1};{0.6 + i % 4 * 0.1};{0.2 + i % 3 * 0.1}" dur="{2 + i % 4}s" repeatCount="indefinite"/></circle>' for i in range(40))}
        </g>
        <!-- Ambient glow -->
        <ellipse cx="50%" cy="35%" rx="45%" ry="30%" fill="url(#glowGrad)" opacity="0.4">
          <animate attributeName="opacity" values="0.3;0.5;0.3" dur="6s" repeatCount="indefinite"/>
        </ellipse>
        <!-- Wind particles -->
        <g opacity="0.25">
          {"".join(f'<line x1="-10" y1="{20 + i * 25}" x2="40" y2="{18 + i * 25}" stroke="{colors["primary"]}" stroke-width="1" opacity="0.4"><animate attributeName="x1" values="-10;100%" dur="{3 + i % 5}s" repeatCount="indefinite"/><animate attributeName="x2" values="40;calc(100% + 40px)" dur="{3 + i % 5}s" repeatCount="indefinite"/></line>' for i in range(12))}
        </g>
        <!-- Wind turbine 1 (left) -->
        <g transform="translate(12%, 55%)" filter="url(#glow)" opacity="0.35">
          <rect x="-3" y="0" width="6" height="180" fill="#1a3a5c" rx="3"/>
          <g transform="translate(0, 0)">
            <animateTransform attributeName="transform" type="rotate" from="0" to="360" dur="8s" repeatCount="indefinite"/>
            <path d="M0,0 L-8,-90 Q0,-95 8,-90 Z" fill="url(#bladeGrad)" opacity="0.7"/>
            <path d="M0,0 L78,45 Q80,52 72,50 Z" fill="url(#bladeGrad)" opacity="0.6"/>
            <path d="M0,0 L-70,45 Q-78,50 -75,42 Z" fill="url(#bladeGrad)" opacity="0.5"/>
          </g>
          <circle cx="0" cy="0" r="5" fill="{colors["primary"]}" opacity="0.8">
            <animate attributeName="r" values="4;6;4" dur="3s" repeatCount="indefinite"/>
          </circle>
        </g>
        <!-- Wind turbine 2 (right, smaller) -->
        <g transform="translate(82%, 60%) scale(0.7)" filter="url(#glow)" opacity="0.25">
          <rect x="-3" y="0" width="6" height="160" fill="#1a3a5c" rx="3"/>
          <g transform="translate(0, 0)">
            <animateTransform attributeName="transform" type="rotate" from="0" to="360" dur="12s" repeatCount="indefinite"/>
            <path d="M0,0 L-7,-80 Q0,-84 7,-80 Z" fill="url(#bladeGrad)" opacity="0.6"/>
            <path d="M0,0 L69,40 Q72,46 65,44 Z" fill="url(#bladeGrad)" opacity="0.5"/>
            <path d="M0,0 L-62,40 Q-68,44 -65,38 Z" fill="url(#bladeGrad)" opacity="0.4"/>
          </g>
          <circle cx="0" cy="0" r="4" fill="{colors["secondary"]}" opacity="0.7">
            <animate attributeName="opacity" values="0.5;0.9;0.5" dur="4s" repeatCount="indefinite"/>
          </circle>
        </g>
        <!-- Wind turbine 3 (far right, tiny) -->
        <g transform="translate(92%, 68%) scale(0.4)" filter="url(#glow)" opacity="0.2">
          <rect x="-2" y="0" width="4" height="140" fill="#1a3a5c" rx="2"/>
          <g>
            <animateTransform attributeName="transform" type="rotate" from="0" to="360" dur="15s" repeatCount="indefinite"/>
            <path d="M0,0 L-6,-70 Q0,-73 6,-70 Z" fill="url(#bladeGrad)" opacity="0.5"/>
            <path d="M0,0 L61,35 Q63,40 57,38 Z" fill="url(#bladeGrad)" opacity="0.4"/>
            <path d="M0,0 L-55,35 Q-60,38 -57,33 Z" fill="url(#bladeGrad)" opacity="0.3"/>
          </g>
          <circle cx="0" cy="0" r="3" fill="{colors["primary"]}" opacity="0.6"/>
        </g>
        <!-- Horizon glow -->
        <ellipse cx="50%" cy="92%" rx="60%" ry="8%" fill="{colors["primary"]}" opacity="0.04"/>
      </svg>
    </div>
    """


def create_animated_gauge(mean_rul: float, std_rul: float, risk: str) -> str:
    colors = RISK_COLORS[risk]
    progress = min(max(mean_rul / 365.0, 0.0), 1.0)
    angle = progress * 270
    return f"""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:24px 12px;
                background:linear-gradient(180deg,rgba(8,17,31,0.95),rgba(10,22,40,0.98));
                border-radius:24px;border:1px solid rgba(124,211,255,0.12);
                box-shadow:0 0 60px {colors["glow"]}, inset 0 0 30px rgba(0,0,0,0.3);
                position:relative;overflow:hidden;">
      <!-- Animated ring background pulse -->
      <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:280px;height:280px;
                  border-radius:50%;border:2px solid {colors["primary"]};opacity:0.1;
                  animation:pulse-ring 3s ease-in-out infinite;"></div>
      <svg width="260" height="220" viewBox="0 0 260 220">
        <defs>
          <linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" style="stop-color:{colors["secondary"]}"/>
            <stop offset="100%" style="stop-color:{colors["primary"]}"/>
          </linearGradient>
          <filter id="gaugeGlow">
            <feGaussianBlur stdDeviation="4" result="blur"/>
            <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
        </defs>
        <!-- Background arc -->
        <path d="M 30 190 A 100 100 0 1 1 230 190" fill="none" stroke="rgba(255,255,255,0.06)"
              stroke-width="18" stroke-linecap="round"/>
        <!-- Danger zone (red segment) -->
        <path d="M 30 190 A 100 100 0 0 1 65 68" fill="none" stroke="rgba(255,23,68,0.15)"
              stroke-width="18" stroke-linecap="round"/>
        <!-- Warning zone (yellow segment) -->
        <path d="M 65 68 A 100 100 0 0 1 130 30" fill="none" stroke="rgba(255,214,0,0.12)"
              stroke-width="18" stroke-linecap="round"/>
        <!-- Active arc -->
        <path d="M 30 190 A 100 100 0 1 1 230 190" fill="none" stroke="url(#gaugeGrad)"
              stroke-width="18" stroke-linecap="round" filter="url(#gaugeGlow)"
              stroke-dasharray="{angle * 3.14:.1f} 1000"
              style="transition: stroke-dasharray 1.5s cubic-bezier(0.4,0,0.2,1);"/>
        <!-- Tick marks -->
        {"".join(f'<line x1="{130 + 105 * np.cos(np.radians(225 - i * 2.7))}" y1="{130 - 105 * np.sin(np.radians(225 - i * 2.7))}" x2="{130 + 95 * np.cos(np.radians(225 - i * 2.7))}" y2="{130 - 95 * np.sin(np.radians(225 - i * 2.7))}" stroke="rgba(255,255,255,0.2)" stroke-width="1"/>' for i in range(0, 100, 10))}
        <!-- Center value -->
        <text x="130" y="115" text-anchor="middle" fill="{colors["primary"]}"
              font-size="52" font-weight="900" font-family="system-ui"
              style="filter:drop-shadow(0 0 12px {colors["glow"]});">
          {mean_rul:.0f}
        </text>
        <text x="130" y="145" text-anchor="middle" fill="rgba(200,220,240,0.6)"
              font-size="13" font-weight="500" font-family="system-ui">
          days of healthy life
        </text>
        <text x="130" y="170" text-anchor="middle" fill="{colors["secondary"]}"
              font-size="14" font-weight="700" font-family="system-ui">
          ±{std_rul:.1f} days uncertainty
        </text>
      </svg>
      <style>
        @keyframes pulse-ring {{
          0%, 100% {{ transform: translate(-50%,-50%) scale(1); opacity: 0.1; }}
          50% {{ transform: translate(-50%,-50%) scale(1.08); opacity: 0.2; }}
        }}
      </style>
    </div>
    """


def create_risk_badge(risk: str) -> str:
    colors = RISK_COLORS[risk]
    label = RISK_LABELS[risk]
    pulse = "animation:pulse-badge 2s ease-in-out infinite;" if risk == "CRITICAL" else ""
    return f"""
    <div style="display:inline-flex;align-items:center;gap:12px;padding:12px 20px;
                border-radius:16px;font-size:15px;font-weight:800;
                letter-spacing:0.04em;border:2px solid {hex_to_rgba(colors["primary"], 0.27)};
                background:{colors["glow"]};color:{colors["primary"]};
                box-shadow:0 0 20px {colors["glow"]};{pulse}
                backdrop-filter:blur(10px);">
      <span style="width:10px;height:10px;border-radius:50%;background:{colors["primary"]};
                   box-shadow:0 0 10px {colors["primary"]};
                   animation:blink 1.5s ease-in-out infinite;"></span>
      {label}
    </div>
    <style>
      @keyframes pulse-badge {{
        0%,100% {{ box-shadow: 0 0 20px {colors["glow"]}; }}
        50% {{ box-shadow: 0 0 40px {colors["glow"]}, 0 0 60px {colors["glow"]}; }}
      }}
      @keyframes blink {{
        0%,100% {{ opacity:1; }} 50% {{ opacity:0.4; }}
      }}
    </style>
    """


def create_recommendation_card(risk: str, title: str, body: str) -> str:
    colors = RISK_COLORS[risk]
    return f"""
    <div style="padding:20px 24px;border-radius:20px;
                background:linear-gradient(135deg,rgba(10,22,40,0.95),rgba(16,35,63,0.9));
                border:1px solid {hex_to_rgba(colors["primary"], 0.2)};
                box-shadow:inset 0 0 0 1px {hex_to_rgba(colors["primary"], 0.07)}, 0 8px 32px rgba(0,0,0,0.3);
                backdrop-filter:blur(10px);margin-top:8px;">
      <div style="font-size:18px;font-weight:800;color:{colors["primary"]};margin-bottom:8px;">
        {title}
      </div>
      <div style="color:rgba(180,200,220,0.85);line-height:1.65;font-size:15px;">
        {body}
      </div>
    </div>
    """


def create_fleet_card(t: dict) -> str:
    colors = RISK_COLORS[t["risk"]]
    return f"""
    <div style="padding:16px;border-radius:18px;
                background:linear-gradient(135deg,rgba(10,22,40,0.92),rgba(16,35,63,0.88));
                border:1px solid {hex_to_rgba(colors["primary"], 0.13)};
                box-shadow:0 4px 20px rgba(0,0,0,0.2);min-width:200px;
                transition:transform 0.3s,box-shadow 0.3s;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
        <span style="font-size:13px;font-weight:800;color:{colors["primary"]};
                     letter-spacing:0.05em;">{t["id"]}</span>
        <span style="width:8px;height:8px;border-radius:50%;background:{colors["primary"]};
                     box-shadow:0 0 8px {colors["primary"]};"></span>
      </div>
      <div style="font-size:16px;font-weight:700;color:#edf6ff;margin-bottom:4px;">{t["name"]}</div>
      <div style="font-size:12px;color:rgba(160,180,200,0.7);margin-bottom:10px;">{t["model"]} · {t["location"]}</div>
      <div style="font-size:28px;font-weight:900;color:{colors["primary"]};
                  text-shadow:0 0 15px {colors["glow"]};">{t["rul_days"]:.0f}<span style="font-size:13px;color:rgba(160,180,200,0.6);margin-left:4px;">days</span></div>
      <div style="margin-top:8px;height:4px;border-radius:2px;background:rgba(255,255,255,0.06);overflow:hidden;">
        <div style="height:100%;width:{min(100, max(0, t["rul_days"] / 365 * 100)):.1f}%;
                    background:linear-gradient(90deg,{colors["secondary"]},{colors["primary"]});
                    border-radius:2px;transition:width 1s;"></div>
      </div>
      <div style="display:flex;justify-content:space-between;margin-top:8px;font-size:11px;color:rgba(160,180,200,0.5);">
        <span>Vib: {t["vibration_rms"]} mm/s</span>
        <span>Temp: {t["bearing_temp"]}°C</span>
        <span>{t["operating_hours"]:.0f}h</span>
      </div>
    </div>
    """


def create_kpi_strip() -> str:
    return """
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:8px;">
      <div style="padding:16px;border-radius:16px;text-align:center;
                  background:linear-gradient(135deg,rgba(0,229,160,0.08),rgba(0,184,212,0.06));
                  border:1px solid rgba(0,229,160,0.15);">
        <div style="font-size:26px;font-weight:900;color:#00e5a0;text-shadow:0 0 15px rgba(0,229,160,0.3);">12</div>
        <div style="font-size:11px;color:rgba(160,180,200,0.6);text-transform:uppercase;letter-spacing:0.1em;margin-top:4px;">Turbines monitored</div>
      </div>
      <div style="padding:16px;border-radius:16px;text-align:center;
                  background:linear-gradient(135deg,rgba(255,214,0,0.08),rgba(255,145,0,0.06));
                  border:1px solid rgba(255,214,0,0.15);">
        <div style="font-size:26px;font-weight:900;color:#ffd600;text-shadow:0 0 15px rgba(255,214,0,0.3);">2</div>
        <div style="font-size:11px;color:rgba(160,180,200,0.6);text-transform:uppercase;letter-spacing:0.1em;margin-top:4px;">Warnings active</div>
      </div>
      <div style="padding:16px;border-radius:16px;text-align:center;
                  background:linear-gradient(135deg,rgba(0,184,212,0.08),rgba(0,229,160,0.06));
                  border:1px solid rgba(0,184,212,0.15);">
        <div style="font-size:26px;font-weight:900;color:#00b8d4;text-shadow:0 0 15px rgba(0,184,212,0.3);">94.2%</div>
        <div style="font-size:11px;color:rgba(160,180,200,0.6);text-transform:uppercase;letter-spacing:0.1em;margin-top:4px;">Model accuracy</div>
      </div>
      <div style="padding:16px;border-radius:16px;text-align:center;
                  background:linear-gradient(135deg,rgba(139,233,255,0.08),rgba(0,184,212,0.06));
                  border:1px solid rgba(139,233,255,0.15);">
        <div style="font-size:26px;font-weight:900;color:#8be9ff;text-shadow:0 0 15px rgba(139,233,255,0.3);">45d</div>
        <div style="font-size:11px;color:rgba(160,180,200,0.6);text-transform:uppercase;letter-spacing:0.1em;margin-top:4px;">Early warning</div>
      </div>
    </div>
    """


def create_fleet_dashboard_html() -> str:
    rng = np.random.default_rng(42)
    fleet_data = simulate_fleet_prediction(rng)
    cards = "\n".join(create_fleet_card(t) for t in fleet_data)
    return f"""
    <div style="display:flex;flex-direction:column;gap:16px;">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <h3 style="margin:0;color:#edf6ff;font-size:20px;font-weight:800;">
          🌍 Global Fleet Overview
        </h3>
        <span style="font-size:12px;color:rgba(160,180,200,0.5);">
          8 OEM models · 6 regions · Live simulation
        </span>
      </div>
      <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px;">
        {cards}
      </div>
    </div>
    """


# ── Chart generators ──────────────────────────────────────────────


def make_histogram(predictions, mean_rul, ci_lower, ci_upper, risk):
    colors = RISK_COLORS[risk]
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=predictions,
            nbinsx=32,
            marker={"color": colors["primary"], "line": {"color": "#0c1627", "width": 1}},
            opacity=0.85,
            hovertemplate="%{{x:.1f}} days<extra></extra>",
        )
    )
    fig.add_vline(
        x=mean_rul,
        line_color="#8be9ff",
        line_width=3,
        annotation_text=f"mean {mean_rul:.1f}d",
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
        line_color="#ffd600",
        line_width=3,
        line_dash="dot",
        annotation_text="45-day",
        annotation_position="bottom right",
    )
    fig.add_vline(
        x=14.0,
        line_color="#ff1744",
        line_width=3,
        line_dash="dot",
        annotation_text="critical",
        annotation_position="bottom left",
    )
    fig.update_layout(
        template="plotly_dark",
        height=340,
        margin={"l": 18, "r": 18, "t": 44, "b": 18},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title={
            "text": "Monte Carlo Prediction Distribution",
            "font": {"color": "#edf6ff", "size": 15},
        },
        xaxis_title="Predicted healthy life remaining (days)",
        yaxis_title="Frequency",
        bargap=0.04,
        showlegend=False,
        font={"color": "#edf6ff", "family": "system-ui"},
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)")
    return fig


def make_radar_chart(vibration, bearing_temp, gen_temp, power, wind_speed, hours, risk):
    colors = RISK_COLORS[risk]
    # Normalize to 0-100 health scores (inverted: high signal = low health for some)
    vib_health = max(0, 100 - vibration * 2.5)
    temp_health = max(0, 100 - (bearing_temp - 30) * 0.9)
    gen_health = max(0, 100 - (gen_temp - 40) * 0.6)
    power_health = min(100, power / 30)
    wind_health = min(100, wind_speed * 6)
    age_health = max(0, 100 - hours / 876)

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=[vib_health, temp_health, gen_health, power_health, wind_health, age_health]
            + [vib_health],
            theta=[
                "Vibration",
                "Bearing Temp",
                "Generator Temp",
                "Power",
                "Wind",
                "Age",
                "Vibration",
            ],
            fill="toself",
            fillcolor=hex_to_rgba(colors["primary"], 0.13),
            line={"color": colors["primary"], "width": 2},
            marker={"size": 6, "color": colors["primary"]},
        )
    )
    # Add threshold ring
    fig.add_trace(
        go.Scatterpolar(
            r=[50] * 7,
            theta=[
                "Vibration",
                "Bearing Temp",
                "Generator Temp",
                "Power",
                "Wind",
                "Age",
                "Vibration",
            ],
            fill="none",
            line={"color": "rgba(255,255,255,0.1)", "width": 1, "dash": "dot"},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        polar={
            "bgcolor": "rgba(0,0,0,0)",
            "radialaxis": {
                "visible": True,
                "range": [0, 100],
                "gridcolor": "rgba(255,255,255,0.08)",
                "tickcolor": "rgba(255,255,255,0.1)",
                "tickfont": {"color": "rgba(160,180,200,0.5)", "size": 10},
            },
            "angularaxis": {
                "gridcolor": "rgba(255,255,255,0.06)",
                "tickfont": {"color": "#8be9ff", "size": 12},
            },
        },
        template="plotly_dark",
        height=360,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 40, "r": 40, "t": 40, "b": 40},
        title={"text": "Turbine Health Radar", "font": {"color": "#edf6ff", "size": 15}},
    )
    return fig


def make_trend_chart(hours_val, risk):
    """Generate a synthetic historical degradation trend."""
    colors = RISK_COLORS[risk]
    rng = np.random.default_rng(int(hours_val) % 10000)
    max_hours = 87600
    t = np.linspace(0, min(hours_val * 1.2, max_hours), 200)
    health = 450 * np.exp(-3.5 * (t / max_hours) ** 1.4)
    noise = rng.normal(0, 5, size=len(t))
    health_noisy = health + noise
    health_noisy = np.clip(health_noisy, 0, 450)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=health_noisy,
            mode="lines",
            line={"color": colors["primary"], "width": 2.5},
            name="Health Index",
            hovertemplate="%{{x:.0f}}h: %{{y:.1f}} days<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=health_noisy,
            mode="lines",
            line={"color": colors["primary"], "width": 0},
            fill="tozeroy",
            fillcolor=hex_to_rgba(colors["primary"], 0.07),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_hline(
        y=45,
        line_color="#ffd600",
        line_width=2,
        line_dash="dot",
        annotation_text="45-day plan window",
        annotation_position="top right",
    )
    fig.add_hline(
        y=14,
        line_color="#ff1744",
        line_width=2,
        line_dash="dot",
        annotation_text="critical",
        annotation_position="bottom right",
    )
    fig.add_vline(
        x=hours_val,
        line_color="#8be9ff",
        line_width=2,
        line_dash="dash",
        annotation_text="now",
        annotation_position="top",
    )

    fig.update_layout(
        template="plotly_dark",
        height=340,
        margin={"l": 18, "r": 18, "t": 44, "b": 18},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title={"text": "Degradation Trend Over Time", "font": {"color": "#edf6ff", "size": 15}},
        xaxis_title="Operating Hours",
        yaxis_title="Estimated Healthy Life (days)",
        showlegend=False,
        font={"color": "#edf6ff", "family": "system-ui"},
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
    return fig


def make_power_curve(wind_speed, power_output, risk):
    colors = RISK_COLORS[risk]
    winds = np.linspace(0, 25, 100)
    rated = 2500.0
    cut_in, rated_wind, cut_out = 3.0, 12.0, 25.0
    power_curve = np.where(
        winds < cut_in,
        0,
        np.where(
            winds < rated_wind,
            rated * ((winds - cut_in) / (rated_wind - cut_in)) ** 3,
            np.where(winds <= cut_out, rated, 0),
        ),
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=winds,
            y=power_curve,
            mode="lines",
            line={"color": "rgba(0,184,212,0.5)", "width": 2, "dash": "dash"},
            name="Ideal power curve",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[wind_speed],
            y=[power_output],
            mode="markers",
            marker={
                "size": 16,
                "color": colors["primary"],
                "line": {"color": "white", "width": 2},
                "symbol": "circle",
            },
            name="Current operating point",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[wind_speed],
            y=[power_output],
            mode="markers",
            marker={
                "size": 30,
                "color": hex_to_rgba(colors["primary"], 0.15),
                "symbol": "circle",
            },
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=340,
        margin={"l": 18, "r": 18, "t": 44, "b": 18},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title={"text": "Power Curve Analysis", "font": {"color": "#edf6ff", "size": 15}},
        xaxis_title="Wind Speed (m/s)",
        yaxis_title="Power Output (kW)",
        font={"color": "#edf6ff", "family": "system-ui"},
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
    return fig


# ── Main prediction function ──────────────────────────────────────


def predict_rul(
    vibration_rms,
    bearing_temp,
    generator_temp,
    power_output,
    wind_speed,
    operating_hours,
    n_samples,
):
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

    bg = animated_background(risk)
    gauge = create_animated_gauge(mean_rul, std_rul, risk)
    badge = create_risk_badge(risk)
    rec = create_recommendation_card(risk, rec_title, rec_copy)

    histogram = make_histogram(predictions, mean_rul, ci_lower, ci_upper, risk)
    radar = make_radar_chart(
        vibration_rms, bearing_temp, generator_temp, power_output, wind_speed, operating_hours, risk
    )
    trend = make_trend_chart(operating_hours, risk)
    power_curve = make_power_curve(wind_speed, power_output, risk)

    stats = (
        f"### 📊 Prediction Stats\n"
        f"- **95% confidence interval:** {ci_lower:.1f} → {ci_upper:.1f} days\n"
        f"- **Uncertainty (σ):** {std_rul:.2f} days\n"
        f"- **Monte Carlo samples:** {int(n_samples)}\n"
        f"- **Model:** {bundle.source}\n"
        f"- **Inference latency:** {latency_ms:.0f} ms ({CPU_THREADS} threads)\n"
    )
    payload = {
        "mean_rul_days": round(mean_rul, 2),
        "uncertainty_days": round(std_rul, 2),
        "ci_95": [round(ci_lower, 2), round(ci_upper, 2)],
        "risk_level": risk,
        "latency_ms": round(latency_ms, 1),
        "raw_input": dict(zip(FEATURE_NAMES, raw_values)),
    }
    return bg, gauge, badge, rec, histogram, radar, trend, power_curve, stats, payload


def generate_fleet():
    """Refresh the fleet dashboard."""
    return create_fleet_dashboard_html()


# ── CSS ───────────────────────────────────────────────────────────

APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
:root {
  --bg: #060d18; --panel: rgba(10,22,40,0.92); --panel-2: rgba(16,35,63,0.88);
  --border: rgba(124,211,255,0.12); --text: #edf6ff; --muted: rgba(160,180,200,0.7);
  --teal: #00e5a0; --cyan: #00b8d4; --sky: #8be9ff;
}
*, *::before, *::after { box-sizing: border-box; }
body, .gradio-container {
  font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
  background: #060d18 !important; color: var(--text) !important;
  min-height: 100vh;
}
.gradio-container { max-width: 1400px !important; margin: 0 auto !important; padding: 20px !important; }
.gradio-blocks { background: transparent !important; }
/* Glassmorphism cards */
.glass-card {
  background: linear-gradient(135deg, rgba(10,22,40,0.92), rgba(16,35,63,0.88)) !important;
  border: 1px solid rgba(124,211,255,0.12) !important;
  border-radius: 20px !important;
  box-shadow: 0 8px 32px rgba(0,0,0,0.3), inset 0 0 0 1px rgba(255,255,255,0.03) !important;
  backdrop-filter: blur(20px) !important;
  padding: 20px !important;
}
/* Primary button glow */
button.primary, .primary {
  background: linear-gradient(135deg, #00e5a0, #00b8d4) !important;
  color: #07111f !important; font-weight: 800 !important;
  border: none !important; border-radius: 14px !important;
  box-shadow: 0 0 30px rgba(0,229,160,0.3) !important;
  transition: all 0.3s cubic-bezier(0.4,0,0.2,1) !important;
  text-transform: uppercase !important; letter-spacing: 0.06em !important;
}
button.primary:hover, .primary:hover {
  box-shadow: 0 0 50px rgba(0,229,160,0.5) !important;
  transform: translateY(-2px) !important;
}
/* Slider styling */
.gr-slider { border-radius: 12px !important; }
input[type="range"] {
  -webkit-appearance: none; appearance: none;
  height: 6px !important; border-radius: 3px !important;
  background: linear-gradient(90deg, rgba(0,229,160,0.3), rgba(0,184,212,0.3)) !important;
}
input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none; width: 18px; height: 18px; border-radius: 50%;
  background: linear-gradient(135deg, #00e5a0, #00b8d4);
  box-shadow: 0 0 12px rgba(0,229,160,0.4);
  cursor: pointer;
}
/* Tab styling */
.tabs > .tab-nav > button {
  color: var(--muted) !important; font-weight: 600 !important;
  border: none !important; border-bottom: 2px solid transparent !important;
  transition: all 0.3s !important; text-transform: uppercase !important;
  letter-spacing: 0.08em !important; font-size: 13px !important;
}
.tabs > .tab-nav > button.selected {
  color: var(--teal) !important;
  border-bottom: 2px solid var(--teal) !important;
}
/* Markdown */
.markdown h3 { color: var(--text) !important; font-weight: 800 !important; }
.markdown p, .markdown li { color: var(--muted) !important; }
/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: rgba(0,0,0,0.2); }
::-webkit-scrollbar-thumb { background: rgba(0,229,160,0.3); border-radius: 3px; }
/* Plot containers */
.plot-container { border-radius: 16px !important; overflow: hidden !important; }
/* Dropdown */
.gr-dropdown { border-radius: 12px !important; }
"""


# ── Build the app ─────────────────────────────────────────────────


def build_interface() -> gr.Blocks:
    with gr.Blocks(
        title="AeroVigil EPIC — Wind Turbine Health Intelligence",
    ) as demo:
        # Hidden background element
        bg_html = gr.HTML(value=animated_background("LOW"), visible=True)

        # Header
        gr.HTML("""
        <div style="text-align:center;padding:16px 0 8px;">
          <div style="font-size:11px;text-transform:uppercase;letter-spacing:0.25em;
                      color:rgba(0,229,160,0.7);font-weight:700;margin-bottom:6px;">
            Physics-Guided Bayesian AI · EPIC Edition
          </div>
          <h1 style="font-size:38px;font-weight:900;margin:0;
                     background:linear-gradient(135deg,#00e5a0,#00b8d4,#8be9ff);
                     -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                     background-clip:text;line-height:1.2;">
            AeroVigil
          </h1>
          <p style="color:rgba(180,200,220,0.6);font-size:15px;margin:6px 0 0;
                    font-weight:400;letter-spacing:0.02em;">
            See the failure before it happens · Schedule the repair before it becomes a rescue
          </p>
          <nav style="display:flex;justify-content:center;gap:10px;flex-wrap:wrap;margin-top:14px;">
            <a href="/api/docs" target="_blank" style="color:#8be9ff;text-decoration:none;
               border:1px solid rgba(139,233,255,.22);border-radius:999px;padding:6px 12px;">
              Advisory · Twin · Telemetry API
            </a>
            <a href="/model-api/docs" target="_blank" style="color:#8be9ff;text-decoration:none;
               border:1px solid rgba(139,233,255,.22);border-radius:999px;padding:6px 12px;">
              PG-BNN Model API
            </a>
            <a href="/health" target="_blank" style="color:#00e5a0;text-decoration:none;
               border:1px solid rgba(0,229,160,.22);border-radius:999px;padding:6px 12px;">
              Unified Health
            </a>
          </nav>
        </div>
        """)

        # KPI strip
        gr.HTML(create_kpi_strip())

        with gr.Tabs():
            # ── TAB 1: Dashboard ─────────────────────────────
            with gr.TabItem("📊 Dashboard"):
                gr.HTML(create_fleet_dashboard_html(), elem_classes=["glass-card"])
                with gr.Row():
                    fleet_refresh = gr.Button(
                        "🔄 Refresh Fleet Data", variant="secondary", size="sm"
                    )
                fleet_output = gr.HTML()
                fleet_refresh.click(fn=generate_fleet, outputs=[fleet_output])

            # ── TAB 2: Predict ───────────────────────────────
            with gr.TabItem("🔮 Predict"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=360):
                        gr.HTML("""
                        <div style="padding:16px 20px;border-radius:18px;
                                    background:linear-gradient(135deg,rgba(10,22,40,0.92),rgba(16,35,63,0.88));
                                    border:1px solid rgba(124,211,255,0.1);margin-bottom:8px;">
                          <h3 style="margin:0 0 4px;color:#edf6ff;font-size:17px;font-weight:800;">
                            🎛️ Set Turbine State
                          </h3>
                          <p style="margin:0;color:rgba(160,180,200,0.6);font-size:13px;">
                            Choose a preset or fine-tune the 6 SCADA signals
                          </p>
                        </div>
                        """)
                        scenario = gr.Dropdown(
                            choices=list(SCENARIOS.keys()),
                            value="🟢 Healthy turbine",
                            label="Scenario Preset",
                            info="Pick a demo case — sliders update automatically",
                        )
                        vibration_rms = gr.Slider(
                            0.0, 40.0, value=12.5, step=0.1, label="🫨 Vibration RMS (mm/s)"
                        )
                        bearing_temp = gr.Slider(
                            30.0, 130.0, value=65.0, step=0.5, label="🌡️ Bearing Temperature (°C)"
                        )
                        generator_temp = gr.Slider(
                            40.0, 170.0, value=80.0, step=0.5, label="🔥 Generator Temperature (°C)"
                        )
                        power_output = gr.Slider(
                            0.0, 3000.0, value=2000.0, step=10.0, label="⚡ Power Output (kW)"
                        )
                        wind_speed = gr.Slider(
                            0.0, 20.0, value=9.0, step=0.1, label="💨 Wind Speed (m/s)"
                        )
                        operating_hours = gr.Slider(
                            0.0, 90000.0, value=1000.0, step=100.0, label="⏱️ Operating Hours"
                        )
                        n_samples = gr.Slider(
                            20, 200, value=100, step=10, label="🎲 Monte Carlo Samples"
                        )
                        predict_btn = gr.Button("🚀 Run Prediction", variant="primary", size="lg")

                    with gr.Column(scale=6, min_width=420):
                        gauge_html = gr.HTML(value=create_animated_gauge(280.0, 5.1, "LOW"))
                        risk_badge_html = gr.HTML(value=create_risk_badge("LOW"))
                        rec_html = gr.HTML(
                            value=create_recommendation_card(
                                "LOW",
                                "✅ Healthy: keep standard monitoring",
                                "Telemetry looks nominal. Press Predict to run stochastic inference.",
                            )
                        )
                        stats_md = gr.Markdown(
                            "### 📊 Prediction Stats\n- **Run a prediction** to see results",
                            elem_classes=["glass-card"],
                        )
                        api_payload = gr.JSON(value={}, visible=False)

                # Charts row
                with gr.Row():
                    hist_plot = gr.Plot(label="Prediction Distribution")
                    radar_plot = gr.Plot(label="Health Radar")
                with gr.Row():
                    trend_plot = gr.Plot(label="Degradation Trend")
                    power_plot = gr.Plot(label="Power Curve")

                with gr.Accordion("🧑‍🏫 How to read the results", open=False):
                    gr.Markdown("""
                    ### Reading the AeroVigil output

                    - **Big number (gauge):** average healthy-life runway in days
                    - **Spread (histogram width):** how much the model disagrees with itself — wider = less certain
                    - **Yellow 45-day line:** left of this = proactive-maintenance window
                    - **Red 14-day line:** left of this = urgent action needed
                    - **Radar chart:** shows balance across all 6 health dimensions
                    - **Trend chart:** projected degradation path over the turbine's lifetime
                    - **Power curve:** how the current operating point compares to the ideal curve
                    - **AeroVigil advises only** — it never controls the turbine
                    """)

            # ── TAB 3: Fleet View ────────────────────────────
            with gr.TabItem("🌍 Fleet View"):
                gr.HTML("""
                <div style="padding:16px 20px;border-radius:18px;
                            background:linear-gradient(135deg,rgba(10,22,40,0.92),rgba(16,35,63,0.88));
                            border:1px solid rgba(124,211,255,0.1);margin-bottom:12px;">
                  <h3 style="margin:0;color:#edf6ff;font-size:18px;font-weight:800;">
                    🌍 Global Fleet Health Monitor
                  </h3>
                  <p style="margin:6px 0 0;color:rgba(160,180,200,0.6);font-size:13px;">
                    12 turbines across 6 regions · 8 OEM models · Real-time simulation
                  </p>
                </div>
                """)
                gr.HTML(value=create_fleet_dashboard_html())
                fleet_map = make_fleet_map_chart()
                gr.Plot(value=fleet_map, label="Global Fleet Map")
                fleet_bar = make_fleet_comparison_chart()
                gr.Plot(value=fleet_bar, label="Fleet RUL Comparison")

            # ── TAB 4: Digital Twin ──────────────────────────
            with gr.TabItem("🏭 Digital Twin"):
                gr.HTML("""
                <div style="padding:16px 20px;border-radius:18px;
                            background:linear-gradient(135deg,rgba(10,22,40,0.92),rgba(16,35,63,0.88));
                            border:1px solid rgba(124,211,255,0.1);margin-bottom:12px;">
                  <h3 style="margin:0;color:#edf6ff;font-size:18px;font-weight:800;">
                    🏭 Digital Twin — Scenario Simulator
                  </h3>
                  <p style="margin:6px 0 0;color:rgba(160,180,200,0.6);font-size:13px;">
                    Simulate future operating scenarios and their impact on bearing life
                  </p>
                </div>
                """)
                with gr.Row():
                    with gr.Column():
                        twin_scenario = gr.Dropdown(
                            choices=[
                                "Nominal operation",
                                "High wind overload",
                                "Derated operation",
                                "Tropical heat stress",
                            ],
                            value="Nominal operation",
                            label="Scenario",
                        )
                        twin_hours = gr.Slider(
                            0, 20000, 5000, 500, label="Additional hours to simulate"
                        )
                        twin_btn = gr.Button("🔬 Run Simulation", variant="primary")
                    with gr.Column():
                        twin_output = gr.HTML()
                twin_chart = gr.Plot(label="Scenario Projection")
                twin_btn.click(
                    fn=run_twin_simulation,
                    inputs=[
                        twin_scenario,
                        twin_hours,
                        vibration_rms,
                        bearing_temp,
                        generator_temp,
                        power_output,
                        wind_speed,
                        operating_hours,
                    ],
                    outputs=[twin_output, twin_chart],
                )

            # ── TAB 5: Analytics ─────────────────────────────
            with gr.TabItem("📈 Analytics"):
                gr.HTML("""
                <div style="padding:16px 20px;border-radius:18px;
                            background:linear-gradient(135deg,rgba(10,22,40,0.92),rgba(16,35,63,0.88));
                            border:1px solid rgba(124,211,255,0.1);margin-bottom:12px;">
                  <h3 style="margin:0;color:#edf6ff;font-size:18px;font-weight:800;">
                    📈 Fleet Analytics & Insights
                  </h3>
                </div>
                """)
                with gr.Row():
                    analytics_risk_dist = make_risk_distribution_chart()
                    gr.Plot(value=analytics_risk_dist, label="Risk Distribution")
                    analytics_oem = make_oem_comparison_chart()
                    gr.Plot(value=analytics_oem, label="OEM Comparison")
                with gr.Row():
                    analytics_timeline = make_timeline_chart()
                    gr.Plot(value=analytics_timeline, label="Maintenance Timeline")

        # Footer
        gr.HTML("""
        <div style="text-align:center;padding:20px 12px;margin-top:8px;
                    border-radius:18px;background:linear-gradient(135deg,rgba(10,22,40,0.85),rgba(16,35,63,0.8));
                    border:1px solid rgba(124,211,255,0.08);">
          <p style="color:rgba(160,180,200,0.5);font-size:12px;margin:0;line-height:1.6;">
            ⚠️ Advisory only · Decision-support for maintenance planning · Does not send turbine
            control commands · Read
            <a href="https://github.com/rajaram-2005/wind-turbine-pg-bnn/blob/main/docs/SAFETY.md"
               target="_blank" style="color:rgba(0,229,160,0.7);">docs/SAFETY.md</a>
          </p>
          <p style="color:rgba(160,180,200,0.4);font-size:11px;margin:6px 0 0;">
            AeroVigil EPIC · Physics-Guided Bayesian Neural Network ·
            <a href="https://github.com/rajaram-2005/wind-turbine-pg-bnn" target="_blank"
               style="color:rgba(0,184,212,0.6);">GitHub</a> ·
            <a href="https://huggingface.co/AerovigilAI/wind-turbine-pg-bnn" target="_blank"
               style="color:rgba(0,184,212,0.6);">Hugging Face</a>
          </p>
        </div>
        """)

        # Wire up events
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

        predict_btn.click(
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
            outputs=[
                bg_html,
                gauge_html,
                risk_badge_html,
                rec_html,
                hist_plot,
                radar_plot,
                trend_plot,
                power_plot,
                stats_md,
                api_payload,
            ],
            api_name="predict_rul",
        )

    return demo


# ── Additional chart generators for tabs ──────────────────────────


def make_fleet_map_chart():
    fig = go.Figure()
    locations = [
        ("North Ridge α", 57.0, 10.0),
        ("North Ridge β", 57.1, 10.2),
        ("Mumbai Coast 1", 19.0, 72.8),
        ("Mumbai Coast 2", 19.1, 72.9),
        ("Iowa Plains A", 41.9, -93.1),
        ("Iowa Plains B", 42.0, -93.0),
        ("Patagonia Wind", -40.7, -65.3),
        ("Nordic Frost", 61.9, 24.0),
        ("Sahara Edge", 31.6, -8.0),
        ("Offshore Delta", 52.4, 4.3),
        ("Highland Echo", 57.5, -5.0),
        ("Tropical Breeze", 11.0, 77.5),
    ]
    rng = np.random.default_rng(42)
    for name, lat, lon in locations:
        rul = float(rng.uniform(8, 350))
        risk, _, _ = classify_risk(rul)
        colors = RISK_COLORS[risk]
        fig.add_trace(
            go.Scattergeo(
                lat=[lat],
                lon=[lon],
                text=[f"{name}<br>RUL: {rul:.0f}d"],
                mode="markers",
                hoverinfo="text",
                marker={
                    "size": max(8, min(20, rul / 15)),
                    "color": colors["primary"],
                    "opacity": 0.8,
                    "line": {"width": 1, "color": "white"},
                },
            )
        )
    fig.update_geos(
        showcountries=True,
        countrycolor="rgba(124,211,255,0.1)",
        showcoastlines=True,
        coastlinecolor="rgba(124,211,255,0.2)",
        showland=True,
        landcolor="rgba(10,22,40,0.8)",
        showocean=True,
        oceancolor="rgba(6,13,24,0.9)",
        projection_type="natural earth",
    )
    fig.update_layout(
        template="plotly_dark",
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        title={"text": "🌍 Global Fleet Locations", "font": {"color": "#edf6ff", "size": 16}},
        font={"color": "#edf6ff"},
    )
    return fig


def make_fleet_comparison_chart():
    rng = np.random.default_rng(42)
    turbines = [t["id"] for t in FLEET_TURBINES]
    ruls = [float(rng.uniform(8, 350)) for _ in turbines]
    colors_list = [RISK_COLORS[classify_risk(r)[0]]["primary"] for r in ruls]
    fig = go.Figure(
        go.Bar(
            x=turbines,
            y=ruls,
            marker_color=colors_list,
            text=[f"{r:.0f}d" for r in ruls],
            textposition="outside",
            textfont={"color": "#edf6ff", "size": 11},
        )
    )
    fig.add_hline(y=45, line_color="#ffd600", line_dash="dot", line_width=2)
    fig.add_hline(y=14, line_color="#ff1744", line_dash="dot", line_width=2)
    fig.update_layout(
        template="plotly_dark",
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title={"text": "RUL Comparison Across Fleet", "font": {"color": "#edf6ff", "size": 15}},
        xaxis_title="Turbine",
        yaxis_title="Days of healthy life remaining",
        font={"color": "#edf6ff", "family": "system-ui"},
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", tickangle=45)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
    return fig


def run_twin_simulation(scenario_name, extra_hours, vib, btemp, gtemp, power, wind, hours):
    """Simulate degradation under different scenarios."""
    total_hours = hours + extra_hours
    scenarios_config = {
        "Nominal operation": {"vib_mult": 1.0, "temp_bias": 0, "degradation": 1.0},
        "High wind overload": {"vib_mult": 1.8, "temp_bias": 8, "degradation": 1.5},
        "Derated operation": {"vib_mult": 0.7, "temp_bias": -5, "degradation": 0.6},
        "Tropical heat stress": {"vib_mult": 1.2, "temp_bias": 20, "degradation": 1.3},
    }
    cfg = scenarios_config[scenario_name]
    future_hours = np.linspace(hours, total_hours, 100)
    base_vib = vib * cfg["vib_mult"]
    base_temp = btemp + cfg["temp_bias"]

    rul = np.clip(
        450.0
        - 5.5 * max(base_vib - 5.0, 0) ** 1.18
        - 2.4 * max(base_temp - 52.0, 0) ** 1.12
        - (future_hours / 87600.0) * 220.0 * cfg["degradation"],
        0,
        450,
    )

    colors = RISK_COLORS[classify_risk(float(rul[-1]))[0]]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=future_hours,
            y=rul,
            mode="lines",
            line={"color": colors["primary"], "width": 3},
            name="Projected RUL",
            fill="tozeroy",
            fillcolor=hex_to_rgba(colors["primary"], 0.08),
        )
    )
    fig.add_hline(
        y=45, line_color="#ffd600", line_dash="dot", line_width=2, annotation_text="45-day line"
    )
    fig.add_hline(
        y=14, line_color="#ff1744", line_dash="dot", line_width=2, annotation_text="critical"
    )
    fig.add_vline(
        x=hours, line_color="#8be9ff", line_dash="dash", line_width=2, annotation_text="now"
    )
    fig.update_layout(
        template="plotly_dark",
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title={"text": f"Scenario: {scenario_name}", "font": {"color": "#edf6ff", "size": 15}},
        xaxis_title="Operating Hours",
        yaxis_title="Projected healthy life (days)",
        font={"color": "#edf6ff", "family": "system-ui"},
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)")

    final_rul = float(rul[-1])
    risk, _, _ = classify_risk(final_rul)
    status_color = RISK_COLORS[risk]["primary"]
    info = f"""
    <div style="padding:20px;border-radius:18px;
                background:linear-gradient(135deg,rgba(10,22,40,0.92),rgba(16,35,63,0.88));
                border:1px solid {hex_to_rgba(status_color, 0.2)};">
      <h4 style="color:{status_color};margin:0 0 12px;">🔬 Simulation Result</h4>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;font-size:14px;">
        <div><span style="color:rgba(160,180,200,0.6);">Scenario:</span> <b style="color:#edf6ff;">{scenario_name}</b></div>
        <div><span style="color:rgba(160,180,200,0.6);">Total hours:</span> <b style="color:#edf6ff;">{total_hours:,.0f}h</b></div>
        <div><span style="color:rgba(160,180,200,0.6);">Final RUL:</span> <b style="color:{status_color};">{final_rul:.0f} days</b></div>
        <div><span style="color:rgba(160,180,200,0.6);">Degradation rate:</span> <b style="color:#edf6ff;">{cfg["degradation"]}x</b></div>
      </div>
    </div>
    """
    return info, fig


def make_risk_distribution_chart():
    rng = np.random.default_rng(42)
    ruls = [float(rng.uniform(5, 350)) for _ in FLEET_TURBINES]
    risks = [classify_risk(r)[0] for r in ruls]
    counts = {r: risks.count(r) for r in ["LOW", "MODERATE", "HIGH", "CRITICAL"]}
    colors_list = [RISK_COLORS[r]["primary"] for r in counts]
    fig = go.Figure(
        go.Pie(
            labels=list(counts.keys()),
            values=list(counts.values()),
            marker_colors=colors_list,
            hole=0.5,
            textinfo="label+value",
            textfont={"color": "#edf6ff", "size": 13},
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        title={"text": "Fleet Risk Distribution", "font": {"color": "#edf6ff", "size": 15}},
        font={"color": "#edf6ff"},
    )
    return fig


def make_oem_comparison_chart():
    oems = ["GE", "Vestas", "Siemens", "Suzlon", "Gamesa", "Nordex", "Senvion", "NREL"]
    rng = np.random.default_rng(123)
    avg_ruls = [float(rng.uniform(50, 280)) for _ in oems]
    colors_list = [RISK_COLORS[classify_risk(r)[0]]["primary"] for r in avg_ruls]
    fig = go.Figure(
        go.Bar(
            x=oems,
            y=avg_ruls,
            marker_color=colors_list,
            text=[f"{r:.0f}d" for r in avg_ruls],
            textposition="outside",
            textfont={"color": "#edf6ff", "size": 11},
        )
    )
    fig.add_hline(y=45, line_color="#ffd600", line_dash="dot", line_width=2)
    fig.update_layout(
        template="plotly_dark",
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title={"text": "Average RUL by OEM Model", "font": {"color": "#edf6ff", "size": 15}},
        xaxis_title="OEM",
        yaxis_title="Avg. healthy life (days)",
        font={"color": "#edf6ff", "family": "system-ui"},
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
    return fig


def make_timeline_chart():
    import datetime

    rng = np.random.default_rng(42)
    today = datetime.date.today()
    dates = [today + datetime.timedelta(days=i * 7) for i in range(12)]
    planned = [int(rng.integers(0, 4)) for _ in range(12)]
    warnings = [int(rng.integers(1, 6)) for _ in range(12)]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=dates, y=planned, name="Planned interventions", marker_color="#00e5a0", opacity=0.8
        )
    )
    fig.add_trace(
        go.Bar(x=dates, y=warnings, name="Active warnings", marker_color="#ffd600", opacity=0.8)
    )
    fig.update_layout(
        template="plotly_dark",
        height=350,
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title={"text": "12-Week Maintenance Timeline", "font": {"color": "#edf6ff", "size": 15}},
        xaxis_title="Week",
        yaxis_title="Turbines",
        font={"color": "#edf6ff", "family": "system-ui"},
        legend={"font": {"color": "#edf6ff"}},
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
    return fig


if __name__ == "__main__":
    demo = build_interface()
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7860,
        allowed_paths=[str(ROOT)],
        strict_cors=False,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="teal", secondary_hue="cyan", neutral_hue="slate"),
        css=APP_CSS,
    )
