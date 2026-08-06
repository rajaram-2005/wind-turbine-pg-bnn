"""Build the 60-second AeroVigil *live dashboard* video.

What this is
------------
A 1280x720 / 24 fps / exactly-60.0 s walkthrough of the live AeroVigil
dashboard (the same UI the Gradio app at `gradio_app/app.py` renders). The
numbers are NOT faked: every scenario runs real 100-sample Monte Carlo VI
inference on the local demo weights in `artifacts/pg_bnn_demo/`, using the
same preprocessing and risk thresholds as the app.

No browser is required: frames are composited from the app's own CSS palette,
widgets (gauge, risk badge, recommendation, stats, 28-bin histogram), and an
animated cursor that "drives" the session like a screen recording.

Outputs
-------
- docs/assets/aerovigil-live-dashboard-60s.mp4   (video + narration + clicks)
- narration clips: docs/assets/audio/live_intro.wav, live_outro.wav

Run (from the repo root, in a venv with torch/numpy/matplotlib/pillow/imageio-ffmpeg):
    python scripts/build_live_dashboard_video.py
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aerovigil_pg_bnn import PhysicsGuidedBNN  # noqa: E402

# ---------------------------------------------------------------------------
# Constants (mirror the Gradio app's palette and layout)
# ---------------------------------------------------------------------------
FRAME_W, FRAME_H = 1280, 720
FPS = 24
DURATION = 60.0
N_FRAMES = int(DURATION * FPS)

BG = "#0a1628"
PANEL = "#10233f"
PANEL_2 = "#163356"
TEXT = "#eef7ff"
MUTED = "#9eb7ce"
TEAL = "#20d3c2"
CYAN = "#8be9ff"
GOLD = "#ffd15c"
RED = "#ff5c70"
ORANGE = "#ff9b5f"
GREEN = "#3dd9b4"

RISK_META = {
    "LOW":       (GREEN,  "#0f241f", "LOW RISK · healthy trend"),
    "MODERATE":  (GOLD,   "#2c2611", "MODERATE RISK · inside 45-day plan window"),
    "HIGH":      (ORANGE, "#2b1b13", "HIGH RISK · intervention soon"),
    "CRITICAL":  (RED,    "#2a1118", "CRITICAL RISK · act now"),
}

SCENARIOS = {
    "healthy": {
        "label": "Healthy turbine",
        "raw": np.array([12.5, 65.0, 80.0, 2000.0, 9.0, 1000.0], dtype=np.float32),
        "chips": ["1.5 mm/s", "65 °C", "80 °C", "2000 kW", "9.0 m/s", "1,000 h"],
    },
    "warning": {
        "label": "Warning: degradation building",
        "raw": np.array([20.0, 88.0, 115.0, 2100.0, 11.0, 52000.0], dtype=np.float32),
        "chips": ["20.0 mm/s", "88 °C", "115 °C", "2100 kW", "11.0 m/s", "52,000 h"],
    },
    "critical": {
        "label": "Critical: act now",
        "raw": np.array([34.0, 118.0, 150.0, 2400.0, 12.0, 78000.0], dtype=np.float32),
        "chips": ["34.0 mm/s", "118 °C", "150 °C", "2400 kW", "12.0 m/s", "78,000 h"],
    },
}

FEATURE_LABELS = ["vibration_rms", "bearing_temp", "generator_temp",
                  "power_output", "wind_speed", "operating_hours"]
CHIP_SHORT_LABELS = ["vibration", "bear temp", "gen temp", "power", "wind", "run hours"]
CHIP_UNITS = ["mm/s", "°C", "°C", "kW", "m/s", "h"]
CHIP_NUMS = {
    "healthy":  [1.5, 65.0, 80.0, 2000.0, 9.0, 1000.0],
    "warning":  [20.0, 88.0, 115.0, 2100.0, 11.0, 52000.0],
    "critical": [34.0, 118.0, 150.0, 2400.0, 12.0, 78000.0],
}


def format_chips(nums: list[float]) -> list[str]:
    out = []
    for i, v in enumerate(nums):
        out.append(f"{v:.1f} {CHIP_UNITS[i]}" if CHIP_UNITS[i] == "mm/s"
                   else f"{v:,.0f} {CHIP_UNITS[i]}")
    return out

REC_TITLES = {
    "LOW": "Healthy outlook: keep standard monitoring",
    "MODERATE": "Use the 45-day planning window",
    "HIGH": "Plan the repair in the next 2–4 weeks",
    "CRITICAL": "Urgent action: stage crane + crew now",
}
REC_BODIES = {
    "LOW": ("Telemetry looks nominal. Keep trend monitoring on the normal cadence "
            "and use this as a baseline for future comparisons."),
    "MODERATE": ("This is your proactive-maintenance zone. Put the turbine on the "
                 "next maintenance schedule so you avoid a surprise breakdown."),
    "HIGH": ("The model still sees time to organize the work, but not to wait. "
             "Confirm parts, watch the trend daily, and reserve your field team."),
    "CRITICAL": ("Failure risk is inside two weeks. Treat this as a rescue-avoidance "
                 "window: lock a crane slot, pre-stage spares, and schedule the "
                 "intervention immediately."),
}

AUDIO_DIR = ROOT / "docs" / "assets" / "audio"
OUT_VIDEO = ROOT / "docs" / "assets" / "aerovigil-live-dashboard-60s.mp4"
OUT_AUDIO = Path("/tmp") / "av_live_mix.wav"
FRAMES_DIR = Path("/tmp") / "av_live_frames"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# Timeline anchors (seconds)
T_LOAD = 8.0          # page fully loaded
T_CLICK1 = 12.0       # Predict (healthy)
T_RES1 = 14.0         # healthy results land
T_SEL_WARN = 23.0     # select warning scenario
T_CLICK2 = 25.0       # Predict (warning)
T_RES2 = 26.5
T_SEL_CRIT = 34.5     # select critical
T_CLICK3 = 37.0
T_RES3 = 38.5
T_SEL_HEAL = 48.0     # back to healthy
T_CLICK4 = 50.5
T_RES4 = 51.5
T_ENDCARD = 54.5

CLICK_TIMES = [T_CLICK1, T_SEL_WARN, T_CLICK2, T_SEL_CRIT, T_CLICK3, T_SEL_HEAL, T_CLICK4]


@dataclass
class Stats:
    name: str
    mean: float
    std: float
    ci_low: float
    ci_high: float
    risk: str
    latency_ms: float
    preds: np.ndarray


def risk_for(mean_rul: float) -> str:
    if mean_rul < 14.0:
        return "CRITICAL"
    if mean_rul < 30.0:
        return "HIGH"
    if mean_rul < 45.0:
        return "MODERATE"
    return "LOW"


def load_stats() -> dict[str, Stats]:
    torch.manual_seed(7)
    np.random.seed(7)
    folder = ROOT / "artifacts" / "pg_bnn_demo"
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    scaler = np.load(folder / "scaler.npz")
    mean, std = scaler["mean"].astype(np.float32), scaler["std"].astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    model = PhysicsGuidedBNN(config)
    model.load_state_dict(torch.load(folder / "bnn_demo.pt", map_location="cpu", weights_only=True))
    model.eval()

    out: dict[str, Stats] = {}
    for name, meta in SCENARIOS.items():
        x = ((meta["raw"] - mean) / std).astype(np.float32)
        tensor = torch.tensor([x.tolist()], dtype=torch.float32)
        started = time.perf_counter()
        preds: list[float] = []
        model.train()
        with torch.no_grad():
            for _ in range(100):
                m, _ = model(tensor)
                preds.append(float(m.squeeze().item()))
        arr = np.array(preds, dtype=np.float32)
        latency_ms = (time.perf_counter() - started) * 1000.0
        m = float(arr.mean())
        out[name] = Stats(
            name=name,
            mean=m,
            std=float(arr.std()),
            ci_low=float(np.percentile(arr, 2.5)),
            ci_high=float(np.percentile(arr, 97.5)),
            risk=risk_for(m),
            latency_ms=latency_ms,
            preds=arr,
        )
        print(f"  {name:9s} mean={m:6.1f} σ={arr.std():5.1f} "
              f"ci=[{np.percentile(arr, 2.5):6.1f},{np.percentile(arr, 97.5):6.1f}] "
              f"risk={risk_for(m):8s} {latency_ms:4.0f} ms")
    return out


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def font_paths() -> dict[str, Path]:
    ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    return {
        "regular": ttf / "DejaVuSans.ttf",
        "bold": ttf / "DejaVuSans-Bold.ttf",
        "mono": ttf / "DejaVuSansMono.ttf",
    }


FONTS = {
    "kicker": ImageFont.truetype(str(font_paths()["bold"]), 17),
    "title": ImageFont.truetype(str(font_paths()["bold"]), 30),
    "h2": ImageFont.truetype(str(font_paths()["bold"]), 24),
    "body": ImageFont.truetype(str(font_paths()["regular"]), 21),
    "small": ImageFont.truetype(str(font_paths()["regular"]), 17),
    "chip": ImageFont.truetype(str(font_paths()["bold"]), 16),
    "tiny": ImageFont.truetype(str(font_paths()["regular"]), 13),
    "mono": ImageFont.truetype(str(font_paths()["mono"]), 17),
    "gauge": ImageFont.truetype(str(font_paths()["bold"]), 64),
    "endtitle": ImageFont.truetype(str(font_paths()["bold"]), 52),
    "endsub": ImageFont.truetype(str(font_paths()["regular"]), 30),
}


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def lerp_color(c1: str, c2: str, t: float) -> str:
    r1, g1, b1 = ImageColor.getrgb(c1)
    r2, g2, b2 = ImageColor.getrgb(c2)
    return "#{:02x}{:02x}{:02x}".format(
        int(lerp(r1, r2, t)), int(lerp(g1, g2, t)), int(lerp(b1, b2, t))
    )


def canvas() -> Image.Image:
    img = Image.new("RGBA", (FRAME_W, FRAME_H), ImageColor.getrgb(BG) + (255,))
    glow = Image.new("RGBA", (FRAME_W, FRAME_H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse((900, -80, 1340, 360), fill=(32, 211, 194, 30))
    gd.ellipse((-140, -100, 340, 300), fill=(139, 233, 255, 22))
    glow = glow.filter(ImageFilter.GaussianBlur(50))
    img.alpha_composite(glow)
    return img


def panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int],
          fill: str = PANEL, outline: str | None = None, radius: int = 22) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill,
                           outline=outline or "rgba(124,211,255,45)", width=1)


def write_wrapped(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int],
                  width: int, font, fill: str, spacing: int = 4) -> int:
    words = text.split()
    lines, cur = [], ""
    for w in words:
        cand = f"{cur} {w}".strip()
        if draw.textbbox((0, 0), cand, font=font)[2] <= width or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    x, y = xy
    for ln in lines:
        draw.text((x, y), ln, font=font, fill=fill)
        y += font.size + spacing
    return y


def center_text(draw: ImageDraw.ImageDraw, cx: int, y: int, text: str, font, fill: str) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text((cx - (bbox[2] - bbox[0]) / 2 - bbox[0], y), text, font=font, fill=fill)
    return y + font.size


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------
def draw_chrome(img: Image.Image, clock: str, pulse: float) -> None:
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, FRAME_W, 44), fill="#0b1a2e")
    d.line((0, 44, FRAME_W, 44), fill="rgba(124,211,255,50)", width=1)
    for i, c in enumerate(["#ff5f57", "#febc2e", "#28c840"]):
        d.ellipse((18 + i * 22, 15, 30 + i * 22, 27), fill=c)
    d.rounded_rectangle((430, 9, 850, 35), radius=13, fill="#0a1628", outline="#21476e", width=1)
    d.ellipse((446, 17, 456, 27), fill="#3dd9b4")
    d.text((464, 13), "https://aerovigil.abacusai.app", font=FONTS["mono"], fill=TEXT)
    # LIVE badge
    a = int(120 + 110 * pulse)
    d.ellipse((1106, 17, 1122, 33), fill=f"rgba(255,92,112,{a})", outline=RED, width=2)
    d.text((1130, 14), "LIVE", font=FONTS["kicker"], fill=RED)
    d.text((1172, 13), clock, font=FONTS["mono"], fill=MUTED)


def draw_hero(img: Image.Image, active: str, load_alpha: float) -> None:
    d = ImageDraw.Draw(img)
    # left: kicker + title
    k = d.textbbox((0, 0), "AEROVIGIL · LIVE DASHBOARD", font=FONTS["kicker"])
    d.rectangle((26, 60, 30 + (k[2] - k[0]), 72), fill=TEAL)
    d.text((40, 58), "AEROVIGIL · LIVE DASHBOARD", font=FONTS["kicker"], fill=CYAN)
    d.text((26, 82), "Wind-turbine RUL early warning", font=FONTS["title"], fill=TEXT)
    # chips
    for i, chip_txt in enumerate(["45-day horizon", "100 MC passes", "Physics-guided · ISO 281", "Uncertainty-aware"]):
        b = d.textbbox((0, 0), chip_txt, font=FONTS["chip"])
        w = b[2] - b[0] + 22
        x = 26 + i * (w + 9)
        d.rounded_rectangle((x, 122, x + w, 148), radius=13,
                            fill=(32, 211, 194, 22), outline=(32, 211, 194, 66))
        d.text((x + 11, 126), chip_txt, font=FONTS["chip"], fill=CYAN)
    # right: scenario segmented control
    d.text((806, 60), "Scenario", font=FONTS["small"], fill=MUTED)
    order = ["healthy", "warning", "critical"]
    labels = {"healthy": "Healthy turbine", "warning": "Warning", "critical": "Critical"}
    x = 806
    for name in order:
        lbl = labels[name]
        b = d.textbbox((0, 0), lbl, font=FONTS["chip"])
        w = b[2] - b[0] + 24
        if name == active:
            d.rounded_rectangle((x, 78, x + w, 112), radius=17, fill=TEAL)
            d.text((x + 12, 86), lbl, font=FONTS["chip"], fill="#07111f")
        else:
            d.rounded_rectangle((x, 78, x + w, 112), radius=17, fill=PANEL, outline="#21476e")
            d.text((x + 12, 86), lbl, font=FONTS["chip"], fill=MUTED)
        x += w + 8


def draw_gauge(img: Image.Image, box: tuple[int, int, int, int], mean_val: float,
               sigma: float, ring_color: str, risk_text: str, risk_color: str,
               fade: float = 1.0) -> None:
    x0, y0, x1, y1 = box
    overlay = Image.new("RGBA", (FRAME_W, FRAME_H), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    panel(d, (x0, y0, x1, y1))
    cx, cy = x0 + 118, y0 + 88
    outer, inner = 82, 55
    d.ellipse((cx - outer, cy - outer, cx + outer, cy + outer), fill="#0c172a")
    d.arc((cx - outer, cy - outer, cx + outer, cy + outer), start=-90, end=270, fill="#33465f", width=14)
    progress = max(0.02, min(mean_val / 365.0, 1.0))
    d.arc((cx - outer, cy - outer, cx + outer, cy + outer),
          start=-90, end=-90 + int(progress * 360), fill=ring_color, width=14)
    d.ellipse((cx - inner, cy - inner, cx + inner, cy + inner), fill="#07111f")
    num = f"{mean_val:.0f}"
    b = d.textbbox((0, 0), num, font=FONTS["gauge"])
    d.text((cx - (b[2] - b[0]) / 2, cy - 32), num, font=FONTS["gauge"], fill=TEXT)
    lbl = "days of healthy life"
    b = d.textbbox((0, 0), lbl, font=FONTS["small"])
    d.text((cx - (b[2] - b[0]) / 2, cy + 24), lbl, font=FONTS["small"], fill=MUTED)
    sub = f"± {sigma:.1f} days uncertainty"
    b = d.textbbox((0, 0), sub, font=FONTS["chip"])
    d.text((cx - (b[2] - b[0]) / 2, cy + 50), sub, font=FONTS["chip"], fill=CYAN)
    # right side: risk + note
    d.text((x0 + 226, y0 + 26), risk_text, font=FONTS["h2"], fill=risk_color)
    write_wrapped(d, "AeroVigil estimates how much runway remains before the turbine "
                     "enters the caution window.", (x0 + 226, y0 + 68), x1 - x0 - 244,
                  FONTS["small"], MUTED, 4)
    if fade < 1.0:
        overlay = overlay.filter(ImageFilter.GaussianBlur(0.4))
    img.alpha_composite(overlay)


def draw_badge(img: Image.Image, box: tuple[int, int, int, int], risk: str, fade: float = 1.0) -> None:
    color, bg, label = RISK_META[risk]
    overlay = Image.new("RGBA", (FRAME_W, FRAME_H), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    d.rounded_rectangle(box, radius=999, fill=bg, outline=color, width=2)
    b = d.textbbox((0, 0), label, font=FONTS["chip"])
    tw = b[2] - b[0]
    d.text(((box[0] + box[2]) / 2 - tw / 2, (box[1] + box[3]) / 2 - 9), label,
           font=FONTS["chip"], fill=color)
    if fade < 1.0:
        overlay = overlay.filter(ImageFilter.GaussianBlur(0.4))
    img.alpha_composite(overlay)


def draw_rec(img: Image.Image, box: tuple[int, int, int, int], risk: str) -> None:
    color, bg, _ = RISK_META[risk]
    x0, y0, x1, y1 = box
    overlay = Image.new("RGBA", (FRAME_W, FRAME_H), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    panel(d, (x0, y0, x1, y1), fill=bg, outline=color)
    d.text((x0 + 22, y0 + 14), "AERO VIGIL RECOMMENDS", font=FONTS["kicker"], fill=color)
    d.text((x0 + 22, y0 + 42), REC_TITLES[risk], font=FONTS["h2"], fill=TEXT)
    write_wrapped(d, REC_BODIES[risk], (x0 + 22, y0 + 76), x1 - x0 - 44,
                  FONTS["small"], MUTED, 5)
    img.alpha_composite(overlay)


def draw_stats(img: Image.Image, box: tuple[int, int, int, int], stats: Stats | None,
               running: bool, latency_ms: float | None) -> None:
    x0, y0, x1, y1 = box
    d = ImageDraw.Draw(img)
    panel(d, (x0, y0, x1, y1))
    d.text((x0 + 20, y0 + 14), "PREDICTION STATS", font=FONTS["kicker"], fill=CYAN)
    rows = [
        ("95% interval", f"{stats.ci_low:.1f} – {stats.ci_high:.1f} days" if stats and not running else "—"),
        ("Uncertainty (σ)", f"{stats.std:.2f} days" if stats and not running else "—"),
        ("Monte Carlo runs", "100"),
        ("Weights source", "local artifacts/pg_bnn_demo"),
        ("CPU latency", f"{stats.latency_ms:.0f} ms on 2 threads" if stats and not running else "—"),
    ]
    ry = y0 + 34
    for i, (k, v) in enumerate(rows):
        d.text((x0 + 20, ry + i * 27), k, font=FONTS["small"], fill=MUTED)
        d.text((x1 - 20, ry + i * 27), v, font=FONTS["mono"], fill=TEXT, anchor="ra")


def draw_hist(img: Image.Image, box: tuple[int, int, int, int], hist: Image.Image,
              alpha: float, scale: float) -> None:
    x0, y0, x1, y1 = box
    d = ImageDraw.Draw(img)
    panel(d, (x0, y0, x1, y1))
    d.text((x0 + 20, y0 + 12), "DISTRIBUTION FROM 100 STOCHASTIC RUNS", font=FONTS["kicker"], fill=CYAN)
    if alpha <= 0.01:
        return
    w, h = x1 - x0 - 40, y1 - y0 - 48
    hh = int(h * max(scale, 0.01))
    scaled = hist.resize((w, hh), Image.Resampling.BILINEAR)
    if alpha < 1.0:
        scaled.putalpha(int(255 * alpha))
    img.alpha_composite(scaled, (x0 + 20, y1 - 24 - hh))


def draw_chips(img: Image.Image, box: tuple[int, int, int, int], values: list[str],
               fade: float = 1.0) -> None:
    x0, y0, x1, y1 = box
    d = ImageDraw.Draw(img)
    d.text((x0, y0 - 4), "LIVE TELEMETRY", font=FONTS["kicker"], fill=CYAN)
    cw, ch, gap = 186, 30, 8
    for i, (label, val) in enumerate(zip(CHIP_SHORT_LABELS, values)):
        col, row = i % 3, i // 3
        cx = x0 + col * (cw + gap)
        cy = y0 + 16 + row * (ch + 6)
        d.rounded_rectangle((cx, cy, cx + cw, cy + ch), radius=10,
                            fill=(16, 51, 86, 140), outline="#21476e")
        d.text((cx + 10, cy + 8), label, font=FONTS["tiny"], fill=MUTED)
        d.text((cx + cw - 10, cy + 6), val, font=FONTS["chip"], fill=TEXT, anchor="ra")


def draw_button(img: Image.Image, box: tuple[int, int, int, int], state: str,
                progress: float) -> None:
    x0, y0, x1, y1 = box
    d = ImageDraw.Draw(img)
    if state == "idle":
        d.rounded_rectangle(box, radius=16, fill="#12d5c8", outline="#4db5ff", width=1)
        d.text((x0 + 22, y0 + 15), "▶  Run prediction — 100 Monte Carlo passes",
               font=FONTS["chip"], fill="#07111f")
    elif state == "running":
        d.rounded_rectangle(box, radius=16, fill="#12314f", outline="#21476e", width=1)
        d.text((x0 + 22, y0 + 15), "Running 100 Monte Carlo passes…",
               font=FONTS["chip"], fill=CYAN)
        bar_x0, bar_x1 = x0 + 22, x1 - 22
        bw = int((bar_x1 - bar_x0) * max(0.02, progress))
        d.rounded_rectangle((bar_x0, y1 - 12, bar_x1, y1 - 6), radius=3, fill="#0a1628")
        d.rounded_rectangle((bar_x0, y1 - 12, bar_x0 + bw, y1 - 6), radius=3, fill=TEAL)
    else:  # done
        d.rounded_rectangle(box, radius=16, fill="#0f241f", outline=GREEN, width=1)
        d.text((x0 + 22, y0 + 15), "✓  100 passes completed — live result below",
               font=FONTS["chip"], fill=GREEN)


def draw_footer(img: Image.Image) -> None:
    d = ImageDraw.Draw(img)
    d.line((24, 694, FRAME_W - 24, 694), fill="rgba(124,211,255,45)", width=1)
    write_wrapped(d, "Advisory only — decision-support for maintenance planning. "
                     "AeroVigil never sends control commands to the turbine.",
                  (24, 700), FRAME_W - 48, FONTS["small"], MUTED, 0)


def draw_cursor(img: Image.Image, x: float, y: float, click_t: float | None, now: float) -> None:
    d = ImageDraw.Draw(img)
    if click_t is not None and now - click_t < 0.45:
        p = (now - click_t) / 0.45
        r = int(lerp(10, 46, ease(p)))
        a = int(220 * (1 - p))
        d.ellipse((x - r, y - r, x + r, y + r), outline=f"rgba(255,255,255,{a})", width=3)
    d.polygon([(x, y), (x + 22, y + 7), (x + 14, y + 13), (x + 18, y + 26),
               (x + 11, y + 24), (x + 7, y + 13), (x, y + 19)], fill="#ffffff", outline="#07111f")


def draw_caption(img: Image.Image, text: str, alpha: float) -> None:
    if alpha <= 0.01:
        return
    d = ImageDraw.Draw(img)
    b = d.textbbox((0, 0), text, font=FONTS["body"])
    tw = b[2] - b[0]
    x0, y0 = (FRAME_W - tw) / 2 - 28, 646
    overlay = Image.new("RGBA", (FRAME_W, FRAME_H), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rounded_rectangle((x0, y0, x0 + tw + 56, y0 + 42), radius=21,
                         fill=(10, 22, 40, int(215 * alpha)), outline=(33, 71, 110, int(255 * alpha)))
    od.text((x0 + 28, y0 + 10), text, font=FONTS["body"], fill=f"rgba(238,247,255,{int(255*alpha)})")
    img.alpha_composite(overlay)


def draw_endcard(img: Image.Image, alpha: float) -> None:
    if alpha <= 0.01:
        return
    overlay = Image.new("RGBA", (FRAME_W, FRAME_H), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    d.rectangle((0, 0, FRAME_W, FRAME_H), fill=(8, 17, 31, int(alpha * 250)))
    d.ellipse((880, -120, 1320, 300), fill=(32, 211, 194, int(30 * alpha)))
    d.ellipse((-180, -100, 260, 260), fill=(139, 233, 255, int(22 * alpha)))
    cx = FRAME_W // 2
    d.text((cx, 210), "AEROVIGIL", font=FONTS["kicker"], fill=CYAN, anchor="ma")
    center_text(d, cx, 250, "See the failure before it happens.", FONTS["endtitle"], TEXT)
    center_text(d, cx, 330, "Plan the repair, not the rescue.", FONTS["endsub"], MUTED)
    y = 420
    for chip_txt in ["Physics-guided · ISO 281", "Uncertainty-aware", "Advisory only"]:
        b = d.textbbox((0, 0), chip_txt, font=FONTS["chip"])
        w = b[2] - b[0] + 26
        x0 = cx - w / 2
        d.rounded_rectangle((x0, y, x0 + w, y + 40), radius=20,
                            fill=(32, 211, 194, int(26 * alpha)), outline=(32, 211, 194, int(70 * alpha)))
        d.text((cx, y + 11), chip_txt, font=FONTS["chip"], fill=CYAN, anchor="ma")
        y += 52
    d.text((cx, 590), "aerovigil.abacusai.app", font=FONTS["mono"], fill=TEXT, anchor="ma")
    d.text((cx, 632), "The check-engine light for wind turbines", font=FONTS["small"], fill=MUTED, anchor="ma")
    img.alpha_composite(overlay)


# ---------------------------------------------------------------------------
# Histogram prerender (matches the app: 28 bins + mean/CI/45/14 lines)
# ---------------------------------------------------------------------------
def prerender_hist(stats: Stats) -> Image.Image:
    color, _, _ = RISK_META[stats.risk]
    fig, ax = plt.subplots(figsize=(6.2, 3.3), dpi=170)
    fig.patch.set_facecolor("#0d1b31")
    ax.set_facecolor("#0d1b31")
    ax.hist(stats.preds, bins=28, color=color, edgecolor="#09111f", alpha=0.92)
    ax.axvline(stats.mean, color=CYAN, linewidth=2.6)
    ax.axvline(stats.ci_low, color="#cfd8e3", linestyle="--", linewidth=1.6)
    ax.axvline(stats.ci_high, color="#cfd8e3", linestyle="--", linewidth=1.6)
    ax.axvline(45.0, color=GOLD, linestyle=":", linewidth=2.4)
    ax.axvline(14.0, color=RED, linestyle=":", linewidth=2.4)
    ax.set_xlim(0, 500)
    ax.set_title("Distribution from 100 stochastic runs", color=TEXT, fontsize=11, pad=8)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlabel("Predicted healthy life remaining (days)", color=MUTED, fontsize=9)
    ax.set_ylabel("How often that value appeared", color=MUTED, fontsize=9)
    for sp in ax.spines.values():
        sp.set_color("#35506e")
    fig.tight_layout()
    tmp = Path("/tmp") / f"hist_{stats.name}.png"
    fig.savefig(tmp, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return Image.open(tmp).convert("RGBA")


# ---------------------------------------------------------------------------
# Timeline / state per frame
# ---------------------------------------------------------------------------
@dataclass
class FrameState:
    clock: str
    load_alpha: float
    cursor_pos: tuple[float, float] | None
    click_t: float | None
    active_scenario: str
    gauge_val: float
    gauge_sigma: float
    gauge_color: str
    risk: str
    stats: Stats | None
    stats_running: bool
    hist_img: Image.Image
    hist_alpha: float
    hist_scale: float
    chips: list[str]
    chip_fade: float
    button: str
    button_progress: float
    caption: tuple[str, float]
    endcard: float


def cursor_path(t: float) -> tuple[float, float] | None:
    """Return the cursor position (or None when off-screen) for time t."""
    segs = [
        (7.0, 10.5, (1210, 660), (312, 669)),   # enter -> predict button
        (10.5, 12.0, (312, 669), (312, 669)),   # hover
        (13.0, 21.5, (312, 669), (1000, 95)),   # to warning pill
        (21.5, 23.0, (1000, 95), (1000, 95)),
        (23.8, 24.8, (1000, 95), (312, 669)),   # back to predict
        (24.8, 25.0, (312, 669), (312, 669)),
        (26.5, 33.5, (312, 669), (1091, 95)),   # to critical pill
        (33.5, 34.5, (1091, 95), (1091, 95)),
        (35.3, 36.5, (1091, 95), (312, 669)),
        (36.5, 37.0, (312, 669), (312, 669)),
        (39.0, 46.0, (312, 669), (878, 95)),    # to healthy pill
        (46.0, 48.0, (878, 95), (878, 95)),
        (48.8, 50.0, (878, 95), (312, 669)),
        (50.0, 50.5, (312, 669), (312, 669)),
        (52.0, 55.5, (312, 669), (760, 200)),   # drift away before end card
    ]
    for t0, t1, p0, p1 in segs:
        if t0 <= t <= t1:
            u = ease((t - t0) / (t1 - t0))
            return lerp(p0[0], p1[0], u), lerp(p0[1], p1[1], u)
    return None


def active_click(t: float) -> float | None:
    for ct in CLICK_TIMES:
        if ct - 0.05 <= t <= ct + 0.45:
            return ct
    return None


def frame_state(t: float, stats: dict[str, Stats], hists: dict[str, Image.Image]) -> FrameState:
    # --- page load ------------------------------------------------------
    load_alpha = ease((t - 0.4) / 3.0) if t < T_LOAD else 1.0

    # --- which scenario is "selected" -----------------------------------
    if t < T_SEL_WARN:
        active = "healthy"
    elif t < T_SEL_CRIT:
        active = "warning"
    elif t < T_SEL_HEAL:
        active = "critical"
    else:
        active = "healthy"

    # --- gauge value sweeps ---------------------------------------------
    gauge_start = 280.0
    gauge_end = stats["healthy"].mean
    gauge_val = gauge_start
    gauge_color = GREEN
    risk = "LOW"

    def sweep(val_from, val_to, t0, t1):
        if t >= t0:
            return lerp(val_from, val_to, ease((t - t0) / max(t1 - t0, 1e-6)))
        return val_from

    if t < T_RES1:
        gauge_val = gauge_start
        gauge_color = GREEN
        risk = "LOW"
    elif t < T_RES2:
        gauge_val = sweep(gauge_start, stats["healthy"].mean, T_RES1, T_RES1 + 2.2)
        risk = "LOW"
        gauge_color = GREEN
    elif t < T_RES3:
        gauge_val = sweep(stats["healthy"].mean, stats["warning"].mean, T_RES2, T_RES2 + 2.2)
        u = ease((t - T_RES2) / 2.2) if t < T_RES2 + 2.2 else 1.0
        gauge_color = lerp_color(GREEN, GOLD, u)
        risk = "MODERATE"
    else:
        gauge_val = sweep(stats["warning"].mean, stats["critical"].mean, T_RES3, T_RES3 + 2.0)
        u = ease((t - T_RES3) / 2.0) if t < T_RES3 + 2.0 else 1.0
        gauge_color = lerp_color(GOLD, RED, u)
        risk = "CRITICAL"

    # back-to-healthy sweep at the end (overrides once it starts)
    if t >= T_RES4:
        gauge_val = sweep(stats["critical"].mean, stats["healthy"].mean, T_RES4, T_RES4 + 2.2)
        u = ease((t - T_RES4) / 2.2) if t < T_RES4 + 2.2 else 1.0
        gauge_color = lerp_color(RED, GREEN, u)
        risk = "LOW"

    # --- shown stats / hist / button -------------------------------------
    running = (T_CLICK1 <= t < T_RES1 + 0.3) or (T_CLICK2 <= t < T_RES2 + 0.3) \
        or (T_CLICK3 <= t < T_RES3 + 0.3) or (T_CLICK4 <= t < T_RES4 + 0.3)
    shown: Stats | None = None
    hist_img = hists["healthy"]
    hist_alpha, hist_scale = 0.0, 0.0
    if t >= T_RES1:
        shown = stats["healthy"]
        hist_alpha = 1.0
        hist_scale = 1.0
    if t >= T_RES2 + 1.0:
        shown = stats["warning"]
        u = ease((t - (T_RES2 + 1.0)) / 1.6) if t < T_RES2 + 2.6 else 1.0
        hist_img = hists["warning"]
        hist_alpha = u
        hist_scale = max(0.15, u)
    if t >= T_RES3 + 1.0:
        shown = stats["critical"]
        u = ease((t - (T_RES3 + 1.0)) / 1.5) if t < T_RES3 + 2.5 else 1.0
        hist_img = hists["critical"]
        hist_alpha = u
        hist_scale = max(0.15, u)
    if t >= T_RES4 + 1.0:
        shown = stats["healthy"]
        u = ease((t - (T_RES4 + 1.0)) / 1.6) if t < T_RES4 + 2.6 else 1.0
        hist_img = hists["healthy"]
        hist_alpha = u
        hist_scale = max(0.15, u)
    if running:
        shown = None

    # --- chips ------------------------------------------------------------
    def chips_at(t_from: float, a_from: str, b_from: str, u: float) -> list[str]:
        a = np.array(CHIP_NUMS[a_from], dtype=float)
        b = np.array(CHIP_NUMS[b_from], dtype=float)
        return format_chips((a + (b - a) * u).tolist())

    if t < T_SEL_WARN:
        chips = format_chips(CHIP_NUMS["healthy"])
    elif t < T_SEL_CRIT:
        u = ease((t - T_SEL_WARN) / 1.5) if t < T_SEL_WARN + 1.5 else 1.0
        chips = chips_at(T_SEL_WARN, "healthy", "warning", u)
    elif t < T_SEL_HEAL:
        u = ease((t - T_SEL_CRIT) / 1.5) if t < T_SEL_CRIT + 1.5 else 1.0
        chips = chips_at(T_SEL_CRIT, "warning", "critical", u)
    else:
        u = ease((t - T_SEL_HEAL) / 1.5) if t < T_SEL_HEAL + 1.5 else 1.0
        chips = chips_at(T_SEL_HEAL, "critical", "healthy", u)

    chip_fade = 1.0  # kept for API stability; chips render opaque

    # --- button state ------------------------------------------------------
    button = "idle"
    progress = 0.0
    for click_t, res_t in [(T_CLICK1, T_RES1), (T_CLICK2, T_RES2), (T_CLICK3, T_RES3), (T_CLICK4, T_RES4)]:
        if click_t <= t < click_t + 0.25:
            button = "running"
            progress = 0.05
            break
        if click_t <= t < res_t:
            button = "running"
            progress = min(1.0, (t - click_t - 0.25) / (res_t - click_t - 0.25))
            break
        if res_t <= t < res_t + 1.2:
            button = "done"
            break

    # --- captions ----------------------------------------------------------
    caption: tuple[str, float] = ("", 0.0)
    caps = [
        (17.0, 20.5, "Healthy — about 8 months of runway left"),
        (30.5, 33.5, "Inside the 45-day planning window — schedule the repair"),
        (42.5, 46.5, "Under 14 days — urgent. AeroVigil advises; humans decide."),
        (52.0, 54.2, "Re-evaluated live on every prediction"),
    ]
    for c0, c1, text in caps:
        if c0 <= t < c1:
            a = min(1.0, (t - c0) / 0.5, (c1 - t) / 0.5)
            caption = (text, a)

    # --- end card ----------------------------------------------------------
    endcard = ease((t - T_ENDCARD) / 1.6) if t > T_ENDCARD else 0.0
    if t > T_ENDCARD + 1.6:
        endcard = 1.0

    # --- clock -------------------------------------------------------------
    secs = int(t)
    clock = f"09:41:{secs:02d}"

    sigma = stats[active].std
    # sigma sweeps roughly with gauge
    if t < T_RES1:
        sigma = stats["healthy"].std
    elif t < T_RES2:
        sigma = lerp(stats["healthy"].std, stats["warning"].std, ease((t - T_RES1) / 2.2))
    elif t < T_RES3:
        sigma = lerp(stats["warning"].std, stats["critical"].std, ease((t - T_RES2) / 2.2))
    elif t < T_RES4:
        sigma = stats["critical"].std
    else:
        sigma = lerp(stats["critical"].std, stats["healthy"].std, ease((t - T_RES4) / 2.2))

    return FrameState(
        clock=clock, load_alpha=load_alpha,
        cursor_pos=cursor_path(t), click_t=active_click(t),
        active_scenario=active,
        gauge_val=gauge_val, gauge_sigma=sigma, gauge_color=gauge_color,
        risk=risk, stats=shown, stats_running=running,
        hist_img=hist_img, hist_alpha=hist_alpha, hist_scale=hist_scale,
        chips=chips, chip_fade=chip_fade,
        button=button, button_progress=progress,
        caption=caption, endcard=endcard,
    )


def render_frame(t: float, st: FrameState) -> Image.Image:
    img = canvas()
    if st.load_alpha < 1.0:
        img = img.filter(ImageFilter.GaussianBlur((1 - st.load_alpha) * 2.2))
    d = ImageDraw.Draw(img)
    draw_chrome(img, st.clock, pulse=0.5 + 0.5 * math.sin(t * 4.0))
    draw_hero(img, st.active_scenario, st.load_alpha)
    risk_color, _, _ = RISK_META[st.risk]
    draw_gauge(img, (24, 156, 600, 336), st.gauge_val, st.gauge_sigma,
               st.gauge_color, st.risk, risk_color, fade=st.load_alpha)
    draw_badge(img, (24, 344, 600, 390), st.risk, fade=st.load_alpha)
    draw_rec(img, (24, 398, 600, 556), st.risk)
    draw_chips(img, (24, 562, 600, 644), st.chips, fade=st.chip_fade)
    draw_button(img, (24, 650, 600, 688), st.button, st.button_progress)
    draw_stats(img, (620, 156, 1256, 330), st.stats, st.stats_running, None)
    draw_hist(img, (620, 338, 1256, 648), st.hist_img, st.hist_alpha, st.hist_scale)
    if st.caption[1] > 0.01:
        draw_caption(img, st.caption[0], st.caption[1])
    draw_footer(img)
    if st.cursor_pos is not None:
        draw_cursor(img, st.cursor_pos[0], st.cursor_pos[1], st.click_t, t)
    if st.endcard > 0.01:
        draw_endcard(img, st.endcard)
    return img


# ---------------------------------------------------------------------------
# Audio: narration + UI clicks -> 60.0 s stereo mix
# ---------------------------------------------------------------------------
def build_audio() -> Path:
    sr = 44100
    n = int(DURATION * sr)
    mix = np.zeros((n, 2), dtype=np.float64)

    def place(path: Path, offset: float, gain: float = 0.96) -> None:
        with wave.open(str(path)) as w:
            ch = w.getnchannels()
            data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768.0
        if ch == 1:
            data = np.column_stack([data, data])
        start = int(offset * sr)
        end = min(n, start + len(data))
        seg = data[: end - start]
        # fades (2-D so it broadcasts over stereo channels)
        f = np.ones((len(seg), 1))
        fade_n = int(0.03 * sr)
        f[:fade_n, 0] = np.linspace(0, 1, fade_n)
        f[-fade_n:, 0] = np.linspace(1, 0, fade_n)
        mix[start:end] += seg * f * gain

    def click(offset: float, gain: float = 0.35) -> None:
        t = np.arange(int(0.045 * sr)) / sr
        env = np.exp(-t * 90.0)
        burst = np.random.default_rng(int(offset * 1000)).uniform(-1, 1, len(t))
        sig = burst * env * gain
        start = int(offset * sr)
        end = min(n, start + len(sig))
        mix[start:end, 0] += sig
        mix[start:end, 1] += sig

    place(AUDIO_DIR / "live_intro.wav", 1.4)
    place(AUDIO_DIR / "live_outro.wav", 30.4)
    for ct in CLICK_TIMES:
        click(ct)

    mix = mix / max(1.0, float(np.abs(mix).max())) * 0.9
    pcm = (mix * 32767.0).astype(np.int16)
    with wave.open(str(OUT_AUDIO), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    print(f"  audio mix: {OUT_AUDIO} ({DURATION:.1f}s, stereo 44.1k)")
    return OUT_AUDIO


def main() -> None:
    t0 = time.perf_counter()
    print("Loading model + running live MCVI inference for each scenario...")
    stats = load_stats()
    hists = {name: prerender_hist(s) for name, s in stats.items()}
    print("Rendering frames...")
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(N_FRAMES):
        t = i / FPS
        st = frame_state(t, stats, hists)
        frame = render_frame(t, st)
        frame.convert("RGB").save(FRAMES_DIR / f"frame_{i:05d}.jpg", quality=91)
        if i % 240 == 0:
            print(f"  {i / FPS:5.1f}s / {DURATION:.0f}s")
    audio = build_audio()
    print("Encoding video + muxing audio...")
    cmd = [
        FFMPEG, "-y",
        "-framerate", str(FPS), "-i", str(FRAMES_DIR / "frame_%05d.jpg"),
        "-i", str(audio),
        "-c:v", "libx264", "-preset", "medium", "-crf", "19",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-t", f"{DURATION:.3f}",
        "-movflags", "+faststart",
        str(OUT_VIDEO),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"Done: {OUT_VIDEO}  ({(time.perf_counter() - t0) / 60.0:.1f} min build time)")


if __name__ == "__main__":
    main()
