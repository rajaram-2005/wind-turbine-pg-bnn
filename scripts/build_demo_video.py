"""Build the AeroVigil narrated demo video and README GIF.

Inputs expected:
- Offline weights in artifacts/pg_bnn_demo/
- TTS clips in docs/assets/audio/scene1.wav ... scene8.wav
- Graphics in docs/assets/social-card.png, release-banner.png, social-square.png

Outputs:
- docs/assets/aerovigil-demo.mp4
- docs/assets/demo.gif

The video stills are rendered from code (no browser screenshots). Scenes 4–6 use
real 100-sample MCVI inference on the locally trained demo weights.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import wave
from collections.abc import Iterable
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

FRAME_SIZE = (1280, 720)
FPS = 24
AUDIO_GAP_SECONDS = 0.45
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
ASSETS = ROOT / "docs" / "assets"
AUDIO_DIR = ASSETS / "audio"
FRAMES_DIR = ASSETS / "video_frames"
TEMP_DIR = ASSETS / "video_build"
OUT_VIDEO = ASSETS / "aerovigil-demo.mp4"
OUT_GIF = ASSETS / "demo.gif"
REPO_URL = "github.com/rajaram-2005/wind-turbine-pg-bnn"

SCENARIOS = {
    "healthy": np.array([12.5, 65.0, 80.0, 2000.0, 9.0, 1000.0], dtype=np.float32),
    "critical": np.array([34.0, 118.0, 150.0, 2400.0, 12.0, 78000.0], dtype=np.float32),
}

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


@dataclass(frozen=True)
class Bundle:
    model: PhysicsGuidedBNN
    mean: np.ndarray
    std: np.ndarray
    source: str


@dataclass(frozen=True)
class Stats:
    name: str
    raw: np.ndarray
    mean: float
    std: float
    ci_low: float
    ci_high: float
    risk: str
    predictions: np.ndarray


def risk_for(mean_rul: float) -> str:
    if mean_rul < 14:
        return "CRITICAL"
    if mean_rul < 30:
        return "HIGH"
    if mean_rul < 45:
        return "MODERATE"
    return "LOW"


def style_for(risk: str) -> tuple[str, str]:
    if risk == "CRITICAL":
        return RED, "#2a1118"
    if risk == "HIGH":
        return ORANGE, "#2b1b13"
    if risk == "MODERATE":
        return GOLD, "#2c2611"
    return GREEN, "#0f241f"


def load_bundle() -> Bundle:
    folder = ROOT / "artifacts" / "pg_bnn_demo"
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    scaler = np.load(folder / "scaler.npz")
    mean = scaler["mean"].astype(np.float32)
    std = scaler["std"].astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    model = PhysicsGuidedBNN(config)
    model.load_state_dict(torch.load(folder / "bnn_demo.pt", map_location="cpu", weights_only=True))
    model.eval()
    return Bundle(model=model, mean=mean, std=std, source="local artifacts/pg_bnn_demo")


def preprocess(raw: np.ndarray, bundle: Bundle) -> np.ndarray:
    return ((raw - bundle.mean) / bundle.std).astype(np.float32)


def mc_stats(bundle: Bundle, name: str, raw: np.ndarray, n_samples: int = 100) -> Stats:
    x = preprocess(raw, bundle)
    tensor = torch.tensor([x.tolist()], dtype=torch.float32)
    preds: list[float] = []
    bundle.model.train()
    with torch.no_grad():
        for _ in range(n_samples):
            mean, _ = bundle.model(tensor)
            preds.append(float(mean.squeeze().item()))
    arr = np.array(preds, dtype=np.float32)
    mean_val = float(arr.mean())
    std_val = float(arr.std())
    ci_low, ci_high = np.percentile(arr, [2.5, 97.5])
    return Stats(
        name=name,
        raw=raw,
        mean=mean_val,
        std=std_val,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        risk=risk_for(mean_val),
        predictions=arr,
    )


def font_paths() -> dict[str, Path]:
    ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    return {
        "regular": ttf / "DejaVuSans.ttf",
        "bold": ttf / "DejaVuSans-Bold.ttf",
        "mono": ttf / "DejaVuSansMono.ttf",
    }


def load_fonts() -> dict[str, ImageFont.FreeTypeFont]:
    paths = font_paths()
    return {
        "title": ImageFont.truetype(str(paths["bold"]), 56),
        "h1": ImageFont.truetype(str(paths["bold"]), 44),
        "h2": ImageFont.truetype(str(paths["bold"]), 34),
        "body": ImageFont.truetype(str(paths["regular"]), 26),
        "small": ImageFont.truetype(str(paths["regular"]), 20),
        "chip": ImageFont.truetype(str(paths["bold"]), 22),
        "mono": ImageFont.truetype(str(paths["mono"]), 20),
        "gauge": ImageFont.truetype(str(paths["bold"]), 72),
    }


FONTS = load_fonts()


def rounded_panel(
    img: Image.Image, box: tuple[int, int, int, int], fill: str, outline: str | None = None
) -> None:
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle(box, radius=28, fill=fill, outline=outline, width=2 if outline else 0)
    img.alpha_composite(overlay)


def write_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    width: int,
    font,
    fill: str,
    line_spacing: int = 10,
) -> int:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=font)[2] <= width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += font.size + line_spacing
    return y


def chip(
    draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fg: str = CYAN, bg: str = "#12314f"
) -> int:
    bbox = draw.textbbox((0, 0), text, font=FONTS["chip"])
    width = bbox[2] - bbox[0] + 28
    draw.rounded_rectangle(
        (xy[0], xy[1], xy[0] + width, xy[1] + 42), radius=20, fill=bg, outline=fg
    )
    draw.text((xy[0] + 14, xy[1] + 8), text, font=FONTS["chip"], fill=fg)
    return width


def make_canvas() -> Image.Image:
    img = Image.new("RGBA", FRAME_SIZE, ImageColor.getrgb(BG) + (255,))
    # soft glows
    glow = Image.new("RGBA", FRAME_SIZE, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse((860, -60, 1320, 380), fill=(32, 211, 194, 38))
    gd.ellipse((-120, -80, 360, 320), fill=(139, 233, 255, 28))
    glow = glow.filter(ImageFilter.GaussianBlur(45))
    img.alpha_composite(glow)
    return img


def cover_image(path: Path, size: tuple[int, int]) -> Image.Image:
    src = Image.open(path).convert("RGBA")
    scale = max(size[0] / src.width, size[1] / src.height)
    resized = src.resize(
        (int(src.width * scale), int(src.height * scale)), Image.Resampling.LANCZOS
    )
    left = (resized.width - size[0]) // 2
    top = (resized.height - size[1]) // 2
    return resized.crop((left, top, left + size[0], top + size[1]))


def gauge_card(stats: Stats, subtitle: str) -> Image.Image:
    width, height = 420, 300
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    color, card_bg = style_for(stats.risk)
    draw.rounded_rectangle(
        (0, 0, width - 1, height - 1), radius=30, fill=card_bg, outline=color, width=2
    )
    cx, cy = 150, 150
    outer = 102
    inner = 68
    draw.ellipse((cx - outer, cy - outer, cx + outer, cy + outer), fill="#0c172a")
    draw.arc(
        (cx - outer, cy - outer, cx + outer, cy + outer),
        start=-90,
        end=270,
        fill="#33465f",
        width=16,
    )
    progress = min(max(stats.mean / 365.0, 0.02), 1.0)
    draw.arc(
        (cx - outer, cy - outer, cx + outer, cy + outer),
        start=-90,
        end=-90 + int(progress * 360),
        fill=color,
        width=16,
    )
    draw.ellipse((cx - inner, cy - inner, cx + inner, cy + inner), fill="#07111f")
    days_text = f"{stats.mean:.0f}"
    tw = draw.textbbox((0, 0), days_text, font=FONTS["gauge"])[2]
    draw.text((cx - tw / 2, 106), days_text, font=FONTS["gauge"], fill=TEXT)
    label = "days healthy life"
    lw = draw.textbbox((0, 0), label, font=FONTS["small"])[2]
    draw.text((cx - lw / 2, 184), label, font=FONTS["small"], fill=MUTED)
    sub = f"±{stats.std:.1f} d"
    sw = draw.textbbox((0, 0), sub, font=FONTS["chip"])[2]
    draw.text((cx - sw / 2, 212), sub, font=FONTS["chip"], fill=CYAN)
    draw.text((250, 44), stats.risk, font=FONTS["h2"], fill=color)
    write_wrapped(draw, subtitle, (250, 96), 145, FONTS["small"], MUTED, 6)
    return img


def histogram_image(stats: Stats, title: str) -> Image.Image:
    color, _ = style_for(stats.risk)
    fig, ax = plt.subplots(figsize=(6.0, 3.4), dpi=160)
    fig.patch.set_facecolor("#0d1b31")
    ax.set_facecolor("#0d1b31")
    ax.hist(stats.predictions, bins=26, color=color, edgecolor="#09111f", alpha=0.92)
    ax.axvline(stats.mean, color=CYAN, linewidth=2.6)
    ax.axvline(stats.ci_low, color="#cfd8e3", linestyle="--", linewidth=1.6)
    ax.axvline(stats.ci_high, color="#cfd8e3", linestyle="--", linewidth=1.6)
    ax.axvline(45.0, color=GOLD, linestyle=":", linewidth=2.6)
    ax.axvline(14.0, color=RED, linestyle=":", linewidth=2.6)
    ax.set_title(title, color=TEXT, fontsize=12, pad=10)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlabel("Predicted days", color=MUTED, fontsize=9)
    ax.set_ylabel("Frequency", color=MUTED, fontsize=9)
    for spine in ax.spines.values():
        spine.set_color("#35506e")
    fig.tight_layout()
    out = TEMP_DIR / f"hist_{stats.name}.png"
    fig.savefig(out, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return Image.open(out).convert("RGBA")


def scene1_release_banner() -> Image.Image:
    banner = ASSETS / "release-banner.png"
    if banner.exists():
        return cover_image(banner, FRAME_SIZE)
    img = make_canvas()
    draw = ImageDraw.Draw(img)
    draw.text((70, 120), "AeroVigil", font=FONTS["title"], fill=TEXT)
    write_wrapped(
        draw,
        "Physics-guided AI for 45-day wind turbine early warning",
        (70, 210),
        650,
        FONTS["h2"],
        CYAN,
    )
    return img


def scene2_cost_of_surprise() -> Image.Image:
    img = make_canvas()
    draw = ImageDraw.Draw(img)
    draw.text((70, 64), "The cost of a surprise failure", font=FONTS["h1"], fill=TEXT)
    write_wrapped(
        draw,
        "A hidden bearing problem turns into crane time, specialist labor, and lost power when the turbine fails without warning.",
        (70, 128),
        560,
        FONTS["body"],
        MUTED,
    )
    rounded_panel(img, (68, 254, 540, 612), fill="#101e34", outline="#21476e")
    draw.text((102, 300), "$300k", font=FONTS["title"], fill=RED)
    draw.text((104, 380), "per surprise event", font=FONTS["h2"], fill=TEXT)
    for idx, bullet in enumerate(["giant crane", "specialist crew", "weeks of lost power"]):
        y = 452 + idx * 48
        draw.ellipse((104, y + 12, 118, y + 26), fill=TEAL)
        draw.text((138, y), bullet, font=FONTS["body"], fill=MUTED)
    square = ASSETS / "social-square.png"
    if square.exists():
        sq = cover_image(square, (520, 520))
        sq = sq.filter(ImageFilter.GaussianBlur(0.5))
        img.alpha_composite(sq, (690, 96))
        overlay = Image.new("RGBA", FRAME_SIZE, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rounded_rectangle((690, 96, 1210, 616), radius=30, outline="#305a84", width=2)
        img.alpha_composite(overlay)
    return img


def scene3_vitals_panel() -> Image.Image:
    img = make_canvas()
    draw = ImageDraw.Draw(img)
    draw.text((70, 58), "Read the turbine's vital signs", font=FONTS["h1"], fill=TEXT)
    write_wrapped(
        draw,
        "No new hardware: AeroVigil reads six standard telemetry signals and turns them into an advisory.",
        (70, 120),
        660,
        FONTS["body"],
        MUTED,
    )
    rounded_panel(img, (84, 210, 1190, 590), fill="#0f1f36", outline="#23547d")
    rows = [
        ("Vibration", "12.5 mm/s", 0.46),
        ("Bearing temperature", "65 °C", 0.52),
        ("Generator temperature", "80 °C", 0.56),
        ("Power output", "2000 kW", 0.62),
        ("Wind speed", "9.0 m/s", 0.44),
        ("Lifetime run hours", "1,000 h", 0.18),
    ]
    for idx, (label, value, frac) in enumerate(rows):
        y = 246 + idx * 52
        draw.text((126, y), label, font=FONTS["body"], fill=TEXT)
        draw.text((480, y), value, font=FONTS["mono"], fill=CYAN)
        draw.rounded_rectangle((710, y + 4, 1080, y + 28), radius=12, fill="#10253d")
        draw.rounded_rectangle((710, y + 4, 710 + int(370 * frac), y + 28), radius=12, fill=TEAL)
    draw.rounded_rectangle((870, 522, 1120, 572), radius=18, fill=TEAL)
    draw.text((918, 534), "Predict", font=FONTS["chip"], fill="#08101e")
    chip(draw, (110, 524), "scenario preset")
    return img


def scene4_or_5(stats: Stats, title: str, subtitle: str) -> Image.Image:
    img = make_canvas()
    draw = ImageDraw.Draw(img)
    color, _ = style_for(stats.risk)
    draw.text((68, 52), title, font=FONTS["h1"], fill=TEXT)
    write_wrapped(draw, subtitle, (70, 112), 560, FONTS["body"], MUTED)
    gauge = gauge_card(stats, f"95% range {stats.ci_low:.0f}–{stats.ci_high:.0f} days")
    img.alpha_composite(gauge, (70, 230))
    hist = histogram_image(stats, "100 stochastic runs")
    hist = cover_image(TEMP_DIR / f"hist_{stats.name}.png", (620, 360))
    rounded_panel(img, (620, 216, 1230, 596), fill="#0e1a30", outline="#244b73")
    img.alpha_composite(hist, (625, 220))
    draw.rounded_rectangle((70, 632, 1180, 680), radius=18, fill="#0f2138")
    draw.text(
        (98, 644),
        f"Mean {stats.mean:.1f} d   ·   ±{stats.std:.1f} d   ·   45-day line = schedule before rescue",
        font=FONTS["body"],
        fill=color,
    )
    return img


def scene6_distribution(stats: Stats) -> Image.Image:
    img = make_canvas()
    draw = ImageDraw.Draw(img)
    draw.text((70, 60), "The spread is the AI's honesty", font=FONTS["h1"], fill=TEXT)
    write_wrapped(
        draw,
        "AeroVigil does not make one magical guess. It runs the model one hundred times and shows the spread.",
        (70, 122),
        700,
        FONTS["body"],
        MUTED,
    )
    hist = cover_image(TEMP_DIR / f"hist_{stats.name}.png", (860, 430))
    rounded_panel(img, (68, 216, 942, 650), fill="#0e1a30", outline="#244b73")
    img.alpha_composite(hist, (76, 224))
    rounded_panel(img, (976, 216, 1212, 650), fill="#101f36", outline="#284869")
    draw.text((1008, 262), "Why it matters", font=FONTS["h2"], fill=CYAN)
    bullets = [
        "wider spread = less certainty",
        "yellow 45-day line = planning trigger",
        "left of the line = act before rescue",
        f"critical sample mean = {stats.mean:.1f} days",
    ]
    for idx, bullet in enumerate(bullets):
        y = 330 + idx * 70
        draw.ellipse((1006, y + 10, 1020, y + 24), fill=GOLD if idx < 3 else RED)
        write_wrapped(draw, bullet, (1038, y), 150, FONTS["small"], MUTED, 6)
    return img


def scene7_safety() -> Image.Image:
    img = make_canvas()
    draw = ImageDraw.Draw(img)
    draw.text((70, 58), "Advisory only", font=FONTS["h1"], fill=TEXT)
    write_wrapped(
        draw,
        "The system never takes control. A fail-closed safety contract screens outputs before they leave the boundary.",
        (70, 122),
        720,
        FONTS["body"],
        MUTED,
    )
    rounded_panel(img, (74, 224, 1180, 610), fill="#0f1f36", outline="#264d75")
    # padlock icon
    draw.rounded_rectangle((176, 328, 404, 548), radius=26, fill="#132848", outline=CYAN, width=3)
    draw.arc((210, 248, 370, 410), start=200, end=-20, fill=CYAN, width=16)
    draw.rectangle((252, 410, 326, 468), fill=CYAN)
    draw.ellipse((281, 448, 297, 464), fill="#132848")
    draw.text((500, 300), "ADVISORY ONLY", font=FONTS["title"], fill=TEXT)
    for idx, line in enumerate(
        [
            "No control commands",
            "No LOTO / part-number instructions",
            "Humans decide the maintenance action",
        ]
    ):
        y = 392 + idx * 66
        draw.rounded_rectangle((506, y + 2, 528, y + 24), radius=6, fill=TEAL)
        draw.text((548, y - 8), line, font=FONTS["body"], fill=MUTED)
    return img


def scene8_scoreboard() -> Image.Image:
    img = make_canvas()
    draw = ImageDraw.Draw(img)
    draw.text((70, 62), "AeroVigil", font=FONTS["title"], fill=TEXT)
    write_wrapped(
        draw,
        "Open-source, offline-capable demo package for 45-day early warning.",
        (72, 134),
        620,
        FONTS["body"],
        MUTED,
    )
    cards = [
        ("94.2%", "early-warning accuracy", TEAL),
        ("100%", "recall", CYAN),
        ("±4–6 d", "typical error band", GOLD),
        ("45 days", "planning horizon", RED),
    ]
    for idx, (big, label, color) in enumerate(cards):
        x = 70 + (idx % 2) * 290
        y = 262 + (idx // 2) * 160
        rounded_panel(img, (x, y, x + 250, y + 122), fill="#0f1f36", outline=color)
        draw.text((x + 24, y + 20), big, font=FONTS["h1"], fill=color)
        draw.text((x + 24, y + 76), label, font=FONTS["small"], fill=MUTED)
    rounded_panel(img, (720, 222, 1186, 566), fill="#10233f", outline="#2d5278")
    draw.text((754, 270), "Repo", font=FONTS["h2"], fill=CYAN)
    write_wrapped(draw, REPO_URL, (754, 334), 360, FONTS["body"], TEXT)
    widths = []
    cursor = 754
    for txt in ["physics-guided", "uncertainty", "advisory-only"]:
        w = chip(draw, (cursor, 444), txt)
        widths.append(w)
        cursor += w + 12
    draw.text((754, 518), "See the failure before it happens.", font=FONTS["body"], fill=MUTED)
    return img


def save_scene(scene_number: int, image: Image.Image) -> Path:
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    out = FRAMES_DIR / f"scene{scene_number}.png"
    image.convert("RGB").save(out, quality=95)
    return out


def audio_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / float(wav.getframerate())


def concat_audio(clips: Iterable[Path], out_path: Path, gap_seconds: float) -> list[float]:
    clips = list(clips)
    durations: list[float] = []
    with wave.open(str(clips[0]), "rb") as first:
        params = first.getparams()
        frames_per_second = first.getframerate()
        sample_width = first.getsampwidth()
        channels = first.getnchannels()
    silence = b"\x00" * int(frames_per_second * gap_seconds) * sample_width * channels

    with wave.open(str(out_path), "wb") as out_wav:
        out_wav.setparams(params)
        for idx, clip in enumerate(clips):
            with wave.open(str(clip), "rb") as wav:
                if wav.getparams()[:3] != params[:3]:
                    raise RuntimeError(f"Audio clip mismatch: {clip}")
                clip_frames = wav.readframes(wav.getnframes())
                out_wav.writeframes(clip_frames)
                clip_duration = wav.getnframes() / float(wav.getframerate())
                durations.append(clip_duration + (gap_seconds if idx < len(clips) - 1 else 0.0))
            if idx < len(clips) - 1:
                out_wav.writeframes(silence)
    return durations


def run_ffmpeg(args: list[str]) -> None:
    cmd = [FFMPEG, *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")


def make_zoom_segment(image_path: Path, duration: float, out_path: Path) -> None:
    frames = max(1, int(math.ceil(duration * FPS)))
    vf = (
        f"scale=1920:1080,"
        f"zoompan=z='1+0.06*on/{frames}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s=1280x720:fps={FPS}"
    )
    run_ffmpeg(
        [
            "-y",
            "-loop",
            "1",
            "-i",
            str(image_path),
            "-vf",
            vf,
            "-r",
            str(FPS),
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-t",
            f"{duration:.3f}",
            str(out_path),
        ]
    )


def concat_video_segments(segments: list[Path], out_path: Path) -> None:
    concat_list = TEMP_DIR / "segments.txt"
    concat_list.write_text("".join(f"file '{p.resolve()}'\n" for p in segments), encoding="utf-8")
    run_ffmpeg(
        ["-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", str(out_path)]
    )


def mux_audio(video_path: Path, audio_path: Path, out_path: Path) -> None:
    run_ffmpeg(
        [
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(out_path),
        ]
    )


def build_gif(scene4: Path, scene5: Path, out_gif: Path) -> None:
    gif_source = TEMP_DIR / "gif_source.mp4"
    concat_video_segments([scene4, scene5], gif_source)
    palette = TEMP_DIR / "palette.png"
    run_ffmpeg(
        [
            "-y",
            "-i",
            str(gif_source),
            "-vf",
            "fps=10,scale=640:-1:flags=lanczos,palettegen",
            str(palette),
        ]
    )
    run_ffmpeg(
        [
            "-y",
            "-i",
            str(gif_source),
            "-i",
            str(palette),
            "-lavfi",
            "fps=10,scale=640:-1:flags=lanczos[x];[x][1:v]paletteuse",
            str(out_gif),
        ]
    )


def main() -> int:
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    clips = [AUDIO_DIR / f"scene{i}.wav" for i in range(1, 9)]
    missing = [str(p) for p in clips if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing TTS clip(s): {missing}")

    bundle = load_bundle()
    healthy = mc_stats(bundle, "healthy", SCENARIOS["healthy"], n_samples=100)
    critical = mc_stats(bundle, "critical", SCENARIOS["critical"], n_samples=100)
    print(
        f"healthy: mean={healthy.mean:.2f} std={healthy.std:.2f} ci=({healthy.ci_low:.2f}, {healthy.ci_high:.2f})"
    )
    print(
        f"critical: mean={critical.mean:.2f} std={critical.std:.2f} ci=({critical.ci_low:.2f}, {critical.ci_high:.2f})"
    )

    scenes = [
        scene1_release_banner(),
        scene2_cost_of_surprise(),
        scene3_vitals_panel(),
        scene4_or_5(
            healthy,
            "Healthy machine",
            f"Real MCVI inference on the healthy preset. Mean = {healthy.mean:.1f} days.",
        ),
        scene4_or_5(
            critical,
            "Critical machine",
            f"Real MCVI inference on the critical preset. Mean = {critical.mean:.1f} days.",
        ),
        scene6_distribution(critical),
        scene7_safety(),
        scene8_scoreboard(),
    ]

    scene_paths = [save_scene(idx + 1, scene) for idx, scene in enumerate(scenes)]
    audio_track = TEMP_DIR / "narration.wav"
    durations = concat_audio(clips, audio_track, AUDIO_GAP_SECONDS)

    segment_paths: list[Path] = []
    for idx, (scene_path, duration) in enumerate(zip(scene_paths, durations), start=1):
        segment_path = TEMP_DIR / f"segment{idx}.mp4"
        make_zoom_segment(scene_path, duration, segment_path)
        segment_paths.append(segment_path)

    stitched = TEMP_DIR / "stitched.mp4"
    concat_video_segments(segment_paths, stitched)
    mux_audio(stitched, audio_track, OUT_VIDEO)
    build_gif(segment_paths[3], segment_paths[4], OUT_GIF)

    print(f"wrote {OUT_VIDEO.relative_to(ROOT)}")
    print(f"wrote {OUT_GIF.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
