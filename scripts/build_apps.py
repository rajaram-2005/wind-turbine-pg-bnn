#!/usr/bin/env python3
"""Cross-platform build and packaging script for AeroVigil native applications.

Builds the Flutter console for Windows, macOS, Linux, and Android, and
packages the release artifacts (zips, tarballs, and APKs) with SHA256 checksums.

Usage:
    python scripts/build_apps.py --platform all
    python scripts/build_apps.py --platform windows
    python scripts/build_apps.py --platform macos
    python scripts/build_apps.py --platform linux
    python scripts/build_apps.py --platform android
    python scripts/build_apps.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FLUTTER_APP_DIR = REPO_ROOT / "apps" / "aerovigilai_flutter"
DIST_DIR = REPO_ROOT / "dist" / "apps"

TARGET_CONFIGS = {
    "windows": {
        "os": "Windows",
        "platforms": "windows",
        "build_cmd": ["flutter", "build", "windows", "--release"],
        "source_path": FLUTTER_APP_DIR / "build" / "windows" / "x64" / "runner" / "Release",
        "artifact_name": "aerovigil-windows-x64.zip",
        "type": "zip_dir",
    },
    "macos": {
        "os": "Darwin",
        "platforms": "macos",
        "build_cmd": ["flutter", "build", "macos", "--release"],
        "source_path": FLUTTER_APP_DIR
        / "build"
        / "macos"
        / "Build"
        / "Products"
        / "Release"
        / "aerovigilai_flutter.app",
        "artifact_name": "aerovigil-macos-universal.zip",
        "type": "zip_dir",
    },
    "linux": {
        "os": "Linux",
        "platforms": "linux",
        "build_cmd": ["flutter", "build", "linux", "--release"],
        "source_path": FLUTTER_APP_DIR / "build" / "linux" / "x64" / "release" / "bundle",
        "artifact_name": "aerovigil-linux-x64.tar.gz",
        "type": "tar_gz_dir",
    },
    "android": {
        "os": "Any",
        "platforms": "android",
        "build_cmd": ["flutter", "build", "apk", "--release"],
        "source_path": FLUTTER_APP_DIR
        / "build"
        / "app"
        / "outputs"
        / "flutter-apk"
        / "app-release.apk",
        "artifact_name": "aerovigil-android.apk",
        "type": "copy_file",
    },
}


def check_tool(name: str) -> bool:
    """Return True if `name` is executable and available on PATH."""
    return shutil.which(name) is not None


def run_command(cmd: list[str], cwd: Path | None = None, dry_run: bool = False) -> int:
    """Run a shell command or simulate it in dry-run mode."""
    display = " ".join(str(c) for c in cmd)
    work_dir = str(cwd or REPO_ROOT)
    print(f"[{work_dir}] $ {display}")
    if dry_run:
        return 0
    res = subprocess.run(cmd, cwd=work_dir, check=False)
    return res.returncode


def sha256_file(path: Path) -> str:
    """Compute the SHA256 hex digest for a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def package_artifact(
    target_key: str, cfg: dict, output_dir: Path, dry_run: bool = False
) -> Path | None:
    """Package compiled build outputs into the target artifact format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / cfg["artifact_name"]
    src_path: Path = cfg["source_path"]

    if dry_run:
        print(f"  [DRY-RUN] Packaging {src_path} -> {out_file}")
        return out_file

    if not src_path.exists():
        print(f"⚠️ Source build path does not exist: {src_path}", file=sys.stderr)
        return None

    print(f"📦 Packaging {cfg['artifact_name']}...")
    if cfg["type"] == "copy_file":
        shutil.copy2(src_path, out_file)
    elif cfg["type"] == "zip_dir":
        with zipfile.ZipFile(out_file, "w", zipfile.ZIP_DEFLATED) as zf:
            if src_path.is_dir():
                for root, _, files in src_path.walk():
                    for f in files:
                        full_path = root / f
                        arcname = full_path.relative_to(
                            src_path.parent if src_path.name.endswith(".app") else src_path
                        )
                        zf.write(full_path, arcname)
            else:
                zf.write(src_path, src_path.name)
    elif cfg["type"] == "tar_gz_dir":
        with tarfile.open(out_file, "w:gz") as tf:
            tf.add(src_path, arcname=src_path.name)

    print(f"✅ Generated: {out_file} ({out_file.stat().st_size / (1024 * 1024):.2f} MB)")
    return out_file


def build_target(target: str, dry_run: bool = False, output_dir: Path = DIST_DIR) -> bool:
    """Build and package a single target platform."""
    if target not in TARGET_CONFIGS:
        print(f"Unknown target platform: {target}", file=sys.stderr)
        return False

    cfg = TARGET_CONFIGS[target]
    current_os = platform.system()

    print("\n========================================================")
    print(f"🔨 Building AeroVigil Native Console for: {target.upper()}")
    print("========================================================")

    # Check Flutter availability
    if not check_tool("flutter") and not dry_run:
        print("❌ 'flutter' SDK was not found on PATH.", file=sys.stderr)
        print(
            "   Please install Flutter SDK (>= 3.10) from https://flutter.dev/docs/get-started/install",
            file=sys.stderr,
        )
        return False

    # Check OS compatibility for native desktop compilers
    if cfg["os"] != "Any" and cfg["os"] != current_os and not dry_run:
        print(
            f"⚠️ Warning: Building {target} requires a {cfg['os']} host (current host is {current_os})."
        )
        print(
            f"   Use GitHub Actions CI or a {cfg['os']} build agent for native desktop compilation."
        )

    # 1. Generate platform runners
    gen_cmd = [
        "flutter",
        "create",
        f"--platforms={cfg['platforms']}",
        "--org",
        "ai.aerovigil",
        "--project-name",
        "aerovigilai_flutter",
        ".",
    ]
    if run_command(gen_cmd, cwd=FLUTTER_APP_DIR, dry_run=dry_run) != 0:
        print(f"❌ Failed to generate platform runners for {target}", file=sys.stderr)
        return False

    # 2. Pub get
    if run_command(["flutter", "pub", "get"], cwd=FLUTTER_APP_DIR, dry_run=dry_run) != 0:
        print(f"❌ 'flutter pub get' failed for {target}", file=sys.stderr)
        return False

    # 3. Build release
    if run_command(cfg["build_cmd"], cwd=FLUTTER_APP_DIR, dry_run=dry_run) != 0:
        print(f"❌ Release build failed for {target}", file=sys.stderr)
        return False

    # 4. Package artifact
    artifact = package_artifact(target, cfg, output_dir, dry_run=dry_run)
    return artifact is not None


def generate_checksums(output_dir: Path = DIST_DIR, dry_run: bool = False) -> Path:
    """Generate SHA256SUMS.txt for all built artifacts in output_dir."""
    checksum_file = output_dir / "SHA256SUMS.txt"
    if dry_run:
        print(f"  [DRY-RUN] Generate checksums -> {checksum_file}")
        return checksum_file

    lines = []
    for f in sorted(output_dir.iterdir()):
        if f.name != "SHA256SUMS.txt" and f.is_file():
            csum = sha256_file(f)
            lines.append(f"{csum}  {f.name}\n")

    if lines:
        checksum_file.write_text("".join(lines), encoding="utf-8")
        print(f"\n🔐 Wrote {len(lines)} checksums to: {checksum_file}")
    return checksum_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--platform",
        choices=["all", "windows", "macos", "linux", "android"],
        default="all",
        help="Target platform to build (default: all)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DIST_DIR),
        help=f"Directory to write packaged artifacts (default: {DIST_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print build and packaging steps without executing Flutter commands",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    targets = (
        ["windows", "macos", "linux", "android"] if args.platform == "all" else [args.platform]
    )

    success = True
    for t in targets:
        ok = build_target(t, dry_run=args.dry_run, output_dir=out_dir)
        if not ok and not args.dry_run:
            success = False

    generate_checksums(out_dir, dry_run=args.dry_run)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
