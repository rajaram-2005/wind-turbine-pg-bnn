"""Tests for multiplatform build tooling and packaging scripts."""

from __future__ import annotations

from pathlib import Path

import scripts.build_apps as build_apps


def test_target_configs_completeness():
    targets = build_apps.TARGET_CONFIGS
    assert set(targets.keys()) == {"windows", "macos", "linux", "android"}

    for _name, cfg in targets.items():
        assert "platforms" in cfg
        assert "build_cmd" in cfg
        assert "artifact_name" in cfg
        assert "type" in cfg
        assert cfg["artifact_name"].startswith("aerovigil-")


def test_dry_run_build_all(capsys):
    rc = build_apps.main(["--platform", "all", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "WINDOWS" in out
    assert "MACOS" in out
    assert "LINUX" in out
    assert "ANDROID" in out
    assert "aerovigil-windows-x64.zip" in out
    assert "aerovigil-macos-universal.zip" in out
    assert "aerovigil-linux-x64.tar.gz" in out
    assert "aerovigil-android.apk" in out


def test_dry_run_build_single_platform(capsys):
    rc = build_apps.main(["--platform", "windows", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "WINDOWS" in out
    assert "MACOS" not in out


def test_multiplatform_doc_exists():
    repo_root = Path(__file__).resolve().parents[1]
    doc = repo_root / "docs" / "MULTIPLATFORM_RELEASE.md"
    assert doc.is_file()
    content = doc.read_text(encoding="utf-8")
    assert "aerovigil-windows-x64.zip" in content
    assert "aerovigil-macos-universal.zip" in content
    assert "aerovigil-linux-x64.tar.gz" in content
    assert "aerovigil-android.apk" in content
