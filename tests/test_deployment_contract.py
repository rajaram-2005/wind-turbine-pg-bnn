"""Regression checks for the canonical one-process, port-8080 deployment."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cpu_and_gpu_images_ship_console_and_probe_unified_health():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert dockerfile.count("ENV PORT=8080") == 2
    assert dockerfile.count("COPY web_console/ ${APP_HOME}/web_console/") == 2
    assert dockerfile.count("urlopen('http://localhost:${PORT}/health')") == 2
    assert "!artifacts/**/*.pt" in (ROOT / ".dockerignore").read_text(encoding="utf-8")


def test_compose_and_kubernetes_expose_runtime_safety_limits():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    for setting in ("AV_TWIN_MAX_ASSETS=1024", "AV_STREAM_HEURISTIC=1", "AV_MODEL_PATH="):
        assert compose.count(setting) == 2

    configmap = (ROOT / "k8s" / "configmap.yaml").read_text(encoding="utf-8")
    for setting in ("AV_TWIN_MAX_ASSETS", "AV_STREAM_HEURISTIC", "AV_MODEL_PATH"):
        assert setting in configmap


def test_entrypoint_and_task_runner_default_to_port_8080():
    entrypoint = (ROOT / "docker" / "entrypoint.sh").read_text(encoding="utf-8")
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "${PORT:-8080}" in entrypoint
    assert "src.unified_app:app --host 0.0.0.0 --port 8080" in makefile


def test_dev_extra_resolves_twine_packaging_floor():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"packaging>=26.1"' in pyproject
