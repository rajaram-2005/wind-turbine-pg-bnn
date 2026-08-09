"""Regression checks for the canonical one-process, port-8080 deployment."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_cpu_and_gpu_images_ship_console_and_probe_unified_health():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert dockerfile.count("ENV PORT=8080") == 2
    assert dockerfile.count("ENV SERVICE_MODE=all") == 2
    assert dockerfile.count("COPY web_console/ ${APP_HOME}/web_console/") == 2
    assert dockerfile.count("urlopen('http://localhost:${PORT}/health')") == 2
    assert "!artifacts/**/*.pt" in (ROOT / ".dockerignore").read_text(encoding="utf-8")


def test_compose_and_kubernetes_expose_runtime_safety_limits():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    for setting in ("AV_TWIN_MAX_ASSETS=1024", "AV_STREAM_HEURISTIC=1", "AV_MODEL_PATH="):
        assert compose.count(setting) == 2

    configmap = yaml.safe_load((ROOT / "k8s" / "configmap.yaml").read_text(encoding="utf-8"))
    data = configmap["data"]
    for setting in ("AV_TWIN_MAX_ASSETS", "AV_STREAM_HEURISTIC", "AV_MODEL_PATH"):
        assert setting in data
    assert data["AEROVIGIL_STORE_DB"].startswith("/app/data/")
    assert data["AV_JOB_DB"].startswith("/app/data/")


def test_kubernetes_keeps_state_in_one_persistent_runtime():
    deployment = yaml.safe_load((ROOT / "k8s" / "deployment.yaml").read_text(encoding="utf-8"))
    spec = deployment["spec"]
    assert spec["replicas"] == 1
    assert spec["strategy"] == {"type": "Recreate"}

    pod_spec = spec["template"]["spec"]
    app = pod_spec["containers"][0]
    assert app["name"] == "app"
    assert app["ports"][0]["containerPort"] == 8080
    mounts = {mount["name"]: mount["mountPath"] for mount in app["volumeMounts"]}
    assert mounts["data"] == "/app/data"
    assert "/app/artifacts" not in mounts.values()  # do not hide the bundled model
    volumes = {volume["name"]: volume for volume in pod_spec["volumes"]}
    assert volumes["data"]["persistentVolumeClaim"]["claimName"] == "aerovigil-data"

    kustomization = yaml.safe_load(
        (ROOT / "k8s" / "kustomization.yaml").read_text(encoding="utf-8")
    )
    resources = set(kustomization["resources"])
    assert {"serviceaccount.yaml", "pvc.yaml", "deployment.yaml"} <= resources
    assert "hpa.yaml" not in resources
    assert "servicemonitor.yaml" not in resources
    assert not (ROOT / "k8s" / "hpa.yaml").exists()
    assert not (ROOT / "k8s" / "servicemonitor.yaml").exists()


def test_entrypoint_converges_legacy_modes_on_the_unified_app():
    entrypoint = (ROOT / "docker" / "entrypoint.sh").read_text(encoding="utf-8")
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "${PORT:-8080}" in entrypoint
    assert "api|model-api|gradio" in entrypoint
    assert "src.api.app:app" not in entrypoint
    assert "src.aerovigil_pg_bnn.api:app" not in entrypoint
    assert "python3 gradio_app/app.py" not in entrypoint
    assert "src.unified_app:app --host 0.0.0.0" in entrypoint
    assert "src.unified_app:app --host 0.0.0.0 --port 8080" in makefile


def test_legacy_launchers_do_not_open_old_network_ports():
    model_api = (ROOT / "src" / "aerovigil_pg_bnn" / "api.py").read_text(encoding="utf-8")
    gradio = (ROOT / "gradio_app" / "app.py").read_text(encoding="utf-8")
    assert "port=8000" not in model_api
    assert "uvicorn.run(app," not in model_api
    assert "from src.unified_app import app as unified_app" in model_api
    assert "server_port=" not in gradio
    assert "standalone Gradio server is retired" in gradio


def test_dev_extra_resolves_packaging_and_onnx_validation_dependencies():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for dependency in (
        '"packaging>=26.1"',
        '"onnx>=1.17"',
        '"onnxruntime>=1.20"',
        '"onnxscript>=0.2"',
    ):
        assert dependency in pyproject
    cloud_image = (ROOT / "docker" / "Dockerfile.cloud").read_text(encoding="utf-8")
    assert "onnx onnxruntime onnxscript" in cloud_image
