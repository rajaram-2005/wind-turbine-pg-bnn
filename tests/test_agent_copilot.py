"""Tests for the restored MIKA + KAI Agent Copilot surface.

Covers the interactive copilot (``POST /api/agent/ask``), the durable human
decision gate (``POST /api/agent/review`` + ``GET /api/agent/reviews``), and
the Scenario Lab parallel-futures comparison (``POST /api/twin/scenarios``) —
all served through the single unified application boundary.
"""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from src.unified_app import create_app  # noqa: E402

_ASSET = "WTG-COP-1"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from src.data.store import reset_store
    from src.jobs.manager import reset_job_manager

    monkeypatch.setenv("AV_STORE_DB", str(tmp_path / "store.sqlite3"))
    monkeypatch.setenv("AV_JOB_DB", str(tmp_path / "jobs.sqlite3"))
    monkeypatch.delenv("AV_MODEL_PATH", raising=False)
    reset_store()
    reset_job_manager()
    with TestClient(create_app(include_dashboard=False)) as c:
        yield c
    reset_store()
    reset_job_manager()


# ------------------------------------------------------------------ agent ask
def test_agent_ask_routes_physics_question_to_kai(client):
    resp = client.post(
        "/api/agent/ask",
        json={"asset_id": _ASSET, "question": "Why is the bearing vibration rising?"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["agent"] == "KAI"
    assert body["asset_id"] == _ASSET
    assert body["answer"]
    assert body["risk_level"] in ("LOW", "MODERATE", "HIGH", "CRITICAL")
    assert "safety_contract" in body["connected_sources"]
    assert body["team"]["team_id"] == "CYBER_PRIME_DUAL_AGENT"
    assert body["advisory_only"] is True


def test_agent_ask_routes_planning_question_to_mika(client):
    resp = client.post(
        "/api/agent/ask",
        json={"asset_id": _ASSET, "question": "When should we plan the inspection crew?"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["agent"] == "MIKA"
    assert "review window" in body["answer"]


def test_agent_ask_general_question_goes_to_council(client):
    resp = client.post(
        "/api/agent/ask",
        json={"asset_id": _ASSET, "question": "Summarise the overall health state."},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["agent"] == "COUNCIL"
    assert "MIKA and KAI agree" in body["answer"]


def test_agent_ask_rejects_empty_question(client):
    resp = client.post("/api/agent/ask", json={"asset_id": _ASSET, "question": ""})
    assert resp.status_code == 422


# ------------------------------------------------------------- human decision gate
def test_agent_review_records_and_returns_trail(client):
    first = client.post(
        "/api/agent/review",
        json={"asset_id": _ASSET, "decision": "Acknowledge evidence"},
    )
    assert first.status_code == 200
    seq1 = first.json()["sequence"]

    second = client.post(
        "/api/agent/review",
        json={
            "asset_id": _ASSET,
            "decision": "Escalate to reliability lead",
            "note": "Vibration trend persists",
        },
    )
    assert second.status_code == 200
    body = second.json()
    assert body["sequence"] == seq1 + 1
    decisions = [item["decision"] for item in body["trail"]]
    assert "Acknowledge evidence" in decisions
    assert "Escalate to reliability lead" in decisions

    listing = client.get("/api/agent/reviews", params={"asset_id": _ASSET}).json()
    assert listing["count"] == 2
    assert listing["reviews"][0]["decision"] == "Escalate to reliability lead"
    assert listing["reviews"][0]["note"] == "Vibration trend persists"


def test_agent_review_rejects_unknown_decision(client):
    resp = client.post(
        "/api/agent/review",
        json={"asset_id": _ASSET, "decision": "Shut down the turbine"},
    )
    assert resp.status_code == 422


# ------------------------------------------------------------------ scenario lab
def test_twin_scenarios_compares_profiles_without_touching_live_twin(client):
    before = client.get("/api/twin/status", params={"asset_id": _ASSET}).json()
    wear_before = before["cumulative_wear"]

    resp = client.post(
        "/api/twin/scenarios",
        json={"asset_id": _ASSET, "hours": 48},
    )
    assert resp.status_code == 200
    body = resp.json()
    profiles = [row["profile"] for row in body["scenarios"]]
    assert profiles == ["nominal", "overload", "derated", "viscosity_loss"]
    assert body["best_profile"] == "derated"
    assert body["worst_profile"] == "overload"
    assert all(row["final_rul_days"] is not None for row in body["scenarios"])
    overload = next(r for r in body["scenarios"] if r["profile"] == "overload")
    assert overload["wear_delta_pct"] > 0

    after = client.get("/api/twin/status", params={"asset_id": _ASSET}).json()
    assert after["cumulative_wear"] == wear_before  # forked twins only


def test_twin_scenarios_rejects_unknown_profile(client):
    resp = client.post(
        "/api/twin/scenarios",
        json={"asset_id": _ASSET, "profiles": ["hyperspeed"]},
    )
    assert resp.status_code == 422


def test_system_stats_counts_reviews(client):
    client.post(
        "/api/agent/review",
        json={"asset_id": _ASSET, "decision": "Request engineering review"},
    )
    stats = client.get("/api/system/stats").json()
    assert stats["tables"]["reviews"] == 1
