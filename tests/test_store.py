"""Tests for the durable SQLite operational store (src.data.store)."""

from src.data.store import Store


def test_telemetry_roundtrip(tmp_path):
    store = Store(tmp_path / "store.sqlite3")
    n = store.record_telemetry(
        [
            {
                "gateway_id": "gw-1",
                "turbine_id": "WTG-1",
                "signal": "rpm",
                "value": 1500.0,
                "unit": "rpm",
                "timestamp": "2026-08-08T10:00:00Z",
            },
            {
                "gateway_id": "gw-1",
                "turbine_id": "WTG-1",
                "signal": "temp",
                "value": 68.0,
                "unit": "C",
                "timestamp": "2026-08-08T10:00:00Z",
            },
        ]
    )
    assert n == 2
    rows = store.latest_telemetry(10)
    assert len(rows) == 2
    assert rows[0]["signal"] == "temp"  # newest first
    assert store.telemetry_count() == 2
    filtered = store.latest_telemetry(10, signal="rpm")
    assert len(filtered) == 1 and filtered[0]["value"] == 1500.0


def test_assets_upsert_and_summary(tmp_path):
    store = Store(tmp_path / "store.sqlite3")
    store.upsert_asset(
        {
            "turbine_id": "WTG-1",
            "status": "Healthy",
            "health_score": 90.0,
            "predicted_rul_days": 250.0,
        }
    )
    store.upsert_asset(
        {"turbine_id": "WTG-1", "status": "Watch", "health_score": 70.0, "predicted_rul_days": 60.0}
    )
    store.upsert_asset(
        {"turbine_id": "WTG-2", "status": "Alert", "health_score": 30.0, "predicted_rul_days": 20.0}
    )
    summary = store.summarize_fleet()
    assert summary["n_assets"] == 2
    assert summary["at_risk_count"] == 2  # both RUL < 104d
    assert summary["mean_rul_days"] == 40.0
    # WTG-1 row was upserted, not duplicated.
    ids = {a["turbine_id"] for a in store.get_assets()}
    assert ids == {"WTG-1", "WTG-2"}


def test_twin_states_reports_imports(tmp_path):
    store = Store(tmp_path / "store.sqlite3")
    store.record_twin_state("WTG-1", {"timestamp": "t1", "cumulative_wear": 0.1})
    store.record_twin_state("WTG-1", {"timestamp": "t2", "cumulative_wear": 0.2})
    assert store.latest_twin_state("WTG-1")["cumulative_wear"] == 0.2
    assert len(store.twin_history("WTG-1")) == 2

    store.record_report("fleet", "# Fleet", title="F1", meta={"n": 1})
    store.record_report("fleet", "# Fleet 2", title="F2", meta={"n": 2})
    latest = store.latest_report("fleet")
    assert latest["title"] == "F2" and latest["meta"]["n"] == 2

    imp = store.record_import("a.csv", "text/csv", 123, "cloud")
    assert imp >= 1
    assert store.list_imports()[0]["source"] == "cloud"

    stats = store.stats()
    assert stats["tables"]["telemetry"] == 0
    assert stats["tables"]["twin_states"] == 2
    assert stats["tables"]["reports"] == 2
    assert stats["tables"]["imports"] == 1
