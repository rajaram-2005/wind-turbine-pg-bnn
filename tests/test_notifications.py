"""Tests for email health reports and severity-based alerts (src/notifications)."""

from datetime import datetime, timedelta, timezone

import pytest

from src.faults.detector import FaultDetector
from src.notifications import (
    AlertTracker,
    EmailNotifier,
    NotificationConfig,
    render_alert_html,
    render_health_report_html,
)

HEALTHY = {
    "vibration_mms": 2.1,
    "temperature_c": 58.0,
    "rpm": 1400.0,
    "oil_viscosity_cst": 32.0,
    "load_pct": 70.0,
}

FAULTY = {
    **HEALTHY,
    "oil_viscosity_cst": 5.0,
    "oil_water_ppm": 1500.0,
    "vibration_mms": 9.0,
    "yaw_error_deg": 25.0,
    "oil_temp_c": 125.0,  # gearbox oil fire range
    "smoke_detector_on": True,
}


@pytest.fixture()
def eml_dir(tmp_path):
    return tmp_path / "notifications"


@pytest.fixture()
def notifier(eml_dir):
    config = NotificationConfig(
        mode="eml",
        alert_recipients=("ops@example.com",),
        report_recipients=("maintenance@example.com",),
        artifact_dir=eml_dir,
    )
    return EmailNotifier(config=config, tracker=AlertTracker(eml_dir / "state.json"))


# --------------------------------------------------------------------------- #
# Rendering                                                                    #
# --------------------------------------------------------------------------- #
def test_render_alert_html_contains_faults_and_actions():
    report = FaultDetector().detect(FAULTY, asset_id="WTG-A", timestamp="2026-08-17T00:00:00Z")
    html = render_alert_html(report)
    assert "AeroVigil Fault Alert" in html
    assert report.asset_id in html
    assert "GB-02" in html  # fault id rendered
    assert "Oil viscosity too low" in html
    assert "Recommended action" in html
    assert "never sends commands" in html  # advisory-only banner


def test_render_health_report_html_covers_multiple_assets():
    healthy = FaultDetector().detect(HEALTHY, asset_id="WTG-OK")
    faulty = FaultDetector().detect(FAULTY, asset_id="WTG-BAD")
    html = render_health_report_html([healthy, faulty], "Nightly fleet digest")
    assert "AeroVigil Health Report" in html
    assert "WTG-OK" in html and "WTG-BAD" in html
    assert "Nightly fleet digest" in html


# --------------------------------------------------------------------------- #
# Delivery                                                                     #
# --------------------------------------------------------------------------- #
def test_eml_fallback_writes_standard_email_file(notifier, eml_dir):
    result = notifier.deliver("Test subject", "<p>html</p>", "text body")
    assert result.channel == "eml"
    assert result.delivered is True
    files = list(eml_dir.glob("*.eml"))
    assert len(files) == 1
    content = files[0].read_text(encoding="utf-8")
    assert "Subject: Test subject" in content
    assert "ops@example.com" in content
    assert "<p>html</p>" in content


def test_no_recipients_skips_delivery(eml_dir):
    config = NotificationConfig(mode="eml", artifact_dir=eml_dir)
    notifier = EmailNotifier(config=config)
    result = notifier.deliver("subject", "<p>x</p>", "text")
    assert result.channel == "skipped"
    assert result.delivered is False
    assert "no recipients" in result.detail


def test_off_mode_sends_nothing(eml_dir):
    config = NotificationConfig(
        mode="off",
        artifact_dir=eml_dir,
        alert_recipients=("a@b.c",),
    )
    notifier = EmailNotifier(config=config)
    report = FaultDetector().detect(FAULTY, asset_id="WTG-A")
    assert notifier.process_report(report) == []


# --------------------------------------------------------------------------- #
# process_report: alerts only for severe, dedupe + escalation                  #
# --------------------------------------------------------------------------- #
def test_process_report_alerts_critical_and_high(notifier):
    report = FaultDetector().detect(FAULTY, asset_id="WTG-A")
    results = notifier.process_report(report)
    assert results, "expected immediate alerts for CRITICAL/HIGH faults"
    subjects = " | ".join(r.subject for r in results)
    assert "CRITICAL" in subjects  # GB-15 gearbox oil fire
    assert all(r.delivered for r in results)


def test_healthy_snapshot_sends_no_alerts(notifier):
    report = FaultDetector().detect(HEALTHY, asset_id="WTG-OK")
    assert notifier.process_report(report) == []


def test_dedupe_cooldown_suppresses_repeats(notifier):
    report = FaultDetector().detect(FAULTY, asset_id="WTG-A")
    first = notifier.process_report(report)
    assert first
    second = notifier.process_report(report)  # same snapshot, same second
    assert second == []
    # After the cooldown window elapses the alert may fire again.
    notifier.tracker._state = {  # simulate elapsed time by rewriting state
        k: {**v, "last_sent": (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()}
        for k, v in notifier.tracker._state.items()
    }
    third = notifier.process_report(report)
    assert third


def test_escalation_fires_immediately(notifier):
    mild = dict(FAULTY)
    mild["oil_viscosity_cst"] = 9.0  # WARN/MEDIUM only for GB-02
    mild.pop("smoke_detector_on")
    mild["oil_temp_c"] = 88.0
    report = FaultDetector().detect(mild, asset_id="WTG-A")
    notifier.process_report(report)  # seed tracker with medium alerts
    severe = FaultDetector().detect(FAULTY, asset_id="WTG-A")  # now CRITICAL
    results = notifier.process_report(severe)
    assert results, "severity escalation must alert immediately despite cooldown"


def test_tracker_persists_across_instances(tmp_path):
    path = tmp_path / "state.json"
    tracker = AlertTracker(path)
    assert tracker.should_send("WTG-1", "GB-02", "HIGH")
    tracker.record("WTG-1", "GB-02", "HIGH")
    assert not tracker.should_send("WTG-1", "GB-02", "HIGH")
    tracker2 = AlertTracker(path)  # reload from disk
    assert not tracker2.should_send("WTG-1", "GB-02", "HIGH")
    # Different asset or fault is unaffected.
    assert tracker2.should_send("WTG-2", "GB-02", "HIGH")
    assert tracker2.should_send("WTG-1", "GB-04", "HIGH")


def test_send_health_report_digest(notifier, eml_dir):
    healthy = FaultDetector().detect(HEALTHY, asset_id="WTG-OK")
    faulty = FaultDetector().detect(FAULTY, asset_id="WTG-BAD")
    result = notifier.send_health_report(
        [healthy, faulty], title="Daily digest", recipients=("ops@example.com",)
    )
    assert result is not None
    assert result.delivered is True
    assert "Daily digest" in result.subject
    files = list(eml_dir.glob("*.eml"))
    assert len(files) == 1
    assert "WTG-BAD" in files[0].read_text(encoding="utf-8")


def test_status_reveals_no_secrets(notifier):
    status = notifier.status()
    assert status["mode"] == "eml"
    assert status["alert_recipients"] == ["ops@example.com"]
    assert "smtp_password" not in status
    assert status["advisory_only"] is True


# --------------------------------------------------------------------------- #
# Environment-driven config                                                    #
# --------------------------------------------------------------------------- #
def test_config_from_env(monkeypatch):
    monkeypatch.setenv("AV_NOTIFY_MODE", "smtp")
    monkeypatch.setenv("AV_SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("AV_SMTP_PORT", "2525")
    monkeypatch.setenv("AV_SMTP_FROM", "aero@example.com")
    monkeypatch.setenv("AV_ALERT_RECIPIENTS", "a@example.com; b@example.com")
    monkeypatch.setenv("AV_REPORT_RECIPIENTS", "c@example.com")
    config = NotificationConfig.from_env()
    assert config.effective_mode == "smtp"
    assert config.smtp_host == "smtp.example.com"
    assert config.smtp_port == 2525
    assert config.alert_recipients == ("a@example.com", "b@example.com")
    assert config.report_recipients == ("c@example.com",)


def test_auto_mode_falls_back_to_eml(monkeypatch, eml_dir):
    monkeypatch.delenv("AV_SMTP_HOST", raising=False)
    config = NotificationConfig.from_env(artifact_dir=eml_dir)
    assert config.effective_mode == "eml"


# --------------------------------------------------------------------------- #
# Digital-twin integration                                                     #
# --------------------------------------------------------------------------- #
def test_twin_update_state_alerts_via_notifier(eml_dir):
    from src.digital_twin.specs import get_spec
    from src.digital_twin.twin import WindTurbineDigitalTwin
    from src.utils.schema import Telemetry

    config = NotificationConfig(
        mode="eml",
        alert_recipients=("ops@example.com",),
        artifact_dir=eml_dir,
    )
    twin = WindTurbineDigitalTwin(
        "WTG-TWN", get_spec("NREL-5MW"), notifier=EmailNotifier(config=config)
    )
    healthy = Telemetry(
        vibration_mms=2.0,
        temperature_c=55.0,
        rpm=950.0,
        oil_viscosity_cst=32.0,
        load_pct=70.0,
    )
    rec = twin.update_state(healthy)
    assert rec["notifications"] == []
    degraded = Telemetry(
        vibration_mms=9.0,
        temperature_c=60.0,
        rpm=1000.0,
        oil_viscosity_cst=5.0,
        load_pct=90.0,
    )
    rec2 = twin.update_state(degraded)
    assert rec2["notifications"], "twin must email alerts on severe faults"
    assert all(n["delivered"] for n in rec2["notifications"])
    assert any("CRITICAL" in n["subject"] or "HIGH" in n["subject"] for n in rec2["notifications"])
    # Repeating the same state does not re-alert (dedupe).
    rec3 = twin.update_state(degraded)
    assert rec3["notifications"] == []


# --------------------------------------------------------------------------- #
# Alert workflow: acknowledge / resolve                                        #
# --------------------------------------------------------------------------- #
def test_acknowledge_stops_realerting(tmp_path):
    tracker = AlertTracker(tmp_path / "state.json")
    tracker.record("WTG-1", "GB-02", "HIGH")
    assert tracker.acknowledge("WTG-1", "GB-02", operator="ops-1") is True
    assert not tracker.should_send("WTG-1", "GB-02", "HIGH")
    # Escalation still breaks through an acknowledgement.
    assert tracker.should_send("WTG-1", "GB-02", "CRITICAL")
    assert tracker.acknowledge("WTG-X", "GB-02") is False  # unknown


def test_resolve_clears_and_allows_fresh_alert(tmp_path):
    tracker = AlertTracker(tmp_path / "state.json")
    tracker.record("WTG-1", "GB-02", "HIGH")
    assert tracker.resolve("WTG-1", "GB-02", operator="crew-7") is True
    # Resolved means a brand-new detection alerts again.
    assert tracker.should_send("WTG-1", "GB-02", "HIGH")
    assert tracker.open_alerts() == []
    assert len(tracker.to_dict()) == 1  # audit trail kept


def test_open_alerts_lists_unresolved_with_state(tmp_path):
    tracker = AlertTracker(tmp_path / "state.json")
    tracker.record("WTG-1", "GB-02", "HIGH")
    tracker.record("WTG-1", "RB-07", "CRITICAL")
    tracker.acknowledge("WTG-1", "GB-02", operator="ops")
    open_alerts = tracker.open_alerts()
    by_fault = {a["fault_id"]: a for a in open_alerts}
    assert set(by_fault) == {"GB-02", "RB-07"}
    assert by_fault["GB-02"]["acknowledged"] is True
    assert by_fault["RB-07"]["acknowledged"] is False
    assert by_fault["RB-07"]["severity"] == "CRITICAL"


# --------------------------------------------------------------------------- #
# Webhook alerts                                                               #
# --------------------------------------------------------------------------- #
def test_detect_format():
    from src.notifications.webhooks import detect_format

    assert detect_format("https://hooks.slack.com/services/T00/B00/xxx") == "slack"
    assert detect_format("https://outlook.office.com/webhook/abc") == "teams"
    assert detect_format("https://example.com/hook") == "generic"


def test_build_payload_formats():
    from src.notifications.webhooks import build_payload

    report = FaultDetector().detect(FAULTY, asset_id="WTG-A")
    slack = build_payload(report, "slack", "SUBJ")
    assert slack["attachments"][0]["color"] in ("danger", "warning")
    teams = build_payload(report, "teams", "SUBJ")
    assert teams["attachments"][0]["content"]["type"] == "AdaptiveCard"
    generic = build_payload(report, "generic", "SUBJ")
    assert generic["subject"] == "SUBJ"
    assert generic["advisory_only"] is True
    assert generic["faults"][0]["fault_id"] == report.faults[0].fault_id


def _local_webhook_server():
    """Tiny HTTP server capturing POSTed webhook payloads."""
    import json as _json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    received: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            received.append(
                {
                    "path": self.path,
                    "body": _json.loads(raw.decode("utf-8")),
                    "content_type": self.headers.get("Content-Type"),
                }
            )
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args):  # silence
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, received, thread


def test_webhook_process_report_posts_slack_payload():
    from src.notifications.webhooks import WebhookNotifier

    server, received, thread = _local_webhook_server()
    try:
        url = f"http://127.0.0.1:{server.server_port}/hook"
        tracker = AlertTracker()
        notifier = WebhookNotifier(urls=[url], tracker=tracker)
        report = FaultDetector().detect(FAULTY, asset_id="WTG-A")
        results = notifier.process_report(report)
        assert results, "expected webhook deliveries for CRITICAL/HIGH"
        assert all(r.delivered for r in results)
        assert any(r.format == "generic" for r in results)  # local URL
        assert received, "server must have received payloads"
        payload = received[0]["body"]
        assert "AeroVigil" in payload["subject"]
        assert payload["advisory_only"] is True
        assert payload["faults"], "payload must carry the detected faults"
        # Dedupe: same snapshot again -> no second delivery.
        notifier.process_report(report)
        before = len(received)
        notifier.process_report(report)
        assert len(received) == before
    finally:
        server.shutdown()
        server.server_close()


def test_webhook_disabled_without_urls():
    from src.notifications.webhooks import WebhookNotifier

    notifier = WebhookNotifier(urls=[])
    report = FaultDetector().detect(FAULTY, asset_id="WTG-A")
    assert notifier.process_report(report) == []
    assert notifier.status()["n_webhooks"] == 0


def test_webhook_delivery_failure_is_reported():
    from src.notifications.webhooks import WebhookNotifier

    notifier = WebhookNotifier(urls=["http://127.0.0.1:1/nope"], timeout_s=1.0)
    report = FaultDetector().detect(FAULTY, asset_id="WTG-A")
    results = notifier.process_report(report)
    assert results
    assert all(not r.delivered for r in results)
    assert results[0].detail  # error message captured


def test_webhook_from_env(monkeypatch):
    from src.notifications.webhooks import WebhookNotifier

    monkeypatch.setenv("AV_WEBHOOK_URLS", "https://hooks.slack.com/services/a/b/c")
    monkeypatch.setenv("AV_WEBHOOK_MODE", "on")
    notifier = WebhookNotifier.from_env()
    assert notifier.enabled is True
    assert notifier.urls == ["https://hooks.slack.com/services/a/b/c"]
    assert notifier.status()["formats"] == ["slack"]


# --------------------------------------------------------------------------- #
# Digest scheduling                                                            #
# --------------------------------------------------------------------------- #
def test_next_digest_delay():
    from datetime import datetime, timezone

    from src.notifications.digest import next_digest_delay

    now = datetime(2026, 8, 17, 12, 30, 0, tzinfo=timezone.utc)
    delay = next_digest_delay(6, now=now)
    assert delay == pytest.approx(17.5 * 3600)  # next 06:00 UTC
    delay_same_hour = next_digest_delay(12, now=now)
    assert delay_same_hour == pytest.approx(23.5 * 3600)  # already past -> tomorrow
    delay_earlier = next_digest_delay(14, now=now)
    assert delay_earlier == pytest.approx(1.5 * 3600)
    with pytest.raises(ValueError):
        next_digest_delay(24, now=now)


def test_digest_config_from_env(monkeypatch):
    from src.notifications.digest import DigestConfig

    monkeypatch.setenv("AV_DIGEST_ENABLED", "1")
    monkeypatch.setenv("AV_DIGEST_HOUR", "5")
    monkeypatch.setenv("AV_DIGEST_RECIPIENTS", "a@b.c; c@d.e")
    monkeypatch.setenv("AV_DIGEST_TITLE", "Nightly")
    config = DigestConfig()
    assert config.enabled is True
    assert config.hour == 5
    assert config.recipients == ("a@b.c", "c@d.e")
    assert config.title == "Nightly"
    assert config.to_dict()["enabled"] is True


def test_build_fleet_digest_collects_reports():
    from src.digital_twin.specs import get_spec
    from src.digital_twin.twin import WindTurbineDigitalTwin
    from src.notifications.digest import build_fleet_digest
    from src.utils.schema import Telemetry

    twins = {}
    for asset in ("WTG-D1", "WTG-D2"):
        twin = WindTurbineDigitalTwin(asset, get_spec("NREL-5MW"))
        twin.update_state(
            Telemetry(
                vibration_mms=2.0,
                temperature_c=55.0,
                rpm=950.0,
                oil_viscosity_cst=32.0,
                load_pct=70.0,
            )
        )
        twins[asset] = twin
    reports = build_fleet_digest(twins)
    assert [r.asset_id for r in reports] == ["WTG-D1", "WTG-D2"]
    assert all(r.overall_status == "OK" for r in reports)


def test_run_digest_emails_fleet_report(eml_dir):
    from src.digital_twin.specs import get_spec
    from src.digital_twin.twin import WindTurbineDigitalTwin
    from src.notifications.digest import run_digest
    from src.utils.schema import Telemetry

    twin = WindTurbineDigitalTwin("WTG-D1", get_spec("NREL-5MW"))
    twin.update_state(
        Telemetry(
            vibration_mms=2.0, temperature_c=55.0, rpm=950.0, oil_viscosity_cst=32.0, load_pct=70.0
        )
    )
    config = NotificationConfig(
        mode="eml", report_recipients=("m@example.com",), artifact_dir=eml_dir
    )
    notifier = EmailNotifier(config=config)
    result = run_digest({"WTG-D1": twin}, notifier, title="Test digest")
    assert result is not None and result.delivered is True
    assert "Test digest" in result.subject
    assert list(eml_dir.glob("*.eml"))
    # No twins -> nothing sent.
    assert run_digest({}, notifier, title="x") is None


# --------------------------------------------------------------------------- #
# Maintenance mode (alert suppression)                                         #
# --------------------------------------------------------------------------- #
def test_suppress_asset_silences_alerts(tmp_path):
    tracker = AlertTracker(tmp_path / "state.json")
    tracker.suppress_asset("WTG-M", reason="gearbox replacement", operator="crew")
    assert tracker.is_suppressed("WTG-M")
    assert not tracker.should_send("WTG-M", "GB-02", "CRITICAL")
    assert tracker.suppressed_assets()[0]["reason"] == "gearbox replacement"
    assert tracker.unsuppress_asset("WTG-M") is True
    assert not tracker.is_suppressed("WTG-M")
    assert tracker.unsuppress_asset("WTG-M") is False  # already cleared


def test_process_report_returns_suppressed_result(notifier):
    notifier.tracker.suppress_asset("WTG-A", reason="service")
    report = FaultDetector().detect(FAULTY, asset_id="WTG-A")
    results = notifier.process_report(report)
    assert len(results) == 1
    assert results[0].channel == "skipped"
    assert "maintenance mode" in results[0].detail


def test_suppression_persists_across_instances(tmp_path):
    path = tmp_path / "state.json"
    tracker = AlertTracker(path)
    tracker.suppress_asset("WTG-P", reason="test")
    tracker2 = AlertTracker(path)
    assert tracker2.is_suppressed("WTG-P")
    # Old v1 flat state files still load (backward compatible).
    path.write_text('{"WTG-O:GB-02": {"severity": "HIGH", "count": 1}}', encoding="utf-8")
    tracker3 = AlertTracker(path)
    assert tracker3.should_send("WTG-O", "GB-02", "CRITICAL")  # escalation
