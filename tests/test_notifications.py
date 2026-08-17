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
        vibration_mms=2.0, temperature_c=55.0, rpm=950.0,
        oil_viscosity_cst=32.0, load_pct=70.0,
    )
    rec = twin.update_state(healthy)
    assert rec["notifications"] == []
    degraded = Telemetry(
        vibration_mms=9.0, temperature_c=60.0, rpm=1000.0,
        oil_viscosity_cst=5.0, load_pct=90.0,
    )
    rec2 = twin.update_state(degraded)
    assert rec2["notifications"], "twin must email alerts on severe faults"
    assert all(n["delivered"] for n in rec2["notifications"])
    assert any("CRITICAL" in n["subject"] or "HIGH" in n["subject"]
               for n in rec2["notifications"])
    # Repeating the same state does not re-alert (dedupe).
    rec3 = twin.update_state(degraded)
    assert rec3["notifications"] == []
