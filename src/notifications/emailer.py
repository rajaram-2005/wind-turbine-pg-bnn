"""Email delivery for AeroVigil fault reports and alerts.

The notifier supports three modes:

* ``smtp``  — deliver through an SMTP server (env ``AV_SMTP_*``).
* ``eml``   — write standard ``.eml`` files under the artifact directory
  (default; used automatically when no SMTP host is configured). Useful for
  previewing reports and for offline/air-gapped deployments.
* ``off``   — disabled; ``process_report`` still records what *would* have
  been sent so dashboards can show the alerting state.

Alert policy (``AlertTracker``)
-------------------------------
* CRITICAL and HIGH faults trigger an alert email immediately (per
  (asset, fault) pair, subject to cooldown and escalation rules).
* MEDIUM faults are rolled into the periodic health report; LOW faults are
  listed there too.
* Escalation: if the same fault reappears at a higher severity than last
  sent, a new alert fires immediately.
* Cooldown: after an alert, the same (asset, fault) is silenced until the
  per-severity cooldown elapses (CRITICAL 6 h, HIGH 24 h, MEDIUM 7 d).
  The tracker state persists to a JSON file so restarts do not reset it.

Every email carries the advisory-only banner: AeroVigil recommends actions,
it never actuates the turbine.
"""

from __future__ import annotations

import json
import os
import smtplib
import ssl
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from html import escape
from pathlib import Path
from typing import Any

from src.faults.detector import FaultReport

SEVERITY_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
DEFAULT_COOLDOWN_HOURS: dict[str, float] = {
    "CRITICAL": 6.0,
    "HIGH": 24.0,
    "MEDIUM": 24.0 * 7.0,
    "LOW": float("inf"),
}
ALERT_SEVERITIES = ("CRITICAL", "HIGH")

_ENV_PREFIX = "AV_"
_PRODUCT = "AeroVigil"
_DEFAULT_FROM = "aerovigil@localhost"
_DEFAULT_ARTIFACT_DIR = Path("artifacts") / "notifications"


@dataclass(frozen=True)
class NotificationResult:
    """Outcome of one notification attempt."""

    subject: str
    recipients: tuple[str, ...]
    channel: str  # "smtp" | "eml" | "skipped"
    delivered: bool
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "recipients": list(self.recipients),
            "channel": self.channel,
            "delivered": self.delivered,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class NotificationConfig:
    """Resolved notification settings (env takes precedence over YAML)."""

    mode: str = "auto"  # auto | smtp | eml | off
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = _DEFAULT_FROM
    smtp_tls: bool = True
    alert_recipients: tuple[str, ...] = ()
    report_recipients: tuple[str, ...] = ()
    artifact_dir: Path = _DEFAULT_ARTIFACT_DIR
    cooldown_hours: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_COOLDOWN_HOURS))
    alert_severities: tuple[str, ...] = ALERT_SEVERITIES

    @classmethod
    def from_env(cls, **overrides) -> NotificationConfig:
        """Build from ``AV_*`` environment variables plus explicit overrides."""
        env = os.environ

        def _flag(name: str, default: bool) -> bool:
            raw = env.get(f"{_ENV_PREFIX}{name}")
            if raw is None:
                return default
            return raw.strip().lower() in ("1", "true", "yes", "on")

        def _recipients(name: str) -> tuple[str, ...]:
            raw = env.get(f"{_ENV_PREFIX}{name}", "")
            return tuple(sorted({r.strip() for r in raw.replace(";", ",").split(",") if r.strip()}))

        base = cls(
            mode=env.get(f"{_ENV_PREFIX}NOTIFY_MODE", "auto").strip().lower(),
            smtp_host=env.get(f"{_ENV_PREFIX}SMTP_HOST", "").strip(),
            smtp_port=int(env.get(f"{_ENV_PREFIX}SMTP_PORT", "587").strip() or "587"),
            smtp_user=env.get(f"{_ENV_PREFIX}SMTP_USER", "").strip(),
            smtp_password=env.get(f"{_ENV_PREFIX}SMTP_PASSWORD", "").strip(),
            smtp_from=env.get(f"{_ENV_PREFIX}SMTP_FROM", _DEFAULT_FROM).strip(),
            smtp_tls=_flag("SMTP_TLS", True),
            alert_recipients=_recipients("ALERT_RECIPIENTS"),
            report_recipients=_recipients("REPORT_RECIPIENTS"),
            artifact_dir=Path(
                env.get(f"{_ENV_PREFIX}NOTIFY_DIR", str(_DEFAULT_ARTIFACT_DIR)).strip()
            ),
        )
        for key, value in overrides.items():
            if value is not None:
                object.__setattr__(base, key, value)
        return base

    @property
    def effective_mode(self) -> str:
        """Resolve ``auto`` to a concrete channel."""
        if self.mode != "auto":
            return self.mode
        if self.smtp_host:
            return "smtp"
        return "eml"


class AlertTracker:
    """Per-(asset, fault) dedupe, escalation and cooldown state."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._state: dict[str, dict[str, Any]] = {}
        if self.path and self.path.exists():
            try:
                self._state = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self._state = {}

    def _key(self, asset_id: str, fault_id: str) -> str:
        return f"{asset_id}:{fault_id}"

    def should_send(
        self,
        asset_id: str,
        fault_id: str,
        severity: str,
        now: datetime | None = None,
    ) -> bool:
        """True when an alert for this (asset, fault) must go out now."""
        if severity not in SEVERITY_RANK:
            return False
        now = now or datetime.now(timezone.utc)
        entry = self._state.get(self._key(asset_id, fault_id))
        if entry is None:
            return True
        # Escalation: same fault, higher severity → alert immediately,
        # even when the previous alert was acknowledged or in cooldown.
        if SEVERITY_RANK.get(severity, 0) > SEVERITY_RANK.get(entry.get("severity", ""), 0):
            return True
        # Acknowledged or resolved alerts are silenced until they escalate or
        # re-appear after resolution.
        if entry.get("acknowledged"):
            return False
        if entry.get("resolved"):
            return False
        cooldown = DEFAULT_COOLDOWN_HOURS.get(severity, float("inf"))
        if not cooldown or cooldown == float("inf"):
            return False
        last = entry.get("last_sent")
        if not last:
            return True
        try:
            last_dt = datetime.fromisoformat(last)
        except ValueError:
            return True
        return now - last_dt >= timedelta(hours=cooldown)

    def record(
        self,
        asset_id: str,
        fault_id: str,
        severity: str,
        now: datetime | None = None,
    ) -> None:
        now = now or datetime.now(timezone.utc)
        self._state[self._key(asset_id, fault_id)] = {
            "severity": severity,
            "last_sent": now.isoformat(),
            "count": self._state.get(self._key(asset_id, fault_id), {}).get("count", 0) + 1,
        }
        self._persist()

    def clear(self) -> None:
        self._state = {}
        if self.path and self.path.exists():
            self.path.unlink(missing_ok=True)

    # -- operator workflow -------------------------------------------------- #
    def acknowledge(self, asset_id: str, fault_id: str, operator: str = "") -> bool:
        """Acknowledge an open alert: it stops re-alerting until escalation or
        resolution. Returns True when an alert existed to acknowledge."""
        key = self._key(asset_id, fault_id)
        if key not in self._state:
            return False
        self._state[key]["acknowledged"] = True
        self._state[key]["acknowledged_at"] = datetime.now(timezone.utc).isoformat()
        self._state[key]["operator"] = operator
        self._persist()
        return True

    def resolve(self, asset_id: str, fault_id: str, operator: str = "") -> bool:
        """Mark an alert resolved (fault fixed). A future re-detection of the
        same fault starts a fresh alert cycle. Returns True when removed."""
        key = self._key(asset_id, fault_id)
        if key not in self._state:
            return False
        entry = self._state.pop(key)
        entry["resolved"] = True
        entry["resolved_at"] = datetime.now(timezone.utc).isoformat()
        entry["operator"] = operator
        self._state[f"{key}::resolved"] = entry  # keep an audit trail
        self._persist()
        return True

    def open_alerts(self) -> list[dict]:
        """Every tracked alert that has not been resolved (newest first)."""
        items = []
        for key, entry in self._state.items():
            if key.endswith("::resolved"):
                continue
            asset_id, fault_id = key.split(":", 1)
            items.append(
                {
                    "asset_id": asset_id,
                    "fault_id": fault_id,
                    "severity": entry.get("severity"),
                    "last_sent": entry.get("last_sent"),
                    "count": entry.get("count", 0),
                    "acknowledged": bool(entry.get("acknowledged")),
                    "acknowledged_at": entry.get("acknowledged_at"),
                    "operator": entry.get("operator", ""),
                }
            )
        items.sort(key=lambda item: item.get("last_sent") or "", reverse=True)
        return items

    def _persist(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(self._state, indent=2, sort_keys=True), encoding="utf-8"
            )

    def to_dict(self) -> dict:
        return dict(self._state)


# --------------------------------------------------------------------------- #
# HTML rendering                                                               #
# --------------------------------------------------------------------------- #
_SEVERITY_COLORS = {
    "LOW": "#8a97a5",
    "MEDIUM": "#f59e0b",
    "HIGH": "#f97316",
    "CRITICAL": "#ef4444",
}

_CSS = """
body{font-family:-apple-system,'Segoe UI',Roboto,Arial,sans-serif;background:#0b1220;color:#e5edf5;margin:0;padding:24px;}
.wrap{max-width:760px;margin:0 auto;background:#101a2e;border:1px solid #22304a;border-radius:14px;overflow:hidden;}
.head{background:linear-gradient(90deg,#0e7490,#155e75);padding:18px 24px;color:#fff;}
.head h1{margin:0;font-size:20px;}
.head p{margin:4px 0 0;font-size:12px;opacity:.85;}
.body{padding:20px 24px;}
.score-row{display:flex;gap:14px;margin:14px 0;}
.score{flex:1;background:#0d1626;border:1px solid #22304a;border-radius:10px;padding:12px 14px;}
.score .label{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:#7d8ea8;}
.score .value{font-size:26px;font-weight:800;margin-top:2px;}
.bar{height:6px;background:#1b2942;border-radius:6px;margin-top:8px;overflow:hidden;}
.bar span{display:block;height:100%;border-radius:6px;}
table{width:100%;border-collapse:collapse;margin:12px 0;font-size:13px;}
th{background:#152238;color:#9fb2cc;text-align:left;padding:8px 10px;font-size:11px;text-transform:uppercase;letter-spacing:.06em;}
td{border-top:1px solid #1b2942;padding:8px 10px;vertical-align:top;}
.badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:700;color:#fff;}
.ok{color:#34d399;font-weight:700;}
.warn{color:#fbbf24;font-weight:700;}
.alarm{color:#f87171;font-weight:700;}
.footer{padding:12px 24px;border-top:1px solid #1b2942;color:#7d8ea8;font-size:11px;}
ul{margin:4px 0;padding-left:18px;}
"""


def _severity_badge(severity: str) -> str:
    color = _SEVERITY_COLORS.get(severity, "#8a97a5")
    return f'<span class="badge" style="background:{color}">{escape(severity)}</span>'


def _score_bar(score: float) -> str:
    color = "#34d399" if score >= 70 else ("#fbbf24" if score >= 40 else "#f87171")
    return f'<div class="bar"><span style="width:{max(0, min(100, score)):.0f}%;background:{color}"></span></div>'


def render_alert_html(report: FaultReport) -> str:
    """HTML body of an immediate fault alert for one asset."""
    rows = []
    for fault in report.faults:
        actions = "".join(f"<li>{escape(a)}</li>" for a in fault.recommended_actions[:3])
        rows.append(
            "<tr>"
            f"<td>{_severity_badge(fault.severity)}</td>"
            f"<td><strong>{escape(fault.fault_id)}</strong> · {escape(fault.name)}<br>"
            f"<span style='color:#7d8ea8;font-size:11px'>{escape(fault.subsystem_label)}</span></td>"
            f"<td>{escape(fault.message)}<br>"
            f"<span style='color:#7d8ea8;font-size:11px'>confidence "
            f"{fault.confidence:.0%}</span></td>"
            f"<td><ul style='margin:0'>{actions}</ul></td>"
            "</tr>"
        )
    rows_html = (
        "".join(rows)
        if rows
        else (
            "<tr><td colspan='4' style='color:#7d8ea8'>No active faults in this snapshot.</td></tr>"
        )
    )
    oil_status = report.oil.overall_status
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>{_CSS}</style></head>
<body><div class="wrap">
<div class="head"><h1>⚠️ AeroVigil Fault Alert</h1>
<p>{escape(report.asset_id)} · {escape(report.timestamp)}</p></div>
<div class="body">
<div class="score-row">
<div class="score"><div class="label">Asset health</div>
<div class="value">{report.health_score:.0f}/100</div>{_score_bar(report.health_score)}</div>
<div class="score"><div class="label">Oil condition</div>
<div class="value">{report.oil.health_score:.0f}/100</div>{_score_bar(report.oil.health_score)}</div>
<div class="score"><div class="label">Overall status</div>
<div class="value" style="font-size:18px;margin-top:8px">
<span class="{oil_status.lower()}">{escape(oil_status)}</span></div></div>
</div>
<p style="font-size:13px;color:#9fb2cc">{report.n_faults} fault(s) detected across
{len(report.by_subsystem())} subsystem(s).</p>
<table><tr><th>Severity</th><th>Fault</th><th>Evidence</th><th>Recommended action</th></tr>
{rows_html}</table>
</div>
<div class="footer">Advisory only — AeroVigil recommends actions and never sends commands to the
turbine. Sent by AeroVigil monitoring ({escape(report.asset_id)}).</div>
</div></body></html>"""


def render_health_report_html(reports: Sequence[FaultReport], title: str) -> str:
    """HTML body of a fleet health report covering several assets."""
    sections = []
    for report in reports:
        status = report.overall_status
        color = {
            "OK": "#34d399",
            "LOW": "#8a97a5",
            "MEDIUM": "#f59e0b",
            "HIGH": "#f97316",
            "CRITICAL": "#ef4444",
        }.get(status, "#8a97a5")
        fault_brief = ", ".join(f"{f.fault_id} ({f.severity})" for f in report.faults[:5]) or "—"
        sections.append(
            "<tr>"
            f"<td><strong>{escape(report.asset_id)}</strong></td>"
            f'<td><span class="badge" style="background:{color}">{escape(status)}</span></td>'
            f"<td>{report.health_score:.0f}</td>"
            f"<td>{report.oil.health_score:.0f} ({escape(report.oil.overall_status)})</td>"
            f"<td>{report.n_faults}</td>"
            f"<td style='font-size:12px'>{escape(fault_brief)}</td>"
            "</tr>"
        )
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>{_CSS}</style></head>
<body><div class="wrap">
<div class="head"><h1>📋 AeroVigil Health Report</h1><p>{escape(title)}</p></div>
<div class="body">
<p style="font-size:13px;color:#9fb2cc">{len(reports)} asset(s) in this report.
CRITICAL/HIGH faults are alerted immediately; this periodic report carries the full fleet picture.</p>
<table><tr><th>Asset</th><th>Status</th><th>Health</th><th>Oil</th><th>Faults</th>
<th>Top findings</th></tr>{''.join(sections)}</table>
</div>
<div class="footer">Advisory only — AeroVigil recommends actions and never sends commands to the
turbine. Generated by AeroVigil monitoring.</div>
</div></body></html>"""


def _report_text(report: FaultReport) -> str:
    lines = [
        f"AeroVigil fault report — {report.asset_id} @ {report.timestamp}",
        f"Overall: {report.overall_status} | Health: {report.health_score:.0f}/100 | "
        f"Oil: {report.oil.health_score:.0f}/100 ({report.oil.overall_status})",
        "",
    ]
    for fault in report.faults:
        lines.append(
            f"[{fault.severity}] {fault.fault_id} {fault.name} ({fault.subsystem_label}) "
            f"conf {fault.confidence:.0%}"
        )
        lines.append(f"    {fault.message}")
        lines.append("    Actions: " + "; ".join(fault.recommended_actions[:3]))
    if not report.faults:
        lines.append("No active faults.")
    lines.append("")
    lines.append("Advisory only — AeroVigil never sends commands to the turbine.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Notifier                                                                    #
# --------------------------------------------------------------------------- #
class EmailNotifier:
    """Delivers fault alerts and health reports by email (or .eml offline)."""

    def __init__(
        self, config: NotificationConfig | None = None, tracker: AlertTracker | None = None
    ):
        self.config = config or NotificationConfig.from_env()
        self.tracker = tracker or AlertTracker(self.config.artifact_dir / "alert_state.json")

    # -- public API --------------------------------------------------------- #
    def process_report(self, report: FaultReport) -> list[NotificationResult]:
        """Send alerts for severe faults in a report; return what was sent.

        Only faults whose severity is in ``alert_severities`` (CRITICAL/HIGH)
        and that pass the tracker's dedupe/cooldown gates trigger an email.
        """
        if self.config.effective_mode == "off":
            return []
        results: list[NotificationResult] = []
        for fault in report.faults:
            if fault.severity not in self.config.alert_severities:
                continue
            if not self.tracker.should_send(report.asset_id, fault.fault_id, fault.severity):
                continue
            subject = (
                f"[AeroVigil {fault.severity}] {report.asset_id}: " f"{fault.fault_id} {fault.name}"
            )
            result = self.deliver(subject, render_alert_html(report), _report_text(report))
            self.tracker.record(report.asset_id, fault.fault_id, fault.severity)
            results.append(result)
        return results

    def send_health_report(
        self,
        reports: Sequence[FaultReport],
        title: str | None = None,
        recipients: Sequence[str] | None = None,
    ) -> NotificationResult | None:
        """Send one digest email covering every provided asset report."""
        if self.config.effective_mode == "off" or not reports:
            return None
        targets = tuple(recipients) if recipients else self.config.report_recipients
        if not targets:
            targets = self.config.alert_recipients
        if not targets:
            return None
        title = title or f"Fleet health — {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}"
        subject = f"[AeroVigil Health Report] {title}"
        return self.deliver(
            subject,
            render_health_report_html(reports, title),
            "\n\n".join(_report_text(r) for r in reports),
            recipients=targets,
        )

    def deliver(
        self,
        subject: str,
        html: str,
        text: str,
        recipients: Sequence[str] | None = None,
    ) -> NotificationResult:
        """Build the message and hand it to the active channel."""
        targets = tuple(recipients) if recipients else self.config.alert_recipients
        if not targets:
            return NotificationResult(
                subject=subject,
                recipients=(),
                channel="skipped",
                delivered=False,
                detail="no recipients configured",
            )
        mode = self.config.effective_mode
        if mode == "smtp":
            return self._deliver_smtp(subject, html, text, targets)
        if mode == "eml":
            return self._deliver_eml(subject, html, text, targets)
        return NotificationResult(
            subject=subject,
            recipients=targets,
            channel="skipped",
            delivered=False,
            detail=f"mode={mode}",
        )

    # -- channels ----------------------------------------------------------- #
    def _build_message(
        self, subject: str, html: str, text: str, targets: tuple[str, ...]
    ) -> EmailMessage:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = self.config.smtp_from
        msg["To"] = ", ".join(targets)
        msg["Date"] = formatdate(localtime=True)
        msg["Message-ID"] = make_msgid(domain="aerovigil")
        msg.set_content(text)
        msg.add_alternative(html, subtype="html")
        return msg

    def _deliver_smtp(
        self, subject: str, html: str, text: str, targets: tuple[str, ...]
    ) -> NotificationResult:
        msg = self._build_message(subject, html, text, targets)
        try:
            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port, timeout=30) as server:
                if self.config.smtp_tls:
                    context = ssl.create_default_context()
                    server.starttls(context=context)
                if self.config.smtp_user:
                    server.login(self.config.smtp_user, self.config.smtp_password)
                server.send_message(msg)
            return NotificationResult(
                subject=subject,
                recipients=targets,
                channel="smtp",
                delivered=True,
                detail=f"{self.config.smtp_host}:{self.config.smtp_port}",
            )
        except (OSError, smtplib.SMTPException) as exc:
            return NotificationResult(
                subject=subject,
                recipients=targets,
                channel="smtp",
                delivered=False,
                detail=str(exc),
            )

    def _deliver_eml(
        self, subject: str, html: str, text: str, targets: tuple[str, ...]
    ) -> NotificationResult:
        msg = self._build_message(subject, html, text, targets)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in subject)[:80]
        self.config.artifact_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.artifact_dir / f"{stamp}-{safe}.eml"
        path.write_bytes(msg.as_bytes())
        return NotificationResult(
            subject=subject,
            recipients=targets,
            channel="eml",
            delivered=True,
            detail=str(path),
        )

    def status(self) -> dict:
        """Human/API-readable configuration summary (no secrets)."""
        cfg = self.config
        return {
            "mode": cfg.effective_mode,
            "configured_mode": cfg.mode,
            "smtp_host": cfg.smtp_host,
            "smtp_port": cfg.smtp_port,
            "smtp_tls": cfg.smtp_tls,
            "from": cfg.smtp_from,
            "alert_recipients": list(cfg.alert_recipients),
            "report_recipients": list(cfg.report_recipients),
            "artifact_dir": str(cfg.artifact_dir),
            "alert_severities": list(cfg.alert_severities),
            "cooldown_hours": {k: v for k, v in cfg.cooldown_hours.items() if v != float("inf")},
            "tracked_alerts": len(self.tracker.to_dict()),
            "advisory_only": True,
        }

    def deliver_test(self, recipients: Sequence[str] | None = None) -> NotificationResult:
        """Send a connectivity-test email to the alert recipients."""
        subject = "[AeroVigil TEST] notification channel check"
        html = (
            "<html><body><h3>AeroVigil notification test</h3>"
            "<p>If you can read this, the email channel is working.</p>"
            "<p><em>Advisory only — AeroVigil never sends commands to the turbine.</em></p>"
            "</body></html>"
        )
        text = (
            "AeroVigil notification test.\n"
            "If you can read this, the email channel is working.\n"
            "Advisory only — AeroVigil never sends commands to the turbine."
        )
        return self.deliver(subject, html, text, recipients=recipients)
