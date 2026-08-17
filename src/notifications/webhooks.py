"""Webhook alerts for CRITICAL/HIGH faults (Slack, Teams, generic).

Complements the email notifier: when a severe fault is detected, the same
dedupe/escalation rules (shared :class:`AlertTracker`) decide whether to POST
a JSON payload to every configured webhook URL.

Payload formats
---------------
* ``slack``   — Slack block payload (``text`` + ``blocks``).
* ``teams``   — Microsoft Teams adaptive-card payload.
* ``generic`` — plain ``{subject, severity, asset_id, faults, ...}`` JSON,
                suitable for any HTTP receiver (n8n, Zapier, PagerDuty
                generic webhook, Twilio Studio, ...).

Configuration
-------------
* env ``AV_WEBHOOK_URLS`` — comma/``;`` separated URLs (auto-detected:
  ``hooks.slack.com`` → slack, ``webhook.office.com`` → teams, else generic).
* env ``AV_WEBHOOK_SEVERITIES`` — e.g. ``CRITICAL,HIGH`` (default).
* env ``AV_WEBHOOK_MODE`` — ``on`` (default when URLs exist) | ``off``.

The notifier never blocks ingestion: delivery failures are returned in the
result dict, and timeouts are capped.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import dataclass
from typing import Any

from src.faults.detector import FaultReport
from src.notifications.emailer import AlertTracker

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = 8.0


@dataclass(frozen=True)
class WebhookResult:
    """Outcome of one webhook delivery."""

    url: str
    format: str  # slack | teams | generic
    delivered: bool
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "format": self.format,
            "delivered": self.delivered,
            "detail": self.detail,
        }


def detect_format(url: str) -> str:
    """Pick a payload format from the webhook URL."""
    lowered = url.lower()
    if "hooks.slack.com" in lowered or "slack.com" in lowered:
        return "slack"
    if "webhook.office.com" in lowered or "outlook.office.com" in lowered:
        return "teams"
    return "generic"


def build_payload(report: FaultReport, fmt: str, subject: str) -> dict[str, Any]:
    """Build the webhook body for one fault report."""
    summary = {
        "asset_id": report.asset_id,
        "timestamp": report.timestamp,
        "overall_status": report.overall_status,
        "health_score": round(report.health_score, 1),
        "oil_health_score": round(report.oil.health_score, 1),
        "oil_status": report.oil.overall_status,
        "n_faults": report.n_faults,
    }
    faults = [
        {
            "fault_id": f.fault_id,
            "name": f.name,
            "subsystem": f.subsystem,
            "subsystem_label": f.subsystem_label,
            "severity": f.severity,
            "confidence": round(f.confidence, 2),
            "message": f.message,
            "recommended_actions": f.recommended_actions[:3],
        }
        for f in report.faults
    ]
    if fmt == "slack":
        color = {
            "CRITICAL": "danger",
            "HIGH": "warning",
            "MEDIUM": "warning",
            "LOW": "good",
        }.get(report.overall_status, "good")
        lines = (
            "\n".join(
                f"• *{f['fault_id']} {f['name']}* [{f['severity']}] — {f['message']}"
                for f in faults
            )
            or "No active faults."
        )
        return {
            "text": f"*AeroVigil* {subject}",
            "attachments": [
                {
                    "color": color,
                    "title": f"AeroVigil fault alert — {report.asset_id}",
                    "text": lines,
                    "footer": "Advisory only — AeroVigil never sends commands to the turbine.",
                }
            ],
        }
    if fmt == "teams":
        return {
            "type": "message",
            "attachments": [
                {
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "content": {
                        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                        "type": "AdaptiveCard",
                        "version": "1.4",
                        "body": [
                            {
                                "type": "TextBlock",
                                "size": "Large",
                                "weight": "Bolder",
                                "text": f"AeroVigil — {subject}",
                            },
                            {
                                "type": "FactSet",
                                "facts": [
                                    {"title": "Asset", "value": report.asset_id},
                                    {"title": "Status", "value": report.overall_status},
                                    {"title": "Health", "value": f"{report.health_score:.0f}/100"},
                                    {"title": "Oil", "value": f"{report.oil.health_score:.0f}/100"},
                                ],
                            },
                            {
                                "type": "TextBlock",
                                "text": "\n".join(
                                    f"{f['fault_id']} {f['name']} [{f['severity']}]" for f in faults
                                )
                                or "No active faults.",
                            },
                        ],
                        "actions": [
                            {
                                "type": "Action.OpenUrl",
                                "title": "Open console",
                                "url": "http://localhost:8080",
                            }
                        ],
                    },
                }
            ],
        }
    return {"subject": subject, "summary": summary, "faults": faults, "advisory_only": True}


class WebhookNotifier:
    """POSTs alert payloads to configured webhook URLs."""

    def __init__(
        self,
        urls: list[str] | None = None,
        tracker: AlertTracker | None = None,
        severities: tuple[str, ...] = ("CRITICAL", "HIGH"),
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        enabled: bool = True,
    ) -> None:
        self.urls = [u.strip() for u in (urls or []) if u.strip()]
        self.tracker = tracker or AlertTracker()
        self.severities = severities
        self.timeout_s = timeout_s
        self.enabled = enabled

    @classmethod
    def from_env(cls, tracker: AlertTracker | None = None) -> WebhookNotifier:
        import os

        raw = os.environ.get("AV_WEBHOOK_URLS", "")
        urls = [u.strip() for u in raw.replace(";", ",").split(",") if u.strip()]
        mode = os.environ.get("AV_WEBHOOK_MODE", "on" if urls else "off").strip().lower()
        severities = tuple(
            s.strip().upper()
            for s in os.environ.get("AV_WEBHOOK_SEVERITIES", "CRITICAL,HIGH").split(",")
            if s.strip()
        )
        return cls(
            urls=urls,
            tracker=tracker,
            severities=severities,
            enabled=mode not in ("off", "0", "false"),
        )

    def process_report(self, report: FaultReport) -> list[WebhookResult]:
        """Alert via webhook for severe faults (dedupe/escalation shared with email)."""
        if not self.enabled or not self.urls:
            return []
        results: list[WebhookResult] = []
        for fault in report.faults:
            if fault.severity not in self.severities:
                continue
            if not self.tracker.should_send(report.asset_id, fault.fault_id, fault.severity):
                continue
            subject = (
                f"[AeroVigil {fault.severity}] {report.asset_id}: {fault.fault_id} {fault.name}"
            )
            for url in self.urls:
                results.append(self._post(url, report, subject))
            self.tracker.record(report.asset_id, fault.fault_id, fault.severity)
        return results

    def send_message(
        self, report: FaultReport, subject: str, urls: list[str] | None = None
    ) -> list[WebhookResult]:
        """Post a message for every URL (no dedupe — used by manual tests)."""
        targets = urls if urls is not None else self.urls
        return [self._post(url, report, subject) for url in targets]

    def _post(self, url: str, report: FaultReport, subject: str) -> WebhookResult:
        fmt = detect_format(url)
        payload = build_payload(report, fmt, subject)
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "AeroVigil/1.0"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                response.read(256)  # drain; status check below
                delivered = 200 <= response.status < 300
            return WebhookResult(
                url=url, format=fmt, delivered=delivered, detail=f"HTTP {response.status}"
            )
        except Exception as exc:  # noqa: BLE001 - delivery must never raise
            logger.warning("webhook delivery failed for %s: %s", url, exc)
            return WebhookResult(url=url, format=fmt, delivered=False, detail=str(exc))

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "n_webhooks": len(self.urls),
            "formats": [detect_format(u) for u in self.urls],
            "severities": list(self.severities),
            "timeout_s": self.timeout_s,
        }
