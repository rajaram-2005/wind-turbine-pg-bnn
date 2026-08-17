"""Email health reports and severity-based alerts.

* :class:`EmailNotifier` turns :class:`src.faults.detector.FaultReport`
  snapshots into HTML/text emails: immediate **alerts** for severe faults
  (CRITICAL / HIGH) and periodic **health reports** for the fleet.
* :class:`WebhookNotifier` (``src.notifications.webhooks``) POSTs the same
  alerts to Slack / Teams / generic webhooks.
* :class:`AlertTracker` prevents alert storms: per (asset, fault) dedupe with
  severity escalation and per-severity cooldowns, persisted to a JSON file.
  It also supports the operator workflow — acknowledge / resolve alerts.
* When SMTP is not configured the notifier falls back to writing standard
  ``.eml`` files under ``artifacts/notifications/`` so the pipeline works
  offline and can be previewed in any mail client.

Configuration is read from environment variables (``AV_*``) so secrets never
live in the repository; the YAML ``notifications`` section carries only
non-secret defaults (mode, recipients, cooldowns).
"""

from src.notifications.emailer import (
    DEFAULT_COOLDOWN_HOURS,
    SEVERITY_RANK,
    AlertTracker,
    EmailNotifier,
    NotificationConfig,
    NotificationResult,
    render_alert_html,
    render_health_report_html,
)
from src.notifications.webhooks import (
    WebhookNotifier,
    WebhookResult,
    build_payload,
    detect_format,
)

__all__ = [
    "DEFAULT_COOLDOWN_HOURS",
    "SEVERITY_RANK",
    "AlertTracker",
    "EmailNotifier",
    "NotificationConfig",
    "NotificationResult",
    "WebhookNotifier",
    "WebhookResult",
    "build_payload",
    "detect_format",
    "render_alert_html",
    "render_health_report_html",
]
