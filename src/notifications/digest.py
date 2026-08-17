"""Scheduled fleet health digest.

The unified app runs a background task (see ``src.unified_app``) that emails
a fleet-wide health report once a day — no external cron required.  This
module holds the pure, testable pieces:

* :func:`build_fleet_digest` — collect the latest :class:`FaultReport` of
  every tracked digital twin.
* :func:`next_digest_delay` — seconds until the next occurrence of the
  configured hour (UTC), so the scheduler can sleep precisely.
* :func:`run_digest` — build + send the digest email (or ``.eml`` offline).

Configuration (env):

* ``AV_DIGEST_ENABLED`` — ``1`` to run the scheduled digest (default off).
* ``AV_DIGEST_HOUR`` — UTC hour 0..23 (default 6).
* ``AV_DIGEST_RECIPIENTS`` — override; falls back to report recipients.
* ``AV_DIGEST_TITLE`` — email subject title.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

from src.faults.detector import FaultReport
from src.notifications.emailer import EmailNotifier, NotificationResult

DEFAULT_HOUR = 6


class DigestConfig:
    """Resolved digest settings from ``AV_*`` environment variables."""

    def __init__(self, env: dict[str, str] | None = None) -> None:
        env = env or os.environ
        raw_enabled = env.get("AV_DIGEST_ENABLED", "0").strip().lower()
        self.enabled = raw_enabled in ("1", "true", "yes", "on")
        try:
            self.hour = int(env.get("AV_DIGEST_HOUR", str(DEFAULT_HOUR)).strip())
        except ValueError:
            self.hour = DEFAULT_HOUR
        self.hour = max(0, min(23, self.hour))
        raw_recipients = env.get("AV_DIGEST_RECIPIENTS", "")
        self.recipients: tuple[str, ...] = tuple(
            sorted({r.strip() for r in raw_recipients.replace(";", ",").split(",") if r.strip()})
        )
        self.title = env.get("AV_DIGEST_TITLE", "Fleet health digest").strip()

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "hour_utc": self.hour,
            "recipients": list(self.recipients),
            "title": self.title,
        }


def build_fleet_digest(twins: Any) -> list[FaultReport]:
    """Latest fault report of every tracked twin (skip twins with none)."""
    reports: list[FaultReport] = []
    for twin in twins.values():
        if twin.last_fault_report is not None:
            reports.append(twin.last_fault_report)
    return reports


def next_digest_delay(hour: int, now: datetime | None = None) -> float:
    """Seconds from ``now`` until the next UTC ``hour`` occurrence.

    A run at exactly the target hour counts as already past (waits one day),
    so repeated app restarts do not re-fire the digest.
    """
    now = now or datetime.now(timezone.utc)
    if not 0 <= hour <= 23:
        raise ValueError(f"hour must be in 0..23, got {hour}")
    target = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    if now >= target:
        target += timedelta(days=1)
    return max(0.0, (target - now).total_seconds())


def run_digest(
    twins: Any,
    notifier: EmailNotifier,
    title: str | None = None,
    recipients: Sequence[str] | None = None,
) -> NotificationResult | None:
    """Email the fleet health digest built from ``twins``."""
    reports = build_fleet_digest(twins)
    if not reports:
        return None
    digest_title = title or "Fleet health digest"
    return notifier.send_health_report(reports, title=digest_title, recipients=recipients)
