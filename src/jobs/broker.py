"""Optional Redis broker for the durable job queue.

SQLite (the ``jobs`` table) is always the source of truth for job state,
status and logs. This module adds an optional fan-out channel so workers
wake instantly when a job is queued instead of polling:

* ``JobBroker.publish(job_id)`` is called by :class:`JobManager` after the
  Pending row is durable,
* ``JobBroker.consume(timeout)`` is used by the standalone worker.

Both methods degrade gracefully when ``redis`` is not installed or the
broker is unreachable (publish is a no-op; consume returns ``None`` so the
worker falls back to SQLite polling).
"""

from __future__ import annotations

import logging

logger = logging.getLogger("jobs.broker")

_QUEUE_KEY = "aerovigil:jobs:pending"


class JobBroker:
    """Redis list-based job-id fan-out with graceful degradation."""

    def __init__(self, url: str) -> None:
        self._url = url
        self._client = None

    async def connect(self) -> bool:
        """Connect lazily; returns True when Redis is reachable."""
        try:
            import redis.asyncio as aioredis

            self._client = aioredis.from_url(self._url)
            await self._client.ping()
            return True
        except Exception as exc:  # noqa: BLE001 - degrade to polling
            logger.warning("Redis broker unavailable (%s); workers will poll SQLite", exc)
            self._client = None
            return False

    async def publish(self, job_id: str) -> bool:
        if self._client is None:
            return False
        try:
            await self._client.lpush(_QUEUE_KEY, job_id)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis publish failed (%s); job stays durable in SQLite", exc)
            return False

    async def consume(self, timeout: float = 0.5) -> str | None:
        if self._client is None:
            return None
        try:
            item = await self._client.brpop(_QUEUE_KEY, timeout=timeout)
            return item[1].decode() if item else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis consume failed (%s)", exc)
            return None
