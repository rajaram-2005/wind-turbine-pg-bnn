"""In-process asyncio job manager with SQLite persistence.

The manager schedules framework jobs as background coroutines. Each job maps to
a CLI command (``python -m src <subcommand> ...`` / ``python main.py ...``) that
is executed with :func:`asyncio.create_subprocess_exec`. Standard output and
standard error are streamed line-by-line into the job's log buffer, so the
``GET /api/jobs/{job_id}`` endpoint can surface recent execution logs while the
job is still running.

Design goals
------------
* **No external broker.** Uses ``asyncio`` only – no Redis/Celery required.
* **Crash-transparent state.** Status transitions and logs are persisted to a
  SQLite file so a status poll after a brief hiccup still works.
* **Safe by construction.** Only a fixed allow-list of job types can be queued,
  and each maps to an explicit, argument-validated command builder.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DB = _REPO_ROOT / "artifacts" / "jobs.sqlite3"
_MAX_LOG_LINES = 500

logger = logging.getLogger("jobs.manager")


class JobStatus(str, Enum):
    """Lifecycle states for a queued job."""

    PENDING = "Pending"
    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"


# Map each public job type to the CLI subcommand that performs the work.
# Values are argument lists appended to ``[python, main.py]``.
ALLOWED_JOB_TYPES: dict[str, list[str]] = {
    "physics": ["train"],
    "train": ["train"],
    "evaluate": ["evaluate"],
    "federated": ["federated"],
    "export": ["export"],
    "active-learning": ["active-sample"],
    "active-sample": ["active-sample"],
    "shap": ["explain"],
    "explain": ["explain"],
}


@dataclass
class JobRecord:
    """Serializable snapshot of a single job."""

    job_id: str
    job_type: str
    status: JobStatus
    created_at: float
    updated_at: float
    args: list[str] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    result: Optional[dict[str, Any]] = None

    def to_public(self, *, max_logs: int = 100) -> dict[str, Any]:
        """Return a JSON-friendly dict with the most recent ``max_logs`` lines."""
        data = asdict(self)
        data["status"] = self.status.value
        data["logs"] = self.logs[-max_logs:]
        return data


class JobManager:
    """Schedule and track framework jobs on the running event loop."""

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        if db_path is None:
            override = os.environ.get("AV_JOB_DB")
            db_path = Path(override) if override else _DEFAULT_DB
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, JobRecord] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        self._init_db()

    # ------------------------------------------------------------------ DB
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id     TEXT PRIMARY KEY,
                    job_type   TEXT NOT NULL,
                    status     TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    args       TEXT NOT NULL,
                    logs       TEXT NOT NULL,
                    result     TEXT
                )
                """
            )
            # Migration: columns used by the durable multi-process worker.
            existing = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)")}
            if "claimed_by" not in existing:
                conn.execute("ALTER TABLE jobs ADD COLUMN claimed_by TEXT")
            if "attempts" not in existing:
                conn.execute("ALTER TABLE jobs ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0")

    def _persist(self, rec: JobRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (job_id, job_type, status, created_at, updated_at, args, logs, result)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    status=excluded.status,
                    updated_at=excluded.updated_at,
                    logs=excluded.logs,
                    result=excluded.result
                """,
                (
                    rec.job_id,
                    rec.job_type,
                    rec.status.value,
                    rec.created_at,
                    rec.updated_at,
                    json.dumps(rec.args),
                    json.dumps(rec.logs[-_MAX_LOG_LINES:]),
                    json.dumps(rec.result) if rec.result is not None else None,
                ),
            )

    def _load(self, job_id: str) -> Optional[JobRecord]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        return JobRecord(
            job_id=row["job_id"],
            job_type=row["job_type"],
            status=JobStatus(row["status"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            args=json.loads(row["args"]),
            logs=json.loads(row["logs"]),
            result=json.loads(row["result"]) if row["result"] else None,
        )

    # -------------------------------------------------------------- public
    def is_allowed(self, job_type: str) -> bool:
        """Return whether ``job_type`` is a queueable framework job."""
        return job_type in ALLOWED_JOB_TYPES

    @staticmethod
    def _inline_execution_enabled() -> bool:
        """Whether this process executes jobs itself (default) or only
        enqueues them for a standalone worker (``AV_JOB_MODE=worker``)."""
        return os.environ.get("AV_JOB_MODE", "inline") != "worker"

    async def queue(self, job_type: str, extra_args: Optional[Sequence[str]] = None) -> str:
        """Queue a job and return its unique ``job_id``.

        The Pending row is persisted *before* anything runs, so the queue is
        durable: in ``AV_JOB_MODE=worker`` a standalone worker process claims
        and executes it; otherwise it is executed inline on this event loop.

        Raises :class:`ValueError` for unknown job types.
        """
        if not self.is_allowed(job_type):
            raise ValueError(f"Unknown job type: {job_type!r}")

        job_id = uuid.uuid4().hex
        now = time.time()
        cli_args = list(ALLOWED_JOB_TYPES[job_type]) + list(extra_args or [])
        rec = JobRecord(
            job_id=job_id,
            job_type=job_type,
            status=JobStatus.PENDING,
            created_at=now,
            updated_at=now,
            args=cli_args,
        )
        async with self._lock:
            self._jobs[job_id] = rec
            self._persist(rec)
            if self._inline_execution_enabled():
                self._tasks[job_id] = asyncio.create_task(self._run(job_id))
            else:
                self._notify_workers(job_id)
        return job_id

    def _notify_workers(self, job_id: str) -> None:
        """Fan out to a Redis broker when configured (best-effort)."""
        broker_url = os.environ.get("AV_JOB_BROKER")
        if not broker_url:
            return
        try:
            from src.jobs.broker import JobBroker

            async def _publish() -> None:
                broker = JobBroker(broker_url)
                await broker.connect()
                await broker.publish(job_id)

            asyncio.create_task(_publish())
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("broker publish failed for %s: %s", job_id, exc)

    def get(self, job_id: str) -> Optional[JobRecord]:
        """Return the current record for ``job_id`` (memory first, then SQLite)."""
        return self._jobs.get(job_id) or self._load(job_id)

    def list_jobs(self, limit: int = 25) -> list[JobRecord]:
        """Return the most recent job records, newest first (DB-backed)."""
        limit = max(1, min(int(limit), 200))
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT job_id FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        out: list[JobRecord] = []
        for row in rows:
            rec = self._jobs.get(row["job_id"]) or self._load(row["job_id"])
            if rec is not None:
                out.append(rec)
        return out

    def claim_pending(self, worker_id: str, limit: int = 1) -> list[JobRecord]:
        """Atomically claim up to ``limit`` Pending jobs for a worker process.

        Claimed rows transition to ``Running`` with ``claimed_by``/``attempts``
        recorded, so a crashed worker's jobs are visible and never double-run
        by two workers at once.
        """
        limit = max(1, int(limit))
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                "SELECT job_id FROM jobs WHERE status = ? ORDER BY created_at ASC LIMIT ?",
                (JobStatus.PENDING.value, limit),
            ).fetchall()
            now = time.time()
            for row in rows:
                conn.execute(
                    "UPDATE jobs SET status = ?, updated_at = ?, claimed_by = ?,"
                    " attempts = attempts + 1 WHERE job_id = ?",
                    (JobStatus.RUNNING.value, now, worker_id, row["job_id"]),
                )
        records: list[JobRecord] = []
        for row in rows:
            rec = self._load(row["job_id"])
            if rec is not None:
                self._jobs[rec.job_id] = rec
                records.append(rec)
        return records

    # -------------------------------------------------------------- worker
    def _touch(self, rec: JobRecord, status: Optional[JobStatus] = None) -> None:
        if status is not None:
            rec.status = status
        rec.updated_at = time.time()
        self._persist(rec)

    async def _run(self, job_id: str) -> None:
        rec = self._jobs[job_id]
        self._touch(rec, JobStatus.RUNNING)
        from src.jobs.executor import run_job_subprocess

        result, code = await run_job_subprocess(
            rec.args,
            repo_root=_REPO_ROOT,
            log_sink=lambda line: self._append_log(rec, line),
        )
        rec.result = result
        self._touch(rec, JobStatus.COMPLETED if code == 0 else JobStatus.FAILED)

    def _append_log(self, rec: JobRecord, line: str) -> None:
        """Append a log line, persisting periodically for live status polls."""
        rec.logs.append(line)
        if len(rec.logs) % 5 == 0:
            try:
                self._touch(rec)
            except Exception:  # pragma: no cover - defensive
                pass


_SINGLETON: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    """Return the process-wide :class:`JobManager` singleton."""
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = JobManager()
    return _SINGLETON


def reset_job_manager() -> None:
    """Drop the cached singleton (test isolation; next get re-reads env)."""
    global _SINGLETON
    _SINGLETON = None
