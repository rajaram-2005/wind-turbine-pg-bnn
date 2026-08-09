"""Tests for the asyncio-backed framework job manager.

Covers the queue allow-list, SQLite-persisted status transitions, and a real
end-to-end run of a lightweight ``federated`` job driven through the same
``python main.py ...`` subprocess path used in production.
"""

import asyncio
import sqlite3

import pytest

from src.jobs import ALLOWED_JOB_TYPES, JobStatus
from src.jobs.manager import JobManager


def test_allow_list_covers_all_framework_jobs():
    for name in (
        "physics",
        "train",
        "federated",
        "export",
        "active-learning",
        "active-sample",
        "shap",
        "explain",
    ):
        assert name in ALLOWED_JOB_TYPES


def test_job_database_uses_wal_journaling(tmp_path):
    db_path = tmp_path / "jobs.sqlite3"
    JobManager(db_path=db_path)
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_queue_rejects_unknown_job_type(tmp_path):
    async def _run():
        mgr = JobManager(db_path=tmp_path / "jobs.sqlite3")
        assert mgr.is_allowed("train") is True
        assert mgr.is_allowed("bogus") is False
        with pytest.raises(ValueError):
            await mgr.queue("bogus")

    asyncio.run(_run())


def test_federated_job_runs_to_completion(tmp_path):
    """Queue a tiny federated job and poll until it reaches a terminal state."""

    async def _run():
        mgr = JobManager(db_path=tmp_path / "jobs.sqlite3")
        ckpt = tmp_path / "fed.pt"
        job_id = await mgr.queue(
            "federated",
            ["--rounds", "1", "--clients", "1", "--local-epochs", "1", "--checkpoint", str(ckpt)],
        )
        assert job_id
        rec = mgr.get(job_id)
        assert rec is not None
        assert rec.status in (JobStatus.PENDING, JobStatus.RUNNING)

        # Poll up to ~120s for the subprocess (imports torch) to finish.
        for _ in range(240):
            rec = mgr.get(job_id)
            if rec.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                break
            await asyncio.sleep(0.5)

        assert rec.status is JobStatus.COMPLETED, "\n".join(rec.logs[-20:])
        assert ckpt.exists()
        assert any("main.py" in line for line in rec.logs)

        # State survives a fresh manager via SQLite persistence.
        mgr2 = JobManager(db_path=tmp_path / "jobs.sqlite3")
        reloaded = mgr2.get(job_id)
        assert reloaded is not None
        assert reloaded.status is JobStatus.COMPLETED

    asyncio.run(_run())
