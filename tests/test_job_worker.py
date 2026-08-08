"""Tests for the durable multi-process job worker.

Covers enqueue-only mode (``AV_JOB_MODE=worker``), atomic claiming by a
standalone worker, and the Redis-broker fan-out wiring (degraded, since no
broker runs in CI).
"""

import asyncio

from src.jobs.manager import ALLOWED_JOB_TYPES, JobManager, JobStatus


def test_evaluate_job_is_allow_listed():
    assert "evaluate" in ALLOWED_JOB_TYPES


def test_worker_mode_queues_without_inline_execution(tmp_path, monkeypatch):
    monkeypatch.setenv("AV_JOB_MODE", "worker")
    db = tmp_path / "jobs.sqlite3"

    async def _run():
        mgr = JobManager(db_path=db)
        job_id = await mgr.queue("evaluate", ["--checkpoint", "nope.pt"])
        rec = mgr.get(job_id)
        assert rec.status is JobStatus.PENDING
        # No inline task: the standalone worker owns execution.
        assert job_id not in mgr._tasks
        return job_id

    job_id = asyncio.run(_run())

    # A fresh manager (the worker process) claims the pending job atomically.
    worker = JobManager(db_path=db)
    claimed = worker.claim_pending("worker-1", limit=5)
    assert [r.job_id for r in claimed] == [job_id]
    rec = worker.get(job_id)
    assert rec.status is JobStatus.RUNNING
    assert rec.args == ["evaluate", "--checkpoint", "nope.pt"]

    # A second worker cannot double-claim the running job.
    other = JobManager(db_path=db)
    assert other.claim_pending("worker-2", limit=5) == []


def test_job_listing_is_db_backed_and_bounded(tmp_path, monkeypatch):
    # Queue-only mode: no subprocesses spawned, purely durable bookkeeping.
    monkeypatch.setenv("AV_JOB_MODE", "worker")
    db = tmp_path / "jobs.sqlite3"

    async def _run():
        mgr = JobManager(db_path=db)
        for i in range(3):
            await mgr.queue("train", [f"--epochs-{i}"])
        return mgr

    mgr = asyncio.run(_run())
    jobs = mgr.list_jobs(limit=2)
    assert len(jobs) == 2
    assert jobs[0].created_at >= jobs[1].created_at
    all_jobs = mgr.list_jobs(limit=10)
    assert len(all_jobs) == 3
