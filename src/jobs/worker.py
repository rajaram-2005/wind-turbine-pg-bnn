"""Standalone durable job worker for multi-process production deployments.

The canonical deployment can run the API in *enqueue-only* mode
(``AV_JOB_MODE=worker``) while one or more worker processes drain the durable
SQLite queue — a Redis/Celery-style architecture with no mandatory broker:

.. code-block:: bash

    # terminal 1 – API process (enqueues only, never executes jobs)
    AV_JOB_MODE=worker uvicorn src.unified_app:app --port 8080

    # terminal 2 – worker processes (execute the jobs)
    python -m src.jobs.worker --workers 2 --db artifacts/jobs.sqlite3

The SQLite ``jobs`` table is the durable queue: ``Pending`` rows survive
restarts, are claimed atomically by workers (``claimed_by``/``attempts``) and
their logs/status are persisted as they stream. When ``--broker redis://...``
is given, job ids are additionally fanned out over a Redis list so workers
wake instantly instead of polling; SQLite remains the source of truth.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import signal
from pathlib import Path
from typing import Any

from src.jobs.executor import run_job_subprocess
from src.jobs.manager import JobManager, JobStatus

logger = logging.getLogger("jobs.worker")

_REPO_ROOT = Path(__file__).resolve().parents[2]


async def _execute_and_finish(mgr: JobManager, rec) -> None:
    """Run one claimed job to completion and persist the outcome."""
    try:
        result, code = await run_job_subprocess(
            rec.args, repo_root=_REPO_ROOT, log_sink=lambda line: _log(mgr, rec, line)
        )
        rec.result = result
        mgr._touch(rec, JobStatus.COMPLETED if code == 0 else JobStatus.FAILED)
    except Exception as exc:  # pragma: no cover - defensive
        _log(mgr, rec, f"[worker] fatal: {exc!r}")
        rec.result = {"error": str(exc)}
        mgr._touch(rec, JobStatus.FAILED)


def _log(mgr: JobManager, rec, line: str) -> None:
    rec.logs.append(line)
    # Persist periodically so a status poll sees fresh logs mid-run.
    if len(rec.logs) % 5 == 0:
        try:
            mgr._touch(rec)
        except Exception:  # pragma: no cover - DB hiccup must not kill worker
            logger.warning("could not persist logs for %s", rec.job_id)


async def _drain_once(mgr: JobManager, broker, workers: int) -> list[str]:
    """Claim and start up to ``workers`` jobs; returns their job ids."""
    ids: list[str] = []
    if broker is not None:
        while len(ids) < workers:
            job_id = await broker.consume(timeout=0.2)
            if job_id is None:
                break
            rec = mgr.get(job_id)
            if rec is not None and rec.status == JobStatus.PENDING:
                ids.append(job_id)
    if len(ids) < workers:
        ids += [r.job_id for r in mgr.claim_pending(f"worker-{workers}", limit=workers - len(ids))]
    return ids


async def run_worker(
    *,
    db_path: Path | str,
    workers: int = 1,
    poll_interval: float = 2.0,
    broker_url: (str | None) = None,
    once: bool = False,
) -> int:
    """Main worker loop: claim pending jobs and execute them concurrently."""
    mgr = JobManager(db_path=db_path)
    broker = None
    if broker_url:
        from src.jobs.broker import JobBroker

        broker = JobBroker(broker_url)
        await broker.connect()
        logger.info("broker connected: %s", broker_url)

    logger.info(
        "job worker started: workers=%d db=%s poll=%.1fs broker=%s",
        workers,
        mgr._db_path,
        poll_interval,
        broker_url or "polling",
    )
    while True:
        job_ids = await _drain_once(mgr, broker, workers)
        if job_ids:
            tasks = [asyncio.create_task(_execute_and_finish(mgr, mgr.get(jid))) for jid in job_ids]
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info("completed batch: %s", job_ids)
        if once:
            break
        await asyncio.sleep(poll_interval)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AeroVigil durable job worker")
    p.add_argument("--db", default="artifacts/jobs.sqlite3", help="Jobs SQLite database path")
    p.add_argument("--workers", type=int, default=1, help="Concurrent job slots")
    p.add_argument("--poll-interval", type=float, default=2.0, help="DB poll interval (s)")
    p.add_argument("--broker", default=None, help="Optional Redis broker URL (redis://...)")
    p.add_argument("--once", action="store_true", help="Drain currently pending jobs and exit")
    return p


def main(argv: (list[str] | None) = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    async def _amain() -> int:
        loop = asyncio.get_running_loop()
        stop = asyncio.Event()

        def _sig(*_: Any) -> None:
            stop.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):  # pragma: no cover - non-POSIX
                loop.add_signal_handler(sig, _sig)
        return await run_worker(
            db_path=args.db,
            workers=max(1, args.workers),
            poll_interval=max(0.2, args.poll_interval),
            broker_url=args.broker,
            once=args.once,
        )

    try:
        return asyncio.run(_amain())
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
