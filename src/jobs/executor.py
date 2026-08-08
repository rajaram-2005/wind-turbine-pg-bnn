"""Shared subprocess execution for framework jobs.

Used by both the in-process :class:`src.jobs.manager.JobManager` (inline mode)
and the standalone multi-process worker (:mod:`src.jobs.worker`) so job
semantics are identical regardless of where the job actually runs:

* command is always ``[python, main.py, <subcommand>, ...args]`` from the
  repository root,
* stdout/stderr are merged and streamed line-by-line to a log sink,
  ``PYTHONUNBUFFERED=1`` keeps the stream live for status polling.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Callable

LogSink = Callable[[str], None]


async def run_job_subprocess(
    args: list[str],
    *,
    repo_root: Path,
    log_sink: LogSink,
) -> tuple[dict[str, Any], int]:
    """Execute ``python main.py <args>`` and stream its output.

    Returns ``(result_meta, exit_code)`` where ``result_meta`` is a small
    JSON-safe dict recorded on the job record.
    """
    cmd = [sys.executable, str(repo_root / "main.py"), *args]
    log_sink(f"$ {' '.join(cmd)}")
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(repo_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        assert proc.stdout is not None
        async for raw in proc.stdout:
            log_sink(raw.decode(errors="replace").rstrip())
        code = await proc.wait()
        if code == 0:
            return {"exit_code": 0}, code
        return {"exit_code": code}, code
    except asyncio.CancelledError:
        # Loop/process shutdown: never leak a child subprocess.
        log_sink("[job-executor] cancelled; terminating subprocess")
        if proc is not None and proc.returncode is None:
            proc.terminate()
        raise
    except Exception as exc:  # pragma: no cover - defensive
        log_sink(f"[job-executor] fatal: {exc!r}")
        return {"error": str(exc)}, -1
