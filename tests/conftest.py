"""Shared pytest fixtures for the AeroVigil test-suite.

The durable store and job queue write to ``artifacts/*.sqlite3`` by default;
a polluted developer checkout could leak state into tests.  This session
fixture wipes those runtime databases once before the suite runs so every
test session starts hermetic (the Store re-creates its schema lazily).
"""

import pytest


@pytest.fixture(scope="session", autouse=True)
def _clean_runtime_databases():
    from pathlib import Path

    from src.data.store import reset_store
    from src.jobs.manager import reset_job_manager

    reset_store()
    reset_job_manager()
    repo_root = Path(__file__).resolve().parents[1]
    for pattern in ("artifacts/aerovigil.sqlite3*", "artifacts/jobs.sqlite3*"):
        for path in repo_root.glob(pattern):
            path.unlink(missing_ok=True)
    yield
    reset_store()
    reset_job_manager()
