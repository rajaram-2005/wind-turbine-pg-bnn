"""Asynchronous job queue for AeroVigil framework jobs.

This package provides a lightweight, dependency-minimal job queue built on
:mod:`asyncio`. Long-running framework operations (physics training,
federated training, ONNX export, active learning, SHAP explanation) are wrapped
as managed worker tasks so they never block the main API event loop.

Job state is persisted to a local SQLite database so status and recent logs
survive within a running process (and across quick restarts on the same host).
"""

from .manager import (
    ALLOWED_JOB_TYPES,
    JobManager,
    JobRecord,
    JobStatus,
    get_job_manager,
)

__all__ = [
    "ALLOWED_JOB_TYPES",
    "JobManager",
    "JobRecord",
    "JobStatus",
    "get_job_manager",
]
