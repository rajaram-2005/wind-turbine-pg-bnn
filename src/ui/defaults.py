"""UI defaults, threaded from configs/default.yaml.

Kept in a streamlit-free module so the values are importable (and testable)
without launching the Streamlit runtime. Values are identical to the previous
hardcoded ``_DEFAULTS`` — the config file is simply the single source of truth.
"""

from __future__ import annotations

from src.utils.config import load_config


def default_snapshot() -> dict[str, float]:
    """Default single-asset telemetry snapshot shown in the UI."""
    return load_config().ui.default_snapshot.model_dump()
