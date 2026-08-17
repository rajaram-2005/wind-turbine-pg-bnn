"""Single source of truth for the AeroVigil application surface metadata.

Every runnable application entry point (``src.unified_app``, the operations
API ``src.api.app``, the CLI ``src.cli``) imports product identity, version,
and the safety banner from here so deployments cannot drift between surfaces.

The packaged model wheel (``src.aerovigil_pg_bnn``) stays self-contained and
does NOT import this module — it pins its own ``__version__``.
"""

from __future__ import annotations

APP_VERSION = "1.0.0"
PRODUCT = "AeroVigil"
WEBSITE = "https://aerovigil.abacusai.app"

# Enterprise Long-Term Support (LTS) Lifecycle: August 2026 – August 2029 (3-Year Cycle)
IS_LTS = True
LTS_RELEASE_TAG = "v1.0.0"
LTS_START_DATE = "2026-08-17"
LTS_END_DATE = "2029-08-17"
LTS_CYCLE_YEARS = 3
NEXT_MAJOR_UPDATE = "2029-08-17"
LTS_STATUS = "Active LTS (Supported through August 2029; next major release cycle: 2029)"

SAFETY_BANNER = (
    f"{PRODUCT} v{APP_VERSION} (wind-turbine-pg-bnn advisory service) — "
    "DECISION-SUPPORT ONLY. https://aerovigil.abacusai.app. "
    "Outputs are not actuation commands; review by a qualified operator "
    "and an OEM documentation cross-check are required before any "
    "maintenance action. See docs/SAFETY.md."
)


def get_lts_info() -> dict[str, object]:
    """Return structured metadata about the current release and LTS support window."""
    return {
        "product": PRODUCT,
        "version": APP_VERSION,
        "is_lts": IS_LTS,
        "lts_tag": LTS_RELEASE_TAG,
        "lts_start": LTS_START_DATE,
        "lts_end": LTS_END_DATE,
        "support_duration_years": LTS_CYCLE_YEARS,
        "next_major_update": NEXT_MAJOR_UPDATE,
        "status": LTS_STATUS,
        "website": WEBSITE,
    }
