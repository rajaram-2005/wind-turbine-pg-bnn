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

SAFETY_BANNER = (
    f"{PRODUCT} v{APP_VERSION} (wind-turbine-pg-bnn advisory service) — "
    "DECISION-SUPPORT ONLY. https://aerovigil.abacusai.app. "
    "Outputs are not actuation commands; review by a qualified operator "
    "and an OEM documentation cross-check are required before any "
    "maintenance action. See docs/SAFETY.md."
)
