"""Module entrypoint for the unified AeroVigil application CLI.

``python -m src`` is equivalent to running the unified ``wind-turbine-bnn``
command (advisory, fleet, report, and the ``twin`` digital-twin group).
"""

from __future__ import annotations

from src.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
