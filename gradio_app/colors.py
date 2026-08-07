"""Color helpers shared by the Gradio dashboard.

Plotly accepts ``#RGB`` and ``#RRGGBB`` hex colors, but not the CSS alpha-hex
forms ``#RGBA`` and ``#RRGGBBAA`` on all supported versions.  Keep opacity
explicit and emit ``rgba(...)`` whenever a transparent Plotly color is needed.
"""

from __future__ import annotations

import math
import re

_HEX_RGB = re.compile(r"^#(?P<red>[0-9a-f]{2})(?P<green>[0-9a-f]{2})(?P<blue>[0-9a-f]{2})$", re.I)


def hex_to_rgba(hex_color: str, opacity: float) -> str:
    """Convert ``#RRGGBB`` and an opacity to a Plotly-safe RGBA color.

    Args:
        hex_color: An opaque six-digit hex color such as ``#00e5a0``.
        opacity: A finite value from 0 (transparent) through 1 (opaque).

    Raises:
        ValueError: If the color is not exactly ``#RRGGBB`` or opacity is out
            of range. In particular, alpha-hex inputs such as ``#00e5a015``
            are rejected so they cannot accidentally be passed to Plotly.
    """
    if not isinstance(hex_color, str):
        raise ValueError("hex_color must be a #RRGGBB string")

    match = _HEX_RGB.fullmatch(hex_color)
    if match is None:
        raise ValueError(
            f"invalid hex color {hex_color!r}; use exactly #RRGGBB and pass opacity separately"
        )

    if isinstance(opacity, bool) or not isinstance(opacity, (int, float)):
        raise ValueError("opacity must be a number between 0 and 1")
    opacity = float(opacity)
    if not math.isfinite(opacity) or not 0.0 <= opacity <= 1.0:
        raise ValueError("opacity must be a finite number between 0 and 1")

    red, green, blue = (int(match.group(channel), 16) for channel in ("red", "green", "blue"))
    return f"rgba({red},{green},{blue},{opacity:g})"
