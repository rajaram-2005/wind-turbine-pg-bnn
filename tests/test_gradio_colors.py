"""Regression tests for Plotly-safe dashboard colors."""

import ast
import math
import re
from pathlib import Path

import pytest

from gradio_app.colors import hex_to_rgba


@pytest.mark.parametrize(
    ("hex_color", "opacity", "expected"),
    [
        ("#00e5a0", 0.15, "rgba(0,229,160,0.15)"),
        ("#00E5A0", 0.2, "rgba(0,229,160,0.2)"),
        ("#000000", 0, "rgba(0,0,0,0)"),
        ("#ffffff", 1, "rgba(255,255,255,1)"),
    ],
)
def test_hex_to_rgba_returns_plotly_safe_color(hex_color, opacity, expected):
    assert hex_to_rgba(hex_color, opacity) == expected


@pytest.mark.parametrize(
    "invalid_color",
    [
        "#00e5a015",  # #RRGGBBAA is not accepted by supported Plotly versions
        "#0ea5",  # #RGBA has the same compatibility problem
        "#00e5a",
        "#00e5a00",
        "00e5a0",
        "transparent",
        None,
    ],
)
def test_hex_to_rgba_rejects_non_rgb_hex_colors(invalid_color):
    with pytest.raises(ValueError, match="#RRGGBB"):
        hex_to_rgba(invalid_color, 0.15)


@pytest.mark.parametrize("invalid_opacity", [-0.01, 1.01, math.inf, math.nan, "0.15", True])
def test_hex_to_rgba_rejects_invalid_opacity(invalid_opacity):
    with pytest.raises(ValueError, match="opacity"):
        hex_to_rgba("#00e5a0", invalid_opacity)


def test_dashboard_does_not_construct_alpha_hex_colors():
    """Catch both literal and dynamic forms such as ``f"{color}15"``."""
    app_path = Path(__file__).parents[1] / "gradio_app" / "app.py"
    source = app_path.read_text(encoding="utf-8")
    assert re.search(r"(?<![0-9a-f])#[0-9a-f]{4}(?:[0-9a-f]{4})?(?![0-9a-f])", source, re.I) is None

    violations = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.JoinedStr):
            continue
        for value, following in zip(node.values, node.values[1:]):
            if not isinstance(value, ast.FormattedValue) or not isinstance(following, ast.Constant):
                continue
            expression = ast.unparse(value.value).lower()
            suffix = following.value[:2] if isinstance(following.value, str) else ""
            is_alpha_suffix = len(suffix) == 2 and all(
                char in "0123456789abcdef" for char in suffix.lower()
            )
            is_color = any(name in expression for name in ("color", "primary", "secondary"))
            if is_alpha_suffix and is_color:
                violations.append((node.lineno, expression, suffix))

    assert violations == []
