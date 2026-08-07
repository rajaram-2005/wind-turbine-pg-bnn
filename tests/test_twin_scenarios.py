"""Tests for the shared Cyber Twin single/comparative scenario engine."""

import numpy as np
import pytest

from gradio_app.twin_scenarios import SCENARIO_CONFIGS, project_scenario

BASE = (5000, 12.5, 65.0, 80.0, 2000.0, 9.0, 1000.0)


def project(name):
    return project_scenario(name, *BASE)


def test_every_scenario_returns_finite_synchronized_projection():
    for name in SCENARIO_CONFIGS:
        result = project(name)
        assert result["scenario"] == name
        assert len(result["hours"]) == 160
        assert len(result["rul"]) == 160
        assert np.all(np.isfinite(result["rul"]))
        assert np.all(result["lower"] <= result["upper"])
        assert result["total_hours"] == pytest.approx(6000.0)
        assert 0.0 <= result["stress_pct"] <= 100.0
        assert result["energy_mwh"] >= 0.0


def test_derating_preserves_more_runway_than_overload():
    derated = project("Derated operation")
    overloaded = project("High wind overload")
    assert derated["final_rul"] > overloaded["final_rul"]
    assert derated["stress_pct"] < overloaded["stress_pct"]


def test_projection_is_deterministic():
    first = project("Grid frequency event")
    second = project("Grid frequency event")
    assert np.array_equal(first["rul"], second["rul"])
    assert first["final_rul"] == second["final_rul"]


def test_projection_validates_scenario_and_resolution():
    with pytest.raises(ValueError, match="unknown twin scenario"):
        project_scenario("Teleport", *BASE)
    with pytest.raises(ValueError, match="points"):
        project_scenario("Nominal operation", *BASE, points=1)
