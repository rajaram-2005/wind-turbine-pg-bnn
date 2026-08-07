"""Tests for the dependency-free Cyber Prime digital-twin renderer."""

import re
from pathlib import Path

import pytest

from gradio_app.cyber_twin import (
    CYBER_TWIN_CSS,
    render_agent_answer,
    render_agent_council,
    render_component_diagnostics,
    render_cyber_twin,
    render_human_review_receipt,
    render_scenario_comparison,
    render_twin_result,
)


@pytest.fixture
def rendered_twin():
    return render_cyber_twin(
        scenario="High wind overload",
        vibration=22.5,
        bearing_temp=91.0,
        generator_temp=117.0,
        power=2450.0,
        wind=14.0,
        operating_hours=56000.0,
        rul_days=38.0,
        risk="MODERATE",
        accent="#ffd600",
    )


def test_cyber_twin_contains_animated_hud(rendered_twin):
    assert "CYBER PRIME" in rendered_twin
    assert "Animated holographic wind turbine digital twin" in rendered_twin
    assert "MIKA // AI" in rendered_twin
    assert "Maintenance strategist" in rendered_twin
    assert "KAI // PHYSICS" in rendered_twin
    assert "Constraint sentinel" in rendered_twin
    assert "High wind overload" in rendered_twin
    assert "38 <small>DAYS</small>" in rendered_twin
    assert "--ct-rotor-speed:" in rendered_twin
    assert "2 agents" in rendered_twin
    assert "consensus" in rendered_twin
    assert "Agent evidence mesh" in rendered_twin
    assert "POINTS LINKED" in rendered_twin


def test_cyber_twin_uses_plotly_safe_color_opacity(rendered_twin):
    assert "--ct-accent:#ffd600" in rendered_twin
    assert "--ct-soft:rgba(255,214,0,0.18)" in rendered_twin
    assert re.search(r"#[0-9a-fA-F]{8}\b", rendered_twin) is None


def test_cyber_twin_escapes_scenario_content():
    rendered = render_cyber_twin(
        '<script>alert("x")</script>', 10, 60, 80, 2000, 9, 1000, 200, "LOW", "#00e5a0"
    )
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered


def test_result_panel_maps_risk_to_escalation_steps():
    low = render_twin_result("Nominal", "LOW", "#00e5a0", 200, 6000, 1.0, 20)
    critical = render_twin_result("Overload", "CRITICAL", "#ff1744", 8, 6000, 1.5, 90)
    assert low.count("ct-result-step on") == 1
    assert critical.count("ct-result-step on") == 4
    assert "8 days" in critical
    assert "MIKA + KAI" in critical
    assert "dual-agent consensus complete" in critical


def test_cyber_twin_css_has_motion_and_reduced_motion_support():
    assert "@keyframes ct-rotor-spin" in CYBER_TWIN_CSS
    assert "@keyframes ct-scan" in CYBER_TWIN_CSS
    assert "prefers-reduced-motion" in CYBER_TWIN_CSS
    assert "@media(max-width:620px)" in CYBER_TWIN_CSS


def test_agent_mesh_connects_every_dashboard_surface():
    app_source = (Path(__file__).parents[1] / "gradio_app" / "app.py").read_text(encoding="utf-8")
    assert "create_agent_mesh_banner" in app_source
    assert "MIKA // MAINTENANCE" in app_source
    assert "KAI // PHYSICS" in app_source
    assert '"agent_team": agent_team' in app_source
    for point in ("SCADA", "PG-BNN", "ISO 281", "TWIN", "FLEET", "HUMAN"):
        assert point in app_source
    for feature in (
        "compare_twin_scenarios",
        "ask_cyber_agents",
        "record_human_twin_review",
        "review_history = gr.State",
    ):
        assert feature in app_source


def test_component_scan_and_agent_council_render_advanced_panels():
    components = render_component_diagnostics(22.5, 91.0, 117.0, 2450.0, 14.0, 38.0)
    council = render_agent_council("High wind overload", 22.5, 91.0, 2450.0, 38.0, "MODERATE")
    assert "COMPONENT RESONANCE SCAN" in components
    assert components.count('class="ct-component"') == 5
    assert "Main bearing" in components
    assert "KAI focus node" in components
    assert "DUAL-AGENT COUNCIL" in council
    assert "MIKA · Maintenance strategy" in council
    assert "KAI · Physics challenge" in council
    assert "Human gate required" in council


def test_agent_answer_and_human_review_escape_operator_content():
    answer = render_agent_answer(
        '<script>alert("question")</script>',
        '<img src=x onerror=alert("answer")>',
        "KAI",
        ["telemetry", "physics_constraints"],
    )
    receipt = render_human_review_receipt("Request engineering review", "Grid <event>")
    assert "<script>" not in answer
    assert "<img" not in answer
    assert "&lt;script&gt;" in answer
    assert "ENGINEERING REVIEW REQUESTED" in receipt
    assert "Grid &lt;event&gt;" in receipt
    with pytest.raises(ValueError, match="unknown human review"):
        render_human_review_receipt("Issue command", "Nominal")


def test_scenario_comparison_ranks_runway_and_labels_estimates():
    html = render_scenario_comparison(
        [
            {
                "scenario": "Overload",
                "accent": "#ff6d00",
                "final_rul": 20.0,
                "final_wear": 90.0,
                "stress_pct": 80.0,
                "energy_mwh": 1200.0,
                "risk": "HIGH",
            },
            {
                "scenario": "Derated",
                "accent": "#8be9ff",
                "final_rul": 180.0,
                "final_wear": 55.0,
                "stress_pct": 30.0,
                "energy_mwh": 700.0,
                "risk": "LOW",
            },
        ]
    )
    assert html.index("Derated") < html.index("Overload")
    assert "simulated energy" in html
    assert "not a commercial forecast" in html
