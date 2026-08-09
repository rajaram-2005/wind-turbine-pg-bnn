"""Static contract checks for the native Flutter agent surfaces.

The Python CI image does not install a Flutter SDK, so these tests protect the
API wiring and named widgets that the separately built native client relies on.
"""

from pathlib import Path


FLUTTER_LIB = Path(__file__).resolve().parents[1] / "apps" / "aerovigilai_flutter" / "lib"


def test_api_service_uses_runtime_base_and_exposes_root_health():
    source = (FLUTTER_LIB / "services" / "api_service.dart").read_text(encoding="utf-8")
    assert "String.fromEnvironment('API_BASE'" in source
    assert "getHealth()" in source
    assert "_uri('/health')" in source


def test_dashboard_renders_live_agent_mesh_banner():
    source = (FLUTTER_LIB / "screens" / "dashboard_screen.dart").read_text(encoding="utf-8")
    assert "widget.api.getHealth()" in source
    assert "class _AgentMeshBanner" in source
    assert "CYBER PRIME DUAL AGENT" in source
    assert "mesh['evidence_path']" in source


def test_twin_renders_latest_state_agent_findings():
    source = (FLUTTER_LIB / "screens" / "digital_twin_screen.dart").read_text(encoding="utf-8")
    assert "last?['agent_team']" in source
    assert "class _TwinAgentBanner" in source
    assert "mika['finding']" in source
    assert "kai['finding']" in source
    assert "agreement_score_pct" in source
    assert "connected_sources" in source
