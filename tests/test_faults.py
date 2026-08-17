"""Tests for the whole-turbine fault detection system (src/faults)."""

import json
import os

import pytest

from src.digital_twin.specs import get_spec
from src.digital_twin.twin import WindTurbineDigitalTwin
from src.faults.detector import FaultDetector, all_fault_ids, covered_fault_ids
from src.faults.oil import analyze_oil, oil_analysis_from_telemetry
from src.faults.taxonomy import (
    FAULT_CATALOG,
    SEVERITIES,
    SUBSYSTEMS,
    catalog_summary,
    faults_by_subsystem,
    get_fault,
    list_faults,
)
from src.utils.schema import Telemetry


# --------------------------------------------------------------------------- #
# Taxonomy: every part has fault types                                         #
# --------------------------------------------------------------------------- #
def test_catalog_covers_all_twelve_subsystems():
    assert set(SUBSYSTEMS) == {
        "rotor_blades",
        "pitch",
        "hub_mainshaft",
        "gearbox",
        "hss_brake",
        "generator",
        "yaw",
        "tower_foundation",
        "nacelle_sensors",
        "cooling_hydraulics",
        "electrical",
        "scada",
    }


def test_every_subsystem_has_multiple_fault_types():
    # SCADA faults are by nature soft (LOW/MEDIUM); every hardware subsystem
    # must include at least one severe (HIGH/CRITICAL) type.
    soft_only = {"scada"}
    for subsystem in SUBSYSTEMS:
        faults = faults_by_subsystem(subsystem)
        assert len(faults) >= 4, f"{subsystem} needs more fault types"
        if subsystem not in soft_only:
            assert any(f.severity in ("HIGH", "CRITICAL") for f in faults), subsystem


def test_fault_ids_are_unique_and_prefixed():
    seen = set()
    for fault in FAULT_CATALOG:
        assert fault.fault_id not in seen
        seen.add(fault.fault_id)
        prefix = fault.fault_id.split("-")[0]
        assert prefix in {"RB", "PT", "HS", "GB", "BR", "GN", "YW", "TF", "NS", "CH", "EL", "SC"}


def test_fault_definitions_are_complete():
    for fault in FAULT_CATALOG:
        assert fault.name
        assert fault.description
        assert fault.severity in SEVERITIES
        assert fault.root_causes
        assert fault.symptoms
        assert fault.detection_signals
        assert fault.recommended_actions
        assert fault.subsystem_label == SUBSYSTEMS[fault.subsystem]


def test_gearbox_has_oil_fault_types():
    gearbox_faults = faults_by_subsystem("gearbox")
    names = " | ".join(f.name.lower() for f in gearbox_faults)
    for keyword in ("viscosity", "water", "particle", "filter", "level", "acid"):
        assert keyword in names


def test_catalog_helpers():
    assert get_fault("GB-02").name == "Oil viscosity too low"
    with pytest.raises(KeyError):
        get_fault("XX-99")
    with pytest.raises(KeyError):
        faults_by_subsystem("not-a-subsystem")
    summary = catalog_summary()
    assert summary["total_fault_types"] == len(FAULT_CATALOG)
    assert summary["n_subsystems"] == 12
    assert len(list_faults("gearbox")) == 14
    assert len(list_faults()) == len(FAULT_CATALOG)


def test_every_catalog_fault_has_detection_coverage():
    """Every fault type must be reachable by the detector (auto rule or an
    inspection flag handled inside a rule)."""
    uncovered = set(all_fault_ids()) - set(covered_fault_ids())
    # Inspection-only faults are detected through inspection_* flags in rules.
    assert not uncovered


# --------------------------------------------------------------------------- #
# Oil analysis                                                                 #
# --------------------------------------------------------------------------- #
def test_oil_analysis_healthy_snapshot():
    oil = analyze_oil(
        oil_viscosity_cst=30.0,
        oil_temp_c=60.0,
        oil_water_ppm=100.0,
        oil_particles_iso4406="17/15/12",
        oil_tan_mgkoh_g=0.4,
        oil_filter_dp_bar=0.8,
        oil_level_pct=80.0,
        oil_pressure_bar=3.0,
    )
    assert oil.overall_status == "OK"
    assert oil.health_score >= 90.0
    assert oil.n_alarms == 0 and oil.n_warnings == 0


def test_oil_analysis_contaminated_snapshot():
    oil = analyze_oil(
        oil_viscosity_cst=6.0,
        oil_water_ppm=1200.0,
        oil_particles_iso4406="20/18/16",
        oil_tan_mgkoh_g=2.4,
        oil_filter_dp_bar=2.8,
        oil_level_pct=8.0,
        oil_pressure_bar=0.8,
        oil_aeration_pct=18.0,
    )
    assert oil.overall_status == "ALARM"
    assert oil.n_alarms >= 5
    assert oil.health_score < 40.0
    statuses = {f.parameter: f.status for f in oil.findings}
    assert statuses["oil_viscosity_cst"] == "ALARM"
    assert statuses["oil_water_ppm"] == "ALARM"
    assert statuses["oil_particles_iso4406"] == "ALARM"
    assert statuses["oil_filter_dp_bar"] == "ALARM"
    assert statuses["oil_level_pct"] == "ALARM"


def test_oil_analysis_spec_viscosity_window():
    # A 9.5 cSt reading is fine for NREL-5MW (min 8) but a fault for GE-1.5 (min 10).
    oil_nrel = analyze_oil(oil_viscosity_cst=9.5, viscosity_min_cst=8.0, viscosity_max_cst=60.0)
    oil_ge = analyze_oil(oil_viscosity_cst=9.5, viscosity_min_cst=10.0, viscosity_max_cst=50.0)
    assert oil_nrel.overall_status == "OK"
    assert oil_ge.overall_status in ("WARN", "ALARM")


def test_oil_analysis_from_telemetry_missing_channels():
    oil = oil_analysis_from_telemetry({"vibration_mms": 2.0})
    assert oil.overall_status == "OK"
    assert oil.health_score == 100.0


# --------------------------------------------------------------------------- #
# Detector: healthy vs faulty snapshots                                        #
# --------------------------------------------------------------------------- #
HEALTHY = {
    "vibration_mms": 2.1,
    "temperature_c": 58.0,
    "oil_temp_c": 58.0,
    "rpm": 1400.0,
    "oil_viscosity_cst": 32.0,
    "load_pct": 70.0,
    "main_bearing_temp_c": 48.0,
    "generator_temp_c": 85.0,
    "oil_water_ppm": 80.0,
    "oil_particles_iso4406": "16/14/11",
    "oil_tan_mgkoh_g": 0.3,
    "oil_filter_dp_bar": 0.7,
    "oil_level_pct": 78.0,
    "oil_pressure_bar": 3.2,
    "wind_speed_mps": 8.0,
    "wind_speed2_mps": 8.2,
    "yaw_error_deg": 4.0,
    "blade_pitch_deviation_deg": 0.2,
    "brake_wear_pct": 35.0,
    "converter_temp_c": 52.0,
}


def test_healthy_snapshot_detects_nothing():
    report = FaultDetector().detect(HEALTHY, asset_id="WTG-T")
    assert report.n_faults == 0
    assert report.overall_status == "OK"
    assert report.health_score == 100.0
    assert report.oil.overall_status == "OK"


def test_oil_faults_found_by_detector():
    telemetry = dict(HEALTHY)
    telemetry.update(
        {
            "oil_viscosity_cst": 5.0,
            "oil_water_ppm": 1400.0,
            "oil_particles_iso4406": "21/19/17",
            "oil_tan_mgkoh_g": 2.6,
            "oil_filter_dp_bar": 3.0,
            "oil_level_pct": 6.0,
            "oil_pressure_bar": 0.7,
        }
    )
    report = FaultDetector().detect(telemetry, asset_id="WTG-T")
    ids = {f.fault_id for f in report.faults}
    assert {"GB-02", "GB-04", "GB-05", "GB-06", "GB-07", "GB-08", "GB-12"} <= ids
    assert report.oil.overall_status == "ALARM"
    assert report.overall_status in ("HIGH", "CRITICAL")


def test_each_subsystem_can_trigger_its_faults():
    detector = FaultDetector()
    scenarios = [
        # (fault_id, telemetry patch)
        ("RB-01", {"blade_1p_amplitude_mms": 2.8}),
        ("RB-06", {"blade_pitch_deviation_deg": 1.8}),
        ("PT-01", {"pitch_torque_pct": 82.0}),
        ("PT-04", {"pitch_hydraulic_pressure_bar": 95.0}),
        ("HS-01", {"main_bearing_temp_c": 72.0, "grease_debris_ppm": 600.0}),
        ("HS-02", {"shaft_axial_displacement_mm": 1.7}),
        ("GB-01", {"oil_temp_c": 88.0}),
        ("GB-09", {"gmf_sideband_amplitude": 3.2}),
        ("GB-11", {"bpfo_amplitude_mms": 2.2}),
        ("GB-13", {"oil_aeration_pct": 12.0}),
        ("BR-01", {"brake_wear_pct": 96.0}),
        ("BR-02", {"rpm": 1900.0}),
        ("BR-03", {"hss_vibration_mms": 5.2}),
        ("GN-01", {"stator_temp_c": 122.0}),
        ("GN-02", {"generator_bearing_temp_c": 92.0}),
        ("GN-03", {"motor_current_imbalance_pct": 11.0}),
        ("GN-05", {"coolant_flow_pct": 25.0}),
        ("GN-06", {"slip_ring_temp_c": 88.0}),
        ("GN-07", {"generator_vibration_mms": 8.0}),
        ("YW-01", {"yaw_torque_pct": 88.0}),
        ("YW-03", {"yaw_error_deg": 22.0}),
        ("YW-04", {"cable_twist_turns": 3.8}),
        ("YW-05", {"yaw_brake_pressure_bar": 50.0}),
        ("TF-01", {"tower_vibration_mms": 6.5}),
        ("TF-02", {"inspection_bolt_loose": True}),
        ("TF-03", {"tower_tilt_deg": 0.6}),
        ("NS-01", {"anemometer_stuck": True}),
        ("NS-03", {"nacelle_oscillation_mms": 3.4}),
        ("NS-05", {"nacelle_temp_c": 52.0}),
        ("CH-01", {"cooling_fan_fault": True}),
        ("CH-02", {"coolant_level_pct": 12.0}),
        ("CH-04", {"hydraulic_pressure_bar": 100.0}),
        ("CH-05", {"hydraulic_pressure_bar": 235.0}),
        ("CH-06", {"hydraulic_oil_particles_iso4406": "19/17/15"}),
        ("EL-01", {"converter_temp_c": 90.0}),
        ("EL-02", {"transformer_temp_c": 115.0}),
        ("EL-03", {"thd_pct": 9.0}),
        ("EL-04", {"partial_discharge_pc": 2500.0}),
        ("EL-05", {"dc_link_ripple_pct": 11.0}),
        ("SC-01", {"telemetry_gap_min": 150.0}),
        ("SC-02", {"sensor_disagreement_pct": 16.0}),
        ("SC-03", {"clock_skew_s": 90.0}),
        ("SC-04", {"controller_faults_24h": 4.0}),
    ]
    for fault_id, patch in scenarios:
        telemetry = dict(HEALTHY)
        telemetry.update(patch)
        report = detector.detect(telemetry, asset_id="WTG-T")
        ids = {f.fault_id for f in report.faults}
        assert fault_id in ids, f"{fault_id} not detected with {patch} (got {sorted(ids)})"


def test_critical_gearbox_tooth_breakage():
    telemetry = dict(HEALTHY)
    telemetry.update({"vibration_mms": 14.0, "oil_iron_ppm": 500.0})
    report = FaultDetector().detect(telemetry, asset_id="WTG-T")
    by_id = {f.fault_id: f for f in report.faults}
    assert by_id["GB-10"].severity == "CRITICAL"
    assert report.overall_status == "CRITICAL"
    assert report.health_score < 60.0


def test_spec_limits_are_honored():
    # NREL-5MW allows 5.0 mm/s vibration; GE-1.5 only 4.5 mm/s.
    nrel = FaultDetector(get_spec("NREL-5MW")).detect({"vibration_mms": 4.8})
    ge = FaultDetector(get_spec("GE-1.5")).detect({"vibration_mms": 4.8})
    assert nrel.n_faults == 0
    assert ge.n_faults >= 1


def test_confirmation_across_history_boosts_confidence():
    detector = FaultDetector()
    faulty = dict(HEALTHY)
    faulty.update({"oil_viscosity_cst": 6.0})
    first = detector.detect(faulty, history=[], asset_id="WTG-T")
    second = detector.detect(faulty, history=[faulty], asset_id="WTG-T")
    third = detector.detect(faulty, history=[faulty, faulty, faulty], asset_id="WTG-T")
    gb02_first = next(f for f in first.faults if f.fault_id == "GB-02")
    gb02_second = next(f for f in second.faults if f.fault_id == "GB-02")
    gb02_third = next(f for f in third.faults if f.fault_id == "GB-02")
    assert gb02_first.confirmations == 0 and gb02_first.new
    assert gb02_second.confirmations == 1 and not gb02_second.new
    assert gb02_third.confirmations == 3
    assert gb02_third.confidence > gb02_first.confidence


def test_report_serialization():
    telemetry = dict(HEALTHY)
    telemetry.update({"oil_viscosity_cst": 5.0, "yaw_error_deg": 25.0})
    report = FaultDetector().detect(telemetry, asset_id="WTG-7", timestamp="2026-08-17T00:00:00Z")
    body = report.to_dict()
    assert body["asset_id"] == "WTG-7"
    assert body["summary"]["n_faults"] == report.n_faults
    assert all("fault_id" in f and "recommended_actions" in f for f in body["faults"])
    # Deterministic ordering: severities ranked, then confidence.
    severities = [f["severity"] for f in body["faults"]]
    assert severities == sorted(
        severities, key=lambda s: {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}[s], reverse=True
    )


# --------------------------------------------------------------------------- #
# Digital twin integration                                                     #
# --------------------------------------------------------------------------- #
def test_twin_update_state_runs_fault_detection():
    twin = WindTurbineDigitalTwin("WTG-TW", get_spec("NREL-5MW"))
    healthy = Telemetry(
        vibration_mms=2.0,
        temperature_c=55.0,
        rpm=950.0,  # below NREL-5MW's 1173.7 rpm HSS limit
        oil_viscosity_cst=32.0,
        load_pct=70.0,
    )
    rec = twin.update_state(healthy)
    assert rec["fault_report"]["summary"]["n_faults"] == 0
    assert twin.last_fault_report is not None

    # Next snapshot: viscosity collapse + vibration spike -> faults appear.
    degraded = Telemetry(
        vibration_mms=6.2,
        temperature_c=72.0,
        rpm=1000.0,
        oil_viscosity_cst=6.0,
        load_pct=95.0,
    )
    rec2 = twin.update_state(degraded)
    ids = {f["fault_id"] for f in rec2["fault_report"]["faults"]}
    assert "GB-02" in ids
    assert rec2["fault_report"]["overall_status"] in ("HIGH", "CRITICAL")


# --------------------------------------------------------------------------- #
# CLI + examples                                                               #
# --------------------------------------------------------------------------- #
def test_example_fault_payload_detects_faults():
    path = os.path.join(os.path.dirname(__file__), "..", "examples", "fault_payload.json")
    with open(path, encoding="utf-8") as fh:
        payload = json.loads(fh.read())
    report = FaultDetector(get_spec(payload["model_key"])).detect(
        payload["telemetry"], asset_id=payload["asset_id"]
    )
    ids = {f.fault_id for f in report.faults}
    # Oil-condition faults + drivetrain faults must show up in the demo payload.
    assert {"GB-02", "GB-04", "GB-05", "GB-08", "GB-09", "GB-11"} <= ids
    assert report.oil.overall_status == "ALARM"
    assert report.overall_status in ("HIGH", "CRITICAL")
