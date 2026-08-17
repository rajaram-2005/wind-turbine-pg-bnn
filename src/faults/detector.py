"""Rule-based whole-turbine fault detection engine.

:class:`FaultDetector` turns a telemetry snapshot (the five canonical SCADA
channels plus any optional condition-monitoring channels such as oil-water
content, ISO 4406 particle counts, yaw error, brake wear, converter
temperature ...) into a :class:`FaultReport` that lists every fault found in
every subsystem, ranked by severity, with confidence, evidence and
recommended actions.

Design notes
------------
* **Declarative rules** — each rule in :data:`_RULES` maps one ``fault_id``
  (from :mod:`src.faults.taxonomy`) to a small evaluator function.  Rules
  return ``None`` when the fault is not present and a
  ``(severity, confidence, evidence)`` tuple otherwise.
* **Spec-aware limits** — a :class:`src.digital_twin.specs.TurbineSpec`
  (e.g. from :data:`src.digital_twin.specs.SPECS_LIBRARY`) overrides the
  generic thresholds (vibration, temperature, RPM, viscosity window).
* **Oil analysis** — every report embeds an :class:`src.faults.oil.OilAnalysis`
  so the gearbox oil-condition faults are scored consistently with the
  dedicated oil module.
* **Confirmation across windows** — pass previous telemetry snapshots as
  ``history``; faults seen in previous windows get a confidence boost and are
  marked as confirmed, while first sightings are flagged as ``new``.
* **Inspection-only faults** (lightning damage, structural cracks, corrosion,
  bolt loosening) are reported when the corresponding inspection flags are
  present in the telemetry dict (``inspection_crack``, ``inspection_corrosion``,
  ``inspection_bolt_loose``, ``inspection_lightning_damage``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from src.faults.oil import OilAnalysis, oil_analysis_from_telemetry
from src.faults.taxonomy import FAULT_CATALOG, SUBSYSTEMS, Severity, get_fault

# Default limits used when no TurbineSpec is supplied.
DEFAULT_VIBRATION_LIMIT_MMS = 4.5
DEFAULT_TEMPERATURE_LIMIT_C = 80.0
DEFAULT_RPM_LIMIT_HSS = 1800.0
DEFAULT_VISCOSITY_MIN_CST = 10.0
DEFAULT_VISCOSITY_MAX_CST = 50.0

_SEVERITY_PENALTY: dict[Severity, float] = {
    "LOW": 4.0,
    "MEDIUM": 10.0,
    "HIGH": 18.0,
    "CRITICAL": 30.0,
}

_SEVERITY_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


@dataclass(frozen=True)
class DetectedFault:
    """One fault found by the detector in the current snapshot."""

    fault_id: str
    name: str
    subsystem: str
    subsystem_label: str
    severity: Severity
    confidence: float  # 0..1
    evidence: dict[str, float | str] = field(default_factory=dict)
    message: str = ""
    recommended_actions: list[str] = field(default_factory=list)
    confirmations: int = 0  # times seen in the supplied history windows
    new: bool = True  # True when this is the first sighting

    def to_dict(self) -> dict:
        return {
            "fault_id": self.fault_id,
            "name": self.name,
            "subsystem": self.subsystem,
            "subsystem_label": self.subsystem_label,
            "severity": self.severity,
            "confidence": round(self.confidence, 3),
            "evidence": dict(self.evidence),
            "message": self.message,
            "recommended_actions": list(self.recommended_actions),
            "confirmations": self.confirmations,
            "new": self.new,
        }


@dataclass(frozen=True)
class FaultReport:
    """Complete detection result for one asset snapshot."""

    asset_id: str
    timestamp: str
    faults: list[DetectedFault] = field(default_factory=list)
    oil: OilAnalysis = field(default_factory=OilAnalysis)
    health_score: float = 100.0
    overall_status: str = "OK"

    @property
    def n_faults(self) -> int:
        return len(self.faults)

    @property
    def n_critical(self) -> int:
        return sum(1 for f in self.faults if f.severity == "CRITICAL")

    @property
    def n_high(self) -> int:
        return sum(1 for f in self.faults if f.severity == "HIGH")

    @property
    def n_medium(self) -> int:
        return sum(1 for f in self.faults if f.severity == "MEDIUM")

    @property
    def n_low(self) -> int:
        return sum(1 for f in self.faults if f.severity == "LOW")

    def by_subsystem(self) -> dict[str, list[DetectedFault]]:
        grouped: dict[str, list[DetectedFault]] = {}
        for fault in self.faults:
            grouped.setdefault(fault.subsystem, []).append(fault)
        return grouped

    def to_dict(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "timestamp": self.timestamp,
            "health_score": round(self.health_score, 1),
            "overall_status": self.overall_status,
            "summary": {
                "n_faults": self.n_faults,
                "n_critical": self.n_critical,
                "n_high": self.n_high,
                "n_medium": self.n_medium,
                "n_low": self.n_low,
                "subsystems_affected": sorted(self.by_subsystem()),
            },
            "oil": self.oil.to_dict(),
            "faults": [f.to_dict() for f in self.faults],
        }


# --------------------------------------------------------------------------- #
# Rule helpers                                                                 #
# --------------------------------------------------------------------------- #
def _num(telemetry: dict, key: str) -> float | None:
    value = telemetry.get(key)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value != value:  # NaN
        return None
    return value


def _flag(telemetry: dict, key: str) -> bool:
    value = telemetry.get(key)
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in ("", "0", "false", "no", "none")
    return bool(value)


def _above(
    value: float, warn: float, alarm: float, severity_map: dict
) -> tuple[Severity, float] | None:
    """Direction 'value too high': returns (severity, confidence) or None.

    Warn edges are strict (value > warn) so a value sitting exactly on the
    reference limit is still healthy; alarm edges are inclusive.
    """
    if value >= alarm:
        severity, base = severity_map["ALARM"]
        ratio = min((value - warn) / max(alarm - warn, 1e-9), 3.0)
        return severity, min(0.99, base + 0.06 * ratio)
    if value > warn:
        warn_sev, warn_conf = severity_map["WARN"]
        return warn_sev, warn_conf
    return None


def _below(
    value: float, warn: float, alarm: float, severity_map: dict
) -> tuple[Severity, float] | None:
    """Direction 'value too low': returns (severity, confidence) or None."""
    if value <= alarm:
        severity, base = severity_map["ALARM"]
        ratio = min((warn - value) / max(warn - alarm, 1e-9), 3.0)
        return severity, min(0.99, base + 0.06 * ratio)
    if value < warn:
        warn_sev, warn_conf = severity_map["WARN"]
        return warn_sev, warn_conf
    return None


def _outside_band(
    value: float, lo: float, hi: float, margin: float, severity_map: dict
) -> tuple[Severity, float] | None:
    if value <= lo - margin or value >= hi + margin:
        alarm_sev, alarm_conf = severity_map["ALARM"]
        return alarm_sev, alarm_conf
    if value < lo or value > hi:
        warn_sev, warn_conf = severity_map["WARN"]
        return warn_sev, warn_conf
    return None


_ABOVE_MAP = {
    "WARN": ("MEDIUM", 0.62),
    "ALARM": ("HIGH", 0.82),
}
_ABOVE_CRIT_MAP = {
    "WARN": ("MEDIUM", 0.62),
    "ALARM": ("CRITICAL", 0.88),
}
_BELOW_MAP = {
    "WARN": ("MEDIUM", 0.62),
    "ALARM": ("HIGH", 0.82),
}
_BELOW_CRIT_MAP = {
    "WARN": ("HIGH", 0.7),
    "ALARM": ("CRITICAL", 0.9),
}


class _Limits:
    """Resolved detection limits (spec-aware, with generic fallbacks)."""

    def __init__(self, spec, overrides: dict[str, float] | None = None) -> None:
        if spec is None:
            self.vibration_limit_mms = DEFAULT_VIBRATION_LIMIT_MMS
            self.temperature_limit_c = DEFAULT_TEMPERATURE_LIMIT_C
            self.rpm_limit_hss = DEFAULT_RPM_LIMIT_HSS
            self.viscosity_min_cst = DEFAULT_VISCOSITY_MIN_CST
            self.viscosity_max_cst = DEFAULT_VISCOSITY_MAX_CST
        else:
            self.vibration_limit_mms = float(spec.vibration_limit_mms)
            self.temperature_limit_c = float(spec.temperature_limit_c)
            self.rpm_limit_hss = float(spec.rpm_limit_hss)
            self.viscosity_min_cst = float(spec.viscosity_min_cst)
            self.viscosity_max_cst = float(spec.viscosity_max_cst)
        if overrides:
            from src.faults.limits import apply_overrides

            apply_overrides(self, overrides)


# --------------------------------------------------------------------------- #
# Rules: one evaluator per fault id                                            #
# --------------------------------------------------------------------------- #
def _rule_rb01(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    amp = _num(t, "blade_1p_amplitude_mms")
    if amp is None:
        return None
    res = _above(amp, 1.5, 2.5, _ABOVE_MAP)
    if res is None:
        return None
    sev, conf = res
    return sev, conf, {"blade_1p_amplitude_mms": amp, "limit": "warn 1.5 / alarm 2.5 mm/s"}


def _rule_rb02(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    dev = _num(t, "aep_deviation_pct")
    if dev is None:
        return None
    res = _above(dev, 5.0, 10.0, _ABOVE_MAP)
    if res is None:
        return None
    sev, conf = res
    return sev, conf, {"aep_deviation_pct": dev, "limit": "warn 5 % / alarm 10 %"}


def _rule_rb03(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    if _flag(t, "ice_detector_on"):
        return "MEDIUM", 0.85, {"ice_detector_on": True}
    ambient = _num(t, "ambient_temp_c")
    power = _num(t, "power_kw")
    wind = _num(t, "wind_speed_mps")
    # Freezing conditions + poor power capture for the wind present.
    if (
        ambient is not None
        and ambient < 0.5
        and power is not None
        and wind is not None
        and wind > 7.0
        and power < 0.3 * (wind**3) * 0.5
    ):
        return (
            "MEDIUM",
            0.6,
            {
                "ambient_temp_c": ambient,
                "wind_speed_mps": wind,
                "power_kw": power,
                "hint": "freezing conditions with degraded power capture",
            },
        )
    return None


def _rule_rb04(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    strikes = _num(t, "lightning_events_24h")
    if strikes is not None:
        res = _above(strikes, 3.0, 10.0, _ABOVE_MAP)
        if res is not None:
            sev, conf = res
            return sev, conf, {"lightning_events_24h": strikes, "limit": "warn 3 / alarm 10 / 24 h"}
    if _flag(t, "inspection_lightning_damage"):
        return "HIGH", 0.9, {"inspection_lightning_damage": True}
    return None


def _rule_rb05(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    if _flag(t, "inspection_crack"):
        return "CRITICAL", 0.92, {"inspection_crack": t.get("inspection_crack")}
    if _flag(t, "blade_acoustic_anomaly"):
        return "HIGH", 0.75, {"blade_acoustic_anomaly": True}
    return None


def _rule_rb06(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    dev = _num(t, "blade_pitch_deviation_deg")
    if dev is None:
        return None
    res = _above(dev, 0.5, 1.5, _ABOVE_MAP)
    if res is None:
        return None
    sev, conf = res
    return sev, conf, {"blade_pitch_deviation_deg": dev, "limit": "warn 0.5° / alarm 1.5°"}


def _rule_pt01(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    torque = _num(t, "pitch_torque_pct")
    if torque is None:
        return None
    res = _above(torque, 60.0, 80.0, _ABOVE_MAP)
    if res is None:
        return None
    sev, conf = res
    return sev, conf, {"pitch_torque_pct": torque, "limit": "warn 60 % / alarm 80 %"}


def _rule_pt02(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    code = _num(t, "pitch_fault_code")
    if code is not None and code > 0:
        return "HIGH", 0.85, {"pitch_fault_code": code}
    torque = _num(t, "pitch_torque_pct")
    if torque is not None and torque >= 90.0:
        return "HIGH", 0.8, {"pitch_torque_pct": torque, "hint": "pitch drive overloaded"}
    return None


def _rule_pt03(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    if _flag(t, "pitch_sensor_disagreement"):
        return "HIGH", 0.85, {"pitch_sensor_disagreement": True}
    err = _num(t, "pitch_position_error_deg")
    if err is not None and err > 2.0:
        return "HIGH", 0.7, {"pitch_position_error_deg": err, "limit": "> 2.0°"}
    return None


def _rule_pt04(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    pressure = _num(t, "pitch_hydraulic_pressure_bar") or _num(t, "hydraulic_pressure_bar")
    if pressure is None:
        return None
    res = _below(pressure, 140.0, 110.0, _BELOW_CRIT_MAP)
    if res is None:
        return None
    sev, conf = res
    return sev, conf, {"hydraulic_pressure_bar": pressure, "limit": "warn < 140 / alarm < 110 bar"}


def _rule_pt05(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    feather = _num(t, "feather_time_s")
    if feather is not None:
        res = _above(feather, 3.5, 5.0, _ABOVE_MAP)
        if res is not None:
            sev, conf = res
            return sev, conf, {"feather_time_s": feather, "limit": "warn 3.5 s / alarm 5 s"}
    decay = _num(t, "hydraulic_pressure_decay_bar_s")
    if decay is not None and decay > 0.5:
        return "MEDIUM", 0.65, {"hydraulic_pressure_decay_bar_s": decay}
    return None


def _rule_pt06(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    err = _num(t, "pitch_position_error_deg")
    if err is None:
        return None
    res = _above(err, 0.5, 1.5, _ABOVE_MAP)
    if res is None:
        return None
    sev, conf = res
    return sev, conf, {"pitch_position_error_deg": err, "limit": "warn 0.5° / alarm 1.5°"}


def _rule_hs01(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    temp = _num(t, "main_bearing_temp_c")
    evidence: dict = {}
    if temp is not None:
        res = _above(temp, 60.0, 70.0, _ABOVE_MAP)
        if res is not None:
            sev, conf = res
            evidence["main_bearing_temp_c"] = temp
            evidence["limit"] = "warn 60 °C / alarm 70 °C"
    debris = _num(t, "grease_debris_ppm")
    if debris is not None:
        res = _above(debris, 200.0, 500.0, _ABOVE_MAP)
        if res is not None:
            sev, conf = res
            evidence["grease_debris_ppm"] = debris
            evidence["limit"] = "warn 200 / alarm 500 ppm"
    vib = _num(t, "vibration_mms")
    if vib is not None:
        # Broadband vibration above the spec limit is the classic main-bearing
        # wear signature (the project's headline RUL target).
        res = _above(vib, lim.vibration_limit_mms, 1.5 * lim.vibration_limit_mms, _ABOVE_MAP)
        if res is not None:
            sev, conf = res
            evidence["vibration_mms"] = vib
            evidence["limit"] = (
                f"warn {lim.vibration_limit_mms:g} / alarm {1.5 * lim.vibration_limit_mms:g} mm/s"
            )
    if not evidence:
        return None
    return sev, conf, evidence


def _rule_hs02(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    disp = _num(t, "shaft_axial_displacement_mm")
    if disp is None:
        return None
    res = _above(disp, 0.8, 1.5, _ABOVE_MAP)
    if res is None:
        return None
    sev, conf = res
    return sev, conf, {"shaft_axial_displacement_mm": disp, "limit": "warn 0.8 / alarm 1.5 mm"}


def _rule_hs03(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    crack = t.get("inspection_crack")
    if isinstance(crack, str) and "hub" in crack.lower():
        return "HIGH", 0.9, {"inspection_crack": crack}
    if _flag(t, "inspection_hub_crack"):
        return "HIGH", 0.9, {"inspection_hub_crack": True}
    return None


def _rule_hs04(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    spike = _num(t, "torque_spike_amplitude")
    if spike is None:
        return None
    res = _above(spike, 15.0, 30.0, _ABOVE_MAP)
    if res is None:
        return None
    sev, conf = res
    return sev, conf, {"torque_spike_amplitude": spike, "limit": "warn 15 % / alarm 30 %"}


def _rule_gb01(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    temp = _num(t, "oil_temp_c") or _num(t, "gearbox_temp_c") or _num(t, "temperature_c")
    if temp is None:
        return None
    limit = lim.temperature_limit_c
    warn = 0.92 * limit
    res = _above(temp, warn, limit, _ABOVE_MAP)
    if res is None:
        return None
    sev, conf = res
    return sev, conf, {"oil_temp_c": temp, "limit": f"warn {warn:.0f} / alarm {limit:.0f} °C"}


def _rule_gb02(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    visc = _num(t, "oil_viscosity_cst")
    if visc is None:
        return None
    lo = lim.viscosity_min_cst
    if visc < lo:
        if visc < 0.85 * lo:
            return (
                "HIGH",
                0.85,
                {
                    "oil_viscosity_cst": visc,
                    "limit": f"min {lo:g} cSt (alarm < {0.85 * lo:.1f})",
                },
            )
        return (
            "MEDIUM",
            0.62,
            {
                "oil_viscosity_cst": visc,
                "limit": f"min {lo:g} cSt",
            },
        )
    return None


def _rule_gb03(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    visc = _num(t, "oil_viscosity_cst")
    if visc is None:
        return None
    hi = lim.viscosity_max_cst
    if visc > hi:
        if visc > 1.15 * hi:
            return (
                "MEDIUM",
                0.8,
                {
                    "oil_viscosity_cst": visc,
                    "limit": f"max {hi:g} cSt (alarm > {1.15 * hi:.1f})",
                },
            )
        return "MEDIUM", 0.6, {"oil_viscosity_cst": visc, "limit": f"max {hi:g} cSt"}
    return None


def _rule_gb04(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    water = _num(t, "oil_water_ppm")
    if water is not None:
        res = _above(water, 300.0, 1000.0, _ABOVE_MAP)
        if res is not None:
            sev, conf = res
            return sev, conf, {"oil_water_ppm": water, "limit": "warn 300 / alarm 1000 ppm"}
    moisture = _num(t, "oil_moisture_pct")
    if moisture is not None:
        res = _above(moisture, 50.0, 80.0, _ABOVE_MAP)
        if res is not None:
            sev, conf = res
            return sev, conf, {"oil_moisture_pct": moisture, "limit": "warn 50 % / alarm 80 %"}
    return None


def _iso4406_largest(code) -> float | None:
    """Largest of the three ISO 4406 codes (e.g. '19/17/15' -> 19)."""
    if code is None:
        return None
    try:
        digits = [float(p) for p in str(code).replace(",", "/").split("/") if p.strip()]
    except (TypeError, ValueError):
        return None
    return max(digits) if digits else None


def _rule_gb05(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    code = t.get("oil_particles_iso4406")
    largest = _iso4406_largest(code)
    if largest is None:
        return None
    res = _above(largest, 17.0, 19.0, _ABOVE_MAP)
    if res is None:
        return None
    sev, conf = res
    return (
        sev,
        conf,
        {"oil_particles_iso4406": str(code), "limit": "target 17/15/12; alarm > 19/17/14"},
    )


def _rule_gb06(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    level = _num(t, "oil_level_pct")
    if level is None:
        return None
    res = _below(level, 20.0, 10.0, _BELOW_MAP)
    if res is None:
        return None
    sev, conf = res
    return sev, conf, {"oil_level_pct": level, "limit": "warn < 20 % / alarm < 10 %"}


def _rule_gb07(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    dp = _num(t, "oil_filter_dp_bar")
    if dp is None:
        return None
    res = _above(dp, 1.5, 2.5, _ABOVE_MAP)
    if res is None:
        return None
    sev, conf = res
    return sev, conf, {"oil_filter_dp_bar": dp, "limit": "warn 1.5 / alarm 2.5 bar"}


def _rule_gb08(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    tan = _num(t, "oil_tan_mgkoh_g")
    if tan is not None:
        res = _above(tan, 0.8, 2.0, _ABOVE_MAP)
        if res is not None:
            sev, conf = res
            return sev, conf, {"oil_tan_mgkoh_g": tan, "limit": "warn 0.8 / alarm 2.0 mg KOH/g"}
    ox = _num(t, "oil_oxidation_pct")
    if ox is not None and ox >= 40.0:
        return "MEDIUM", 0.7, {"oil_oxidation_pct": ox}
    return None


def _rule_gb09(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    sb = _num(t, "gmf_sideband_amplitude") or _num(t, "gmf_sideband_amplitude_mms")
    if sb is not None:
        res = _above(sb, 2.0, 4.0, _ABOVE_MAP)
        if res is not None:
            sev, conf = res
            return sev, conf, {"gmf_sideband_amplitude": sb, "limit": "warn 2 / alarm 4 mm/s"}
    iron = _num(t, "oil_iron_ppm")
    if iron is not None and iron >= 300.0:
        return "HIGH", 0.75, {"oil_iron_ppm": iron, "limit": "alarm >= 300 ppm"}
    return None


def _rule_gb10(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    vib = _num(t, "vibration_mms")
    iron = _num(t, "oil_iron_ppm")
    spike = _num(t, "torque_spike_amplitude")
    if vib is not None and vib > 2.5 * lim.vibration_limit_mms and (iron is None or iron >= 300.0):
        return (
            "CRITICAL",
            0.9,
            {
                "vibration_mms": vib,
                "oil_iron_ppm": iron,
                "hint": "extreme vibration with heavy wear debris",
            },
        )
    if spike is not None and spike >= 50.0:
        return "CRITICAL", 0.8, {"torque_spike_amplitude": spike, "limit": ">= 50 %"}
    return None


def _rule_gb11(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    bpfo = _num(t, "bpfo_amplitude_mms")
    if bpfo is not None:
        res = _above(bpfo, 1.5, 3.0, _ABOVE_MAP)
        if res is not None:
            sev, conf = res
            return sev, conf, {"bpfo_amplitude_mms": bpfo, "limit": "warn 1.5 / alarm 3 mm/s"}
    temp = _num(t, "bearing_temp_c")
    if temp is not None:
        res = _above(temp, 70.0, 80.0, _ABOVE_MAP)
        if res is not None:
            sev, conf = res
            return sev, conf, {"bearing_temp_c": temp, "limit": "warn 70 °C / alarm 80 °C"}
    return None


def _rule_gb12(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    pressure = _num(t, "oil_pressure_bar")
    if pressure is None:
        return None
    res = _below(pressure, 1.5, 1.0, _BELOW_MAP)
    if res is None:
        return None
    sev, conf = res
    return sev, conf, {"oil_pressure_bar": pressure, "limit": "warn < 1.5 / alarm < 1.0 bar"}


def _rule_gb13(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    aeration = _num(t, "oil_aeration_pct")
    if aeration is not None:
        res = _above(aeration, 6.0, 15.0, _ABOVE_MAP)
        if res is not None:
            sev, conf = res
            return sev, conf, {"oil_aeration_pct": aeration, "limit": "warn 6 % / alarm 15 %"}
    foam = _num(t, "oil_foam_pct")
    if foam is not None and foam >= 25.0:
        return "MEDIUM", 0.7, {"oil_foam_pct": foam}
    return None


def _rule_gb14(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    backlash = _num(t, "backlash_mm")
    if backlash is not None:
        res = _above(backlash, 0.3, 0.6, _ABOVE_MAP)
        if res is not None:
            sev, conf = res
            return sev, conf, {"backlash_mm": backlash, "limit": "warn 0.3 / alarm 0.6 mm"}
    rattle = _num(t, "gear_rattle_index")
    if rattle is not None and rattle >= 0.5:
        return "LOW", 0.55, {"gear_rattle_index": rattle}
    return None


def _rule_br01(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    wear = _num(t, "brake_wear_pct")
    if wear is None:
        return None
    res = _above(wear, 80.0, 95.0, _ABOVE_MAP)
    if res is None:
        return None
    sev, conf = res
    return sev, conf, {"brake_wear_pct": wear, "limit": "warn 80 % / alarm 95 %"}


def _rule_br02(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    rpm = _num(t, "rpm")
    if rpm is not None and rpm > lim.rpm_limit_hss:
        return "CRITICAL", 0.9, {"rpm": rpm, "limit": f"max {lim.rpm_limit_hss:g} rpm"}
    trips = _num(t, "overspeed_trips_24h")
    if trips is not None and trips >= 1.0:
        res = _above(trips, 1.0, 3.0, _ABOVE_MAP)
        if res is not None:
            sev, conf = res
            return sev, conf, {"overspeed_trips_24h": trips, "limit": "warn 1 / alarm 3 trips"}
    return None


def _rule_br03(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    vib = _num(t, "hss_vibration_mms")
    if vib is None:
        return None
    res = _above(vib, 4.5, 7.1, _ABOVE_MAP)
    if res is None:
        return None
    sev, conf = res
    return sev, conf, {"hss_vibration_mms": vib, "limit": "warn 4.5 / alarm 7.1 mm/s (ISO 10816)"}


def _rule_br04(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    temp = _num(t, "brake_temp_c")
    if temp is not None:
        res = _above(temp, 60.0, 80.0, _ABOVE_MAP)
        if res is not None:
            sev, conf = res
            return sev, conf, {"brake_temp_c": temp, "limit": "warn 60 °C / alarm 80 °C"}
    drag = _num(t, "brake_drag_current_pct")
    if drag is not None and drag > 10.0:
        return "MEDIUM", 0.65, {"brake_drag_current_pct": drag, "limit": "> 10 %"}
    return None


def _rule_gn01(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    temp = (
        _num(t, "stator_temp_c")
        or _num(t, "generator_temp_c")
        or _num(t, "generator_winding_temp_c")
    )
    if temp is None:
        return None
    res = _above(temp, 105.0, 120.0, _ABOVE_MAP)
    if res is None:
        return None
    sev, conf = res
    return sev, conf, {"stator_temp_c": temp, "limit": "warn 105 °C / alarm 120 °C"}


def _rule_gn02(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    temp = _num(t, "generator_bearing_temp_c")
    if temp is None:
        return None
    res = _above(temp, 75.0, 90.0, _ABOVE_MAP)
    if res is None:
        return None
    sev, conf = res
    return sev, conf, {"generator_bearing_temp_c": temp, "limit": "warn 75 °C / alarm 90 °C"}


def _rule_gn03(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    imbalance = _num(t, "motor_current_imbalance_pct")
    if imbalance is None:
        return None
    res = _above(imbalance, 5.0, 10.0, _ABOVE_MAP)
    if res is None:
        return None
    sev, conf = res
    return sev, conf, {"motor_current_imbalance_pct": imbalance, "limit": "warn 5 % / alarm 10 %"}


def _rule_gn04(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    ir = _num(t, "insulation_resistance_mohm")
    if ir is None:
        return None
    res = _below(ir, 100.0, 10.0, _BELOW_MAP)
    if res is None:
        return None
    sev, conf = res
    return sev, conf, {"insulation_resistance_mohm": ir, "limit": "warn < 100 / alarm < 10 MΩ"}


def _rule_gn05(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    flow = _num(t, "coolant_flow_pct")
    if flow is not None:
        res = _below(flow, 60.0, 30.0, _BELOW_MAP)
        if res is not None:
            sev, conf = res
            return sev, conf, {"coolant_flow_pct": flow, "limit": "warn < 60 % / alarm < 30 %"}
    pressure = _num(t, "coolant_pressure_bar")
    if pressure is not None and pressure < 1.5:
        return "HIGH", 0.7, {"coolant_pressure_bar": pressure}
    return None


def _rule_gn06(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    temp = _num(t, "slip_ring_temp_c")
    if temp is None:
        return None
    res = _above(temp, 70.0, 85.0, _ABOVE_MAP)
    if res is None:
        return None
    sev, conf = res
    return sev, conf, {"slip_ring_temp_c": temp, "limit": "warn 70 °C / alarm 85 °C"}


def _rule_gn07(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    vib = _num(t, "generator_vibration_mms")
    if vib is None:
        return None
    res = _above(vib, 4.5, 7.1, _ABOVE_MAP)
    if res is None:
        return None
    sev, conf = res
    return sev, conf, {"generator_vibration_mms": vib, "limit": "warn 4.5 / alarm 7.1 mm/s"}


def _rule_yw01(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    torque = _num(t, "yaw_torque_pct")
    if torque is None:
        return None
    res = _above(torque, 60.0, 85.0, _ABOVE_MAP)
    if res is None:
        return None
    sev, conf = res
    return sev, conf, {"yaw_torque_pct": torque, "limit": "warn 60 % / alarm 85 %"}


def _rule_yw02(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    faults = _num(t, "yaw_drive_faults_24h")
    if faults is not None and faults >= 1.0:
        res = _above(faults, 1.0, 3.0, _ABOVE_MAP)
        if res is not None:
            sev, conf = res
            return sev, conf, {"yaw_drive_faults_24h": faults, "limit": "warn 1 / alarm 3 faults"}
    torque = _num(t, "yaw_torque_pct")
    if torque is not None and torque >= 85.0:
        return "MEDIUM", 0.7, {"yaw_torque_pct": torque, "hint": "yaw drive overloaded"}
    return None


def _rule_yw03(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    err = _num(t, "yaw_error_deg")
    if err is None:
        return None
    res = _above(err, 10.0, 20.0, _ABOVE_MAP)
    if res is None:
        return None
    sev, conf = res
    return sev, conf, {"yaw_error_deg": err, "limit": "warn 10° / alarm 20°"}


def _rule_yw04(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    twist = _num(t, "cable_twist_turns")
    if twist is None:
        return None
    res = _above(twist, 2.5, 3.5, _ABOVE_MAP)
    if res is None:
        return None
    sev, conf = res
    return sev, conf, {"cable_twist_turns": twist, "limit": "warn 2.5 / alarm 3.5 turns"}


def _rule_yw05(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    pressure = _num(t, "yaw_brake_pressure_bar")
    if pressure is None:
        return None
    res = _below(pressure, 100.0, 60.0, _BELOW_MAP)
    if res is None:
        return None
    sev, conf = res
    return sev, conf, {"yaw_brake_pressure_bar": pressure, "limit": "warn < 100 / alarm < 60 bar"}


def _rule_tf01(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    vib = _num(t, "tower_vibration_mms")
    if vib is not None:
        res = _above(vib, 3.5, 6.0, _ABOVE_MAP)
        if res is not None:
            sev, conf = res
            return sev, conf, {"tower_vibration_mms": vib, "limit": "warn 3.5 / alarm 6 mm/s"}
    accel = _num(t, "tower_accel_g")
    if accel is not None:
        res = _above(accel, 0.15, 0.25, _ABOVE_MAP)
        if res is not None:
            sev, conf = res
            return sev, conf, {"tower_accel_g": accel, "limit": "warn 0.15 g / alarm 0.25 g"}
    return None


def _rule_tf02(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    if _flag(t, "inspection_bolt_loose"):
        return "HIGH", 0.9, {"inspection_bolt_loose": True}
    dev = _num(t, "bolt_tension_deviation_pct")
    if dev is not None:
        res = _above(dev, 10.0, 20.0, _ABOVE_MAP)
        if res is not None:
            sev, conf = res
            return sev, conf, {"bolt_tension_deviation_pct": dev, "limit": "warn 10 % / alarm 20 %"}
    return None


def _rule_tf03(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    tilt = _num(t, "tower_tilt_deg")
    if tilt is None:
        return None
    res = _above(tilt, 0.2, 0.5, _ABOVE_MAP)
    if res is None:
        return None
    sev, conf = res
    return sev, conf, {"tower_tilt_deg": tilt, "limit": "warn 0.2° / alarm 0.5°"}


def _rule_tf04(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    if _flag(t, "inspection_corrosion"):
        return "LOW", 0.85, {"inspection_corrosion": t.get("inspection_corrosion")}
    humidity = _num(t, "tower_humidity_pct")
    if humidity is not None and humidity >= 80.0:
        return "LOW", 0.55, {"tower_humidity_pct": humidity, "hint": "corrosion risk window"}
    return None


def _rule_ns01(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    if _flag(t, "anemometer_stuck"):
        return "MEDIUM", 0.85, {"anemometer_stuck": True}
    ws1 = _num(t, "wind_speed_mps")
    ws2 = _num(t, "wind_speed2_mps")
    if ws1 is not None and ws2 is not None and ws1 > 3.0:
        dev = abs(ws1 - ws2) / max(ws1, ws2) * 100.0
        if dev >= 20.0:
            return (
                "MEDIUM",
                0.7,
                {
                    "wind_speed_mps": ws1,
                    "wind_speed2_mps": ws2,
                    "disagreement_pct": round(dev, 1),
                    "limit": ">= 20 %",
                },
            )
    return None


def _rule_ns02(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    v1 = _num(t, "wind_vane_deg")
    v2 = _num(t, "wind_vane2_deg")
    if v1 is not None and v2 is not None:
        dev = abs((v1 - v2 + 180.0) % 360.0 - 180.0)
        if dev >= 15.0:
            return (
                "MEDIUM",
                0.7,
                {
                    "wind_vane_deg": v1,
                    "wind_vane2_deg": v2,
                    "disagreement_deg": round(dev, 1),
                    "limit": ">= 15°",
                },
            )
    return None


def _rule_ns03(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    osc = _num(t, "nacelle_oscillation_mms")
    if osc is None:
        return None
    res = _above(osc, 2.5, 5.0, _ABOVE_MAP)
    if res is None:
        return None
    sev, conf = res
    return sev, conf, {"nacelle_oscillation_mms": osc, "limit": "warn 2.5 / alarm 5 mm/s"}


def _rule_ns04(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    if _flag(t, "inspection_lightning_damage"):
        return "LOW", 0.8, {"inspection_lightning_damage": True}
    strikes = _num(t, "lightning_events_24h")
    if strikes is not None and strikes >= 3.0:
        return "LOW", 0.55, {"lightning_events_24h": strikes}
    return None


def _rule_ns05(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    temp = _num(t, "nacelle_temp_c")
    if temp is not None:
        res = _above(temp, 40.0, 50.0, _ABOVE_MAP)
        if res is not None:
            sev, conf = res
            return sev, conf, {"nacelle_temp_c": temp, "limit": "warn 40 °C / alarm 50 °C"}
    humidity = _num(t, "nacelle_humidity_pct")
    if humidity is not None and humidity > 80.0:
        return "LOW", 0.55, {"nacelle_humidity_pct": humidity, "hint": "condensation risk"}
    return None


def _rule_ns06(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    if _flag(t, "smoke_detector_on"):
        return "CRITICAL", 0.95, {"smoke_detector_on": True}
    temp = _num(t, "nacelle_temp_c")
    if temp is not None and temp >= 60.0:
        return "CRITICAL", 0.8, {"nacelle_temp_c": temp, "hint": "nacelle heat spike"}
    return None


def _rule_ch01(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    if _flag(t, "cooling_fan_fault"):
        return "MEDIUM", 0.85, {"cooling_fan_fault": True}
    coolant = _num(t, "coolant_temp_c")
    runtime = _num(t, "fan_runtime_pct")
    if coolant is not None and coolant > 55.0 and runtime is not None and runtime < 20.0:
        return (
            "MEDIUM",
            0.7,
            {
                "coolant_temp_c": coolant,
                "fan_runtime_pct": runtime,
                "hint": "hot coolant with the fan barely running",
            },
        )
    return None


def _rule_ch02(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    level = _num(t, "coolant_level_pct")
    if level is None:
        return None
    res = _below(level, 30.0, 15.0, _BELOW_MAP)
    if res is None:
        return None
    sev, conf = res
    return sev, conf, {"coolant_level_pct": level, "limit": "warn < 30 % / alarm < 15 %"}


def _rule_ch03(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    dt = _num(t, "heat_exchanger_delta_t")
    if dt is None:
        return None
    res = _above(dt, 12.0, 18.0, _ABOVE_MAP)
    if res is None:
        return None
    sev, conf = res
    return sev, conf, {"heat_exchanger_delta_t": dt, "limit": "warn 12 K / alarm 18 K approach"}


def _rule_ch04(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    pressure = _num(t, "hydraulic_pressure_bar")
    if pressure is None:
        return None
    res = _below(pressure, 140.0, 110.0, _BELOW_MAP)
    if res is None:
        return None
    sev, conf = res
    return sev, conf, {"hydraulic_pressure_bar": pressure, "limit": "warn < 140 / alarm < 110 bar"}


def _rule_ch05(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    pressure = _num(t, "hydraulic_pressure_bar")
    if pressure is None:
        return None
    res = _above(pressure, 210.0, 230.0, _ABOVE_MAP)
    if res is None:
        return None
    sev, conf = res
    return sev, conf, {"hydraulic_pressure_bar": pressure, "limit": "warn > 210 / alarm > 230 bar"}


def _rule_ch06(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    code = t.get("hydraulic_oil_particles_iso4406")
    largest = _iso4406_largest(code)
    if largest is not None:
        res = _above(largest, 17.0, 19.0, _ABOVE_MAP)
        if res is not None:
            sev, conf = res
            return sev, conf, {"hydraulic_oil_particles_iso4406": str(code)}
    water = _num(t, "hydraulic_oil_water_ppm")
    if water is not None and water > 300.0:
        return "MEDIUM", 0.7, {"hydraulic_oil_water_ppm": water}
    return None


def _rule_el01(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    temp = _num(t, "converter_temp_c")
    if temp is not None:
        res = _above(temp, 70.0, 85.0, _ABOVE_MAP)
        if res is not None:
            sev, conf = res
            return sev, conf, {"converter_temp_c": temp, "limit": "warn 70 °C / alarm 85 °C"}
    faults = _num(t, "converter_faults_24h")
    if faults is not None and faults >= 1.0:
        res = _above(faults, 1.0, 3.0, _ABOVE_MAP)
        if res is not None:
            sev, conf = res
            return sev, conf, {"converter_faults_24h": faults, "limit": "warn 1 / alarm 3 faults"}
    return None


def _rule_el02(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    temp = _num(t, "transformer_temp_c")
    if temp is not None:
        res = _above(temp, 95.0, 110.0, _ABOVE_MAP)
        if res is not None:
            sev, conf = res
            return sev, conf, {"transformer_temp_c": temp, "limit": "warn 95 °C / alarm 110 °C"}
    level = _num(t, "transformer_oil_level_pct")
    if level is not None and level < 30.0:
        return "MEDIUM", 0.6, {"transformer_oil_level_pct": level}
    return None


def _rule_el03(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    thd = _num(t, "thd_pct")
    if thd is not None:
        res = _above(thd, 5.0, 8.0, _ABOVE_MAP)
        if res is not None:
            sev, conf = res
            return sev, conf, {"thd_pct": thd, "limit": "warn 5 % / alarm 8 %"}
    unbalance = _num(t, "voltage_unbalance_pct")
    if unbalance is not None:
        res = _above(unbalance, 2.0, 4.0, _ABOVE_MAP)
        if res is not None:
            sev, conf = res
            return sev, conf, {"voltage_unbalance_pct": unbalance, "limit": "warn 2 % / alarm 4 %"}
    return None


def _rule_el04(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    pd = _num(t, "partial_discharge_pc")
    if pd is not None:
        res = _above(pd, 500.0, 2000.0, _ABOVE_MAP)
        if res is not None:
            sev, conf = res
            return sev, conf, {"partial_discharge_pc": pd, "limit": "warn 500 / alarm 2000 pC"}
    ir = _num(t, "insulation_resistance_mohm")
    if ir is not None and ir < 100.0:
        return "HIGH", 0.7, {"insulation_resistance_mohm": ir}
    return None


def _rule_el05(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    ripple = _num(t, "dc_link_ripple_pct")
    if ripple is None:
        return None
    res = _above(ripple, 5.0, 10.0, _ABOVE_MAP)
    if res is None:
        return None
    sev, conf = res
    return sev, conf, {"dc_link_ripple_pct": ripple, "limit": "warn 5 % / alarm 10 %"}


def _rule_sc01(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    gap = _num(t, "telemetry_gap_min")
    if gap is not None:
        res = _above(gap, 30.0, 120.0, _ABOVE_MAP)
        if res is not None:
            sev, conf = res
            return sev, conf, {"telemetry_gap_min": gap, "limit": "warn 30 / alarm 120 min"}
    uptime = _num(t, "comms_uptime_pct")
    if uptime is not None:
        res = _below(uptime, 99.0, 95.0, _BELOW_MAP)
        if res is not None:
            sev, conf = res
            return sev, conf, {"comms_uptime_pct": uptime, "limit": "warn < 99 % / alarm < 95 %"}
    return None


def _rule_sc02(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    dev = _num(t, "sensor_disagreement_pct")
    if dev is None:
        return None
    res = _above(dev, 5.0, 15.0, _ABOVE_MAP)
    if res is None:
        return None
    sev, conf = res
    return sev, conf, {"sensor_disagreement_pct": dev, "limit": "warn 5 % / alarm 15 %"}


def _rule_sc03(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    skew = _num(t, "clock_skew_s")
    if skew is None:
        return None
    res = _above(skew, 5.0, 60.0, _ABOVE_MAP)
    if res is None:
        return None
    sev, conf = res
    return sev, conf, {"clock_skew_s": skew, "limit": "warn 5 s / alarm 60 s"}


def _rule_sc04(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    faults = _num(t, "controller_faults_24h")
    if faults is not None and faults >= 1.0:
        res = _above(faults, 1.0, 3.0, _ABOVE_MAP)
        if res is not None:
            sev, conf = res
            return sev, conf, {"controller_faults_24h": faults, "limit": "warn 1 / alarm 3 faults"}
    restarts = _num(t, "restarts_24h")
    if restarts is not None and restarts >= 3.0:
        res = _above(restarts, 3.0, 5.0, _ABOVE_MAP)
        if res is not None:
            sev, conf = res
            return sev, conf, {"restarts_24h": restarts, "limit": "warn 3 / alarm 5 restarts"}
    return None


def _rule_rb07(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    """Blade fire / lightning fire (CRITICAL)."""
    if _flag(t, "blade_fire_alarm"):
        return "CRITICAL", 0.95, {"blade_fire_alarm": t.get("blade_fire_alarm")}
    blade_temp = _num(t, "blade_temp_c")
    strikes = _num(t, "lightning_events_24h")
    if blade_temp is not None and blade_temp >= 90.0:
        evidence: dict = {"blade_temp_c": blade_temp}
        if strikes is not None and strikes >= 1.0:
            evidence["lightning_events_24h"] = strikes
            evidence["hint"] = "hot blade after lightning activity"
        return "CRITICAL", 0.85, evidence
    return None


def _rule_rb08(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    """Blade tip deflection / excessive flex (MEDIUM)."""
    deflection = _num(t, "blade_tip_deflection_pct")
    if deflection is None:
        return None
    res = _above(deflection, 8.0, 15.0, _ABOVE_MAP)
    if res is None:
        return None
    sev, conf = res
    return sev, conf, {"blade_tip_deflection_pct": deflection, "limit": "warn 8 % / alarm 15 %"}


def _rule_rb09(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    """Trailing-edge split / delamination (HIGH)."""
    if _flag(t, "inspection_blade_delamination"):
        return "HIGH", 0.9, {"inspection_blade_delamination": True}
    if _flag(t, "blade_acoustic_anomaly"):
        dev = _num(t, "aep_deviation_pct")
        if dev is not None and dev > 5.0:
            return (
                "HIGH",
                0.75,
                {
                    "blade_acoustic_anomaly": True,
                    "aep_deviation_pct": dev,
                    "hint": "acoustic anomaly with energy loss",
                },
            )
    return None


def _rule_rb10(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    """Blade root bolt tension loss (HIGH)."""
    dev = _num(t, "blade_bolt_tension_deviation_pct")
    if dev is not None:
        res = _above(dev, 10.0, 20.0, _ABOVE_MAP)
        if res is not None:
            sev, conf = res
            return (
                sev,
                conf,
                {
                    "blade_bolt_tension_deviation_pct": dev,
                    "limit": "warn 10 % / alarm 20 %",
                },
            )
    if _flag(t, "inspection_bolt_loose"):
        detail = str(t.get("inspection_bolt_loose")).lower()
        if "blade" in detail or "root" in detail:
            return "HIGH", 0.9, {"inspection_bolt_loose": t.get("inspection_bolt_loose")}
    return None


def _rule_br05(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    """Brake fire (CRITICAL)."""
    if _flag(t, "brake_fire_alarm"):
        return "CRITICAL", 0.95, {"brake_fire_alarm": True}
    temp = _num(t, "brake_temp_c")
    if temp is not None and temp >= 120.0:
        return "CRITICAL", 0.9, {"brake_temp_c": temp, "hint": "brake thermal runaway"}
    if _flag(t, "smoke_detector_on"):
        temp = _num(t, "brake_temp_c")
        if temp is not None and temp >= 80.0:
            return "CRITICAL", 0.8, {"smoke_detector_on": True, "brake_temp_c": temp}
    return None


def _rule_gb15(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    """Gearbox oil fire (CRITICAL)."""
    oil_temp = _num(t, "oil_temp_c") or _num(t, "temperature_c")
    if oil_temp is not None and oil_temp >= 120.0:
        return (
            "CRITICAL",
            0.9,
            {"oil_temp_c": oil_temp, "hint": "oil temperature at ignition range"},
        )
    if (_flag(t, "oil_smoke_detector_on") or _flag(t, "smoke_detector_on")) and (
        oil_temp is not None and oil_temp >= 95.0
    ):
        return (
            "CRITICAL",
            0.85,
            {
                "smoke_detector_on": True,
                "oil_temp_c": oil_temp,
                "hint": "smoke with very hot gearbox oil",
            },
        )
    if _flag(t, "fire_suppression_released"):
        oil_temp = _num(t, "oil_temp_c") or _num(t, "temperature_c")
        if oil_temp is not None and oil_temp >= 95.0:
            return (
                "CRITICAL",
                0.85,
                {
                    "fire_suppression_released": True,
                    "oil_temp_c": oil_temp,
                },
            )
    return None


def _rule_el06(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    """Electrical cabinet / transformer fire (CRITICAL)."""
    if _flag(t, "cabinet_fire_alarm"):
        return "CRITICAL", 0.95, {"cabinet_fire_alarm": True}
    temp = _num(t, "transformer_temp_c")
    if _flag(t, "smoke_detector_on") and temp is not None and temp >= 110.0:
        return "CRITICAL", 0.85, {"smoke_detector_on": True, "transformer_temp_c": temp}
    if temp is not None and temp >= 130.0:
        return "CRITICAL", 0.9, {"transformer_temp_c": temp, "hint": "transformer thermal runaway"}
    if _flag(t, "fire_suppression_released") and temp is not None and temp >= 95.0:
        return "CRITICAL", 0.8, {"fire_suppression_released": True, "transformer_temp_c": temp}
    return None


def _rule_tf05(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    """Tower base fire (CRITICAL)."""
    if _flag(t, "tower_fire_alarm") or _flag(t, "tower_smoke_detector_on"):
        return (
            "CRITICAL",
            0.95,
            {
                "tower_fire_alarm": _flag(t, "tower_fire_alarm"),
                "tower_smoke_detector_on": _flag(t, "tower_smoke_detector_on"),
            },
        )
    return None


def _rule_ns07(t: dict, lim: _Limits) -> tuple[Severity, float, dict] | None:
    """Fire suppression system fault (HIGH)."""
    status = t.get("fire_suppression_status")
    if isinstance(status, str):
        lowered = status.strip().lower()
        if lowered in ("fault", "unavailable", "discharged", "expired", "maintenance"):
            return "HIGH", 0.9, {"fire_suppression_status": status}
    if _flag(t, "fire_suppression_fault"):
        return "HIGH", 0.85, {"fire_suppression_fault": True}
    if _flag(t, "fire_suppression_released") and not (
        _flag(t, "smoke_detector_on")
        or _flag(t, "blade_fire_alarm")
        or _flag(t, "cabinet_fire_alarm")
        or _flag(t, "tower_fire_alarm")
        or _flag(t, "brake_fire_alarm")
        or _flag(t, "oil_smoke_detector_on")
    ):
        # Suppression released without any fire evidence = system fault.
        return (
            "HIGH",
            0.7,
            {
                "fire_suppression_released": True,
                "hint": "suppression released without fire evidence",
            },
        )
    return None


# --------------------------------------------------------------------------- #
# Rule registry: every auto-detectable fault id -> evaluator                    #
# --------------------------------------------------------------------------- #
_RULES: dict[str, Callable[[dict, _Limits], tuple[Severity, float, dict] | None]] = {
    "RB-01": _rule_rb01,
    "RB-02": _rule_rb02,
    "RB-03": _rule_rb03,
    "RB-04": _rule_rb04,
    "RB-05": _rule_rb05,
    "RB-06": _rule_rb06,
    "RB-07": _rule_rb07,
    "RB-08": _rule_rb08,
    "RB-09": _rule_rb09,
    "RB-10": _rule_rb10,
    "PT-01": _rule_pt01,
    "PT-02": _rule_pt02,
    "PT-03": _rule_pt03,
    "PT-04": _rule_pt04,
    "PT-05": _rule_pt05,
    "PT-06": _rule_pt06,
    "HS-01": _rule_hs01,
    "HS-02": _rule_hs02,
    "HS-03": _rule_hs03,
    "HS-04": _rule_hs04,
    "GB-01": _rule_gb01,
    "GB-02": _rule_gb02,
    "GB-03": _rule_gb03,
    "GB-04": _rule_gb04,
    "GB-05": _rule_gb05,
    "GB-06": _rule_gb06,
    "GB-07": _rule_gb07,
    "GB-08": _rule_gb08,
    "GB-09": _rule_gb09,
    "GB-10": _rule_gb10,
    "GB-11": _rule_gb11,
    "GB-12": _rule_gb12,
    "GB-13": _rule_gb13,
    "GB-14": _rule_gb14,
    "GB-15": _rule_gb15,
    "BR-01": _rule_br01,
    "BR-02": _rule_br02,
    "BR-03": _rule_br03,
    "BR-04": _rule_br04,
    "BR-05": _rule_br05,
    "GN-01": _rule_gn01,
    "GN-02": _rule_gn02,
    "GN-03": _rule_gn03,
    "GN-04": _rule_gn04,
    "GN-05": _rule_gn05,
    "GN-06": _rule_gn06,
    "GN-07": _rule_gn07,
    "YW-01": _rule_yw01,
    "YW-02": _rule_yw02,
    "YW-03": _rule_yw03,
    "YW-04": _rule_yw04,
    "YW-05": _rule_yw05,
    "TF-01": _rule_tf01,
    "TF-02": _rule_tf02,
    "TF-03": _rule_tf03,
    "TF-04": _rule_tf04,
    "TF-05": _rule_tf05,
    "NS-01": _rule_ns01,
    "NS-02": _rule_ns02,
    "NS-03": _rule_ns03,
    "NS-04": _rule_ns04,
    "NS-05": _rule_ns05,
    "NS-06": _rule_ns06,
    "NS-07": _rule_ns07,
    "CH-01": _rule_ch01,
    "CH-02": _rule_ch02,
    "CH-03": _rule_ch03,
    "CH-04": _rule_ch04,
    "CH-05": _rule_ch05,
    "CH-06": _rule_ch06,
    "EL-01": _rule_el01,
    "EL-02": _rule_el02,
    "EL-03": _rule_el03,
    "EL-04": _rule_el04,
    "EL-05": _rule_el05,
    "EL-06": _rule_el06,
    "SC-01": _rule_sc01,
    "SC-02": _rule_sc02,
    "SC-03": _rule_sc03,
    "SC-04": _rule_sc04,
}


# --------------------------------------------------------------------------- #
# Detector                                                                    #
# --------------------------------------------------------------------------- #
class FaultDetector:
    """Whole-turbine fault detector over one or more telemetry snapshots."""

    def __init__(self, spec=None, overrides: dict[str, float] | None = None) -> None:
        self._limits = _Limits(spec, overrides=overrides)
        self._spec = spec
        self._overrides = dict(overrides) if overrides else {}

    @property
    def spec(self):
        return self._spec

    @property
    def effective_limits(self) -> dict[str, float]:
        """The resolved detection limits (spec + overrides) for reporting."""
        return {
            "vibration_limit_mms": self._limits.vibration_limit_mms,
            "temperature_limit_c": self._limits.temperature_limit_c,
            "rpm_limit_hss": self._limits.rpm_limit_hss,
            "viscosity_min_cst": self._limits.viscosity_min_cst,
            "viscosity_max_cst": self._limits.viscosity_max_cst,
        }

    def detect(
        self,
        telemetry: dict,
        history: list[dict] | None = None,
        asset_id: str = "WTG-000",
        timestamp: str = "",
    ) -> FaultReport:
        """Evaluate one snapshot; ``history`` supplies confirmation counts."""
        current = self._evaluate_snapshot(telemetry)

        confirmations: dict[str, int] = {}
        if history:
            for window in history[-3:]:
                for fault in self._evaluate_snapshot(window):
                    confirmations[fault.fault_id] = confirmations.get(fault.fault_id, 0) + 1

        faults: list[DetectedFault] = []
        for raw in current:
            fault_id = raw.fault_id
            conf = raw.confidence + 0.12 * min(confirmations.get(fault_id, 0), 3)
            faults.append(
                DetectedFault(
                    fault_id=fault_id,
                    name=raw.name,
                    subsystem=raw.subsystem,
                    subsystem_label=raw.subsystem_label,
                    severity=raw.severity,
                    confidence=min(0.99, conf),
                    evidence=raw.evidence,
                    message=raw.message,
                    recommended_actions=raw.recommended_actions,
                    confirmations=confirmations.get(fault_id, 0),
                    new=confirmations.get(fault_id, 0) == 0,
                )
            )

        faults.sort(
            key=lambda f: (
                -_SEVERITY_ORDER[f.severity],
                -f.confidence,
                f.fault_id,
            )
        )

        oil = oil_analysis_from_telemetry(
            telemetry,
            viscosity_min_cst=self._limits.viscosity_min_cst,
            viscosity_max_cst=self._limits.viscosity_max_cst,
        )
        score = 100.0 - sum(_SEVERITY_PENALTY[f.severity] for f in faults)
        if oil.overall_status == "ALARM":
            score -= 12.0
        elif oil.overall_status == "WARN":
            score -= 5.0
        score = max(0.0, min(100.0, score))
        if any(f.severity == "CRITICAL" for f in faults):
            status = "CRITICAL"
        elif any(f.severity == "HIGH" for f in faults) or oil.overall_status == "ALARM":
            status = "HIGH"
        elif any(f.severity == "MEDIUM" for f in faults) or oil.overall_status == "WARN":
            status = "MEDIUM"
        elif faults:
            status = "LOW"
        else:
            status = "OK"

        return FaultReport(
            asset_id=asset_id,
            timestamp=timestamp,
            faults=faults,
            oil=oil,
            health_score=score,
            overall_status=status,
        )

    def _evaluate_snapshot(self, telemetry: dict) -> list[DetectedFault]:
        found: list[DetectedFault] = []
        for fault_id, rule in _RULES.items():
            try:
                result = rule(telemetry, self._limits)
            except Exception:  # a bad channel value must not break detection
                continue
            if result is None:
                continue
            severity, confidence, evidence = result
            definition = get_fault(fault_id)
            found.append(
                DetectedFault(
                    fault_id=fault_id,
                    name=definition.name,
                    subsystem=definition.subsystem,
                    subsystem_label=definition.subsystem_label,
                    severity=severity,
                    confidence=confidence,
                    evidence=evidence,
                    message=_compose_message(definition.name, evidence),
                    recommended_actions=definition.recommended_actions,
                )
            )
        return found


def _compose_message(name: str, evidence: dict) -> str:
    parts = []
    for key, value in evidence.items():
        if key in ("limit", "hint"):
            continue
        if isinstance(value, float):
            parts.append(f"{key}={value:g}")
        else:
            parts.append(f"{key}={value}")
    snippet = ", ".join(parts)
    hint = evidence.get("hint")
    limit = evidence.get("limit")
    message = name
    if snippet:
        message += f" — {snippet}"
    if limit:
        message += f" (limit: {limit})"
    if hint:
        message += f"; {hint}"
    return message


def all_fault_ids() -> list[str]:
    """Every fault id in the catalog, in catalog order."""
    return [f.fault_id for f in FAULT_CATALOG]


def covered_fault_ids() -> set[str]:
    """Fault ids that have an automatic detection rule."""
    return set(_RULES)


def uncovered_fault_ids() -> set[str]:
    """Catalog faults without an automatic rule (inspection-flagged ones are
    still detected via their inspection flags inside other rules)."""
    return set(all_fault_ids()) - set(_RULES)


def subsystem_labels() -> dict[str, str]:
    return dict(SUBSYSTEMS)
