"""Gearbox oil-condition analysis: the 'using oil' fault finder.

Wind-turbine gearbox faults are discovered largely *through the oil*: viscosity,
water, particles, acidity, filter differential pressure, level, aeration and
pressure.  This module turns raw oil-condition readings into actionable
findings and a single 0–100 oil health score.

Typical healthy reference windows (ISO 4406 targets for wind gearboxes,
common OEM guidance):

* kinematic viscosity at operating temperature ... ``[10, 50] cSt``
* water content ........................ ``< 300 ppm`` (alarm ``>= 1000 ppm``)
* moisture (% saturation) .............. ``< 50 %``   (alarm ``>= 80 %``)
* ISO 4406 cleanliness ................. ``<= 17/15/12`` (alarm ``> 19/17/14``)
* total acid number (TAN) .............. ``< 0.8 mg KOH/g`` (alarm ``>= 2.0``)
* filter differential pressure ......... ``< 1.5 bar`` (alarm ``>= 2.5 bar``)
* oil level ............................ ``>= 50 %``  (alarm ``< 10 %``)
* oil pressure (if monitored) .......... ``>= 1.5 bar`` (alarm ``< 1.0 bar``)
* air entrainment ...................... ``< 6 %``    (alarm ``>= 15 %``)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Status = Literal["OK", "WARN", "ALARM"]

_DEFAULT_LIMITS: dict[str, dict[str, float]] = {
    "viscosity": {"min": 10.0, "max": 50.0},
    "water_ppm": {"warn": 300.0, "alarm": 1000.0},
    "moisture_pct": {"warn": 50.0, "alarm": 80.0},
    "iso4406": {"warn": 17.0, "alarm": 19.0},  # compare against the largest code
    "tan": {"warn": 0.8, "alarm": 2.0},
    "filter_dp_bar": {"warn": 1.5, "alarm": 2.5},
    "level_pct": {"warn": 20.0, "alarm": 10.0},
    "pressure_bar": {"warn": 1.5, "alarm": 1.0},
    "aeration_pct": {"warn": 6.0, "alarm": 15.0},
    "foam_pct": {"warn": 10.0, "alarm": 25.0},
    "iron_ppm": {"warn": 100.0, "alarm": 300.0},
}

# Severity contributed to the oil health score by each non-OK finding.
_SEVERITY_PENALTY = {"OK": 0.0, "WARN": 8.0, "ALARM": 22.0}


@dataclass(frozen=True)
class OilFinding:
    """One oil-condition parameter verdict."""

    parameter: str
    label: str
    value: float | None
    unit: str
    status: Status
    limit: str
    message: str

    def to_dict(self) -> dict:
        return {
            "parameter": self.parameter,
            "label": self.label,
            "value": self.value,
            "unit": self.unit,
            "status": self.status,
            "limit": self.limit,
            "message": self.message,
        }


@dataclass(frozen=True)
class OilAnalysis:
    """Complete oil-condition assessment for one snapshot."""

    findings: list[OilFinding] = field(default_factory=list)
    health_score: float = 100.0
    overall_status: Status = "OK"

    def to_dict(self) -> dict:
        return {
            "health_score": round(self.health_score, 1),
            "overall_status": self.overall_status,
            "findings": [f.to_dict() for f in self.findings],
        }

    @property
    def n_warnings(self) -> int:
        return sum(1 for f in self.findings if f.status == "WARN")

    @property
    def n_alarms(self) -> int:
        return sum(1 for f in self.findings if f.status == "ALARM")


def _status_below(value: float, warn: float, alarm: float) -> Status:
    """Value too high: warn edges are strict, alarm edges are inclusive."""
    if value >= alarm:
        return "ALARM"
    if value > warn:
        return "WARN"
    return "OK"


def _status_above(value: float, warn: float, alarm: float) -> Status:
    """Value too low: warn edges are strict, alarm edges are inclusive."""
    if value <= alarm:
        return "ALARM"
    if value < warn:
        return "WARN"
    return "OK"


def _status_band(value: float, lo: float, hi: float, margin: float) -> Status:
    """In-band is OK; outside the band by more than ``margin`` is ALARM."""
    if value <= lo - margin or value >= hi + margin:
        return "ALARM"
    if value < lo or value > hi:
        return "WARN"
    return "OK"


def _iso4406_max_code(code: float | None) -> float:
    """Largest of the three ISO 4406 codes, e.g. 19/17/14 -> 19."""
    if code is None:
        return 0.0
    return max(float(p) for p in str(code).replace(",", "/").split("/") if p.strip())


def analyze_oil(
    *,
    oil_viscosity_cst: float | None = None,
    oil_temp_c: float | None = None,
    oil_water_ppm: float | None = None,
    oil_moisture_pct: float | None = None,
    oil_particles_iso4406: float | None = None,
    oil_tan_mgkoh_g: float | None = None,
    oil_filter_dp_bar: float | None = None,
    oil_level_pct: float | None = None,
    oil_pressure_bar: float | None = None,
    oil_aeration_pct: float | None = None,
    oil_foam_pct: float | None = None,
    oil_iron_ppm: float | None = None,
    viscosity_min_cst: float | None = None,
    viscosity_max_cst: float | None = None,
) -> OilAnalysis:
    """Score one oil snapshot against reference limits.

    ``viscosity_min_cst`` / ``viscosity_max_cst`` override the default
    :data:`_DEFAULT_LIMITS` window (e.g. from a ``TurbineSpec``).  Parameters
    left ``None`` are simply not assessed (no penalty).
    """
    lo = viscosity_min_cst if viscosity_min_cst is not None else _DEFAULT_LIMITS["viscosity"]["min"]
    hi = viscosity_max_cst if viscosity_max_cst is not None else _DEFAULT_LIMITS["viscosity"]["max"]
    margin = max(0.05 * (hi - lo), 1.0)

    findings: list[OilFinding] = []

    def add(
        parameter: str,
        label: str,
        value: float | None,
        unit: str,
        status: Status,
        limit: str,
        message: str,
    ) -> None:
        findings.append(
            OilFinding(
                parameter=parameter,
                label=label,
                value=value,
                unit=unit,
                status=status,
                limit=limit,
                message=message,
            )
        )

    if oil_viscosity_cst is not None:
        status = _status_band(oil_viscosity_cst, lo, hi, margin)
        add(
            "oil_viscosity_cst",
            "Kinematic viscosity",
            oil_viscosity_cst,
            "cSt",
            status,
            f"window [{lo:g}, {hi:g}] cSt (±{margin:g} margin)",
            {
                "OK": "Viscosity within the operating window.",
                "WARN": "Viscosity near/outside the window — monitor and sample.",
                "ALARM": "Viscosity outside the operating window — oil condition degraded.",
            }[status],
        )

    if oil_water_ppm is not None:
        status = _status_below(
            oil_water_ppm,
            _DEFAULT_LIMITS["water_ppm"]["warn"],
            _DEFAULT_LIMITS["water_ppm"]["alarm"],
        )
        add(
            "oil_water_ppm",
            "Water content",
            oil_water_ppm,
            "ppm",
            status,
            "warn >= 300 ppm, alarm >= 1000 ppm",
            {
                "OK": "Water content acceptable.",
                "WARN": "Elevated water — check desiccant breather and cooler.",
                "ALARM": "Water contamination — find the ingress path, filter/drain.",
            }[status],
        )

    if oil_moisture_pct is not None:
        status = _status_below(
            oil_moisture_pct,
            _DEFAULT_LIMITS["moisture_pct"]["warn"],
            _DEFAULT_LIMITS["moisture_pct"]["alarm"],
        )
        add(
            "oil_moisture_pct",
            "Moisture saturation",
            oil_moisture_pct,
            "%",
            status,
            "warn >= 50 %, alarm >= 80 %",
            {
                "OK": "Moisture saturation acceptable.",
                "WARN": "Moisture rising — monitor trend.",
                "ALARM": "High moisture — risk of corrosion and emulsion.",
            }[status],
        )

    if oil_particles_iso4406 is not None:
        code = _iso4406_max_code(oil_particles_iso4406)
        status = _status_below(
            code, _DEFAULT_LIMITS["iso4406"]["warn"], _DEFAULT_LIMITS["iso4406"]["alarm"]
        )
        add(
            "oil_particles_iso4406",
            "Particle count (ISO 4406)",
            oil_particles_iso4406,
            "code",
            status,
            "largest code: warn > 17, alarm > 19 (target <= 17/15/12)",
            {
                "OK": "Cleanliness within target.",
                "WARN": "Particle count elevated — check filters and breather.",
                "ALARM": "Heavy particle load — abrasive wear risk, filter + analyze.",
            }[status],
        )

    if oil_tan_mgkoh_g is not None:
        status = _status_below(
            oil_tan_mgkoh_g,
            _DEFAULT_LIMITS["tan"]["warn"],
            _DEFAULT_LIMITS["tan"]["alarm"],
        )
        add(
            "oil_tan_mgkoh_g",
            "Total acid number (TAN)",
            oil_tan_mgkoh_g,
            "mg KOH/g",
            status,
            "warn >= 0.8, alarm >= 2.0",
            {
                "OK": "Acidity acceptable.",
                "WARN": "Oxidation starting — monitor TAN trend.",
                "ALARM": "Oil oxidized — plan oil change.",
            }[status],
        )

    if oil_filter_dp_bar is not None:
        status = _status_below(
            oil_filter_dp_bar,
            _DEFAULT_LIMITS["filter_dp_bar"]["warn"],
            _DEFAULT_LIMITS["filter_dp_bar"]["alarm"],
        )
        add(
            "oil_filter_dp_bar",
            "Filter differential pressure",
            oil_filter_dp_bar,
            "bar",
            status,
            "warn >= 1.5, alarm >= 2.5",
            {
                "OK": "Filter ΔP normal.",
                "WARN": "Filter loading — plan element change.",
                "ALARM": "Filter clogged — bypass may open, change immediately.",
            }[status],
        )

    if oil_level_pct is not None:
        status = _status_above(
            oil_level_pct,
            _DEFAULT_LIMITS["level_pct"]["warn"],
            _DEFAULT_LIMITS["level_pct"]["alarm"],
        )
        add(
            "oil_level_pct",
            "Oil level",
            oil_level_pct,
            "%",
            status,
            "warn < 20 %, alarm < 10 %",
            {
                "OK": "Oil level acceptable.",
                "WARN": "Oil level low — top up and look for leaks.",
                "ALARM": "Oil level critical — starvation risk.",
            }[status],
        )

    if oil_pressure_bar is not None:
        status = _status_above(
            oil_pressure_bar,
            _DEFAULT_LIMITS["pressure_bar"]["warn"],
            _DEFAULT_LIMITS["pressure_bar"]["alarm"],
        )
        add(
            "oil_pressure_bar",
            "Oil supply pressure",
            oil_pressure_bar,
            "bar",
            status,
            "warn < 1.5, alarm < 1.0",
            {
                "OK": "Supply pressure adequate.",
                "WARN": "Pressure low — check pump and strainer.",
                "ALARM": "Lubrication starvation risk.",
            }[status],
        )

    if oil_aeration_pct is not None:
        status = _status_below(
            oil_aeration_pct,
            _DEFAULT_LIMITS["aeration_pct"]["warn"],
            _DEFAULT_LIMITS["aeration_pct"]["alarm"],
        )
        add(
            "oil_aeration_pct",
            "Air entrainment",
            oil_aeration_pct,
            "%",
            status,
            "warn >= 6 %, alarm >= 15 %",
            {
                "OK": "Air content acceptable.",
                "WARN": "Aeration rising — check level and return lines.",
                "ALARM": "Severe aeration — cavitation and film-loss risk.",
            }[status],
        )

    if oil_foam_pct is not None:
        status = _status_below(
            oil_foam_pct,
            _DEFAULT_LIMITS["foam_pct"]["warn"],
            _DEFAULT_LIMITS["foam_pct"]["alarm"],
        )
        add(
            "oil_foam_pct",
            "Surface foam",
            oil_foam_pct,
            "%",
            status,
            "warn >= 10 %, alarm >= 25 %",
            {
                "OK": "Foaming acceptable.",
                "WARN": "Foaming elevated.",
                "ALARM": "Excessive foaming — additive/level issue.",
            }[status],
        )

    if oil_iron_ppm is not None:
        status = _status_below(
            oil_iron_ppm,
            _DEFAULT_LIMITS["iron_ppm"]["warn"],
            _DEFAULT_LIMITS["iron_ppm"]["alarm"],
        )
        add(
            "oil_iron_ppm",
            "Iron / wear-metal content",
            oil_iron_ppm,
            "ppm",
            status,
            "warn >= 100, alarm >= 300",
            {
                "OK": "Wear-metal content acceptable.",
                "WARN": "Wear metals rising — trending wear.",
                "ALARM": "Heavy wear — gear/bearing damage suspected.",
            }[status],
        )

    if oil_temp_c is not None:
        # Temperature is informational here (the oil-temperature fault lives in
        # the detector), but a very hot oil explains low viscosity readings.
        temp_status: Status = "OK"
        if oil_temp_c >= 90.0:
            temp_status = "ALARM"
        elif oil_temp_c >= 80.0:
            temp_status = "WARN"
        add(
            "oil_temp_c",
            "Oil temperature",
            oil_temp_c,
            "°C",
            temp_status,
            "warn >= 80 °C, alarm >= 90 °C (monitor)",
            {
                "OK": "Oil temperature acceptable.",
                "WARN": "Oil hot — verify cooler.",
                "ALARM": "Oil very hot — cooling fault suspected.",
            }[temp_status],
        )

    score = 100.0 - sum(_SEVERITY_PENALTY[f.status] for f in findings)
    score = max(0.0, min(100.0, score))
    n_alarms = sum(1 for f in findings if f.status == "ALARM")
    n_warns = sum(1 for f in findings if f.status == "WARN")
    overall: Status = "ALARM" if n_alarms else ("WARN" if n_warns else "OK")
    return OilAnalysis(findings=findings, health_score=score, overall_status=overall)


def oil_analysis_from_telemetry(
    telemetry: dict,
    viscosity_min_cst: float | None = None,
    viscosity_max_cst: float | None = None,
) -> OilAnalysis:
    """Run :func:`analyze_oil` pulling whatever oil channels are present."""
    return analyze_oil(
        oil_viscosity_cst=_num(telemetry, "oil_viscosity_cst"),
        oil_temp_c=_num(telemetry, "oil_temp_c") or _num(telemetry, "temperature_c"),
        oil_water_ppm=_num(telemetry, "oil_water_ppm"),
        oil_moisture_pct=_num(telemetry, "oil_moisture_pct"),
        oil_particles_iso4406=_num(telemetry, "oil_particles_iso4406"),
        oil_tan_mgkoh_g=_num(telemetry, "oil_tan_mgkoh_g"),
        oil_filter_dp_bar=_num(telemetry, "oil_filter_dp_bar"),
        oil_level_pct=_num(telemetry, "oil_level_pct"),
        oil_pressure_bar=_num(telemetry, "oil_pressure_bar"),
        oil_aeration_pct=_num(telemetry, "oil_aeration_pct"),
        oil_foam_pct=_num(telemetry, "oil_foam_pct"),
        oil_iron_ppm=_num(telemetry, "oil_iron_ppm"),
        viscosity_min_cst=viscosity_min_cst,
        viscosity_max_cst=viscosity_max_cst,
    )


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
