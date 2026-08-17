"""Whole-turbine fault taxonomy: every subsystem, every fault type.

AeroVigil's fault catalog covers all twelve wind-turbine subsystems:

    1. Rotor & Blades
    2. Pitch System
    3. Hub & Main Shaft
    4. Gearbox                      (including oil-condition faults)
    5. High-Speed Shaft & Brake
    6. Generator
    7. Yaw System
    8. Tower & Foundation
    9. Nacelle & Sensors
    10. Cooling & Hydraulics
    11. Electrical & Power Conversion
    12. SCADA & Communication

Each :class:`FaultDefinition` records the fault id, human-readable name,
severity, description, root causes, symptoms, the signals it is detected
from, and the recommended maintenance actions. The catalog is the single
source of truth consumed by:

* ``src.faults.detector``  — automatic rule-based detection per snapshot,
* ``src.faults.oil``       — gearbox-oil-condition analysis,
* the ``faults`` CLI subcommand in ``main.py``,
* the ``/faults/*`` API routes in ``src.api.app``,
* the digital-twin fault report (``src.digital_twin.twin``).

Severity levels follow a maintenance-urgency scale:

* ``LOW``      — watch item; schedule with the next routine service.
* ``MEDIUM``   — plan an inspection within days-weeks; monitor closely.
* ``HIGH``     — engineering review required; plan repair soon.
* ``CRITICAL`` — act immediately; risk of damage or forced outage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Severity = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]

SEVERITIES: tuple[Severity, ...] = ("LOW", "MEDIUM", "HIGH", "CRITICAL")


@dataclass(frozen=True)
class FaultDefinition:
    """One fault type of one wind-turbine subsystem."""

    fault_id: str
    name: str
    subsystem: str  # canonical subsystem key, e.g. "gearbox"
    subsystem_label: str  # human label, e.g. "Gearbox"
    severity: Severity
    description: str
    root_causes: list[str] = field(default_factory=list)
    symptoms: list[str] = field(default_factory=list)
    detection_signals: list[str] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serializable summary (used by the CLI and the API)."""
        return {
            "fault_id": self.fault_id,
            "name": self.name,
            "subsystem": self.subsystem,
            "subsystem_label": self.subsystem_label,
            "severity": self.severity,
            "description": self.description,
            "root_causes": list(self.root_causes),
            "symptoms": list(self.symptoms),
            "detection_signals": list(self.detection_signals),
            "recommended_actions": list(self.recommended_actions),
        }


# --------------------------------------------------------------------------- #
# Subsystem registry                                                           #
# --------------------------------------------------------------------------- #
SUBSYSTEMS: dict[str, str] = {
    "rotor_blades": "Rotor & Blades",
    "pitch": "Pitch System",
    "hub_mainshaft": "Hub & Main Shaft",
    "gearbox": "Gearbox",
    "hss_brake": "High-Speed Shaft & Brake",
    "generator": "Generator",
    "yaw": "Yaw System",
    "tower_foundation": "Tower & Foundation",
    "nacelle_sensors": "Nacelle & Sensors",
    "cooling_hydraulics": "Cooling & Hydraulics",
    "electrical": "Electrical & Power",
    "scada": "SCADA & Communication",
}


# --------------------------------------------------------------------------- #
# The full catalog                                                             #
# --------------------------------------------------------------------------- #
FAULT_CATALOG: list[FaultDefinition] = [
    # ── 1. Rotor & Blades ──────────────────────────────────────────────────
    FaultDefinition(
        fault_id="RB-01",
        name="Blade mass imbalance",
        subsystem="rotor_blades",
        subsystem_label="Rotor & Blades",
        severity="MEDIUM",
        description="One blade carries more mass than the others, producing a 1P "
        "rotor-frequency vibration and asymmetric loads on the main shaft.",
        root_causes=[
            "Ice accretion that does not shed evenly",
            "Blade repair adds mass",
            "Water ingress inside the blade",
            "Manufacturing/tolerance deviation",
        ],
        symptoms=[
            "Rising 1P (rotor-speed) vibration component",
            "Side-to-side nacelle shaking in phase with rotor rotation",
            "Main bearing temperature slowly increasing",
        ],
        detection_signals=["vibration_mms", "blade_1p_amplitude_mms", "rotor_speed_rpm"],
        recommended_actions=[
            "Inspect blade mass distribution",
            "Check for water ingress",
            "Balance blades during next planned maintenance",
        ],
    ),
    FaultDefinition(
        fault_id="RB-02",
        name="Leading-edge erosion",
        subsystem="rotor_blades",
        subsystem_label="Rotor & Blades",
        severity="LOW",
        description="Progressive erosion of the blade leading-edge coating reduces "
        "aerodynamic efficiency and lifts noise and vibration.",
        root_causes=[
            "Rain/sand/particle impact",
            "Coating age and UV degradation",
            "Insect debris accumulation",
        ],
        symptoms=[
            "Gradual annual energy-production (AEP) loss",
            "Higher noise emissions",
            "Power curve droop at high wind speed",
        ],
        detection_signals=["aep_deviation_pct", "power_kw", "wind_speed_mps"],
        recommended_actions=[
            "Visual/drone blade inspection",
            "Leading-edge protection tape",
            "Scheduled re-coating",
        ],
    ),
    FaultDefinition(
        fault_id="RB-03",
        name="Blade icing",
        subsystem="rotor_blades",
        subsystem_label="Rotor & Blades",
        severity="MEDIUM",
        description="Ice accumulates on the blades, adding mass, changing the profile, "
        "reducing power and throwing ice hazardously.",
        root_causes=[
            "Freezing rain or rime conditions",
            "Inoperative blade heating system",
            "Cold-start without de-icing cycle",
        ],
        symptoms=[
            "Power drops while wind speed stays high",
            "Mass imbalance vibration (1P)",
            "Ice-detector / power-curve deviation alarms",
        ],
        detection_signals=["ambient_temp_c", "power_kw", "wind_speed_mps", "ice_detector_on"],
        recommended_actions=[
            "Run de-icing cycle",
            "Check blade heating elements",
            "Derate until ice sheds safely",
        ],
    ),
    FaultDefinition(
        fault_id="RB-04",
        name="Lightning strike damage",
        subsystem="rotor_blades",
        subsystem_label="Rotor & Blades",
        severity="HIGH",
        description="Lightning hit compromises the blade receptor, bonding or internal "
        "structure; may also damage the pitch system electronics.",
        root_causes=[
            "Lightning storm exposure",
            "Failed lightning protection system",
            "Deteriorated receptors/conductors",
        ],
        symptoms=[
            "Lightning-counter event registered",
            "Pitch or sensor electronics glitches",
            "Visible scorch marks found on inspection",
        ],
        detection_signals=["lightning_events_24h", "inspection_lightning_damage"],
        recommended_actions=[
            "Inspect blade receptors and down-conductor",
            "Test lightning protection continuity",
            "Repair damaged laminate",
        ],
    ),
    FaultDefinition(
        fault_id="RB-05",
        name="Blade crack / structural damage",
        subsystem="rotor_blades",
        subsystem_label="Rotor & Blades",
        severity="CRITICAL",
        description="Structural crack of the blade shell, spar or bonding line — a "
        "catastrophic-failure risk that must be grounded and inspected.",
        root_causes=[
            "Fatigue over lifetime",
            "Manufacturing defect",
            "Overload / extreme gusts",
            "Unrepaired erosion or impact damage",
        ],
        symptoms=[
            "Sudden rise in vibration",
            "Unusual acoustic emission",
            "Shedding debris observed by cameras",
        ],
        detection_signals=["vibration_mms", "blade_acoustic_anomaly", "inspection_crack"],
        recommended_actions=[
            "Stop turbine immediately",
            "Rope-access or drone inspection",
            "Structural repair or blade replacement",
        ],
    ),
    FaultDefinition(
        fault_id="RB-06",
        name="Blade pitch-angle asymmetry (aerodynamic imbalance)",
        subsystem="rotor_blades",
        subsystem_label="Rotor & Blades",
        severity="MEDIUM",
        description="One blade consistently operates at a slightly different pitch "
        "angle, causing asymmetric thrust, vibration and power loss.",
        root_causes=[
            "Pitch calibration drift",
            "Incorrect pitch servo zero-point",
            "Mechanical play in pitch bearing",
        ],
        symptoms=[
            "Power curve deviation",
            "1P vibration component",
            "Pitch angle scatter differs between blades",
        ],
        detection_signals=["pitch_angle_deg", "blade_pitch_deviation_deg", "power_kw"],
        recommended_actions=[
            "Re-calibrate pitch angle sensors",
            "Check pitch bearing play",
            "Verify pitch control gains",
        ],
    ),
    FaultDefinition(
        fault_id="RB-07",
        name="Blade fire / lightning fire",
        subsystem="rotor_blades",
        subsystem_label="Rotor & Blades",
        severity="CRITICAL",
        description="Fire on or inside a blade — typically lightning-initiated or "
        "from hot pitch-system electronics; requires immediate shutdown and "
        "fire response.",
        root_causes=[
            "Lightning strike",
            "Electrical fault in pitch/hub electronics",
            "Leading-edge tape burning from friction",
            "Arcing in blade heater",
        ],
        symptoms=[
            "Blade fire alarm",
            "Smoke near rotor",
            "Lightning event followed by smoke",
            "Infrared camera hotspot on blade",
        ],
        detection_signals=[
            "blade_fire_alarm",
            "blade_temp_c",
            "smoke_detector_on",
            "lightning_events_24h",
        ],
        recommended_actions=[
            "Shut down and yaw to safe position",
            "Trigger fire suppression",
            "Evacuate tower base, call emergency services",
        ],
    ),
    FaultDefinition(
        fault_id="RB-08",
        name="Blade tip deflection / excessive flex",
        subsystem="rotor_blades",
        subsystem_label="Rotor & Blades",
        severity="MEDIUM",
        description="Blade tip deflection exceeds the design envelope, risking "
        "tower strike in extreme cases; indicates stiffness loss or overload.",
        root_causes=[
            "Structural stiffness loss",
            "Over-speed operation",
            "Extreme gusts",
            "Damaged spar cap",
        ],
        symptoms=[
            "Tip deflection sensor/strain above limit",
            "Tower proximity alarms",
            "AEP loss at high wind",
        ],
        detection_signals=["blade_tip_deflection_pct", "blade_strain_ue", "rotor_speed_rpm"],
        recommended_actions=[
            "Reduce load setpoints",
            "Strain/inspection review",
            "Check for spar damage",
        ],
    ),
    FaultDefinition(
        fault_id="RB-09",
        name="Trailing-edge split / delamination",
        subsystem="rotor_blades",
        subsystem_label="Rotor & Blades",
        severity="HIGH",
        description="Trailing-edge opening or laminate delamination grows with "
        "cycling and can shed material; detected acoustically or by inspection.",
        root_causes=[
            "Fatigue of trailing-edge bond",
            "Manufacturing void",
            "Impact damage",
            "Water ingress freezing",
        ],
        symptoms=["Blade acoustic anomaly", "AEP deviation", "Found on drone inspection"],
        detection_signals=[
            "blade_acoustic_anomaly",
            "aep_deviation_pct",
            "inspection_blade_delamination",
        ],
        recommended_actions=[
            "Inspect with drone/rope access",
            "Repair trailing-edge bond",
            "Monitor growth rate",
        ],
    ),
    FaultDefinition(
        fault_id="RB-10",
        name="Blade root bolt tension loss",
        subsystem="rotor_blades",
        subsystem_label="Rotor & Blades",
        severity="HIGH",
        description="Tension loss of blade root studs/bolts — a classic finding of "
        "bolt audits; if undetected it can progress to bolt failure.",
        root_causes=[
            "Pre-load relaxation",
            "Oversized holes / bedding wear",
            "Corrosion",
            "Overload cycles",
        ],
        symptoms=[
            "Bolt tension audit deviation",
            "Hub-side stud corrosion",
            "Micro-movement at root flange",
        ],
        detection_signals=["blade_bolt_tension_deviation_pct", "inspection_bolt_loose"],
        recommended_actions=[
            "Bolt tension audit",
            "Re-torque per OEM procedure",
            "Replace degraded studs",
        ],
    ),
    # ── 2. Pitch System ────────────────────────────────────────────────────
    FaultDefinition(
        fault_id="PT-01",
        name="Pitch bearing wear / spalling",
        subsystem="pitch",
        subsystem_label="Pitch System",
        severity="MEDIUM",
        description="Wear or spalling of the pitch bearing raceways produces "
        "increased pitch torque demand, noise and play.",
        root_causes=[
            "Grease contamination or starvation",
            "Oscillating micro-motion (fretting)",
            "Overload from gusts",
            "Brinelling during standstill",
        ],
        symptoms=[
            "Pitch torque above normal",
            "Grinding noise during pitching",
            "Pitch position tracking error",
        ],
        detection_signals=["pitch_torque_pct", "pitch_position_error_deg", "pitch_motor_current_a"],
        recommended_actions=[
            "Grease analysis and re-lubrication",
            "Inspect bearing raceways",
            "Plan pitch bearing replacement",
        ],
    ),
    FaultDefinition(
        fault_id="PT-02",
        name="Pitch drive motor / gearbox fault",
        subsystem="pitch",
        subsystem_label="Pitch System",
        severity="HIGH",
        description="The pitch drive (motor, its gearbox or VFD) cannot move the "
        "blade reliably — a safety-critical actuator for shutdown.",
        root_causes=[
            "Motor winding failure",
            "Pitch gearbox wear",
            "VFD/brake fault",
            "Overheating",
        ],
        symptoms=[
            "Pitch fault alarms",
            "Motor current/torque anomalies",
            "Blade fails to reach feather on command",
        ],
        detection_signals=["pitch_motor_current_a", "pitch_torque_pct", "pitch_fault_code"],
        recommended_actions=[
            "Run pitch function test",
            "Check motor and drive electronics",
            "Replace pitch drive unit if faulty",
        ],
    ),
    FaultDefinition(
        fault_id="PT-03",
        name="Pitch angle sensor (encoder) fault",
        subsystem="pitch",
        subsystem_label="Pitch System",
        severity="HIGH",
        description="The pitch position encoder reads an angle that disagrees with "
        "the actual blade position — the controller then fights phantom errors.",
        root_causes=[
            "Encoder failure / drift",
            "Cable damage in the hub",
            "Connector corrosion",
            "Battery-backed memory fault",
        ],
        symptoms=[
            "Pitch sensor disagreement alarms",
            "Blade position jumps",
            "Emergency feather events",
        ],
        detection_signals=["pitch_position_error_deg", "pitch_sensor_disagreement"],
        recommended_actions=[
            "Cross-check with redundant sensor",
            "Re-calibrate encoder",
            "Replace encoder and cable",
        ],
    ),
    FaultDefinition(
        fault_id="PT-04",
        name="Hydraulic pitch pressure loss",
        subsystem="pitch",
        subsystem_label="Pitch System",
        severity="CRITICAL",
        description="Hydraulic pitch systems cannot feather the blades if accumulator "
        "pressure is lost — the turbine must shut down immediately.",
        root_causes=[
            "Hydraulic leak",
            "Pump failure",
            "Accumulator bladder rupture",
            "Pressure switch failure",
        ],
        symptoms=["Low hydraulic pressure alarms", "Frequent pump starts", "Feather tests fail"],
        detection_signals=["hydraulic_pressure_bar", "pump_runtime_pct"],
        recommended_actions=[
            "Shut down turbine",
            "Inspect hydraulic circuit for leaks",
            "Test accumulator pre-charge",
        ],
    ),
    FaultDefinition(
        fault_id="PT-05",
        name="Pitch accumulator pre-charge loss",
        subsystem="pitch",
        subsystem_label="Pitch System",
        severity="MEDIUM",
        description="The pitch accumulator lost nitrogen pre-charge, reducing the "
        "energy available for an emergency feather.",
        root_causes=["Bladder permeability", "Valve leakage", "Failed pre-charge check"],
        symptoms=["Pressure decays fast when pump stops", "Emergency feather slower than spec"],
        detection_signals=["hydraulic_pressure_bar", "feather_time_s"],
        recommended_actions=["Re-charge accumulator", "Replace bladder", "Retest feather time"],
    ),
    FaultDefinition(
        fault_id="PT-06",
        name="Pitch angle drift / deviation",
        subsystem="pitch",
        subsystem_label="Pitch System",
        severity="MEDIUM",
        description="Blade pitch angles deviate from the commanded setpoint by more "
        "than tolerance, degrading power performance and loading.",
        root_causes=[
            "Calibration drift",
            "Hydraulic valve hysteresis",
            "Mechanical backlash in pitch linkage",
        ],
        symptoms=["Pitch tracking error", "Power curve droop", "Cyclic 1P load increase"],
        detection_signals=["blade_pitch_deviation_deg", "pitch_position_error_deg"],
        recommended_actions=[
            "Re-calibrate pitch system",
            "Check valve hysteresis",
            "Tighten/align pitch linkage",
        ],
    ),
    # ── 3. Hub & Main Shaft ────────────────────────────────────────────────
    FaultDefinition(
        fault_id="HS-01",
        name="Main (rotor) bearing wear",
        subsystem="hub_mainshaft",
        subsystem_label="Hub & Main Shaft",
        severity="HIGH",
        description="Wear of the main shaft bearing — the project's headline RUL "
        "target — driven by fatigue, contamination and load cycles.",
        root_causes=[
            "Fatigue per ISO 281 L10 life",
            "Lubrication starvation",
            "Contamination / water in grease",
            "Misalignment",
            "Extreme loads",
        ],
        symptoms=[
            "Rising bearing temperature",
            "Low-frequency vibration",
            "Rising oil/grease debris counts",
            "RUL forecast shortening",
        ],
        detection_signals=[
            "main_bearing_temp_c",
            "vibration_mms",
            "grease_debris_ppm",
            "predicted_rul_days",
        ],
        recommended_actions=[
            "Increase condition-monitoring frequency",
            "Grease analysis",
            "Plan bearing replacement before failure",
        ],
    ),
    FaultDefinition(
        fault_id="HS-02",
        name="Main shaft misalignment / bending",
        subsystem="hub_mainshaft",
        subsystem_label="Hub & Main Shaft",
        severity="HIGH",
        description="The main shaft is bent or misaligned with the gearbox input, "
        "increasing coupling loads and vibration.",
        root_causes=[
            "Rotor overhang load",
            "Bearing wear allowing sag",
            "Thermal distortion",
            "Coupling misalignment during assembly",
        ],
        symptoms=[
            "Axial vibration spikes",
            "Coupling wear debris",
            "Gearbox LSS bearing temperature rise",
        ],
        detection_signals=["vibration_mms", "shaft_axial_displacement_mm", "coupling_temp_c"],
        recommended_actions=[
            "Check shaft run-out and alignment",
            "Inspect coupling",
            "Realign or straighten shaft",
        ],
    ),
    FaultDefinition(
        fault_id="HS-03",
        name="Hub / spinner structural crack",
        subsystem="hub_mainshaft",
        subsystem_label="Hub & Main Shaft",
        severity="HIGH",
        description="Crack in the hub casting or spinner shell found by inspection or "
        "suggested by anomalous low-frequency vibration.",
        root_causes=["Fatigue", "Casting defect", "Bolted-joint loosening"],
        symptoms=["Low-frequency vibration", "Found during scheduled inspection"],
        detection_signals=["vibration_mms", "inspection_crack"],
        recommended_actions=[
            "Inspect hub and bolts",
            "Ultrasonic testing of casting",
            "Repair or replace hub",
        ],
    ),
    FaultDefinition(
        fault_id="HS-04",
        name="Coupling / shrink-disc slip",
        subsystem="hub_mainshaft",
        subsystem_label="Hub & Main Shaft",
        severity="HIGH",
        description="The main-shaft-to-gearbox coupling slips or loses pre-load, "
        "causing torque fluctuations and fretting.",
        root_causes=[
            "Insufficient pre-load",
            "Lubricant on friction surface",
            "Torque spikes",
            "Worn coupling elements",
        ],
        symptoms=[
            "Torque spikes at rotor frequency",
            "Fretting debris",
            "Rotor speed vs gearbox speed mismatch",
        ],
        detection_signals=["torque_spike_amplitude", "rpm", "rotor_speed_rpm"],
        recommended_actions=[
            "Verify shrink-disc torque",
            "Clean friction surfaces",
            "Replace coupling elements",
        ],
    ),
    # ── 4. Gearbox (incl. oil-condition faults) ────────────────────────────
    FaultDefinition(
        fault_id="GB-01",
        name="Gearbox oil temperature too high",
        subsystem="gearbox",
        subsystem_label="Gearbox",
        severity="MEDIUM",
        description="Gearbox oil temperature exceeds the OEM limit, accelerating "
        "oxidation and reducing film strength.",
        root_causes=[
            "Cooler fouling / fan failure",
            "Oil level too low",
            "Overload",
            "Internal friction (wear, misalignment)",
        ],
        symptoms=[
            "Oil temperature alarm",
            "Viscosity dropping with temperature",
            "Cooler runs at maximum",
        ],
        detection_signals=["oil_temp_c", "temperature_c", "coolant_temp_c"],
        recommended_actions=[
            "Check cooler and fan",
            "Verify oil level",
            "Oil sampling and analysis",
        ],
    ),
    FaultDefinition(
        fault_id="GB-02",
        name="Oil viscosity too low",
        subsystem="gearbox",
        subsystem_label="Gearbox",
        severity="HIGH",
        description="Kinematic viscosity below the minimum limit (e.g. < 10 cSt at "
        "operating temperature) — thin film, metal-to-metal contact risk.",
        root_causes=[
            "Water or fuel contamination",
            "Wrong oil grade topped up",
            "Thermal cracking / oxidation thinning",
            "Overheating",
        ],
        symptoms=[
            "Viscosity reading below limit",
            "Rising gear/bearing temperatures",
            "Increased wear debris",
        ],
        detection_signals=["oil_viscosity_cst", "oil_temp_c", "oil_water_ppm"],
        recommended_actions=[
            "Oil sampling: water, TAN, particle count",
            "Locate and stop contamination source",
            "Oil change if degraded",
        ],
    ),
    FaultDefinition(
        fault_id="GB-03",
        name="Oil viscosity too high",
        subsystem="gearbox",
        subsystem_label="Gearbox",
        severity="MEDIUM",
        description="Viscosity above the maximum limit (e.g. > 50 cSt) — poor pumpability, "
        "oil starvation at start-up, high churning losses.",
        root_causes=[
            "Oxidation thickening",
            "Wrong grade topped up",
            "Cold start below minimum temperature",
            "Evaporation of light ends",
        ],
        symptoms=[
            "Viscosity reading above limit",
            "High pressure across filter",
            "Slow oil circulation at cold start",
        ],
        detection_signals=["oil_viscosity_cst", "oil_filter_dp_bar", "oil_temp_c"],
        recommended_actions=[
            "Confirm oil grade",
            "Check for oxidation (TAN rise)",
            "Consider oil change / dilution flush",
        ],
    ),
    FaultDefinition(
        fault_id="GB-04",
        name="Water contamination in gearbox oil",
        subsystem="gearbox",
        subsystem_label="Gearbox",
        severity="HIGH",
        description="Water in the oil (ppm or % saturation) destroys film strength, "
        "corrodes bearings and promotes emulsification and bacterial growth.",
        root_causes=["Breather/desiccant failure", "Cooler leak", "Condensation", "Seal damage"],
        symptoms=[
            "Water alarm from online sensor",
            "Milky/emulsified oil",
            "Viscosity drop",
            "Corrosion debris in filters",
        ],
        detection_signals=["oil_water_ppm", "oil_moisture_pct", "oil_viscosity_cst"],
        recommended_actions=[
            "Find and fix water entry path",
            "Run offline filtration",
            "Replace desiccant breather",
            "Drain water if separated",
        ],
    ),
    FaultDefinition(
        fault_id="GB-05",
        name="Particle contamination (ISO 4406)",
        subsystem="gearbox",
        subsystem_label="Gearbox",
        severity="MEDIUM",
        description="Particle counts above the ISO 4406 cleanliness target (e.g. "
        "> 19/17/14) indicate abrasive wear or ingression; abrasive particles "
        "accelerate bearing and gear wear.",
        root_causes=[
            "Normal wear debris accumulation",
            "Dust ingression via breather/seals",
            "Filter bypass or clogging",
            "Contaminated oil top-up",
        ],
        symptoms=[
            "ISO 4406 code above target",
            "Filter ΔP rising",
            "Wear metals in spectrometric analysis",
        ],
        detection_signals=["oil_particles_iso4406", "oil_filter_dp_bar", "oil_iron_ppm"],
        recommended_actions=[
            "Particle count analysis (ISO 4406)",
            "Inspect/change filters",
            "Oil filtration or change",
            "Check breather and seals",
        ],
    ),
    FaultDefinition(
        fault_id="GB-06",
        name="Gearbox oil level too low",
        subsystem="gearbox",
        subsystem_label="Gearbox",
        severity="HIGH",
        description="Oil level below the minimum — lubrication starvation, aeration, "
        "and overheating of gears and bearings.",
        root_causes=["External leak", "Internal leak", "Incomplete filling", "Foaming losses"],
        symptoms=["Low-level alarm", "Oil temperature rising", "Filter ΔP fluctuating"],
        detection_signals=["oil_level_pct", "oil_temp_c"],
        recommended_actions=[
            "Top up with correct grade",
            "Inspect for leaks",
            "Investigate where the oil went",
        ],
    ),
    FaultDefinition(
        fault_id="GB-07",
        name="Oil filter clogging / high ΔP",
        subsystem="gearbox",
        subsystem_label="Gearbox",
        severity="MEDIUM",
        description="Differential pressure across the oil filter exceeds limit — the "
        "filter bypass may open, sending unfiltered oil through the gearbox.",
        root_causes=[
            "Particle loading",
            "Cold oil at start-up",
            "Filter degradation",
            "Contamination event",
        ],
        symptoms=["Filter ΔP alarm", "Bypass valve opens", "Particle counts rising"],
        detection_signals=["oil_filter_dp_bar", "oil_particles_iso4406"],
        recommended_actions=[
            "Change filter elements",
            "Analyze filter debris (ferrography)",
            "Check cold-start behavior",
        ],
    ),
    FaultDefinition(
        fault_id="GB-08",
        name="Oil oxidation / high acidity (TAN)",
        subsystem="gearbox",
        subsystem_label="Gearbox",
        severity="MEDIUM",
        description="Total acid number (TAN) above limit indicates oil oxidation, "
        "which thickens oil, forms sludge and attacks seals.",
        root_causes=[
            "Overheating",
            "Extended oil drain interval",
            "Water contamination",
            "Catalytic wear metals",
        ],
        symptoms=[
            "TAN rising toward 2 mg KOH/g",
            "Dark oil color",
            "Sludge deposits",
            "Viscosity drift up",
        ],
        detection_signals=["oil_tan_mgkoh_g", "oil_viscosity_cst", "oil_oxidation_pct"],
        recommended_actions=[
            "Confirm with laboratory analysis",
            "Shorten drain interval",
            "Plan oil change",
            "Address root cause (heat/water)",
        ],
    ),
    FaultDefinition(
        fault_id="GB-09",
        name="Gear tooth pitting / surface fatigue",
        subsystem="gearbox",
        subsystem_label="Gearbox",
        severity="HIGH",
        description="Pitting or micropitting on gear flanks from surface fatigue — "
        "detected via vibration and ferrography before tooth breakage.",
        root_causes=[
            "Surface fatigue cycles",
            "Marginal lubrication",
            "Contamination",
            "Tooth load concentration / misalignment",
        ],
        symptoms=[
            "Gear-mesh frequency (GMF) sidebands in spectrum",
            "Ferrous debris in oil",
            "Vibration trend rising",
        ],
        detection_signals=["vibration_mms", "oil_iron_ppm", "gmf_sideband_amplitude"],
        recommended_actions=[
            "Vibration analysis with gearbox specialist",
            "Oil debris analysis",
            "Borescope inspection",
            "Plan gearbox repair",
        ],
    ),
    FaultDefinition(
        fault_id="GB-10",
        name="Gear tooth breakage / scuffing",
        subsystem="gearbox",
        subsystem_label="Gearbox",
        severity="CRITICAL",
        description="A gear tooth breaks or scuffs — imminent gearbox failure with "
        "debris risk to all downstream stages.",
        root_causes=[
            "Tooth bending fatigue",
            "Impact load / torque spike",
            "Lubrication loss",
            "Advanced pitting",
        ],
        symptoms=[
            "Sudden vibration jump",
            "Large ferrous debris",
            "Gearbox noise change",
            "Oil filter loaded with metal",
        ],
        detection_signals=["vibration_mms", "oil_iron_ppm", "torque_spike_amplitude"],
        recommended_actions=[
            "Stop turbine immediately",
            "Borescope inspection",
            "Major gearbox repair or replacement",
        ],
    ),
    FaultDefinition(
        fault_id="GB-11",
        name="Gearbox bearing wear (HSS / ISS / LSS)",
        subsystem="gearbox",
        subsystem_label="Gearbox",
        severity="HIGH",
        description="Wear of gearbox bearings (high-, intermediate- or low-speed "
        "shaft) — the classic vibration-analysis finding with characteristic "
        "defect frequencies.",
        root_causes=["Fatigue (ISO 281)", "Poor lubrication", "Misalignment", "Contamination"],
        symptoms=[
            "BPFO/BPFI/BSF/FTF peaks in spectrum",
            "Bearing temperature rise",
            "Wear debris trend",
        ],
        detection_signals=["vibration_mms", "bearing_temp_c", "oil_iron_ppm", "bpfo_amplitude_mms"],
        recommended_actions=[
            "Spectrum analysis to locate bearing",
            "Increase monitoring",
            "Plan bearing replacement",
        ],
    ),
    FaultDefinition(
        fault_id="GB-12",
        name="Lubrication starvation",
        subsystem="gearbox",
        subsystem_label="Gearbox",
        severity="HIGH",
        description="Insufficient oil reaches gears/bearings (pump, line, nozzle or "
        "level problem) — rapid wear without intervention.",
        root_causes=[
            "Oil pump failure",
            "Suction strainer clog",
            "Low oil level",
            "Blocked spray nozzles",
            "Wrong viscosity (too high)",
        ],
        symptoms=["Oil pressure low", "Temperatures rising", "Gear noise"],
        detection_signals=["oil_pressure_bar", "oil_level_pct", "oil_temp_c"],
        recommended_actions=[
            "Check pump and pressure",
            "Inspect strainers and nozzles",
            "Verify level and viscosity",
        ],
    ),
    FaultDefinition(
        fault_id="GB-13",
        name="Oil aeration / foaming",
        subsystem="gearbox",
        subsystem_label="Gearbox",
        severity="MEDIUM",
        description="Air entrainment or surface foam in the oil reduces lubrication "
        "performance and can cause cavitation and erratic level readings.",
        root_causes=[
            "Low oil level allowing vortexing",
            "Return-line placement",
            "Water contamination",
            "Additive depletion",
        ],
        symptoms=["Aeration sensor above limit", "Level sensor noise", "Hydraulic noise"],
        detection_signals=["oil_aeration_pct", "oil_foam_pct", "oil_level_pct"],
        recommended_actions=[
            "Check level and return line",
            "Defoamant additive",
            "Oil analysis for water",
        ],
    ),
    FaultDefinition(
        fault_id="GB-14",
        name="Gear backlash increase / rattle",
        subsystem="gearbox",
        subsystem_label="Gearbox",
        severity="LOW",
        description="Increased gear backlash or rattle under low load indicates wear "
        "of tooth flanks or bearing clearance.",
        root_causes=["Flank wear", "Bearing clearance", "Torque reversals in grid events"],
        symptoms=["Rattle at low load", "Tooth-mesh modulation"],
        detection_signals=["vibration_mms", "backlash_mm"],
        recommended_actions=["Monitor trend", "Inspect at next service", "Adjust if possible"],
    ),
    FaultDefinition(
        fault_id="GB-15",
        name="Gearbox oil fire",
        subsystem="gearbox",
        subsystem_label="Gearbox",
        severity="CRITICAL",
        description="Gearbox oil ignites (hot oil spray on a hot surface) — smoke "
        "detectors and extreme oil temperature mark the event.",
        root_causes=[
            "Oil leak onto hot gearbox surface",
            "Oil mist ignition",
            "Extreme overheating",
            "Electrical spark near oil mist",
        ],
        symptoms=[
            "Oil temperature extreme (> 120 °C)",
            "Smoke detector near gearbox",
            "Fire suppression released",
        ],
        detection_signals=["oil_temp_c", "smoke_detector_on", "fire_suppression_released"],
        recommended_actions=[
            "Shut down and disconnect from grid",
            "Trigger fire suppression",
            "Evacuate tower base, call emergency services",
        ],
    ),
    # ── 5. High-Speed Shaft & Brake ────────────────────────────────────────
    FaultDefinition(
        fault_id="BR-01",
        name="Brake pad / caliper wear",
        subsystem="hss_brake",
        subsystem_label="High-Speed Shaft & Brake",
        severity="MEDIUM",
        description="Wear of the high-speed-shaft brake pads reduces braking torque; "
        "worn pads may also produce dust and heat.",
        root_causes=["Normal wear", "Dragging brake", "Misaligned caliper", "Contaminated pads"],
        symptoms=["Brake wear sensor near limit", "Braking time increase", "Brake dust/heat"],
        detection_signals=["brake_wear_pct", "brake_temp_c"],
        recommended_actions=[
            "Replace brake pads",
            "Check caliper alignment",
            "Verify braking torque in test",
        ],
    ),
    FaultDefinition(
        fault_id="BR-02",
        name="High-speed shaft over-speed",
        subsystem="hss_brake",
        subsystem_label="High-Speed Shaft & Brake",
        severity="CRITICAL",
        description="HSS speed exceeds the design limit — a protection-trip event; "
        "repeated trips stress the gearbox and brake.",
        root_causes=[
            "Grid loss while generating",
            "Controller failure",
            "Brake failure",
            "Coupling failure on generator side",
        ],
        symptoms=["Overspeed trip events", "HSS RPM above limit"],
        detection_signals=["rpm", "overspeed_trips_24h"],
        recommended_actions=[
            "Investigate trip sequence",
            "Test brake and controller",
            "Check HSS coupling",
        ],
    ),
    FaultDefinition(
        fault_id="BR-03",
        name="High-speed shaft vibration / coupling fault",
        subsystem="hss_brake",
        subsystem_label="High-Speed Shaft & Brake",
        severity="MEDIUM",
        description="Vibration on the high-speed side — coupling wear, HSS bearing "
        "or gearbox output issues.",
        root_causes=[
            "Coupling element wear",
            "HSS bearing wear",
            "Misalignment",
            "Generator-side imbalance",
        ],
        symptoms=["HSS vibration sensor trend", "2×/1× RPM peaks"],
        detection_signals=["vibration_mms", "hss_vibration_mms"],
        recommended_actions=["Vibration analysis", "Inspect coupling", "Check alignment"],
    ),
    FaultDefinition(
        fault_id="BR-04",
        name="Brake dragging / not releasing",
        subsystem="hss_brake",
        subsystem_label="High-Speed Shaft & Brake",
        severity="MEDIUM",
        description="The high-speed-shaft brake does not fully release after a stop, "
        "causing continuous friction, heat and premature pad wear.",
        root_causes=[
            "Brake release spring/actuator fault",
            "Caliper sticking",
            "Hydraulic pressure not reaching release",
            "Contaminated sliding surfaces",
        ],
        symptoms=[
            "Brake temperature high while running",
            "Pad wear faster than expected",
            "Brake drag current/position alarms",
        ],
        detection_signals=["brake_temp_c", "brake_drag_current_pct"],
        recommended_actions=[
            "Inspect caliper and actuator",
            "Verify release pressure",
            "Clean and re-grease sliding surfaces",
        ],
    ),
    FaultDefinition(
        fault_id="BR-05",
        name="Brake fire",
        subsystem="hss_brake",
        subsystem_label="High-Speed Shaft & Brake",
        severity="CRITICAL",
        description="Brake disc/pad overheat ignites — from prolonged dragging or a "
        "failed emergency brake; smoke and heat detectors fire.",
        root_causes=[
            "Prolonged brake dragging",
            "Failed release",
            "Disc overheating",
            "Combustible contamination on disc",
        ],
        symptoms=[
            "Brake temperature extreme",
            "Smoke detector in nacelle/drive train",
            "Burning smell, brake dust ignition",
        ],
        detection_signals=["brake_temp_c", "smoke_detector_on", "brake_fire_alarm"],
        recommended_actions=[
            "Shut down and disconnect from grid",
            "Trigger fire suppression",
            "Evacuate tower base, call emergency services",
        ],
    ),
    # ── 6. Generator ───────────────────────────────────────────────────────
    FaultDefinition(
        fault_id="GN-01",
        name="Stator winding over-temperature",
        subsystem="generator",
        subsystem_label="Generator",
        severity="HIGH",
        description="Stator/rotor winding temperature above the limit (e.g. 120 °C) "
        "degrades insulation and can lead to winding failure.",
        root_causes=["Cooling failure", "Overload", "Voltage imbalance", "Insulation aging"],
        symptoms=["Winding temperature alarm", "Power derating", "Hot spots"],
        detection_signals=["generator_temp_c", "stator_temp_c", "coolant_temp_c"],
        recommended_actions=[
            "Check cooling system",
            "Reduce load",
            "Thermography inspection",
            "Insulation resistance test",
        ],
    ),
    FaultDefinition(
        fault_id="GN-02",
        name="Generator bearing wear",
        subsystem="generator",
        subsystem_label="Generator",
        severity="MEDIUM",
        description="DE/NDE generator bearing wear detected by vibration spectra and "
        "temperature.",
        root_causes=["Fatigue", "Grease starvation", "Electrical pitting (EDM currents)"],
        symptoms=[
            "Bearing defect frequencies",
            "Generator bearing temperature rise",
            "Grease debris",
        ],
        detection_signals=["generator_bearing_temp_c", "vibration_mms", "bpfo_amplitude_mms"],
        recommended_actions=[
            "Vibration analysis",
            "Grease service",
            "Plan bearing replacement",
            "Check shaft grounding brushes",
        ],
    ),
    FaultDefinition(
        fault_id="GN-03",
        name="Air-gap eccentricity",
        subsystem="generator",
        subsystem_label="Generator",
        severity="MEDIUM",
        description="Rotor-stator air gap no longer uniform, producing unbalanced "
        "magnetic pull, vibration at 2× line frequency and heating.",
        root_causes=["Bearing wear", "Rotor bow", "Stator frame deformation"],
        symptoms=[
            "2× line-frequency vibration",
            "Current signature anomalies",
            "Uneven stator temperatures",
        ],
        detection_signals=["vibration_mms", "motor_current_imbalance_pct", "stator_temp_c"],
        recommended_actions=[
            "Motor current signature analysis",
            "Check bearings",
            "Generator specialist inspection",
        ],
    ),
    FaultDefinition(
        fault_id="GN-04",
        name="Winding insulation degradation",
        subsystem="generator",
        subsystem_label="Generator",
        severity="HIGH",
        description="Insulation resistance / polarization index below limits — "
        "precursor to winding short-circuit.",
        root_causes=["Aging", "Moisture", "Over-temperature", "Voltage surges"],
        symptoms=["Insulation resistance (IR) trending down", "Partial discharge rise"],
        detection_signals=["insulation_resistance_mohm", "partial_discharge_pc"],
        recommended_actions=["IR/PI test", "Bake-out or dry the windings", "Plan rewind"],
    ),
    FaultDefinition(
        fault_id="GN-05",
        name="Generator cooling failure",
        subsystem="generator",
        subsystem_label="Generator",
        severity="HIGH",
        description="Air/water cooling circuit failure causes rapid winding " "temperature rise.",
        root_causes=[
            "Cooling fan/pump fault",
            "Heat exchanger fouling",
            "Coolant leak",
            "Duct blockage",
        ],
        symptoms=["Generator temperature rising", "Coolant flow low", "Cooler outlet hot"],
        detection_signals=["coolant_temp_c", "coolant_flow_pct", "generator_temp_c"],
        recommended_actions=["Check fans/pumps", "Clean heat exchanger", "Repair coolant leak"],
    ),
    FaultDefinition(
        fault_id="GN-06",
        name="Slip ring / brush wear (DFIG)",
        subsystem="generator",
        subsystem_label="Generator",
        severity="MEDIUM",
        description="Slip-ring brush wear and contamination increase sparking and "
        "rotor-circuit resistance (doubly-fed machines).",
        root_causes=[
            "Brush wear",
            "Ring grooving",
            "Brush-spring pressure loss",
            "Dust accumulation",
        ],
        symptoms=["Slip-ring temperature rise", "Brush sparking", "Rotor current imbalance"],
        detection_signals=["slip_ring_temp_c", "rotor_current_imbalance_pct"],
        recommended_actions=["Replace brushes", "Clean/grind slip rings", "Check spring pressure"],
    ),
    FaultDefinition(
        fault_id="GN-07",
        name="Generator vibration",
        subsystem="generator",
        subsystem_label="Generator",
        severity="MEDIUM",
        description="Excessive generator vibration from mechanical or electrical "
        "causes (imbalance, misalignment, eccentricity).",
        root_causes=["Rotor imbalance", "Coupling misalignment", "Bearing wear", "Magnetic pull"],
        symptoms=["Vibration alarm on generator sensor", "Noise", "Bearing wear"],
        detection_signals=["generator_vibration_mms", "vibration_mms"],
        recommended_actions=[
            "Balance/align generator",
            "Inspect bearings and coupling",
            "Electrical tests if magnetic cause suspected",
        ],
    ),
    # ── 7. Yaw System ──────────────────────────────────────────────────────
    FaultDefinition(
        fault_id="YW-01",
        name="Yaw bearing wear",
        subsystem="yaw",
        subsystem_label="Yaw System",
        severity="MEDIUM",
        description="Wear of the yaw bearing raceways and ring gear increases yaw "
        "drive torque and backlash.",
        root_causes=[
            "Grease starvation",
            "Fretting corrosion",
            "Contamination",
            "Excessive yaw activity",
        ],
        symptoms=["Yaw torque above normal", "Yaw noise", "Ring-gear wear debris"],
        detection_signals=["yaw_torque_pct", "yaw_motor_current_a"],
        recommended_actions=[
            "Re-grease yaw bearing",
            "Inspect ring gear",
            "Plan yaw bearing replacement",
        ],
    ),
    FaultDefinition(
        fault_id="YW-02",
        name="Yaw drive / motor fault",
        subsystem="yaw",
        subsystem_label="Yaw System",
        severity="MEDIUM",
        description="One or more yaw drive motors or pinions fail, reducing yaw "
        "capability and loading the remaining drives.",
        root_causes=["Motor failure", "Pinion/ring gear wear", "VFD fault", "Drive overheating"],
        symptoms=["Yaw drive fault alarms", "Slow yaw response", "Uneven drive currents"],
        detection_signals=["yaw_motor_current_a", "yaw_position_error_deg"],
        recommended_actions=[
            "Inspect all yaw drives",
            "Check pinion meshing",
            "Replace failed drive",
        ],
    ),
    FaultDefinition(
        fault_id="YW-03",
        name="Yaw misalignment (wind tracking error)",
        subsystem="yaw",
        subsystem_label="Yaw System",
        severity="LOW",
        description="Nacelle persistently misaligned with the wind, costing power "
        "and adding asymmetric loads.",
        root_causes=[
            "Wind vane fault",
            "Yaw controller settings",
            "Yaw brake drag",
            "Cable twist limits",
        ],
        symptoms=[
            "Yaw error above 10–20° for long periods",
            "Power loss vs wind direction",
            "Asymmetric loading",
        ],
        detection_signals=["yaw_error_deg", "wind_vane_deg", "power_kw"],
        recommended_actions=[
            "Verify wind vane calibration",
            "Check yaw control settings",
            "Inspect yaw brake",
        ],
    ),
    FaultDefinition(
        fault_id="YW-04",
        name="Cable twist / unwinding fault",
        subsystem="yaw",
        subsystem_label="Yaw System",
        severity="HIGH",
        description="Yaw cable bundle twist count approaching the unwinding limit — "
        "risk of cable damage and power loss.",
        root_causes=[
            "Yaw controller not unwinding",
            "Twist counter fault",
            "Persistent one-direction wind",
        ],
        symptoms=["Twist counter near limit", "Yaw unwinding events"],
        detection_signals=["cable_twist_turns", "yaw_turns_24h"],
        recommended_actions=[
            "Trigger unwinding procedure",
            "Check twist counter",
            "Inspect cable bundle",
        ],
    ),
    FaultDefinition(
        fault_id="YW-05",
        name="Yaw brake failure",
        subsystem="yaw",
        subsystem_label="Yaw System",
        severity="MEDIUM",
        description="Yaw brakes do not hold the nacelle, allowing drift and "
        "oscillation around the wind direction.",
        root_causes=["Brake pad wear", "Hydraulic pressure loss", "Brake adjustment drift"],
        symptoms=["Nacelle drift during parking", "Yaw error oscillation", "Yaw brake alarms"],
        detection_signals=["yaw_brake_pressure_bar", "yaw_error_deg"],
        recommended_actions=["Adjust/rebuild yaw brakes", "Check brake circuit pressure"],
    ),
    # ── 8. Tower & Foundation ──────────────────────────────────────────────
    FaultDefinition(
        fault_id="TF-01",
        name="Tower resonance / excessive vibration",
        subsystem="tower_foundation",
        subsystem_label="Tower & Foundation",
        severity="HIGH",
        description="Tower vibration above limits — possible resonance with rotor "
        "harmonics, damping degradation or structural damage.",
        root_causes=[
            "1P/3P excitation near tower natural frequency",
            "Damper (TMD) fault",
            "Bolt loosening",
            "Structural crack",
        ],
        symptoms=["Tower acceleration alarm", "Nacelle oscillation", "Damping decay"],
        detection_signals=["tower_vibration_mms", "tower_accel_g", "nacelle_oscillation_mms"],
        recommended_actions=[
            "Analyze tower frequency spectrum",
            "Check tuned mass damper",
            "Inspect tower bolts and welds",
        ],
    ),
    FaultDefinition(
        fault_id="TF-02",
        name="Foundation bolt loosening",
        subsystem="tower_foundation",
        subsystem_label="Tower & Foundation",
        severity="HIGH",
        description="Loosening of foundation / tower-flange bolts detected by "
        "ultrasonic checks or characteristic vibration.",
        root_causes=["Pre-load relaxation", "Grout degradation", "Corrosion", "Fatigue"],
        symptoms=["Found on bolt-tension audit", "Tower-flange micro-movement"],
        detection_signals=["bolt_tension_deviation_pct", "inspection_bolt_loose"],
        recommended_actions=[
            "Tension audit of flange bolts",
            "Re-torque per OEM procedure",
            "Grout inspection",
        ],
    ),
    FaultDefinition(
        fault_id="TF-03",
        name="Tower tilt / foundation settlement",
        subsystem="tower_foundation",
        subsystem_label="Tower & Foundation",
        severity="MEDIUM",
        description="Progressive tilt of the tower due to foundation settlement — "
        "checked by tilt sensors or surveys.",
        root_causes=["Soil settlement", "Grout failure", "Drainage erosion around foundation"],
        symptoms=["Tilt sensor reading increasing", "Door misalignment", "Cracks in foundation"],
        detection_signals=["tower_tilt_deg", "inspection_foundation_crack"],
        recommended_actions=["Leveling survey", "Foundation inspection", "Grout repair"],
    ),
    FaultDefinition(
        fault_id="TF-04",
        name="Tower / foundation corrosion",
        subsystem="tower_foundation",
        subsystem_label="Tower & Foundation",
        severity="LOW",
        description="Corrosion of tower sections, flanges or foundation rebar "
        "reduces structural margins over time.",
        root_causes=[
            "Coating damage",
            "Coastal/saline environment",
            "Water pooling",
            "Galvanic contact",
        ],
        symptoms=["Rust staining", "Coating blistering", "Found on inspection"],
        detection_signals=["inspection_corrosion", "tower_humidity_pct"],
        recommended_actions=[
            "Coating repair",
            "Cathodic protection check",
            "Corrosion monitoring program",
        ],
    ),
    FaultDefinition(
        fault_id="TF-05",
        name="Tower base fire",
        subsystem="tower_foundation",
        subsystem_label="Tower & Foundation",
        severity="CRITICAL",
        description="Fire at the tower base — transformer, switchgear or cable "
        "compartment; tower-base smoke/heat detectors fire.",
        root_causes=[
            "Transformer fault at tower base",
            "Cable fault in tower",
            "Switchgear arcing",
            "External fire spreading",
        ],
        symptoms=[
            "Tower-base smoke alarm",
            "Tower-base heat detector",
            "Fire suppression released",
        ],
        detection_signals=[
            "tower_smoke_detector_on",
            "tower_fire_alarm",
            "fire_suppression_released",
        ],
        recommended_actions=[
            "Shut down and disconnect from grid",
            "Trigger fire suppression",
            "Evacuate tower base, call emergency services",
        ],
    ),
    # ── 9. Nacelle & Sensors ───────────────────────────────────────────────
    FaultDefinition(
        fault_id="NS-01",
        name="Anemometer fault / icing",
        subsystem="nacelle_sensors",
        subsystem_label="Nacelle & Sensors",
        severity="MEDIUM",
        description="The nacelle anemometer reads wrong (frozen, worn bearings, "
        "heater failure), corrupting the power curve and control inputs.",
        root_causes=["Bearing wear", "Icing", "Heater failure", "Impact damage"],
        symptoms=[
            "Wind speed disagrees with redundant unit",
            "Power curve deviation",
            "Anemometer signal flat/stuck",
        ],
        detection_signals=["wind_speed_mps", "wind_speed2_mps", "anemometer_heater_on"],
        recommended_actions=[
            "Cross-check redundant anemometers",
            "Replace anemometer",
            "Check heater circuit",
        ],
    ),
    FaultDefinition(
        fault_id="NS-02",
        name="Wind vane fault",
        subsystem="nacelle_sensors",
        subsystem_label="Nacelle & Sensors",
        severity="MEDIUM",
        description="The wind vane reports wrong direction, causing yaw tracking "
        "errors and power loss.",
        root_causes=["Mechanical damage", "Icing", "Signal/calibration fault"],
        symptoms=[
            "Yaw error large while vane disagrees with redundant unit",
            "Frequent unnecessary yawing",
        ],
        detection_signals=["wind_vane_deg", "yaw_error_deg"],
        recommended_actions=["Calibrate/replace vane", "Check wiring"],
    ),
    FaultDefinition(
        fault_id="NS-03",
        name="Nacelle oscillation / vibration",
        subsystem="nacelle_sensors",
        subsystem_label="Nacelle & Sensors",
        severity="MEDIUM",
        description="Excessive nacelle fore-aft or side-to-side oscillation — "
        "usually coupled with tower or rotor issues.",
        root_causes=[
            "Tower damper fault",
            "Rotor imbalance",
            "Yaw brake chatter",
            "Structural looseness",
        ],
        symptoms=["Nacelle acceleration alarms", "Visible swaying"],
        detection_signals=["nacelle_oscillation_mms", "tower_vibration_mms"],
        recommended_actions=["Check damper", "Analyze rotor harmonics", "Inspect nacelle mounts"],
    ),
    FaultDefinition(
        fault_id="NS-04",
        name="Lightning protection failure",
        subsystem="nacelle_sensors",
        subsystem_label="Nacelle & Sensors",
        severity="LOW",
        description="Nacelle lightning protection (air terminals, down-conductors, "
        "SPDs) degraded, increasing electronics damage risk.",
        root_causes=["Corrosion", "SPD aging", "Cable damage"],
        symptoms=["Surge damage events", "Found on inspection"],
        detection_signals=["lightning_events_24h", "inspection_lightning_damage"],
        recommended_actions=["Test continuity", "Replace SPDs", "Repair down-conductor"],
    ),
    FaultDefinition(
        fault_id="NS-05",
        name="Nacelle HVAC failure",
        subsystem="nacelle_sensors",
        subsystem_label="Nacelle & Sensors",
        severity="LOW",
        description="Nacelle heating/ventilation failure leads to condensation, "
        "overheating of electronics and corrosion.",
        root_causes=[
            "Fan failure",
            "Heater element failure",
            "Thermostat fault",
            "Filter clogging",
        ],
        symptoms=["Nacelle temperature/humidity out of band", "Condensation on equipment"],
        detection_signals=["nacelle_temp_c", "nacelle_humidity_pct"],
        recommended_actions=["Service HVAC unit", "Replace filters", "Check thermostat"],
    ),
    FaultDefinition(
        fault_id="NS-06",
        name="Nacelle fire / smoke detection",
        subsystem="nacelle_sensors",
        subsystem_label="Nacelle & Sensors",
        severity="CRITICAL",
        description="Smoke or fire detected in the nacelle — immediate shutdown "
        "and emergency response required.",
        root_causes=[
            "Electrical fault (converter/brake)",
            "Oil leak onto hot surfaces",
            "Lightning strike",
            "Brake fire",
        ],
        symptoms=["Smoke detector alarm", "Nacelle temperature spike", "Fire-suppression release"],
        detection_signals=["smoke_detector_on", "nacelle_temp_c"],
        recommended_actions=[
            "Shut down and disconnect from grid",
            "Trigger fire suppression",
            "Evacuate tower base, call emergency services",
        ],
    ),
    FaultDefinition(
        fault_id="NS-07",
        name="Fire suppression system fault",
        subsystem="nacelle_sensors",
        subsystem_label="Nacelle & Sensors",
        severity="HIGH",
        description="The fire suppression system is unavailable, faulted, or "
        "discharged — the turbine loses its first line of fire defense.",
        root_causes=[
            "Suppression bottle discharged/expired",
            "Detector fault",
            "Actuation circuit fault",
        ],
        symptoms=["Suppression status alarm", "System test failed", "Pressure low in bottle"],
        detection_signals=["fire_suppression_status", "fire_suppression_released"],
        recommended_actions=[
            "Recharge/replace suppression bottle",
            "Test detectors",
            "Repair actuation circuit",
        ],
    ),
    # ── 10. Cooling & Hydraulics ───────────────────────────────────────────
    FaultDefinition(
        fault_id="CH-01",
        name="Cooling fan / pump failure",
        subsystem="cooling_hydraulics",
        subsystem_label="Cooling & Hydraulics",
        severity="MEDIUM",
        description="Gearbox/generator cooling fan or oil-pump failure reduces heat "
        "rejection; temperatures climb.",
        root_causes=["Motor failure", "VFD fault", "Bearing failure in fan", "Control fault"],
        symptoms=["Coolant/oil temperature rising", "Fan running hours anomaly"],
        detection_signals=["coolant_temp_c", "fan_runtime_pct", "oil_temp_c"],
        recommended_actions=["Inspect fan/pump and motor", "Check VFD", "Repair or replace"],
    ),
    FaultDefinition(
        fault_id="CH-02",
        name="Coolant level low / leak",
        subsystem="cooling_hydraulics",
        subsystem_label="Cooling & Hydraulics",
        severity="MEDIUM",
        description="Coolant loss from the liquid-cooling circuit reduces heat "
        "removal and risks pump cavitation.",
        root_causes=["Hose/radiator leak", "Expansion-tank loss", "Pump seal wear"],
        symptoms=["Coolant level alarm", "System pressure dropping", "Temperature rising"],
        detection_signals=["coolant_level_pct", "coolant_pressure_bar", "coolant_temp_c"],
        recommended_actions=[
            "Locate and repair leak",
            "Top up with specified coolant",
            "Pressure-test circuit",
        ],
    ),
    FaultDefinition(
        fault_id="CH-03",
        name="Heat exchanger fouling",
        subsystem="cooling_hydraulics",
        subsystem_label="Cooling & Hydraulics",
        severity="LOW",
        description="Fouling of radiators/heat exchangers degrades heat transfer; "
        "observed as rising temperatures at constant load.",
        root_causes=["Dust/debris accumulation", "Biological fouling", "Scale"],
        symptoms=["Temperature spread across exchanger rising", "Cooler runs harder"],
        detection_signals=["coolant_temp_c", "heat_exchanger_delta_t"],
        recommended_actions=["Clean heat exchanger", "Check air filters", "Water treatment"],
    ),
    FaultDefinition(
        fault_id="CH-04",
        name="Hydraulic system pressure too low",
        subsystem="cooling_hydraulics",
        subsystem_label="Cooling & Hydraulics",
        severity="HIGH",
        description="Hydraulic pressure below setpoint (yaw brake, pitch, service "
        "crane circuits) — degraded actuation and braking.",
        root_causes=["Pump failure", "Leak", "Pressure switch fault", "Low fluid level"],
        symptoms=["Low-pressure alarm", "Slow actuator motion", "Frequent pump starts"],
        detection_signals=["hydraulic_pressure_bar", "hydraulic_level_pct"],
        recommended_actions=["Check pump and accumulator", "Leak inspection", "Top up fluid"],
    ),
    FaultDefinition(
        fault_id="CH-05",
        name="Hydraulic system pressure too high",
        subsystem="cooling_hydraulics",
        subsystem_label="Cooling & Hydraulics",
        severity="MEDIUM",
        description="Hydraulic pressure above setpoint risks seal and hose damage.",
        root_causes=["Relief valve stuck", "Accumulator over-charge", "Temperature rise"],
        symptoms=["High-pressure alarm", "Relief valve chattering"],
        detection_signals=["hydraulic_pressure_bar"],
        recommended_actions=[
            "Test relief valve",
            "Check accumulator pre-charge",
            "Verify pump regulator",
        ],
    ),
    FaultDefinition(
        fault_id="CH-06",
        name="Hydraulic oil contamination",
        subsystem="cooling_hydraulics",
        subsystem_label="Cooling & Hydraulics",
        severity="MEDIUM",
        description="Contaminated hydraulic fluid (particles/water) wears valves " "and actuators.",
        root_causes=["Filter clogging", "Breather failure", "Water ingress", "Wear debris"],
        symptoms=["Particle counts high", "Valve sluggishness", "Fluid cloudy"],
        detection_signals=["hydraulic_oil_particles_iso4406", "hydraulic_oil_water_ppm"],
        recommended_actions=["Fluid analysis", "Change filters", "Flush system if severe"],
    ),
    # ── 11. Electrical & Power Conversion ──────────────────────────────────
    FaultDefinition(
        fault_id="EL-01",
        name="Converter (IGBT) over-temperature / fault",
        subsystem="electrical",
        subsystem_label="Electrical & Power",
        severity="HIGH",
        description="Power-converter IGBT junction/module temperature above limit "
        "or converter fault codes — a major availability driver.",
        root_causes=[
            "Cooling failure",
            "Grid disturbances",
            "Module aging",
            "DC-link issues",
            "Fan failure",
        ],
        symptoms=["Converter temperature alarm", "Converter trip events", "Derating"],
        detection_signals=["converter_temp_c", "converter_faults_24h", "grid_frequency_hz"],
        recommended_actions=[
            "Check converter cooling",
            "Analyze fault codes",
            "Thermal imaging",
            "Plan module replacement",
        ],
    ),
    FaultDefinition(
        fault_id="EL-02",
        name="Transformer over-temperature",
        subsystem="electrical",
        subsystem_label="Electrical & Power",
        severity="MEDIUM",
        description="Tower/pad-mount transformer temperature above limit degrades "
        "insulation and accelerates aging.",
        root_causes=["Overload", "Cooling failure", "Harmonics", "Oil level low"],
        symptoms=["Transformer temperature alarm", "Oil level low", "Odor"],
        detection_signals=["transformer_temp_c", "transformer_oil_level_pct"],
        recommended_actions=[
            "Check cooling and level",
            "Load management",
            "Oil dissolved-gas analysis",
        ],
    ),
    FaultDefinition(
        fault_id="EL-03",
        name="Grid harmonics / power-quality issue",
        subsystem="electrical",
        subsystem_label="Electrical & Power",
        severity="LOW",
        description="Elevated harmonics or voltage imbalance stress the converter, "
        "transformer and generator.",
        root_causes=["Weak grid", "Resonance", "Other site equipment", "Converter control"],
        symptoms=["THD above limit", "Converter trips", "Transformer temperature rise"],
        detection_signals=["thd_pct", "voltage_unbalance_pct", "power_factor"],
        recommended_actions=["Power-quality study", "Check with utility", "Filter tuning"],
    ),
    FaultDefinition(
        fault_id="EL-04",
        name="Cable / insulation fault",
        subsystem="electrical",
        subsystem_label="Electrical & Power",
        severity="HIGH",
        description="Cable or insulation degradation (partial discharge, moisture) "
        "— precursor to short-circuit.",
        root_causes=["Aging", "Mechanical damage", "Moisture ingress", "Overheating"],
        symptoms=["Partial discharge activity", "Insulation resistance drop", "Cable temperature"],
        detection_signals=["partial_discharge_pc", "insulation_resistance_mohm"],
        recommended_actions=["PD measurement", "Megger testing", "Cable replacement"],
    ),
    FaultDefinition(
        fault_id="EL-05",
        name="DC-link / rectifier fault",
        subsystem="electrical",
        subsystem_label="Electrical & Power",
        severity="HIGH",
        description="DC-link capacitor or rectifier degradation causes converter "
        "trips and voltage ripple.",
        root_causes=["Capacitor aging", "Rectifier failure", "Cooling issues"],
        symptoms=["DC-link voltage ripple", "Converter trips", "Capacitor bulge (inspection)"],
        detection_signals=["dc_link_ripple_pct", "converter_faults_24h"],
        recommended_actions=[
            "Check DC-link capacitors",
            "Analyze converter events",
            "Replace capacitors",
        ],
    ),
    FaultDefinition(
        fault_id="EL-06",
        name="Electrical cabinet / transformer fire",
        subsystem="electrical",
        subsystem_label="Electrical & Power",
        severity="CRITICAL",
        description="Fire in a converter cabinet, transformer or cable compartment "
        "— the most common electrical fire source in wind turbines.",
        root_causes=[
            "Converter/IGBT thermal runaway",
            "Transformer fault",
            "Cable insulation failure",
            "Loose connection arcing",
        ],
        symptoms=[
            "Cabinet smoke alarm",
            "Transformer temperature extreme",
            "Fire suppression released",
        ],
        detection_signals=[
            "smoke_detector_on",
            "cabinet_fire_alarm",
            "transformer_temp_c",
            "fire_suppression_released",
        ],
        recommended_actions=[
            "Shut down and disconnect from grid",
            "Trigger fire suppression",
            "Evacuate tower base, call emergency services",
        ],
    ),
    # ── 12. SCADA & Communication ──────────────────────────────────────────
    FaultDefinition(
        fault_id="SC-01",
        name="Telemetry dropout / comms loss",
        subsystem="scada",
        subsystem_label="SCADA & Communication",
        severity="LOW",
        description="Missing or late SCADA samples — the condition-monitoring "
        "system cannot see the asset.",
        root_causes=["Network outage", "Controller reboot", "RTU failure", "Radio interference"],
        symptoms=["Gaps in telemetry", "Comm watchdog alarms"],
        detection_signals=["telemetry_gap_min", "comms_uptime_pct"],
        recommended_actions=["Check network path", "Inspect RTU/controller", "Restart comms stack"],
    ),
    FaultDefinition(
        fault_id="SC-02",
        name="Sensor drift / miscalibration",
        subsystem="scada",
        subsystem_label="SCADA & Communication",
        severity="LOW",
        description="A sensor drifts or saturates; readings become implausible or "
        "disagree with redundant sensors.",
        root_causes=[
            "Aging electronics",
            "Moisture",
            "Calibration interval exceeded",
            "Wiring fault",
        ],
        symptoms=["Redundant-sensor disagreement", "Unphysical trends", "Stuck values"],
        detection_signals=["sensor_disagreement_pct", "telemetry_gap_min"],
        recommended_actions=["Cross-calibrate sensors", "Replace suspect sensor", "Verify wiring"],
    ),
    FaultDefinition(
        fault_id="SC-03",
        name="Timestamp / synchronization error",
        subsystem="scada",
        subsystem_label="SCADA & Communication",
        severity="LOW",
        description="Clock drift breaks time alignment between channels and "
        "farms, corrupting analytics.",
        root_causes=["NTP failure", "Controller clock battery", "DST misconfiguration"],
        symptoms=["Channel skew", "Time jumps in historian"],
        detection_signals=["clock_skew_s"],
        recommended_actions=["Fix time sync", "Verify NTP servers", "Re-align historian"],
    ),
    FaultDefinition(
        fault_id="SC-04",
        name="Controller fault / watchdog trip",
        subsystem="scada",
        subsystem_label="SCADA & Communication",
        severity="MEDIUM",
        description="Turbine controller fault code or repeated watchdog restarts "
        "— intermittent control failures.",
        root_causes=["Software bug", "Power supply glitch", "I/O fault", "EMC issue"],
        symptoms=["Fault codes logged", "Unexpected restarts", "Mode changes"],
        detection_signals=["controller_faults_24h", "restarts_24h"],
        recommended_actions=["Read fault log", "Check power supply", "Firmware update"],
    ),
]


# --------------------------------------------------------------------------- #
# Catalog helpers                                                              #
# --------------------------------------------------------------------------- #
_CATALOG_BY_ID: dict[str, FaultDefinition] = {f.fault_id: f for f in FAULT_CATALOG}
_CATALOG_BY_SUBSYSTEM: dict[str, list[FaultDefinition]] = {}
for _f in FAULT_CATALOG:
    _CATALOG_BY_SUBSYSTEM.setdefault(_f.subsystem, []).append(_f)


def get_fault(fault_id: str) -> FaultDefinition:
    """Look up a fault definition by id (e.g. ``GB-02``)."""
    if fault_id not in _CATALOG_BY_ID:
        raise KeyError(f"unknown fault id '{fault_id}'; catalog has {len(FAULT_CATALOG)} faults")
    return _CATALOG_BY_ID[fault_id]


def faults_by_subsystem(subsystem: str) -> list[FaultDefinition]:
    """All fault types defined for one subsystem key."""
    if subsystem not in SUBSYSTEMS:
        raise KeyError(f"unknown subsystem '{subsystem}'; available: {sorted(SUBSYSTEMS)}")
    return list(_CATALOG_BY_SUBSYSTEM.get(subsystem, []))


def list_faults(subsystem: str | None = None) -> list[dict]:
    """Serializable summary of the catalog, optionally filtered by subsystem."""
    source = FAULT_CATALOG if subsystem is None else faults_by_subsystem(subsystem)
    return [f.to_dict() for f in source]


def catalog_summary() -> dict:
    """Per-subsystem counts and severity roll-up (used by CLI/API/docs)."""
    summary: dict = {}
    for key, label in SUBSYSTEMS.items():
        faults = _CATALOG_BY_SUBSYSTEM.get(key, [])
        summary[key] = {
            "subsystem": key,
            "label": label,
            "n_fault_types": len(faults),
            "by_severity": {sev: sum(1 for f in faults if f.severity == sev) for sev in SEVERITIES},
        }
    summary["total_fault_types"] = len(FAULT_CATALOG)
    summary["n_subsystems"] = len(SUBSYSTEMS)
    return summary
