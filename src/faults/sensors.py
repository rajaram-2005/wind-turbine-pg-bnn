"""Sensor catalog: which sensors measure what, and which faults they catch.

Every wind-turbine fault needs a sensor that can see it.  This catalog maps
the full chain — **sensor → measured channel → fault types** — for all 12
subsystems, so the user guide can answer "what hardware do I need and where
does it go?" and the detector can answer "which fault does this reading
feed?".

Sensor ids are grouped by category:

* ``WS``  wind measurement        * ``OC``  oil condition
* ``TH``  temperature             * ``FS``  fire & safety
* ``VB``  vibration / CMS         * ``PS``  position & actuation
* ``RM``  mechanical / RPM        * ``EL``  electrical
* ``DA``  data acquisition & comms

The ``channels`` of each sensor are the exact signal names understood by
:mod:`src.faults.detector` and the hardware gateway (see
``src.api.gateway_routes.SIGNAL_ALIASES``), so a sensor's reading feeds the
corresponding detection rules directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.faults.taxonomy import SUBSYSTEMS


@dataclass(frozen=True)
class SensorSpec:
    """One physical sensor / measurement device."""

    sensor_id: str
    name: str
    category: str  # WS | TH | VB | OC | FS | PS | RM | EL | DA
    placement: str  # where on the turbine / in the farm it is installed
    technology: str  # measurement principle
    output: str  # signal type: 4-20 mA, Modbus, CAN, digital, analog, ethernet
    channels: list[str] = field(default_factory=list)  # signal names consumed
    feeds_fault_ids: list[str] = field(default_factory=list)  # fault types it can reveal
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "sensor_id": self.sensor_id,
            "name": self.name,
            "category": self.category,
            "placement": self.placement,
            "technology": self.technology,
            "output": self.output,
            "channels": list(self.channels),
            "feeds_fault_ids": list(self.feeds_fault_ids),
            "notes": self.notes,
        }


SENSOR_CATALOG: list[SensorSpec] = [
    # ── Wind measurement ──────────────────────────────────────────────────
    SensorSpec(
        "WS-01",
        "Cup anemometer",
        "WS",
        "Nacelle roof (x2 for redundancy)",
        "Rotating cups -> pulse/analog",
        "4-20 mA / pulse",
        ["wind_speed_mps", "wind_speed2_mps"],
        ["NS-01"],
        "Heated version available for icing sites.",
    ),
    SensorSpec(
        "WS-02",
        "Ultrasonic anemometer",
        "WS",
        "Nacelle roof or spinner",
        "Ultrasonic transit-time",
        "Modbus RTU/TCP",
        ["wind_speed_mps", "wind_speed2_mps"],
        ["NS-01"],
        "No moving parts; better in icing.",
    ),
    SensorSpec(
        "WS-03",
        "Wind vane",
        "WS",
        "Nacelle roof (x2 for redundancy)",
        "Potentiometer / encoder",
        "4-20 mA / analog",
        ["wind_vane_deg", "wind_vane2_deg"],
        ["NS-02", "YW-03"],
    ),
    SensorSpec(
        "WS-04",
        "Ice detector",
        "WS",
        "Nacelle roof / blade root",
        "Resonance / conductivity change",
        "Digital / Modbus",
        ["ice_detector_on", "ambient_temp_c"],
        ["RB-03"],
    ),
    # ── Temperature ───────────────────────────────────────────────────────
    SensorSpec(
        "TH-01",
        "Main bearing RTD",
        "TH",
        "Main bearing housing",
        "Resistance temperature detector (Pt100)",
        "4-20 mA",
        ["main_bearing_temp_c"],
        ["HS-01"],
    ),
    SensorSpec(
        "TH-02",
        "Gearbox oil temperature sensor",
        "TH",
        "Gearbox oil sump / inlet",
        "RTD / thermocouple",
        "4-20 mA / Modbus",
        ["oil_temp_c", "gearbox_temp_c", "temperature_c"],
        ["GB-01", "GB-15", "GB-04"],
    ),
    SensorSpec(
        "TH-03",
        "Generator winding RTD",
        "TH",
        "Generator stator slots",
        "RTD (Pt100)",
        "4-20 mA / Modbus",
        ["stator_temp_c", "generator_temp_c", "generator_winding_temp_c"],
        ["GN-01", "GN-03"],
    ),
    SensorSpec(
        "TH-04",
        "Generator bearing thermocouple",
        "TH",
        "DE & NDE bearing caps",
        "Thermocouple (K)",
        "4-20 mA",
        ["generator_bearing_temp_c"],
        ["GN-02"],
    ),
    SensorSpec(
        "TH-05",
        "Ambient temperature sensor",
        "TH",
        "Nacelle exterior / met mast",
        "RTD / thermistor",
        "4-20 mA",
        ["ambient_temp_c"],
        ["RB-03", "RB-07"],
    ),
    SensorSpec(
        "TH-06",
        "Nacelle temperature & humidity",
        "TH",
        "Nacelle interior",
        "Combined T/RH probe",
        "Modbus",
        ["nacelle_temp_c", "nacelle_humidity_pct"],
        ["NS-05", "NS-06"],
    ),
    SensorSpec(
        "TH-07",
        "Transformer temperature sensor",
        "TH",
        "Tower base / pad-mount transformer",
        "RTD / winding temp indicator",
        "4-20 mA / Modbus",
        ["transformer_temp_c"],
        ["EL-02", "EL-06"],
    ),
    SensorSpec(
        "TH-08",
        "Converter IGBT temperature sensor",
        "TH",
        "Converter power modules",
        "NTC thermistor on module",
        "Modbus (converter)",
        ["converter_temp_c"],
        ["EL-01"],
    ),
    SensorSpec(
        "TH-09",
        "Coolant temperature sensor",
        "TH",
        "Cooling circuit in/out",
        "RTD",
        "4-20 mA",
        ["coolant_temp_c"],
        ["CH-01", "CH-02", "CH-03"],
    ),
    SensorSpec(
        "TH-10",
        "Brake disc temperature sensor",
        "TH",
        "HSS brake caliper",
        "Thermocouple / IR",
        "4-20 mA",
        ["brake_temp_c"],
        ["BR-04", "BR-05"],
    ),
    SensorSpec(
        "TH-11",
        "Slip-ring temperature sensor",
        "TH",
        "Generator slip-ring housing",
        "RTD",
        "4-20 mA",
        ["slip_ring_temp_c"],
        ["GN-06"],
    ),
    SensorSpec(
        "TH-12",
        "Infrared thermography camera",
        "TH",
        "Nacelle interior / drone payload",
        "IR imaging",
        "Ethernet / video",
        ["blade_temp_c", "inspection_hotspot"],
        ["RB-07", "EL-01", "NS-06"],
        "Drone or fixed camera for hotspots.",
    ),
    # ── Vibration / condition monitoring ──────────────────────────────────
    SensorSpec(
        "VB-01",
        "CMS accelerometer (drive train)",
        "VB",
        "Main bearing, gearbox, generator",
        "IEPE accelerometer, 10 Hz-10 kHz",
        "4-20 mA / IEPE / Ethernet",
        ["vibration_mms", "hss_vibration_mms", "generator_vibration_mms", "blade_1p_amplitude_mms"],
        ["HS-01", "RB-01", "GB-09", "GB-10", "GB-11", "BR-03", "GN-07"],
        "The backbone of the condition monitoring system.",
    ),
    SensorSpec(
        "TH-13",
        "Coolant flow / pressure meter",
        "TH",
        "Cooling circuit",
        "Flow meter + pressure transmitter",
        "4-20 mA",
        ["coolant_flow_pct", "coolant_pressure_bar"],
        ["GN-05", "CH-01", "CH-02"],
    ),
    SensorSpec(
        "VB-02",
        "Gearbox high-frequency vibration sensor",
        "VB",
        "Gearbox housing per stage",
        "Accelerometer, high-freq envelope",
        "IEPE / Ethernet",
        ["gmf_sideband_amplitude", "gmf_sideband_amplitude_mms", "gear_rattle_index"],
        ["GB-09", "GB-10", "GB-14"],
    ),
    SensorSpec(
        "VB-03",
        "Bearing vibration / envelope sensor",
        "VB",
        "Main + gearbox + generator bearings",
        "Accelerometer + envelope (BPFO/BPFI/BSF/FTF)",
        "Ethernet CMS",
        ["bpfo_amplitude_mms", "bearing_temp_c"],
        ["GB-11", "GN-02"],
    ),
    SensorSpec(
        "VB-04",
        "Tower accelerometer",
        "VB",
        "Tower top / mid tower",
        "DC-response accelerometer",
        "4-20 mA",
        ["tower_vibration_mms", "tower_accel_g"],
        ["TF-01", "NS-03"],
    ),
    SensorSpec(
        "VB-05",
        "Blade strain gauge",
        "VB",
        "Blade root / spar (per blade)",
        "Fibre-optic or foil strain gauge",
        "Ethernet / analog",
        ["blade_strain_ue", "blade_tip_deflection_pct"],
        ["RB-08", "RB-02"],
        "Fibre-optic strain sensing also gives blade load histograms.",
    ),
    SensorSpec(
        "VB-06",
        "Nacelle oscillation sensor",
        "VB",
        "Nacelle frame",
        "Accelerometer (fore-aft / side-side)",
        "4-20 mA",
        ["nacelle_oscillation_mms"],
        ["NS-03", "TF-01"],
    ),
    SensorSpec(
        "VB-07",
        "Tilt sensor",
        "VB",
        "Tower flange / foundation",
        "MEMS inclinometer",
        "Modbus / 4-20 mA",
        ["tower_tilt_deg"],
        ["TF-03"],
    ),
    SensorSpec(
        "VB-08",
        "Acoustic emission sensor",
        "VB",
        "Blade shell / gearbox",
        "AE transducer (ultrasonic events)",
        "Ethernet",
        ["blade_acoustic_anomaly"],
        ["RB-05", "RB-09"],
    ),
    # ── Oil condition ─────────────────────────────────────────────────────
    SensorSpec(
        "OC-01",
        "Online viscometer",
        "OC",
        "Gearbox oil line",
        "Vibrating tuning-fork viscosity",
        "4-20 mA / Modbus",
        ["oil_viscosity_cst"],
        ["GB-02", "GB-03", "GB-13"],
    ),
    SensorSpec(
        "OC-02",
        "Water-in-oil sensor",
        "OC",
        "Gearbox oil line",
        "Capacitive / hygrometry",
        "4-20 mA / Modbus",
        ["oil_water_ppm", "oil_moisture_pct"],
        ["GB-04"],
    ),
    SensorSpec(
        "OC-03",
        "Online particle counter (ISO 4406)",
        "OC",
        "Gearbox oil return line",
        "Optical / laser occlusion counting",
        "Modbus",
        ["oil_particles_iso4406"],
        ["GB-05", "GB-07"],
    ),
    SensorSpec(
        "OC-04",
        "Oil level sensor",
        "OC",
        "Gearbox sump",
        "Ultrasonic / capacitive / float",
        "4-20 mA",
        ["oil_level_pct"],
        ["GB-06"],
    ),
    SensorSpec(
        "OC-05",
        "Oil filter differential-pressure switch",
        "OC",
        "Gearbox filter housing",
        "Pressure switch / DP transmitter",
        "Digital / 4-20 mA",
        ["oil_filter_dp_bar"],
        ["GB-07"],
    ),
    SensorSpec(
        "OC-06",
        "Oil pressure transmitter",
        "OC",
        "Gearbox supply line",
        "Piezoresistive pressure cell",
        "4-20 mA",
        ["oil_pressure_bar"],
        ["GB-12"],
    ),
    SensorSpec(
        "OC-07",
        "Oil aeration / foam sensor",
        "OC",
        "Gearbox return line / tank",
        "Optical / ultrasonic",
        "4-20 mA / Modbus",
        ["oil_aeration_pct", "oil_foam_pct"],
        ["GB-13"],
    ),
    SensorSpec(
        "OC-08",
        "Ferrous debris (wear metal) sensor",
        "OC",
        "Gearbox oil line / magnetic plug",
        "Inductive particle / spectrometric (lab)",
        "Modbus / lab report",
        ["oil_iron_ppm", "grease_debris_ppm"],
        ["GB-09", "GB-10", "GB-11", "HS-01"],
    ),
    SensorSpec(
        "OC-09",
        "Oil sampling port + laboratory analysis",
        "OC",
        "Gearbox / hydraulic system",
        "Offline lab: TAN, viscosity, FTIR, spectroscopy",
        "Lab report (periodic)",
        ["oil_tan_mgkoh_g", "oil_oxidation_pct"],
        ["GB-08", "GB-02", "GB-03"],
        "Monthly sampling complements the online sensors.",
    ),
    # ── Fire & safety ─────────────────────────────────────────────────────
    SensorSpec(
        "FS-01",
        "Smoke detector",
        "FS",
        "Nacelle, hub, tower base, converter cabinet",
        "Photoelectric / ionisation",
        "Digital / relay",
        ["smoke_detector_on"],
        ["NS-06", "BR-05", "GB-15", "EL-06"],
    ),
    SensorSpec(
        "FS-02",
        "Heat detector",
        "FS",
        "Nacelle, gearbox, brake area",
        "Rate-of-rise / fixed temperature",
        "Digital / relay",
        ["blade_temp_c", "nacelle_temp_c"],
        ["NS-06", "BR-05", "GB-15"],
    ),
    SensorSpec(
        "FS-03",
        "Flame detector (IR/UV)",
        "FS",
        "Nacelle, converter cabinet, tower base",
        "IR/UV flame flicker sensing",
        "Relay / Modbus",
        ["blade_fire_alarm", "cabinet_fire_alarm", "tower_fire_alarm", "brake_fire_alarm"],
        ["RB-07", "EL-06", "TF-05", "BR-05"],
    ),
    SensorSpec(
        "FS-04",
        "Gas detector (CO / CO2)",
        "FS",
        "Nacelle, battery/cabinet areas",
        "Electrochemical / NDIR",
        "4-20 mA / Modbus",
        ["smoke_detector_on"],
        ["NS-06", "EL-06"],
        "Early incipient-fire detection (overheated cables).",
    ),
    SensorSpec(
        "FS-05",
        "Fire suppression system status",
        "FS",
        "Suppression bottles, nacelle & tower base",
        "Pressure switches, release circuit monitoring",
        "Digital / Modbus",
        ["fire_suppression_status", "fire_suppression_released"],
        ["NS-07", "GB-15", "EL-06", "TF-05"],
    ),
    SensorSpec(
        "FS-06",
        "Lightning counter & SPD monitor",
        "FS",
        "Blade receptors, nacelle, tower",
        "Event counter / surge protection device status",
        "Modbus",
        ["lightning_events_24h", "inspection_lightning_damage"],
        ["RB-04", "RB-07", "NS-04"],
    ),
    # ── Position & actuation ──────────────────────────────────────────────
    SensorSpec(
        "PS-01",
        "Pitch angle encoder",
        "PS",
        "Pitch bearing / blade root (per blade)",
        "Resolver / absolute encoder",
        "CANopen / SSI",
        ["pitch_angle_deg", "blade_pitch_deviation_deg", "pitch_position_error_deg"],
        ["RB-06", "PT-03", "PT-06"],
    ),
    SensorSpec(
        "PS-02",
        "Pitch motor current / torque monitor",
        "PS",
        "Pitch drive cabinet",
        "Current transformer / drive telemetry",
        "Modbus",
        ["pitch_torque_pct", "pitch_motor_current_a", "pitch_fault_code"],
        ["PT-01", "PT-02"],
    ),
    SensorSpec(
        "PS-03",
        "Yaw angle encoder",
        "PS",
        "Yaw bearing / ring gear",
        "Incremental / absolute encoder",
        "CANopen / Modbus",
        ["yaw_error_deg", "yaw_turns_24h", "cable_twist_turns"],
        ["YW-03", "YW-04"],
    ),
    SensorSpec(
        "PS-04",
        "Yaw drive current / torque monitor",
        "PS",
        "Yaw drive cabinet",
        "Current transformer / drive telemetry",
        "Modbus",
        ["yaw_torque_pct", "yaw_motor_current_a", "yaw_drive_faults_24h"],
        ["YW-01", "YW-02"],
    ),
    SensorSpec(
        "PS-05",
        "Brake wear sensor",
        "PS",
        "HSS brake caliper",
        "Wear pin / limit switch",
        "Digital",
        ["brake_wear_pct"],
        ["BR-01"],
    ),
    SensorSpec(
        "PS-06",
        "Hydraulic pressure transmitter",
        "PS",
        "Hydraulic power unit",
        "Piezoresistive pressure cell",
        "4-20 mA",
        ["hydraulic_pressure_bar", "pitch_hydraulic_pressure_bar", "yaw_brake_pressure_bar"],
        ["PT-04", "PT-05", "CH-04", "CH-05", "YW-05"],
    ),
    SensorSpec(
        "PS-07",
        "Accumulator pressure / feather-time monitor",
        "PS",
        "Pitch accumulator",
        "Pressure switch + function test",
        "Digital",
        ["hydraulic_pressure_decay_bar_s", "feather_time_s"],
        ["PT-05", "PT-04"],
    ),
    SensorSpec(
        "PS-08",
        "Hydraulic oil condition sensor",
        "PS",
        "Hydraulic power unit",
        "Particle counter + water-in-oil",
        "Modbus",
        ["hydraulic_oil_particles_iso4406", "hydraulic_oil_water_ppm"],
        ["CH-06"],
    ),
    # ── Mechanical / RPM ──────────────────────────────────────────────────
    SensorSpec(
        "RM-01",
        "HSS speed encoder",
        "RM",
        "High-speed shaft / gearbox output",
        "Magnetic pickup / encoder",
        "Pulse / CAN",
        ["rpm", "overspeed_trips_24h"],
        ["BR-02"],
    ),
    SensorSpec(
        "RM-02",
        "Rotor speed sensor",
        "RM",
        "Main shaft / hub",
        "Magnetic pickup / encoder",
        "Pulse / CAN",
        ["rotor_speed_rpm"],
        ["RB-08", "HS-04"],
    ),
    SensorSpec(
        "RM-03",
        "Shaft axial displacement probe",
        "RM",
        "Main shaft thrust collar",
        "Eddy-current proximity probe",
        "4-20 mA",
        ["shaft_axial_displacement_mm"],
        ["HS-02"],
    ),
    SensorSpec(
        "RM-04",
        "Torque sensor",
        "RM",
        "Main shaft / HSS",
        "Strain-gauge torque flange",
        "CAN / 4-20 mA",
        ["torque_spike_amplitude"],
        ["HS-04", "GB-10"],
    ),
    # ── Electrical ────────────────────────────────────────────────────────
    SensorSpec(
        "EL-01",
        "Power meter",
        "EL",
        "Turbine LV/MV switchboard",
        "Current + voltage transformers, energy metering",
        "Modbus / IEC 61850",
        ["power_kw", "power_factor", "aep_deviation_pct"],
        ["RB-02", "RB-03", "EL-03"],
    ),
    SensorSpec(
        "EL-02",
        "Power quality analyzer (THD / unbalance)",
        "EL",
        "Turbine terminals",
        "Harmonic analysis of V/I",
        "Modbus / Ethernet",
        ["thd_pct", "voltage_unbalance_pct", "grid_frequency_hz"],
        ["EL-03"],
    ),
    SensorSpec(
        "EL-03",
        "Partial discharge sensor",
        "EL",
        "Generator winding, cables, transformer",
        "HFCT / UHF coupler",
        "Ethernet",
        ["partial_discharge_pc"],
        ["EL-04", "GN-04"],
    ),
    SensorSpec(
        "EL-04",
        "DC-link monitor",
        "EL",
        "Converter DC bus",
        "Voltage ripple measurement",
        "Modbus (converter)",
        ["dc_link_ripple_pct"],
        ["EL-05"],
    ),
    SensorSpec(
        "EL-05",
        "Insulation resistance monitor",
        "EL",
        "Generator / cables",
        "Megger / IR measurement",
        "Modbus",
        ["insulation_resistance_mohm"],
        ["GN-04", "EL-04"],
    ),
    SensorSpec(
        "EL-06",
        "Current signature analyzer (MCSA)",
        "EL",
        "Generator / motor supply",
        "Motor current signature analysis",
        "Ethernet",
        ["motor_current_imbalance_pct", "rotor_current_imbalance_pct"],
        ["GN-03", "GN-06"],
    ),
    # ── Data acquisition & communication ──────────────────────────────────
    SensorSpec(
        "DA-01",
        "Turbine controller / PLC",
        "DA",
        "Nacelle control cabinet",
        "Programmable logic controller",
        "Modbus TCP / OPC-UA",
        [
            "controller_faults_24h",
            "restarts_24h",
            "pitch_fault_code",
            "yaw_drive_faults_24h",
            "sensor_disagreement_pct",
        ],
        ["SC-04", "SC-02", "PT-02", "YW-02"],
    ),
    SensorSpec(
        "DA-02",
        "RTU / protocol gateway",
        "DA",
        "Tower base / substation",
        "Protocol conversion (Modbus, OPC-UA, MQTT, IEC 61850)",
        "Ethernet / cellular",
        ["telemetry_gap_min", "comms_uptime_pct", "sensor_disagreement_pct"],
        ["SC-01", "SC-02"],
    ),
    SensorSpec(
        "DA-03",
        "Edge AI node",
        "DA",
        "Nacelle or tower base",
        "ESP32 / STM32 / Raspberry Pi running edge firmware",
        "WiFi / LTE / Ethernet",
        ["telemetry_gap_min"],
        ["SC-01"],
        "See edge/ directory for ESP32 & STM32 firmware.",
    ),
    SensorSpec(
        "DA-04",
        "Drone / rope-access inspection camera",
        "DA",
        "Blades, tower, hub (periodic)",
        "High-res imaging + ML defect detection",
        "Offline reports",
        [
            "inspection_crack",
            "inspection_corrosion",
            "inspection_blade_delamination",
            "inspection_bolt_loose",
            "inspection_lightning_damage",
            "inspection_hub_crack",
        ],
        ["RB-04", "RB-05", "RB-09", "RB-10", "HS-03", "TF-02", "TF-04", "NS-04"],
    ),
    SensorSpec(
        "DA-05",
        "Time sync (NTP / GPS)",
        "DA",
        "Network-wide",
        "NTP / PTP / GPS clock",
        "Network",
        ["clock_skew_s"],
        ["SC-03"],
    ),
]

_CATALOG_BY_ID: dict[str, SensorSpec] = {s.sensor_id: s for s in SENSOR_CATALOG}


def get_sensor(sensor_id: str) -> SensorSpec:
    """Look up one sensor by id (e.g. ``OC-01``)."""
    if sensor_id not in _CATALOG_BY_ID:
        raise KeyError(
            f"unknown sensor id '{sensor_id}'; catalog has {len(SENSOR_CATALOG)} sensors"
        )
    return _CATALOG_BY_ID[sensor_id]


def sensors_by_category(category: str) -> list[SensorSpec]:
    """Sensors in one category (``WS``, ``TH``, ``VB``, ``OC``, ``FS``, ...)."""
    return [s for s in SENSOR_CATALOG if s.category == category]


def sensors_for_fault(fault_id: str) -> list[SensorSpec]:
    """Sensors that can reveal a given fault type."""
    return [s for s in SENSOR_CATALOG if fault_id in s.feeds_fault_ids]


def sensors_for_subsystem(subsystem: str) -> list[SensorSpec]:
    """Sensors whose fault list touches a given subsystem."""
    if subsystem not in SUBSYSTEMS:
        raise KeyError(f"unknown subsystem '{subsystem}'; available: {sorted(SUBSYSTEMS)}")
    prefix = subsystem_to_prefix(subsystem)
    return [s for s in SENSOR_CATALOG if any(f.startswith(prefix) for f in s.feeds_fault_ids)]


def subsystem_to_prefix(subsystem: str) -> str:
    """Map a subsystem key to its fault-id prefix (e.g. 'gearbox' -> 'GB')."""
    mapping = {
        "rotor_blades": "RB",
        "pitch": "PT",
        "hub_mainshaft": "HS",
        "gearbox": "GB",
        "hss_brake": "BR",
        "generator": "GN",
        "yaw": "YW",
        "tower_foundation": "TF",
        "nacelle_sensors": "NS",
        "cooling_hydraulics": "CH",
        "electrical": "EL",
        "scada": "SC",
    }
    return mapping[subsystem]


def sensor_catalog_dict(subsystem: str | None = None) -> dict:
    """Serializable catalog, optionally filtered by subsystem."""
    sensors = SENSOR_CATALOG if subsystem is None else sensors_for_subsystem(subsystem)
    by_category: dict[str, int] = {}
    for sensor in SENSOR_CATALOG:
        by_category[sensor.category] = by_category.get(sensor.category, 0) + 1
    return {
        "summary": {
            "n_sensors": len(SENSOR_CATALOG),
            "n_sensors_filtered": len(sensors),
            "filtered_subsystem": subsystem,
            "by_category": by_category,
            "n_fault_types_covered": len(
                {fid for s in SENSOR_CATALOG for fid in s.feeds_fault_ids}
            ),
        },
        "sensors": [s.to_dict() for s in sensors],
    }
