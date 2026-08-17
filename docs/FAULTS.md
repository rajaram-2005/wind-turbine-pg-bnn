# 🔍 Whole-Turbine Fault Detection

AeroVigil detects faults in **every part** of a wind turbine — blades, pitch,
hub & main shaft, gearbox (including oil-condition faults), high-speed shaft
& brake, generator, yaw, tower & foundation, nacelle & sensors, cooling &
hydraulics, electrical & power conversion, and SCADA & communication.

The fault catalog (`src/faults/taxonomy.py`) defines **71 fault types across
12 subsystems**. For every fault it records the symptoms, root causes, the
signals it is detected from, severity, and recommended maintenance actions.

## How it finds faults

| Layer | Module | What it does |
| --- | --- | --- |
| Catalog | `src/faults/taxonomy.py` | Every subsystem × every fault type, with symptoms, root causes, actions |
| Oil analysis | `src/faults/oil.py` | Gearbox oil-condition scoring: viscosity, water, ISO 4406 particles, TAN, filter ΔP, level, aeration, pressure, wear metals → 0–100 oil health score |
| Detector | `src/faults/detector.py` | Rule engine that turns one telemetry snapshot into a ranked fault report with confidence, evidence, severity and recommended actions |
| Digital twin | `src/digital_twin/twin.py` | Runs detection on every state update; each state record embeds `fault_report` |
| API | `POST /api/faults/detect`, `GET /api/faults/catalog` | Detect from any snapshot; browse the catalog |
| CLI | `python main.py faults ...` | Detect from a JSON snapshot or print the catalog |

## The fault catalog (all 12 subsystems)

### 1 · Rotor & Blades (`RB`)
| ID | Fault type | Severity | Detected from |
| --- | --- | --- | --- |
| RB-01 | Blade mass imbalance | MEDIUM | `blade_1p_amplitude_mms` > 1.5 / 2.5 |
| RB-02 | Leading-edge erosion | LOW | `aep_deviation_pct` > 5 / 10 |
| RB-03 | Blade icing | MEDIUM | `ice_detector_on`, or freezing + power droop |
| RB-04 | Lightning strike damage | HIGH | `lightning_events_24h` ≥ 3 / 10, `inspection_lightning_damage` |
| RB-05 | Blade crack / structural damage | CRITICAL | `inspection_crack`, `blade_acoustic_anomaly` |
| RB-06 | Pitch-angle asymmetry | MEDIUM | `blade_pitch_deviation_deg` > 0.5° / 1.5° |

### 2 · Pitch System (`PT`)
| ID | Fault type | Severity | Detected from |
| --- | --- | --- | --- |
| PT-01 | Pitch bearing wear / spalling | MEDIUM | `pitch_torque_pct` > 60 / 80 |
| PT-02 | Pitch drive motor / gearbox fault | HIGH | `pitch_fault_code`, torque ≥ 90 % |
| PT-03 | Pitch angle sensor (encoder) fault | HIGH | `pitch_sensor_disagreement`, error > 2° |
| PT-04 | Hydraulic pitch pressure loss | CRITICAL | `pitch_hydraulic_pressure_bar` < 140 / 110 |
| PT-05 | Accumulator pre-charge loss | MEDIUM | `feather_time_s` > 3.5 / 5 |
| PT-06 | Pitch angle drift / deviation | MEDIUM | `pitch_position_error_deg` > 0.5° / 1.5° |

### 3 · Hub & Main Shaft (`HS`)
| ID | Fault type | Severity | Detected from |
| --- | --- | --- | --- |
| HS-01 | Main (rotor) bearing wear | HIGH | `main_bearing_temp_c` > 60/70, `grease_debris_ppm` > 200/500, vibration > spec limit |
| HS-02 | Main shaft misalignment / bending | HIGH | `shaft_axial_displacement_mm` > 0.8 / 1.5 |
| HS-03 | Hub / spinner structural crack | HIGH | `inspection_hub_crack` |
| HS-04 | Coupling / shrink-disc slip | HIGH | `torque_spike_amplitude` > 15 / 30 % |

### 4 · Gearbox (`GB`) — including all oil faults
| ID | Fault type | Severity | Detected from |
| --- | --- | --- | --- |
| GB-01 | Oil temperature too high | MEDIUM | `oil_temp_c` > 92 % / 100 % of spec limit |
| GB-02 | **Oil viscosity too low** | HIGH | `oil_viscosity_cst` < spec min / 0.85 × min |
| GB-03 | **Oil viscosity too high** | MEDIUM | `oil_viscosity_cst` > spec max / 1.15 × max |
| GB-04 | **Water contamination** | HIGH | `oil_water_ppm` ≥ 300 / 1000, `oil_moisture_pct` ≥ 50 / 80 |
| GB-05 | **Particle contamination (ISO 4406)** | MEDIUM | code > 17 / 19 (target 17/15/12) |
| GB-06 | **Oil level too low** | HIGH | `oil_level_pct` < 20 / 10 |
| GB-07 | **Oil filter clogging (ΔP)** | MEDIUM | `oil_filter_dp_bar` ≥ 1.5 / 2.5 |
| GB-08 | **Oil oxidation / high TAN** | MEDIUM | `oil_tan_mgkoh_g` ≥ 0.8 / 2.0 |
| GB-09 | Gear tooth pitting / surface fatigue | HIGH | GMF sidebands > 2 / 4, `oil_iron_ppm` ≥ 300 |
| GB-10 | Gear tooth breakage / scuffing | CRITICAL | vibration > 2.5 × limit + iron ≥ 300 ppm |
| GB-11 | Gearbox bearing wear (HSS/ISS/LSS) | HIGH | `bpfo_amplitude_mms` > 1.5 / 3, `bearing_temp_c` > 70/80 |
| GB-12 | Lubrication starvation | HIGH | `oil_pressure_bar` < 1.5 / 1.0 |
| GB-13 | Oil aeration / foaming | MEDIUM | `oil_aeration_pct` ≥ 6 / 15, `oil_foam_pct` ≥ 25 |
| GB-14 | Gear backlash increase / rattle | LOW | `backlash_mm` > 0.3 / 0.6 |

### 5 · High-Speed Shaft & Brake (`BR`)
| ID | Fault type | Severity | Detected from |
| --- | --- | --- | --- |
| BR-01 | Brake pad / caliper wear | MEDIUM | `brake_wear_pct` ≥ 80 / 95 |
| BR-02 | HSS over-speed | CRITICAL | `rpm` > spec limit, `overspeed_trips_24h` ≥ 1 / 3 |
| BR-03 | HSS vibration / coupling fault | MEDIUM | `hss_vibration_mms` > 4.5 / 7.1 |
| BR-04 | Brake dragging / not releasing | MEDIUM | `brake_temp_c` > 60/80, `brake_drag_current_pct` > 10 |

### 6 · Generator (`GN`)
| ID | Fault type | Severity | Detected from |
| --- | --- | --- | --- |
| GN-01 | Stator winding over-temperature | HIGH | `stator_temp_c`/`generator_temp_c` > 105 / 120 |
| GN-02 | Generator bearing wear | MEDIUM | `generator_bearing_temp_c` > 75 / 90 |
| GN-03 | Air-gap eccentricity | MEDIUM | `motor_current_imbalance_pct` > 5 / 10 |
| GN-04 | Winding insulation degradation | HIGH | `insulation_resistance_mohm` < 100 / 10 |
| GN-05 | Generator cooling failure | HIGH | `coolant_flow_pct` < 60 / 30 |
| GN-06 | Slip ring / brush wear (DFIG) | MEDIUM | `slip_ring_temp_c` > 70 / 85 |
| GN-07 | Generator vibration | MEDIUM | `generator_vibration_mms` > 4.5 / 7.1 |

### 7 · Yaw System (`YW`)
| ID | Fault type | Severity | Detected from |
| --- | --- | --- | --- |
| YW-01 | Yaw bearing wear | MEDIUM | `yaw_torque_pct` > 60 / 85 |
| YW-02 | Yaw drive / motor fault | MEDIUM | `yaw_drive_faults_24h` ≥ 1 / 3 |
| YW-03 | Yaw misalignment (tracking error) | LOW | `yaw_error_deg` > 10° / 20° |
| YW-04 | Cable twist / unwinding fault | HIGH | `cable_twist_turns` ≥ 2.5 / 3.5 |
| YW-05 | Yaw brake failure | MEDIUM | `yaw_brake_pressure_bar` < 100 / 60 |

### 8 · Tower & Foundation (`TF`)
| ID | Fault type | Severity | Detected from |
| --- | --- | --- | --- |
| TF-01 | Tower resonance / vibration | HIGH | `tower_vibration_mms` > 3.5 / 6, `tower_accel_g` > 0.15 / 0.25 |
| TF-02 | Foundation bolt loosening | HIGH | `bolt_tension_deviation_pct` > 10 / 20, `inspection_bolt_loose` |
| TF-03 | Tower tilt / settlement | MEDIUM | `tower_tilt_deg` > 0.2° / 0.5° |
| TF-04 | Tower / foundation corrosion | LOW | `inspection_corrosion`, `tower_humidity_pct` ≥ 80 |

### 9 · Nacelle & Sensors (`NS`)
| ID | Fault type | Severity | Detected from |
| --- | --- | --- | --- |
| NS-01 | Anemometer fault / icing | MEDIUM | `anemometer_stuck`, redundant-sensor disagreement ≥ 20 % |
| NS-02 | Wind vane fault | MEDIUM | vane disagreement ≥ 15° |
| NS-03 | Nacelle oscillation | MEDIUM | `nacelle_oscillation_mms` > 2.5 / 5 |
| NS-04 | Lightning protection failure | LOW | `lightning_events_24h` ≥ 3, `inspection_lightning_damage` |
| NS-05 | Nacelle HVAC failure | LOW | `nacelle_temp_c` > 40/50, humidity ≥ 80 % |
| NS-06 | Nacelle fire / smoke detection | CRITICAL | `smoke_detector_on`, `nacelle_temp_c` ≥ 60 |

### 10 · Cooling & Hydraulics (`CH`)
| ID | Fault type | Severity | Detected from |
| --- | --- | --- | --- |
| CH-01 | Cooling fan / pump failure | MEDIUM | `cooling_fan_fault`, hot coolant + low fan runtime |
| CH-02 | Coolant level low / leak | MEDIUM | `coolant_level_pct` < 30 / 15 |
| CH-03 | Heat exchanger fouling | LOW | `heat_exchanger_delta_t` > 12 / 18 K |
| CH-04 | Hydraulic pressure too low | HIGH | `hydraulic_pressure_bar` < 140 / 110 |
| CH-05 | Hydraulic pressure too high | MEDIUM | `hydraulic_pressure_bar` > 210 / 230 |
| CH-06 | Hydraulic oil contamination | MEDIUM | ISO 4406 > 17/19, `hydraulic_oil_water_ppm` ≥ 300 |

### 11 · Electrical & Power (`EL`)
| ID | Fault type | Severity | Detected from |
| --- | --- | --- | --- |
| EL-01 | Converter (IGBT) over-temperature / fault | HIGH | `converter_temp_c` > 70/85, `converter_faults_24h` ≥ 1/3 |
| EL-02 | Transformer over-temperature | MEDIUM | `transformer_temp_c` > 95/110, oil level < 30 % |
| EL-03 | Grid harmonics / power quality | LOW | `thd_pct` > 5/8, `voltage_unbalance_pct` > 2/4 |
| EL-04 | Cable / insulation fault | HIGH | `partial_discharge_pc` > 500/2000, IR < 100 MΩ |
| EL-05 | DC-link / rectifier fault | HIGH | `dc_link_ripple_pct` > 5 / 10 |

### 12 · SCADA & Communication (`SC`)
| ID | Fault type | Severity | Detected from |
| --- | --- | --- | --- |
| SC-01 | Telemetry dropout / comms loss | LOW | `telemetry_gap_min` > 30/120, `comms_uptime_pct` < 99/95 |
| SC-02 | Sensor drift / miscalibration | LOW | `sensor_disagreement_pct` > 5 / 15 |
| SC-03 | Timestamp / sync error | LOW | `clock_skew_s` > 5 / 60 |
| SC-04 | Controller fault / watchdog trip | MEDIUM | `controller_faults_24h` ≥ 1/3, `restarts_24h` ≥ 3/5 |

## The oil-condition score

`src/faults/oil.py` scores gearbox oil on up to ten parameters. Each is
`OK` / `WARN` / `ALARM`; warnings cost 8 points and alarms 22 points off a
100-point oil health score:

| Parameter | Channel | Warn | Alarm |
| --- | --- | --- | --- |
| Kinematic viscosity | `oil_viscosity_cst` | outside spec window | beyond window ± margin |
| Water content | `oil_water_ppm` | ≥ 300 ppm | ≥ 1000 ppm |
| Moisture saturation | `oil_moisture_pct` | ≥ 50 % | ≥ 80 % |
| Particles (ISO 4406) | `oil_particles_iso4406` | code > 17 | code > 19 |
| Total acid number | `oil_tan_mgkoh_g` | ≥ 0.8 | ≥ 2.0 |
| Filter ΔP | `oil_filter_dp_bar` | ≥ 1.5 bar | ≥ 2.5 bar |
| Oil level | `oil_level_pct` | < 20 % | < 10 % |
| Supply pressure | `oil_pressure_bar` | < 1.5 bar | < 1.0 bar |
| Air entrainment | `oil_aeration_pct` | ≥ 6 % | ≥ 15 % |
| Wear metals (Fe) | `oil_iron_ppm` | ≥ 100 ppm | ≥ 300 ppm |

## Using it

CLI:

```bash
# Print the whole catalog (71 fault types, 12 subsystems)
python main.py faults --list

# One subsystem
python main.py faults --subsystem gearbox

# Detect faults from a telemetry snapshot (JSON file)
python main.py faults --snapshot examples/fault_payload.json --model NREL-5MW
```

API:

```bash
curl -X POST http://localhost:8080/api/faults/detect \
  -H 'Content-Type: application/json' \
  -d @examples/fault_payload.json

curl 'http://localhost:8080/api/faults/catalog?subsystem=generator'
```

Python:

```python
from src.digital_twin.specs import get_spec
from src.faults import FaultDetector

detector = FaultDetector(get_spec("NREL-5MW"))
report = detector.detect(
    {"vibration_mms": 6.8, "oil_viscosity_cst": 6.2, "oil_water_ppm": 850.0, ...},
    history=last_windows,          # optional: confirmation counts
    asset_id="WTG-014",
)
print(report.to_dict())
```

Digital twin: every `update_state` call runs detection automatically and
embeds `fault_report` in the returned state record; `twin.last_fault_report`
holds the latest report.

## Notes

* The canonical five SCADA channels are always evaluated. Optional
  condition-monitoring channels simply unlock more of the catalog.
* Faults that need eyes (lightning damage, structural cracks, corrosion,
  loose bolts) are detected through `inspection_*` flags.
* Every fault record is **advisory only** — it recommends actions but never
  sends commands to the turbine.
* Thresholds are research-grade defaults; tune them to your OEM limits before
  operations depend on them.
