"""Whole-turbine fault detection: taxonomy, oil analysis and detector.

This package finds faults in **every part** of a wind turbine from
SCADA/condition-monitoring telemetry:

* :mod:`src.faults.taxonomy` — the complete fault catalog: 12 subsystems
  (rotor & blades, pitch, hub & main shaft, gearbox, HSS & brake, generator,
  yaw, tower & foundation, nacelle & sensors, cooling & hydraulics,
  electrical & power, SCADA & communication) with all their fault types,
  symptoms, root causes and recommended actions.
* :mod:`src.faults.oil` — gearbox oil-condition analysis (viscosity, water,
  ISO 4406 particles, TAN, filter ΔP, level, aeration, pressure ...) with a
  0–100 oil health score.
* :mod:`src.faults.detector` — :class:`FaultDetector`, the rule engine that
  turns a telemetry snapshot into a ranked :class:`FaultReport` with
  confidence, evidence and recommended actions per detected fault.
"""

from src.faults.detector import (
    DetectedFault,
    FaultDetector,
    FaultReport,
    all_fault_ids,
    covered_fault_ids,
    subsystem_labels,
    uncovered_fault_ids,
)
from src.faults.oil import OilAnalysis, OilFinding, analyze_oil, oil_analysis_from_telemetry
from src.faults.taxonomy import (
    FAULT_CATALOG,
    SEVERITIES,
    SUBSYSTEMS,
    FaultDefinition,
    catalog_summary,
    faults_by_subsystem,
    get_fault,
    list_faults,
)

__all__ = [
    "FAULT_CATALOG",
    "SEVERITIES",
    "SUBSYSTEMS",
    "DetectedFault",
    "FaultDefinition",
    "FaultDetector",
    "FaultReport",
    "OilAnalysis",
    "OilFinding",
    "all_fault_ids",
    "analyze_oil",
    "catalog_summary",
    "covered_fault_ids",
    "faults_by_subsystem",
    "get_fault",
    "list_faults",
    "oil_analysis_from_telemetry",
    "subsystem_labels",
    "uncovered_fault_ids",
]
