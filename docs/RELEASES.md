# 🚀 AeroVigil Release History & 3-Year Enterprise LTS Roadmap

This document outlines the official release history of the **AeroVigil** physics-guided AI predictive maintenance platform, from the initial prototype release (`v0.1.0`) to the production 3-Year Enterprise Long-Term Support release (`v1.0.0 LTS`), along with future release milestones.

---

## 📅 Release Timeline Overview

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               3-YEAR ENTERPRISE LTS WINDOW                             │
│                                                                                        │
│   2026-08 (v0.1.0)           2026-08 (v1.0.0 LTS)                       2029-08 (v2.0.0 LTS)  │
│   Initial Core Engine        Production LTS Launch                      Next Scheduled Update │
│   ───────────────────        ─────────────────────                      ───────────────────── │
│   • PG-BNN Architecture      • 12 Subsystems & 80 Fault Modes          • Next-Gen Physics     │
│   • ISO 281 Physics Loss     • Multi-Platform Native Flutter Apps      • Multi-Farm P2P Mesh  │
│   • MLOps Pipeline           • Edge IoT (ESP32/STM32 Firmware)         • Autonomous Workflows │
│   • Gradio Space Demo        • MIKA + KAI Dual Copilot Agents          • 3-Year LTS Refresh   │
│                              • 3-Year Zero-Breaking Guarantee                                 │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Releases

### [v1.0.0 LTS] — 2026-08-17 (Current Production LTS)

> **Enterprise LTS Commitment**: Active Support through **August 17, 2029**. No mandatory breaking changes for 3 years. See [`docs/LTS_POLICY.md`](LTS_POLICY.md).

#### 🌟 Major Highlights & Architecture

1. **Whole-Turbine Subsystem Diagnostics (12 Subsystems, 80 Fault Types)**:
   - Complete coverage: Rotor & Blades, Pitch, Hub & Main Shaft, Gearbox, High-Speed Shaft & Brake, Generator, Yaw, Tower & Foundation, Nacelle & Sensors, Cooling & Hydraulics, Electrical & Power, SCADA & Communication.
   - Comprehensive gearbox **oil condition monitoring**: Viscosity, water PPM, ISO 4406 particulate cleanliness, Total Acid Number (TAN), filter differential pressure ($\Delta P$), aeration/foaming, supply pressure, and elemental wear metals (Fe, Cu).
   - Whole-turbine **fire fault detection**: 7 distinct fire risk categories including gearbox oil smoke, brake friction overheat, electrical cabinet arcing, tower-base fires, and fire suppression discharge anomalies.

2. **Physics-Guided AI Framework (v2 Extension)**:
   - **Aerodynamics Loss**: Incorporates Heier power coefficient $C_p(\lambda, \beta)$, theoretical Betz limit enforcement ($16/27$), and Jensen analytical wake deficit.
   - **Drivetrain Dynamics**: ISO 281 $L_{10}$ bearing fatigue life calculations, gear torque transfer conservation, and irreversible cumulative wear regularisation.
   - **Thermal Lumped-Parameter Network**: Generator and bearing heat dissipation ODEs, stator insulation limits, and thermal steady-state constraints.
   - **PINO (Physics-Informed Neural Operator)**: Fourier Neural Operator (FNO) with 2D Navier-Stokes residual modeling for wake field prediction.
   - **Uncertainty & Active Learning**: Monte Carlo dropout epistemic uncertainty sampling generating automated maintenance alerts.
   - **Physics-Ground SHAP**: Transparent explainability mapping feature attributions directly to physical root causes.
   - **Federated Fleet Learning**: Privacy-preserving Flower `NumPyClient` for multi-farm collaborative training.

3. **Multi-Platform Native Consoles & Unified Web Dashboard**:
   - **Native Desktop & Mobile Apps**: Flutter client targeting Windows x64, macOS Universal, Linux x64, and Android APK.
   - **Connected Browser Console**: Single-port high-performance operator console with live SVG digital twin carbon replica, interactive scenario lab, and human decision gate.
   - **MIKA & KAI Copilot Mesh**: Integrated mechanical kinematics advisory (MIKA) and telemetry anomaly inference (KAI) copilot agents.

4. **Edge Hardware & Industrial IoT Connectivity**:
   - ESP32 and STM32 embedded C/C++ firmware prototypes.
   - Device simulation scripts (`edge/simulate_device.py`) for cloud-microcontroller telemetry validation.
   - Store-and-forward local SQLite buffering for remote offshore and intermittent network environments.

5. **Durable Enterprise Automation & Notifications**:
   - Asynchronous job queue worker with SQLite persistence.
   - Multi-channel notification pipeline: transactional email alerts, scheduled daily fleet health digests, and webhook integrations (Slack, Microsoft Teams, generic HTTP).
   - On-demand CSV and HTML fleet health report generators.

---

### [v0.1.0] — 2026-08-06 (Initial Release)

The initial foundation and open-source release of the AeroVigil PG-BNN package.

#### 🌟 Key Deliverables

- **Physics-Guided Bayesian Neural Network (PG-BNN)**:
  - Variational Bayes-by-Backprop implementation in PyTorch.
  - Decomposition of uncertainty into epistemic (model epistemic variance) and aleatoric (heteroscedastic noise) components.
  - Soft regularisation enforcing monotonic wear and ISO 281 bearing life bounds.
- **Python Packaging & CLI**:
  - `aerovigil-pg-bnn` PyPI package with `aerovigil-infer` CLI command.
  - Standalone FastAPI inference server (`src/aerovigil_pg_bnn/api.py`).
- **MLOps & Infrastructure**:
  - Multi-stage `Dockerfile` and local `docker-compose.yml`.
  - Kubernetes production deployment manifests (`k8s/`).
  - Pre-trained demonstration model weights and synthetic SCADA benchmark datasets.
- **Advisory Safety Contract**:
  - Enforced decision-support boundary forbidding automated mechanical actuation without human review.

---

## 🔮 3-Year Release Cadence & Future Milestones

| Target Date | Version / Milestone | Scope & Focus |
| :--- | :--- | :--- |
| **August 2026** | **v1.0.0 LTS** | Initial 3-Year Enterprise LTS baseline (Active Support through August 2029). |
| **2026 – 2029** | **v1.0.x Patches** | Non-breaking security fixes, dependency updates, new OEM turbine profiles. |
| **August 2028** | **v2.0.0 Preview** | 12-month advance notice, RFCs, and API migration guidelines for next LTS. |
| **February 2029**| **v2.0.0-rc1** | Public release candidate and staging migration sandbox. |
| **August 2029** | **v2.0.0 LTS** | Next 3-Year Enterprise LTS release (Multi-farm P2P mesh, autonomous scheduling). |
| **August 2030** | **v1.0.0 EOL** | Conclusion of extended security overlap support for v1.0.0. |

---

*For detailed enterprise support policies, see [`docs/LTS_POLICY.md`](LTS_POLICY.md).*
