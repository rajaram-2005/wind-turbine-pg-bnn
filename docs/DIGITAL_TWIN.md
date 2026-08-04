# Wind Turbine Digital Twin (Advisory-Only)

This subpackage adds a virtual physical representation (Digital Twin) for any wind turbine in the specs library. It couples physics-based ISO 281 bearing life estimations and dynamic cumulative wear modeling with our Bayesian Neural Network (BNN) probabilistic Remaining Useful Life (RUL) predictions.

All operations and recommendations are strictly aligned with the advisory-only engineering guidelines documented in `docs/SAFETY.md`.

## Features

1. **Turbine Specifications Library (`src.digital_twin.specs`)**:
   Predefined structural, mechanical, and gearbox limitations for common wind turbines:
   - **GE 1.5 SLE** (GE-1.5)
   - **Vestas V90-2.0 MW** (Vestas-V90)
   - **Siemens SWT-2.3-101** (Siemens-SWT-2.3)
   - **NREL 5MW Reference Turbine** (NREL-5MW)

2. **WindTurbineDigitalTwin (`src.digital_twin.twin`)**:
   Tracks the physical representation of a specific turbine.
   - Calculates **ISO 281 Bearing $L_{10}$ Life** under actual generator load and RPM.
   - Computes **Cumulative Physical Wear Index** based on vibration, temperature, load, speed, and active physical violations.
   - Simulates operational scenarios under different profiles (**nominal**, **overload**, **derated**, **viscosity_loss**).

3. **Copilot / LLM Prompt Context (`src.digital_twin.prompts`)**:
   Generates comprehensive, contextual prompt templates for AI reliability advisors. It structures all metrics, design specs, active violations, and BNN RUL outputs into an advisory-only prompt block that strictly prevents actuation recommendations.

---

## Command Line Interface (CLI)

The package installs three commands under your environment:

### 1. `twin-status`
Display the current status, specifications, and physical wear index of a turbine twin:
```bash
twin-status --asset-id WTG-042 --model Vestas-V90
```
To update the twin with an actual telemetry payload:
```bash
twin-status --asset-id WTG-042 --model Vestas-V90 --payload examples/payload.json
```

### 2. `twin-simulate`
Run physical state progression simulations over time under hypothetical scenario profiles:
```bash
# Simulate overload operations on NREL 5MW for 12 hours
twin-simulate --asset-id WTG-SIM-01 --model NREL-5MW --profile overload --hours 12
```
Options: `--profile {nominal,overload,derated,viscosity_loss}`, `--hours <hours>`, and `-o <output.json>` to save history.

### 3. `twin-prompt`
Compile and generate a safety-bounded context block and instruction prompt for LLMs or copilot integrations:
```bash
twin-prompt --asset-id WTG-099 --model GE-1.5
```

---

## Safety & Non-Actuation Contract
As with all modules in `wind-turbine-pg-bnn`, the digital twin does not command or dictate physical systems:
- No manual actuator, throttle, or LOTO procedures are suggested.
- All bearing calculations ($L_{10}$), wear indices, and prompt directives serve strictly as informational decision-support for reliability planners and engineers.
