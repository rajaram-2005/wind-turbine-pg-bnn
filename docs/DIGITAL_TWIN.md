# Wind Turbine Digital Twin (Advisory-Only)

This subpackage adds a virtual physical representation (Digital Twin) for any wind turbine in the specs library. It couples physics-based ISO 281 bearing life estimations and dynamic cumulative wear modeling with our Bayesian Neural Network (BNN) probabilistic Remaining Useful Life (RUL) predictions.

All operations and recommendations are strictly aligned with the advisory-only engineering guidelines documented in `docs/SAFETY.md`.

## Features

1. **Turbine Specifications Library (`src.digital_twin.specs`)**:
   Predefined structural, mechanical, and gearbox limitations for 8 common wind turbines:
   - **GE 1.5 SLE** (GE-1.5)
   - **Vestas V90-2.0 MW** (Vestas-V90)
   - **Siemens SWT-2.3-101** (Siemens-SWT-2.3)
   - **Suzlon S97 2.1 MW** (Suzlon-S97) — most-deployed turbine in Indian wind farms
   - **Gamesa G114-2.0 MW** (Gamesa-G114)
   - **Nordex N100/2500** (Nordex-N100)
   - **Senvion MM92** (Senvion-MM92)
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

The twin commands live in the **unified application CLI** under the `twin`
group, and remain available as standalone commands for backwards
compatibility:

```bash
# Unified application CLI (one tool for the whole application surface)
python -m src twin status    --asset-id WTG-042 --model Vestas-V90
python -m src twin simulate  --asset-id WTG-SIM-01 --model NREL-5MW --profile overload --hours 12
python -m src twin prompt    --asset-id WTG-099 --model GE-1.5

# Standalone entrypoints (same parsers, same flags)
twin-status    --asset-id WTG-042 --model Vestas-V90 --payload examples/payload.json
twin-simulate  --asset-id WTG-SIM-01 --model NREL-5MW --profile overload --hours 12 -o sim.json
twin-prompt    --asset-id WTG-099 --model GE-1.5
```

### `twin status`
Display the current status, specifications, and physical wear index of a turbine twin.
`--payload` ingests a telemetry snapshot first; `--advisory` prints the advisory
engine output (trained-model path with `--model-path`, else the `bnn_state` block).
`--format json` emits machine-readable output.

### `twin simulate`
Run physical state progression simulations over time under hypothetical scenario
profiles: `--profile {nominal,overload,derated,viscosity_loss}`, `--hours <hours>`,
and `-o <output.json>` to save history. Durations must be positive finite numbers
no larger than one year (8760 h); fractional hours round up to whole hourly steps.

### `twin prompt`
Compile and generate a safety-bounded context block and instruction prompt for
LLMs or copilot integrations.

---

## Runtime hardening

The twin runtime is hardened for long-running, unattended operation:

- **Deterministic simulation.** Scenario fluctuations use a per-asset seeded
  RNG (CRC32 of the asset id), so the same twin + profile + duration always
  reproduces the same trajectory — across processes and Python hash-seed
  settings. (Previously Python's process-randomized `hash()` made every run
  slightly different.)
- **Bounded memory.** Each twin retains at most `max_history` state records
  (default 10 000) and a 512-snapshot advisory feature buffer. The FastAPI
  twin registry is LRU-bounded too — `AV_TWIN_MAX_ASSETS` (default 1024)
  caps concurrent in-memory twins, evicting the least recently used asset
  first. `/health` on the unified app reports `assets_tracked` / `max_assets`.
- **Input validation.** Non-finite telemetry or BNN values (NaN/Inf) are
  rejected with a clear error instead of silently corrupting wear physics,
  and invalid simulation durations are refused.
- **Advisory failover.** If the attached serving model raises during a state
  update, the twin falls back to the `bnn_state` path and records
  `advisory_error` on the state record — state ingestion never dies with the
  model.
- **Safe seeding.** API twin creation seeds from the spec's nominal operating
  point; a failed seed returns 422 and never leaves a half-initialized twin
  in the registry.

---

## Connected Cyber Prime agent mesh

Every twin state now carries one shared `agent_team` brief produced by
`src.agents.cyber_team`. This prevents separate product surfaces from inventing
contradictory interpretations:

- **MIKA** translates RUL, uncertainty, and risk into maintenance-planning language.
- **KAI** explains physical violations, ISO 281 bearing evidence, telemetry, and wear.
- The same brief flows through twin history, `/advisory`, `/advisory/fleet`,
  `/twin/status`, `/twin/simulate`, engineering prompts, CLI output, prediction
  cards, fleet cards, and the Cyber Twin HUD.
- `connected_sources` records which evidence points contributed. The agreement
  score is a coordination indicator, not a calibrated probability.

The Cyber Twin command center also provides:

- **Component Resonance Scan** — scenario-relative health indicators for the
  rotor, main bearing, gearbox, generator, and converter.
- **Scenario Lab** — synchronized comparison of multiple operating futures,
  including RUL runway, wear, stress, uncertainty, and illustrative energy.
- **Agent Copilot** — evidence-grounded MIKA/KAI answers for maintenance and
  physics questions; this is deterministic synthesis, not an unbounded LLM.
- **Human decision gate** — acknowledge, request engineering review, or
  escalate the evidence without sending control commands to the turbine.

```text
SCADA → PG-BNN → ISO 281 / constraints → Digital Twin wear → Fleet priority → Human review
                  ↘ MIKA + KAI shared advisory brief ↗
```

## Safety & Non-Actuation Contract
As with all modules in `wind-turbine-pg-bnn`, the digital twin does not command or dictate physical systems:
- No manual actuator, throttle, or LOTO procedures are suggested.
- All bearing calculations ($L_{10}$), wear indices, and prompt directives serve strictly as informational decision-support for reliability planners and engineers.
