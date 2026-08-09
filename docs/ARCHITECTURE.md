# AeroVigil architecture — how the modules connect

AeroVigil v1.0.0 · <https://aerovigil.abacusai.app> · **advisory / decision-support only**

This document maps every module in `wind-turbine-pg-bnn` and the data flow that
connects them: *config → data → physics → model/serving → predictor → safety →
reporting → api/ui/cli*, plus the digital-twin and meta/hermes paths.

## Single deployment boundary

`src/unified_app.py` is the canonical runtime. It binds the static eight-page
operator console, complete operations API, and low-level model API into one ASGI
process and one port. `/health` discovers the whole system, `/api/*` connects
advisory, fleet, twin, telemetry, jobs, and reporting, `/model-api/*` exposes
raw PG-BNN inference, and `/` serves the console. Mounted child lifespans are
explicitly managed by the parent, so model initialization is not skipped.

Former standalone apps remain import-compatible, but they are not deployments.
Legacy container service-mode names converge on the unified app, and the Gradio
script no longer opens a port. `docker compose up aerovigil` and `make serve`
both start the same boundary. Kubernetes deliberately runs one pod with a PVC
at `/app/data`: the twin registry is in memory and the operational/job stores
use SQLite WAL, so horizontal replicas would create divergent state. The
separate `docker/` training/federation stack is offline model-production
infrastructure, not an operator HTTP surface.

Rendered Mermaid versions of these diagrams — including the inference pipeline
sequence — live in [`DIAGRAMS.md`](DIAGRAMS.md).

---

## Connection diagram

```
                        configs/default.yaml
                                │
                    src/utils/config.py  (AppConfig; FAIL-CLOSED unless advisory_only)
                                │
      ┌───────────────┬─────────┼──────────────┬──────────────────────────┐
      ▼               ▼         ▼              ▼                          ▼
physics limits   45-d horizon  bnn/telemetry  meta.reptile + hermes    UI defaults
(constraints.py) (calibration) (ingest/bnn)   (meta/, agents/hermes)   (ui/defaults.py)
      │               │         │                    │
      │               │         ▼                    │
      │               │   src/data/synthetic.py      │
      │               │   src/data/ingest.py ────────┼──► robust_normalize + sliding_features
      │               │         │                    │        (window 60 / stride 20 / 5 stats)
      │               │         ▼                    │
      │               │   src/models/bnn.py ◄────────┘  MCVI train (elbo_loss) / predict
      │               │         │  TRAIN
      │               │         ▼
      │               │   src/utils/artifacts.py  (checkpoint + scaler + JSON sidecar ── artifacts/)
      │               │         │  SAVE / LOAD
      │               │         ▼
      │               │   src/models/serving.py  (load_serving_model: model + feature_fn)
      │               │         │          ▲
      │               │         │          │ export (meta / hermes bundles)
      ▼               ▼         ▼          │
src/models/predictor.py  run_advisory(payload, model?, feature_vector?)
      │   physics check_violations ──► physics limits above
      │   early-warning horizon ─────► EARLY_WARNING_HORIZON_DAYS (45d, config-sourced)
      │   RUL source: served model (model+features)  OR  payload bnn_state (fallback)
      ▼
src/utils/safety.py  enforce_safety_contract  ── screens EVERY outgoing payload
      │
      ├───────────────┬──────────────────┬───────────────────────────────┐
      ▼               ▼                  ▼                               ▼
 src/reporting/   src/api/app.py      src/cli.py (+cli_twin.py)      src/ui/app.py
 reports.py       FastAPI service     unified CLI: advisory / fleet / Streamlit UI
 (markdown/JSON)      │               report / twin status|simulate|prompt
                      ▼
   /health /advisory /advisory/fleet /twin/* /telemetry/* /fleet/report
```

### Digital twin path

```
src/digital_twin/specs.py (TurbineSpec library)
        │
src/digital_twin/twin.py  WindTurbineDigitalTwin
        ├── update_state(telemetry, bnn_state?)
        │      ├── physics violations + ISO 281 L10 + cumulative wear
        │      └── advisory bridge ──► run_advisory
        │             ├── serving model attached?  rolling buffer → features → model RUL
        │             └── else bnn_state block (previous behavior)
        ├── simulate_scenario(profile) ──► update_state per step
        │        (seeded RNG per asset → deterministic; hours validated ≤ 1 year)
        ├── runtime guards ── non-finite values rejected · history capped
        │        (max_history) · advisory failover bnn_state on model errors
        └── src/digital_twin/prompts.py generate_engineering_prompt (advisory-aware)
        │
   Consumers: API /twin/status · /twin/simulate · /twin/prompt (LRU-bounded
              registry, AV_TWIN_MAX_ASSETS), unified CLI `twin` group, UI tab
```

### Meta-learning / Hermes onboarding path

```
src/meta/tasks.py (AdaptationTask from telemetry) ─► src/meta/reptile.py (meta_train)
        │                                                    │
src/agents/hermes.py  ADAPT → SELF-TRAIN (σ ≤ τ pseudo-labels) → GATE (fail-closed)
        │                                                    │
        └── OnboardingReport + adapted model clone ──► src/utils/artifacts.py
              export_onboarding_bundle  →  artifacts/…/*.pt + *_report.json
              → load_serving_model → run_advisory  (deployment loop closed)
```

### AeroZip telemetry path

```
src/models/telemetry/aerozip.py (delta + deadband + quantize, anomaly bypass)
        │
src/models/telemetry/pipeline.py  compress_window / restore_window (surfaces anomaly_score)
        ├── src/data/ingest.py: write_aerozip_csv / load_aerozip_csv
        ├── API POST /telemetry/compress, POST /telemetry/restore
        └── UI "Telemetry (AeroZip)" metrics panel
```

---

## Module inventory

| Path | Role in the system | Consumed by |
| --- | --- | --- |
| `configs/default.yaml` + `src/utils/config.py` | single source of truth; fail-closed advisory-only loader | physics, eval, ingest windows, bnn, meta, hermes, UI, serving |
| `src/data/synthetic.py` | seeded synthetic fleet generator (labels = RUL days) | demos, pipeline, tests |
| `src/data/ingest.py` | CSV ingest, `robust_normalize`, `sliding_features` (60/20), AeroZip CSV path | training, serving, pipeline |
| `src/physics/constraints.py` | OEM-style hard limits (config-sourced), differentiable `physics_loss`, ISO 281 L10 | predictor, bnn training, twin |
| `src/models/bnn.py` | MCVI PG-BNN (`elbo_loss`, `predict` with epistemic/aleatoric split) | serving, meta, hermes, pipeline |
| `src/utils/artifacts.py` | artifact registry: checkpoint + scaler + JSON sidecar, onboarding export | serving, demos, hermes |
| `src/models/serving.py` | `load_serving_model` → model + feature function → model-based `run_advisory` | API, UI, CLI, twin, hermes export check |
| `src/models/telemetry/aerozip.py` | compressor core | pipeline |
| `src/models/telemetry/pipeline.py` | window ⇄ payload wrappers, anomaly score | ingest, API, UI |
| `src/models/predictor.py` | `run_advisory`: physics + failure-mode + rationale + 45-d early warning | everything |
| `src/eval/calibration.py` | ECE, utilization, **45-day early-warning metrics** (config-sourced horizon) | API fleet, eval scripts, predictor |
| `src/meta/reptile.py` + `src/meta/tasks.py` | Reptile meta-learning over fleet tasks | hermes, onboard demo |
| `src/agents/hermes.py` | self-training onboarding agent → `OnboardingReport` (safety-screened) | onboard demo, artifacts export |
| `src/utils/safety.py` | `enforce_safety_contract` — fail-closed key scanner on every boundary | predictor, API, CLI, UI, artifacts |
| `src/reporting/reports.py` | advisory records → markdown/JSON, `build_fleet_report` | CLI, UI, API `/fleet/report` |
| `src/digital_twin/` | spec library, hardened twin runtime (bounded history, deterministic sim, advisory failover), copilot prompts | API `/twin/*`, `twin` CLI group |
| `src/api/` | FastAPI surface (payload-based and model-serving advisory, twin, telemetry, reporting); LRU-bounded twin registry | deployment |
| `src/cli.py` + `src/cli_twin.py` | unified `wind-turbine-bnn` CLI with `twin status|simulate|prompt` group (standalone `twin-*` kept for compat) | operators |
| `src/ui/` | Streamlit advisory UI (single, fleet, digital twin, AeroZip) | operators |
| `scripts/run_pipeline.py` | config→data→train→eval→export→advisory smoke→`artifacts/pipeline_report.md` | release verification |
| `scripts/e2e_smoke.py` | in-process FastAPI e2e over every endpoint; non-zero exit on failure | release verification |

## Boundary / safety invariants

1. **Advisory-only, fail closed everywhere.** `run_advisory` output and every
   API/CLI/UI/report payload pass `enforce_safety_contract`; keys matching
   throttle/torque/pitch/rpm-setpoint/breaker/LOTO/part-number/actuation are
   rejected, by design, before leaving the system.
2. **Config gate.** `load_config` refuses any config that is not
   `safety.mode: advisory_only` with `allow_actuation: false`.
3. **Backward compatibility.** `POST /advisory` with a `bnn_state` block is
   byte-for-byte the pre-integration behavior; the model path only activates
   when a serving model is loaded AND the request carries a `telemetry_window`.
4. **Feature determinism.** Serving computes features with the *training*
   scaler stored in the bundle (never refit) and the canonical
   window/stride/stat order; a dedicated test asserts bit-level agreement
   with the training pipeline.
5. **Metadata `torch.load(weights_only=True)`.** Bundle checkpoints carry no
   arbitrary pickles for the model payload itself.
6. **One state owner.** The canonical deployment runs one application process.
   Kubernetes uses one pod and a persistent `/app/data` claim; do not add HPA or
   replicas until SQLite and the in-memory twin registry move to shared external
   services.
