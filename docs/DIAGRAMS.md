# AeroVigil diagrams — architecture & inference

Rendered [Mermaid](https://mermaid.js.org/) companions to the ASCII maps in
[`ARCHITECTURE.md`](ARCHITECTURE.md). GitHub renders these inline.

AeroVigil v1.0.0 · **advisory / decision-support only**

---

## System architecture

```mermaid
flowchart TD
    CFG["configs/default.yaml<br/>AppConfig · fail-closed unless advisory_only"]

    subgraph TRAIN["Training path"]
        SYN["src/data/synthetic.py<br/>seeded synthetic fleet"]
        ING["src/data/ingest.py<br/>robust_normalize + sliding_features<br/>(window 60 / stride 20 / 5 stats)"]
        BNN["src/models/bnn.py<br/>MCVI PG-BNN · elbo_loss"]
        ART["src/utils/artifacts.py<br/>checkpoint + scaler + JSON sidecar"]
        SYN --> ING --> BNN --> ART
    end

    subgraph SERVE["Serving / inference path"]
        SRV["src/models/serving.py<br/>load_serving_model: model + feature_fn"]
        PRED["src/models/predictor.py<br/>run_advisory"]
        EWC["45-day early-warning horizon<br/>(config-sourced)"]
        SRV --> PRED
        EWC --> PRED
    end

    subgraph PHY["Physics"]
        CONS["src/physics/constraints.py<br/>OEM hard limits · physics_loss · ISO 281 L10"]
    end

    subgraph SAFE["Safety boundary"]
        CONTRACT["src/utils/safety.py<br/>enforce_safety_contract<br/>screens EVERY outgoing payload"]
    end

    subgraph OUT["Delivery boundaries"]
        API["src/api/app.py<br/>/health /advisory /advisory/fleet<br/>/twin/* /telemetry/* /fleet/report"]
        CLI["src/cli.py · src/cli_twin.py<br/>wind-turbine-bnn · twin status|simulate|prompt"]
        UI["src/ui/app.py<br/>Streamlit advisory UI<br/>(no actuation controls)"]
        RPT["src/reporting/reports.py<br/>markdown / JSON fleet reports"]
    end

    CFG --> CONS
    CFG --> EWC
    CFG --> ING
    CFG --> META["src/meta/reptile.py +<br/>src/agents/hermes.py"]
    CFG --> UID["src/ui/defaults.py"]

    ART --> SRV
    META -.->|export onboarding bundle| ART
    CONS --> PRED
    PRED --> CONTRACT
    CONTRACT --> API
    CONTRACT --> CLI
    CONTRACT --> UI
    CONTRACT --> RPT
```

## Inference pipeline — model-serving advisory

`POST /advisory` with a `telemetry_window` while a serving bundle is loaded:

```mermaid
sequenceDiagram
    autonumber
    participant C as Client / UI / CLI
    participant API as FastAPI /advisory
    participant FEAT as feature_fn (training scaler,<br/>window 60 / stride 20)
    participant BNN as PG-BNN (MCVI predict)
    participant PHY as physics check_violations
    participant SAFE as enforce_safety_contract

    C->>API: payload + telemetry_window
    API->>FEAT: raw window → canonical features
    Note over FEAT: training scaler reused — never refit<br/>bit-level parity with training pipeline
    FEAT-->>API: feature vector
    API->>BNN: stochastic forward passes
    BNN-->>API: RUL mean · epistemic/aleatoric σ
    API->>PHY: limits + ISO 281 L10 cross-check
    PHY-->>API: violations · failure-mode rationale
    API->>API: 45-day early-warning flag (config horizon)
    API->>SAFE: screen assembled advisory payload
    SAFE-->>API: pass (fail-closed on actuation keys)
    API-->>C: advisory JSON (decision-support only)
```

## Inference fallback — payload-based advisory (backward compatible)

`POST /advisory` with a `bnn_state` block is byte-for-byte the pre-integration
behavior; the model path activates only when a serving model is loaded **and**
the request carries a `telemetry_window`.

```mermaid
flowchart LR
    REQ["POST /advisory"] --> Q{serving model loaded<br/>AND telemetry_window?}
    Q -->|yes| MP[model path:<br/>features → BNN RUL]
    Q -->|no| BP[payload path:<br/>bnn_state statistics used as-is]
    MP --> ADV[run_advisory core:<br/>physics + rationale + 45-d warning]
    BP --> ADV
    ADV --> SAFE[enforce_safety_contract]
    SAFE --> RESP[advisory response]
```

## Digital twin loop

```mermaid
flowchart TD
    SPEC["src/digital_twin/specs.py<br/>TurbineSpec library"] --> TWIN["WindTurbineDigitalTwin"]
    TEL["telemetry (+ optional bnn_state)"] --> TWIN
    TWIN --> UPD["update_state"]
    UPD --> ISO["physics violations + ISO 281 L10<br/>+ cumulative wear"]
    UPD --> BRIDGE{"advisory bridge"}
    BRIDGE -->|serving model attached| RB["rolling buffer → features → model RUL"]
    BRIDGE -->|else| BS["bnn_state block (previous behavior)"]
    RB --> RA["run_advisory"]
    BS --> RA
    TWIN --> SIM["simulate_scenario(profile)<br/>→ update_state per step"]
    TWIN --> PRM["generate_engineering_prompt<br/>(advisory-aware)"]
    RA --> SAFE2["safety-screened twin status"]
    SAFE2 --> CONS["API /twin/status · /twin/simulate · /twin/prompt<br/>twin-* CLIs · UI Digital Twin tab"]
```

## Meta-learning / Hermes onboarding

```mermaid
flowchart LR
    TASKS["src/meta/tasks.py<br/>AdaptationTask from telemetry"] --> REPTILE["src/meta/reptile.py<br/>meta_train"]
    REPTILE --> HERMES["src/agents/hermes.py"]
    subgraph HERMESFLOW["Hermes gates (fail-closed)"]
        ADAPT["ADAPT"] --> SELF["SELF-TRAIN<br/>σ ≤ τ pseudo-labels"] --> GATE["GATE"]
    end
    HERMES --> HERMESFLOW
    GATE --> REPORT["OnboardingReport + adapted clone"]
    REPORT --> EXPORT["export_onboarding_bundle<br/>artifacts/…/*.pt + *_report.json"]
    EXPORT --> SRV2["load_serving_model → run_advisory<br/>(deployment loop closed)"]
```

## AeroZip telemetry path

```mermaid
flowchart LR
    WIN["telemetry window"] --> AZ["src/models/telemetry/aerozip.py<br/>delta + deadband + quantize<br/>(anomaly bypass)"]
    AZ --> PIPE["src/models/telemetry/pipeline.py<br/>compress_window / restore_window<br/>surfaces anomaly_score"]
    PIPE --> CSV["src/data/ingest.py<br/>write/load_aerozip_csv"]
    PIPE --> APIA["API POST /telemetry/compress<br/>POST /telemetry/restore"]
    PIPE --> UIA["UI Telemetry (AeroZip)<br/>metrics panel"]
```

---

## Safety invariants shown by every diagram

1. **Advisory-only, fail closed.** Every payload crossing a boundary
   (`predictor → API/CLI/UI/report`) passes `enforce_safety_contract`; keys
   matching throttle/torque/pitch/rpm-setpoint/breaker/LOTO/part-number/
   actuation are rejected before leaving the system.
2. **Config gate.** `load_config` refuses anything that is not
   `safety.mode: advisory_only` with `allow_actuation: false`.
3. **Determinism.** Serving features use the training scaler (never refit) and
   the canonical window/stride/stat order.
4. **No arbitrary pickles.** Bundle metadata loads with
   `torch.load(weights_only=True)`.
