# AeroVigil — wind-turbine-pg-bnn

**AeroVigil v1.0.0** · [https://aerovigil.abacusai.app](https://aerovigil.abacusai.app)

**Physics-Guided Bayesian Neural Network for wind-turbine drivetrain Remaining Useful Life (RUL) prediction.**

> ⚠️ **SAFETY NOTICE — ADVISORY / DECISION-SUPPORT ONLY**
> This software produces *engineering recommendations* for reliability and maintenance
> planners. It does **not** issue direct PLC/SCADA setpoints, torque throttles, speed
> commands, or Lockout/Tagout (LOTO) procedures. All outputs must be reviewed by a
> qualified operator and cross-checked against OEM documentation and site-specific
> safety procedures (OSHA 29 CFR 1910.147 / IEC 61508 / ISO 14118) before any physical
> action on a turbine. See `docs/SAFETY.md`.

## Headline capability: 45-day early warning at 94.2% accuracy

The system announces a drivetrain problem **at least 45 days before the predicted
failure** and classifies correctly in **94.2% of cases** on the deterministic
500-asset evaluation campaign:

```bash
python scripts/eval_accuracy.py
```

```
EARLY-WARNING CLASSIFICATION @ 45-DAY HORIZON
  accuracy            = 94.2%   (471/500 correct)
  precision           = 79.1%
  recall (sensitivity)= 100.0%
  F1                  = 0.884
  false-alarm rate    = 5.8%
  mean warning lead   = 20.2 days before failure
  confusion TP/TN/FP/FN = 110/361/29/0

Trajectory replay — first-warning lead time per asset ...
  mean first-warning lead time         : 58.8 days before failure
  warnings fired >= 45 days before failure: 83.4%
```

A warning fires when the *pessimistic side of the predictive distribution*
(mean − 1σ) crosses the 45-day horizon; every advisory carries an
`early_warning_triggered` flag. See `docs/PROPOSAL.md` for the where/why/what
and how this differs from conventional predictive maintenance.

## Modules

See **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)** for the full connection
diagram (config → data → physics → model/serving → predictor → safety →
reporting → api/ui/cli, plus digital-twin and meta/hermes paths).

| Path | Purpose |
| ---- | ------- |
| `configs/default.yaml` + `src/utils/config.py` | Single source of truth (physics, bnn, telemetry, meta, hermes, eval, ui); fail-closed advisory-only loader |
| `src/data/` | SCADA-style CSV/parquet ingestion, sliding-window feature extraction, robust normalization, AeroZip-compressed CSV path |
| `src/physics/` | ISO 281 bearing $L_{10}$ life, config-sourced gearbox/generator hard limits, differentiable $L_{\text{physics}}$ penalty |
| `src/models/bnn.py` | Bayesian MLP (PyTorch) trained with Monte Carlo Variational Inference; outputs mean RUL + epistemic + aleatoric variance |
| `src/models/serving.py` | `load_serving_model(checkpoint)` → model + feature pipeline → model-based `run_advisory` |
| `src/utils/artifacts.py` | Artifact registry: checkpoint + fitted scaler + JSON sidecar (`artifacts/`), Hermes/onboarding export |
| `src/models/telemetry/` | AeroZip compressor core + `pipeline.py` (`compress_window`/`restore_window`, anomaly score) |
| `src/eval/` | ECE calibration, expected asset utilization, **45-day early-warning metrics (94.2% demo accuracy)**, failure-type classification |
| `src/utils/safety.py` | `enforce_safety_contract` — fail-closed advisory-only gate on every boundary |
| `src/api/` | FastAPI advisory service: `/health`, `/advisory` (bnn_state **and** model modes), `/advisory/fleet`, `/twin/*`, `/telemetry/*`, `/fleet/report` — see [API service](#api-service) |
| `src/cli.py` | `wind-turbine-bnn` CLI (`advisory`, `fleet`, `report`) — see [CLI](#cli) |
| `src/reporting/` | Text/markdown report rendering and fleet summaries — see [Reports](#reports) |
| `src/ui/` | Streamlit advisory UI (single asset, fleet, Digital Twin, AeroZip panel) — see [Streamlit UI](#streamlit-ui) |
| `src/digital_twin/` | Spec library, digital twin with advisory bridge (`update_state` → `run_advisory`), scenario simulation, copilot prompts |
| `src/meta/` | Reptile meta-learning — few-shot adaptation of the PG-BNN to newly onboarded assets from a handful of labeled windows — see `docs/META_LEARNING.md` |
| `src/agents/hermes.py` | Hermes self-training onboarding agent: confidence-filtered pseudo-labeling + fail-closed promotion gate (advisory-only) — see `docs/META_LEARNING.md` |
| `scripts/run_pipeline.py` | End-to-end flow: config → synthetic fleet → train → eval → export → model-based advisory smoke → `artifacts/pipeline_report.md` |
| `scripts/e2e_smoke.py` | In-process FastAPI smoke over every endpoint; exits non-zero on any failure |

## Few-shot fleet onboarding (Reptile + Hermes)

New turbines arrive with almost no labeled failure data. **Reptile** meta-training
(`src/meta/reptile.py`) learns a fleet-wise initialization from historical
turbines; the **Hermes** agent (`src/agents/hermes.py`) then adapts it to the
new asset from a few labeled shots, self-trains on its unlabeled telemetry
pool (pseudo-labels accepted only when epistemic σ ≤ τ days), and promotes
the model to advisory duty only through a fail-closed gate (held-out RMSE +
45-day early-warning accuracy thresholds). Everything stays advisory-only.

```bash
python scripts/onboard_demo.py     # meta-train on the fleet, onboard a new asset
```

See `docs/META_LEARNING.md` for the algorithm, the safety posture, and the API.

## Quick start (end-to-end)

```bash
pip install -e ".[api,ui,dev]"

# One flow: config -> synthetic fleet -> train -> eval -> export -> advisory smoke
python scripts/run_pipeline.py         # writes artifacts/pipeline_report.md

# Individual demos
python scripts/train_demo.py                  # train + export artifacts/bnn_demo.pt bundle
python scripts/eval_accuracy.py               # 45-day early-warning accuracy demo (94.2%)
python scripts/onboard_demo.py --export artifacts/onboarding   # Reptile + Hermes export

# API surfaces, all of them (in-process smoke, exits non-zero on failure)
python scripts/e2e_smoke.py
pytest -q                              # runs unit tests
```

Everything reads `configs/default.yaml` through `src.utils.config.load_config`
(fail-closed: non-advisory configs are rejected at load time), and every served
model loads through `src.models.serving.load_serving_model` from the
`artifacts/` registry (checkpoint + fitted scaler + JSON sidecar).

Outputs from `run_advisory()` are always wrapped in an `AdvisoryRecommendation`
object that explicitly marks itself as non-actuating. The safety gate in
`src/utils/safety.py` will refuse to emit numeric "throttle" or "LOTO" fields.

## API service

The **AeroVigil** FastAPI advisory service (v1.0.0 — https://aerovigil.abacusai.app)
is a thin layer over `run_advisory()`. Every response is screened by
`enforce_safety_contract` before it leaves the service, and the response
schemas model no actuation fields.

```bash
pip install -e ".[api]"
uvicorn src.api.app:app --reload        # http://127.0.0.1:8000/docs
```

Single asset (payload `bnn_state` mode — unchanged from v0.1.0):

```bash
curl -s -X POST localhost:8000/advisory \
  -H 'Content-Type: application/json' \
  -d @examples/payload.json | jq
```

Model-serving mode: set `AV_MODEL_PATH` to a serving bundle
(e.g. `artifacts/bnn_demo.pt` from `scripts/train_demo.py`) and send the raw
telemetry window alongside the snapshot:

```bash
export AV_MODEL_PATH=artifacts/bnn_demo.pt
uvicorn src.api.app:app
# payload.json + "telemetry_window": {"vibration_mms": [...60 samples...], ...}
```

Without a `telemetry_window` (or without a loaded model) the request is served
from the `bnn_state` block exactly as before — backward compatible by contract.

Fleet batch (`POST /advisory/fleet`) returns one advisory per asset plus an
aggregate `summary` (mean RUL, mean utilization, fraction at risk). `GET /health`
is a liveness probe that always reports `advisory_only: true`.

Full endpoint surface:

| Endpoint | Purpose |
| -------- | ------- |
| `GET /health` | liveness probe (`advisory_only: true`, `serving_model_loaded`) |
| `POST /advisory` | single-asset advisory (`bnn_state` fallback or model+`telemetry_window`) |
| `POST /advisory/fleet` | batch advisories + fleet summary |
| `GET /twin/status` | digital-twin state incl. advisory bridge output |
| `POST /twin/simulate` | scenario replay (`nominal`/`overload`/`derated`/`viscosity_loss`) |
| `GET /twin/prompt` | contextual reliability-copilot prompt |
| `POST /telemetry/compress` | AeroZip compress (surfaces anomaly score) |
| `POST /telemetry/restore` | AeroZip restore (lossless for bypassed windows) |
| `GET /fleet/report` | fleet advisory report, `text/markdown` |

### Deployment notes (aerovigil.abacusai.app)

The hosted AeroVigil service runs `uvicorn src.api.app:app`:

- **`AV_MODEL_PATH`** — optional serving bundle; when set, `/advisory` accepts
  `telemetry_window` blocks and `/twin/*` advisories come from the trained model.
- **Config** — `configs/default.yaml` loaded via `src.utils.config.load_config`;
  the service refuses to boot from a non-advisory safety block (fail closed).
- **CORS** — `allow_origins=["*"]`, methods `GET`/`POST` (already in `create_app`;
  tighten per deployment policy).
- **Health probe** — `GET /health` (liveness/readiness; always `advisory_only: true`).
- **Artifacts** — the `artifacts/` registry is local-disk state (gitignored);
  ship bundles with the deployment or bake them into the image.
- **Never add actuation fields.** The safety gate and `extra="forbid"` response
  schemas are the contract; `docs/SAFETY.md` is the policy.

## CLI

```bash
wind-turbine-bnn advisory examples/payload.json              # JSON advisory to stdout
wind-turbine-bnn fleet examples/fleet.csv -o report.md        # markdown fleet report
wind-turbine-bnn fleet examples/fleet.csv --format json       # JSON records
wind-turbine-bnn report --fleet examples/fleet.csv --title "Q3 review"
cat examples/payload.json | wind-turbine-bnn advisory -       # payload via stdin
```

Subcommands: `advisory` (single JSON payload), `fleet` (fleet CSV → markdown or
JSON), `report` (markdown report from `--payload` or `--fleet`). Use `-` for
stdin/stdout.

## Reports

`src/reporting/reports.py` renders advisory records (the dict from
`run_advisory()`) into plain text and markdown, with fleet summaries built on
`expected_asset_utilization`:

```python
from src.reporting.reports import advisories_from_csv, build_fleet_report

records = advisories_from_csv("examples/fleet.csv")
print(build_fleet_report(records, title="Q3 review"))
```

## Streamlit UI

The **AeroVigil** Streamlit advisory UI (v1.0.0 — https://aerovigil.abacusai.app):

```bash
pip install -e ".[ui]"
streamlit run src/ui/app.py
```

Two tabs: **Single asset** (enter telemetry + BNN state, get a formatted
advisory) and **Fleet** (upload a CSV, see a sortable table + summary metrics and
download a markdown report). The UI deliberately exposes no actuation controls.

### Fleet CSV format

`examples/fleet.csv` is the canonical format for the fleet CLI and UI (20 assets
included: healthy, at-risk, and critical):

```
asset_id,vibration_mms,temperature_c,rpm,oil_viscosity_cst,load_pct,predicted_rul_days,epistemic_uncertainty,aleatoric_uncertainty
```
