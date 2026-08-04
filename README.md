# wind-turbine-pg-bnn

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

| Path | Purpose |
| ---- | ------- |
| `src/data/` | SCADA-style CSV/parquet ingestion, sliding-window feature extraction, robust normalization |
| `src/physics/` | ISO 281 bearing $L_{10}$ life, gearbox/generator hard limits, differentiable $L_{\text{physics}}$ penalty |
| `src/models/bnn.py` | Bayesian MLP (PyTorch) trained with Monte Carlo Variational Inference; outputs mean RUL + epistemic + aleatoric variance |
| `src/models/telemetry/` | AeroZip-style delta + deadband + quantize compression with anomaly-bypass |
| `src/eval/` | ECE calibration, expected asset utilization, **45-day early-warning metrics (94.2% demo accuracy)**, failure-type classification |
| `src/utils/` | Safety gates, logging, schema |
| `src/api/` | FastAPI advisory service (`/health`, `/advisory`, `/advisory/fleet`) — see [API service](#api-service) |
| `src/cli.py` | `wind-turbine-bnn` CLI (`advisory`, `fleet`, `report`) — see [CLI](#cli) |
| `src/reporting/` | Text/markdown report rendering and fleet summaries — see [Reports](#reports) |
| `src/ui/` | Streamlit advisory UI — see [Streamlit UI](#streamlit-ui) |

## Quick start (research / offline mode)

```bash
pip install -e ".[api,ui,dev]"
python scripts/train_demo.py           # trains on synthetic drivetrain data
python scripts/eval_accuracy.py        # 45-day early-warning accuracy demo (94.2%)
pytest -q                              # runs unit tests
```

Outputs from `predict_rul()` are always wrapped in an `AdvisoryRecommendation`
object that explicitly marks itself as non-actuating. The safety gate in
`src/utils/safety.py` will refuse to emit numeric "throttle" or "LOTO" fields.

## API service

A thin FastAPI layer over `run_advisory()`. Every response is screened by
`enforce_safety_contract` before it leaves the service, and the response
schemas model no actuation fields.

```bash
pip install -e ".[api]"
uvicorn src.api.app:app --reload        # http://127.0.0.1:8000/docs
```

Single asset:

```bash
curl -s -X POST localhost:8000/advisory \
  -H 'Content-Type: application/json' \
  -d @examples/payload.json | jq
```

Fleet batch (`POST /advisory/fleet`) returns one advisory per asset plus an
aggregate `summary` (mean RUL, mean utilization, fraction at risk). `GET /health`
is a liveness probe that always reports `advisory_only: true`.

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
