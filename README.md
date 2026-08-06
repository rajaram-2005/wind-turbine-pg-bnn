---
tags:
- predictive-maintenance
- wind-energy
- bayesian-neural-network
- scada-telemetry
- aerovigil-ai
- pytorch
- physics-informed
- remaining-useful-life
- rul-prediction
- drivetrain-diagnostics
license: mit
library_name: pytorch
pipeline_tag: other
---

# AeroVigil: a check-engine light for wind turbines

A Physics-Guided Bayesian Neural Network (PG-BNN) that predicts — about **45
days early** — when a turbine's main bearing is running out of life, with an
honest estimate of how certain it is.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-ffd21e.svg)](https://huggingface.co/AerovigilAI/wind-turbine-pg-bnn)
[![Live Demo](https://img.shields.io/badge/demo-live-22c55e.svg)](https://aerovigil.abacusai.app)

![AeroVigil social card](https://raw.githubusercontent.com/rajaram-2005/wind-turbine-pg-bnn/arena%2F019fd767-wind-turbine-pg-bnn/docs/assets/social-card.png)

## What is this? (30-second read)

A wind turbine's main bearing spins **around one billion times** over its life.
When it fails with no warning, the operator suddenly needs a crane, a specialist
crew, and an emergency repair that can cost roughly **$150k–$300k per event**.

**AeroVigil turns everyday turbine telemetry into an early warning.** It tells
you *how much healthy life is left, how sure the model is, and whether you still
have time to schedule the repair* instead of reacting to a surprise failure.

| AeroVigil at a glance | |
|---|---|
| Early warning | ~45 days before failure |
| Inputs | 6 standard SCADA signals — no new sensors needed |
| Outputs | Days of life left, confidence range, risk level |
| Demo accuracy | 94.2% early-warning accuracy |
| Live demo | [aerovigil.abacusai.app](https://aerovigil.abacusai.app) |

### Three things to know

1. **It is physics-guided, not pure pattern-matching.** Predictions are checked
   against ISO 281 bearing-life logic, so answers stay physically sensible.
2. **It is honest about uncertainty.** You get a range and a risk level, not one
   overconfident number.
3. **It advises; it never controls.** AeroVigil helps humans plan maintenance.
   It does not send commands to the turbine.

> **New to the topic?** Read the zero-jargon [explainer](docs/EXPLAINER.md).
> **Pitching or evaluating?** Open the [investor one-pager](docs/PITCH.md).

## Try it right now — offline demo in 2 commands

[![AeroVigil demo GIF](https://raw.githubusercontent.com/rajaram-2005/wind-turbine-pg-bnn/arena%2F019fd767-wind-turbine-pg-bnn/docs/assets/demo.gif)](docs/assets/aerovigil-demo.mp4)

```bash
python scripts/train_pg_demo.py   # trains a small demo model (a few minutes)
python gradio_app/app.py          # opens the demo UI in your browser
```

The app prefers local weights from `artifacts/pg_bnn_demo/`, so the demo still
works with **no internet** after the training step.

- Full narrated demo video: [`docs/assets/aerovigil-demo.mp4`](docs/assets/aerovigil-demo.mp4)
- Demo script and fallback plan: [`docs/DEMO_RUNBOOK.md`](docs/DEMO_RUNBOOK.md)

## Quick start (for developers)

### 1. Install

```bash
git clone https://github.com/rajaram-2005/wind-turbine-pg-bnn.git
cd wind-turbine-pg-bnn
python -m pip install -e .
```

### 2. Predict in Python

```python
import torch
from aerovigil_pg_bnn import MonteCarloVI, PhysicsGuidedBNN

# Downloads config.json and bnn_demo.pt from the Hugging Face model repo.
model = PhysicsGuidedBNN.from_pretrained("AerovigilAI/wind-turbine-pg-bnn")

vi = MonteCarloVI(model, num_samples=100)
result = vi.predict_single([1.5, 45.0, 60.0, 2000.0, 9.0, 1000.0])
print(result)
```

`predict_single` returns the mean RUL, uncertainty, 95% interval, risk level,
and whether maintenance planning is recommended at the 45-day threshold.

### 3. Or use the command line

Save `telemetry.json`:

```json
{
  "vibration_rms": 1.5,
  "bearing_temp": 45.0,
  "generator_temp": 60.0,
  "power_output": 2000.0,
  "wind_speed": 9.0,
  "operating_hours": 1000.0
}
```

Then run:

```bash
aerovigil-infer --input telemetry.json --samples 100
```

## What the answers look like

| Signal | Meaning |
|---|---|
| 🟢 45+ days of life left | Healthy — keep monitoring |
| 🟡 14–45 days | Schedule maintenance |
| 🔴 under 14 days | Urgent — act now |

The model also reports how certain it is. Wide uncertainty bands mean the
prediction should be reviewed by a human before acting on it.

## Honest limits (read this)

- **Advisory only.** Outputs are decision-support for reliability engineers and
  maintenance planners. AeroVigil never emits turbine control commands, LOTO
  procedures, or work instructions. See [`docs/SAFETY.md`](docs/SAFETY.md).
- **Demo numbers are not field-validated.** The headline metrics come from a
  deterministic synthetic demo campaign; validate on real site data before
  operations depend on them.
- **Built for wind drivetrain bearings.** Other rotating machinery needs domain
  adaptation before use.
- **45-day horizon.** Longer-term predictions carry increasing uncertainty.
- **Simplified physics.** The physics term is an ISO 281-inspired operating-hours
  constraint, not a full load-history bearing-life model.

---

# For developers and researchers

## Technical details

### What the model is

PG-BNN couples data-driven learning with first-principles physics to estimate
the remaining useful life (RUL) of wind turbine drivetrain bearings. Unlike a
black-box point estimator, it returns a **probabilistic** RUL estimate: Bayesian
layers capture epistemic uncertainty, Monte Carlo Variational Inference (MCVI)
samples the posterior, and a simplified ISO 281 L10-life relationship acts as a
soft physics constraint during training.

### Key innovations

| Feature | Benefit |
|---------|---------|
| ISO 281 physics integration | Grounds predictions in tribological first principles |
| Bayesian layers | Captures epistemic uncertainty in sparse-data regimes |
| SCADA telemetry fusion | Uses vibration, temperature, power, wind-speed, and runtime signals |
| Variational inference | Repeated stochastic forward passes give uncertainty estimates |

### Architecture

| Component | Specification |
|-----------|---------------|
| Input layer | 6 SCADA telemetry features |
| Hidden layers | Bayesian linear layers: 128 → 64 → 32 |
| Activations | ReLU with dropout (`p=0.2`) |
| Output heads | RUL mean and log-variance |
| Variational family | Mean-field Gaussian weights and biases |
| Physics term | MSE against clamped ISO 281 L10-life reference minus operating hours |
| Loss | Gaussian NLL + β·KL term + physics loss |
| Inference | Stochastic forward passes; mean, std, and percentile intervals |

### Input format

| Feature | Description | Units |
|---------|-------------|-------|
| `vibration_rms` | Drive-train vibration RMS | mm/s |
| `bearing_temp` | Main bearing temperature | °C |
| `generator_temp` | Generator winding temperature | °C |
| `power_output` | Active power generation | kW |
| `wind_speed` | Nacelle wind speed | m/s |
| `operating_hours` | Cumulative operating time | hours |

### Output format

`PhysicsGuidedBNN.forward` returns `rul_mean` and `rul_log_var`, each of shape
`(batch, 1)`. The higher-level `MonteCarloVI.predict_single` returns a
dictionary with `predicted_rul_days`, `uncertainty_days`,
`confidence_interval_95`, `risk_level`, and `maintenance_recommended`.

### Training

Training uses deterministic synthetic data for demonstration and CI (production
use would require historical SCADA/condition-monitoring data), z-score
normalization, and a chronological 70/15/15 train/validation/test split.

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam, lr 1e-3 with cosine annealing |
| Batch size | 64 |
| Epochs | 200 |
| Physics loss weight | 0.1 |
| ELBO weight (β) | 0.01 |
| Early stopping patience | 20 epochs |

### Evaluation

> **Metric caveat.** Headline values come from the deterministic 500-asset demo
> campaign in [`scripts/eval_accuracy.py`](scripts/eval_accuracy.py). Useful for
> illustrating the model, not a substitute for site-specific validation.

| Metric | Value |
|--------|-------|
| MAE | 4.3 days |
| RMSE | 6.1 days |
| Accuracy @ 45 days | 94.2% (471/500) |
| Recall | 100% — no within-horizon failures missed |
| Calibration error | 0.03 |

| Benchmark comparison | MAE (days) | Uncertainty | Physics Valid |
|----------------------|------------|-------------|---------------|
| Standard LSTM | 8.7 | ❌ No | ❌ No |
| Deep Ensembles | 5.2 | ✅ Yes | ❌ No |
| **PG-BNN** | **4.3** | **✅ Yes** | **✅ Yes** |

Run the evaluation locally: `python scripts/eval_accuracy.py`.

### Loading the model manually

```python
import json
import torch
from huggingface_hub import hf_hub_download
from aerovigil_pg_bnn import PhysicsGuidedBNN

repo_id = "AerovigilAI/wind-turbine-pg-bnn"
config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
model_path = hf_hub_download(repo_id=repo_id, filename="bnn_demo.pt")

with open(config_path) as f:
    config = json.load(f)

model = PhysicsGuidedBNN(config)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
```

For MCVI, keep dropout active by calling `model.train()` before repeated forward
passes, or use `MonteCarloVI` as shown above.

## REST API

```bash
python -m pip install -e ".[api]"
python -m aerovigil_pg_bnn.api    # listens on 0.0.0.0:8000
```

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Model-health check |
| `GET /model/info` | Model metadata |
| `POST /predict` | Single-telemetry RUL prediction |
| `POST /predict/batch` | Batch predictions |
| `POST /predict/stream` | Streamed Monte Carlo samples |
| `GET /docs` | OpenAPI UI |

```bash
curl -X POST "http://localhost:8000/predict?n_mcmc_samples=100" \
  -H "Content-Type: application/json" \
  -d @telemetry.json
```

The broader AeroVigil advisory service — including `/advisory`, digital-twin,
telemetry-compression, and fleet-report endpoints — lives in
[`src/api/app.py`](src/api/app.py).

## Intended use

**Who it is for:** wind-farm operators optimizing maintenance schedules,
condition-monitoring engineers needing uncertainty-aware RUL estimates, and
asset managers planning around inspection and CMMS workflows.

**Out of scope:** real-time turbine control; other rotating machinery without
domain adaptation; replacing OEM service manuals, inspections, oil analysis, or
qualified engineering review; extreme environments with contaminated
lubrication, sensor outages, or unusual operating regimes.

## Repository layout

| Path | Purpose |
|------|---------|
| [`model.py`](model.py) | Compact standalone model reference |
| [`src/aerovigil_pg_bnn/`](src/aerovigil_pg_bnn) | Packaged model, MCVI utility, CLI, and inference API |
| [`src/models/`](src/models) | Research BNN, predictor, serving, and telemetry pipeline |
| [`src/physics/`](src/physics) | ISO 281 and operating-limit physics constraints |
| [`src/digital_twin/`](src/digital_twin) | Turbine specs, virtual asset, and scenario simulation |
| [`src/agents/hermes.py`](src/agents/hermes.py) | Few-shot onboarding and promotion gating |
| [`scripts/`](scripts) | Training, evaluation, pipeline, and smoke-test scripts |
| [`tests/`](tests) | Unit and integration tests |
| [`docs/`](docs) | Plain-language, safety, and architecture documentation |

## Development

```bash
python -m pip install -e ".[dev]"
make ci                  # full local quality gate: lint, format, type, security, test, build

# Individual checks
make lint                # ruff check
make format-check        # ruff format --check
make typecheck           # mypy
make security            # bandit
make test                # pytest
```

Container images and Kubernetes manifests are provided for API and demo
deployments: `docker compose up api`.

## Learn more

| Document | What it covers |
|----------|----------------|
| [`docs/EXPLAINER.md`](docs/EXPLAINER.md) | The idea with zero jargon |
| [`docs/PITCH.md`](docs/PITCH.md) | Investor one-pager |
| [`docs/SAFETY.md`](docs/SAFETY.md) | Advisory-only safety contract |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System architecture |
| [`docs/DIAGRAMS.md`](docs/DIAGRAMS.md) | Diagrams |
| [`docs/PROPOSAL.md`](docs/PROPOSAL.md) | Project proposal |
| [`docs/DEMO_RUNBOOK.md`](docs/DEMO_RUNBOOK.md) | Stage script and fallback plan |
| [`docs/DEMO_VIDEO_SCRIPT.md`](docs/DEMO_VIDEO_SCRIPT.md) | Narration and shot list |

## Citation

If you use this model in your research, please cite:

```bibtex
@software{aerovigil_pgbnn_2026,
  author = {Aerovigil AI},
  title = {Physics-Guided Bayesian Neural Network for Wind Turbine RUL Prediction},
  year = {2026},
  url = {https://huggingface.co/AerovigilAI/wind-turbine-pg-bnn}
}
```

## Model Card Contact

- **Organization**: [Aerovigil AI](https://huggingface.co/AerovigilAI)
- **Repository Issues**: [github.com/rajaram-2005/wind-turbine-pg-bnn/issues](https://github.com/rajaram-2005/wind-turbine-pg-bnn/issues)
- **Hugging Face Hub**: [huggingface.co/AerovigilAI/wind-turbine-pg-bnn](https://huggingface.co/AerovigilAI/wind-turbine-pg-bnn)

---

*Last updated: August 2026*
