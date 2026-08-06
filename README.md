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

# Physics-Guided Bayesian Neural Network for Wind Turbine RUL Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-ffd21e.svg)](https://huggingface.co/AerovigilAI/wind-turbine-pg-bnn)
[![Live Demo](https://img.shields.io/badge/demo-live-22c55e.svg)](https://aerovigil.abacusai.app)

**Aerovigil AI** presents a Physics-Guided Bayesian Neural Network (PG-BNN) for
**Remaining Useful Life (RUL) prediction and diagnostic advisory** of wind
turbine drivetrain components. The model combines ISO 281 bearing-life physics
with Monte Carlo Variational Inference (MCVI) to provide uncertainty-aware
prognostics for predictive-maintenance planning.

> **Advisory only.** This repository produces decision-support outputs for
> reliability engineers and maintenance planners. It does not emit turbine
> control commands, LOTO procedures, part numbers, or authoritative work
> instructions. See [`docs/SAFETY.md`](docs/SAFETY.md).

| Attribute | Value |
|-----------|-------|
| **Developer** | [Aerovigil AI](https://huggingface.co/AerovigilAI) |
| **Model Name** | `wind-turbine-pg-bnn` |
| **Domain** | Wind turbine predictive maintenance |
| **Task** | Remaining Useful Life (RUL) prediction and diagnostic advisory |
| **Architecture** | Bayesian neural network with physics constraints |
| **Framework** | PyTorch |
| **Inference** | Monte Carlo Variational Inference (MCVI) |
| **Physics Engine** | ISO 281 bearing-life theory |
| **Early Warning Horizon** | 45 days |
| **Demo Accuracy** | 94.2% early-warning accuracy |
| **Demo Recall** | 100% |
| **Live Demo** | [aerovigil.abacusai.app](https://aerovigil.abacusai.app) |
| **Source Code** | [github.com/rajaram-2005/wind-turbine-pg-bnn](https://github.com/rajaram-2005/wind-turbine-pg-bnn) |

---

## Model Description

The PG-BNN couples data-driven deep learning with first-principles physics to
predict remaining useful life for wind turbine drivetrain bearings. Unlike a
black-box point estimator, PG-BNN returns a probabilistic RUL estimate and uses
ISO 281 bearing-fatigue guidance as a soft physics constraint during training.

Key properties:

- **Uncertainty quantification:** probabilistic RUL estimates from variational
  Bayesian layers and MC sampling.
- **Physics compliance:** predictions are regularized against a simplified ISO
  281 L10-life relationship based on cumulative operating hours.
- **Early detection:** the advisory threshold and evaluation horizon are 45
  days.
- **Decision support:** outputs are intended for inspection and maintenance
  planning, not direct turbine actuation.

### Key Innovations

| Feature | Benefit |
|---------|---------|
| ISO 281 physics integration | Grounds predictions in tribological first principles |
| Bayesian layers | Captures epistemic uncertainty in sparse-data regimes |
| SCADA telemetry fusion | Uses vibration, temperature, power, wind-speed, and runtime signals |
| Variational inference | Enables repeated stochastic forward passes for uncertainty estimates |

---

## Quick Start

### Option 1: Use the packaged model code locally

```bash
git clone https://github.com/rajaram-2005/wind-turbine-pg-bnn.git
cd wind-turbine-pg-bnn
python -m pip install -e .
```

For development and the full quality pipeline:

```bash
python -m pip install -e ".[dev]"
make ci
```

### Option 2: Install only the model dependencies for a Hugging Face download

If you are using the model class from this repository with weights hosted on
Hugging Face, install PyTorch and Hugging Face Hub:

```bash
python -m pip install torch huggingface_hub
```

### Python inference

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

### Command-line inference

Create `telemetry.json`:

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

### REST API

Install API dependencies and start the lightweight inference server:

```bash
python -m pip install -e ".[api]"
python -m aerovigil_pg_bnn.api
```

The server listens on `0.0.0.0:8000` and exposes:

- `GET /health` — model-health check
- `GET /model/info` — model metadata
- `POST /predict` — single-telemetry RUL prediction
- `POST /predict/batch` — batch predictions
- `POST /predict/stream` — streamed Monte Carlo samples
- `GET /docs` — OpenAPI UI

Example request:

```bash
curl -X POST "http://localhost:8000/predict?n_mcmc_samples=100" \
  -H "Content-Type: application/json" \
  -d @telemetry.json
```

The repository also contains the broader AeroVigil advisory service in
[`src/api/app.py`](src/api/app.py), including `/advisory`, digital-twin,
telemetry-compression, and fleet-report endpoints.

---

## Intended Use

### Direct Use

- **Wind-farm operators** optimizing maintenance schedules and reducing
  unplanned downtime.
- **Condition-monitoring engineers** needing uncertainty-aware RUL estimates
  for drivetrain bearings.
- **Asset managers** evaluating the remaining life of turbine mechanical
  components alongside inspection and CMMS workflows.

### Out-of-Scope Use

- This model is designed for **wind turbine drivetrain bearings** and should
  not be applied to other rotating machinery without domain adaptation.
- It is not intended for real-time turbine control.
- Predictions should not replace OEM service manuals, inspections, oil
  analysis, or qualified engineering review.
- The simplified ISO 281 constraint may be insufficient for extreme
  environments, contaminated lubrication, sensor outages, or unusual operating
  regimes.

---

## Loading the Model Manually

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

For MCVI, keep dropout active by calling `model.train()` before repeated
forward passes, or use `MonteCarloVI` as shown above.

---

## Technical Specifications

### Model Architecture

| Component | Specification |
|-----------|---------------|
| Input layer | 6 SCADA telemetry features |
| Hidden layers | Bayesian linear layers: 128 → 64 → 32 |
| Activations | ReLU with dropout (`p=0.2`) |
| Output heads | RUL mean and log-variance |
| Variational family | Mean-field Gaussian weights and biases |
| Physics term | MSE against clamped ISO 281 L10-life reference minus operating hours |
| Loss | Gaussian negative log-likelihood + β KL term + physics loss |
| Inference | Stochastic forward passes; mean, standard deviation, and percentile intervals |

### Input Format

| Feature | Description | Units |
|---------|-------------|-------|
| `vibration_rms` | Drive-train vibration RMS | mm/s |
| `bearing_temp` | Main bearing temperature | °C |
| `generator_temp` | Generator winding temperature | °C |
| `power_output` | Active power generation | kW |
| `wind_speed` | Nacelle wind speed | m/s |
| `operating_hours` | Cumulative operating time | hours |

### Output Format

`PhysicsGuidedBNN.forward` returns:

| Output | Description | Shape |
|--------|-------------|-------|
| `rul_mean` | Expected remaining useful life | `(batch, 1)` |
| `rul_log_var` | Log-variance for uncertainty modeling | `(batch, 1)` |

The higher-level `MonteCarloVI.predict_single` helper returns a dictionary with
`predicted_rul_days`, `uncertainty_days`, `confidence_interval_95`,
`risk_level`, and `maintenance_recommended`.

---

## Training Details

### Training Data

The repository includes deterministic synthetic-data and evaluation scripts
for demonstration and CI. Historical SCADA/condition-monitoring data from
operational wind farms would be required for production calibration.

Preprocessing in the current configuration uses z-score normalization and
physics-informed feature handling. The configuration declares a chronological
70/15/15 train/validation/test split as the intended training regime.

### Training Regime

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 1e-3 with cosine annealing |
| Batch size | 64 |
| Epochs | 200 |
| Physics loss weight | 0.1 |
| ELBO weight (β) | 0.01 |
| Early stopping patience | 20 epochs |

---

## Evaluation

> **Metric caveat.** The headline values below come from the repository's
> deterministic 500-asset demonstration campaign in
> [`scripts/eval_accuracy.py`](scripts/eval_accuracy.py). They are useful for
> illustrating the model and early-warning logic, but should not be interpreted
> as field-validated production performance without site-specific validation.

| Metric | Value | Notes |
|--------|-------|-------|
| MAE | 4.3 days | Configuration/reporting value for the 45-day horizon |
| RMSE | 6.1 days | Root mean squared error |
| Accuracy @ 45 days | 94.2% | Early-warning classification demo result (471/500) |
| Recall | 100% | No within-horizon failures missed in the demo campaign |
| Calibration error | 0.03 | Expected calibration error |

### Benchmark Comparison

| Model | MAE (days) | Uncertainty | Physics Valid |
|-------|------------|-------------|---------------|
| Standard LSTM | 8.7 | ❌ No | ❌ No |
| Deep Ensembles | 5.2 | ✅ Yes | ❌ No |
| **PG-BNN** | **4.3** | **✅ Yes** | **✅ Yes** |

Run the evaluation locally:

```bash
python scripts/eval_accuracy.py
```

---

## Repository Layout

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
| [`docs/`](docs) | Architecture, proposal, and safety documentation |

---

## Bias, Risks, and Limitations

### Known Limitations

- **Data dependency:** performance can degrade under sensor drift, missing
  channels, calibration faults, or distribution shift.
- **Domain specificity:** the model and specs focus on onshore horizontal-axis
  turbines; offshore, vertical-axis, or OEM-specific deployments require
  retraining and validation.
- **Physics simplification:** the training physics term is a simplified ISO
  281 operating-hours constraint, not a full load-history bearing-life model.
- **Temporal scope:** the 45-day horizon is the supported early-warning target;
  longer predictions carry increasing uncertainty.
- **Synthetic demonstration metrics:** repository numbers should be validated
  on real site data before operational decisions depend on them.

### Risk Mitigation

- Use predictions as one input to maintenance planning, together with
  inspections, SCADA trends, vibration/CMS trends, oil analysis, and OEM
  guidance.
- Retrain or recalibrate periodically with newly labeled operational data.
- Route high-uncertainty predictions to human review.
- Keep all outputs in advisory/decision-support workflows.

---

## Development

```bash
# Install development dependencies
python -m pip install -e ".[dev]"

# Run the full local quality gate
make ci

# Individual checks
make lint
make format-check
make typecheck
make security
make test
```

Container images and Kubernetes manifests are also provided for API and demo
deployments:

```bash
docker compose up api
```

---

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

---

## Model Card Contact

- **Organization**: [Aerovigil AI](https://huggingface.co/AerovigilAI)
- **Repository Issues**: [github.com/rajaram-2005/wind-turbine-pg-bnn/issues](https://github.com/rajaram-2005/wind-turbine-pg-bnn/issues)
- **Hugging Face Hub**: [huggingface.co/AerovigilAI/wind-turbine-pg-bnn](https://huggingface.co/AerovigilAI/wind-turbine-pg-bnn)

---

*Last updated: August 2026*
