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

## Model Overview

**Aerovigil AI** presents a Physics-Guided Bayesian Neural Network (PG-BNN) designed for **Remaining Useful Life (RUL) prediction and diagnostic advisory** of wind turbine drivetrain components. This model integrates ISO 281 bearing physics with Monte Carlo Variational Inference to deliver uncertainty-aware prognostics for predictive maintenance in wind energy operations.

| Attribute | Value |
|-----------|-------|
| **Developer** | Aerovigil AI |
| **Model Name** | `wind-turbine-pg-bnn` |
| **Domain** | Wind Turbine Predictive Maintenance |
| **Task** | Remaining Useful Life (RUL) Prediction & Diagnostic Advisory |
| **Architecture** | Bayesian Neural Network with Physics Constraints |
| **Framework** | PyTorch |
| **Inference** | Monte Carlo Variational Inference (MCVI) |
| **Physics Engine** | ISO 281 Bearing Life Theory |
| **Early Warning Horizon** | 45 days |
| **Accuracy** | 94.2% |
| **Recall** | 100% |
| **Live Demo** | [aerovigil.abacusai.app](https://aerovigil.abacusai.app) |
| **Source Code** | [github.com/rajaram-2005/wind-turbine-pg-bnn](https://github.com/rajaram-2005/wind-turbine-pg-bnn) |

---

## Model Description

The Physics-Guided Bayesian Neural Network combines data-driven deep learning with first-principles physics to predict the remaining useful life of wind turbine drivetrain bearings. Unlike traditional black-box models, PG-BNN embeds ISO 281 bearing fatigue life equations as soft physics constraints within a Bayesian neural architecture. This hybrid approach ensures:

- **Uncertainty Quantification**: Probabilistic RUL estimates via Monte Carlo dropout variational inference
- **Physics Compliance**: Predictions respect fundamental bearing degradation mechanics
- **Early Detection**: 45-day advance warning horizon for maintenance scheduling
- **High Reliability**: 100% recall ensures no critical failures are missed

### Key Innovations

| Feature | Benefit |
|---------|---------|
| ISO 281 Physics Integration | Grounds predictions in tribological first principles |
| Bayesian Layers | Captures epistemic uncertainty in sparse data regimes |
| SCADA Telemetry Fusion | Ingests operational vibration, temperature, and power signals |
| Variational Inference | Enables real-time probabilistic inference at scale |

---

## Intended Use

### Direct Use
- **Wind farm operators** seeking to optimize maintenance schedules and reduce unplanned downtime
- **Condition monitoring engineers** requiring uncertainty-aware RUL estimates for drivetrain bearings
- **Asset managers** evaluating remaining life of turbine mechanical components

### Out-of-Scope Use
- This model is trained specifically for **wind turbine drivetrain bearings** and should not be applied to other rotating machinery without domain adaptation
- Not intended for real-time control decisions without human-in-the-loop validation
- Predictions assume standard ISO 281 operating conditions; extreme environments may require recalibration

---

## How to Use

### Installation

```bash
pip install torch huggingface_hub
```

### Loading the Model

```python
import torch
from huggingface_hub import hf_hub_download

# Download model weights from Hugging Face Hub
model_path = hf_hub_download(
    repo_id="rajaram-2005/wind-turbine-pg-bnn",
    filename="bnn_demo.pt"
)

# Load into your PyTorch model architecture
model = YourBNNArchitecture()  # Replace with your model class
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()
```

### Inference Example

```python
import torch

# Enable Monte Carlo dropout for uncertainty estimation
model.train()  # Keep dropout active for MCVI

# Run multiple forward passes for probabilistic prediction
n_mcmc_samples = 100
predictions = []

with torch.no_grad():
    for _ in range(n_mcmc_samples):
        rul_pred = model(input_telemetry)
        predictions.append(rul_pred)

# Compute mean prediction and epistemic uncertainty
mean_rul = torch.stack(predictions).mean(dim=0)
uncertainty = torch.stack(predictions).std(dim=0)

print(f"Predicted RUL: {mean_rul.item():.1f} days")
print(f"Uncertainty (±): {uncertainty.item():.1f} days")
```

---

## Technical Specifications

### Model Architecture

| Component | Specification |
|-----------|---------------|
| Input Layer | SCADA telemetry features (vibration, temperature, power, wind speed) |
| Hidden Layers | Fully-connected Bayesian layers with learnable mean/variance |
| Physics Layer | ISO 281 bearing life constraint regularization |
| Output Layer | Gaussian-distributed RUL prediction (mean + log-variance) |
| Activation | ReLU with Monte Carlo dropout (p=0.2) |
| Inference | Mean-field variational approximation |

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

| Output | Description | Shape |
|--------|-------------|-------|
| `rul_mean` | Expected remaining useful life | `(batch, 1)` |
| `rul_log_var` | Log-variance for uncertainty | `(batch, 1)` |

---

## Training Details

### Training Data
- **Source**: Historical SCADA and condition monitoring data from operational wind farms
- **Preprocessing**: ISO 281 physics-informed feature engineering, z-score normalization
- **Split**: 70% training, 15% validation, 15% testing (chronological)

### Training Regime

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 1e-3 with cosine annealing |
| Batch Size | 64 |
| Epochs | 200 |
| Physics Loss Weight | 0.1 |
| ELBO Weight (β) | 0.01 (annealed) |

---

## Evaluation

### Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| MAE (Mean Absolute Error) | 4.3 days | On 45-day horizon |
| RMSE | 6.1 days | Root mean squared error |
| Accuracy @ 45 days | 94.2% | Within ±5 day tolerance |
| Recall | 100% | No missed critical failures |
| Calibration Error | 0.03 | Well-calibrated uncertainties |

### Benchmark Comparison

| Model | MAE (days) | Uncertainty | Physics Valid |
|-------|------------|-------------|---------------|
| Standard LSTM | 8.7 | ❌ No | ❌ No |
| Deep Ensembles | 5.2 | ✅ Yes | ❌ No |
| **PG-BNN (Ours)** | **4.3** | **✅ Yes** | **✅ Yes** |

---

## Bias, Risks, and Limitations

### Known Limitations
- **Data dependency**: Performance degrades with sensor drift or missing telemetry channels
- **Domain specificity**: Trained on onshore horizontal-axis turbines; offshore or vertical-axis applications require retraining
- **Physics simplification**: ISO 281 assumes standard lubrication; contaminated or starved conditions not modeled
- **Temporal scope**: 45-day horizon validated; longer predictions have increasing uncertainty

### Risk Mitigation
- Always combine model output with scheduled inspections
- Retrain model quarterly with new operational data
- Flag predictions with uncertainty >10 days for human review

---

## Citation

If you use this model in your research, please cite:

```bibtex
@software{aerovigil_pgbnn_2024,
  author = {Aerovigil AI},
  title = {Physics-Guided Bayesian Neural Network for Wind Turbine RUL Prediction},
  year = {2024},
  url = {https://huggingface.co/rajaram-2005/wind-turbine-pg-bnn}
}
```

---

## Model Card Contact

- **Organization**: [Aerovigil AI](https://aerovigil.abacusai.app)
- **Repository Issues**: [github.com/rajaram-2005/wind-turbine-pg-bnn/issues](https://github.com/rajaram-2005/wind-turbine-pg-bnn/issues)
- **Hugging Face Hub**: [huggingface.co/rajaram-2005/wind-turbine-pg-bnn](https://huggingface.co/rajaram-2005/wind-turbine-pg-bnn)

---

*Last updated: August 2026*
