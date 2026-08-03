# wind-turbine-pg-bnn

**Physics-Guided Bayesian Neural Network for wind-turbine drivetrain Remaining Useful Life (RUL) prediction.**

> ⚠️ **SAFETY NOTICE — ADVISORY / DECISION-SUPPORT ONLY**
> This software produces *engineering recommendations* for reliability and maintenance
> planners. It does **not** issue direct PLC/SCADA setpoints, torque throttles, speed
> commands, or Lockout/Tagout (LOTO) procedures. All outputs must be reviewed by a
> qualified operator and cross-checked against OEM documentation and site-specific
> safety procedures (OSHA 29 CFR 1910.147 / IEC 61508 / ISO 14118) before any physical
> action on a turbine. See `docs/SAFETY.md`.

## Modules

| Path | Purpose |
| ---- | ------- |
| `src/data/` | SCADA-style CSV/parquet ingestion, sliding-window feature extraction, robust normalization |
| `src/physics/` | ISO 281 bearing $L_{10}$ life, gearbox/generator hard limits, differentiable $L_{\text{physics}}$ penalty |
| `src/models/bnn.py` | Bayesian MLP (PyTorch) trained with Monte Carlo Variational Inference; outputs mean RUL + epistemic + aleatoric variance |
| `src/models/telemetry/` | AeroZip-style delta + deadband + quantize compression with anomaly-bypass |
| `src/eval/` | ECE calibration, expected asset utilization, failure-type classification metrics |
| `src/utils/` | Safety gates, logging, schema |

## Quick start (research / offline mode)

```bash
pip install -e ".[dev]"
python scripts/train_demo.py           # trains on synthetic drivetrain data
pytest -q                              # runs unit tests
```

Outputs from `predict_rul()` are always wrapped in an `AdvisoryRecommendation`
object that explicitly marks itself as non-actuating. The safety gate in
`src/utils/safety.py` will refuse to emit numeric "throttle" or "LOTO" fields.
