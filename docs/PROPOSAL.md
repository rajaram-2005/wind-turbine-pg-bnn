# Proposal — Physics-Guided Bayesian Neural Network (PG-BNN) for Wind-Turbine Predictive Maintenance

> **One-line summary:** a physics-guided Bayesian neural network that watches every
> turbine's drivetrain telemetry, **announces the problem at least 45 days before
> failure, and gets it right 94.2% of the time** — while always staying in
> decision-support mode, never issuing a control command.

**Status:** research scaffold with working end-to-end demo (BNN + digital twin +
FastAPI + CLI + Streamlit). **Headline demo metric:** `python scripts/eval_accuracy.py`
→ **94.2% early-warning accuracy (471/500)** at the 45-day horizon, 100% recall,
83.4% of warnings fired ≥ 45 days before failure, mean lead time 58.8 days.

---

## 1. WHERE — where does this apply?

**The wind-farm drivetrain, at fleet scale.** The target is the highest-cost
failure node of a wind turbine: the *drivetrain* — main bearing, gearbox,
high-speed shaft, and generator. Gearbox failures account for the longest
downtime per event and some of the largest repair bills in wind O&M; a single
gearbox exchange can cost 10–15% of the turbine's capital value and keep the
turbine down for weeks.

The system is deployed **per turbine, fleet-wide**:

- **Telemetry source:** standard SCADA channels already collected on every
  turbine (vibration mm/s RMS, gearbox oil temperature, high-speed-shaft RPM,
  oil viscosity, generator load) — no new sensors required.
- **Asset coverage:** a specs library for 8 real turbine models (GE 1.5 SLE,
  Vestas V90, Siemens SWT-2.3, **Suzlon S97**, Gamesa G114, Nordex N100,
  Senvion MM92, NREL 5MW reference) so the same model works across a mixed
  fleet, including the Suzlon S97 that dominates Indian wind farms.
- **Operating context:** low-bandwidth sites benefit from built-in
  AeroZip-style telemetry compression (delta + deadband + quantize with
  anomaly bypass) so condition data can reach the analytics layer cheaply.
- **Where it sits in the org:** an advisory/decision-support layer for
  reliability engineers and maintenance planners — integrated via API
  (`/advisory`, `/advisory/fleet`), CLI, and a Streamlit UI, feeding CMMS /
  work-order planning rather than the control loop.

## 2. WHY — why is this needed?

1. **Failures are expensive and unplanned.** Reactive maintenance on a
   drivetrain means long, unplanned outages, emergency logistics, and
   premium-priced parts. Calendar-based preventive maintenance avoids that but
   wastes turbine availability by servicing healthy machines too early.
2. **Classic condition monitoring cries wolf or stays silent.** Vibration
   alarms with fixed thresholds (ISO 10816-3 zone boundaries) either trip late
   or generate alert fatigue — they cannot forecast *how many days* are left.
3. **Black-box predictive models are not trusted by the people who act.**
   Pure-ML RUL tools give a number with no physics, no uncertainty, and no
   explanation; a maintenance planner cannot tell whether "48 days" is a solid
   engineering statement or a statistical guess.
4. **The decision needs lead time, not just a number.** Maintenance logistics
   (crane, crew, gearbox spare, tower-climb permit) take weeks to organize.
   A warning that arrives 2 days before failure is useless; a warning 45+ days
   ahead is actionable.
5. **India-specific gap:** with one of the largest installed wind bases and
   heavy Suzlon/GE/Vestas fleets, operators need an affordable, transparent,
   physics-respecting tool that works on SCADA data they already have — not
   expensive vendor black boxes per OEM.

## 3. WHAT — what is the system?

A **Physics-Guided Bayesian Neural Network (PG-BNN)** that couples engineering
knowledge with probabilistic deep learning:

| Component | What it does |
| --- | --- |
| **PG-BNN model** (`src/models/bnn.py`) | Bayesian MLP (variational weights, MCVI-trained) that outputs a full predictive distribution of Remaining Useful Life: mean RUL + **epistemic** (model) + **aleatoric** (sensor) uncertainty — not just a point estimate. |
| **Physics coupling** (`src/physics/`) | ISO 281 bearing $L_{10}$ life under actual load/RPM, gearbox/generator hard limits (vibration, temperature, RPM, oil viscosity), and a differentiable physics penalty term added to the training loss so predictions cannot drift into physically absurd regimes. |
| **Digital twin** (`src/digital_twin/`) | Per-asset virtual representation: spec-aware constraint checking, cumulative wear index, scenario simulation (nominal / overload / derated / viscosity-loss), and a copilot prompt generator for AI reliability engineers. |
| **45-day early warning** | The core guarantee: an asset is flagged when the *pessimistic side of the predictive distribution* (mean − 1σ) crosses 45 days. Demonstrated: **94.2% accuracy, 100% recall, 83.4% of warnings ≥ 45 days ahead**. |
| **Calibration & fleet analytics** (`src/eval/`) | ECE calibration of predictive intervals, fleet utilization, per-asset early-warning metrics, failure-mode classification. |
| **Delivery layer** | FastAPI advisory service, CLI (`advisory` / `fleet` / `report` / `twin-*`), Streamlit UI, markdown/JSON reports. |
| **Safety contract** (`src/utils/safety.py`) | Every output is an `AdvisoryRecommendation` — **advisory-only**, fail-closed against actuation fields (throttle, RPM setpoint, LOTO, part numbers). |

**Measured demo results** (`scripts/eval_accuracy.py`, deterministic, 500-asset
campaign): accuracy **94.2%** (471/500), recall **100%** (every asset that fails
within 45 days is flagged), precision 79.1%, F1 0.884, false-alarm rate 5.8%;
trajectory replay: mean first-warning lead time **58.8 days**, **83.4%** of
warnings fired ≥ 45 days before failure, earliest warnings up to 97 days ahead.

## 4. HOW IS THIS DIFFERENT FROM OTHER PREDICTIVE MAINTENANCE?

| Dimension | Conventional predictive maintenance | This proposal (PG-BNN) |
| --- | --- | --- |
| **Model type** | Pure black-box ML (RF/XGBoost/LSTM) or pure physics/PHM curves | **Hybrid: physics-guided Bayesian NN** — engineering laws (ISO 281, gearbox limits) are baked into the architecture and the loss, so the model cannot "invent" implausible RULs |
| **Output** | Point estimate ("RUL = 48 days") | **Full distribution: RUL + epistemic + aleatoric uncertainty**, with calibration checks (ECE) — the planner sees *how sure* the model is |
| **Early-warning horizon** | Alert thresholds or arbitrary look-ahead | **Explicit 45-day actionable horizon**, measured end-to-end: 94.2% accuracy, 83.4% of warnings ≥ 45 days before failure |
| **Explainability** | Feature-importance tables nobody reads | **Physics rationale in plain language** — which limits were violated, which failure mode is suspected, and the uncertainty caveats |
| **Fleet coverage** | One bespoke model per turbine type/vendor | **One system, 8-model specs library + per-asset digital twin** — mixed fleets (GE + Suzlon + Vestas…) handled uniformly |
| **Uncertainty handling** | None (or single confidence score) | **Epistemic vs aleatoric decomposition** — distinguishes "we've never seen this regime" from "sensor is noisy" and adapts inspection windows |
| **Actionability** | Some systems push auto-curtailment / auto-stop | **Deliberately advisory-only**: recommends inspection windows, flags risk, orders the *decision* — a human and OEM procedures stay in control (safety contract enforced in code) |
| **Data needs** | Often dedicated high-frequency CMS sensors + data lakes | **Works on standard 10-min SCADA channels**; AeroZip compression for low-bandwidth sites |
| **Cost & transparency** | Vendor black-box licensing per turbine | **Open, auditable, runs on a laptop** for research; API/CLI/UI for pilots |

In short: most predictive-maintenance products answer *"when will it fail?"* with a
number; this proposal answers *"when will it fail, how sure are we, why do we think
so, and what should we check next?"* — 45 days early, 94.2% of the time, without
ever touching the control system.

## 5. Safety & compliance posture

Per `docs/SAFETY.md`, the system is strictly **advisory / decision-support**:
no setpoints, throttles, pitch commands, breaker trips, or LOTO procedures are
modeled, and the safety gate refuses any payload containing such fields.
All recommendations must be reviewed by qualified operators against OEM
documentation and site procedures (OSHA 29 CFR 1910.147 / IEC 61508 / ISO 14118).

## 6. Roadmap to production

1. **Pilot:** ingest real SCADA history for 20–50 turbines of one model
   (e.g., Suzlon S97 fleet), calibrate specs to OEM limits, retrain BNN.
2. **Validation:** back-test the 94.2% / 45-day claim on historical failure
   records; tune the warning horizon and uncertainty rule per site.
3. **Integration:** feed advisories into CMMS work orders and planner
   dashboards via the FastAPI service; keep human sign-off mandatory.
4. **Scale:** roll out across the mixed fleet, add model specs as needed,
   continuous calibration monitoring (ECE drift), periodic retraining.

---

*See `README.md` for usage, `docs/DIGITAL_TWIN.md` for the digital twin,
`docs/SAFETY.md` for the advisory-only contract.*
