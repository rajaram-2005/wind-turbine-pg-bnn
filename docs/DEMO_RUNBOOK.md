# AeroVigil demo-day runbook

## Goal

Deliver a confident **5-minute live demo** that makes officers and non-technical judges understand three things fast:

1. the problem is expensive and real
2. the UI is easy to understand
3. the system is trustworthy because it shows uncertainty and stays advisory-only

---

## Pre-flight checklist

### The day before
- Run `python scripts/train_pg_demo.py`
- Confirm these files exist:
  - `artifacts/pg_bnn_demo/bnn_demo.pt`
  - `artifacts/pg_bnn_demo/config.json`
  - `artifacts/pg_bnn_demo/scaler.npz`
- Run `python -m src.unified_app`
- Open `http://localhost:8080` and confirm all eight console pages load
- On **Inference**, run healthy and critical six-signal snapshots
- Confirm `/health`, `/api/docs`, and `/model-api/docs` share port 8080
- Keep `docs/assets/aerovigil-demo.mp4` ready as backup

### 10 minutes before stage time
- Open the unified console and leave it on the **Inference** page
- Put the browser in **F11 / full-screen** mode
- Keep these tabs ready:
  - live app
  - `docs/PITCH.md`
  - `docs/EXPLAINER.md`
  - demo video backup
- Close noisy tabs / notifications
- Plug into power
- Verify audio works in case you need to play the narrated backup video
- Confirm hotspot fallback is ready if venue Wi-Fi is shaky

---

## 30-second hook

> "A bearing inside a wind turbine can fail as a surprise event that costs up to three hundred thousand dollars once you need a crane, a specialist crew, and lost power. AeroVigil is built to spot that failure about forty-five days early, so operators schedule the repair instead of paying for a rescue." 

Pause. Let the cost land.

---

## 30-second "what it is"

> "This is a physics-guided AI for wind turbine health. It reads signals turbines already produce — vibration, temperatures, power, wind, and total run hours — and turns them into a plain-language maintenance advisory. It never controls the turbine. It advises humans; humans decide." 

---

## Live demo sequence

### Demo table

| Step | What you click / show | What you say | What the audience should notice |
|---|---|---|---|
| 1 | Show the live **MIKA + KAI** mesh | "Every page uses one evidence path and one server." | Connected status and animated SCADA → HUMAN path |
| 2 | Open **Inference** with the healthy values | "This is a machine that looks normal." | Six validated SCADA signals on the same app |
| 3 | Press **Run inference** | "The model reports healthy runway and its uncertainty." | RUL gauge, risk chips, confidence interval table |
| 4 | Raise vibration/temperatures/hours to critical values | "Same turbine family, now after hidden degradation." | Input evidence changes visibly |
| 5 | Run inference again | "Now the estimate contracts into a maintenance window." | Risk and maintenance-review chips update |
| 6 | Open **Digital Twin** | "MIKA plans the response while KAI explains the physics." | Two live findings, agreement, ISO 281 evidence |
| 7 | Open **System** | "This is one app—not a collection of disconnected demos." | `/health`, services, twins, and durable rows on port 8080 |

### Key callouts during the live demo
- **Healthy → green** means the machine still has runway.
- **Critical → red** means the machine is inside an urgent response window.
- **Confidence interval = honesty**. Wider uncertainty means more human review.
- **MIKA + KAI** connect maintenance planning to physics evidence.
- **System health** proves every browser/API surface shares one process and port.

---

## 45-second trust section

> "There are four reasons to trust the output more than a generic black-box score. First, it is physics-guided, so the model is shaped by bearing-life intuition. Second, it shows uncertainty instead of pretending to know everything. Third, it is safety-gated and advisory-only. And fourth, the demo scoreboard is strong: 94.2 percent early-warning accuracy, 100 percent recall, and typical error in the plus-or-minus four to six day range." 

---

## 45-second engineering credibility section

> "This is not just a slide deck model. The repo includes the model package, one FastAPI deployment boundary, an eight-page browser console, CLI tools, digital twins, telemetry compression, onboarding logic, and durable jobs. Even the narrated video is rendered from real model inference, not canned screenshots."

---

## Close

> "AeroVigil helps operators see a costly bearing failure early enough to act. The value is simple: fewer surprises, cheaper maintenance planning, and more confidence in what the AI does and does not know." 

---

## Q&A cheat sheet

### 1) "Is this real?"
**Answer:**
Yes. The code runs end to end in this repository. The live demo uses local weights trained in the repo, the UI performs real inference, and the narrated video is built from real model outputs.

### 2) "Can it control the turbine?"
**Answer:**
No. It is explicitly advisory-only. It does not send control commands or automate actuation.

### 3) "Why is uncertainty important?"
**Answer:**
Because maintenance decisions are expensive. A single point estimate hides doubt; AeroVigil shows the spread so planners know how confident the model is.

### 4) "Does it need new hardware?"
**Answer:**
For the demo, no. It uses standard SCADA-style signals the turbine already emits.

### 5) "What happens with no internet?"
**Answer:**
The demo still works. The unified app loads local weights from `artifacts/pg_bnn_demo`; the browser console and every API are served locally on port 8080.

### 6) "How far ahead does it warn?"
**Answer:**
The primary planning threshold is 45 days. That's enough time to schedule labor, parts, and crane access.

### 7) "What's the next step after the demo?"
**Answer:**
A pilot fleet. Real site data lets us calibrate the model, measure avoided surprise failures, and turn the package into a field case study.

---

## Failure fallback plan

### Level 1 — minor hiccup
If the app is slow or the click misses:
- refresh once
- reselect the preset
- narrate while the histogram loads

### Level 2 — app trouble
If the live UI is unstable:
- switch immediately to `docs/assets/aerovigil-demo.mp4`
- say: "This backup is generated from the same real inference flow you were about to see live."

### Level 3 — total environment failure
If the machine/projector setup collapses:
- present `docs/PITCH.md`
- show `docs/EXPLAINER.md`
- use the social card / release banner as still visuals
- keep the close simple: problem, solution, trust, ask
