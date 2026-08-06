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
- Run `python gradio_app/app.py`
- Test all three presets: healthy, warning, critical
- Confirm the app reports **Weights source: local artifacts/pg_bnn_demo**
- Confirm the critical preset lands around **~4 days** mean RUL
- Keep `docs/assets/aerovigil-demo.mp4` ready as backup

### 10 minutes before stage time
- Open the app and leave it on the main prediction view
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
| 1 | Choose **Healthy turbine** | "This is a machine that looks normal." | Green / low-risk result |
| 2 | Press **Predict the 45-day outlook** | "The model says there's lots of healthy runway left." | Gauge stays high; histogram mostly right of 45-day line |
| 3 | Point at histogram width | "This spread is the model being honest about uncertainty." | Uncertainty is visible, not hidden |
| 4 | Choose **Critical: act now** | "Same turbine family, now after months of hidden degradation." | Sliders jump to harsher values |
| 5 | Press predict again | "Now the estimate collapses into the red zone." | Red badge, low days remaining |
| 6 | Point to yellow line | "Anything left of 45 days means schedule the repair now, not later." | Planning threshold is intuitive |
| 7 | Open the accordion | "If someone is not technical, the UI still teaches them how to read it." | UX feels productized |

### Key callouts during the live demo
- **Healthy -> green** means the machine still has runway.
- **Critical -> red** means the machine is inside an urgent response window.
- **Distribution spread = honesty**. Wider histogram = lower certainty.
- **45-day line** is the maintenance-planning trigger.
- **Accordion** proves the UI is explainable, not just technical.

---

## 45-second trust section

> "There are four reasons to trust the output more than a generic black-box score. First, it is physics-guided, so the model is shaped by bearing-life intuition. Second, it shows uncertainty instead of pretending to know everything. Third, it is safety-gated and advisory-only. And fourth, the demo scoreboard is strong: 94.2 percent early-warning accuracy, 100 percent recall, and typical error in the plus-or-minus four to six day range." 

---

## 45-second engineering credibility section

> "This is not just a slide deck model. The repo already includes the model package, FastAPI endpoints, CLI tools, a digital twin workflow, telemetry compression, onboarding logic, and this offline-capable Gradio demo. Even the narrated video is rendered from real model inference, not canned screenshots." 

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
The demo still works. We train and load local weights from `artifacts/pg_bnn_demo` and the Gradio app prefers those automatically.

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
