# AeroVigil investor one-pager

## The problem

Wind turbines fail expensively when drivetrain issues are discovered too late.

A single surprise bearing event can cost **$150,000–$300,000** once you include:

- crane mobilization
- specialist labor
- expedited parts
- lost generation

Across the industry, wind operations and maintenance is a **multi-billion-dollar** spend category. Operators do not need another dashboard full of raw sensor charts. They need an earlier, more credible answer to one question:

**Which turbine needs intervention soon enough that we can plan the repair instead of reacting to a failure?**

---

## The solution

**AeroVigil** is a physics-guided Bayesian AI for wind turbine bearing health.

It turns standard SCADA telemetry into a plain-language maintenance advisory:

- estimated healthy life remaining
- uncertainty range
- risk band
- recommended planning urgency

The product's promise is simple: **be the check-engine light for wind turbines, ~45 days before failure.**

---

## Why we win: the moat

| Moat element | Why it matters |
|---|---|
| **ISO 281 physics guidance** | Grounds the model in real bearing-life intuition instead of pure curve fitting |
| **MCVI uncertainty** | Shows confidence spread, which makes the output more usable for high-consequence planning |
| **Fail-closed safety contract** | The system is built advisory-only; it screens outputs before they leave the boundary |
| **Full product surface** | Not just a notebook: FastAPI, CLI, Streamlit, Gradio demo, digital twin, Hermes few-shot onboarding, and AeroZip telemetry compression |

---

## Proof already in the package

| Signal | Demo proof point |
|---|---|
| Early warning accuracy | **94.2%** |
| Recall at the warning horizon | **100%** |
| Typical error band | **±4–6 days** |
| Planning horizon | **45 days** |
| Deployment posture | **Open source + offline-capable + advisory-only** |

What officers and operators see immediately:

- a live UI with healthy / warning / critical scenarios
- a narrated demo video built from real model inference
- offline local weights so the demo still works with no internet
- documentation for engineers, executives, and stage presenters

---

## Business model

AeroVigil can monetize as a staged reliability product:

1. **Pilot deployment** per fleet / site
2. **Annual platform subscription** for advisory analytics and model updates
3. **Integration services** for SCADA / CMMS / reporting workflows
4. **Premium modules** for fleet benchmarking, digital twin scenarios, and compressed telemetry workflows

The path to revenue is practical: start with the bearing use case, prove avoided surprise failures, then expand to broader drivetrain reliability workflows.

---

## Near-term roadmap

### 0–3 months
- secure one pilot fleet
- validate site-specific calibration
- integrate SCADA export / ingestion path

### 3–6 months
- fleet-level reporting
- alert routing into maintenance workflows
- operator-specific dashboards and role-based summaries

### 6–12 months
- expanded component coverage
- cross-site benchmarking
- workflow integrations with CMMS / OEM service processes

---

## The ask

**One pilot fleet.**

That is the next unlock.

With one real operating fleet, AeroVigil can:

- calibrate against site history
- prove avoided surprise-failure economics
- turn a strong technical package into a commercial case study

---

## Why now

The market does not need more generic AI. It needs reliability tools that are:

- easy to explain
- trustworthy under uncertainty
- safe by design
- deployable even with weak connectivity

AeroVigil already looks and feels like a product, not just a model artifact.

**The pitch is not “we built a neural network.” The pitch is “we help operators see expensive failures early enough to do something cheaper.”**
