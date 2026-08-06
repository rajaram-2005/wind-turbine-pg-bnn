# AeroVigil in plain language

## The $300k surprise bearing

A wind turbine bearing is supposed to be boring.

But inside every turbine, that bearing spins **roughly a billion times** over its working life. When it fails with no warning, the repair is not a small service call. It can mean:

- a giant crane booking
- a specialist crew
- weeks of lost generation
- a repair bill that can climb into the **$150,000–$300,000** range

AeroVigil is built to prevent that kind of surprise.

In one sentence: **it works like a wind turbine's check-engine light, but it turns on about 45 days early so humans can schedule the repair instead of reacting to a breakdown.**

---

## What AeroVigil looks at

AeroVigil reads the signals many turbines already have in their SCADA system:

1. vibration
2. main bearing temperature
3. generator temperature
4. power output
5. wind speed
6. total run hours

It does **not** require a fancy new sensor pack for the demo.

---

## How it works in 5 steps

### 1) Read the turbine's vital signs
The model takes six operating signals that already tell a story about wear, heat, load, and age.

### 2) Compare this turbine to patterns it has learned
The AI has been trained to connect those signals to how much healthy life is likely left in the drivetrain bearing.

### 3) Check that the answer still makes physical sense
This is not a pure black box. AeroVigil is **physics-guided**, meaning it is designed around bearing-life logic inspired by ISO 281 rather than free-form guessing.

### 4) Run the model many times, not once
Instead of giving one overconfident answer, AeroVigil runs the model repeatedly and measures the spread.

- tight spread = the model is more sure
- wide spread = the model is less sure

That spread is the system being honest about uncertainty.

### 5) Turn the prediction into a maintenance planning signal
The output is shown as **days of healthy life remaining** with a caution threshold at **45 days**.

- **45+ days:** healthy / routine monitoring
- **14–45 days:** schedule maintenance
- **under 14 days:** urgent action

---

## Demo scoreboard

| Demo metric | What it means |
|---|---|
| **94.2% accuracy** | In the demo campaign, AeroVigil usually got the early-warning call right |
| **100% recall** | In the demo campaign, it did not miss a turbine that was truly in the warning window |
| **±4–6 days** | Typical prediction error band shown in the demo materials |
| **45-day horizon** | The main planning line: enough time to organize a repair instead of a rescue |

---

## What AeroVigil will NOT do

This is important.

AeroVigil is **advisory only**. It will **not**:

- control the turbine
- change torque, pitch, or RPM
- issue lockout/tagout instructions
- replace an engineer's judgment
- replace inspection, vibration review, oil analysis, or OEM procedures

It tells humans: **"this machine looks healthy"** or **"this machine is moving into a danger window"**.

Humans decide what to do next.

---

## Why this matters operationally

The difference between 4 days and 45 days is not just accuracy. It is logistics.

With 45 days of warning, an operator can:

- group work with other planned maintenance
- order parts normally instead of expediting
- reserve crane time before the market gets tight
- reduce lost generation
- avoid emergency-response pricing

That is why AeroVigil is framed as a planning tool, not just an AI model.

---

## The simple takeaway

AeroVigil watches for the subtle combination of shaking, heat, and age that suggests a turbine bearing is quietly running out of life.

If the model thinks the turbine still has lots of runway left, it shows green.
If the runway is collapsing toward the 45-day line, it warns you early.
If the turbine is close to failure, it says so clearly.

**See the failure before it happens — and schedule the repair before it becomes a rescue.**
