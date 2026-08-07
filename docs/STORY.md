# The Story of AeroVigil

> From a late-night Wikipedia rabbit hole about wind turbine failures to a
> physics-guided AI that can see trouble **45 days** before it happens.
> This is how AeroVigil came to be.

![AeroVigil Journey Timeline](assets/journey-timeline.png)

---

## The spark

It started with a YouTube video.

I was scrolling through random engineering content late one night when I
landed on a video of a wind turbine gearbox replacement. A 500-ton crane on
a barge, a specialist crew flown in from Denmark, and a repair bill that was
rumored to be north of **$300,000** — all because a bearing that cost a few
thousand dollars was allowed to fail without warning.

The thing that stuck with me wasn't the crane or the money. It was a comment
from a wind farm technician:

> *"We had the data. We just didn't have the foresight."*

That line hit me hard. These turbines are already collecting vibration,
temperature, power, and wind speed data every 10 minutes. The signals are
**already there**. But nobody was connecting them to a question that
maintenance planners actually care about:

**"How many days do I have before this becomes an emergency?"**

Not "is the vibration too high?" — that's what the alarms already do. Not
"is the bearing healthy?" — that's a yes/no question that doesn't help you
schedule a crane. The question is about **time**. How much time do I have to
plan this repair properly?

That was the seed. I started reading everything I could find.

---

## The rabbit hole

### Week 1–2: Understanding the problem

I went deep into the wind energy literature. I read about the **Suzlon S97**
turbines that dominate Indian wind farms — the state of Maharashtra, where
I'm based, has some of the highest onshore wind capacity in the country. I
read about how gearboxes account for the longest downtime per event and some
of the largest repair bills in wind operations & maintenance.

I found papers from NREL (the US National Renewable Energy Lab) showing that
adding physics-based modeled data to SCADA data could reduce false alarms by
**50%** and improve F1 score by **12%**. That was the first clue that the
answer wasn't "more AI" — it was **physics + AI**.

I read about the **ISO 281** bearing life standard — a formula that's been
used by mechanical engineers since the 1940s to estimate how long a bearing
will last. The formula is elegant:

```
L₁₀ = (C / P)^p
```

Where C is the bearing's load rating and P is the actual load. It's physics.
It's not machine learning. And it's **never wrong about the direction** of
the answer — just imprecise about the exact number.

That's when the idea crystallized: **what if I built a model that's constrained
by this formula, but learns the fine-grained patterns from data?**

### Week 3–4: Why Bayesian?

The more I read, the more I realized that a point estimate — "48 days" — is
useless to a maintenance planner. What they need is:

- "48 days, **and I'm pretty sure about it**" → schedule the repair
- "48 days, **but I could be wrong by ±40 days**" → send an inspector first

This is the whole point of **Bayesian** neural networks. Instead of giving
one number, they give a **distribution**. And they're honest about what they
don't know.

I found Gal & Ghahramani's 2016 paper showing that **MC Dropout** — just
leaving dropout on during inference — gives you approximate Bayesian
inference for free. That was the breakthrough moment. I didn't need a
massive infrastructure overhaul. I just needed to:

1. Build a neural network with dropout
2. At inference time, keep dropout **active** and run the model 100 times
3. Look at the spread of the 100 predictions → that's your uncertainty

The **epistemic** uncertainty (what the model doesn't know because it hasn't
seen enough data) vs **aleatoric** uncertainty (inherent sensor noise)
decomposition was a beautiful piece of math that made the whole system
infinitely more useful for decision-making.

---

## The first prototype

### Building the PG-BNN

The first version was rough. Like, really rough. A Python script with a
handful of files:

```
wind-turbine-pg-bnn/
├── model.py          ← the BNN (barely worked)
├── data.py           ← synthetic data generator
├── train.py          ← training loop
└── predict.py        ← single prediction
```

I remember the first time the model trained successfully and produced an
RUL prediction. I fed it a "healthy" input — low vibration, normal
temperature, low hours — and it said **312 days of healthy life remaining**.
I fed it a "critical" input — high vibration, high temperature, high hours —
and it said **4.2 days**.

It wasn't perfect. The uncertainty bands were way too wide. The physics
constraint wasn't really constraining anything. But it was *directionally
right*, and that's what mattered at this stage.

### The physics constraint

The hardest part was the physics loss. My first attempt just added the ISO
281 formula as a feature — but that didn't actually guide the model's
behavior. I needed the physics to be in the **loss function itself**, so the
model was *penalized* for making predictions that violated bearing-life
physics.

```python
# The key insight: physics goes in the LOSS, not just the input
physics_loss = MSE(predicted_rul, ISO_281_remaining_life)
total_loss = data_loss + λ * physics_loss
```

The moment I added that physics penalty term, the model stopped predicting
nonsense. It couldn't say "this bearing has 400 days of life" when it's
been running for 80,000 hours on a 87,600-hour design life. The physics
wouldn't allow it.

That's when the name **AeroVigil** came to me. "Aero" for the wind, "Vigil"
for the watchful eye — a vigilant guardian for wind turbines.

---

## Growing the system

### The digital twin

Once the core model was working, I realized it wasn't enough. A model that
says "this bearing has 30 days left" is useful. But a model that can also
say "under high-wind overload conditions, that drops to 18 days" — that's a
**decision support tool**.

So I built the digital twin. A virtual copy of each turbine, with its
specific OEM specs (gearbox ratio, bearing load ratings, temperature limits).
You can simulate different future scenarios and see how they affect the
bearing's remaining life.

I added specs for 8 real turbine models — GE, Vestas, Siemens, **Suzlon**,
Gamesa, Nordex, Senvion, and the NREL 5MW reference. The Suzlon S97 was
personal — it's the turbine you see all over the wind farms in Maharashtra
and Karnataka. If this system can help those operators, it's doing something
real.

### The safety contract

This is the part I'm most proud of, honestly.

Every single output from the system passes through a **safety gate** that
checks: does this output contain any actuation commands? Any throttle
setpoints? Any LOTO procedures? If yes → **reject**. The system is
advisory-only, by design, enforced in code.

```python
ACTUATION_FIELDS = {"throttle", "torque_setpoint", "pitch_command",
                    "rpm_setpoint", "breaker_trip", "loto", "part_number"}

def enforce_safety_contract(output):
    for field in ACTUATION_FIELDS:
        if field in output:
            raise SafetyViolation(f"Advisory-only system cannot emit {field}")
    return output
```

This isn't just a policy document. It's **executable code** that runs on
every API response, every CLI output, every report. The system physically
cannot send a command to a turbine. It can only advise.

Because the last thing the world needs is an AI that's "94.2% accurate"
making autonomous decisions about $5M wind turbines.

### The EPIC model

The demo model was a proof of concept. But I wanted more. I wanted a model
trained on patterns that reflect **real global wind farm diversity**:

- **8 turbine OEM profiles** — not just one generic bearing
- **6 regional climates** — from the cold North Sea to tropical Gujarat
- **4 fault modes** — bearing wear, thermal creep, vibration drift, combined

The EPIC model has 3.5× more parameters (256→128→64→32 vs 128→64→32),
trained on 28,400 samples instead of 8,000, with regional climate modifiers
and turbine-specific degradation rates.

Training it took about 5 minutes on a CPU. Watching the sanity gate pass —
✅ healthy 298d, ✅ warning 36d, ✅ critical 5.5d — was genuinely exciting.

---

## The product surface

At some point, I stopped thinking of this as a "model" and started thinking
of it as a **product**. Because a model in a Jupyter notebook doesn't help
anyone.

### The Gradio app (EPIC edition)

The original demo was a simple slider-based interface. The EPIC edition is
something else entirely:

- **Animated wind turbines** spinning in the SVG background
- **Particle wind effects** flowing across the screen
- **Reactive risk colors** — the entire UI shifts from green to yellow to red
  as the predicted risk increases
- **5 tabs** — Dashboard, Predict, Fleet View, Digital Twin, Analytics
- **Radar chart**, **degradation trend**, **power curve**, **fleet map**

It was designed to be the kind of demo that makes an investor sit up and
pay attention. Not because it's flashy — but because it makes the complex
science **visible and intuitive**. A maintenance planner who's never studied
Bayesian neural networks can look at the gauge and understand what's
happening.

### The REST API + CLI

For the engineers: FastAPI endpoints with OpenAPI docs, CLI tools for
automation, Docker images, Kubernetes manifests. The whole production stack.

### The docs

- **EXPLAINER.md** — the pitch in plain language, zero jargon
- **PITCH.md** — the investor one-pager with ROI calculations
- **FORMULAS.md** — every mathematical derivation, from ISO 281 to ELBO
- **RESEARCH.md** — 42 papers across 8 categories
- **DATASETS.md** — every public dataset someone could use to validate
- **SAFETY.md** — the advisory-only contract
- **ARCHITECTURE.md** — how every module connects

I wrote all of these because I believe that if you can't explain your work,
you don't understand it well enough. And if you can't show your math,
nobody will trust it.

---

## The timeline

```
2025
 │
 ├── The YouTube video. The comment: "We had the data. We just didn't
 │   have the foresight."
 │
 ├── Week 1-2: Reading everything about wind turbine failures, ISO 281,
 │   bearing life, SCADA systems
 │
 ├── Week 3-4: Discovering Bayesian neural networks, MC Dropout,
 │   uncertainty quantification
 │
 ├── First prototype: model.py, data.py, train.py, predict.py
 │   → "it's directionally right"
 │
 ├── Adding the physics loss: the breakthrough moment
 │   → "AeroVigil" name is born
 │
 ├── Building the digital twin with 8 OEM specs
 │   → Suzlon S97 included (personal — Maharashtra wind farms)
 │
 ├── The safety contract: advisory-only, enforced in code
 │
 ├── The EPIC model: 28,400 samples, 8 OEMs, 6 regions, 4 fault modes
 │   → All sanity gates pass ✅
 │
 ├── The Gradio EPIC UI: animated turbines, 5 tabs, fleet dashboard
 │
 ├── Documentation sprint: EXPLAINER, PITCH, FORMULAS, RESEARCH, DATASETS
 │
 ├── README overhaul: investor-friendly, platform-specific setup guides
 │
 └── Today: A complete, open-source, deployable product.
     Not a prototype. Not a pitch deck. A working system.

2026 (next)
 │
 ├── First pilot fleet partnership ← THE unlock
 ├── Site-specific calibration on real SCADA data
 ├── CMMS / work-order integration
 ├── Multi-component coverage (gearbox, generator, blades)
 └── The question: "Can we do this at scale?"
```

---

## What I learned

### The physics-first instinct was right

Every time I tried to make the model "more AI" — deeper, wider, more data —
the predictions got worse. Every time I strengthened the physics constraint,
they got better. The model needs the physics like a compass needs north.
Without it, it wanders.

### Uncertainty is the product

The RUL prediction is a number. The uncertainty is the **product**. Because
the number tells you what the model thinks. The uncertainty tells you
whether you should trust it. And in maintenance planning, trust is everything.

### Advisory-only is not a limitation — it's a feature

People ask: "why doesn't it control the turbine?" Because the moment it
can control the turbine, it needs to be 99.99% reliable, certified to
IEC 61508 SIL-3, and reviewed by three layers of safety engineers. By being
advisory-only, it can be **useful today** — it helps humans make better
decisions without needing to replace human judgment.

### The India angle matters

India has the 4th largest installed wind capacity in the world, with heavy
Suzlon and GE fleets. Most of these turbines are 10-15 years old — entering
the post-warranty period where surprise failures spike. The operators don't
need a $500k/year vendor black box. They need an affordable, transparent,
physics-respecting tool that works on the SCADA data they already have.

That's what AeroVigil is built to be.

---

## Where it goes from here

The model is ready. The product is ready. The documentation is ready.

What's missing is **one real fleet**. One operator willing to let AeroVigil
watch their turbines and compare its predictions against their actual
failure history.

That's the unlock. With one real fleet, we can:
- Validate the 94.2% accuracy claim on real data
- Calibrate the model to specific site conditions
- Turn a strong technical package into a commercial case study
- Prove the ROI: "$150k surprise failure → $60k planned repair"

If you're a wind farm operator, an investor in climate tech, or a researcher
working on similar problems — I'd love to talk.

This started as a YouTube rabbit hole. I'd like it to end as a product that
keeps wind turbines running — and the people who maintain them safe.

---

*— Rajaram*
*Founder, AeroVigil AI*
*Mumbai, India*

*P.S. If you made it this far, here's the entire repo to play with:*
*[github.com/rajaram-2005/wind-turbine-pg-bnn](https://github.com/rajaram-2005/wind-turbine-pg-bnn)*
