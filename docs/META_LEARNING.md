# Few-Shot Fleet Onboarding — Reptile Meta-Learning + the Hermes Self-Training Agent

> **One-line summary:** a new turbine joins the fleet with eight labeled
> failure windows and a week of unlabeled SCADA — Reptile gives the model a
> fleet-wise initialization, and Hermes self-trains on the unlabeled pool,
> promotes itself only through a fail-closed gate, and stays advisory-only
> the whole time.

---

## 1. The problem: the cold start of every new asset

The PG-BNN early-warning system works because it has seen many turbines
degrade and fail. A *newly commissioned or newly instrumented* turbine has
none of that history: no failure labels, maybe a few inspection outcomes, and
a stream of unlabeled SCADA telemetry. The conventional options are all bad:

- **Wait** months for labeled data to accumulate — the exact window in which
  infant-mortality drivetrain faults appear.
- **Retrain from scratch** on a handful of labels — overfits instantly.
- **Apply the fleet model as-is** — site- and asset-specific bias goes
  uncorrected, and the epistemic uncertainty (correctly) explodes.

Few-shot fleet onboarding is the alternative: learn **how to adapt** from the
turbines the fleet already knows, then adapt to the newcomer with very little
data.

## 2. Reptile meta-learning over the fleet (`src/meta/reptile.py`)

Reptile (Nichol, Achiam & Schulman, 2018; arXiv:1803.02999) is a first-order
meta-learning algorithm. Each existing turbine is a *task*
(`src/meta/tasks.AdaptationTask`): a small **support** set of labeled feature
windows (what a new asset would actually have) and a larger **query** set used
only to score adaptation.

```
repeat meta_iterations times:
    sample a batch of tasks (turbines)
    for each task k:
        θ_k = inner_steps of Adam(φ) on support_k      # inner loop
    φ ← φ + ε · mean_k(θ_k − φ)                        # meta-update (ε = meta_lr)
```

The meta-initialization φ is not itself a good model — it is a good *starting
point*: a few Adam steps on a new turbine's shots land at an accurate,
well-uncalibrated-to-the-fleet model instead of wandering from a random or
fleet-biased init. The inner loop reuses the same ELBO objective
(`src/models/bnn.elbo_loss`), so meta-training preserves the interpretable
uncertainty decomposition (epistemic vs. aleatoric) the early-warning system
depends on.

```python
from src.meta.reptile import ReptileConfig, meta_train, evaluate_few_shot
from src.meta.tasks import tasks_from_synthetic_fleet
from src.models.bnn import BayesianNeuralNetwork

tasks = tasks_from_synthetic_fleet(n_support=8)          # one task per turbine
meta = BayesianNeuralNetwork(in_features=tasks[0].feature_dim, hidden_sizes=(64, 64))
meta, history = meta_train(meta, tasks, ReptileConfig(meta_iterations=25))
report = evaluate_few_shot(meta, holdout_tasks, ReptileConfig())  # RMSE + 45-day acc
```

## 3. Hermes: the self-training onboarding agent (`src/agents/hermes.py`)

Hermes runs the onboarding playbook for a single new asset:

1. **ADAPT** — few-shot adaptation of the fleet meta-model on the asset's
   labeled shots. The meta-model is never mutated; Hermes works on a clone.
2. **SELF-TRAIN** — classic confidence-filtered pseudo-labeling: predict on
   the asset's unlabeled window pool, keep windows whose **epistemic σ ≤ τ
   days** (`confidence_tau_days`), add `(window, predicted RUL)` to the
   training set, re-adapt, repeat for at most `max_rounds`. Pseudo-labels are
   clipped to the physical RUL range and every round is capped
   (`max_pseudo_per_round`) so self-training cannot flood the few real shots.
3. **GATE** — fail-closed promotion. The adapted model is evaluated on a
   labeled evaluation split it never adapted on. Promotion requires
   `RMSE ≤ promotion_max_rmse_days` **and** 45-day early-warning accuracy
   `≥ promotion_min_accuracy`, with at least `min_eval_shots` evaluation
   windows. If anything is missing or below threshold, the asset stays in
   **shadow mode**: its report says exactly why and what data to collect, and
   its advisories require human review.

The only output is an `OnboardingReport` — an advisory-only payload screened
by the same `enforce_safety_contract` as every other system output, so the
agent can never smuggle a control field upstream.

```python
from src.agents.hermes import HermesAgent, HermesConfig

agent = HermesAgent(meta_model, HermesConfig())
model, report = agent.onboard(
    asset_id="turbine-NEW-001",
    support_x=shots_x, support_y=shots_y,        # the few labeled windows
    unlabeled_x=pool_x,                           # unlabeled telemetry windows
    eval_x=eval_x, eval_y=eval_y,                 # labeled validation windows
)
print(report.status, report.eval_rmse_days)       # "promoted" | "shadow"
```

## 4. Safety posture

- **Advisory-only, end to end.** Meta-training and self-training touch model
  *weights* and feature tensors only. Promoted models feed the same
  `run_advisory()` pipeline as any other model; the same safety gate screens
  the `OnboardingReport`.
- **Fail-closed.** No eval split, not enough eval shots, missed RMSE budget,
  missed accuracy floor → shadow mode, no exceptions.
- **Bounded self-training.** Hard caps on rounds and per-round pseudo-labels,
  an uncertainty threshold in physical units (RUL days), and label clipping.
- **Fleet model protection.** Adaptation always operates on deep copies; the
  shared meta-initialization cannot be corrupted by a single asset's bad data.

## 5. Demo

```bash
python scripts/onboard_demo.py
```

Meta-trains on a synthetic 7-turbine fleet, compares few-shot query RMSE and
45-day early-warning accuracy before/after meta-training on 3 held-out
turbines, then runs Hermes end-to-end on the newest asset and prints the
onboarding report as JSON.

## 6. Limitations (research scaffold)

- Reptile is a heuristic approximation to gradient-based meta-learning; it
  shares little formal grounding with full MAML but is far cheaper and stable
  with MCVI noise.
- Pseudo-label quality is only as good as the epistemic estimate; a
  miscalibrated model can pass its own confidence test. The promotion gate —
  not the confidence filter — is the real guard; keep `min_eval_shots` and the
  thresholds strict in production.
- Task construction assumes per-turbine normalization; cross-site fleets
  should build tasks per site (`task_from_telemetry`) rather than pooling
  raw channels.
