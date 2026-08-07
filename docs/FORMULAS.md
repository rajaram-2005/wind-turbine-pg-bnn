# AeroVigil — Mathematical Formulations & Derivations

> Complete mathematical foundation for the Physics-Guided Bayesian Neural
> Network (PG-BNN). Each section includes definitions, derivations, and
> their connection to the implementation.

---

## Table of Contents

1. [ISO 281 Bearing Life Model](#1-iso-281-bearing-life-model)
2. [Bayesian Neural Network Architecture](#2-bayesian-neural-network-architecture)
3. [Variational Inference & ELBO](#3-variational-inference--elbo)
4. [Monte Carlo Variational Inference (MCVI)](#4-monte-carlo-variational-inference-mcvi)
5. [Physics-Guided Loss Function](#5-physics-guided-loss-function)
6. [Uncertainty Decomposition](#6-uncertainty-decomposition)
7. [Risk Classification](#7-risk-classification)
8. [Feature Normalization](#8-feature-normalization)

---

## 1. ISO 281 Bearing Life Model

### 1.1 Basic Rating Life (L₁₀)

The ISO 281 standard defines the **basic rating life** L₁₀ as the number of
million revolutions that 90% of a group of identical bearings will complete or
exceed before fatigue failure.

$$
L_{10} = \left(\frac{C}{P}\right)^p
$$

Where:
- **L₁₀** = basic rating life (million revolutions)
- **C** = basic dynamic load rating (kN) — from bearing manufacturer tables
- **P** = equivalent dynamic bearing load (kN)
- **p** = life exponent: p = 3 for ball bearings, p = 10/3 for roller bearings

### 1.2 Derivation from Hertzian Contact Stress

The life exponent arises from the relationship between contact stress and
fatigue crack propagation:

1. **Hertzian contact stress** for a rolling element on a raceway:

$$
\sigma_H \propto \frac{F}{(R_1 \cdot R_2)^{1/2}}
$$

where F is the contact force and R₁, R₂ are principal radii of curvature.

2. **Lundberg-Palmgren theory** (1947) relates stress to life probability:

$$
\ln\left(\frac{1}{S}\right) \propto \frac{\sigma_H^c \cdot V^e}{z^h}
$$

where S is survival probability, V is stressed volume, z is track depth,
and c, e, h are material constants.

3. **Simplification** for constant load gives:

$$
L_{10} \propto \frac{1}{P^p} \quad \text{with } p = \frac{c}{e}
$$

For modern bearings: c ≈ 10/3, giving p = 3 (ball) or p = 10/3 (roller).

### 1.3 Adjusted Rating Life (ISO 281:2007)

The **adjusted** rating life accounts for lubrication, contamination, and
fatigue load limit:

$$
L_{nm} = a_1 \cdot a_{ISO} \cdot L_{10}
$$

Where:
- **a₁** = reliability factor (a₁ = 1 for 90% reliability)
- **a_ISO** = life modification factor from lubrication and contamination

In AeroVigil, we use a simplified operating-hours version:

$$
L_{10,\text{hours}} = \frac{10^6 \cdot L_{10}}{60 \cdot n}
$$

where n is the rotational speed in RPM.

### 1.4 Operating Hours Constraint

AeroVigil's physics constraint simplifies to an operating-hours fraction:

$$
\text{Physics Reference} = L_{10,\text{ref}} - h_{\text{op}}
$$

where:
- L_{10,ref} = 87,600 hours (= 10 years of continuous operation)
- h_op = current operating hours

This creates a **soft constraint**: the model's predicted RUL should be
physically consistent with the remaining fraction of the bearing's design life.

---

## 2. Bayesian Neural Network Architecture

### 2.1 Bayesian Layer Definition

Each Bayesian layer replaces deterministic weights with **probability
distributions**. Instead of a fixed weight w, we have:

$$
w \sim q_\theta(w) = \mathcal{N}(\mu_w, \sigma_w^2)
$$

The forward pass for layer l with Bayesian weights:

$$
z^{(l)} = W^{(l)} x^{(l-1)} + b^{(l)}
$$

where W^(l) and b^(l) are sampled from their variational distributions.

### 2.2 AeroVigil Architecture

```
Input (6 features)
    │
    ▼
Bayesian Linear(6 → 256) + ReLU + Dropout(0.15)     ← Epic model
Bayesian Linear(256 → 128) + ReLU + Dropout(0.15)
Bayesian Linear(128 → 64) + ReLU + Dropout(0.15)
Bayesian Linear(64 → 32) + ReLU + Dropout(0.15)
    │
    ├──→ Output Head 1: rul_mean    (linear, 32 → 1)
    └──→ Output Head 2: rul_log_var (linear, 32 → 1)
```

The **dual output heads** parameterize a Gaussian predictive distribution:

$$
p(y \mid x, \theta) = \mathcal{N}(\mu_{\text{rul}}, \exp(\log \sigma^2_{\text{rul}}))
$$

### 2.3 Mean-Field Variational Family

The variational posterior factorizes across all parameters:

$$
q_\theta(\mathbf{w}) = \prod_{i} \mathcal{N}(w_i \mid \mu_i, \sigma_i^2)
$$

This is the **mean-field approximation** — it assumes independence between
weights, making the KL divergence computable in closed form.

---

## 3. Variational Inference & ELBO

### 3.1 The Bayesian Objective

Given training data D = {(x_i, y_i)}_{i=1}^N, we want to compute the
posterior over weights:

$$
p(\mathbf{w} \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \mathbf{w}) \, p(\mathbf{w})}{p(\mathcal{D})}
$$

The evidence p(D) is intractable (requires integrating over all possible weight
values), so we use **variational inference** to approximate the posterior with
a simpler distribution q_θ(w).

### 3.2 Evidence Lower Bound (ELBO) Derivation

We minimize the KL divergence between q and the true posterior:

$$
\text{KL}(q_\theta(\mathbf{w}) \,\|\, p(\mathbf{w} \mid \mathcal{D}))
= \int q_\theta(\mathbf{w}) \log \frac{q_\theta(\mathbf{w})}{p(\mathbf{w} \mid \mathcal{D})} d\mathbf{w}
$$

Expand using Bayes' rule:

$$
= \int q_\theta(\mathbf{w}) \log \frac{q_\theta(\mathbf{w}) \cdot p(\mathcal{D})}{p(\mathcal{D} \mid \mathbf{w}) \cdot p(\mathbf{w})} d\mathbf{w}
$$

$$
= \int q_\theta(\mathbf{w}) \left[\log q_\theta(\mathbf{w}) - \log p(\mathcal{D} \mid \mathbf{w}) - \log p(\mathbf{w}) + \log p(\mathcal{D})\right] d\mathbf{w}
$$

Since log p(D) is a constant with respect to w:

$$
= \log p(\mathcal{D}) - \left[\int q_\theta(\mathbf{w}) \log \frac{p(\mathcal{D} \mid \mathbf{w}) \, p(\mathbf{w})}{q_\theta(\mathbf{w})} d\mathbf{w}\right]
$$

Therefore:

$$
\log p(\mathcal{D}) = \text{KL}(q_\theta \,\|\, p) + \underbrace{\mathbb{E}_{q_\theta}[\log p(\mathcal{D} \mid \mathbf{w})] - \text{KL}(q_\theta(\mathbf{w}) \,\|\, p(\mathbf{w}))}_{\text{ELBO}}
$$

Since KL ≥ 0, we get:

$$
\boxed{\text{ELBO} \leq \log p(\mathcal{D})}
$$

**Maximizing the ELBO** minimizes the KL to the true posterior, giving the
tightest possible variational approximation.

### 3.3 ELBO in Practice

The ELBO decomposes into:

$$
\mathcal{L}_{\text{ELBO}} = \underbrace{\mathbb{E}_{q_\theta}[\log p(\mathcal{D} \mid \mathbf{w})]}_{\text{Expected log-likelihood}} - \underbrace{\beta \cdot \text{KL}(q_\theta(\mathbf{w}) \,\|\, p(\mathbf{w}))}_{\text{Complexity penalty}}
$$

For Gaussian likelihood p(y|x,w) = N(f_w(x), σ²):

$$
\mathbb{E}_{q_\theta}[\log p(\mathcal{D} \mid \mathbf{w})] = -\frac{N}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{N}(y_i - f_{\mathbf{w}}(x_i))^2
$$

For mean-field Gaussian prior p(w) = N(0, I):

$$
\text{KL}(q_\theta \,\|\, p) = \frac{1}{2}\sum_{j}\left[\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1\right]
$$

### 3.4 Implementation in AeroVigil

```python
# Loss = -ELBO = NLL + β·KL
loss = gaussian_nll(pred_mean, pred_log_var, y_true) + β * kl_divergence
```

where β = 0.01 (ELBO weight, acts as a temperature parameter controlling the
trade-off between data fit and prior regularization).

---

## 4. Monte Carlo Variational Inference (MCVI)

### 4.1 Core Idea

After training, we have the variational posterior q_θ*(w). At test time,
instead of using the mean weights (which would give a point estimate), we:

1. Sample weights from the posterior: w^(k) ~ q_θ*(w)
2. Run a forward pass: ŷ^(k) = f(x; w^(k))
3. Repeat K times to get a distribution of predictions

### 4.2 Connection to MC Dropout

Gal & Ghahramani (2016) showed that **MC Dropout** is mathematically equivalent
to approximate variational inference in a deep Gaussian process:

$$
\text{MC Dropout at test time} \approx \text{Sampling from } q_\theta(\mathbf{w})
$$

**Proof sketch:**
- Dropout masks the activation: h^(l) = h^(l-1) · diag(m) where m ~ Bernoulli(1-p)
- This is equivalent to sampling a sub-network
- Over many forward passes with different masks, the ensemble of sub-networks
  approximates the posterior predictive distribution

### 4.3 Predictive Distribution

The **Monte Carlo estimate** of the posterior predictive:

$$
p(y \mid x, \mathcal{D}) \approx \frac{1}{K}\sum_{k=1}^{K} p(y \mid x, \mathbf{w}^{(k)})
$$

Point estimate (mean RUL):

$$
\hat{\mu}_{\text{rul}} = \frac{1}{K}\sum_{k=1}^{K} \hat{y}^{(k)}
$$

Predictive variance:

$$
\hat{\sigma}^2_{\text{rul}} = \frac{1}{K}\sum_{k=1}^{K} (\hat{y}^{(k)} - \hat{\mu}_{\text{rul}})^2
$$

### 4.4 Confidence Intervals

The 95% credible interval from the MC samples:

$$
\text{CI}_{95\%} = \left[\text{Percentile}_{2.5}(\{\hat{y}^{(k)}\}), \; \text{Percentile}_{97.5}(\{\hat{y}^{(k)}\})\right]
$$

### 4.5 AeroVigil Implementation

```python
def run_inference(model, x, n_samples=100):
    model.train()  # keep dropout ACTIVE for MCVI
    predictions = []
    with torch.no_grad():
        for _ in range(n_samples):
            mean, _ = model(x)
            predictions.append(mean.item())
    return np.array(predictions)  # distribution of K predictions
```

---

## 5. Physics-Guided Loss Function

### 5.1 Total Loss

The PG-BNN training loss combines three components:

$$
\boxed{\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{NLL}} + \beta \cdot \mathcal{L}_{\text{KL}} + \lambda \cdot \mathcal{L}_{\text{physics}}}
$$

Where:
- L_NLL = Gaussian negative log-likelihood (data fit)
- L_KL = KL divergence to prior (model complexity)
- L_physics = Physics constraint loss (ISO 281 grounding)
- β = 0.01 (ELBO temperature)
- λ = 0.1 (physics weight)

### 5.2 Gaussian Negative Log-Likelihood

The model outputs μ (mean) and log σ² (log-variance). The NLL for a single
sample:

$$
\mathcal{L}_{\text{NLL}} = \frac{1}{2}\left[\log(2\pi) + \log\sigma^2 + \frac{(y - \mu)^2}{\sigma^2}\right]
$$

**Intuition:** The model learns to predict not just the RUL but also how
uncertain it is. When the model is uncertain (large σ²), the penalty for a
wrong prediction is lower — this naturally calibrates uncertainty.

### 5.3 KL Divergence (Mean-Field Gaussian)

$$
\mathcal{L}_{\text{KL}} = \frac{1}{2}\sum_{j=1}^{D}\left[\mu_j^2 + \sigma_j^2 - \log(\sigma_j^2) - 1\right]
$$

This penalizes weights that drift too far from the standard normal prior.

### 5.4 Physics Loss (ISO 281 Constraint)

The physics loss enforces that the model's predictions are consistent with
bearing-life physics:

$$
\mathcal{L}_{\text{physics}} = \text{MSE}\left(\hat{y}, \max(L_{10,\text{ref}} - h_{\text{op}}, 0)\right)
$$

Where:
- L_{10,ref} = 87,600 hours (design life reference)
- h_op = operating hours from input features
- The MSE is between the model's RUL prediction and the physics-based
  remaining life estimate

**Why this works:** The physics term acts as a regularizer that prevents the
model from predicting RUL values that are physically impossible given the
bearing's design life and current operating hours.

---

## 6. Uncertainty Decomposition

### 6.1 Epistemic Uncertainty (Model Uncertainty)

Epistemic uncertainty captures uncertainty about the **model parameters** —
what the model doesn't know due to limited data:

$$
\sigma^2_{\text{epistemic}} = \text{Var}_{\mathbf{w} \sim q_\theta}\left[\mathbb{E}[y \mid x, \mathbf{w}]\right]
$$

In MCVI: estimated by the variance across the K stochastic forward passes.

**Properties:**
- Decreases with more training data
- Can be reduced by collecting more data
- Higher in out-of-distribution regions

### 6.2 Aleatoric Uncertainty (Data Uncertainty)

Aleatoric uncertainty captures noise inherent in the **observations** —
sensor noise, natural variability:

$$
\sigma^2_{\text{aleatoric}} = \mathbb{E}_{\mathbf{w} \sim q_\theta}\left[\text{Var}[y \mid x, \mathbf{w}]\right]
$$

In AeroVigil: read directly from the model's log-variance output head.

**Properties:**
- Cannot be reduced by collecting more data
- Intrinsic to the measurement process
- Related to sensor quality and environmental variability

### 6.3 Total Predictive Uncertainty

$$
\sigma^2_{\text{total}} = \sigma^2_{\text{epistemic}} + \sigma^2_{\text{aleatoric}}
$$

In AeroVigil's MCVI with K samples:

```python
predictions = [model_forward(x) for _ in range(K)]  # each sample from posterior
epistemic = np.var(predictions)                       # variance of means
aleatoric = np.mean([model.log_var for ...])          # mean of predicted variances
total = epistemic + aleatoric
```

---

## 7. Risk Classification

### 7.1 Decision Boundaries

AeroVigil maps continuous RUL predictions to discrete risk levels using
clinically-inspired thresholds:

$$
\text{risk} = \begin{cases}
\text{CRITICAL} & \text{if } \hat{\mu}_{\text{rul}} < 14 \text{ days} \\
\text{HIGH} & \text{if } 14 \leq \hat{\mu}_{\text{rul}} < 30 \text{ days} \\
\text{MODERATE} & \text{if } 30 \leq \hat{\mu}_{\text{rul}} < 45 \text{ days} \\
\text{LOW} & \text{if } \hat{\mu}_{\text{rul}} \geq 45 \text{ days}
\end{cases}
$$

### 7.2 Justification of Thresholds

| Threshold | Rationale |
|-----------|-----------|
| **45 days** | Minimum lead time for planned maintenance: crane scheduling, parts procurement, crew availability |
| **30 days** | Transition from "plan ahead" to "act soon" — parts ordering deadline for most bearing suppliers |
| **14 days** | Emergency threshold — unplanned intervention required; crane mobilization at premium rates |

### 7.3 Uncertainty-Weighted Risk

In practice, AeroVigil also considers the **confidence interval** in its risk
assessment. If the lower bound of the 95% CI falls below a threshold while the
mean is above it, the risk is elevated:

$$
\text{risk}_{\text{adjusted}} = \begin{cases}
\text{elevate by 1 level} & \text{if } \text{CI}_{\text{lower}} < \text{threshold} \text{ and } \hat{\mu} > \text{threshold} \\
\text{unchanged} & \text{otherwise}
\end{cases}
$$

---

## 8. Feature Normalization

### 8.1 Z-Score Normalization

All input features are z-score normalized before being fed to the model:

$$
x_i^{\text{normalized}} = \frac{x_i - \mu_i}{\sigma_i}
$$

where μ_i and σ_i are the mean and standard deviation computed from the
training set (stored in `scaler.npz`).

### 8.2 Why Normalization Matters for BNNs

1. **Bayesian priors assume unit scale** — the N(0,1) prior on weights works
   best when inputs are centered and scaled.
2. **Gradient flow** — prevents some features from dominating the loss gradient
   due to magnitude differences (e.g., operating hours ~50,000 vs vibration ~10).
3. **Uncertainty calibration** — the model's uncertainty estimates are more
   reliable when inputs are on a consistent scale.

### 8.3 Feature Statistics (from training data)

| Feature | Typical Range | Unit | Description |
|---------|---------------|------|-------------|
| vibration_rms | 2 – 40 | mm/s | Drive-train vibration RMS |
| bearing_temp | 30 – 130 | °C | Main bearing temperature |
| generator_temp | 40 – 170 | °C | Generator winding temperature |
| power_output | 0 – 5200 | kW | Active power generation |
| wind_speed | 0 – 25 | m/s | Nacelle anemometer wind speed |
| operating_hours | 0 – 87,600 | hours | Cumulative operating time |

---

## Appendix A: Loss Landscape & Optimization

### A.1 Learning Rate Schedule

AeroVigil uses **cosine annealing** with warm restart:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t \cdot \pi}{T}\right)\right)
$$

Where:
- η_max = 2×10⁻³ (Epic), 3×10⁻³ (Demo)
- η_min = 1×10⁻⁵
- T = total epochs (350 Epic, 250 Demo)

### A.2 Gradient Clipping

To prevent exploding gradients in the Bayesian layers:

$$
\mathbf{g} \leftarrow \mathbf{g} \cdot \min\left(1, \frac{\tau}{\|\mathbf{g}\|_2}\right)
$$

where τ = 1.0 (clip threshold).

---

## Appendix B: Synthetic Teaching Rule (Demo Model)

The demo model uses a synthetic teaching rule to generate training targets:

$$
y = 430 - 5.2 \cdot \max(\text{vib} - 5, 0)^{1.15} - 2.2 \cdot \max(\text{temp}_{\text{bearing}} - 55, 0)^{1.10} - 1.1 \cdot \max(\text{temp}_{\text{gen}} - 70, 0)^{1.05} - \frac{h_{\text{op}}}{87600} \cdot 210 + \epsilon
$$

where ε ~ N(0, σ²) with σ = 5.0 days.

This rule captures the physical intuition:
- **Higher vibration** → more degradation → less life remaining
- **Higher temperatures** → accelerated wear → less life remaining
- **More operating hours** → closer to end of design life

The power-law exponents (>1) create **non-linear acceleration** — degradation
gets progressively worse as the signals increase, matching real bearing behavior.

---

*This document provides the mathematical backbone of AeroVigil. For implementation
details, see the source code in [`src/aerovigil_pg_bnn/`](../src/aerovigil_pg_bnn/).*

*Last updated: August 2026*
