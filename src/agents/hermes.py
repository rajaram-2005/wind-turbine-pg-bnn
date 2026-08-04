"""Hermes — the self-training fleet-onboarding agent (advisory-only).

When a new turbine joins the fleet it arrives with, at best, a handful of
labeled failure windows. Hermes bootstraps an advisory model for it:

1. **ADAPT** — few-shot adaptation of the fleet meta-model (the Reptile
   initialization from `src/meta/reptile.py`) on the asset's labeled shots.
2. **SELF-TRAIN** — pseudo-label the asset's unlabeled telemetry windows
   where the model's own epistemic uncertainty is low enough (σ ≤ τ days),
   then re-adapt on labeled + pseudo-labeled data; repeat for a bounded
   number of rounds.
3. **GATE** — a fail-closed promotion gate evaluates the adapted model on a
   labeled evaluation split it never adapted on. Only if RMSE and 45-day
   early-warning accuracy clear their thresholds is the model promoted to
   advisory duty; otherwise the asset stays in "shadow" mode and the report
   tells the human what to collect next.

Hermes never actuates. Its only output is an `OnboardingReport` — screened by
the same safety contract (`src/utils/safety.enforce_safety_contract`) as every
other system output — and the adapted model is a *clone*: the fleet
meta-model is never mutated.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field

import numpy as np
import torch

from src.eval.calibration import early_warning_metrics
from src.meta.reptile import ReptileConfig, few_shot_adapt
from src.models.bnn import BayesianNeuralNetwork, predict
from src.utils.safety import enforce_safety_contract


RUL_CLIP_MAX_DAYS = 3650.0  # matches utils.schema.BNNState upper bound


@dataclass
class HermesConfig:
    adaptation: ReptileConfig = field(default_factory=ReptileConfig)
    confidence_tau_days: float = 40.0   # pseudo-label only if epistemic σ ≤ τ (RUL days)
    max_rounds: int = 4
    max_pseudo_per_round: int = 32
    promotion_max_rmse_days: float = 120.0
    promotion_min_accuracy: float = 0.8
    min_eval_shots: int = 4
    eval_mc_samples: int = 16
    rul_clip_days: float = RUL_CLIP_MAX_DAYS
    seed: int = 0


@dataclass(frozen=True)
class OnboardingReport:
    """Decision-support outcome of an onboarding run.

    Informational only: no setpoints, no commands. `to_dict()` passes through
    the advisory-only safety contract before it can leave the system.
    """

    asset_id: str
    status: str                                # "promoted" | "shadow"
    promoted: bool
    rounds_completed: int
    n_labeled_shots: int
    n_pseudo_labels: int
    eval_rmse_days: float | None
    eval_early_warning_accuracy: float | None
    promotion_thresholds: dict[str, float]
    rationale: str
    advisory_only: bool = True
    generated_at: str = field(
        default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        payload = {
            "asset_id": self.asset_id,
            "status": self.status,
            "promoted": bool(self.promoted),
            "rounds_completed": int(self.rounds_completed),
            "n_labeled_shots": int(self.n_labeled_shots),
            "n_pseudo_labels": int(self.n_pseudo_labels),
            "eval_rmse_days": None if self.eval_rmse_days is None else float(self.eval_rmse_days),
            "eval_early_warning_accuracy": (
                None if self.eval_early_warning_accuracy is None else float(self.eval_early_warning_accuracy)
            ),
            "promotion_thresholds": {k: float(v) for k, v in self.promotion_thresholds.items()},
            "rationale": self.rationale,
            "advisory_only": True,
            "generated_at": self.generated_at,
            "disclaimer": (
                "Decision-support only. Hermes configures advisory models and "
                "issues no control commands; promotion decisions must be "
                "reviewed by a qualified reliability engineer."
            ),
        }
        return enforce_safety_contract(payload)


def select_confident(
    pred: dict[str, torch.Tensor],
    tau_days: float,
    k: int,
) -> np.ndarray:
    """Indices of up to `k` pseudo-label candidates whose epistemic σ ≤ τ,
    ordered most-confident-first. `pred` is the output of `src.models.bnn.predict`.
    """
    epi = pred["epistemic_std"].numpy()
    order = np.argsort(epi, kind="stable")
    sel = [int(i) for i in order if float(epi[i]) <= tau_days][:k]
    return np.asarray(sel, dtype=np.int64)


class HermesAgent:
    """Onboards one new asset at a time against a fleet meta-model."""

    def __init__(self, meta_model: BayesianNeuralNetwork, cfg: HermesConfig | None = None):
        self.meta_model = meta_model
        self.cfg = cfg or HermesConfig()

    # ------------------------------------------------------------------
    def onboard(
        self,
        asset_id: str,
        support_x: np.ndarray,
        support_y: np.ndarray,
        unlabeled_x: np.ndarray | None = None,
        eval_x: np.ndarray | None = None,
        eval_y: np.ndarray | None = None,
    ) -> tuple[BayesianNeuralNetwork, OnboardingReport]:
        """Run ADAPT → SELF-TRAIN → GATE for one asset.

        Returns (adapted model clone, OnboardingReport). The fleet meta-model
        is never mutated; nothing is actuated.
        """
        cfg = self.cfg
        torch.manual_seed(cfg.seed)

        support_x = np.asarray(support_x, dtype=np.float32)
        support_y = np.asarray(support_y, dtype=np.float32).ravel()
        if support_x.ndim != 2:
            raise ValueError("support_x must be a 2-D array of feature windows")
        if support_y.shape[0] < 1:
            raise ValueError("Hermes needs at least one labeled shot to onboard")
        if support_x.shape[0] != support_y.shape[0]:
            raise ValueError("support_x and support_y must contain the same number of samples")
        feature_dim = support_x.shape[1]

        pool = None
        if unlabeled_x is not None:
            pool = np.asarray(unlabeled_x, dtype=np.float32)
            if pool.ndim != 2 or pool.shape[1] != feature_dim:
                raise ValueError("unlabeled_x must be (n, d) with the same feature dim as support_x")

        eval_available = False
        ex = ey = None
        n_eval = 0
        if eval_x is not None and eval_y is not None:
            ex = np.asarray(eval_x, dtype=np.float32)
            ey = np.asarray(eval_y, dtype=np.float32).ravel()
            if ex.ndim != 2 or ex.shape[0] != ey.shape[0] or ex.shape[1] != feature_dim:
                raise ValueError("eval_x/eval_y shapes must agree and match the support feature dim")
            n_eval = int(ey.shape[0])
            eval_available = n_eval >= cfg.min_eval_shots

        # ---- 1. ADAPT (few-shot) --------------------------------------
        train_x, train_y = support_x.copy(), support_y.copy()
        adapted, _ = few_shot_adapt(self.meta_model, train_x, train_y, cfg.adaptation)

        # ---- 2. SELF-TRAIN (bounded pseudo-labeling rounds) ------------
        rounds = 0
        n_pseudo = 0
        if pool is not None and pool.shape[0] > 0:
            for _ in range(cfg.max_rounds):
                if pool.shape[0] == 0:  # pool exhausted by the previous round
                    break
                pred = predict(adapted, torch.tensor(pool), mc_samples=cfg.eval_mc_samples)
                idx = select_confident(pred, cfg.confidence_tau_days, cfg.max_pseudo_per_round)
                if idx.size == 0:
                    break
                pseudo_y = np.clip(pred["mean_pred"].numpy(), 0.0, cfg.rul_clip_days)[idx].astype(np.float32)
                train_x = np.concatenate([train_x, pool[idx]], axis=0)
                train_y = np.concatenate([train_y, pseudo_y], axis=0)
                keep = np.ones(pool.shape[0], dtype=bool)
                keep[idx] = False
                pool = pool[keep]
                n_pseudo += int(idx.size)
                adapted, _ = few_shot_adapt(self.meta_model, train_x, train_y, cfg.adaptation)
                rounds += 1

        # ---- 3. GATE (fail-closed promotion) ---------------------------
        rmse: float | None = None
        acc: float | None = None
        reasons: list[str] = []
        if not eval_available:
            reasons.append(
                f"insufficient labeled evaluation data ({n_eval}/{cfg.min_eval_shots} shots): "
                "promotion requires human-scored validation windows"
            )
        else:
            out = predict(adapted, torch.tensor(ex), mc_samples=cfg.eval_mc_samples)
            pred_y = out["mean_pred"].numpy()
            rmse = float(np.sqrt(np.mean((pred_y - ey) ** 2)))
            acc = float(early_warning_metrics(ey, pred_y)["accuracy"])
            if rmse > cfg.promotion_max_rmse_days:
                reasons.append(
                    f"RMSE {rmse:.1f} d exceeds promotion budget {cfg.promotion_max_rmse_days:.1f} d"
                )
            if acc < cfg.promotion_min_accuracy:
                reasons.append(
                    f"early-warning accuracy {acc:.3f} below promotion floor "
                    f"{cfg.promotion_min_accuracy:.3f}"
                )
        promoted = eval_available and not reasons
        status = "promoted" if promoted else "shadow"

        rationale_parts = [
            f"Hermes onboarding for asset '{asset_id}': few-shot adaptation on "
            f"{support_y.shape[0]} labeled shot(s), {rounds} self-training round(s) "
            f"adding {n_pseudo} pseudo-label(s) (epistemic σ ≤ {cfg.confidence_tau_days:.1f} d)."
        ]
        if rmse is not None:
            rationale_parts.append(
                f"Held-out evaluation: RMSE {rmse:.1f} d, 45-day early-warning accuracy {acc:.3f}."
            )
        if promoted:
            rationale_parts.append(
                "Promotion gate PASSED — model promoted to advisory duty. A reliability "
                "engineer must still review each advisory before any maintenance action."
            )
        else:
            rationale_parts.append(
                "Promotion gate BLOCKED — asset remains in shadow mode: " + "; ".join(reasons) + ". "
                "Collect more labeled inspection outcomes and re-run onboarding; advisories "
                "from this asset require human review."
            )
        rationale_parts.append("Advisory only — Hermes issues no control commands.")

        report = OnboardingReport(
            asset_id=asset_id,
            status=status,
            promoted=promoted,
            rounds_completed=rounds,
            n_labeled_shots=int(support_y.shape[0]),
            n_pseudo_labels=n_pseudo,
            eval_rmse_days=rmse,
            eval_early_warning_accuracy=acc,
            promotion_thresholds={
                "max_rmse_days": cfg.promotion_max_rmse_days,
                "min_early_warning_accuracy": cfg.promotion_min_accuracy,
                "min_eval_shots": float(cfg.min_eval_shots),
                "confidence_tau_days": cfg.confidence_tau_days,
            },
            rationale=" ".join(rationale_parts),
        )
        return adapted, report
