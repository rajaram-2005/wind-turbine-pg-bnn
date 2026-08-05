"""Fleet onboarding demo: Reptile meta-training, then Hermes self-training.

Meta-trains a fleet initialization on historical turbines (Reptile), then
onboards a brand-new turbine from a handful of labeled windows plus a pool of
unlabeled telemetry — the few-shot fleet-onboarding path described in
docs/META_LEARNING.md.

Usage:
    python scripts/onboard_demo.py
    python scripts/onboard_demo.py --export artifacts/onboarding
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from src.agents.hermes import HermesAgent, HermesConfig  # noqa: E402
from src.data.synthetic import SyntheticConfig, generate  # noqa: E402
from src.meta.reptile import ReptileConfig, evaluate_few_shot, meta_train  # noqa: E402
from src.meta.tasks import task_from_telemetry  # noqa: E402
from src.models.bnn import BayesianNeuralNetwork  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    from src.utils.encoding import configure_utf8_stdio  # noqa: E402

    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--export",
        default=None,
        metavar="DIR",
        help="Export the meta model, the Hermes-adapted model, and the "
        "OnboardingReport (JSON) as serving bundles under DIR, then verify "
        "the adapted model loads through the Phase-2 serving path.",
    )
    args = parser.parse_args(argv)
    print("[1/5] Generating synthetic fleet (meta-train turbines + new asset) ...")
    cfg_syn = SyntheticConfig(n_turbines=10, seq_len=1200, seed=11)
    seqs = generate(cfg_syn)
    meta_seqs, holdout_seqs = seqs[:7], seqs[7:]

    meta_tasks = [
        task_from_telemetry(
            asset_id=f"meta-{i}", df=df, rul_end_days=rul,
            n_support=8, sample_interval_s=cfg_syn.sample_interval_s, seed=100 + i,
        )
        for i, (df, rul) in enumerate(meta_seqs)
    ]
    holdout_tasks = [
        task_from_telemetry(
            asset_id=f"holdout-{i}", df=df, rul_end_days=rul,
            n_support=8, sample_interval_s=cfg_syn.sample_interval_s, seed=200 + i,
        )
        for i, (df, rul) in enumerate(holdout_seqs)
    ]
    print(f"    meta-train tasks={len(meta_tasks)}  holdout tasks={len(holdout_tasks)}"
          f"  shots/task={meta_tasks[0].n_support}  feature_dim={meta_tasks[0].feature_dim}")

    reptile_cfg = ReptileConfig(
        inner_lr=8e-3, inner_steps=15, meta_lr=0.5, tasks_per_iter=4,
        meta_iterations=20, num_samples=2, eval_mc_samples=12, seed=3,
    )
    meta_model = BayesianNeuralNetwork(in_features=meta_tasks[0].feature_dim, hidden_sizes=(32, 16))

    print("[2/5] Few-shot evaluation of the RAW initialization on new assets ...")
    before = evaluate_few_shot(meta_model, holdout_tasks, reptile_cfg)
    print(f"    query RMSE = {before['mean_query_rmse_days']:.1f} days   "
          f"early-warning acc = {before['mean_early_warning_accuracy']:.3f}   "
          f"(naive support-mean baseline: {before['naive_mean_rmse_days']:.1f} days)")

    print("[3/5] Reptile meta-training over the fleet ...")
    meta_model, history = meta_train(meta_model, meta_tasks, reptile_cfg)
    losses = [it["mean_support_loss"] for it in history["iterations"]]
    print(f"    {len(losses)} meta-iterations  support ELBO {losses[0]:.3f} -> {losses[-1]:.3f}")

    print("[4/5] Few-shot evaluation of the META initialization on new assets ...")
    after = evaluate_few_shot(meta_model, holdout_tasks, reptile_cfg)
    print(f"    query RMSE = {after['mean_query_rmse_days']:.1f} days   "
          f"early-warning acc = {after['mean_early_warning_accuracy']:.3f}")
    gain = 100.0 * (before["mean_query_rmse_days"] - after["mean_query_rmse_days"]) \
        / before["mean_query_rmse_days"]
    print(f"    few-shot RMSE improvement from meta-training: {gain:+.1f}%")

    print("[5/5] Hermes onboarding of the newest asset (8 shots + unlabeled pool) ...")
    new = holdout_tasks[0]
    rng = np.random.default_rng(0)
    perm = rng.permutation(new.n_query)
    half = new.n_query // 2
    pool_x = new.query_x[perm[:half]]              # labels WITHHELD from the agent
    eval_x, eval_y = new.query_x[perm[half:]], new.query_y[perm[half:]]

    hermes = HermesAgent(meta_model, HermesConfig(adaptation=reptile_cfg, seed=3))
    adapted, report = hermes.onboard(
        asset_id="turbine-NEW-001",
        support_x=new.support_x, support_y=new.support_y,
        unlabeled_x=pool_x, eval_x=eval_x, eval_y=eval_y,
    )
    print(json.dumps(report.to_dict(), indent=2))

    if args.export:
        print(f"[export] Writing deployable bundles to {args.export} ...")
        from src.utils.artifacts import export_onboarding_bundle, save_model_bundle

        # The meta-initialization itself is the fleet asset (re-shared for every
        # new turbine); the adapted clone serves the onboarded asset.
        meta_bundle = save_model_bundle(
            meta_model,
            os.path.join(args.export, "meta_model.pt"),
            metadata={"produced_by": "scripts/onboard_demo.py", "kind": "reptile-meta"},
        )
        print(f"    meta model:          {meta_bundle.checkpoint_path}")

        paths = export_onboarding_bundle(
            adapted,
            report,
            os.path.join(args.export, "turbine-NEW-001"),
            extra_metadata={"produced_by": "scripts/onboard_demo.py"},
        )
        for k, p in paths.items():
            print(f"    {k:<20} {p}")

        # Deployment verification: both checkpoints must load through the
        # Phase-2 serving path and produce an advisory.
        from src.models.serving import load_serving_model
        from src.utils.schema import Telemetry, TurbinePayload

        holdout_df = holdout_seqs[0][0]
        snap = holdout_df.iloc[-1]
        payload = TurbinePayload(
            asset_id="turbine-NEW-001",
            telemetry=Telemetry(**{c: float(snap[c]) for c in holdout_df.columns if c != "timestamp"}),
        )
        for name, ckpt in (("meta", meta_bundle.checkpoint_path), ("adapted", paths["checkpoint"])):
            serving = load_serving_model(ckpt)
            rec = serving.advisory(payload, holdout_df.drop(columns=["timestamp"]))
            print(f"    [{name}] serving advisory OK: RUL={rec['predicted_rul_days']:.1f} d "
                  f"(epistemic σ={rec['epistemic_std']:.3f})")

    print("Done. Reminder: onboarding configures ADVISORY models only — no control path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
