"""Run everything: raw CSVs in, model, metrics and charts out.

    python -m delivery_risk.pipeline --raw-dir data/raw
"""

from __future__ import annotations

import argparse
import time

from . import config, data, features, report, select, tracking, train


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=str(config.RAW_DIR), help="folder holding the Olist CSV files")
    parser.add_argument("--skip-selection", action="store_true", help="skip the development-month comparison")
    parser.add_argument("--mlflow", action="store_true", help="also log the run to MLflow (needs mlflow installed)")
    args = parser.parse_args(argv)

    started = time.time()
    raw = data.load_raw(args.raw_dir)
    orders = data.build_orders(raw)
    summary = {"raw_orders": int(len(raw["orders"])), **data.label_summary(orders)}
    print(f"{summary['orders']:,} delivered orders, {summary['late_orders']:,} late ({summary['late_rate']:.1%})")

    features_by_stage = {name: features.build_features(orders, stage) for name, stage in config.STAGES.items()}
    if not args.skip_selection:
        select.run(features_by_stage)

    results, backtest_scores = report.run(features_by_stage, summary)
    reference = train.fit_and_save(features_by_stage["handoff"], config.HANDOFF, backtest_scores["handoff"])
    if args.mlflow:
        tracking.log_results(results)

    print(f"\nEvaluation months {config.EVAL_FOLDS[0]} to {config.EVAL_FOLDS[-1]}, alerting on the riskiest 10%")
    for name, stage_results in results["stages"].items():
        for who in ("model", "rule", "logistic", "narrow_model", "shuffled_split"):
            m = stage_results[who]
            print(
                f"  {name:8s} {who:14s} AP {m['average_precision']:.3f}  AUC {m['roc_auc']:.3f}  "
                f"recall {m['recall_at_budget']:.1%}  precision {m['precision_at_budget']:.1%}"
            )
    gap = results["stages"]["handoff"]["model_vs_rule"]
    print(
        f"\nHandoff model against the slack rule: {gap['recall_gap'] * 100:+.1f} points of late orders caught, "
        f"95% interval {gap['interval_95_low'] * 100:+.1f} to {gap['interval_95_high'] * 100:+.1f} "
        f"({gap['resamples']} paired resamples)"
    )
    print(f"\nServing model trained on {reference['training_rows']:,} orders through {reference['trained_through']}")
    print(f"Alert threshold {reference['alert_threshold']:.4f}, set from {reference['scores_from']} scores")
    print(f"Wrote {config.REPORT_DIR} and {config.ARTIFACT_DIR} in {time.time() - started:.0f}s")
    return results


if __name__ == "__main__":
    main()
