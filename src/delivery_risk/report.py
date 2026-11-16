"""Run the evaluation backtest and write the numbers and charts the README quotes."""

from __future__ import annotations

import json

import pandas as pd
from sklearn.inspection import permutation_importance

from . import backtest, config, metrics, model, monitor, plots


def evaluate_stage(features: pd.DataFrame, stage: config.Stage) -> dict:
    """Walk-forward results on the evaluation months for the model and its two baselines."""
    preds = backtest.walk_forward(features, stage, config.EVAL_FOLDS, lambda: model.make_gbm(stage))
    logistic = backtest.walk_forward(features, stage, config.EVAL_FOLDS, lambda: model.make_logistic(stage))
    narrow = config.narrow(stage)
    narrow_preds = backtest.walk_forward(features, narrow, config.EVAL_FOLDS, lambda: model.make_gbm(narrow))
    shuffled = backtest.random_split_scores(features, stage, config.EVAL_FOLDS, lambda: model.make_gbm(stage))
    shuffled = shuffled[shuffled["month"] == "in_window"]
    return {
        "preds": preds,
        "model": metrics.pooled(preds),
        "rule": metrics.pooled(preds, "rule"),
        "model_vs_rule": metrics.bootstrap_gap(preds),
        "logistic": metrics.pooled(logistic),
        "narrow_model": metrics.pooled(narrow_preds),
        "shuffled_split": metrics.summarize(shuffled["y"], shuffled["score"]),
        "by_month_model": metrics.by_month(preds),
        "by_month_rule": metrics.by_month(preds, "rule"),
    }


def handoff_importance(features: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """Permutation importance on the last evaluation month, with the model trained for that month."""
    stage = config.HANDOFF
    train, test = backtest.fold_split(features, stage, config.EVAL_FOLDS[-1])
    fitted = model.make_gbm(stage).fit(train[stage.features], train[config.TARGET])
    result = permutation_importance(
        fitted, test[stage.features], test[config.TARGET],
        scoring="average_precision", n_repeats=5, random_state=config.SEED,
    )  # fmt: skip
    table = pd.DataFrame({"feature": stage.features, "importance": result.importances_mean})
    return table.sort_values("importance", ascending=False).head(top).round(4)


def handoff_by_region(features: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """Alert rate, recall and precision by the customer's region."""
    region_of = {state: region for region, states in config.REGIONS.items() for state in states}
    customer_state = preds["order_id"].map(features.set_index("order_id")["customer_state"])
    return metrics.by_group(preds, customer_state.map(region_of).fillna("Unknown")).rename(columns={"group": "region"})


def run(features_by_stage: dict[str, pd.DataFrame], data_summary: dict | None = None) -> tuple[dict, dict]:
    """Evaluate both stages and write reports/.

    Returns the headline numbers and, per stage, the out-of-time score of every evaluated order.
    """
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    stages = {name: evaluate_stage(features_by_stage[name], stage) for name, stage in config.STAGES.items()}
    handoff, checkout = stages["handoff"], stages["checkout"]

    monitoring = monitor.replay(features_by_stage["handoff"], handoff["preds"], config.HANDOFF)
    importance = handoff_importance(features_by_stage["handoff"])

    results = {
        "data": data_summary or {},
        "evaluation_months": config.EVAL_FOLDS,
        "alert_budget": config.ALERT_BUDGET,
        "stages": {
            name: {k: s[k] for k in ("model", "rule", "model_vs_rule", "logistic", "narrow_model", "shuffled_split")}
            for name, s in stages.items()
        },
    }
    (config.REPORT_DIR / "metrics.json").write_text(json.dumps(results, indent=2))

    monthly = []
    for name, s in stages.items():
        both = s["by_month_model"].merge(
            s["by_month_rule"][["month", "average_precision", "roc_auc", "recall_at_budget"]],
            on="month", suffixes=("", "_rule"),
        )  # fmt: skip
        monthly.append(both.assign(stage=name))
    pd.concat(monthly).to_csv(config.REPORT_DIR / "backtest_by_month.csv", index=False)
    monitoring.to_csv(config.REPORT_DIR / "monitoring.csv", index=False)
    importance.to_csv(config.REPORT_DIR / "importance.csv", index=False)
    by_region = handoff_by_region(features_by_stage["handoff"], handoff["preds"])
    by_region.to_csv(config.REPORT_DIR / "by_region.csv", index=False)

    plots.recall_by_month(
        handoff["by_month_model"], handoff["by_month_rule"], checkout["by_month_model"],
        config.ALERT_BUDGET, config.FIGURE_DIR / "recall_by_month.png",
    )  # fmt: skip
    plots.score_vs_actual(monitoring, config.FIGURE_DIR / "score_vs_actual.png")
    plots.leak_check(
        [
            {
                "label": f"Scored at {name}",
                "walk_forward": s["model"]["roc_auc"],
                "random_split": s["shuffled_split"]["roc_auc"],
            }
            for name, s in stages.items()
        ],
        config.FIGURE_DIR / "leak_check.png",
    )
    plots.importance(importance, config.FIGURE_DIR / "importance.png")
    return results, {name: s["preds"] for name, s in stages.items()}
