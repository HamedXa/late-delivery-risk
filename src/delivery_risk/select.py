"""Compare candidate settings on the development months only.

The evaluation months are not touched here. Whatever wins is written into
model.SELECTED by hand, so the choice is visible in a diff.
"""

from __future__ import annotations

import json

import pandas as pd

from . import backtest, config, metrics, model


def compare(features: pd.DataFrame, stage: config.Stage) -> pd.DataFrame:
    """Average precision and recall on the development folds for each candidate."""
    rows = []

    def record(name: str, preds: pd.DataFrame, score_col: str = "score") -> None:
        monthly = metrics.by_month(preds, score_col)
        pooled = metrics.pooled(preds, score_col)
        rows.append(
            {
                "candidate": name,
                "average_precision_by_month": round(float(monthly["average_precision"].mean()), 4),
                "average_precision_pooled": pooled["average_precision"],
                "roc_auc_pooled": pooled["roc_auc"],
                "recall_at_budget": pooled["recall_at_budget"],
            }
        )

    preds = None
    for name, params in model.CANDIDATES.items():
        preds = backtest.walk_forward(
            features, stage, config.DEV_FOLDS, lambda params=params: model.make_gbm(stage, **params)
        )
        record(name, preds)
    narrow = config.narrow(stage)
    record(
        f"narrow feature set ({len(narrow.features)} inputs)",
        backtest.walk_forward(features, narrow, config.DEV_FOLDS, lambda: model.make_gbm(narrow)),
    )
    record("logistic", backtest.walk_forward(features, stage, config.DEV_FOLDS, lambda: model.make_logistic(stage)))
    record(f"rule ({stage.rule})", preds, "rule")
    return pd.DataFrame(rows)


def run(features_by_stage: dict[str, pd.DataFrame]) -> dict:
    out = {}
    for name, stage in config.STAGES.items():
        table = compare(features_by_stage[name], stage)
        out[name] = {"selected": model.SELECTED[name], "candidates": table.to_dict(orient="records")}
        print(f"\n{name}: development months {config.DEV_FOLDS[0]} to {config.DEV_FOLDS[-1]}")
        print(table.to_string(index=False))
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORT_DIR / "selection.json").write_text(json.dumps(out, indent=2))
    return out
