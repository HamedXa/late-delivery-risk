"""Metrics for a rare-event alerting problem."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from . import config


def alert_mask(score: np.ndarray, budget: float = config.ALERT_BUDGET) -> np.ndarray:
    """True for the riskiest `budget` share of orders."""
    n_alerts = max(int(round(len(score) * budget)), 1)
    mask = np.zeros(len(score), dtype=bool)
    mask[np.argsort(-np.asarray(score), kind="stable")[:n_alerts]] = True
    return mask


def recall_at_budget(y, score, budget: float = config.ALERT_BUDGET) -> float:
    """Share of late orders caught when alerting on the riskiest `budget` of orders."""
    y = np.asarray(y)
    return float(y[alert_mask(score, budget)].sum() / max(y.sum(), 1))


def precision_at_budget(y, score, budget: float = config.ALERT_BUDGET) -> float:
    """Share of alerts that turn out late."""
    return float(np.asarray(y)[alert_mask(score, budget)].mean())


def summarize(y, score) -> dict[str, float]:
    """The handful of numbers reported everywhere in this project."""
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    base = float(y.mean())
    recall = recall_at_budget(y, score)
    return {
        "orders": int(len(y)),
        "late_rate": round(base, 4),
        "average_precision": round(float(average_precision_score(y, score)), 4),
        "roc_auc": round(float(roc_auc_score(y, score)), 4),
        "recall_at_budget": round(recall, 4),
        "precision_at_budget": round(precision_at_budget(y, score), 4),
        "lift_at_budget": round(recall / config.ALERT_BUDGET, 2),
    }


def by_month(preds: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    """One row of metrics per backtest month. Alerts are budgeted within each month."""
    rows = []
    for month, part in preds.groupby("month", sort=True):
        row = {"month": month, **summarize(part["y"], part[score_col])}
        if score_col == "score" and part["score"].between(0, 1).all():  # probabilities only
            row["mean_score"] = round(float(part["score"].mean()), 4)
            row["brier"] = round(float(brier_score_loss(part["y"], part["score"])), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_gap(preds: pd.DataFrame, resamples: int = 1000, seed: int = config.SEED) -> dict[str, float]:
    """How much of the model's lead over the rule could be luck in which orders came in?

    Each resample draws every month's orders again with replacement, issues
    both sets of alerts on the redrawn month, and records the share of late
    orders caught by the model minus the share caught by the rule. Both
    methods always see the same redrawn orders, so the comparison is paired.
    """
    rng = np.random.default_rng(seed)
    months = [part[["y", "score", "rule"]].to_numpy(dtype=float) for _, part in preds.groupby("month", sort=True)]
    gaps = np.empty(resamples)
    for i in range(resamples):
        caught_by_model = caught_by_rule = late = 0.0
        for month in months:
            drawn = month[rng.integers(0, len(month), len(month))]
            y = drawn[:, 0]
            caught_by_model += y[alert_mask(drawn[:, 1])].sum()
            caught_by_rule += y[alert_mask(drawn[:, 2])].sum()
            late += y.sum()
        gaps[i] = (caught_by_model - caught_by_rule) / max(late, 1.0)
    low, high = np.quantile(gaps, [0.025, 0.975])
    observed = pooled(preds)["recall_at_budget"] - pooled(preds, "rule")["recall_at_budget"]
    return {
        "recall_gap": round(float(observed), 4),
        "interval_95_low": round(float(low), 4),
        "interval_95_high": round(float(high), 4),
        "share_of_resamples_model_ahead": round(float((gaps > 0).mean()), 3),
        "resamples": resamples,
    }


def by_group(preds: pd.DataFrame, group: pd.Series) -> pd.DataFrame:
    """How the alerts fall across groups of orders, and how well they work in each.

    Alerts are budgeted month by month over all orders, as in the backtest, and
    then counted per group. `group` is aligned with `preds` row by row.
    """
    alert = np.zeros(len(preds), dtype=bool)
    for rows in preds.groupby("month").indices.values():
        alert[rows] = alert_mask(preds["score"].to_numpy()[rows])
    table = pd.DataFrame({"group": group.to_numpy(), "late": preds["y"].to_numpy(), "alert": alert})
    table["caught"] = table["late"] * table["alert"]
    out = table.groupby("group").agg(
        orders=("late", "size"), late_orders=("late", "sum"), alerts=("alert", "sum"), caught=("caught", "sum")
    )
    out["late_rate"] = (out["late_orders"] / out["orders"]).round(4)
    out["alert_rate"] = (out["alerts"] / out["orders"]).round(4)
    out["recall"] = (out["caught"] / out["late_orders"].clip(lower=1)).round(4)
    out["precision"] = (out["caught"] / out["alerts"].clip(lower=1)).round(4)
    return out.sort_values("orders", ascending=False).reset_index()


def pooled(preds: pd.DataFrame, score_col: str = "score") -> dict[str, float]:
    """Metrics over all months together, with the alert budget applied month by month."""
    out = summarize(preds["y"], preds[score_col])
    monthly = by_month(preds, score_col)
    weights = monthly["orders"] * monthly["late_rate"]
    recall = float((monthly["recall_at_budget"] * weights).sum() / max(weights.sum(), 1e-9))
    alerts = float(
        (monthly["precision_at_budget"] * monthly["orders"]).sum() / monthly["orders"].sum()
    )
    out.update(
        recall_at_budget=round(recall, 4),
        precision_at_budget=round(alerts, 4),
        lift_at_budget=round(recall / config.ALERT_BUDGET, 2),
    )
    return out
