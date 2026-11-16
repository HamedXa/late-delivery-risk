"""Drift checks: has the input data moved, and do scores still match outcomes?

PSI (population stability index) compares two distributions of one feature.
A common reading: under 0.1 stable, 0.1 to 0.25 worth a look, over 0.25 shifted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import backtest, config

PSI_ALERT = 0.25


def decile_edges(reference: np.ndarray) -> np.ndarray:
    """Inner bin edges from the reference data's deciles."""
    values = np.asarray(reference, dtype=float)
    return np.unique(np.nanquantile(values, np.linspace(0.1, 0.9, 9)))


def bin_shares(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Share of values in each bin, with one extra bin for missing values."""
    values = np.asarray(values, dtype=float)
    missing = np.isnan(values)
    counts = np.bincount(np.searchsorted(edges, values[~missing], side="right"), minlength=len(edges) + 1)
    counts = np.append(counts, missing.sum())
    return counts / max(counts.sum(), 1)


def psi(reference: np.ndarray, current: np.ndarray) -> float:
    """Population stability index between a reference sample and a current sample."""
    edges = decile_edges(reference)
    floor = 1e-4  # avoids log(0) when a bin is empty on one side
    ref = np.clip(bin_shares(reference, edges), floor, None)
    cur = np.clip(bin_shares(current, edges), floor, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def feature_drift(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str]) -> pd.Series:
    """PSI per numeric feature, largest first."""
    scores = {c: psi(reference[c].to_numpy(), current[c].to_numpy()) for c in columns}
    return pd.Series(scores).sort_values(ascending=False).round(3)


def replay(features: pd.DataFrame, preds: pd.DataFrame, stage: config.Stage) -> pd.DataFrame:
    """Replay the evaluation months as if the monitor had been running.

    For each month: how far the inputs moved from the data the model was
    trained on, and how the average score compared with the late rate that
    was later observed.
    """
    rows = []
    for month, part in preds.groupby("month", sort=True):
        train, test = backtest.fold_split(features, stage, month)
        recent = train[
            train[stage.asof] >= train[stage.asof].max() - pd.Timedelta(days=config.REFERENCE_DAYS)
        ]
        # Network and state features move together for every order in a month, so a
        # distribution test on them always fires. Drift is measured on the order's own inputs.
        own_inputs = [c for c in stage.numeric if not config.is_context(c)]
        drift = feature_drift(recent, test, own_inputs)
        rows.append(
            {
                "month": month,
                "orders": int(len(part)),
                "mean_score": round(float(part["score"].mean()), 4),
                "actual_late_rate": round(float(part["y"].mean()), 4),
                "most_drifted_feature": drift.index[0],
                "max_psi": float(drift.iloc[0]),
                "features_over_alert": int((drift > PSI_ALERT).sum()),
            }
        )
    return pd.DataFrame(rows)
