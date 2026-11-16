"""The time split, the alerting metrics and the drift check."""

from __future__ import annotations

import numpy as np
import pandas as pd

from delivery_risk import backtest, config, metrics, monitor


def test_fold_split_trains_only_on_outcomes_known_before_the_month(handoff_features):
    stage = config.HANDOFF
    train, test = backtest.fold_split(handoff_features, stage, "2018-04")
    start = pd.Timestamp("2018-04-01")

    assert len(train) > 50 and len(test) > 50
    assert (train["delivered_ts"] < start).all()
    assert (test[stage.asof] >= start).all() and (test[stage.asof] < pd.Timestamp("2018-05-01")).all()
    # Orders handed over in March but still in transit on April 1 are in neither set.
    in_transit = handoff_features[(handoff_features[stage.asof] < start) & (handoff_features["delivered_ts"] >= start)]
    assert len(in_transit) > 0
    assert not set(in_transit["order_id"]) & (set(train["order_id"]) | set(test["order_id"]))


def test_alerts_respect_the_budget_and_recall_counts_late_orders():
    y = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    score = np.array([0.9, 0.8, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    assert metrics.alert_mask(score, budget=0.2).sum() == 2
    assert metrics.recall_at_budget(y, score, budget=0.2) == 0.5  # one of two late orders is in the top two
    assert metrics.precision_at_budget(y, score, budget=0.2) == 0.5


def test_bootstrap_gap_is_zero_for_identical_rankings_and_positive_for_a_better_one():
    rng = np.random.default_rng(1)
    y = (rng.random(4000) < 0.1).astype(int)
    noise = rng.normal(0, 1, 4000)
    preds = pd.DataFrame({"month": np.repeat(["2018-04", "2018-05"], 2000), "y": y, "rule": y + noise})

    same = metrics.bootstrap_gap(preds.assign(score=preds["rule"]), resamples=50)
    better = metrics.bootstrap_gap(preds.assign(score=y + 0.3 * noise), resamples=200)

    assert same["recall_gap"] == 0 and same["interval_95_low"] == 0 and same["interval_95_high"] == 0
    assert better["recall_gap"] > 0 and better["interval_95_low"] > 0
    assert better["share_of_resamples_model_ahead"] == 1.0


def test_by_group_counts_alerts_per_group_after_budgeting_over_everyone():
    preds = pd.DataFrame(
        {
            "month": ["2018-04"] * 10,
            "y": [1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
            "score": [0.9, 0.8, 0.7, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1],
        }
    )
    group = pd.Series(["north"] * 5 + ["south"] * 5)

    table = metrics.by_group(preds, group).set_index("group")

    # A 10% budget over ten orders is one alert, and it lands on the top score, in the north.
    assert table.loc["north", "alerts"] == 1 and table.loc["south", "alerts"] == 0
    assert table.loc["north", "recall"] == 0.5 and table.loc["north", "precision"] == 1.0
    assert table.loc["south", "recall"] == 0.0


def test_psi_is_near_zero_for_the_same_distribution_and_large_after_a_shift():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 5000)

    assert monitor.psi(reference, rng.normal(0, 1, 5000)) < 0.05
    assert monitor.psi(reference, rng.normal(1.5, 1, 5000)) > monitor.PSI_ALERT
