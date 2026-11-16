"""Walk-forward backtest: train on the past, score the next month, move on.

The split is by time, and it respects when labels become known. An order bought
on the 28th and delivered on the 9th of the next month has no outcome yet when
that next month starts, so it is not in the training set for that month.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def fold_split(features: pd.DataFrame, stage: config.Stage, month: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training rows whose outcome was known before `month` began, and the rows scored in `month`."""
    start = pd.Timestamp(f"{month}-01")
    end = start + pd.offsets.MonthBegin(1)
    asof = features[stage.asof]
    train = features[(asof < start) & (features["delivered_ts"] < start)]
    test = features[(asof >= start) & (asof < end)]
    return train, test


def walk_forward(
    features: pd.DataFrame,
    stage: config.Stage,
    months: list[str],
    make_model: Callable[[], object],
) -> pd.DataFrame:
    """Out-of-time scores for every order in `months`. One model per month."""
    parts = []
    for month in months:
        train, test = fold_split(features, stage, month)
        model = make_model().fit(train[stage.features], train[config.TARGET])
        parts.append(
            pd.DataFrame(
                {
                    "order_id": test["order_id"].to_numpy(),
                    "month": month,
                    "n_train": len(train),
                    "y": test[config.TARGET].to_numpy(),
                    "score": model.predict_proba(test[stage.features])[:, 1],
                    "rule": test[stage.rule].to_numpy(),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def random_split_scores(
    features: pd.DataFrame, stage: config.Stage, months: list[str], make_model: Callable[[], object]
) -> pd.DataFrame:
    """The evaluation you should not trust: shuffle all orders up to the last month, hold out 20%.

    Orders from the same days land on both sides of the split, so the model
    sees the network conditions of the test orders during training. Kept here
    only to measure how much that flatters the result.
    """
    end = pd.Timestamp(f"{months[-1]}-01") + pd.offsets.MonthBegin(1)
    pool = features[features[stage.asof] < end]
    train, test = train_test_split(
        pool, test_size=0.2, random_state=config.SEED, stratify=pool[config.TARGET]
    )
    model = make_model().fit(train[stage.features], train[config.TARGET])
    return pd.DataFrame(
        {
            "y": test[config.TARGET].to_numpy(),
            "score": model.predict_proba(test[stage.features])[:, 1],
            "rule": test[stage.rule].to_numpy(),
            "month": np.where(
                test[stage.asof] >= pd.Timestamp(f"{months[0]}-01"), "in_window", "before_window"
            ),
        }
    )
