"""Fit the model that gets served, and save everything the API needs next to it."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from . import config, model, monitor
from .features import haversine_km


def recent_rows(features: pd.DataFrame, stage: config.Stage, days: int = config.REFERENCE_DAYS) -> pd.DataFrame:
    """Orders scored in the last `days` days of the data."""
    asof = features[stage.asof]
    return features[asof >= asof.max() - pd.Timedelta(days=days)]


def route_medians(rows: pd.DataFrame, column: str = "route_transit_days_60d") -> dict[str, float]:
    """Median of one input for each seller state > customer state pair."""
    route = rows["seller_state"] + ">" + rows["customer_state"]
    return rows.groupby(route)[column].median().round(2).to_dict()


def reference_scores(recent: pd.DataFrame, stage: config.Stage, fitted, backtest: pd.DataFrame | None):
    """Scores of recent orders, from backtest models that had not seen them when possible.

    The served model has seen the outcome of every recent order, so its own
    scores on them run high for the late ones. An alert threshold taken from
    those scores would be too strict for new orders. Without a backtest (the
    unit tests) there is nothing better than the model's own training rows.
    """
    if backtest is not None:
        unseen = backtest.loc[backtest["order_id"].isin(recent["order_id"]), "score"]
        if len(unseen):
            return unseen.to_numpy(), "walk-forward backtest"
    return fitted.predict_proba(recent[stage.features])[:, 1], "training rows"


def build_reference(
    features: pd.DataFrame, stage: config.Stage, fitted, backtest: pd.DataFrame | None = None
) -> dict:
    """Typical recent values, the score distribution and drift baselines.

    The API uses these to fill inputs the caller leaves out, to say how a score
    ranks against recent orders, and to explain a score against a typical order.
    """
    recent = recent_rows(features, stage)
    latest = recent_rows(features, stage, config.CONDITIONS_DAYS)
    scores, scores_from = reference_scores(recent, stage, fitted, backtest)
    centroids = pd.concat(
        [
            recent[["customer_state", "customer_lat", "customer_lng"]].set_axis(["state", "lat", "lng"], axis=1),
            recent[["seller_state", "seller_lat", "seller_lng"]].set_axis(["state", "lat", "lng"], axis=1),
        ]
    ).groupby("state")[["lat", "lng"]].mean()
    state_inputs = [c for c in stage.numeric if c.startswith("state_")]
    context = [c for c in stage.numeric if config.is_context(c)]
    own_inputs = [c for c in stage.numeric if c not in context]
    # A typical order comes from six months of orders. Conditions come from the last two.
    typical = {**recent[own_inputs].median().to_dict(), **latest[context].median().to_dict()}
    return {
        "stage": stage.name,
        "features": stage.features,
        "alert_budget": config.ALERT_BUDGET,
        "alert_threshold": float(np.quantile(scores, 1 - config.ALERT_BUDGET)),
        "scores_from": scores_from,
        "score_percentiles": np.quantile(scores, np.linspace(0, 1, 101)).round(5).tolist(),
        "typical": {name: round(float(typical[name]), 4) for name in stage.numeric},
        "by_state": latest.groupby("customer_state")[state_inputs].median().round(4).to_dict(orient="index"),
        # Routes with no parcels in the last two months keep their six-month value.
        "route_transit_days": {**route_medians(recent), **route_medians(latest)},
        "route_distance_km": route_medians(recent, "distance_km"),
        "state_centroids": centroids.round(3).to_dict(orient="index"),
        "rule_threshold": float(np.quantile(recent[stage.rule].dropna(), 1 - config.ALERT_BUDGET)),
        "drift_edges": {c: monitor.decile_edges(recent[c].to_numpy()).round(4).tolist() for c in own_inputs},
        "trained_through": str(features[stage.asof].max().date()),
        "training_rows": int(len(features)),
    }


def route_distance_km(reference: dict, seller_state: str, customer_state: str) -> float | None:
    """A distance for callers who do not send one.

    The median distance of recent orders on the same route. For a pair of
    states with no recent orders, the distance between the two state centres.
    """
    known = reference["route_distance_km"].get(f"{seller_state}>{customer_state}")
    if known is not None:
        return float(known)
    points = reference["state_centroids"]
    if seller_state not in points or customer_state not in points:
        return None
    a, b = points[seller_state], points[customer_state]
    return float(haversine_km(a["lat"], a["lng"], b["lat"], b["lng"]))


def fit_and_save(
    features: pd.DataFrame, stage: config.Stage = config.HANDOFF, backtest: pd.DataFrame | None = None
) -> dict:
    """Train on every order with a known outcome and write the artifacts.

    `backtest` is the walk-forward output for this stage (order_id and score).
    """
    fitted = model.make_gbm(stage).fit(features[stage.features], features[config.TARGET])
    reference = build_reference(features, stage, fitted, backtest)
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted, config.ARTIFACT_DIR / f"{stage.name}_model.joblib")
    (config.ARTIFACT_DIR / f"{stage.name}_reference.json").write_text(json.dumps(reference, indent=1))
    return reference
