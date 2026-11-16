"""Features known at the moment an order is scored.

One rule runs through this module: a feature for an order scored at time t may
only use things that had already happened before t. Another order's outcome
counts only once that order has been delivered. tests/test_features.py checks
the rule by rewriting the future and asserting that earlier features do not move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

DAY = 86_400  # seconds
PRIOR_LATE, PRIOR_N = 1.0, 15.0  # weak starting belief: about 7% of orders are late
SELLER_SMOOTHING = 20.0  # a seller needs about 20 known outcomes to outweigh the network rate
WINDOW_SMOOTHING = 10.0
MIN_ROUTE_ORDERS = 5  # below this a route average is too noisy, so fall back to the network


def to_seconds(timestamps: pd.Series) -> np.ndarray:
    """Timestamps as whole seconds since 1970, so they sort and compare as plain integers."""
    return pd.to_datetime(timestamps).to_numpy(dtype="datetime64[s]").astype("int64")


def days_between(start: pd.Series, end: pd.Series) -> pd.Series:
    return (end - start).dt.total_seconds() / DAY


def haversine_km(lat1, lng1, lat2, lng2) -> np.ndarray:
    """Great-circle distance in kilometres between two sets of points."""
    lat1, lng1, lat2, lng2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lng1, lat2, lng2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def sum_before(event_times: np.ndarray, values: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    """For each query time, the sum of `values` over events strictly before it.

    Pass an array of ones as `values` to count events.
    """
    order = np.argsort(event_times, kind="stable")
    running = np.concatenate([[0.0], np.cumsum(values[order])])
    return running[np.searchsorted(event_times[order], query_times, side="left")]


def grouped_sum_before(
    df: pd.DataFrame,
    key: str,
    event_col: str,
    asof: str,
    value_col: str | None = None,
    days_back: float = 0,
) -> np.ndarray:
    """Like sum_before, but only over rows that share the same `key` (a seller, a state).

    Each row asks: among rows with my key, what had happened before my scoring
    time, moved `days_back` days earlier? Rows with a missing event time or a
    missing value never count.
    """
    queries = to_seconds(df[asof]) - int(days_back * DAY)
    usable = df[event_col].notna().to_numpy()
    events = to_seconds(df[event_col])
    if value_col is None:
        values = np.ones(len(df))
    else:
        values = df[value_col].to_numpy(dtype=float)
        usable = usable & ~np.isnan(values)
    out = np.zeros(len(df))
    for rows in df.groupby(key, sort=False).indices.values():
        known = rows[usable[rows]]
        out[rows] = sum_before(events[known], values[known], queries[rows])
    return out


def add_order_features(df: pd.DataFrame) -> pd.DataFrame:
    """Facts about the order itself: the promise, the basket, the route, the handoff."""
    out = df.copy()
    out["promised_days"] = days_between(out["purchase_ts"], out["promised_date"])
    out["purchase_dow"] = out["purchase_ts"].dt.dayofweek
    out["purchase_hour"] = out["purchase_ts"].dt.hour
    out["freight_ratio"] = out["freight_value"] / (out["order_value"] + out["freight_value"]).clip(lower=0.01)
    out["same_state"] = (out["customer_state"] == out["seller_state"]).astype(int)
    out["distance_km"] = haversine_km(
        out["customer_lat"], out["customer_lng"], out["seller_lat"], out["seller_lng"]
    )
    # A few records show the carrier scan before the purchase. Treat those as handed over at purchase.
    out["handoff_ts"] = out[["carrier_ts", "purchase_ts"]].max(axis=1).where(out["carrier_ts"].notna())
    out["handling_days"] = days_between(out["purchase_ts"], out["handoff_ts"])
    out["remaining_days"] = days_between(out["handoff_ts"], out["overdue_ts"])
    out["handoff_dow"] = out["handoff_ts"].dt.dayofweek
    out["handoff_hour"] = out["handoff_ts"].dt.hour
    # Durations that become known only at delivery. They feed the history features, never a model directly.
    out["delivery_days"] = days_between(out["purchase_ts"], out["delivered_ts"])
    out["transit_days"] = days_between(out["handoff_ts"], out["delivered_ts"])
    return out


def add_history_features(df: pd.DataFrame, asof: str) -> pd.DataFrame:
    """What the seller, the route and the network looked like just before each scoring time.

    `df` must hold every delivered order, because other orders are the history.
    An order is open from purchase until delivery, in transit from handoff until
    delivery, and overdue once the promised day has ended without a delivery.
    """
    out = df[df[asof].notna()].copy()
    out["_all"] = 0  # a single group holding every order, for network-wide numbers
    out["_route"] = out["seller_state"].astype(str) + ">" + out["customer_state"].astype(str)
    out["_closed_ts"] = out[["overdue_ts", "delivered_ts"]].max(axis=1)  # overdue and since delivered

    def before(key, event_col, value_col=None, days_back=0):
        return grouped_sum_before(out, key, event_col, asof, value_col, days_back)

    def recent(key, value_col, days):
        """Count and sum of `value_col` over same-key orders delivered in the last `days` days."""
        n = before(key, "delivered_ts") - before(key, "delivered_ts", days_back=days)
        total = before(key, "delivered_ts", value_col) - before(key, "delivered_ts", value_col, days_back=days)
        return n, total

    def recent_mean(key, value_col, days, fallback):
        n, total = recent(key, value_col, days)
        return np.where(n >= MIN_ROUTE_ORDERS, total / np.clip(n, 1, None), fallback)

    # Network-wide belief about lateness, from every outcome known so far.
    known_n = before("_all", "delivered_ts")
    known_late = before("_all", "delivered_ts", config.TARGET)
    base_rate = (known_late + PRIOR_LATE) / (known_n + PRIOR_N)

    # The network in the last two weeks, and right now.
    n_14d, late_14d = recent("_all", config.TARGET, 14)
    out["net_late_rate_14d"] = (late_14d + WINDOW_SMOOTHING * base_rate) / (n_14d + WINDOW_SMOOTHING)
    out["net_delivery_days_14d"] = recent_mean("_all", "delivery_days", 14, np.nan)
    out["net_transit_days_14d"] = recent_mean("_all", "transit_days", 14, np.nan)
    out["net_open_orders"] = before("_all", "purchase_ts") - known_n
    out["net_in_transit"] = before("_all", "handoff_ts") - known_n
    net_overdue = before("_all", "overdue_ts") - before("_all", "_closed_ts")
    out["net_overdue_share"] = net_overdue / out["net_open_orders"].clip(lower=1)
    out["net_orders_7d"] = before("_all", "purchase_ts") - before("_all", "purchase_ts", days_back=7)

    # The customer's state.
    key = "customer_state"
    state_n, state_late = recent(key, config.TARGET, 30)
    out["state_late_rate_30d"] = (state_late + WINDOW_SMOOTHING * base_rate) / (state_n + WINDOW_SMOOTHING)
    out["state_transit_days_14d"] = recent_mean(key, "transit_days", 14, out["net_transit_days_14d"])
    state_open = before(key, "purchase_ts") - before(key, "delivered_ts")
    state_overdue = before(key, "overdue_ts") - before(key, "_closed_ts")
    out["state_overdue_share"] = state_overdue / np.clip(state_open, 1, None)

    # The route, seller state to customer state.
    out["route_delivery_days_60d"] = recent_mean("_route", "delivery_days", 60, out["net_delivery_days_14d"])
    out["route_transit_days_60d"] = recent_mean("_route", "transit_days", 60, out["net_transit_days_14d"])
    # Slack: the time the promise leaves, minus the time this route has been taking lately.
    out["promise_slack_days"] = out["promised_days"] - out["route_delivery_days_60d"]
    out["transit_slack_days"] = out["remaining_days"] - out["route_transit_days_60d"]

    # The seller.
    key = "seller_id"
    seller_n = before(key, "delivered_ts")
    seller_late = before(key, "delivered_ts", config.TARGET)
    out["seller_orders_prior"] = seller_n
    out["seller_late_rate_prior"] = (seller_late + SELLER_SMOOTHING * base_rate) / (seller_n + SELLER_SMOOTHING)
    handed_over = before(key, "handoff_ts")
    out["seller_handling_days_prior"] = np.where(
        handed_over > 0, before(key, "handoff_ts", "handling_days") / np.clip(handed_over, 1, None), np.nan
    )
    out["seller_open_orders"] = before(key, "purchase_ts") - seller_n
    out["seller_overdue_open"] = before(key, "overdue_ts") - before(key, "_closed_ts")

    # Rules of thumb the models have to beat. Higher means riskier.
    out["rule_short_promise"] = -out["promise_slack_days"].fillna(out["promised_days"])
    out["rule_low_slack"] = -out["transit_slack_days"].fillna(out["remaining_days"])

    return out.drop(columns=["_all", "_route", "_closed_ts"])


def build_features(orders: pd.DataFrame, stage: config.Stage) -> pd.DataFrame:
    """Feature table for one stage: a row per order with the stage's features, keys and label."""
    return add_history_features(add_order_features(orders), asof=stage.asof)
