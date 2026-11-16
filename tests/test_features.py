"""The feature rules: correct arithmetic, and nothing from the future."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from delivery_risk import config, data, features


def test_haversine_sao_paulo_to_rio():
    km = features.haversine_km(-23.55, -46.63, -22.91, -43.17)
    assert 350 < float(km) < 370


def test_sum_before_ignores_events_at_the_same_instant():
    events = np.array([10, 20, 30])
    values = np.array([1.0, 1.0, 1.0])
    counted = features.sum_before(events, values, np.array([5, 20, 31]))
    assert counted.tolist() == [0.0, 1.0, 3.0]  # the event at 20 is not "before" 20


def test_seller_history_waits_for_the_delivery():
    """A late order only counts against its seller once it has actually been delivered."""
    day = pd.Timestamp("2018-03-01")
    orders = pd.DataFrame(
        {
            "order_id": ["a", "b", "c"],
            "seller_id": ["s1", "s1", "s1"],
            "purchase_ts": [day, day + pd.Timedelta(days=3), day + pd.Timedelta(days=12)],
            "carrier_ts": [day + pd.Timedelta(days=1), day + pd.Timedelta(days=4), day + pd.Timedelta(days=13)],
            "delivered_ts": [day + pd.Timedelta(days=10), day + pd.Timedelta(days=8), day + pd.Timedelta(days=20)],
            "promised_date": [day + pd.Timedelta(days=5), day + pd.Timedelta(days=15), day + pd.Timedelta(days=25)],
            "customer_state": "RJ",
            "seller_state": "SP",
            "customer_lat": -22.9, "customer_lng": -43.2, "seller_lat": -23.5, "seller_lng": -46.6,
            "n_items": 1, "n_sellers": 1, "order_value": 100.0, "freight_value": 10.0,
            "weight_kg": 1.0, "volume_l": 5.0, "payment_type": "credit_card",
            "payment_installments": 1, "category": "toys",
        }
    )  # fmt: skip
    built = features.build_features(data.add_label(orders), config.CHECKOUT).set_index("order_id")

    assert built.loc["a", config.TARGET] == 1  # delivered five days after the promise
    # Order b was bought on day 3. Order a is already running, but its outcome is unknown until day 10.
    assert built.loc["b", "seller_orders_prior"] == 0
    assert built.loc["b", "seller_open_orders"] == 1
    # Order c was bought on day 12, after both earlier deliveries. One of the two was late.
    assert built.loc["c", "seller_orders_prior"] == 2
    assert built.loc["c", "seller_late_rate_prior"] > built.loc["b", "seller_late_rate_prior"]


def test_route_average_matches_the_slow_obvious_calculation(handoff_features):
    """The fast cumulative-sum code against a plain loop.

    The feature is the mean transit time of same-route orders delivered in the 60 days before scoring.
    """
    table = handoff_features
    route = table["seller_state"] + ">" + table["customer_state"]
    checked = 0
    for index in table.index[::23]:  # every 23rd order is plenty
        scored_at = table.loc[index, "handoff_ts"]
        window = (
            (route == route[index])
            & (table["delivered_ts"] < scored_at)
            & (table["delivered_ts"] >= scored_at - pd.Timedelta(days=60))
        )
        if window.sum() >= features.MIN_ROUTE_ORDERS:
            assert table.loc[index, "route_transit_days_60d"] == pytest.approx(table.loc[window, "transit_days"].mean())
            checked += 1
    assert checked >= 10


@pytest.mark.parametrize("stage", list(config.STAGES.values()), ids=list(config.STAGES))
def test_features_do_not_change_when_the_future_does(orders, stage):
    """Rewrite everything after a cutoff. Features of orders scored before it must not move.

    This is the leakage test. If any feature peeks at an outcome that was not
    yet known at scoring time, the two tables differ and the test fails.
    """
    before = features.build_features(orders, stage)
    cutoff = before[stage.asof].quantile(0.6)

    rewritten = orders[orders["purchase_ts"] < cutoff].copy()  # orders placed later vanish
    late_handoff = rewritten["carrier_ts"] >= cutoff
    rewritten.loc[late_handoff, "carrier_ts"] += pd.Timedelta(days=2)  # later handoffs move
    late_delivery = rewritten["delivered_ts"] >= cutoff
    rewritten.loc[late_delivery, "delivered_ts"] += pd.Timedelta(days=9)  # later deliveries move, labels flip
    after = features.build_features(data.add_label(rewritten), stage)

    past = before[before[stage.asof] < cutoff].set_index("order_id")
    same_orders = after[after[stage.asof] < cutoff].set_index("order_id").loc[past.index]
    assert len(past) > 100
    assert (past[config.TARGET] != same_orders[config.TARGET]).any(), "the rewrite should change some labels"
    pd.testing.assert_frame_equal(past[stage.features], same_orders[stage.features], check_exact=False, rtol=1e-9)


@pytest.mark.parametrize("stage", list(config.STAGES.values()), ids=list(config.STAGES))
def test_every_model_input_is_built(orders, stage):
    built = features.build_features(orders, stage)
    missing = [c for c in [*stage.features, *stage.narrow_numeric, stage.rule] if c not in built.columns]
    assert not missing
    assert built[stage.asof].notna().all()
