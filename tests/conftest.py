"""A small made-up dataset, so the tests run in seconds without the Olist files."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from delivery_risk import config, data, features

STATES = {"SP": (-23.5, -46.6), "RJ": (-22.9, -43.2), "MG": (-19.9, -43.9), "BA": (-12.9, -38.5)}


def make_orders(n: int = 700, seed: int = 0) -> pd.DataFrame:
    """Delivered orders with the columns data.build_orders produces."""
    rng = np.random.default_rng(seed)
    purchase = pd.Timestamp("2018-01-01") + pd.to_timedelta(np.sort(rng.uniform(0, 160, n)) * 86_400, unit="s")
    carrier = purchase + pd.to_timedelta(rng.gamma(2.0, 1.2, n) * 86_400, unit="s")
    delivered = carrier + pd.to_timedelta(rng.gamma(3.0, 3.0, n) * 86_400, unit="s")
    customer_state = rng.choice(list(STATES), n)
    seller_state = rng.choice(["SP", "MG"], n)
    orders = pd.DataFrame(
        {
            "order_id": [f"order-{i}" for i in range(n)],
            "purchase_ts": purchase.round("s"),
            "carrier_ts": carrier.round("s"),
            "delivered_ts": delivered.round("s"),
            "promised_date": purchase + pd.to_timedelta(rng.integers(6, 26, n), unit="D"),
            "customer_state": customer_state,
            "seller_state": seller_state,
            "seller_id": rng.choice([f"seller-{i}" for i in range(15)], n),
            "customer_lat": [STATES[s][0] for s in customer_state] + rng.normal(0, 0.5, n),
            "customer_lng": [STATES[s][1] for s in customer_state] + rng.normal(0, 0.5, n),
            "seller_lat": [STATES[s][0] for s in seller_state] + rng.normal(0, 0.5, n),
            "seller_lng": [STATES[s][1] for s in seller_state] + rng.normal(0, 0.5, n),
            "n_items": rng.integers(1, 4, n),
            "n_sellers": 1,
            "order_value": rng.gamma(2.0, 60.0, n).round(2),
            "freight_value": rng.gamma(2.0, 10.0, n).round(2),
            "weight_kg": rng.gamma(1.5, 1.2, n).round(2) + 0.05,
            "volume_l": rng.gamma(2.0, 6.0, n).round(2) + 0.1,
            "payment_type": rng.choice(["credit_card", "boleto"], n),
            "payment_installments": rng.integers(1, 6, n),
            "category": rng.choice(["toys", "health_beauty", "auto"], n),
        }
    )
    return data.add_label(orders)


@pytest.fixture(scope="session")
def orders() -> pd.DataFrame:
    return make_orders()


@pytest.fixture(scope="session")
def handoff_features(orders) -> pd.DataFrame:
    return features.build_features(orders, config.HANDOFF)
