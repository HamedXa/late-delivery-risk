"""Joining the raw tables and defining the label."""

from __future__ import annotations

import pandas as pd

from delivery_risk import config, data


def tiny_raw() -> dict[str, pd.DataFrame]:
    ts = pd.Timestamp
    orders = pd.DataFrame(
        {
            "order_id": ["late", "same_day", "cancelled"],
            "customer_id": ["c1", "c2", "c3"],
            "order_status": ["delivered", "delivered", "canceled"],
            "order_purchase_timestamp": [ts("2018-01-01 10:00"), ts("2018-01-02 10:00"), ts("2018-01-03 10:00")],
            "order_approved_at": [ts("2018-01-01 11:00"), ts("2018-01-02 11:00"), pd.NaT],
            "order_delivered_carrier_date": [ts("2018-01-02 09:00"), ts("2018-01-03 09:00"), pd.NaT],
            "order_delivered_customer_date": [ts("2018-01-12 08:00"), ts("2018-01-10 23:30"), pd.NaT],
            "order_estimated_delivery_date": [ts("2018-01-10"), ts("2018-01-10"), ts("2018-01-15")],
        }
    )
    items = pd.DataFrame(
        {
            "order_id": ["late", "late", "same_day", "cancelled"],
            "order_item_id": [1, 2, 1, 1],
            "product_id": ["p1", "p2", "p1", "p1"],
            "seller_id": ["s1", "s2", "s1", "s1"],
            "shipping_limit_date": ["2018-01-05"] * 4,
            "price": [50.0, 200.0, 50.0, 50.0],
            "freight_value": [10.0, 20.0, 10.0, 10.0],
        }
    )
    products = pd.DataFrame(
        {
            "product_id": ["p1", "p2"],
            "product_category_name": ["brinquedos", "sem_traducao"],
            "product_weight_g": [500, 2000],
            "product_length_cm": [10, 20],
            "product_height_cm": [10, 20],
            "product_width_cm": [10, 20],
        }
    )
    return {
        "orders": orders,
        "items": items,
        "products": products,
        "categories": pd.DataFrame(
            {"product_category_name": ["brinquedos"], "product_category_name_english": ["toys"]}
        ),
        "customers": pd.DataFrame(
            {
                "customer_id": ["c1", "c2", "c3"],
                "customer_zip_code_prefix": [20000, 20000, 99999],
                "customer_state": ["RJ", "RJ", "RJ"],
            }
        ),
        "sellers": pd.DataFrame(
            {"seller_id": ["s1", "s2"], "seller_zip_code_prefix": [1000, 1000], "seller_state": ["SP", "SP"]}
        ),
        "payments": pd.DataFrame(
            {
                "order_id": ["late", "late", "same_day"],
                "payment_sequential": [1, 2, 1],
                "payment_type": ["voucher", "credit_card", "boleto"],
                "payment_installments": [1, 6, 1],
                "payment_value": [30.0, 250.0, 60.0],
            }
        ),
        "geolocation": pd.DataFrame(
            {
                "geolocation_zip_code_prefix": [20000, 1000, 1000],
                "geolocation_lat": [-22.9, -23.5, 60.0],  # the last point is outside Brazil and gets dropped
                "geolocation_lng": [-43.2, -46.6, 10.0],
                "geolocation_state": ["RJ", "SP", "SP"],
            }
        ),
    }


def test_build_orders_keeps_delivered_orders_and_labels_them():
    built = data.build_orders(tiny_raw()).set_index("order_id")

    assert sorted(built.index) == ["late", "same_day"]  # the cancelled order has no outcome
    assert built.loc["late", config.TARGET] == 1
    assert built.loc["same_day", config.TARGET] == 0  # arriving on the promised day is on time


def test_build_orders_summarizes_the_basket():
    row = data.build_orders(tiny_raw()).set_index("order_id").loc["late"]

    assert row["n_items"] == 2 and row["n_sellers"] == 2
    assert row["seller_id"] == "s2"  # the most expensive item decides the seller
    assert row["category"] == "unknown"  # a category with no translation
    assert row["payment_type"] == "credit_card"  # the largest payment decides the type
    assert row["weight_kg"] == 2.5
    assert abs(row["seller_lat"] - (-23.5)) < 1e-9  # the point outside Brazil was ignored
