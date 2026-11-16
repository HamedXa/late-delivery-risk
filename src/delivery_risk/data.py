"""Load the raw Olist tables and build one row per delivered order.

Nothing here looks at the future on purpose: this module only joins tables and
defines the label. Everything time-sensitive lives in features.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config

ORDER_DATES = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]


def load_raw(raw_dir: Path | str = config.RAW_DIR) -> dict[str, pd.DataFrame]:
    """Read the eight CSVs this project uses. Raises a clear error if one is missing."""
    raw_dir = Path(raw_dir)
    missing = [f for f in config.RAW_FILES.values() if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing files in {raw_dir}: {missing}. See data/raw/README.md for the download link."
        )
    raw = {}
    for name, filename in config.RAW_FILES.items():
        parse = ORDER_DATES if name == "orders" else None
        raw[name] = pd.read_csv(raw_dir / filename, parse_dates=parse)
    return raw


def zip_centroids(geolocation: pd.DataFrame) -> pd.DataFrame:
    """Mean latitude and longitude per zip prefix, after dropping points outside Brazil."""
    geo = geolocation.rename(
        columns={
            "geolocation_zip_code_prefix": "zip_prefix",
            "geolocation_lat": "lat",
            "geolocation_lng": "lng",
            "geolocation_state": "state",
        }
    )
    inside = geo["lat"].between(*config.LAT_RANGE) & geo["lng"].between(*config.LNG_RANGE)
    geo = geo.loc[inside, ["zip_prefix", "state", "lat", "lng"]]
    return geo.groupby("zip_prefix", as_index=False).agg(
        lat=("lat", "mean"), lng=("lng", "mean"), state=("state", "first")
    )


def state_centroids(zips: pd.DataFrame) -> pd.DataFrame:
    """One point per state, used when a zip prefix has no coordinates."""
    return zips.groupby("state", as_index=False).agg(lat=("lat", "mean"), lng=("lng", "mean"))


def _order_items(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Collapse order items to one row per order."""
    products = raw["products"].merge(raw["categories"], on="product_category_name", how="left")
    products["category"] = products["product_category_name_english"].fillna("unknown")
    products["volume_l"] = (
        products["product_length_cm"] * products["product_height_cm"] * products["product_width_cm"]
    ) / 1000.0
    products["weight_kg"] = products["product_weight_g"] / 1000.0

    items = raw["items"].merge(
        products[["product_id", "category", "volume_l", "weight_kg"]], on="product_id", how="left"
    )
    # The most expensive item decides the order's seller and category.
    items = items.sort_values(["order_id", "price"], ascending=[True, False], kind="stable")
    main = items.drop_duplicates("order_id")[["order_id", "seller_id", "category"]]
    totals = items.groupby("order_id", as_index=False).agg(
        n_items=("order_item_id", "count"),
        n_sellers=("seller_id", "nunique"),
        order_value=("price", "sum"),
        freight_value=("freight_value", "sum"),
        weight_kg=("weight_kg", lambda s: s.sum(min_count=1)),
        volume_l=("volume_l", lambda s: s.sum(min_count=1)),
    )
    return totals.merge(main, on="order_id", how="left")


def _order_payments(payments: pd.DataFrame) -> pd.DataFrame:
    """Main payment type (largest amount) and the highest number of installments."""
    ordered = payments.sort_values(
        ["order_id", "payment_value"], ascending=[True, False], kind="stable"
    )
    main = ordered.drop_duplicates("order_id")[["order_id", "payment_type"]]
    inst = payments.groupby("order_id", as_index=False).agg(
        payment_installments=("payment_installments", "max")
    )
    return main.merge(inst, on="order_id", how="left")


def add_label(orders: pd.DataFrame) -> pd.DataFrame:
    """Add the label and the moment an order becomes overdue.

    An order is late when it reaches the customer on a calendar day after the
    promised date. Delivery on the promised day counts as on time.
    """
    out = orders.copy()
    out["promised_date"] = out["promised_date"].dt.normalize()
    out[config.TARGET] = (out["delivered_ts"].dt.normalize() > out["promised_date"]).astype(int)
    # First moment the order counts as overdue: the start of the day after the promise.
    out["overdue_ts"] = out["promised_date"] + pd.Timedelta(days=1)
    return out


def build_orders(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per delivered order, with the label and the columns features.py needs."""
    orders = raw["orders"].rename(
        columns={
            "order_purchase_timestamp": "purchase_ts",
            "order_delivered_carrier_date": "carrier_ts",
            "order_delivered_customer_date": "delivered_ts",
            "order_estimated_delivery_date": "promised_date",
        }
    )
    delivered = (
        (orders["order_status"] == "delivered")
        & orders["delivered_ts"].notna()
        & orders["promised_date"].notna()
    )
    orders = orders.loc[
        delivered,
        ["order_id", "customer_id", "purchase_ts", "carrier_ts", "delivered_ts", "promised_date"],
    ].copy()

    orders = add_label(orders)

    customers = raw["customers"].rename(columns={"customer_zip_code_prefix": "customer_zip"})
    sellers = raw["sellers"].rename(columns={"seller_zip_code_prefix": "seller_zip"})

    df = (
        orders.merge(
            customers[["customer_id", "customer_zip", "customer_state"]], on="customer_id", how="left"
        )
        .merge(_order_items(raw), on="order_id", how="inner")
        .merge(sellers[["seller_id", "seller_zip", "seller_state"]], on="seller_id", how="left")
        .merge(_order_payments(raw["payments"]), on="order_id", how="left")
    )

    zips = zip_centroids(raw["geolocation"])
    states = state_centroids(zips).set_index("state")
    for side in ("customer", "seller"):
        df = df.merge(
            zips[["zip_prefix", "lat", "lng"]].rename(
                columns={"zip_prefix": f"{side}_zip", "lat": f"{side}_lat", "lng": f"{side}_lng"}
            ),
            on=f"{side}_zip",
            how="left",
        )
        # Fall back to the state centre when the zip prefix has no coordinates.
        for axis in ("lat", "lng"):
            fallback = df[f"{side}_state"].map(states[axis])
            df[f"{side}_{axis}"] = df[f"{side}_{axis}"].fillna(fallback)

    df["payment_type"] = df["payment_type"].fillna("unknown")
    df["payment_installments"] = df["payment_installments"].fillna(1)
    df["category"] = df["category"].fillna("unknown")
    return df.sort_values("purchase_ts", kind="stable").reset_index(drop=True)


def label_summary(df: pd.DataFrame) -> dict[str, float]:
    """Counts used in the README, so the headline numbers are reproducible."""
    return {
        "orders": int(len(df)),
        "late_orders": int(df[config.TARGET].sum()),
        "late_rate": float(np.round(df[config.TARGET].mean(), 4)),
        "first_purchase": str(df["purchase_ts"].min().date()),
        "last_purchase": str(df["purchase_ts"].max().date()),
    }
