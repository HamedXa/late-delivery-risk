"""Paths and constants. Change them here, nowhere else."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

# Repo root. Override with DELIVERY_RISK_HOME when the package is installed elsewhere.
ROOT = Path(os.environ.get("DELIVERY_RISK_HOME", Path(__file__).resolve().parents[2]))
RAW_DIR = ROOT / "data" / "raw"
ARTIFACT_DIR = Path(os.environ.get("DELIVERY_RISK_ARTIFACTS", ROOT / "artifacts"))
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

RAW_FILES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "categories": "product_category_name_translation.csv",
}

SEED = 42
TARGET = "is_late"

# Brazil's five regions, for reporting results by where the customer lives.
REGIONS = {
    "Southeast": ("SP", "RJ", "MG", "ES"),
    "South": ("PR", "SC", "RS"),
    "Northeast": ("BA", "PE", "CE", "MA", "PB", "RN", "AL", "PI", "SE"),
    "Centre-West": ("DF", "GO", "MT", "MS"),
    "North": ("PA", "AM", "RO", "TO", "AC", "AP", "RR"),
}

# Rough bounding box for Brazil, used to drop bad geolocation points.
LAT_RANGE = (-34.0, 5.5)
LNG_RANGE = (-74.0, -34.0)

# Walk-forward backtest. Each fold trains on orders whose outcome was already
# known when the fold's month began, then scores that month.
# Development folds are for comparing features and settings.
# Evaluation folds produce the numbers reported in the README.
DEV_FOLDS = ["2017-10", "2017-11", "2017-12", "2018-01"]
EVAL_FOLDS = ["2018-02", "2018-03", "2018-04", "2018-05", "2018-06", "2018-07", "2018-08"]

# Share of orders the operations team can act on. Alerts go to the riskiest
# ALERT_BUDGET of orders.
ALERT_BUDGET = 0.10

# "Recent" for reference values: the typical order, the score distribution, drift baselines.
REFERENCE_DAYS = 180
# Network, state and route conditions change faster, so their defaults use a shorter window.
CONDITIONS_DAYS = 60


@dataclass(frozen=True)
class Stage:
    """A moment in an order's life at which it gets scored."""

    name: str
    asof: str  # column holding the scoring time; features may only use what happened before it
    numeric: tuple[str, ...]
    rule: str  # one-column rule of thumb the model has to beat; higher means riskier
    categorical: tuple[str, ...] = ()
    # Features the model may only use in one direction: -1 means more of it never raises the risk.
    monotone: dict[str, int] = field(default_factory=dict)
    # A smaller feature set that was also tried. See README, "What did not work".
    narrow_numeric: tuple[str, ...] = ()

    @property
    def features(self) -> list[str]:
        return [*self.categorical, *self.numeric]


ORDER_FEATURES = ("n_items", "order_value", "freight_value", "weight_kg", "volume_l", "distance_km", "same_state")

CHECKOUT = Stage(
    name="checkout",
    asof="purchase_ts",
    categorical=("customer_state", "seller_state", "payment_type", "category"),
    numeric=(
        # the promise and the calendar
        "promised_days",
        "promise_slack_days",  # the promise minus what this route has been taking lately
        "purchase_dow",
        "purchase_hour",
        # the order and the route
        *ORDER_FEATURES,
        "n_sellers",
        "freight_ratio",
        "payment_installments",
        # the seller, from outcomes known before this purchase
        "seller_orders_prior",
        "seller_late_rate_prior",
        "seller_handling_days_prior",
        "seller_open_orders",
        "seller_overdue_open",
        # the network at the moment of purchase
        "net_open_orders",
        "net_overdue_share",
        "net_late_rate_14d",
        "net_orders_7d",
        "net_delivery_days_14d",
        "state_overdue_share",
        "state_late_rate_30d",
        "route_delivery_days_60d",
    ),
    rule="rule_short_promise",
    narrow_numeric=(
        "promised_days",
        "promise_slack_days",
        "seller_late_rate_prior",
        "seller_handling_days_prior",
        "net_late_rate_14d",
    ),
)

HANDOFF = Stage(
    name="handoff",
    asof="handoff_ts",
    categorical=("customer_state", "seller_state"),
    numeric=(
        # known once the seller hands the parcel to the carrier
        "handling_days",  # purchase to handoff
        "remaining_days",  # handoff to the end of the promised day
        "transit_slack_days",  # remaining days minus this route's recent transit time
        "promised_days",
        "handoff_dow",
        "handoff_hour",
        *ORDER_FEATURES,
        "seller_late_rate_prior",
        # the carrier network at the moment of handoff
        "net_in_transit",
        "net_overdue_share",
        "net_late_rate_14d",
        "net_transit_days_14d",
        "state_overdue_share",
        "state_late_rate_30d",
        "state_transit_days_14d",
        "route_transit_days_60d",
    ),
    rule="rule_low_slack",
    monotone={"remaining_days": -1, "transit_slack_days": -1},
    narrow_numeric=(
        "handling_days",
        "remaining_days",
        "transit_slack_days",
        "promised_days",
        "handoff_dow",
        "distance_km",
        "same_state",
        "weight_kg",
        "freight_value",
        "net_late_rate_14d",
        "state_late_rate_30d",
    ),
)

STAGES = {s.name: s for s in (CHECKOUT, HANDOFF)}


def narrow(stage: Stage) -> Stage:
    """The same stage with the smaller feature set and no categorical inputs."""
    return replace(stage, numeric=stage.narrow_numeric, categorical=())


def is_context(feature: str) -> bool:
    """True for features describing the network, a state or a route, not the order itself."""
    return feature.startswith(("net_", "state_", "route_"))
