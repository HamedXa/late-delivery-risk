"""HTTP API: score an order for late-delivery risk at carrier handoff.

    uvicorn delivery_risk.serve:app --port 7860
"""

from __future__ import annotations

import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from . import __version__, config
from .features import DAY
from .schemas import HandoffOrder, Reason, RiskResponse
from .train import route_distance_km

STAGE = config.HANDOFF
PAGE = config.ROOT / "app" / "static" / "index.html"

# Inputs that are explained together, because changing one alone makes no sense.
REASON_GROUPS = {
    "Time left on the promise": ["handling_days", "remaining_days", "promised_days"],
    "Transit time on this route": ["route_transit_days_60d"],
    "Network conditions": [f for f in STAGE.numeric if f.startswith("net_")],
    "Conditions in the customer's state": [f for f in STAGE.numeric if f.startswith("state_")],
    "Parcel size and value": ["weight_kg", "volume_l", "order_value", "freight_value", "n_items"],
    "Distance": ["distance_km", "same_state"],
    "Seller's record": ["seller_late_rate_prior"],
    "Day and hour of handoff": ["handoff_dow", "handoff_hour"],
}

# How an input reads inside an explanation: number format, then what the number is.
DESCRIBE = {
    "handling_days": ("{:.1f} days", "from purchase to handoff"),
    "remaining_days": ("{:.1f} days", "left on the promise"),
    "promised_days": ("{:.1f} days", "promised at checkout"),
    "route_transit_days_60d": ("{:.1f} days", "recent transit on this route"),
    "net_late_rate_14d": ("{:.1%}", "of deliveries late in the last 14 days"),
    "net_overdue_share": ("{:.1%}", "of open orders overdue"),
    "net_transit_days_14d": ("{:.1f} days", "recent transit, all routes"),
    "state_late_rate_30d": ("{:.1%}", "of the state's deliveries late in the last 30 days"),
    "state_overdue_share": ("{:.1%}", "of the state's open orders overdue"),
    "state_transit_days_14d": ("{:.1f} days", "recent transit into the state"),
    "weight_kg": ("{:.1f} kg", "parcel weight"),
    "order_value": ("R$ {:,.0f}", "order value"),
    "freight_value": ("R$ {:,.0f}", "freight"),
    "distance_km": ("{:,.0f} km", "seller to customer"),
    "seller_late_rate_prior": ("{:.1%}", "of the seller's past deliveries late"),
}
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

app = FastAPI(
    title="Late-delivery risk",
    version=__version__,
    description="Scores an order at carrier handoff. Trained on the public Olist dataset (Brazil, 2016 to 2018).",
)


@lru_cache
def load() -> tuple[object, dict]:
    """The fitted pipeline and its reference values, read once."""
    fitted = joblib.load(config.ARTIFACT_DIR / f"{STAGE.name}_model.joblib")
    reference = json.loads((config.ARTIFACT_DIR / f"{STAGE.name}_reference.json").read_text())
    return fitted, reference


def with_slack(inputs: dict) -> dict:
    """Slack is derived, so recompute it whenever its parts change."""
    return {**inputs, "transit_slack_days": inputs["remaining_days"] - inputs["route_transit_days_60d"]}


def build_inputs(order: HandoffOrder, reference: dict) -> dict:
    """Turn a request into the model's inputs.

    Network, state and route conditions start from their values over the last
    two months of the training data and are overridden by `conditions` when supplied.
    """
    typical = reference["typical"]
    purchase, handoff = pd.Timestamp(order.purchase_ts), pd.Timestamp(order.handoff_ts)
    promised = pd.Timestamp(order.promised_date)
    route = f"{order.seller_state}>{order.customer_state}"

    inputs = {name: typical[name] for name in STAGE.numeric}
    inputs.update(reference["by_state"].get(order.customer_state, {}))
    inputs["route_transit_days_60d"] = reference["route_transit_days"].get(route, typical["net_transit_days_14d"])
    inputs.update(order.conditions or {})

    distance = order.distance_km
    if distance is None:
        distance = route_distance_km(reference, order.seller_state, order.customer_state)
    inputs.update(
        customer_state=order.customer_state,
        seller_state=order.seller_state,
        handling_days=(handoff - purchase).total_seconds() / DAY,
        remaining_days=(promised + pd.Timedelta(days=1) - handoff).total_seconds() / DAY,
        promised_days=(promised - purchase).total_seconds() / DAY,
        handoff_dow=handoff.dayofweek,
        handoff_hour=handoff.hour,
        n_items=order.n_items,
        order_value=order.order_value,
        freight_value=order.freight_value,
        weight_kg=order.weight_kg,
        volume_l=typical["volume_l"] if order.volume_l is None else order.volume_l,
        distance_km=typical["distance_km"] if distance is None else distance,
        same_state=int(order.customer_state == order.seller_state),
        seller_late_rate_prior=(
            typical["seller_late_rate_prior"] if order.seller_late_rate is None else order.seller_late_rate
        ),
    )
    return with_slack(inputs)


def describe(names: list[str], inputs: dict, typical: dict) -> str:
    """One line of plain words for a group of inputs, each next to its typical value."""
    if "handoff_dow" in names:
        return f"handed over on a {WEEKDAYS[int(inputs['handoff_dow'])]} at {int(inputs['handoff_hour']):02d}:00"
    parts = []
    for name in names:
        if name in DESCRIBE:
            number, meaning = DESCRIBE[name]
            parts.append(f"{number.format(inputs[name])} {meaning} (typical {number.format(typical[name])})")
    return "; ".join(parts)


def explain(inputs: dict, reference: dict, fitted) -> tuple[float, list[Reason]]:
    """Score the order, then re-score it with each group of inputs set to typical values.

    The difference is that group's effect. It is a what-if against a typical
    order, which is easier to read than a per-feature attribution.
    """
    typical = reference["typical"]
    variants, labels = [inputs], []
    for label, names in REASON_GROUPS.items():
        variants.append(with_slack({**inputs, **{n: typical[n] for n in names}}))
        labels.append(label)
    scores = fitted.predict_proba(pd.DataFrame(variants)[STAGE.features])[:, 1]
    reasons = []
    for label, counterfactual in zip(labels, scores[1:], strict=True):
        effect = float(scores[0] - counterfactual)
        if abs(effect) >= 0.005:
            detail = describe(REASON_GROUPS[label], inputs, typical)
            reasons.append(Reason(factor=label, effect=round(effect, 4), detail=detail))
    return float(scores[0]), sorted(reasons, key=lambda r: -r.effect)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def demo_page() -> str:
    return PAGE.read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict:
    _, reference = load()
    return {
        "status": "ok",
        "version": __version__,
        "stage": reference["stage"],
        "trained_through": reference["trained_through"],
        "training_rows": reference["training_rows"],
        "alert_threshold_from": reference["scores_from"],
    }


@app.get("/reference")
def reference_values() -> dict:
    """Known states, and the conditions the API assumes when a request sends none."""
    _, reference = load()
    return {
        "states": sorted(reference["state_centroids"]),
        "alert_threshold": reference["alert_threshold"],
        "typical_conditions": {k: v for k, v in reference["typical"].items() if config.is_context(k)},
        "trained_through": reference["trained_through"],
    }


@app.post("/predict", response_model=RiskResponse)
def predict(order: HandoffOrder) -> RiskResponse:
    fitted, reference = load()
    inputs = build_inputs(order, reference)
    score, reasons = explain(inputs, reference, fitted)
    percentile = int(np.searchsorted(reference["score_percentiles"], score, side="right")) - 1
    return RiskResponse(
        risk_score=round(score, 4),
        percentile=int(np.clip(percentile, 0, 100)),
        alert=score >= reference["alert_threshold"],
        alert_threshold=round(reference["alert_threshold"], 4),
        slack_days=round(inputs["transit_slack_days"], 2),
        rule_alert=-inputs["transit_slack_days"] >= reference["rule_threshold"],
        reasons=reasons,
        inputs={k: (v if isinstance(v, str) else round(float(v), 4)) for k, v in inputs.items()},
    )
