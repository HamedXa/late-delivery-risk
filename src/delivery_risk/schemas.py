"""Request and response shapes for the API."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field, field_validator, model_validator

from . import config

CONDITION_INPUTS = sorted(f for f in config.HANDOFF.numeric if config.is_context(f))


class HandoffOrder(BaseModel):
    """An order at the moment the seller hands the parcel to the carrier."""

    purchase_ts: datetime = Field(description="When the customer placed the order.")
    handoff_ts: datetime = Field(description="When the carrier received the parcel.")
    promised_date: date = Field(description="The delivery date shown to the customer.")
    customer_state: str = Field(min_length=2, max_length=2, description="Two-letter state code.")
    seller_state: str = Field(min_length=2, max_length=2, description="Two-letter state code.")
    order_value: float = Field(ge=0, description="Total price of the items.")
    freight_value: float = Field(ge=0, description="Total freight charged.")
    weight_kg: float = Field(gt=0, le=500)
    volume_l: float | None = Field(default=None, gt=0, description="Parcel volume in litres.")
    n_items: int = Field(default=1, ge=1, le=50)
    distance_km: float | None = Field(
        default=None, ge=0, description="Seller to customer. Filled with the route's usual distance when omitted."
    )
    seller_late_rate: float | None = Field(
        default=None, ge=0, le=1, description="Share of this seller's past deliveries that were late."
    )
    conditions: dict[str, float] | None = Field(
        default=None,
        description=f"Live network or state conditions, if you have them. Allowed keys: {CONDITION_INPUTS}.",
    )

    @field_validator("customer_state", "seller_state")
    @classmethod
    def upper_case(cls, value: str) -> str:
        return value.upper()

    @field_validator("purchase_ts", "handoff_ts")
    @classmethod
    def drop_timezone(cls, value: datetime) -> datetime:
        return value.replace(tzinfo=None)

    @field_validator("conditions")
    @classmethod
    def known_conditions(cls, value: dict[str, float] | None) -> dict[str, float] | None:
        unknown = sorted(set(value or {}) - set(CONDITION_INPUTS))
        if unknown:
            raise ValueError(f"unknown condition inputs: {unknown}")
        return value

    @model_validator(mode="after")
    def dates_in_order(self) -> HandoffOrder:
        if self.handoff_ts < self.purchase_ts:
            raise ValueError("handoff_ts is before purchase_ts")
        if self.promised_date < self.purchase_ts.date():
            raise ValueError("promised_date is before the purchase date")
        return self


class Reason(BaseModel):
    """How much one group of inputs moves the score, against a typical recent order."""

    factor: str
    effect: float = Field(description="Score with this order's values minus score with typical values.")
    detail: str


class RiskResponse(BaseModel):
    risk_score: float = Field(description="Model score from 0 to 1. A ranking score, not a calibrated probability.")
    percentile: int = Field(description="Share of recent orders that scored lower, 0 to 100.")
    alert: bool = Field(description="True when the score is in the riskiest 10% of recent orders.")
    alert_threshold: float
    slack_days: float = Field(description="Days left on the promise minus this route's recent transit time.")
    rule_alert: bool = Field(description="What the one-line slack rule would have said.")
    reasons: list[Reason]
    inputs: dict[str, float | str] = Field(description="Every value the model was given.")
