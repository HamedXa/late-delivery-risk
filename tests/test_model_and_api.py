"""The model pipeline and the HTTP API, on a model trained from the made-up data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from delivery_risk import config, features, model, serve, train

ORDER = {
    "purchase_ts": "2018-05-01T10:00:00",
    "handoff_ts": "2018-05-03T15:00:00",
    "promised_date": "2018-05-12",
    "customer_state": "rj",
    "seller_state": "SP",
    "order_value": 120.0,
    "freight_value": 18.0,
    "weight_kg": 1.2,
}


@pytest.mark.parametrize("stage", list(config.STAGES.values()), ids=list(config.STAGES))
def test_model_trains_and_copes_with_gaps(orders, stage):
    table = features.build_features(orders, stage)
    fitted = model.make_gbm(stage).fit(table[stage.features], table[config.TARGET])

    odd = table[stage.features].head(5).copy()
    odd.loc[odd.index[0], stage.numeric[0]] = np.nan  # a missing number
    odd.loc[odd.index[1], stage.categorical[0]] = "ZZ"  # a state the model has never seen
    scores = fitted.predict_proba(odd)[:, 1]
    assert ((scores >= 0) & (scores <= 1)).all()


def test_alert_threshold_comes_from_scores_the_model_had_not_trained_on(handoff_features):
    stage = config.HANDOFF
    fitted = model.make_gbm(stage).fit(handoff_features[stage.features], handoff_features[config.TARGET])
    recent = train.recent_rows(handoff_features, stage)
    backtest = pd.DataFrame({"order_id": recent["order_id"], "score": 0.5})

    from_backtest = train.build_reference(handoff_features, stage, fitted, backtest)
    without = train.build_reference(handoff_features, stage, fitted)

    assert from_backtest["scores_from"] == "walk-forward backtest"
    assert from_backtest["alert_threshold"] == 0.5
    assert without["scores_from"] == "training rows"


@pytest.fixture(scope="module")
def client(handoff_features, tmp_path_factory):
    """An API backed by a throwaway model, so the test does not depend on the shipped artifacts."""
    patch = pytest.MonkeyPatch()
    patch.setattr(config, "ARTIFACT_DIR", tmp_path_factory.mktemp("artifacts"))
    train.fit_and_save(handoff_features, config.HANDOFF)
    serve.load.cache_clear()
    yield TestClient(serve.app)
    patch.undo()
    serve.load.cache_clear()


def test_health_and_demo_page(client):
    assert client.get("/health").json()["status"] == "ok"
    page = client.get("/")
    assert page.status_code == 200 and "Late-delivery risk" in page.text


def test_predict_returns_a_score_with_reasons(client):
    body = client.post("/predict", json=ORDER).json()

    assert 0 <= body["risk_score"] <= 1
    assert 0 <= body["percentile"] <= 100
    assert body["inputs"]["customer_state"] == "RJ"  # lower-case input is normalized
    assert all("typical" in r["detail"] or "handed over" in r["detail"] for r in body["reasons"])
    assert abs(body["inputs"]["handling_days"] - 2.2083) < 1e-3
    assert abs(body["inputs"]["remaining_days"] - 9.375) < 1e-3  # to the end of the promised day
    assert abs(body["slack_days"] - body["inputs"]["transit_slack_days"]) < 0.01


def test_more_time_on_the_promise_never_raises_the_score(client):
    """The monotonic constraint, checked from the outside."""
    scores = []
    for promised in ("2018-05-04", "2018-05-08", "2018-05-12", "2018-05-20", "2018-06-10"):
        scores.append(client.post("/predict", json={**ORDER, "promised_date": promised}).json()["risk_score"])
    assert all(later <= earlier + 1e-9 for earlier, later in zip(scores, scores[1:], strict=False)), scores


@pytest.mark.parametrize(
    "change",
    [
        {"handoff_ts": "2018-04-30T10:00:00"},  # handed over before it was bought
        {"promised_date": "2018-04-20"},  # promised before it was bought
        {"weight_kg": 0},
        {"conditions": {"not_a_real_input": 1.0}},
        {"customer_state": "Rio"},
    ],
)
def test_bad_requests_are_rejected(client, change):
    assert client.post("/predict", json={**ORDER, **change}).status_code == 422


def test_live_conditions_override_the_defaults(client):
    calm = client.post("/predict", json=ORDER).json()
    stressed = client.post("/predict", json={**ORDER, "conditions": {"net_late_rate_14d": 0.4}}).json()
    assert stressed["inputs"]["net_late_rate_14d"] == 0.4
    assert calm["inputs"]["net_late_rate_14d"] != 0.4


def test_shipped_model_loads_and_scores():
    """Catches a saved model that no longer matches the pinned library versions."""
    path = config.ARTIFACT_DIR / "handoff_model.joblib"
    if not path.exists():
        pytest.skip("no shipped model in this checkout")
    serve.load.cache_clear()
    body = TestClient(serve.app).post("/predict", json=ORDER).json()
    serve.load.cache_clear()
    assert 0 <= body["risk_score"] <= 1
