"""The SQL version of the slack rule has to agree with the Python feature."""

from __future__ import annotations

import numpy as np

from delivery_risk import sql_rule


def test_sql_slack_matches_the_python_feature(handoff_features):
    result = sql_rule.run(handoff_features)

    expected = handoff_features["transit_slack_days"].to_numpy()
    computed = result["slack_days"].to_numpy(dtype=float)
    assert np.isfinite(expected).sum() > 500  # most orders have enough history to compare
    np.testing.assert_allclose(computed, expected, rtol=1e-9, atol=1e-9, equal_nan=True)


def test_sql_alerts_are_the_tenth_of_each_month_with_the_least_slack(handoff_features):
    result = sql_rule.run(handoff_features)
    result["sort_key"] = result["remaining_days"] - result["usual_transit_days"].fillna(0)

    for _, month in result.groupby("handoff_month"):
        alerts = month[month["alert"] == 1]
        assert abs(len(alerts) - 0.10 * len(month)) <= 0.5
        if len(alerts):
            assert alerts["sort_key"].max() <= month.loc[month["alert"] == 0, "sort_key"].min()
