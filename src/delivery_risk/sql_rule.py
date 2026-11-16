"""The slack rule in SQL, for a team that wants the cheap version of this project.

    python -m delivery_risk.sql_rule

sql/slack_rule.sql computes slack and a 10% alert list with window functions.
This module loads the orders into an in-memory SQLite database, runs the query
and compares the result with the Python feature.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from . import config, data, features

QUERY = config.ROOT / "sql" / "slack_rule.sql"


def run(table: pd.DataFrame) -> pd.DataFrame:
    """Run the query on a handoff feature table. Returns one row per order, in the table's order."""
    orders = pd.DataFrame(
        {
            "order_id": table["order_id"].to_numpy(),
            "seller_state": table["seller_state"].to_numpy(),
            "customer_state": table["customer_state"].to_numpy(),
            "handoff_ts": features.to_seconds(table["handoff_ts"]),
            "delivered_ts": features.to_seconds(table["delivered_ts"]),
            "overdue_ts": features.to_seconds(table["overdue_ts"]),
        }
    )
    connection = sqlite3.connect(":memory:")
    try:
        orders.to_sql("orders", connection, index=False)
        result = pd.read_sql_query(QUERY.read_text(encoding="utf-8"), connection)
    finally:
        connection.close()
    return result.set_index("order_id").loc[orders["order_id"]].reset_index()


def main() -> None:
    table = features.build_features(data.build_orders(data.load_raw()), config.HANDOFF)
    result = run(table)
    gap = np.nanmax(np.abs(result["slack_days"].to_numpy(dtype=float) - table["transit_slack_days"].to_numpy()))
    print(f"{len(result):,} orders, largest difference from the Python slack feature: {gap:.1e} days")

    evaluated = result["handoff_month"].isin(config.EVAL_FOLDS).to_numpy()
    late = table[config.TARGET].to_numpy()[evaluated]
    alert = result["alert"].to_numpy(dtype=bool)[evaluated]
    print(
        f"{config.EVAL_FOLDS[0]} to {config.EVAL_FOLDS[-1]}: {alert.sum():,} alerts, "
        f"{late[alert].sum():,} of {late.sum():,} late orders caught ({late[alert].sum() / late.sum():.1%})"
    )


if __name__ == "__main__":
    main()
