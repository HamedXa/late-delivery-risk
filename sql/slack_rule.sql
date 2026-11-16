-- The slack rule as one query.
--
-- For every order, at the moment the carrier receives the parcel:
--     slack = days left on the promise - usual transit time on the route
-- The 10% of each month's handoffs with the least slack get an alert.
--
-- "Usual transit time" is the average transit of parcels DELIVERED on the same
-- route (seller state to customer state) in the 60 days before the handoff.
-- A parcel still on the road has no transit time yet, so it cannot leak into
-- the average. A route with fewer than five such parcels falls back to the
-- whole network over the last 14 days.
--
-- Input: a table `orders`, one row per order, with
--     order_id, seller_state, customer_state,
--     handoff_ts, delivered_ts, overdue_ts
-- Times are Unix seconds. overdue_ts is the start of the day after the promised date.
--
-- Written for SQLite 3.28 or later, which added RANGE offsets in window frames.
-- tests/test_sql_rule.py checks the result against the Python feature.

WITH events AS (
    -- A delivery adds one known transit time to its route.
    SELECT seller_state || '>' || customer_state AS route,
           delivered_ts AS ts,
           (delivered_ts - handoff_ts) / 86400.0 AS transit_days,
           NULL AS order_id
    FROM orders
    UNION ALL
    -- A handoff asks what was known just before it.
    SELECT seller_state || '>' || customer_state AS route,
           handoff_ts AS ts,
           NULL AS transit_days,
           order_id
    FROM orders
),
history AS (
    SELECT order_id,
           COUNT(transit_days) OVER route_60d   AS route_parcels,
           AVG(transit_days)   OVER route_60d   AS route_transit_days,
           COUNT(transit_days) OVER network_14d AS network_parcels,
           AVG(transit_days)   OVER network_14d AS network_transit_days
    FROM events
    -- 5184000 seconds is 60 days, 1209600 is 14 days. "1 PRECEDING" keeps the frame strictly in the past.
    WINDOW route_60d   AS (PARTITION BY route ORDER BY ts RANGE BETWEEN 5184000 PRECEDING AND 1 PRECEDING),
           network_14d AS (ORDER BY ts RANGE BETWEEN 1209600 PRECEDING AND 1 PRECEDING)
),
slack AS (
    SELECT o.order_id,
           strftime('%Y-%m', o.handoff_ts, 'unixepoch') AS handoff_month,
           (o.overdue_ts - o.handoff_ts) / 86400.0 AS remaining_days,
           CASE WHEN h.route_parcels >= 5 THEN h.route_transit_days
                WHEN h.network_parcels >= 5 THEN h.network_transit_days
           END AS usual_transit_days
    FROM orders AS o
    JOIN history AS h ON h.order_id = o.order_id
)
SELECT order_id,
       handoff_month,
       remaining_days,
       usual_transit_days,
       remaining_days - usual_transit_days AS slack_days,
       ROW_NUMBER() OVER least_slack_first <= ROUND(0.10 * COUNT(*) OVER (PARTITION BY handoff_month)) AS alert
FROM slack
WINDOW least_slack_first AS (
    PARTITION BY handoff_month
    ORDER BY remaining_days - COALESCE(usual_transit_days, 0), order_id
)
ORDER BY order_id;
