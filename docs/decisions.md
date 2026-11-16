# Decisions and the evidence behind them

Each entry gives the choice, the reason, and where to check it. Numbers come from `reports/` unless a snippet is given.

## 1. Late means a calendar day after the promised date

The promised date in the data has no time of day. A parcel that arrives at 3 pm on the promised day kept the promise, so `data.add_label` compares dates. Comparing raw timestamps would count that parcel as late and lift the late rate from 6.8% to 8.1%.

## 2. Score at carrier handoff

Two scoring moments were built with the same code: checkout and carrier handoff. `config.Stage` holds what differs between them, which is the scoring time, the inputs and the rule to beat.

At checkout the best method caught 26.9% of late orders at a 10% alert budget. At handoff the slack rule alone caught 34.7%. The seller's handling time is unknown at checkout, and it decides how much of the window the carrier gets. The handoff model is the one served.

## 3. Features are computed as of the scoring time

Rules, enforced in `features.py`:

- An input for an order scored at time *t* uses only events strictly before *t*.
- Another order's outcome (late or on time, its transit time) counts only once that order has been delivered.
- An order is open from purchase to delivery, and overdue once the promised day has ended without a delivery.

Every history input is a sum over earlier events. `sum_before` sorts the events once, takes a cumulative sum, and looks up each scoring time with a binary search. A count over the last 14 days is the difference between two such lookups. The table of 96,470 orders builds in about two seconds per scoring moment.

Checks in `tests/test_features.py`:

- The future-invariance test deletes all later purchases, moves later handoffs by two days and later deliveries by nine, and asserts that the inputs of earlier orders are unchanged.
- One input is recomputed with a plain loop and compared with the fast version.
- A three-order case checks by hand that a late order does not count against its seller until it has been delivered.

Planting a leak makes these fail. Three were tried: counting events at the same instant, counting a seller's late orders from the purchase date, and averaging route transit times from the handoff date.

## 4. Walk-forward backtest, with a wait for labels

`backtest.fold_split` trains on orders delivered before the month starts and scores the orders handed over during the month. Orders in transit on the first of the month are in neither set.

The comparison that justifies this is in `reports/metrics.json` under `shuffled_split`. A random 80/20 split of the same orders reports AUC 0.886 at handoff. The walk-forward number is 0.794.

## 5. The metric is late orders caught at a 10% alert budget

A team can act on a limited number of orders. The headline metric is the share of late orders inside the riskiest 10% of each month, with the budget applied month by month. Average precision is reported because it measures ranking quality when the positive class is rare, and ROC AUC because readers expect it.

Accuracy is not reported. Predicting "on time" for every order scores 93%.

## 6. Two baselines before any boosted model

- The slack rule: sort orders by days left on the promise minus the route's recent transit time.
- Logistic regression on the same inputs.

The boosted model had to beat both. At handoff it did, narrowly: 36.2% against 34.7% and 35.2%. At checkout it lost to both.

How sure is the lead over the rule? `metrics.bootstrap_gap` redraws each month's orders with replacement 1,000 times, re-issues both sets of alerts on the redrawn orders, and records the difference. At handoff the lead is 1.5 points with a 95% interval of 0.4 to 2.4, and the model is ahead in 99% of the resamples. At checkout the model trails the rule by 0.9 points, with an interval of -2.4 to +0.5. The interval measures the luck of which orders arrived. It says nothing about choices made while building the model, which section 9 covers.

The rule also exists as SQL. `sql/slack_rule.sql` puts every delivery and every handoff on one timeline per route and reads the 60-day average off a window frame that ends one second before the handoff, so a parcel still on the road cannot enter the average. `python -m delivery_risk.sql_rule` runs it in SQLite on the full dataset: the slack it returns differs from the Python feature by at most 2e-12 days across all 96,469 orders. It issues 4,645 alerts where Python issues 4,644, because SQL rounds 692.5 up and Python rounds it to the even number, and both catch 1,309 late orders.

## 7. scikit-learn's histogram gradient boosting

It handles missing values without imputation, takes categorical inputs, and supports monotonic constraints. It adds no dependency beyond scikit-learn. XGBoost and LightGBM implement the same idea and were not tried.

## 8. Three settings compared, and they barely differ

On the development months (October 2017 to January 2018), late orders caught at handoff:

| Candidate | Late orders caught | Average precision per month |
| --- | ---: | ---: |
| `gbm_default` | 33.0% | 0.305 |
| `gbm_regularized` (chosen) | 33.1% | 0.308 |
| `gbm_shallow` | 32.4% | 0.308 |
| 11-input feature set | 35.4% | 0.322 |
| Logistic regression | 26.2% | 0.245 |
| Slack rule | 32.7% | 0.284 |

The choice among the three settings is close to a coin flip. The choice of inputs moves the result more than the settings do.

## 9. The 24-input model stayed, against the selection rule

The 11-input set won on the development months and lost on the evaluation months (34.0% against 36.2%). Evaluation-month results for the 24-input model had been seen during exploration, before the split was frozen, and that model was kept.

This is the weakest point in the project. Following the rule strictly would have shipped the 11-input model, which finished below the slack rule (34.7%). The reported lead of the model over the rule is the favourable case.

## 10. Two constrained inputs, and six that were left free

More days left on the promise and more slack can never raise the score. The constraint stops the model from learning a noisy reversal in a thin region of the data, and it helped on the development months.

Constraining the six network and state inputs in the other direction was tested and dropped.

| Development months, handoff | Late orders caught | Pooled average precision |
| --- | ---: | ---: |
| No constraints | 32.1% | 0.269 |
| Two constraints (shipped) | 33.1% | 0.280 |
| Eight constraints | 32.6% | 0.243 |

To rerun it:

```python
from dataclasses import replace

from delivery_risk import backtest, config, data, features, metrics, model

table = features.build_features(data.build_orders(data.load_raw()), config.HANDOFF)
worse_means_riskier = dict.fromkeys(
    ["net_overdue_share", "net_late_rate_14d", "net_transit_days_14d",
     "state_overdue_share", "state_late_rate_30d", "state_transit_days_14d"],
    1,
)
stages = {
    "no constraints": replace(config.HANDOFF, monotone={}),
    "two constraints (shipped)": config.HANDOFF,
    "eight constraints": replace(config.HANDOFF, monotone={**config.HANDOFF.monotone, **worse_means_riskier}),
}
for label, stage in stages.items():
    preds = backtest.walk_forward(table, stage, config.DEV_FOLDS, lambda stage=stage: model.make_gbm(stage))
    print(label, metrics.pooled(preds))
```

The reason the extra constraints hurt is in the data. Among orders handed over since October 2017, the fifth with the highest 14-day network late rate were late 5.8% of the time, against 8.0% overall. A burst of lateness is usually over by the time the trailing rate peaks. With `table` from the snippet above:

```python
import pandas as pd

recent = table[table["handoff_ts"] >= "2017-10-01"]
fifth = pd.qcut(recent["net_late_rate_14d"], 5, labels=False)
print(recent.groupby(fifth)["is_late"].mean().round(3))
print(round(recent["is_late"].mean(), 3))
```

## 11. The alert threshold comes from scores on unseen orders

The served model is trained on every order, so its own scores for recent late orders run high. The 90th percentile of those in-sample scores is 0.136. The 90th percentile of backtest scores for the same period, from models that had not seen the orders, is 0.110. Using the in-sample value would have alerted on 8% of new orders instead of 10%. `train.reference_scores` uses the backtest scores.

## 12. Explanations are grouped what-ifs

For each group of inputs, the API re-scores the order with that group set to typical values and reports the difference. Inputs that only make sense together are grouped, and slack is recomputed whenever one of its parts changes.

SHAP values were not used. Slack is derived from two other inputs, and per-input attributions would split one effect across three columns. The what-if answers the question a support agent asks: what about this order is unusual, and how much does it matter.

## 13. Defaults in the API

A caller knows the order. It usually does not know the state of the network. The API fills those inputs from the training data.

- Order-level typical values are medians over the last 180 days.
- Network, state and route conditions use the last 60 days. The 180-day median was stale: for São Paulo to Rio de Janeiro it gave 17.0 days of route transit, a figure dominated by the disruption of early 2018, against 9.0 days over the last 60.
- A missing distance is filled with the median distance of recent orders on the same route. The first version used the distance between state centres, which is zero for a same-state order.

A caller with live values passes them in `conditions`.

## 14. Drift is measured on the order's own inputs

Network inputs take nearly the same value for every order in a given week, so a distribution test on them fires every month and says nothing. In the seven evaluation months, at least seven of the eight network, state and route inputs were over the 0.25 threshold every time. `monitor.replay` computes the population stability index on order-level inputs only.

It still flags at least one input in all seven evaluation months. The seller's late-rate input is smoothed toward the network rate and moves with it. The thresholds need work before this could page anyone.

One flag was a real change in the source data. The index for `handoff_hour` reached 4.5 in May 2018 (`reports/monitoring.csv`). The share of carrier scans stamped at 18:00 or later, by month, with `table` from the snippet in section 10:

```python
evening = table["handoff_hour"] >= 18
print(evening.groupby(table["handoff_ts"].dt.to_period("M")).mean().loc["2018-01":"2018-08"].round(3))
```

It runs between 0.53 and 0.65 from January to April and at about 0.03 from May on.

## 15. Pinned versions and a shipped model

A scikit-learn model saved with joblib loads reliably only with the version that wrote it. `requirements.txt` pins every package, and `test_shipped_model_loads_and_scores` fails if the saved model and the installed versions drift apart. The model is 0.5 MB and is committed so the API, the tests and the Docker image work without the data.

## What would come next

1. A test period that nobody has looked at. With this dataset that is no longer possible, so it means new data.
2. An experiment that measures what an alert is worth: act on a random half of the alerts and compare outcomes.
3. A separate alert budget per region. The model card shows how uneven the alerts are.
4. Tuned drift thresholds, input by input.
