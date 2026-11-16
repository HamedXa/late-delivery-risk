# Model card: late-delivery risk at carrier handoff

Version 1.0.0. Trained on 96,469 Olist orders handed to the carrier through 11 September 2018.

## What the model does

It scores an e-commerce order at the moment the carrier receives the parcel. A higher score means the order is more likely to reach the customer after the promised date. The model is scikit-learn's `HistGradientBoostingClassifier` inside a pipeline that encodes the two state inputs.

## Intended use

Ranking open orders so that a support or logistics team with limited time looks at the riskiest ones first. The team might expedite the parcel, chase the carrier, or message the customer before the delay shows.

In batch use, score the day's handoffs and review the top 10%. The API also returns a fixed threshold for single-order calls. That threshold is a convenience for the demo and should be recomputed on current scores in any real use.

## Not intended for

- Decisions about people. The score describes a parcel's journey. It should not be used to penalize a seller or to limit service to a customer.
- Quoting a probability. The score ranks orders. Its level drifts with the network's late rate (see Calibration).
- Another marketplace or carrier network without retraining and a fresh backtest.
- Scoring at checkout. A separate checkout model was tested and was much weaker.

## Data

The public Olist dataset: 96,470 delivered orders bought between 15 September 2016 and 29 August 2018 on a Brazilian marketplace. 6,534 of them (6.8%) were late.

An order is late when it reaches the customer on a calendar day after the promised date. Delivery on the promised day counts as on time.

About 3% of orders in the raw data never reached the delivered status. They have no outcome and are left out.

## Inputs

24 inputs, all known at handoff.

| Group | Inputs |
| --- | --- |
| The promise | `handling_days`, `remaining_days`, `transit_slack_days`, `promised_days` |
| The handoff | `handoff_dow`, `handoff_hour` |
| The parcel | `n_items`, `order_value`, `freight_value`, `weight_kg`, `volume_l` |
| The route | `customer_state`, `seller_state`, `distance_km`, `same_state`, `route_transit_days_60d` |
| The seller | `seller_late_rate_prior` |
| The network | `net_in_transit`, `net_overdue_share`, `net_late_rate_14d`, `net_transit_days_14d` |
| The customer's state | `state_overdue_share`, `state_late_rate_30d`, `state_transit_days_14d` |

Every history input uses only events before the handoff, and another order's outcome counts only after that order was delivered. The model receives no names, addresses or customer identifiers. Location enters as the two states and a distance computed from zip-code prefixes.

Two inputs are constrained: more `remaining_days` and more `transit_slack_days` can never raise the score.

## Training

Settings were compared on four development months (October 2017 to January 2018) by the share of late orders caught at a 10% alert budget. The chosen settings: learning rate 0.05, 200 iterations, at most 15 leaves per tree, at least 200 orders per leaf, L2 regularization 10. Random seed 42. `reports/selection.json` holds the comparison.

## Evaluation

Walk-forward over seven months, February to August 2018. Each month is scored by a model trained only on orders delivered before that month began. 46,431 orders, 3,770 late (8.1%), alerts on the riskiest 10% of each month.

| Method | Late orders caught | Alerts that were late | Average precision | ROC AUC |
| --- | ---: | ---: | ---: | ---: |
| Random alerts | 10.0% | 8.1% | 0.081 | 0.500 |
| Slack rule | 34.7% | 28.2% | 0.315 | 0.741 |
| Logistic regression | 35.2% | 28.6% | 0.324 | 0.801 |
| This model | 36.2% | 29.4% | 0.344 | 0.794 |

The model's lead over the slack rule is 1.5 points. A paired bootstrap over orders (1,000 resamples) gives a 95% interval of 0.4 to 2.4 points.

By month, the share of late orders caught ranged from 24.7% (March 2018) to 64.0% (August 2018). In March 20.3% of orders were late, so a 10% budget could have caught 49% at most.

The evaluation months are not an untouched test set. The README explains what was seen before the split was frozen.

### By customer region

| Region | Orders | Late | Share alerted | Late orders caught | Alerts that were late |
| --- | ---: | ---: | ---: | ---: | ---: |
| Southeast | 32,475 | 7.3% | 8.8% | 41.7% | 34.6% |
| South | 6,397 | 6.8% | 6.6% | 22.5% | 23.3% |
| Northeast | 4,088 | 15.4% | 27.1% | 35.0% | 19.9% |
| Centre-West | 2,725 | 8.6% | 6.0% | 18.3% | 26.4% |
| North | 746 | 13.7% | 12.3% | 14.7% | 16.3% |

The model works best where most of its training data comes from. In the Southeast it catches 42% of late orders. In the North, with 746 orders, it catches 15%. Northeast orders are alerted three times as often as Southeast orders, and four in five of those alerts are false. A team using the alerts should expect that imbalance, and a per-region budget may serve it better. The table is rewritten to `reports/by_region.csv` on every pipeline run.

## Calibration

The score is not a calibrated probability. In March 2018 the average score was 7.3% while 20.3% of orders were late. In June 2018 the average score was 2.5% while 1.5% were late. Lateness arrives in bursts that the backward-looking inputs register only afterwards.

## Monitoring

`monitor.py` compares each month's inputs with the previous 180 days using the population stability index, and compares the average score with the late rate once outcomes arrive. In the replay it flagged a real change in the source data: from May 2018 carrier scans stopped being stamped in the evening. It also flagged at least one input in every month, so its thresholds need tuning before use.

## Retraining

`python -m delivery_risk.pipeline` rebuilds the features, reruns the backtest and refits the served model on all orders with a known outcome. The backtest retrains monthly, and the reported numbers assume that cadence.

## Limits

- One marketplace, one country, 2016 to 2018.
- A backtest cannot show whether acting on an alert prevents a late delivery.
- Open-order counts leave out orders that were later cancelled.
- The saved model loads only with the pinned scikit-learn version in `requirements.txt`. A test checks that it loads.
