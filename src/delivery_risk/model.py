"""Model definitions. Small on purpose: one boosted-tree pipeline and one linear baseline."""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from . import config

# Settings compared on the development folds. See reports/selection.json for the outcome.
CANDIDATES: dict[str, dict] = {
    "gbm_default": {"max_iter": 300, "max_leaf_nodes": 31, "min_samples_leaf": 50, "l2_regularization": 1.0},
    "gbm_regularized": {"max_iter": 200, "max_leaf_nodes": 15, "min_samples_leaf": 200, "l2_regularization": 10.0},
    "gbm_shallow": {"max_iter": 400, "max_depth": 3, "min_samples_leaf": 100, "l2_regularization": 5.0},
}
# The winner on the development folds, per stage. Set by hand after running select.py.
SELECTED = {"checkout": "gbm_shallow", "handoff": "gbm_regularized"}


def make_gbm(stage: config.Stage, **overrides) -> Pipeline:
    """Gradient-boosted trees. Missing numbers are handled by the trees themselves.

    Categorical inputs, when a stage has any, become integer codes with rare
    values sharing one code.
    """
    encode = ColumnTransformer(
        [
            (
                "categories",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, max_categories=25),
                list(stage.categorical),
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    params = {
        "learning_rate": 0.05,
        "categorical_features": list(range(len(stage.categorical))) or None,
        "monotonic_cst": [stage.monotone.get(name, 0) for name in stage.features],
        "random_state": config.SEED,
        **CANDIDATES[SELECTED[stage.name]],
        **overrides,
    }
    return Pipeline([("encode", encode), ("trees", HistGradientBoostingClassifier(**params))])


def make_logistic(stage: config.Stage) -> Pipeline:
    """Logistic regression baseline: one-hot categories, scaled numbers, median for missing."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    prepare = ColumnTransformer(
        [
            (
                "categories",
                OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=25),
                list(stage.categorical),
            ),
            ("numbers", numeric, list(stage.numeric)),
        ]
    )
    return Pipeline([("prepare", prepare), ("logit", LogisticRegression(max_iter=2000, C=0.5))])
