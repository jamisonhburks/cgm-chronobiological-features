"""XGBoost classifier factory and fit/predict helpers.

The hyperparameters were duplicated verbatim across four notebook cells; they now
live here as a single source of truth (the grid-search-selected configuration).
"""

from __future__ import annotations

import xgboost as xgb

from .data import feature_matrix

#: Best configuration from the grid search in the paper.
DEFAULT_PARAMS = dict(
    n_estimators=500,
    max_depth=3,
    learning_rate=1,
    objective="multi:softmax",
    enable_categorical=True,
    tree_method="auto",
    importance_type="gain",
)


def build_classifier(random_state: int = 42, **overrides) -> xgb.XGBClassifier:
    """Construct the XGBoost classifier used throughout the analysis.

    Pass keyword ``overrides`` to sweep individual hyperparameters
    (e.g. ``build_classifier(n_estimators=200, max_depth=5)`` for grid search).
    """
    params = {**DEFAULT_PARAMS, **overrides}
    return xgb.XGBClassifier(random_state=random_state, **params)


def fit_predict(model, Xtrain, ytrain, Xtest):
    """Fit on the training features and return ``(preds, proba)`` for the test set.

    Feature selection (dropping the identifier/demographic columns) is applied
    internally via :func:`data.feature_matrix`, so callers pass the full frames.
    """
    model.fit(feature_matrix(Xtrain), ytrain)
    preds = model.predict(feature_matrix(Xtest))
    proba = model.predict_proba(feature_matrix(Xtest))
    return preds, proba
