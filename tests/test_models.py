"""Tests for the classifier factory and fit/predict wiring."""

from __future__ import annotations

from src import data, models


def test_build_classifier_applies_defaults_and_overrides():
    clf = models.build_classifier(random_state=7)
    params = clf.get_params()
    assert params["n_estimators"] == models.DEFAULT_PARAMS["n_estimators"]
    assert params["random_state"] == 7

    swept = models.build_classifier(n_estimators=25, max_depth=5)
    assert swept.get_params()["n_estimators"] == 25
    assert swept.get_params()["max_depth"] == 5


def test_fit_predict_shapes_and_feature_selection(synthetic_frame):
    X, y = data.split_features_label(synthetic_frame)
    Xtrain, Xtest, ytrain, _ = data.stratified_split(X, y, random_state=0)

    clf = models.build_classifier(n_estimators=5, random_state=0)  # tiny = fast
    preds, proba = models.fit_predict(clf, Xtrain, ytrain, Xtest)

    assert len(preds) == len(Xtest)
    assert proba.shape == (len(Xtest), y.nunique())
    # fit_predict must strip the meta columns internally, never training on them.
    assert clf.n_features_in_ == X.shape[1] - len(data.META_COLUMNS)
