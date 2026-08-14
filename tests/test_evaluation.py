"""Tests for the metric helpers (no model training required)."""

from __future__ import annotations

import numpy as np

from cgm_features import evaluation


def test_per_class_roc_auc_perfect_separation():
    # Class i gets all its probability mass -> perfect one-vs-rest ranking.
    ytest = np.array([0, 1, 2, 0, 1, 2])
    proba = np.eye(3)[ytest]  # one-hot == perfectly separable
    roc = evaluation.per_class_roc_auc(ytest, proba)

    assert set(roc) == {0, 1, 2}
    for c in (0, 1, 2):
        assert roc[c]["auc"] == 1.0
        assert {"fpr", "tpr", "thresholds", "auc"} <= set(roc[c])


def test_calibration_points_one_entry_per_class():
    rng = np.random.default_rng(0)
    n = 200
    ytest = rng.integers(0, 3, size=n)
    proba = rng.dirichlet([1, 1, 1], size=n)

    points = evaluation.calibration_points(ytest, proba, n_classes=3, n_bins=5)
    assert len(points) == 3
    for prob_true, prob_pred in points:
        assert len(prob_true) == len(prob_pred)
        assert np.all((prob_true >= 0) & (prob_true <= 1))
