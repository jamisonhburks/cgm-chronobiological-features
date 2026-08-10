"""Model evaluation: per-class ROC/AUC, cross-validation, calibration points."""

from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             log_loss, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .data import feature_matrix


def per_class_roc_auc(ytest, proba, classes=(0, 1, 2)) -> dict:
    """One-vs-rest ROC curve and AUC for each class.

    Returns ``{class: {"fpr", "tpr", "thresholds", "auc"}}``.
    """
    results = {}
    for c in classes:
        fpr, tpr, thresholds = roc_curve(ytest, proba[:, c], pos_label=c)
        results[c] = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds,
                      "auc": auc(fpr, tpr)}
    return results


def cross_validate_report(model, Xtrain, ytrain, Xtest, ytest,
                          n_splits: int = 5, random_state: int = 42,
                          verbose: bool = True) -> dict:
    """Stratified K-fold CV on the training set + a held-out test evaluation.

    Mirrors the notebook's reporting block (accuracy + log-loss per fold, then a
    classification report on the test set). Returns the collected metrics.
    """
    Xtr, Xte = feature_matrix(Xtrain), feature_matrix(Xtest)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    accuracy_scores = cross_val_score(model, Xtr, ytrain, cv=kf, scoring="accuracy")
    log_losses = cross_val_score(model, Xtr, ytrain, cv=kf, scoring="neg_log_loss")

    model.fit(Xtr, ytrain)
    preds = model.predict(Xte)
    proba = model.predict_proba(Xte)

    metrics = {
        "cv_accuracy": accuracy_scores,
        "cv_log_loss": -log_losses,
        "test_accuracy": accuracy_score(ytest, preds),
        "test_log_loss": log_loss(ytest, proba),
        "report": classification_report(ytest, preds),
        "preds": preds,
        "proba": proba,
    }

    if verbose:
        print(f"Cross-Validation Accuracy Scores: {metrics['cv_accuracy']}")
        print(f"Cross-Validation Log Loss Scores: {list(metrics['cv_log_loss'])}")
        print(f"Mean CV Accuracy: {metrics['cv_accuracy'].mean()}")
        print(f"Mean CV Log Loss: {metrics['cv_log_loss'].mean()}")
        print(f"Test Accuracy: {metrics['test_accuracy']}")
        print(f"Test Log Loss: {metrics['test_log_loss']}")
        print("\nClassification Report on Test Data:")
        print(metrics["report"])

    return metrics


def calibration_points(ytest, proba, n_classes: int = 3, n_bins: int = 10):
    """One-vs-rest calibration curve points per class.

    Returns a list of ``(prob_true, prob_pred)`` arrays, one per class.
    """
    ytest = np.asarray(ytest)
    points = []
    for c in range(n_classes):
        binary = (ytest == c).astype(int)
        prob_true, prob_pred = calibration_curve(binary, proba[:, c],
                                                 pos_label=1, n_bins=n_bins)
        points.append((prob_true, prob_pred))
    return points
