"""Figure helpers: calibration curves, feature-importance comparison, PCA biplot.

Each function returns the Matplotlib ``Axes`` so callers can further tweak or save
the figure (as both .png and .pdf, per project convention).
"""

from __future__ import annotations

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import norm

from .data import COMPLEXITY_TEMPORAL_FEATURES
from .evaluation import calibration_points

CLASS_LABELS = ["Lower AAR", "Similar AAR", "Greater AAR"]


def cosine_similarity(a, b) -> float:
    """Cosine similarity between two vectors."""
    return np.dot(a, b) / (norm(a) * norm(b))


def plot_calibration_curves(ytest, proba, title: str,
                            class_labels=CLASS_LABELS, ax=None):
    """Reliability (calibration) scatter for each one-vs-rest class."""
    if ax is None:
        _, ax = plt.subplots(figsize=(2, 2), dpi=200)
    for prob_true, prob_pred in calibration_points(ytest, proba, n_classes=len(class_labels)):
        ax.scatter(prob_pred, prob_true, alpha=0.75, s=5)
    ax.plot([0, 1], [0, 1], linestyle="dotted", c="k", alpha=0.5, zorder=-5)
    ax.set_ylabel("True")
    ax.set_xlabel("Pred")
    ax.set_yticks([0, 0.5, 1])
    ax.set_xticks([0, 0.5, 1])
    ax.legend(class_labels, fontsize=7)
    ax.set_title(title, fontsize=8)
    return ax


def plot_feature_importance_comparison(model_full, feats_full, model_stat, feats_stat,
                                       top_n: int = 20,
                                       complexity_temporal_features=COMPLEXITY_TEMPORAL_FEATURES,
                                       ax=None):
    """Horizontal bar chart comparing gain importance of the two models.

    Tick labels are colored by feature *type* (statistical vs. complexity/temporal)
    from ``complexity_temporal_features``, replacing the notebook's hardcoded
    ``complexTicks``/``statTicks`` index lists so it stays correct if the feature
    set changes.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=200)

    full = pd.DataFrame(model_full.feature_importances_, index=feats_full)
    stat = pd.DataFrame(model_stat.feature_importances_, index=feats_stat)
    imp = pd.concat([full, stat], axis=1)
    imp.columns = ["Both", "Stat"]
    imp["Feature"] = imp.index
    imp = imp.sort_values("Both", ascending=True).iloc[-top_n:, :]

    y = np.arange(len(imp.Feature))
    h_stat = ax.barh(y + 0.2, imp.Stat, 0.4, color="gray")
    h_both = ax.barh(y - 0.2, imp.Both, 0.4, color="black")
    ax.set_yticks(y, imp.Feature, fontsize=8)

    # Color each tick label by whether it is a statistical or complexity/temporal feature.
    complex_set = set(complexity_temporal_features)
    for label, feat in zip(ax.get_yticklabels(), imp.Feature):
        label.set_color("C1" if feat in complex_set else "C0")
        label.set_fontweight("bold")

    ax.set_title("XGBoost Model Feature Importances", fontsize=10)
    ax.set_xlabel("Feature Importance: Gain", fontsize=10)
    ax.legend([h_stat, h_both],
              ["Model 1: Statistical Features", "Model 2: Model 1 Features + Complexity/Temporal"])
    return ax


def pca_biplot(pca, scores, feature_names,
               complexity_temporal_features=COMPLEXITY_TEMPORAL_FEATURES,
               label_threshold: float = 0.25, ax=None):
    """PCA biplot of feature loadings, colored by feature type.

    Draws each feature's loading vector, labels the strong ones, and overlays the
    mean vector for the statistical vs. complexity/temporal groups plus the angle
    between them. Group membership comes from ``complexity_temporal_features``
    instead of the notebook's hardcoded index ranges.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6), dpi=200)

    complex_set = set(complexity_temporal_features)
    is_stat = np.array([f not in complex_set for f in feature_names])
    colors = np.where(is_stat, "C0", "C1")

    coeffs = np.transpose(pca.components_[0:2, :])
    coeff_mag = np.sqrt(coeffs[:, 0] ** 2 + coeffs[:, 1] ** 2)

    for i, name in enumerate(feature_names):
        ax.arrow(0, 0, coeffs[i, 0], coeffs[i, 1], color=colors[i],
                 alpha=0.35, linestyle="dotted", linewidth=2, zorder=0)
        if coeff_mag[i] >= label_threshold:
            ax.text(coeffs[i, 0], coeffs[i, 1] * 1.05, name, color="k",
                    ha="center", va="center", fontsize=8, fontweight="bold", zorder=2)

    # Mean loading vector for each feature group and the angle between them.
    mean_stat = coeffs[is_stat].mean(axis=0)
    mean_comp = coeffs[~is_stat].mean(axis=0)
    ax.arrow(0, 0, *mean_comp, color="C1", alpha=1, linewidth=4, zorder=1)
    ax.arrow(0, 0, *mean_stat, color="C0", alpha=1, linewidth=4, zorder=1)
    deg = int(np.round(np.rad2deg(np.arccos(cosine_similarity(mean_stat, mean_comp)))))

    ax.set_xlim(-0.4, 0.4)
    ax.set_ylim(-0.15, 0.4)
    ax.set_xlabel(f"PC1 (Explained Variance: {int(round(pca.explained_variance_ratio_[0] * 100))}%)")
    ax.set_ylabel(f"PC2 (Explained Variance: {int(round(pca.explained_variance_ratio_[1] * 100))}%)")
    ax.grid(True)
    ax.set_title("Features with Correlation to PCs", fontsize=10)
    ax.text(-0.35, -0.13, f"Angle Between Feature-Type Mean Vectors: {deg}°", fontweight="bold")
    return ax, deg
