"""Reusable analysis code for the chronobiological CGM prediction paper.

Public API for the notebook to import, e.g.::

    from src import (load_dataset, split_features_label, stratified_split,
                     build_classifier, fit_predict, per_class_roc_auc,
                     cross_validate_report, plot_calibration_curves)
"""

__version__ = "1.0.0"

from .data import (COMPLEXITY_FEATURES, COMPLEXITY_TEMPORAL_FEATURES,
                   DEFAULT_DATA_PATH, INTRADAY_STAT_FEATURES, LABEL_COLUMN,
                   META_COLUMNS, MODEL1_DROP_FEATURES, TEMPORAL_FEATURES,
                   feature_matrix, load_dataset, split_features_label,
                   stratified_split)
from .evaluation import (calibration_points, cross_validate_report,
                         per_class_roc_auc)
from .models import build_classifier, fit_predict
from .plots import (cosine_similarity, pca_biplot,
                    plot_calibration_curves, plot_feature_importance_comparison)

__all__ = [
    "DEFAULT_DATA_PATH", "LABEL_COLUMN", "META_COLUMNS",
    "COMPLEXITY_FEATURES", "TEMPORAL_FEATURES", "COMPLEXITY_TEMPORAL_FEATURES",
    "INTRADAY_STAT_FEATURES", "MODEL1_DROP_FEATURES",
    "feature_matrix", "load_dataset", "split_features_label", "stratified_split",
    "build_classifier", "fit_predict",
    "per_class_roc_auc", "cross_validate_report", "calibration_points",
    "plot_calibration_curves", "plot_feature_importance_comparison",
    "pca_biplot", "cosine_similarity",
]
