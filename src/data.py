"""Data loading, feature grouping, and train/test splitting.

Centralizes every assumption the notebook previously encoded as inline magic
(the ``.iloc[:, 4:]`` slice, the hand-typed drop list, the ``range(29)`` PCA
coloring) so those choices live in exactly one place.

Two distinct feature groupings are used in the paper and both are made explicit
here:

* **Model 1 vs Model 2** -- Model 1 uses only whole-day *statistical* summaries
  (:data:`MODEL1_STATISTICAL_FEATURES`); Model 2 adds the intraday, complexity,
  and temporal features. Model 1 is obtained by dropping
  :data:`MODEL1_DROP_FEATURES`.
* **Feature *type* (for PCA / importance coloring)** -- every feature is either
  "statistical" or "complexity/temporal"; the latter is
  :data:`COMPLEXITY_TEMPORAL_FEATURES` (the CI and Poincaré/SD families). Note
  the intraday sub-window stats count as *statistical* under this grouping.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_DATA_PATH = "processed_CGM_data_for_ML.parquet"

#: Identifier / demographic columns. Used to *stratify* the split, but NOT fed to
#: the model as features (the notebook trained on ``.iloc[:, 4:]``, i.e. everything
#: after these four columns).
META_COLUMNS = ["PID", "Age", "Gender", "Treatment"]

#: Prediction target: next-day change in area above 180 mg/dL (0 lower, 1 similar, 2 greater).
LABEL_COLUMN = "AreaAboveLabel"

# --- Feature *type* grouping (used for PCA and importance coloring) --------------

#: Multiscale complexity-index features.
COMPLEXITY_FEATURES = ["CI_all", "CI_early", "CI_late", "CI_mid", "CI_sleep"]

#: Poincaré / temporal variability features.
TEMPORAL_FEATURES = [
    "SD1", "SD2", "SDRatio", "SD1sleep", "SD2sleep", "SDRatiosleep",
    "SD1_24h_sleep", "SD2_24h_sleep", "SDR_24h_sleep",
    "YesterToday_SD1", "YesterToday_SD2", "YesterToday_SDR", "SleepEucDist",
]

#: The non-statistical feature type: complexity + temporal (18 features). Used to
#: color the PCA biplot and importance chart by feature type.
COMPLEXITY_TEMPORAL_FEATURES = COMPLEXITY_FEATURES + TEMPORAL_FEATURES

# --- Model 1 (statistical-only) grouping ----------------------------------------

#: Intraday (early/mid/late sub-window) statistical features. These are statistical
#: by *type*, but are excluded from the whole-day-only Model 1.
INTRADAY_STAT_FEATURES = [
    "AreaAbove_early", "AreaAbove_mid", "AreaAbove_late",
    "Median_early", "Median_mid", "Median_late",
    "STD_early", "STD_mid", "STD_late",
    "Kurt_early", "Kurt_mid", "Kurt_late",
    "MAD_early", "MAD_mid", "MAD_late",
]

#: Columns dropped to reduce the full table to Model 1's whole-day statistical
#: feature set (complexity + temporal + intraday sub-window stats).
MODEL1_DROP_FEATURES = COMPLEXITY_TEMPORAL_FEATURES + INTRADAY_STAT_FEATURES


def load_dataset(path: str = DEFAULT_DATA_PATH, statistical_only: bool = False) -> pd.DataFrame:
    """Load the processed CGM table, optionally reduced to Model 1's features.

    Parameters
    ----------
    path
        Path to ``processed_CGM_data_for_ML.parquet`` (download from DOI 10.6075/J0BR8SK9).
    statistical_only
        If True, drop complexity/temporal/intraday features (Model 1 feature set).
        If False, keep all features (Model 2).

    Rows with any missing value are dropped, matching the original analysis.
    """
    data = pd.read_parquet(path)
    if statistical_only:
        data = data.drop(columns=MODEL1_DROP_FEATURES)
    return data.dropna()


def split_features_label(data: pd.DataFrame):
    """Return (X, y): every column except the label, and the label."""
    y = data[LABEL_COLUMN]
    X = data.drop(columns=[LABEL_COLUMN])
    return X, y


def feature_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Drop identifier/demographic columns, leaving only model input features.

    Replaces the fragile positional ``X.iloc[:, 4:]`` with an explicit,
    order-independent column drop.
    """
    return X.drop(columns=[c for c in META_COLUMNS if c in X.columns])


def stratified_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2,
                     random_state: int = 42):
    """80/20 split stratified jointly on PID, Age, Gender, and Treatment."""
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=X.loc[:, META_COLUMNS],
        random_state=random_state,
    )
