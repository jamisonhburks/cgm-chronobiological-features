"""Tests for feature grouping, splitting, and loading."""

from __future__ import annotations

import pytest

from src import data


def test_feature_groupings_are_disjoint_and_sized():
    # Complexity (5) + temporal (13) = 18 non-statistical features.
    assert len(data.COMPLEXITY_TEMPORAL_FEATURES) == 18
    assert set(data.COMPLEXITY_FEATURES).isdisjoint(data.TEMPORAL_FEATURES)
    # Model 1 drops complexity/temporal (18) + intraday sub-window stats (15) = 33.
    assert len(data.MODEL1_DROP_FEATURES) == 33
    assert len(set(data.MODEL1_DROP_FEATURES)) == 33  # no duplicates


def test_split_features_label_separates_target(synthetic_frame):
    X, y = data.split_features_label(synthetic_frame)
    assert data.LABEL_COLUMN not in X.columns
    assert y.name == data.LABEL_COLUMN
    assert len(X) == len(y) == len(synthetic_frame)


def test_feature_matrix_drops_only_meta_columns(synthetic_frame):
    X, _ = data.split_features_label(synthetic_frame)
    feats = data.feature_matrix(X)
    assert not set(data.META_COLUMNS) & set(feats.columns)
    assert set(feats.columns) == set(X.columns) - set(data.META_COLUMNS)


def test_load_dataset_missing_file_points_to_doi(tmp_path):
    with pytest.raises(FileNotFoundError, match="10.6075/J0BR8SK9"):
        data.load_dataset(str(tmp_path / "does_not_exist.parquet"))


def test_load_dataset_statistical_only_drops_complex_features(synthetic_frame, tmp_path):
    path = tmp_path / "frame.parquet"
    synthetic_frame.to_parquet(path)

    full = data.load_dataset(str(path), statistical_only=False)
    stat = data.load_dataset(str(path), statistical_only=True)

    assert set(data.MODEL1_DROP_FEATURES) <= set(full.columns)
    assert not set(data.MODEL1_DROP_FEATURES) & set(stat.columns)
    # Only the drop-set columns differ.
    assert set(full.columns) - set(stat.columns) == set(data.MODEL1_DROP_FEATURES)


def test_stratified_split_is_reproducible_and_sized(synthetic_frame):
    X, y = data.split_features_label(synthetic_frame)
    a = data.stratified_split(X, y, test_size=0.2, random_state=42)
    b = data.stratified_split(X, y, test_size=0.2, random_state=42)

    Xtrain, Xtest, _, _ = a
    assert len(Xtest) == pytest.approx(len(X) * 0.2, abs=1)
    assert len(Xtrain) + len(Xtest) == len(X)
    # Same seed -> identical split.
    assert list(a[1].index) == list(b[1].index)
