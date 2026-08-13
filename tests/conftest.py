"""Shared fixtures: a tiny synthetic frame with the real column structure.

Lets the suite run in milliseconds without the ~331 MB parquet or any model
training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import data


@pytest.fixture
def synthetic_frame() -> pd.DataFrame:
    """60 rows, 4 patients (15 rows each), real column names.

    Columns = META_COLUMNS + four kept whole-day stats + every
    MODEL1_DROP_FEATURES column + the label. Demographics are held constant so
    each patient forms one stratifiable group of 15 for ``stratified_split``
    (few enough strata that a 20% test split still covers every group).
    """
    rng = np.random.default_rng(0)
    n = 60

    frame = {c: rng.normal(size=n) for c in ["Mean", "STD", "Max", "Min"]}
    for col in data.MODEL1_DROP_FEATURES:
        frame[col] = rng.normal(size=n)

    df = pd.DataFrame(frame)
    df["PID"] = np.repeat(np.arange(4), 15)
    df["Age"] = 50
    df["Gender"] = pd.Categorical(["Male"] * n)
    df["Treatment"] = pd.Categorical(["no"] * n)
    df[data.LABEL_COLUMN] = rng.integers(0, 3, size=n)
    return df
