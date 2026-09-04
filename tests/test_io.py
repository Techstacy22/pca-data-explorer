import io

import numpy as np
import pandas as pd
import pytest

from pca_explorer.io import load_csv, validate_features, DataValidationError


def _csv_bytes(df: pd.DataFrame) -> io.StringIO:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


def test_load_csv_splits_off_label_column():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "species": ["x", "y", "x"]})
    features, labels = load_csv(_csv_bytes(df), label_column="species")
    assert list(features.columns) == ["a", "b"]
    assert list(labels) == ["x", "y", "x"]


def test_load_csv_without_label_column():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    features, labels = load_csv(_csv_bytes(df))
    assert labels is None
    assert list(features.columns) == ["a", "b"]


def test_load_csv_missing_label_column_raises():
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(DataValidationError):
        load_csv(_csv_bytes(df), label_column="nonexistent")


def test_validate_features_drops_non_numeric_columns():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "note": ["x", "y"]})
    result = validate_features(df)
    assert list(result.columns) == ["a", "b"]


def test_validate_features_rejects_all_non_numeric():
    df = pd.DataFrame({"note": ["x", "y"], "other": ["z", "w"]})
    with pytest.raises(DataValidationError):
        validate_features(df)


def test_validate_features_rejects_missing_values():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [1.0, 2.0, 3.0]})
    with pytest.raises(DataValidationError):
        validate_features(df)


def test_validate_features_rejects_too_few_rows():
    df = pd.DataFrame({"a": [1.0]})
    with pytest.raises(DataValidationError):
        validate_features(df)


def test_validate_features_rejects_all_constant_columns():
    df = pd.DataFrame({"a": [5.0, 5.0, 5.0], "b": [1.0, 1.0, 1.0]})
    with pytest.raises(DataValidationError):
        validate_features(df)


def test_validate_features_allows_some_constant_columns():
    df = pd.DataFrame({"a": [5.0, 5.0, 5.0], "b": [1.0, 2.0, 3.0]})
    result = validate_features(df)
    assert result.shape == (3, 2)
