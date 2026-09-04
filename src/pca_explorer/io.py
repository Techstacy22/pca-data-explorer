r"""
CSV loading and validation for user-uploaded datasets.

PCA requires a purely numeric feature matrix. Real uploaded CSVs commonly
include a non-numeric "label" column (a class/category to color points
by, not a feature to analyze) and sometimes stray non-numeric or missing
values in otherwise-numeric columns. This module separates the numeric
feature matrix from an optional label column and validates it before any
PCA computation is attempted, so failures are reported clearly rather
than surfacing as a cryptic linear-algebra error.
"""

from __future__ import annotations

import io as _io

import numpy as np
import pandas as pd


class DataValidationError(ValueError):
    pass


def load_csv(file_like_or_path, label_column: str | None = None) -> tuple[pd.DataFrame, pd.Series | None]:
    """Load a CSV, split off an optional label column, and return
    (numeric_features_df, labels_or_None).
    """
    df = pd.read_csv(file_like_or_path)
    labels = None
    if label_column is not None:
        if label_column not in df.columns:
            raise DataValidationError(f"Label column '{label_column}' not found in CSV columns: {list(df.columns)}")
        labels = df[label_column]
        df = df.drop(columns=[label_column])

    return df, labels


def validate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only numeric columns, drop columns that are entirely missing,
    and raise a clear error for the failure modes that would otherwise
    surface as an opaque linear-algebra exception.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    dropped = set(df.columns) - set(numeric_df.columns)

    if numeric_df.shape[1] == 0:
        raise DataValidationError(
            "No numeric columns found. PCA requires at least one numeric feature column "
            f"(non-numeric columns present: {sorted(dropped) if dropped else 'none'})."
        )

    all_missing = [c for c in numeric_df.columns if numeric_df[c].isna().all()]
    numeric_df = numeric_df.drop(columns=all_missing)
    if numeric_df.shape[1] == 0:
        raise DataValidationError("All numeric columns are entirely missing values.")

    if numeric_df.isna().any().any():
        n_missing = int(numeric_df.isna().sum().sum())
        raise DataValidationError(
            f"{n_missing} missing value(s) found in numeric columns "
            f"{[c for c in numeric_df.columns if numeric_df[c].isna().any()]}. "
            "Remove or impute missing values before running PCA."
        )

    constant_cols = [c for c in numeric_df.columns if numeric_df[c].nunique() <= 1]
    if len(constant_cols) == numeric_df.shape[1]:
        raise DataValidationError("Every numeric column is constant (zero variance); PCA has nothing to analyze.")

    if numeric_df.shape[0] < 2:
        raise DataValidationError(f"Need at least 2 rows (samples) to run PCA; found {numeric_df.shape[0]}.")

    return numeric_df


def dataset_summary(df: pd.DataFrame, dropped_columns: list[str] | None = None) -> dict:
    return {
        "n_samples": df.shape[0],
        "n_features": df.shape[1],
        "feature_names": list(df.columns),
        "dropped_columns": dropped_columns or [],
        "constant_columns": [c for c in df.columns if df[c].nunique() <= 1],
    }
