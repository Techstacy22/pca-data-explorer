r"""
Preprocessing: centering and (optional) standardization.

**Status: exact reimplementation** of the two preprocessing conventions
relevant to PCA:

- **Centering** (subtracting each feature's mean) is mandatory for PCA:
  without it, the first "principal component" would just point toward
  the data's centroid rather than its direction of maximum variance.
  This matches `sklearn.decomposition.PCA`, which always centers (and
  never scales) by default.

- **Standardization** (additionally dividing each feature by its standard
  deviation) is offered as an option, because real uploaded CSVs
  routinely mix features on very different scales (e.g. a "price in
  dollars" column and a "rating out of 5" column) -- without it, PCA is
  dominated by whichever feature happens to have the largest numeric
  range, which is rarely what a user exploring their own data wants. This
  matches the standard practice of applying `StandardScaler` before PCA
  on heterogeneous tabular data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CenterResult:
    centered: np.ndarray
    mean: np.ndarray


def center(X: np.ndarray) -> CenterResult:
    """Subtract each column's mean. Returns the centered matrix and the
    means (needed later to un-center reconstructions)."""
    mean = X.mean(axis=0)
    return CenterResult(centered=X - mean, mean=mean)


@dataclass
class StandardizeResult:
    standardized: np.ndarray
    mean: np.ndarray
    std: np.ndarray


def standardize(X: np.ndarray, ddof: int = 0) -> StandardizeResult:
    """Center and scale each column to unit variance.

    A column with zero variance (constant across all samples) would
    divide by zero; such columns are left centered-but-unscaled (a
    constant column carries no information for PCA either way, so this
    only avoids a NaN, not a modeling choice).
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=ddof)
    safe_std = np.where(std == 0, 1.0, std)
    return StandardizeResult(standardized=(X - mean) / safe_std, mean=mean, std=std)
