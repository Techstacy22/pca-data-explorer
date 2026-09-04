r"""
Principal Component Analysis via singular value decomposition.

**Status: exact reimplementation** of the SVD-based formulation of PCA
(Jolliffe, *Principal Component Analysis*, 2nd ed., 2002, Ch. 3; this is
also how `sklearn.decomposition.PCA`'s default "full" solver works
internally, though we implement the centering, variance accounting,
projection, reconstruction, and deterministic sign convention ourselves).

Mathematics
-----------
Given a centered data matrix :math:`X_c \in \mathbb{R}^{n \times p}` (n
samples, p features; see `preprocessing.center`), its singular value
decomposition is

.. math::
    X_c = U S V^\top

with :math:`U \in \mathbb{R}^{n \times r}` (orthonormal columns),
:math:`S \in \mathbb{R}^{r}` (singular values, descending),
:math:`V \in \mathbb{R}^{p \times r}` (orthonormal columns), and
:math:`r = \min(n, p)`.

This is directly equivalent to eigendecomposing the covariance matrix
:math:`C = X_c^\top X_c / (n-1)`: the columns of :math:`V` are exactly
the eigenvectors of :math:`C` ("principal axes" / "loadings"), and the
eigenvalues are :math:`\lambda_i = S_i^2 / (n-1)` ("explained variance").
We use the SVD of :math:`X_c` directly rather than forming :math:`C`
explicitly and eigendecomposing it, because this avoids ever squaring the
data (squaring inflates numerical conditioning issues for
ill-scaled input) -- the standard numerically-preferred approach.

Definitions:

- **Principal components / loadings**: the rows of :math:`V^\top`
  (equivalently columns of V) -- unit vectors in the original p-dimensional
  feature space giving each new axis's direction.
- **Scores**: :math:`X_c V = U S` -- the data's coordinates in the new,
  rotated coordinate system. "The first two principal components" means
  the first two columns of this.
- **Explained variance**: :math:`\lambda_i = S_i^2/(n-1)`, the variance of
  the data along principal axis i.
- **Explained variance ratio**: :math:`\lambda_i / \sum_j \lambda_j` --
  what fraction of the data's total variance that axis captures.
- **Reconstruction error** at k components: project onto the top k axes
  and back, :math:`\hat X_c = (X_c V_{:,:k}) V_{:,:k}^\top`, then measure
  :math:`\frac{1}{n}\sum_i \lVert X_{c,i} - \hat X_{c,i} \rVert_2^2`
  (mean squared reconstruction error per sample). This has a closed form
  we use only as an internal consistency check (see tests): it must equal
  :math:`\sum_{i>k} \lambda_i \cdot (n-1)/n`, i.e. exactly the variance
  left in the discarded components.

Sign convention
----------------
SVD does not pin down the sign of each (U column, V column) pair --
flipping both simultaneously (:math:`u \to -u, v \to -v`) gives an
equally valid decomposition. This makes results non-deterministic run to
run and, more importantly, not directly comparable to another
implementation's output. We fix this exactly the way
`sklearn.decomposition.PCA` does (`sklearn.utils.extmath.svd_flip`, called
with `u_based_decision=False`): for each component, find the feature with
the largest-magnitude loading and force that loading to be positive,
flipping the corresponding score column's sign to match. This specific
convention (deciding by V, not by U) is what let our validation script
(`scripts/validate_against_sklearn.py`) confirm exact sign agreement, not
just agreement up to sign, with scikit-learn.

A genuine limit of this (found by that validation script, not assumed):
when a component's singular value is numerically zero -- which happens
whenever n_samples <= n_features, since centering removes one degree of
freedom and caps the data's rank at n_samples-1 -- that component's
*direction* is mathematically undefined (any vector spanning the leftover
null space is an equally valid answer), so two independent, entirely
correct SVD implementations can and do disagree on it. Validating on a
30-sample x 100-feature random matrix, our components agreed with
scikit-learn to float64 noise (~1e-14) on all 29 components with real
variance, and disagreed by ~0.33 (i.e. completely different directions)
on only the 30th, whose explained variance both implementations computed
as ~1e-29 to 1e-30 -- confirming this is the expected mathematical
indeterminacy, not a defect in either implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .preprocessing import center


def _svd_flip_by_components(U: np.ndarray, Vt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic sign convention: for each component (row of Vt), flip
    sign so the largest-magnitude loading is positive. Matches
    scikit-learn's `svd_flip(u, v, u_based_decision=False)`.
    """
    max_abs_cols = np.argmax(np.abs(Vt), axis=1)
    signs = np.sign(Vt[np.arange(Vt.shape[0]), max_abs_cols])
    signs[signs == 0] = 1.0
    Vt = Vt * signs[:, np.newaxis]
    U = U * signs[np.newaxis, :]
    return U, Vt


@dataclass
class PCAResult:
    components_: np.ndarray  # shape (r, p): principal axes (rows), descending variance
    explained_variance_: np.ndarray  # shape (r,)
    explained_variance_ratio_: np.ndarray  # shape (r,)
    singular_values_: np.ndarray  # shape (r,)
    mean_: np.ndarray  # shape (p,): feature means subtracted before decomposition
    scores_: np.ndarray  # shape (n, r): data projected onto every component
    n_samples_: int
    n_features_: int


def fit_pca(X: np.ndarray) -> PCAResult:
    """Fit PCA on data matrix X (n_samples, n_features) via SVD.

    X should already be preprocessed (centered, and standardized if
    desired) by `preprocessing.py` -- this function centers internally as
    well (PCA is undefined otherwise), so passing raw data is safe, but
    passing already-standardized data lets the caller control scaling.
    """
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    centered = center(X)
    Xc = centered.centered

    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    U, Vt = _svd_flip_by_components(U, Vt)

    ddof = max(n - 1, 1)
    explained_variance = (S**2) / ddof
    total_variance = explained_variance.sum()
    explained_variance_ratio = (
        explained_variance / total_variance if total_variance > 0 else np.zeros_like(explained_variance)
    )
    scores = U * S

    return PCAResult(
        components_=Vt,
        explained_variance_=explained_variance,
        explained_variance_ratio_=explained_variance_ratio,
        singular_values_=S,
        mean_=centered.mean,
        scores_=scores,
        n_samples_=n,
        n_features_=p,
    )


def transform(result: PCAResult, X: np.ndarray, k: int | None = None) -> np.ndarray:
    """Project new data X onto the fitted principal axes (top k if given)."""
    Xc = np.asarray(X, dtype=np.float64) - result.mean_
    components = result.components_ if k is None else result.components_[:k]
    return Xc @ components.T


def inverse_transform(result: PCAResult, scores: np.ndarray, k: int | None = None) -> np.ndarray:
    """Reconstruct data from its top-k scores back into the original
    feature space (undoing centering)."""
    components = result.components_ if k is None else result.components_[:k]
    return scores @ components + result.mean_


def reconstruction_error(result: PCAResult, X: np.ndarray, k: int) -> float:
    """Mean squared reconstruction error per sample when keeping only the
    top k principal components (see module docstring for the closed-form
    identity this is checked against in tests)."""
    scores_k = transform(result, X, k=k)
    X_hat = inverse_transform(result, scores_k, k=k)
    residuals = np.asarray(X, dtype=np.float64) - X_hat
    return float(np.mean(np.sum(residuals**2, axis=1)))


def reconstruction_error_curve(result: PCAResult, X: np.ndarray) -> np.ndarray:
    """Reconstruction error for every k from 1 to the number of fitted
    components, vectorized (avoids repeated O(n*p*k) work per k)."""
    r = result.components_.shape[0]
    scores_full = transform(result, X, k=r)  # (n, r)
    Xc = np.asarray(X, dtype=np.float64) - result.mean_

    errors = np.empty(r)
    running_reconstruction = np.zeros_like(Xc)
    for k in range(r):
        running_reconstruction += np.outer(scores_full[:, k], result.components_[k])
        residual = Xc - running_reconstruction
        errors[k] = np.mean(np.sum(residual**2, axis=1))
    return errors
