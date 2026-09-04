import numpy as np
import pytest

from pca_explorer.pca import (
    fit_pca,
    transform,
    inverse_transform,
    reconstruction_error,
    reconstruction_error_curve,
)


def test_pca_hand_calculated_rank_one_data():
    # All variance along the [1,1] direction, none orthogonal to it --
    # covariance matrix is [[4,4],[4,4]], eigenvalues 8 and 0 (worked out
    # by hand in the module's test docstring-equivalent comment below).
    #   X^T X (already centered, mean=[0,0]) = [[8,8],[8,8]]; /(n-1=2) = [[4,4],[4,4]]
    #   trace=8, det=0 -> eigenvalues {8, 0}; eigenvector for 8 is [1,1]/sqrt(2)
    X = np.array([[2.0, 2.0], [0.0, 0.0], [-2.0, -2.0]])
    result = fit_pca(X)

    np.testing.assert_allclose(result.explained_variance_, [8.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(result.explained_variance_ratio_, [1.0, 0.0], atol=1e-10)

    expected_component1 = np.array([1.0, 1.0]) / np.sqrt(2)
    np.testing.assert_allclose(result.components_[0], expected_component1, atol=1e-10)

    expected_scores_col0 = np.array([2 * np.sqrt(2), 0.0, -2 * np.sqrt(2)])
    np.testing.assert_allclose(result.scores_[:, 0], expected_scores_col0, atol=1e-10)
    # second component carries zero variance -> every point's score on it is ~0
    np.testing.assert_allclose(result.scores_[:, 1], 0.0, atol=1e-10)


def test_components_are_orthonormal():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 6))
    result = fit_pca(X)
    V = result.components_
    np.testing.assert_allclose(V @ V.T, np.eye(V.shape[0]), atol=1e-8)


def test_explained_variance_ratio_sums_to_one():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(80, 5)) * np.array([10, 1, 5, 0.1, 3])
    result = fit_pca(X)
    assert result.explained_variance_ratio_.sum() == pytest.approx(1.0, abs=1e-10)


def test_explained_variance_matches_score_variance():
    # By construction, PC scores are uncorrelated with variance = explained_variance_.
    rng = np.random.default_rng(4)
    X = rng.normal(size=(300, 4))
    result = fit_pca(X)
    score_variances = result.scores_.var(axis=0, ddof=1)
    np.testing.assert_allclose(score_variances, result.explained_variance_, rtol=1e-8)

    score_cov = np.cov(result.scores_, rowvar=False)
    off_diagonal = score_cov - np.diag(np.diag(score_cov))
    np.testing.assert_allclose(off_diagonal, 0.0, atol=1e-8)


def test_full_reconstruction_is_exact():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(40, 7))
    result = fit_pca(X)
    scores_full = transform(result, X)
    X_hat = inverse_transform(result, scores_full)
    np.testing.assert_allclose(X, X_hat, atol=1e-8)


def test_reconstruction_error_matches_closed_form_discarded_variance():
    # Residual sum-of-squares after keeping k components must equal
    # exactly the variance left in the discarded components, scaled by
    # (n-1)/n (see pca.py docstring for the derivation).
    rng = np.random.default_rng(6)
    X = rng.normal(size=(60, 5)) * np.array([20, 5, 1, 0.5, 0.1])
    result = fit_pca(X)
    n = X.shape[0]

    for k in range(1, 5):
        err = reconstruction_error(result, X, k=k)
        closed_form = result.explained_variance_[k:].sum() * (n - 1) / n
        assert err == pytest.approx(closed_form, rel=1e-6)


def test_reconstruction_error_curve_is_non_increasing_and_ends_near_zero():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(50, 6))
    result = fit_pca(X)
    errors = reconstruction_error_curve(result, X)

    assert np.all(np.diff(errors) <= 1e-8)  # monotonically non-increasing
    assert errors[-1] == pytest.approx(0.0, abs=1e-6)  # keeping all components -> no error


def test_transform_of_new_data_uses_fitted_mean_and_components():
    rng = np.random.default_rng(8)
    X_train = rng.normal(size=(100, 3))
    result = fit_pca(X_train)

    X_new = np.array([[1.0, 2.0, 3.0]])
    projected = transform(result, X_new, k=2)
    expected = (X_new - result.mean_) @ result.components_[:2].T
    np.testing.assert_allclose(projected, expected)
