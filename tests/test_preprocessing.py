import numpy as np
import pytest

from pca_explorer.preprocessing import center, standardize


def test_center_hand_calculated():
    X = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
    result = center(X)
    np.testing.assert_allclose(result.mean, [3.0, 20.0])
    np.testing.assert_allclose(result.centered, [[-2.0, -10.0], [0.0, 0.0], [2.0, 10.0]])


def test_center_output_has_zero_mean():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=[5, -3, 100], scale=[1, 2, 50], size=(200, 3))
    result = center(X)
    np.testing.assert_allclose(result.centered.mean(axis=0), 0.0, atol=1e-10)


def test_standardize_hand_calculated():
    X = np.array([[1.0], [2.0], [3.0]])
    result = standardize(X, ddof=0)
    expected_std = np.std([1.0, 2.0, 3.0])
    np.testing.assert_allclose(result.mean, [2.0])
    np.testing.assert_allclose(result.std, [expected_std])
    np.testing.assert_allclose(result.standardized.ravel(), (X.ravel() - 2.0) / expected_std)


def test_standardize_output_has_unit_variance():
    rng = np.random.default_rng(1)
    X = rng.normal(loc=[5, -3, 100], scale=[1, 2, 50], size=(500, 3))
    result = standardize(X, ddof=0)
    np.testing.assert_allclose(result.standardized.std(axis=0), 1.0, atol=1e-10)
    np.testing.assert_allclose(result.standardized.mean(axis=0), 0.0, atol=1e-10)


def test_standardize_constant_column_does_not_produce_nan():
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    result = standardize(X)
    assert not np.isnan(result.standardized).any()
    # constant column ends up all-zero after centering (std forced to 1)
    np.testing.assert_allclose(result.standardized[:, 1], 0.0)
