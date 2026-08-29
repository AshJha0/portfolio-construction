"""Tests for the pinned linear algebra kernels."""
import numpy as np
import pytest

import portopt as po


def _spd(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n))
    return a @ a.T + n * np.eye(n)


def test_cholesky_matches_numpy_grid():
    for n in (1, 2, 5, 8, 12):
        a = _spd(n, 100 + n)
        low = po.cholesky(a)
        assert np.allclose(low, np.linalg.cholesky(a), atol=1e-12)
        assert np.allclose(low @ low.T, a, atol=1e-10)


def test_chol_solve_matches_numpy():
    a = _spd(8, 7)
    b = np.arange(1.0, 9.0)
    x = po.spd_solve(a, b)
    assert np.allclose(x, np.linalg.solve(a, b), atol=1e-11)


def test_spd_inverse():
    a = _spd(6, 3)
    inv = po.spd_inverse(a)
    assert np.allclose(inv @ a, np.eye(6), atol=1e-10)


def test_cholesky_rejects_indefinite():
    a = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues 3, -1
    with pytest.raises(ValueError, match="not positive definite"):
        po.cholesky(a)


def test_cholesky_rejects_singular_duplicated_asset(returns):
    dup = np.column_stack([returns, returns[:, 0]])
    cov = po.sample_cov(dup)
    with pytest.raises(ValueError, match="not positive definite"):
        po.cholesky(cov)


def test_cholesky_rejects_asymmetric_and_nonsquare():
    with pytest.raises(ValueError, match="not symmetric"):
        po.cholesky(np.array([[1.0, 0.5], [0.1, 1.0]]))
    with pytest.raises(ValueError, match="square"):
        po.cholesky(np.ones((2, 3)))


def test_cholesky_rejects_nan():
    a = np.eye(3)
    a[1, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        po.cholesky(a)
