"""Tests for sample moments and Ledoit-Wolf shrinkage."""
import numpy as np
import pytest

import portopt as po


def test_sample_moments_match_numpy(returns):
    assert np.allclose(po.sample_mean(returns), returns.mean(axis=0), atol=1e-15)
    assert np.allclose(
        po.sample_cov(returns), np.cov(returns, rowvar=False, ddof=1), atol=1e-15
    )


def test_shrinkage_intensity_in_unit_interval(returns):
    res = po.shrink_covariance(returns)
    assert 0.0 <= res.intensity <= 1.0
    assert -1.0 < res.rbar < 1.0


def test_shrunk_matrix_is_psd(returns):
    res = po.shrink_covariance(returns)
    po.cholesky(res.sigma)  # must not raise


def test_shrinkage_target_structure(returns):
    res = po.shrink_covariance(returns)
    s, f = res.sample, res.target
    assert np.allclose(np.diag(f), np.diag(s), atol=1e-15)
    sd = np.sqrt(np.diag(s))
    n = s.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            assert f[i, j] == pytest.approx(res.rbar * sd[i] * sd[j], abs=1e-15)


def test_shrinkage_convex_combination(returns):
    res = po.shrink_covariance(returns)
    recon = res.intensity * res.target + (1.0 - res.intensity) * res.sample
    assert np.allclose(res.sigma, recon, atol=1e-15)


def test_shrinkage_pulls_correlations_toward_rbar(returns):
    """Property-style grid check: every off-diagonal shrunk correlation lies
    (weakly) between the sample correlation and rbar."""
    res = po.shrink_covariance(returns)
    s, sh = res.sample, res.sigma
    n = s.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            c_s = s[i, j] / np.sqrt(s[i, i] * s[j, j])
            c_f = res.rbar
            c_sh = sh[i, j] / np.sqrt(sh[i, i] * sh[j, j])
            lo, hi = min(c_s, c_f) - 1e-12, max(c_s, c_f) + 1e-12
            assert lo <= c_sh <= hi


def test_annualization():
    mu = np.array([0.001, 0.002])
    cov = np.array([[1e-4, 2e-5], [2e-5, 3e-4]])
    assert np.allclose(po.annualize_mean(mu), 252 * mu)
    assert np.allclose(po.annualize_cov(cov), 252 * cov)
    with pytest.raises(ValueError):
        po.annualize_mean(mu, periods=0)


def test_input_validation_errors():
    with pytest.raises(ValueError, match="2-D"):
        po.sample_mean(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="at least 2 rows"):
        po.sample_cov(np.ones((1, 3)))
    bad = np.ones((10, 2))
    bad[3, 1] = np.inf
    with pytest.raises(ValueError, match="NaN or infinite"):
        po.sample_cov(bad)
    with pytest.raises(ValueError, match="at least 2 assets"):
        po.shrink_covariance(np.random.default_rng(0).standard_normal((50, 1)))
    with pytest.raises(ValueError, match="at least one asset"):
        po.sample_mean(np.ones((5, 0)))
