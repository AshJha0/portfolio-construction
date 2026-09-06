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
        po.shrink_covariance(0.01 * np.random.default_rng(0).standard_normal((50, 1)))
    with pytest.raises(ValueError, match="at least one asset"):
        po.sample_mean(np.ones((5, 0)))
    with pytest.raises(ValueError, match="<= -1"):
        po.sample_mean(np.array([[0.01, -1.0], [0.02, 0.01]]))
    for bad in (0, -252, np.nan):
        with pytest.raises(ValueError, match="periods"):
            po.annualize_cov(np.eye(2), periods=bad)


def _covcor(x: np.ndarray):
    """Direct transcription of Ledoit & Wolf's covCor.m (JPM 2004, App. B):
    divisor T throughout, both theta terms per ordered pair."""
    t, n = x.shape
    xc = x - x.mean(axis=0)
    sample = (xc.T @ xc) / t
    var = np.diag(sample)
    sqrtvar = np.sqrt(var)
    rbar = (np.sum(sample / np.outer(sqrtvar, sqrtvar)) - n) / (n * (n - 1))
    prior = rbar * np.outer(sqrtvar, sqrtvar)
    np.fill_diagonal(prior, var)
    y = xc**2
    phi_mat = (y.T @ y) / t - 2 * (xc.T @ xc) * sample / t + sample**2
    phi = phi_mat.sum()
    term1 = ((xc**3).T @ xc) / t
    help_ = (xc.T @ xc) / t
    theta = term1 - np.diag(help_)[:, None] * sample - help_ * var[:, None] + var[:, None] * sample
    np.fill_diagonal(theta, 0.0)
    rho = np.sum(np.diag(phi_mat)) + rbar * np.sum(np.outer(1 / sqrtvar, sqrtvar) * theta)
    gamma = np.linalg.norm(sample - prior, "fro") ** 2
    delta = max(0.0, min(1.0, (phi - rho) / gamma / t))
    return delta, rbar, delta * prior + (1 - delta) * sample


def test_lw_rho_matches_paper_formula():
    """PT-1 (CRIT-1): a 12x3 hand panel with unequal vols, so that
    theta_ii,ij != theta_jj,ij and the old halved rho_hat would be wrong."""
    x = np.array(
        [
            [0.010, 0.020, -0.010],
            [-0.020, 0.000, 0.010],
            [0.030, -0.010, 0.020],
            [0.005, 0.040, -0.030],
            [-0.015, -0.030, 0.000],
            [0.025, 0.010, 0.015],
            [-0.010, 0.050, -0.020],
            [0.000, -0.020, 0.025],
            [0.020, 0.030, 0.005],
            [-0.030, -0.040, -0.010],
            [0.015, 0.000, 0.030],
            [0.010, 0.060, -0.005],
        ]
    )
    delta_paper, rbar_paper, sigma_paper = _covcor(x)
    res = po.shrink_covariance(x)
    assert abs(res.intensity - delta_paper) < 1e-12
    assert abs(res.rbar - rbar_paper) < 1e-12
    t = x.shape[0]
    # paper's estimator rescaled to the ddof=1 convention of API_SPEC 2.1
    assert np.allclose(res.sigma * (t - 1) / t, sigma_paper, atol=1e-16, rtol=0)
    # unequal vols (theta_ii,ij != theta_jj,ij): the halved rho would differ
    x3 = x * np.array([1.0, 3.0, 0.5])
    d3, _, _ = _covcor(x3)
    assert abs(po.shrink_covariance(x3).intensity - d3) < 1e-12
    # n = 2: the constant-correlation target equals S (gamma_hat = 0), so
    # there is nothing to shrink: delta = 0 and sigma == sample (covCor.m
    # would divide by zero here)
    res2 = po.shrink_covariance(x[:, :2])
    assert res2.intensity == 0.0
    assert np.array_equal(res2.sigma, res2.sample)


def test_lw_bundled_data_matches_paper(returns):
    delta_paper, rbar_paper, sigma_paper = _covcor(returns)
    res = po.shrink_covariance(returns)
    assert abs(res.intensity - delta_paper) < 1e-14
    assert abs(res.intensity - 0.00721176522548) < 1e-12  # paper value, not 0.009430
    assert abs(res.rbar - rbar_paper) < 1e-14
    t = returns.shape[0]
    assert np.allclose(res.sigma * (t - 1) / t, sigma_paper, atol=1e-18, rtol=0)
    assert np.allclose(res.sample, po.sample_cov(returns), atol=0, rtol=0)


def test_lw_intensity_small_t_repairs_singular_sample(returns):
    """PT-2: T = 5 <= N = 8 — the sample covariance is singular but the
    shrunk matrix passes the Cholesky gate and 0 < delta <= 1."""
    x = returns[:5]
    res = po.shrink_covariance(x)
    assert 0.0 < res.intensity <= 1.0
    with pytest.raises(ValueError, match="not positive definite"):
        po.cholesky(res.sample)
    po.cholesky(res.sigma)  # must not raise
    assert abs(res.intensity - _covcor(x)[0]) < 1e-12


def test_shrinkage_zero_variance_rejected():
    """PT-9: a constant column has zero sample variance."""
    x = np.column_stack([np.linspace(-0.01, 0.01, 10), np.full(10, 0.004)])
    with pytest.raises(ValueError, match="strictly positive sample variances"):
        po.shrink_covariance(x)


def test_sample_moments_bit_stable_under_row_order_change(returns):
    """The correctly rounded sums make the mean/covariance independent of
    the summation order (row permutation) to the last bit."""
    perm = np.random.default_rng(9).permutation(returns.shape[0])
    assert np.array_equal(po.sample_mean(returns), po.sample_mean(returns[perm]))
    assert np.array_equal(po.sample_cov(returns), po.sample_cov(returns[perm]))
