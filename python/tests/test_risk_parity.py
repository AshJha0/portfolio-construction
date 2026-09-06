"""Tests for risk contributions, ERC CCD, and risk budgeting."""
import numpy as np
import pytest
from scipy.optimize import minimize  # cross-check only

import portopt as po


def test_rc_identity_sums_to_one(moments):
    """Sum of normalized risk contributions == 1 (portfolio variance
    normalized), identity to 1e-12, for a grid of portfolios."""
    sig = moments["sigma"]
    n = sig.shape[0]
    rng = np.random.default_rng(11)
    for _ in range(10):
        w = rng.dirichlet(np.ones(n))
        rc = po.risk_contributions(w, sig)
        assert abs(rc.sum() - 1.0) < 1e-12


def test_erc_contributions_equal(moments):
    sig = moments["sigma"]
    w, sweeps, conv = po.risk_budget_solve(sig)
    assert conv and sweeps < 10_000
    rc = po.risk_contributions(w, sig)
    n = sig.shape[0]
    assert np.max(np.abs(rc - 1.0 / n)) < 1e-9
    assert np.max(rc) - np.min(rc) < 1e-9
    assert np.all(w > 0)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)


def test_erc_matches_scipy(moments):
    """Cross-check the CCD solution against a scipy SLSQP least-squares
    formulation of the ERC condition."""
    sig = moments["sigma"]
    n = sig.shape[0]
    w_ccd = po.erc_weights(sig)

    def objective(w):
        sw = sig @ w
        rc = w * sw
        return np.sum((rc[:, None] - rc[None, :]) ** 2) * 1e6

    res = minimize(
        objective,
        np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(1e-8, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"ftol": 1e-16, "maxiter": 1000},
    )
    assert res.success
    assert np.max(np.abs(w_ccd - res.x)) < 1e-5


def test_risk_budget_targets_met(moments):
    sig = moments["sigma"]
    b = np.array([0.20, 0.20, 0.15, 0.15, 0.10, 0.10, 0.05, 0.05])
    w = po.risk_budget_weights(sig, b)
    rc = po.risk_contributions(w, sig)
    assert np.max(np.abs(rc - b)) < 1e-9
    # ERC is the equal-budget special case
    assert np.allclose(po.erc_weights(sig), po.risk_budget_weights(sig), atol=0)


def test_budget_normalization(moments):
    """Budgets are normalized internally: scaling them changes nothing."""
    sig = moments["sigma"]
    b = np.array([2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 0.5, 0.5])
    w1 = po.risk_budget_weights(sig, b)
    w2 = po.risk_budget_weights(sig, b / b.sum())
    assert np.allclose(w1, w2, atol=1e-14)


def test_inverse_vol_weights(moments):
    sig = moments["sigma"]
    w = po.inverse_vol_weights(sig)
    inv = 1.0 / np.sqrt(np.diag(sig))
    assert np.allclose(w, inv / inv.sum(), atol=1e-15)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)


def test_erc_diagonal_case_equals_inverse_vol():
    """For diagonal covariance, ERC == inverse vol analytically."""
    sig = np.diag([0.04, 0.09, 0.0225, 0.01])
    assert np.allclose(
        po.erc_weights(sig), po.inverse_vol_weights(sig), atol=1e-10
    )


def test_erc_two_asset_analytic():
    """n=2 ERC has the closed form w1 = s2/(s1+s2) regardless of correlation."""
    for rho in (-0.5, 0.0, 0.7):
        s1, s2 = 0.2, 0.1
        sig = np.array([[s1**2, rho * s1 * s2], [rho * s1 * s2, s2**2]])
        w = po.erc_weights(sig)
        assert w[0] == pytest.approx(s2 / (s1 + s2), abs=1e-10)


def test_erc_with_almost_zero_vol_asset():
    """A near-zero-vol asset dominates the ERC portfolio but the CCD
    still converges and budgets are met."""
    sig = np.diag([0.04, 0.0225, 1e-8])
    sig[0, 1] = sig[1, 0] = 0.3 * 0.2 * 0.15
    w, sweeps, conv = po.risk_budget_solve(sig)
    assert conv
    rc = po.risk_contributions(w, sig)
    assert np.max(np.abs(rc - 1.0 / 3.0)) < 1e-8
    assert w[2] > 0.9  # tiny-vol asset takes nearly all the weight


def test_risk_parity_error_paths(moments):
    sig = moments["sigma"]
    n = sig.shape[0]
    with pytest.raises(ValueError, match="positive"):
        po.risk_budget_weights(sig, np.zeros(n))
    with pytest.raises(ValueError, match="length"):
        po.risk_budget_weights(sig, np.ones(3))
    with pytest.raises(ValueError, match="not positive definite"):
        po.erc_weights(np.ones((3, 3)))  # rank-1, singular
    with pytest.raises(ValueError, match="strictly positive"):
        po.risk_contributions(np.zeros(n), sig)
    bad = sig.copy()
    with pytest.raises(ValueError, match="length"):
        po.risk_contributions(np.ones(n + 1), bad)


def test_ccd_determinism(moments):
    sig = moments["sigma"]
    assert np.array_equal(po.erc_weights(sig), po.erc_weights(sig))


def test_erc_permutation_invariance(moments):
    """PT-12: erc_weights(P Sigma P') == P erc_weights(Sigma) — a sweep-order
    bug would break this."""
    sig = moments["sigma"]
    perm = np.array([5, 2, 7, 0, 3, 6, 1, 4])
    p = np.eye(8)[perm]
    w = po.erc_weights(sig)
    w_perm = po.erc_weights(p @ sig @ p.T)
    assert np.max(np.abs(w_perm - p @ w)) < 1e-10
    b = np.array([0.20, 0.20, 0.15, 0.15, 0.10, 0.10, 0.05, 0.05])
    w_rb = po.risk_budget_weights(sig, b)
    w_rb_perm = po.risk_budget_weights(p @ sig @ p.T, p @ b)
    assert np.max(np.abs(w_rb_perm - p @ w_rb)) < 1e-10


def test_erc_scale_invariance(returns):
    """PT-13: ERC weights are invariant to the covariance scale (daily vs
    annualized), and both solves converge (API_SPEC 6.3 'scale-invariant')."""
    sig_d = po.sample_cov(returns)
    w_d, s_d, c_d = po.risk_budget_solve(sig_d)
    w_a, s_a, c_a = po.risk_budget_solve(252.0 * sig_d)
    assert c_d and c_a
    assert np.max(np.abs(w_d - w_a)) < 1e-10
    w_k, _, c_k = po.risk_budget_solve(1e-6 * sig_d)
    assert c_k and np.max(np.abs(w_k - w_d)) < 1e-10


def test_ccd_cap_reported_not_raised(moments):
    sig = moments["sigma"]
    w, sweeps, conv = po.risk_budget_solve(sig, max_sweeps=1)
    assert not conv and sweeps == 1
    assert w.sum() == pytest.approx(1.0, abs=1e-12) and np.all(w > 0)
    with pytest.raises(ValueError, match="tol"):
        po.risk_budget_solve(sig, tol=0.0)
    with pytest.raises(ValueError, match="max_sweeps"):
        po.risk_budget_solve(sig, max_sweeps=0)
