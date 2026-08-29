"""Tests for closed-form Markowitz, simplex projection, and PGD."""
import numpy as np
import pytest
from scipy.optimize import minimize  # cross-check only; package itself is scipy-free

import portopt as po

RF = 0.02


def test_two_fund_identities(moments):
    mu, sig = moments["mu"], moments["sigma"]
    sc = po.two_fund_scalars(mu, sig)
    inv = np.linalg.inv(sig)
    ones = np.ones(len(mu))
    assert sc.a == pytest.approx(ones @ inv @ mu, abs=1e-10)
    assert sc.b == pytest.approx(mu @ inv @ mu, abs=1e-10)
    assert sc.c == pytest.approx(ones @ inv @ ones, abs=1e-8)
    assert sc.d == pytest.approx(sc.b * sc.c - sc.a**2, abs=1e-10)
    assert sc.c > 0 and sc.b > 0 and sc.d > 0


def test_minvar_closed_form_vs_lstsq(moments):
    sig = moments["sigma"]
    w = po.min_variance_weights(sig)
    y = np.linalg.lstsq(sig, np.ones(sig.shape[0]), rcond=None)[0]
    assert np.allclose(w, y / y.sum(), atol=1e-11)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)


def test_minvar_is_frontier_minimum(moments):
    """Property check: min-var portfolio has the smallest variance on a
    grid of frontier portfolios."""
    mu, sig = moments["mu"], moments["sigma"]
    w_mv = po.min_variance_weights(sig)
    v_mv = w_mv @ sig @ w_mv
    targets, _, variances = po.efficient_frontier(mu, sig, 25)
    assert np.all(variances >= v_mv - 1e-14)


def test_frontier_variance_convex_in_target(moments):
    """Grid check: sigma^2(m) = (C m^2 - 2 A m + B)/D, convex in m."""
    mu, sig = moments["mu"], moments["sigma"]
    targets, _, var = po.efficient_frontier(mu, sig, 25)
    second_diff = var[2:] - 2 * var[1:-1] + var[:-2]
    assert np.all(second_diff > -1e-14)
    sc = po.two_fund_scalars(mu, sig)
    var_analytic = (sc.c * targets**2 - 2 * sc.a * targets + sc.b) / sc.d
    assert np.allclose(var, var_analytic, atol=1e-12)


def test_tangency_has_max_sharpe_on_frontier_grid(moments):
    mu, sig = moments["mu"], moments["sigma"]
    w_tan = po.tangency_weights(mu, sig, RF)
    sharpe_tan = (mu @ w_tan - RF) / np.sqrt(w_tan @ sig @ w_tan)
    targets, weights, var = po.efficient_frontier(mu, sig, 25)
    for k in range(len(targets)):
        sharpe_k = (mu @ weights[k] - RF) / np.sqrt(var[k])
        assert sharpe_k <= sharpe_tan + 1e-12


def test_frontier_weights_hit_target(moments):
    mu, sig = moments["mu"], moments["sigma"]
    w = po.frontier_weights(mu, sig, 0.08)
    assert mu @ w == pytest.approx(0.08, abs=1e-12)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)


def test_frontier_target_below_minvar_rejected(moments):
    mu, sig = moments["mu"], moments["sigma"]
    sc = po.two_fund_scalars(mu, sig)
    with pytest.raises(ValueError, match="below the minimum-variance return"):
        po.frontier_weights(mu, sig, sc.a / sc.c - 0.01)
    with pytest.raises(ValueError, match="below the minimum-variance return"):
        po.frontier_weights(mu, sig, -0.05)


def test_degenerate_frontier_rejected():
    sig = np.diag([0.04, 0.09, 0.01])
    mu = np.array([0.05, 0.05, 0.05])  # proportional to ones -> D = 0
    with pytest.raises(ValueError, match="degenerate frontier"):
        po.frontier_weights(mu, sig, 0.05)


def test_tangency_rf_too_high_rejected(moments):
    mu, sig = moments["mu"], moments["sigma"]
    with pytest.raises(ValueError, match="tangency undefined"):
        po.tangency_weights(mu, sig, float(np.max(mu)) + 0.01)


def test_n1_trivial_weights():
    sig = np.array([[0.04]])
    mu = np.array([0.05])
    assert po.min_variance_weights(sig) == pytest.approx([1.0])
    assert po.tangency_weights(mu, sig, 0.01) == pytest.approx([1.0])
    assert po.frontier_weights(mu, sig, 0.05) == pytest.approx([1.0])
    with pytest.raises(ValueError, match="n=1"):
        po.frontier_weights(mu, sig, 0.06)
    with pytest.raises(ValueError, match="tangency undefined"):
        po.tangency_weights(mu, sig, 0.10)


def test_singular_covariance_rejected(returns):
    dup = np.column_stack([returns[:, 0], returns[:, 0], returns[:, 1]])
    sig = po.sample_cov(dup)
    with pytest.raises(ValueError, match="not positive definite"):
        po.min_variance_weights(sig)


# ---------------------------------------------------------------- projection

def test_simplex_projection_hand_cases():
    # Classic Duchi example.
    w = po.project_simplex_box(np.array([0.9, 0.5, -0.2]))
    assert np.allclose(w, [0.7, 0.3, 0.0], atol=1e-14)
    # Already on the simplex: fixed point.
    v = np.array([0.2, 0.3, 0.5])
    assert np.allclose(po.project_simplex_box(v), v, atol=1e-14)
    # Uniform shift only.
    w = po.project_simplex_box(np.array([1.0, 1.0]))
    assert np.allclose(w, [0.5, 0.5], atol=1e-14)
    # Single coordinate.
    assert np.allclose(po.project_simplex_box(np.array([5.0])), [1.0], atol=1e-14)


def test_simplex_projection_with_box_bounds():
    w = po.project_simplex_box(np.array([0.8, 0.6, 0.4, -0.3]), 1.0, lb=0.05, ub=0.5)
    assert np.allclose(w, [0.5, 0.325, 0.125, 0.05], atol=1e-14)
    assert w.sum() == pytest.approx(1.0, abs=1e-14)


def test_simplex_projection_is_euclidean_optimal():
    """Property grid: our projection beats random feasible points in
    Euclidean distance."""
    rng = np.random.default_rng(5)
    for k in range(20):
        v = rng.standard_normal(6)
        w = po.project_simplex_box(v, 1.0, lb=0.0, ub=0.6)
        assert abs(w.sum() - 1.0) < 1e-12
        assert np.all(w >= -1e-14) and np.all(w <= 0.6 + 1e-14)
        for _ in range(10):
            z = rng.dirichlet(np.ones(6)) * 0.9 + 0.1 / 6  # feasible-ish
            z = np.clip(z, 0.0, 0.6)
            z = z / z.sum()
            if np.all(z <= 0.6):
                assert np.sum((w - v) ** 2) <= np.sum((z - v) ** 2) + 1e-10


def test_simplex_projection_infeasible_bounds():
    with pytest.raises(ValueError, match="sum\\(lb\\)"):
        po.project_simplex_box(np.zeros(4), 1.0, lb=0.3)  # 4*0.3 > 1
    with pytest.raises(ValueError, match="sum\\(ub\\)"):
        po.project_simplex_box(np.zeros(4), 1.0, ub=0.2)  # 4*0.2 < 1
    with pytest.raises(ValueError, match="lb_i > ub_i"):
        po.project_simplex_box(np.zeros(2), 1.0, lb=[0.6, 0.0], ub=[0.5, 1.0])
    with pytest.raises(ValueError, match="NaN"):
        po.project_simplex_box(np.array([np.nan, 0.0]))


# ----------------------------------------------------------------------- PGD

def test_pgd_matches_closed_form_when_constraints_inactive(moments):
    """With wide box bounds the unconstrained sum-to-one minimizer is
    interior, so PGD must recover the closed form to 1e-6."""
    sig = moments["sigma"]
    w_cf = po.min_variance_weights(sig)
    w_pgd = po.min_variance_longonly(sig, lb=-10.0, ub=10.0)
    assert np.max(np.abs(w_pgd - w_cf)) < 1e-6


def test_pgd_longonly_matches_scipy(moments):
    sig = moments["sigma"]
    n = sig.shape[0]
    w_pgd = po.min_variance_longonly(sig)
    res = minimize(
        lambda w: 0.5 * w @ sig @ w,
        np.full(n, 1.0 / n),
        jac=lambda w: sig @ w,
        method="SLSQP",
        bounds=[(0.0, None)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"ftol": 1e-14, "maxiter": 500},
    )
    assert res.success
    assert np.max(np.abs(w_pgd - res.x)) < 1e-6
    assert np.all(w_pgd >= -1e-14)
    assert w_pgd.sum() == pytest.approx(1.0, abs=1e-12)


def test_constrained_frontier_penalty_target(moments):
    mu, sig = moments["mu"], moments["sigma"]
    w = po.constrained_frontier_weights(mu, sig, 0.08)
    assert np.all(w >= -1e-14)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    # penalty formulation: achieved return within O(1/penalty) of target
    assert abs(mu @ w - 0.08) < 1e-3


def test_constrained_frontier_variance_increasing_tail(moments):
    """Above the long-only min-variance return, constrained frontier
    variance is non-decreasing in the target (grid property check)."""
    mu, sig = moments["mu"], moments["sigma"]
    targets, weights, var = po.constrained_frontier(mu, sig, 12)
    w_lo = po.min_variance_longonly(sig)
    r_lo = mu @ w_lo
    mask = targets >= r_lo
    v = var[mask]
    assert np.all(np.diff(v) > -1e-10)


def test_pgd_determinism(moments):
    sig = moments["sigma"]
    w1 = po.min_variance_longonly(sig)
    w2 = po.min_variance_longonly(sig)
    assert np.array_equal(w1, w2)
