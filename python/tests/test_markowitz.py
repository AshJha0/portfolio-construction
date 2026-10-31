"""Tests for closed-form Markowitz, simplex projection, and the active-set QP."""

import numpy as np
import pytest
from scipy.optimize import minimize  # cross-check only; package itself is scipy-free

import portopt as po

RF = 0.02


def _slsqp(sig, mu, target, lb, ub):
    n = sig.shape[0]
    lo = np.broadcast_to(np.asarray(lb, dtype=float), (n,))
    hi = np.broadcast_to(np.asarray(ub, dtype=float), (n,))
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    if target is not None:
        cons.append({"type": "eq", "fun": lambda w: mu @ w - target})
    res = minimize(
        lambda w: 0.5 * w @ sig @ w,
        np.clip(np.full(n, 1.0 / n), lo, hi),
        jac=lambda w: sig @ w,
        method="SLSQP",
        bounds=[(lo[i], None if hi[i] == np.inf else hi[i]) for i in range(n)],
        constraints=cons,
        options={"ftol": 1e-15, "maxiter": 1000},
    )
    assert res.success
    return res.x


def _reference_objective(sig, mu, target, lb, ub):
    """Variance 0.5 w'Sw of a feasible reference point from scipy.

    SLSQP is tried first. Its convergence on random problems depends on the
    platform's BLAS (CI saw "Iteration limit reached" on macOS and "Positive
    directional derivative" on Windows for problems it solves on Linux), so a
    failed run falls back to trust-constr. Any point that is feasible to 1e-9
    bounds the optimum from above, which is all the comparison needs; the
    QP's own optimality is checked separately through its KKT certificate.
    """
    from scipy.optimize import Bounds, LinearConstraint

    n = sig.shape[0]
    lo = np.broadcast_to(np.asarray(lb, dtype=float), (n,))
    hi = np.broadcast_to(np.asarray(ub, dtype=float), (n,))

    def feasible(w):
        ok = abs(w.sum() - 1.0) <= 1e-9 and np.all(w >= lo - 1e-9) and np.all(w <= hi + 1e-9)
        return ok and (target is None or abs(mu @ w - target) <= 1e-9)

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    if target is not None:
        cons.append({"type": "eq", "fun": lambda w: mu @ w - target})
    x0 = np.clip(np.full(n, 1.0 / n), lo, hi)
    res = minimize(
        lambda w: 0.5 * w @ sig @ w,
        x0,
        jac=lambda w: sig @ w,
        method="SLSQP",
        bounds=[(lo[i], None if hi[i] == np.inf else hi[i]) for i in range(n)],
        constraints=cons,
        options={"ftol": 1e-15, "maxiter": 1000},
    )
    candidates = [res.x] if res.success and feasible(res.x) else []
    if not candidates:
        rows = [np.ones(n)] + ([mu] if target is not None else [])
        rhs = [1.0] + ([target] if target is not None else [])
        tc = minimize(
            lambda w: 0.5 * w @ sig @ w,
            x0,
            jac=lambda w: sig @ w,
            hess=lambda w: sig,
            method="trust-constr",
            bounds=Bounds(lo, hi),
            constraints=[LinearConstraint(np.array(rows), rhs, rhs)],
            options={"gtol": 1e-14, "xtol": 1e-15, "maxiter": 20000},
        )
        if feasible(tc.x):
            candidates.append(tc.x)
    assert candidates, "no scipy reference point was feasible"
    return min(float(w @ sig @ w) for w in candidates)


def _kkt_residual_longonly(w, grad):
    """Projected-gradient KKT residual for {sum w = 1, w >= 0}: with the
    equality multiplier lam = min_i grad_i over the support, every free
    coordinate must have grad_i = lam and every zero coordinate grad_i >= lam."""
    support = w > 1e-14
    lam = np.min(grad[support])
    res = np.max(np.abs(grad[support] - lam))
    if np.any(~support):
        res = max(res, max(0.0, lam - np.min(grad[~support])))
    return res


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
    mu, sig = moments["mu"], moments["sigma"]
    w_mv = po.min_variance_weights(sig)
    v_mv = w_mv @ sig @ w_mv
    targets, _, variances = po.efficient_frontier(mu, sig, 25)
    assert np.all(variances >= v_mv - 1e-14)
    assert targets[0] == pytest.approx(mu @ w_mv, abs=1e-15)
    assert targets[-1] == np.max(mu)  # pinned: last grid point is exactly hi


def test_frontier_variance_convex_in_target(moments):
    mu, sig = moments["mu"], moments["sigma"]
    targets, weights, var = po.efficient_frontier(mu, sig, 25)
    second_diff = var[2:] - 2 * var[1:-1] + var[:-2]
    assert np.all(second_diff > -1e-14)
    sc = po.two_fund_scalars(mu, sig)
    var_analytic = (sc.c * targets**2 - 2 * sc.a * targets + sc.b) / sc.d
    assert np.allclose(var, var_analytic, atol=1e-12)
    # the sweep reuses one pair of solves: it must equal frontier_weights point-wise
    for k in (0, 7, 24):
        assert np.allclose(weights[k], po.frontier_weights(mu, sig, targets[k]), atol=1e-15)


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
    with pytest.raises(ValueError, match="degenerate frontier"):
        po.efficient_frontier(mu, sig, 5)


def test_frontier_grid_when_gmv_return_exceeds_max_mu():
    """GMV shorts the low-return asset, so A/C = 0.1185 > max(mu) = 0.10:
    the pinned grid would be decreasing; both sweeps must raise a clear
    error while explicit targets above A/C keep working (MAJ-3)."""
    mu = np.array([0.10, 0.05])
    sig = np.array([[0.01, 0.027], [0.027, 0.09]])
    sc = po.two_fund_scalars(mu, sig)
    assert sc.a / sc.c > np.max(mu)
    with pytest.raises(ValueError, match="frontier grid empty"):
        po.efficient_frontier(mu, sig, 25)
    w = po.frontier_weights(mu, sig, 0.13)
    assert mu @ w == pytest.approx(0.13, abs=1e-12)
    # long-only: the constrained GMV is the max-return vertex itself,
    # so the constrained grid is empty too
    with pytest.raises(ValueError, match="frontier grid empty"):
        po.constrained_frontier(mu, sig, 25)


def test_tangency_rf_too_high_rejected(moments):
    mu, sig = moments["mu"], moments["sigma"]
    with pytest.raises(po.TangencyUndefinedError, match="tangency undefined"):
        po.tangency_weights(mu, sig, float(np.max(mu)) + 0.01)
    with pytest.raises(ValueError, match="rf must be finite"):
        po.tangency_weights(mu, sig, np.nan)


def test_tangency_near_gmv_rejected(moments):
    """Gross leverage scales like 1 / (A/C - rf); the guard rejects any
    tangency whose gross leverage exceeds max_leverage (default 10), which
    covers rf within rounding of A/C (leverage 1e6+) and rf merely close to
    it (the 94x case of the round-2 review), while rf = 0.02 on the full
    sample (4.14x) and rf = A/C - 0.1 pass."""
    mu, sig = moments["mu"], moments["sigma"]
    sc = po.two_fund_scalars(mu, sig)
    r_gmv = sc.a / sc.c
    for gap in (1e-13, 1e-9, 1e-3):
        with pytest.raises(po.TangencyUndefinedError, match="ill-conditioned"):
            po.tangency_weights(mu, sig, r_gmv - gap)
    w = po.tangency_weights(mu, sig, 0.02)
    assert 1.0 <= po.gross_leverage(w) <= po.TANGENCY_MAX_LEVERAGE
    # The cap is a parameter: the 1e-3 gap passes when the caller allows it.
    w = po.tangency_weights(mu, sig, r_gmv - 1e-3, max_leverage=1e3)
    assert w.sum() == pytest.approx(1.0, abs=1e-8)
    lev = po.gross_leverage(w)
    assert po.TANGENCY_MAX_LEVERAGE < lev <= 1e3
    assert lev == pytest.approx(np.sum(np.abs(w)), abs=1e-12)
    w_inf = po.tangency_weights(mu, sig, r_gmv - 1e-3, max_leverage=np.inf)
    assert np.array_equal(w, w_inf)
    # Exactly at the cap passes, just below it raises.
    lev_02 = po.gross_leverage(po.tangency_weights(mu, sig, 0.02))
    po.tangency_weights(mu, sig, 0.02, max_leverage=lev_02)
    with pytest.raises(po.TangencyUndefinedError):
        po.tangency_weights(mu, sig, 0.02, max_leverage=lev_02 * (1.0 - 1e-12))
    for bad in (0.5, np.nan):
        with pytest.raises(ValueError, match="max_leverage"):
            po.tangency_weights(mu, sig, 0.02, max_leverage=bad)
    assert po.gross_leverage(np.array([0.2, 0.3, 0.5])) == pytest.approx(1.0, abs=1e-15)
    assert po.gross_leverage(np.array([1.5, -0.5])) == pytest.approx(2.0, abs=1e-15)
    with pytest.raises(ValueError):
        po.gross_leverage(np.array([np.nan]))


def test_tangency_leverage_is_scale_invariant(moments):
    """Leverage is dimensionless: scaling Sigma and the excess returns by
    the same positive factor (e.g. daily vs annual units) leaves the weights
    and therefore the guard decision unchanged."""
    mu, sig = moments["mu"], moments["sigma"]
    w_ann = po.tangency_weights(mu, sig, 0.02)
    w_day = po.tangency_weights(mu / 252.0, sig / 252.0, 0.02 / 252.0)
    assert np.allclose(w_ann, w_day, atol=1e-12)


def test_n1_trivial_weights():
    sig = np.array([[0.04]])
    mu = np.array([0.05])
    assert po.min_variance_weights(sig) == pytest.approx([1.0])
    assert po.tangency_weights(mu, sig, 0.01) == pytest.approx([1.0])
    assert po.frontier_weights(mu, sig, 0.05) == pytest.approx([1.0])
    assert po.min_variance_longonly(sig) == pytest.approx([1.0])
    with pytest.raises(ValueError, match="n=1"):
        po.frontier_weights(mu, sig, 0.06)
    with pytest.raises(po.TangencyUndefinedError):
        po.tangency_weights(mu, sig, 0.10)
    # MIN-11: the n = 1 frontier path validates the covariance too
    with pytest.raises(ValueError, match="positive and finite"):
        po.frontier_weights(mu, np.array([[-1.0]]), 0.05)
    with pytest.raises(ValueError, match="positive and finite"):
        po.tangency_weights(mu, np.array([[0.0]]), 0.01)


def test_singular_covariance_rejected(returns):
    dup = np.column_stack([returns[:, 0], returns[:, 0], returns[:, 1]])
    sig = po.sample_cov(dup)
    with pytest.raises(ValueError, match="not positive definite"):
        po.min_variance_weights(sig)
    with pytest.raises(ValueError, match="not positive definite"):
        po.min_variance_longonly(sig)


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
    for _ in range(20):
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
    with pytest.raises(ValueError, match="lower bounds must be finite"):
        po.project_simplex_box(np.zeros(2), 1.0, lb=-np.inf)
    with pytest.raises(ValueError, match="empty"):
        po.project_simplex_box(np.zeros(0))


# ------------------------------------------------------------------------ QP


def test_qp_matches_closed_form_when_constraints_inactive(moments):
    """With wide box bounds the unconstrained sum-to-one minimizer is
    interior, so the QP must recover the closed form to rounding (the
    old PGD only reached 1e-6)."""
    sig = moments["sigma"]
    w_cf = po.min_variance_weights(sig)
    res = po.min_variance_longonly_solve(sig, lb=-10.0, ub=10.0)
    assert res.converged and res.iterations <= 2  # one add + the final "nothing violated" pass
    assert np.max(np.abs(res.weights - w_cf)) < 1e-14


def test_qp_longonly_matches_scipy_and_kkt(moments):
    sig = moments["sigma"]
    res = po.min_variance_longonly_solve(sig)
    assert res.converged
    assert res.kkt_residual < 1e-12
    w = res.weights
    assert np.max(np.abs(w - _slsqp(sig, None, None, 0.0, np.inf))) < 1e-6
    assert np.all(w >= 0.0)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    assert _kkt_residual_longonly(w, sig @ w) < 1e-13


def test_qp_box_bounds_respected_and_optimal(moments):
    """lb = 0.02, ub = 0.40: bounds hold exactly, the solution beats every
    random feasible point and matches SLSQP (PT-14)."""
    sig = moments["sigma"]
    n = sig.shape[0]
    res = po.min_variance_longonly_solve(sig, lb=0.02, ub=0.40)
    w = res.weights
    assert res.converged and res.kkt_residual < 1e-12
    assert np.all(w >= 0.02) and np.all(w <= 0.40)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    var = w @ sig @ w
    rng = np.random.default_rng(3)
    for _ in range(20):
        z = po.project_simplex_box(rng.standard_normal(n), 1.0, lb=0.02, ub=0.40)
        assert var <= z @ sig @ z + 1e-15
    assert np.max(np.abs(w - _slsqp(sig, None, None, 0.02, 0.40))) < 1e-6


def test_constrained_frontier_reports_convergence_and_hits_target(moments):
    """The frontier point at 0.08 is certified (converged=True, KKT residual
    < 1e-9 * scale) and meets the target exactly (PT-3)."""
    mu, sig = moments["mu"], moments["sigma"]
    res = po.constrained_frontier_solve(mu, sig, 0.08)
    assert res.converged
    assert res.kkt_residual < 1e-9
    w = res.weights
    assert np.all(w >= 0.0)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    assert mu @ w == pytest.approx(0.08, abs=1e-12)
    targets, weights, var = po.constrained_frontier(mu, sig, 25)
    for k in range(25):
        assert mu @ weights[k] == pytest.approx(targets[k], abs=1e-12)
        assert np.all(weights[k] >= 0.0)
        assert var[k] == pytest.approx(weights[k] @ sig @ weights[k], abs=1e-15)


def test_constrained_frontier_matches_reference_qp(moments):
    """scipy SLSQP with the equality mu'w = m and long-only bounds must not
    beat the pinned solution (PT-4)."""
    mu, sig = moments["mu"], moments["sigma"]
    for tgt in (0.04, 0.06, 0.08):
        w = po.constrained_frontier_weights(mu, sig, tgt)
        w_ref = _slsqp(sig, mu, tgt, 0.0, np.inf)
        assert w @ sig @ w <= w_ref @ sig @ w_ref + 1e-10
        assert abs(mu @ w - tgt) <= 1e-12
        assert np.max(np.abs(w - w_ref)) < 1e-5


def test_constrained_frontier_grid_and_monotone_variance(moments):
    """Grid from the constrained GMV return to the max attainable return;
    variance is non-decreasing along it (efficient branch only)."""
    mu, sig = moments["mu"], moments["sigma"]
    targets, weights, var = po.constrained_frontier(mu, sig, 12)
    w_lo = po.min_variance_longonly(sig)
    r_min, r_max = po.attainable_return_range(mu)
    assert targets[0] == pytest.approx(mu @ w_lo, abs=1e-15)
    assert targets[-1] == r_max == np.max(mu)
    assert np.allclose(weights[0], w_lo, atol=1e-15)
    assert np.all(np.diff(var) > -1e-14)
    # the max-return endpoint is the single-asset vertex
    assert np.allclose(weights[-1], np.eye(8)[np.argmax(mu)], atol=1e-15)


def test_constrained_frontier_rejects_unattainable_targets(moments):
    mu, sig = moments["mu"], moments["sigma"]
    w_lo = po.min_variance_longonly(sig)
    with pytest.raises(ValueError, match="below the constrained minimum-variance"):
        po.constrained_frontier_weights(mu, sig, mu @ w_lo - 1e-3)
    with pytest.raises(ValueError, match="not attainable"):
        po.constrained_frontier_weights(mu, sig, np.max(mu) + 1e-3)
    with pytest.raises(ValueError, match="not attainable"):
        po.constrained_frontier_weights(mu, sig, 0.09, ub=0.4)  # 0.4*0.094+... < 0.09
    with pytest.raises(ValueError, match="must be finite"):
        po.constrained_frontier_weights(mu, sig, np.inf)
    with pytest.raises(ValueError, match="max_iter"):
        po.constrained_frontier_weights(mu, sig, 0.06, max_iter=0)
    with pytest.raises(ValueError, match="n_points"):
        po.constrained_frontier(mu, sig, 1)


def test_attainable_return_range(moments):
    mu = moments["mu"]
    r_min, r_max = po.attainable_return_range(mu)
    assert r_min == np.min(mu) and r_max == np.max(mu)
    # caps at 0.4: max return = 0.4*mu(1) + 0.4*mu(2) + 0.2*mu(3) in descending order
    s = np.sort(mu)[::-1]
    r_min_c, r_max_c = po.attainable_return_range(mu, lb=0.0, ub=0.4)
    assert r_max_c == pytest.approx(0.4 * s[0] + 0.4 * s[1] + 0.2 * s[2], abs=1e-15)
    assert r_min_c == pytest.approx(0.4 * s[-1] + 0.4 * s[-2] + 0.2 * s[-3], abs=1e-15)
    with pytest.raises(ValueError, match="infeasible"):
        po.attainable_return_range(mu, lb=0.2)


def test_qp_degenerate_inputs_certified(moments):
    """Fixed assets (lb == ub), tied expected returns and the max-return
    endpoint are primal-degenerate; the solver must still certify."""
    mu, sig = moments["mu"], moments["sigma"]
    n = sig.shape[0]
    lb = np.zeros(n)
    ub = np.full(n, np.inf)
    lb[5] = ub[5] = 0.30  # fixed US_BOND weight
    res = po.min_variance_longonly_solve(sig, lb=lb, ub=ub)
    assert res.converged and res.weights[5] == 0.30
    w_ref = _slsqp(sig, None, None, lb, ub)
    assert res.weights @ sig @ res.weights <= w_ref @ sig @ w_ref + 1e-12
    assert np.max(np.abs(res.weights - w_ref)) < 1e-6
    # tied top returns with caps: endpoint face has two free assets
    mu2 = mu.copy()
    mu2[1] = mu2[0]
    res2 = po.constrained_frontier_solve(mu2, sig, np.max(mu2), lb=0.0, ub=0.7)
    assert res2.converged and res2.kkt_residual < 1e-12
    assert mu2 @ res2.weights == pytest.approx(np.max(mu2) * 0.7 + 0.3 * np.max(mu2), abs=1e-12)
    assert res2.weights[0] + res2.weights[1] == pytest.approx(1.0, abs=1e-12)
    # all assets equal return: the target row is redundant, result is the GMV
    res3 = po.constrained_frontier_solve(np.full(n, 0.05), sig, 0.05)
    assert res3.converged and np.allclose(res3.weights, po.min_variance_longonly(sig), atol=0)


def test_qp_random_problems_vs_scipy():
    rng = np.random.default_rng(2024)
    for _ in range(15):
        n = int(rng.integers(2, 9))
        a = rng.standard_normal((n, n))
        sig = a @ a.T / n + 1e-3 * np.eye(n)
        mu = rng.normal(0.05, 0.04, n)
        lb, ub = 0.0, float(rng.uniform(1.0 / n + 0.05, 1.0))
        res = po.min_variance_longonly_solve(sig, lb=lb, ub=ub)
        assert res.converged
        v_ref = _reference_objective(sig, None, None, lb, ub)
        assert res.weights @ sig @ res.weights <= v_ref + 1e-12
        r_min, r_max = po.attainable_return_range(mu, lb, ub)
        tgt = mu @ res.weights + 0.5 * (r_max - mu @ res.weights)
        fr = po.constrained_frontier_solve(mu, sig, tgt, lb=lb, ub=ub)
        assert fr.converged and abs(mu @ fr.weights - tgt) < 1e-12
        v_ref = _reference_objective(sig, mu, tgt, lb, ub)
        assert fr.weights @ sig @ fr.weights <= v_ref + 1e-12


def test_qp_determinism(moments):
    sig = moments["sigma"]
    w1 = po.min_variance_longonly(sig)
    w2 = po.min_variance_longonly(sig)
    assert np.array_equal(w1, w2)
    mu = moments["mu"]
    a = po.constrained_frontier(mu, sig, 7)
    b = po.constrained_frontier(mu, sig, 7)
    assert np.array_equal(a[1], b[1])


def test_qp_result_is_frozen_and_weights_only_wrappers_agree(moments):
    mu, sig = moments["mu"], moments["sigma"]
    res = po.min_variance_longonly_solve(sig)
    assert np.array_equal(res.weights, po.min_variance_longonly(sig))
    res = po.constrained_frontier_solve(mu, sig, 0.07)
    assert np.array_equal(res.weights, po.constrained_frontier_weights(mu, sig, 0.07))
    with pytest.raises(AttributeError):
        res.converged = False  # type: ignore[misc]


def test_kkt_certificate_checks_dual_feasibility():
    """min 1/2 |w|^2 s.t. sum w = 1, w >= 0 (S = I, n = 2).  The true active
    set {sum} gives w = (1/2, 1/2), u = 1/2 and is certified.  Forcing the
    wrong active set {sum, w_0 >= 0} gives w = (0, 1), which is primal
    feasible and stationary (S w = 1*(1,1) - 1*e_0) but the bound
    multiplier is -1 < 0: the certificate must reject it."""
    from portopt.markowitz import _kkt_certificate

    s = np.eye(2)
    normals = [np.ones(2), np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    rhs = np.array([1.0, 0.0, 0.0])
    ok, resid = _kkt_certificate(s, normals, rhs, 1, [0], [0.5], np.array([0.5, 0.5]))
    assert ok and resid == 0.0
    ok, resid = _kkt_certificate(s, normals, rhs, 1, [0, 1], [1.0, -1.0], np.array([0.0, 1.0]))
    assert not ok and resid == 1.0
    # A rounding-level negative bound multiplier is tolerated (scaled tol).
    ok, resid = _kkt_certificate(s, normals, rhs, 1, [0, 1], [0.5, -1e-12], np.array([0.5, 0.5]))
    assert ok and resid == pytest.approx(1e-12, rel=1e-3)
    # An equality multiplier may be negative.
    ok, _ = _kkt_certificate(
        s, normals[:1], np.array([-1.0]), 1, [0], [-0.5], np.array([-0.5, -0.5])
    )
    assert ok


def test_qp_with_active_bounds_still_certified(moments):
    """Dual feasibility holds at the true optimum: the long-only and boxed
    problems (several bounds active) remain certified, with the same
    residual bound as before."""
    mu, sig = moments["mu"], moments["sigma"]
    for lb, ub in ((0.0, np.inf), (0.02, 0.40), (0.0, 0.25)):
        res = po.min_variance_longonly_solve(sig, lb=lb, ub=ub)
        assert res.converged and res.kkt_residual < 1e-12
        lo = np.broadcast_to(lb, res.weights.shape)
        hi = np.broadcast_to(ub, res.weights.shape)
        assert np.any(res.weights == lo) or np.any(res.weights == hi)
    fr = po.constrained_frontier_solve(mu, sig, 0.07, lb=0.0, ub=0.5)
    assert fr.converged and fr.kkt_residual < 1e-9


# ------------------------------------------------------------ cost-aware QP


def _turnover_objective(sig, c, w0, kappa, w):
    return 0.5 * w @ sig @ w - c @ w + kappa * np.abs(w - w0).sum()


def _reference_turnover_objective(sig, c, w0, kappa, lb, ub, cap):
    """Objective of a feasible scipy point for the split problem
    (w, b, s), w - b + s = w0, b, s >= 0.  SLSQP first, trust-constr as
    the fallback (same reasoning as _reference_objective)."""
    from scipy.optimize import Bounds, LinearConstraint

    n = sig.shape[0]
    lo = np.broadcast_to(np.asarray(lb, dtype=float), (n,))
    hi = np.broadcast_to(np.asarray(ub, dtype=float), (n,))

    def split_obj(z):
        return 0.5 * z[:n] @ sig @ z[:n] - c @ z[:n] + kappa * z[n:].sum()

    def split_jac(z):
        return np.concatenate([sig @ z[:n] - c, np.full(2 * n, kappa)])

    def feasible(w):
        ok = abs(w.sum() - 1.0) <= 1e-9 and np.all(w >= lo - 1e-9) and np.all(w <= hi + 1e-9)
        return ok and (cap is None or np.abs(w - w0).sum() <= cap + 1e-9)

    x0 = np.clip(w0, lo, hi)
    x0 = po.project_simplex_box(x0, 1.0, lo, hi)
    d = x0 - w0
    z0 = np.concatenate([x0, np.maximum(d, 0.0), np.maximum(-d, 0.0)])
    cons = [
        {"type": "eq", "fun": lambda z: z[:n].sum() - 1.0},
        {"type": "eq", "fun": lambda z: z[:n] - z[n : 2 * n] + z[2 * n :] - w0},
    ]
    if cap is not None:
        cons.append({"type": "ineq", "fun": lambda z: cap - z[n:].sum()})
    bounds = [(lo[i], None if hi[i] == np.inf else hi[i]) for i in range(n)]
    bounds += [(0.0, None)] * (2 * n)
    res = minimize(
        split_obj,
        z0,
        jac=split_jac,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"ftol": 1e-15, "maxiter": 2000},
    )
    candidates = [res.x[:n]] if res.success and feasible(res.x[:n]) else []
    if not candidates:
        rows = np.zeros((n + 1, 3 * n))
        rows[0, :n] = 1.0
        for i in range(n):
            rows[1 + i, i], rows[1 + i, n + i], rows[1 + i, 2 * n + i] = 1.0, -1.0, 1.0
        rhs = np.concatenate([[1.0], w0])
        lin = [LinearConstraint(rows, rhs, rhs)]
        if cap is not None:
            row = np.concatenate([np.zeros(n), np.ones(2 * n)])
            lin.append(LinearConstraint(row[None, :], -np.inf, cap))
        hess = np.zeros((3 * n, 3 * n))
        hess[:n, :n] = sig
        tc = minimize(
            split_obj,
            z0,
            jac=split_jac,
            hess=lambda z: hess,
            method="trust-constr",
            bounds=Bounds(
                np.concatenate([lo, np.zeros(2 * n)]),
                np.concatenate([hi, np.full(2 * n, np.inf)]),
            ),
            constraints=lin,
            options={"gtol": 1e-14, "xtol": 1e-15, "maxiter": 20000},
        )
        if feasible(tc.x[:n]):
            candidates.append(tc.x[:n])
    assert candidates, "no scipy reference point was feasible"
    return min(_turnover_objective(sig, c, w0, kappa, w) for w in candidates)


def test_turnover_kappa_zero_matches_longonly_qp(moments):
    sig = moments["sigma"]
    n = sig.shape[0]
    for w0 in (np.full(n, 1.0 / n), np.eye(n)[0], np.linspace(-0.2, 0.5, n)):
        for lb, ub in ((0.0, np.inf), (0.02, 0.40), (0.0, 0.25)):
            res = po.min_variance_turnover_solve(sig, w0, 0.0, lb=lb, ub=ub)
            assert res.converged and res.kkt_residual < 1e-12
            ref = po.min_variance_longonly(sig, lb=lb, ub=ub)
            assert np.max(np.abs(res.weights - ref)) < 1e-12
            assert res.turnover == pytest.approx(np.abs(ref - w0).sum(), abs=1e-12)


def test_turnover_large_kappa_returns_feasible_w0(moments):
    sig = moments["sigma"]
    n = sig.shape[0]
    caps = np.array([30.0, 20.0, 8.0, 6.0, 7.0, 25.0, 15.0, 10.0])
    for w0, lb, ub in (
        (np.full(n, 1.0 / n), 0.0, np.inf),
        (caps / caps.sum(), 0.02, 0.40),
        (np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0]), 0.0, np.inf),
    ):
        res = po.min_variance_turnover_solve(sig, w0, 1.0, lb=lb, ub=ub)
        assert res.converged
        assert np.max(np.abs(res.weights - w0)) < 1e-12
        assert res.turnover < 1e-12
        mv = po.mean_variance_turnover_solve(moments["mu"], sig, w0, 2.5, 10.0, lb=lb, ub=ub)
        assert mv.converged and np.max(np.abs(mv.weights - w0)) < 1e-12


def test_turnover_decreases_monotonically_in_kappa(moments):
    """(k2 - k1)(TO(k2) - TO(k1)) <= 0 follows from comparing the two
    optimality conditions; the objective net of costs rises with kappa."""
    mu, sig = moments["mu"], moments["sigma"]
    n = sig.shape[0]
    caps = np.array([30.0, 20.0, 8.0, 6.0, 7.0, 25.0, 15.0, 10.0])
    kappas = [0.0, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 1e-1]
    for solve in (
        lambda k: po.min_variance_turnover_solve(sig, np.full(n, 1.0 / n), k),
        lambda k: po.mean_variance_turnover_solve(
            mu, sig, caps / caps.sum(), 2.5, k, lb=0.02, ub=0.40
        ),
    ):
        tos = []
        for k in kappas:
            res = solve(k)
            assert res.converged
            tos.append(res.turnover)
        assert all(b <= a + 1e-12 for a, b in zip(tos, tos[1:], strict=False)), tos
        assert tos[0] > 0.3 and tos[-1] < 1e-12


def test_turnover_cap_binds_and_is_certified(moments):
    sig = moments["sigma"]
    n = sig.shape[0]
    w0 = np.full(n, 1.0 / n)
    free = po.min_variance_turnover_solve(sig, w0, 1e-4)
    for cap in (0.8, 0.5, 0.2, 0.05, 0.0):
        res = po.min_variance_turnover_solve(sig, w0, 1e-4, max_turnover=cap)
        assert res.converged and res.kkt_residual < 1e-12
        assert res.turnover <= cap + 1e-12
        assert res.turnover == pytest.approx(min(cap, free.turnover), abs=1e-12)
    # A slack cap changes nothing.
    loose = po.min_variance_turnover_solve(sig, w0, 1e-4, max_turnover=5.0)
    assert np.max(np.abs(loose.weights - free.weights)) < 1e-12
    # Cap below the smallest attainable turnover: w0 outside the box.
    w_bad = np.array([0.6, 0.4, 0, 0, 0, 0, 0, 0.0])
    with pytest.raises(ValueError, match="infeasible turnover cap"):
        po.min_variance_turnover_solve(sig, w_bad, 0.0, ub=0.3, max_turnover=0.7)
    # Smallest attainable: |0.6-0.3| + |0.4-0.3| + budget gap 0.4 = 0.8.
    res = po.min_variance_turnover_solve(sig, w_bad, 0.0, ub=0.3, max_turnover=0.8)
    assert res.converged and res.turnover == pytest.approx(0.8, abs=1e-12)


def test_turnover_matches_scipy_on_random_problems():
    rng = np.random.default_rng(31)
    for trial in range(15):
        n = int(rng.integers(2, 9))
        a = rng.standard_normal((n, n))
        sig = a @ a.T / n + 1e-3 * np.eye(n)
        mu = rng.normal(0.05, 0.04, n)
        w0 = rng.dirichlet(np.ones(n))
        kappa = float(10.0 ** rng.uniform(-4, -1))
        ub = float(rng.uniform(1.0 / n + 0.05, 1.0))
        wc = np.clip(w0, 0.0, ub)
        to_min = np.abs(wc - w0).sum() + abs(1.0 - wc.sum())
        cap = to_min + float(rng.uniform(0.02, 0.4)) if trial % 3 == 0 else None
        for c, res in (
            (np.zeros(n), po.min_variance_turnover_solve(sig, w0, kappa, ub=ub, max_turnover=cap)),
            (
                mu / 2.0,
                po.mean_variance_turnover_solve(mu, sig, w0, 2.0, kappa, ub=ub, max_turnover=cap),
            ),
        ):
            assert res.converged, (trial, res.kkt_residual)
            f_qp = _turnover_objective(sig, c, w0, kappa, res.weights)
            f_ref = _reference_turnover_objective(sig, c, w0, kappa, 0.0, ub, cap)
            assert f_qp <= f_ref + 1e-12


def test_turnover_certificate_rejects_wrong_points(moments):
    """The certificate is the KKT system of the nonsmooth problem: the
    optimum passes, a perturbed point fails stationarity, a wrong budget
    multiplier fails, and a negative cap multiplier fails."""
    from portopt.markowitz import _turnover_certificate

    sig = moments["sigma"]
    n = sig.shape[0]
    w0 = np.full(n, 1.0 / n)
    kappa = 1e-3
    res = po.min_variance_turnover_solve(sig, w0, kappa)
    lo, hi, c = np.zeros(n), np.full(n, np.inf), np.zeros(n)
    w = res.weights
    # Recover nu from a free (not at bound, not at w0) coordinate.
    free = [i for i in range(n) if w[i] > 1e-9 and abs(w[i] - w0[i]) > 1e-9]
    i = free[0]
    nu = float((sig @ w)[i] + kappa * np.sign(w[i] - w0[i]))
    ok, resid = _turnover_certificate(sig, c, w0, kappa, lo, hi, np.inf, nu, 0.0, w)
    assert ok and resid < 1e-12
    ok, _ = _turnover_certificate(sig, c, w0, kappa, lo, hi, np.inf, nu * 1.01, 0.0, w)
    assert not ok
    w_pert = w.copy()
    w_pert[free[0]] += 1e-4
    w_pert[free[1]] -= 1e-4
    ok, _ = _turnover_certificate(sig, c, w0, kappa, lo, hi, np.inf, nu, 0.0, w_pert)
    assert not ok
    ok, _ = _turnover_certificate(sig, c, w0, kappa, lo, hi, 10.0, nu, -1e-3, w)
    assert not ok
    # tau > 0 with a slack cap breaks complementarity.
    ok, _ = _turnover_certificate(sig, c, w0, kappa, lo, hi, 10.0, nu, 1e-3, w)
    assert not ok
    # The wrapper agrees with the solve variant.
    assert np.array_equal(po.min_variance_turnover(sig, w0, kappa), w)


def test_turnover_validation_and_trivial_cases(moments):
    mu, sig = moments["mu"], moments["sigma"]
    n = sig.shape[0]
    w0 = np.full(n, 1.0 / n)
    with pytest.raises(ValueError, match="kappa"):
        po.min_variance_turnover_solve(sig, w0, -1e-3)
    with pytest.raises(ValueError, match="kappa"):
        po.min_variance_turnover_solve(sig, w0, np.nan)
    with pytest.raises(ValueError, match="w0 length"):
        po.min_variance_turnover_solve(sig, w0[:3], 0.0)
    with pytest.raises(ValueError, match="w0 contains"):
        po.min_variance_turnover_solve(sig, np.full(n, np.nan), 0.0)
    with pytest.raises(ValueError, match="max_turnover"):
        po.min_variance_turnover_solve(sig, w0, 0.0, max_turnover=-0.1)
    with pytest.raises(ValueError, match="risk_aversion"):
        po.mean_variance_turnover_solve(mu, sig, w0, 0.0, 1e-3)
    with pytest.raises(ValueError, match="infeasible bounds"):
        po.min_variance_turnover_solve(sig, w0, 0.0, ub=0.1)
    with pytest.raises(ValueError, match="not positive definite"):
        dup = sig.copy()
        dup[1, :], dup[:, 1] = dup[0, :], dup[:, 0]
        po.min_variance_turnover_solve(dup, w0, 0.0)
    one = po.min_variance_turnover_solve(np.array([[0.04]]), np.array([0.3]), 1e-3)
    assert one.converged and one.weights.tolist() == [1.0] and one.turnover == pytest.approx(0.7)
    # Deterministic and frozen.
    a = po.mean_variance_turnover_solve(mu, sig, w0, 2.5, 1e-3, lb=0.02, ub=0.4)
    b = po.mean_variance_turnover_solve(mu, sig, w0, 2.5, 1e-3, lb=0.02, ub=0.4)
    assert np.array_equal(a.weights, b.weights) and a.iterations == b.iterations
    with pytest.raises(AttributeError):
        a.converged = False  # type: ignore[misc]
    assert np.array_equal(
        po.mean_variance_turnover(mu, sig, w0, 2.5, 1e-3, lb=0.02, ub=0.4), a.weights
    )


def test_turnover_wrapper_raises_when_not_certified(moments):
    sig = moments["sigma"]
    n = sig.shape[0]
    res = po.min_variance_turnover_solve(sig, np.full(n, 1.0 / n), 1e-3, max_iter=2)
    assert not res.converged
    with pytest.raises(ValueError, match="not certified"):
        po.min_variance_turnover(sig, np.full(n, 1.0 / n), 1e-3, max_iter=2)


def test_turnover_meanvar_kappa_zero_is_box_mean_variance(moments):
    """kappa = 0: the optimum is the box-constrained mean-variance
    portfolio, independent of w0; check it against scipy."""
    mu, sig = moments["mu"], moments["sigma"]
    n = sig.shape[0]
    a = po.mean_variance_turnover_solve(mu, sig, np.full(n, 1.0 / n), 2.5, 0.0, lb=0.02, ub=0.4)
    b = po.mean_variance_turnover_solve(mu, sig, np.eye(n)[3], 2.5, 0.0, lb=0.02, ub=0.4)
    assert a.converged and b.converged
    assert np.max(np.abs(a.weights - b.weights)) < 1e-12
    f_ref = _reference_turnover_objective(sig, mu / 2.5, np.full(n, 1.0 / n), 0.0, 0.02, 0.4, None)
    assert _turnover_objective(sig, mu / 2.5, np.full(n, 1.0 / n), 0.0, a.weights) <= f_ref + 1e-12
