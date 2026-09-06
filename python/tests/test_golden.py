"""Golden-value suite: recompute every case from the bundled data and
assert agreement with data/golden/golden.json within its tolerance.

This is the same contract every language port must satisfy.
"""
from __future__ import annotations

import numpy as np
import pytest

import portopt as po

GOLDEN_CASE_COUNT = 23


def _vec(expect: dict, prefix: str) -> np.ndarray:
    keys = sorted(
        (k for k in expect if k.startswith(prefix + "_") and k[len(prefix) + 1 :].isdigit()),
        key=lambda k: int(k.rsplit("_", 1)[1]),
    )
    return np.array([expect[k] for k in keys])


@pytest.fixture(scope="module")
def ctx(returns, market_caps, views_doc, asset_names, config):
    mu_d = po.sample_mean(returns)
    cov_d = po.sample_cov(returns)
    mu = po.annualize_mean(mu_d)
    sig = po.annualize_cov(cov_d)
    return {
        "x": returns,
        "mu_d": mu_d,
        "cov_d": cov_d,
        "mu": mu,
        "sig": sig,
        "caps": market_caps,
        "views": views_doc,
        "names": asset_names,
        "cfg": config,
    }


def check(expect: dict, got: dict, tol: float) -> None:
    for key, want in expect.items():
        assert got[key] == pytest.approx(want, abs=tol), key


def test_golden_case_count(golden):
    assert len(golden) == GOLDEN_CASE_COUNT


def test_every_golden_case_is_asserted(golden):
    """Guard: each case name is referenced by exactly one test below."""
    import inspect
    import sys

    src = inspect.getsource(sys.modules[__name__])
    for name in golden:
        assert f'"{name}"' in src, f"golden case {name} has no test"


def test_sample_moments_daily(golden, ctx):
    c = golden["sample_moments_daily"]
    got = {
        "mean_0": ctx["mu_d"][0],
        "mean_4": ctx["mu_d"][4],
        "mean_7": ctx["mu_d"][7],
        "cov_0_0": ctx["cov_d"][0, 0],
        "cov_0_1": ctx["cov_d"][0, 1],
        "cov_4_4": ctx["cov_d"][4, 4],
        "cov_6_7": ctx["cov_d"][6, 7],
    }
    check(c["expect"], got, c["tol"])


def test_shrinkage_const_corr_daily(golden, ctx):
    c = golden["shrinkage_const_corr_daily"]
    res = po.shrink_covariance(ctx["x"])
    got = {
        "intensity": res.intensity,
        "rbar": res.rbar,
        "shrunk_0_0": res.sigma[0, 0],
        "shrunk_0_1": res.sigma[0, 1],
        "shrunk_2_5": res.sigma[2, 5],
        "shrunk_7_7": res.sigma[7, 7],
    }
    check(c["expect"], got, c["tol"])


def test_two_fund_scalars(golden, ctx):
    c = golden["two_fund_scalars_annualized"]
    sc = po.two_fund_scalars(ctx["mu"], ctx["sig"])
    check(c["expect"], {"a": sc.a, "b": sc.b, "c": sc.c, "d": sc.d}, c["tol"])


def test_minvar_unconstrained(golden, ctx):
    c = golden["minvar_unconstrained"]
    w = po.min_variance_weights(ctx["sig"])
    assert np.allclose(w, _vec(c["expect"], "w"), atol=c["tol"])
    got = {"ret": ctx["mu"] @ w, "vol": float(np.sqrt(w @ ctx["sig"] @ w))}
    check({k: c["expect"][k] for k in ("ret", "vol")}, got, c["tol"])


def test_tangency_unconstrained(golden, ctx):
    c = golden["tangency_unconstrained"]
    rf = c["inputs"]["rf"]
    w = po.tangency_weights(ctx["mu"], ctx["sig"], rf)
    assert np.allclose(w, _vec(c["expect"], "w"), atol=c["tol"])
    sharpe = (ctx["mu"] @ w - rf) / np.sqrt(w @ ctx["sig"] @ w)
    assert sharpe == pytest.approx(c["expect"]["sharpe"], abs=c["tol"])
    assert po.gross_leverage(w) == pytest.approx(c["expect"]["gross_leverage"], abs=c["tol"])


def test_frontier_point(golden, ctx):
    c = golden["frontier_point_t080"]
    w = po.frontier_weights(ctx["mu"], ctx["sig"], c["inputs"]["target"])
    assert np.allclose(w, _vec(c["expect"], "w"), atol=c["tol"])
    assert w @ ctx["sig"] @ w == pytest.approx(c["expect"]["variance"], abs=c["tol"])


def test_minvar_longonly_qp(golden, ctx):
    c = golden["minvar_longonly_qp"]
    res = po.min_variance_longonly_solve(ctx["sig"])
    assert res.converged
    assert np.allclose(res.weights, _vec(c["expect"], "w"), atol=c["tol"])
    w = po.min_variance_longonly(ctx["sig"])
    assert w @ ctx["sig"] @ w == pytest.approx(c["expect"]["variance"], abs=c["tol"])


def test_minvar_box_qp(golden, ctx):
    c = golden["minvar_box_qp"]
    w = po.min_variance_longonly(ctx["sig"], lb=c["inputs"]["lb"], ub=c["inputs"]["ub"])
    assert np.allclose(w, _vec(c["expect"], "w"), atol=c["tol"])
    assert w @ ctx["sig"] @ w == pytest.approx(c["expect"]["variance"], abs=c["tol"])


def test_frontier_constrained(golden, ctx):
    c = golden["frontier_constrained_t080"]
    res = po.constrained_frontier_solve(ctx["mu"], ctx["sig"], c["inputs"]["target"])
    assert res.converged
    assert np.allclose(res.weights, _vec(c["expect"], "w"), atol=c["tol"])
    w = res.weights
    assert w @ ctx["sig"] @ w == pytest.approx(c["expect"]["variance"], abs=c["tol"])
    assert ctx["mu"] @ w == pytest.approx(c["inputs"]["target"], abs=1e-12)


def test_frontier_constrained_box(golden, ctx):
    c = golden["frontier_constrained_box_t060"]
    w = po.constrained_frontier_weights(
        ctx["mu"], ctx["sig"], c["inputs"]["target"], lb=c["inputs"]["lb"], ub=c["inputs"]["ub"]
    )
    assert np.allclose(w, _vec(c["expect"], "w"), atol=c["tol"])
    assert w @ ctx["sig"] @ w == pytest.approx(c["expect"]["variance"], abs=c["tol"])
    assert ctx["mu"] @ w == pytest.approx(c["inputs"]["target"], abs=1e-12)


def test_cholesky_annualized_cov(golden, ctx):
    c = golden["cholesky_annualized_cov"]
    low = po.cholesky(ctx["sig"])
    got = {"l_0_0": low[0, 0], "l_7_0": low[7, 0], "l_7_7": low[7, 7]}
    check(c["expect"], got, c["tol"])


def test_simplex_projection_golden(golden):
    c = golden["simplex_projection_hand"]
    v = np.array([c["inputs"][f"v_{i}"] for i in range(3)])
    w = po.project_simplex_box(v, c["inputs"]["total"])
    assert np.allclose(w, _vec(c["expect"], "w"), atol=c["tol"])

    c = golden["simplex_projection_box"]
    v = np.array([c["inputs"][f"v_{i}"] for i in range(4)])
    w = po.project_simplex_box(
        v, c["inputs"]["total"], lb=c["inputs"]["lb"], ub=c["inputs"]["ub"]
    )
    assert np.allclose(w, _vec(c["expect"], "w"), atol=c["tol"])


def _bl_parts(ctx):
    w_mkt = ctx["caps"] / ctx["caps"].sum()
    doc = ctx["views"]
    pi = po.implied_equilibrium_returns(ctx["sig"], w_mkt, doc["delta"])
    p, q, conf = po.views_to_matrices(doc["views"], ctx["names"])
    return pi, p, q, conf, doc["tau"]


def test_implied_pi(golden, ctx):
    c = golden["implied_pi"]
    pi, *_ = _bl_parts(ctx)
    assert np.allclose(pi, _vec(c["expect"], "pi"), atol=c["tol"])


def test_bl_omega_default(golden, ctx):
    c = golden["bl_omega_default"]
    _, p, _, conf, tau = _bl_parts(ctx)
    om = po.default_omega(ctx["sig"], p, tau, conf)
    check(c["expect"], {"omega_0_0": om[0, 0], "omega_1_1": om[1, 1]}, c["tol"])


def test_bl_posterior_returns_and_weights(golden, ctx):
    pi, p, q, conf, tau = _bl_parts(ctx)
    res = po.bl_posterior(ctx["sig"], pi, p, q, tau, confidences=conf)
    c = golden["bl_posterior_returns"]
    assert np.allclose(res.mu_post, _vec(c["expect"], "mu"), atol=c["tol"])
    c = golden["bl_posterior_weights"]
    w = po.tangency_weights(res.mu_post, res.sigma_post, c["inputs"]["rf"])
    assert np.allclose(w, _vec(c["expect"], "w"), atol=c["tol"])


def test_inverse_vol_golden(golden, ctx):
    c = golden["inverse_vol"]
    assert np.allclose(
        po.inverse_vol_weights(ctx["sig"]), _vec(c["expect"], "w"), atol=c["tol"]
    )


def test_erc_golden(golden, ctx):
    c = golden["erc_weights"]
    w = po.erc_weights(ctx["sig"])
    assert np.allclose(w, _vec(c["expect"], "w"), atol=c["tol"])
    rc = po.risk_contributions(w, ctx["sig"])
    spread = float(np.max(rc) - np.min(rc))
    assert spread == pytest.approx(c["expect"]["rc_spread"], abs=c["tol"])


def test_risk_budget_golden(golden, ctx):
    c = golden["risk_budget_pinned"]
    b = np.array([c["inputs"][f"b_{i}"] for i in range(8)])
    w = po.risk_budget_weights(ctx["sig"], b)
    assert np.allclose(w, _vec(c["expect"], "w"), atol=c["tol"])


def test_backtest_golden(golden, ctx):
    cfg = ctx["cfg"]
    rf, win, step = cfg["risk_free_rate"], cfg["estimation_window"], cfg["rebalance_step"]

    c = golden["backtest_equal_weight"]
    res = po.run_backtest(
        ctx["x"], lambda h: np.full(h.shape[1], 1.0 / h.shape[1]), rf=rf, window=win, step=step
    )
    got = {
        "ann_return": res.ann_return,
        "ann_vol": res.ann_vol,
        "sharpe": res.sharpe,
        "max_drawdown": res.max_drawdown,
    }
    check(c["expect"], got, c["tol"])

    c = golden["backtest_erc"]
    res = po.run_backtest(
        ctx["x"], lambda h: po.erc_weights(po.sample_cov(h)), rf=rf, window=win, step=step
    )
    check(c["expect"], {"sharpe": res.sharpe, "max_drawdown": res.max_drawdown}, c["tol"])

    c = golden["turnover_minvar_longonly"]
    res = po.run_backtest(
        ctx["x"],
        lambda h: po.min_variance_longonly(po.annualize_cov(po.sample_cov(h))),
        rf=rf,
        window=win,
        step=step,
    )
    got = {
        "avg_turnover": res.avg_turnover,
        "n_rebalances": float(len(res.rebalance_dates)),
    }
    check(c["expect"], got, c["tol"])


def test_qp_goldens_are_input_stable(golden, ctx):
    """Scaling the returns by (1 + 2^-52) must move the QP goldens by far
    less than their tolerance — the solver is an exact function of the
    active set, not a truncated iteration (was 1.9e-8 with the old PGD)."""
    x2 = ctx["x"] * (1.0 + 2.0 ** -52)
    sig2 = po.annualize_cov(po.sample_cov(x2))
    c = golden["minvar_longonly_qp"]
    assert np.max(np.abs(po.min_variance_longonly(sig2) - _vec(c["expect"], "w"))) < 1e-13
    c = golden["turnover_minvar_longonly"]
    res = po.run_backtest(
        x2,
        lambda h: po.min_variance_longonly(po.annualize_cov(po.sample_cov(h))),
        rf=0.0,
        window=500,
        step=63,
    )
    assert abs(res.avg_turnover - c["expect"]["avg_turnover"]) < 1e-12
