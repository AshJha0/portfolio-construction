"""Tests for portopt.research.strategies_v2 weight functions."""

from __future__ import annotations

import math

import numpy as np
import pytest

import portopt as po
from portopt import research as rs
from portopt.research import estimators as est
from portopt.research import strategies_v2 as v2


@pytest.fixture(scope="module")
def history(returns) -> np.ndarray:
    return returns[:500]


def all_estimators() -> list[v2.CovarianceEstimator]:
    return [
        v2.sample_estimator(),
        v2.lw_constant_corr_estimator(),
        v2.lw_identity_estimator(),
        v2.nonlinear_estimator(),
        v2.ewma_estimator(60.0),
        v2.ewma_estimator(math.inf, demean=False),
        v2.ewma_cc_estimator(60.0),
        v2.pca_estimator(3),
    ]


def test_estimators_annualize_the_research_functions(history):
    pairs = [
        (v2.sample_estimator(), po.sample_cov(history)),
        (v2.lw_constant_corr_estimator(), po.shrink_covariance(history).sigma),
        (v2.lw_identity_estimator(), est.ledoit_wolf_identity(history).sigma),
        (v2.nonlinear_estimator(), est.nonlinear_shrinkage(history).sigma),
        (v2.ewma_estimator(60.0), est.ewma_covariance(history, 60.0)),
        (v2.ewma_cc_estimator(60.0), est.ewma_constant_correlation(history, 60.0)),
        (v2.pca_estimator(3), est.pca_factor_covariance(history, 3).sigma),
    ]
    for e, daily in pairs:
        np.testing.assert_array_equal(e(history), 252.0 * daily)
    names = [e.name for e in all_estimators()]
    assert names == ["sample", "lwcc", "lwid", "nls", "ewma60", "ewmainf", "ewmacc60", "pca3"]
    assert repr(v2.nonlinear_estimator()) == "CovarianceEstimator('nls')"


def test_estimator_validation():
    with pytest.raises(ValueError, match="half_life"):
        v2.ewma_estimator(0.0)
    with pytest.raises(ValueError, match="half_life"):
        v2.ewma_cc_estimator(math.nan)
    for k in (0, 2.5, True):
        with pytest.raises(ValueError, match="k must be"):
            v2.pca_estimator(k)
    with pytest.raises(ValueError, match="specific_floor"):
        v2.pca_estimator(2, specific_floor=1.0)
    with pytest.raises(ValueError, match="T >= 2"):
        v2.sample_estimator()(np.zeros((1, 3)))
    with pytest.raises(ValueError, match="CovarianceEstimator"):
        v2.erc_with(po.sample_cov)  # type: ignore[arg-type]


@pytest.mark.parametrize("e", all_estimators(), ids=lambda e: e.name)
def test_min_variance_and_erc_on_each_estimator(history, e):
    for rule in (v2.min_variance_longonly_with(e), v2.erc_with(e)):
        w = rule(history)
        assert w.shape == (history.shape[1],)
        assert np.isfinite(w).all()
        assert math.fsum(w.tolist()) == pytest.approx(1.0, abs=1e-9)
        assert np.all(w >= -1e-12)
        np.testing.assert_array_equal(rule(history), w)
        assert rule.fallback_dates == []
    sig = e(history)
    rc = po.risk_contributions(v2.erc_with(e)(history), sig)
    np.testing.assert_allclose(rc, rc.mean(), rtol=1e-6)


def test_matches_existing_rules_on_core_estimators(history):
    np.testing.assert_allclose(
        v2.erc_with(v2.sample_estimator())(history), rs.erc()(history), rtol=1e-12
    )
    np.testing.assert_allclose(
        v2.min_variance_longonly_with(v2.lw_constant_corr_estimator())(history),
        rs.min_variance_longonly(shrink=True)(history),
        rtol=1e-12,
        atol=1e-15,
    )
    assert v2.erc_with(v2.ewma_estimator(30)).name == "erc_ewma30"
    assert v2.min_variance_longonly_with(v2.pca_estimator(2)).name == "min_var_lo_pca2"


def test_min_variance_lower_in_sample_variance_than_erc(history):
    for e in all_estimators():
        sig = e(history)
        wmv = v2.min_variance_longonly_with(e)(history)
        werc = v2.erc_with(e)(history)
        assert wmv @ sig @ wmv <= werc @ sig @ werc + 1e-15


def test_max_sharpe_bayes_stein(history):
    rule = v2.max_sharpe_bayes_stein(rf=0.0)
    w = rule(history)
    assert math.fsum(w.tolist()) == pytest.approx(1.0, abs=1e-9)
    assert np.all(w >= -1e-12)
    assert rule.name == "max_sharpe_lo_bs_sample"
    # Shrinking the means toward a common value spreads the weights at least
    # as much as the sample-mean rule does on this history.
    plain = rs.max_sharpe_longonly(rf=0.0)(history)
    assert np.count_nonzero(w > 1e-8) >= np.count_nonzero(plain > 1e-8)
    with_nls = v2.max_sharpe_bayes_stein(rf=0.0, estimator=v2.nonlinear_estimator())
    assert with_nls.name == "max_sharpe_lo_bs_nls"
    assert math.fsum(with_nls(history).tolist()) == pytest.approx(1.0, abs=1e-9)


def test_max_sharpe_bayes_stein_fallbacks(history):
    rule = v2.max_sharpe_bayes_stein(rf=0.0)
    short = history[:9]  # T = 9 <= N + 2 = 10
    w = rule(short)
    np.testing.assert_allclose(w, po.min_variance_longonly(po.annualize_cov(po.sample_cov(short))))
    assert rule.fallback_dates == [9]
    high = v2.max_sharpe_bayes_stein(rf=10.0)
    high(history)
    assert high.fallback_dates == [500]


def test_max_sharpe_bayes_stein_validation():
    with pytest.raises(ValueError, match="rf"):
        v2.max_sharpe_bayes_stein(rf=math.inf)
    for bad in (1, 2.5, True):
        with pytest.raises(ValueError, match="n_points"):
            v2.max_sharpe_bayes_stein(rf=0.0, n_points=bad)


def test_vol_target_hits_target(history):
    e = v2.ewma_estimator(30.0)
    inner = v2.erc_with(v2.sample_estimator())
    rule = v2.vol_target(inner, 0.08, e)
    w = rule(history)
    u = inner(history)
    s = rule.scales[-1]
    np.testing.assert_allclose(w, s * u, rtol=1e-15)
    assert math.sqrt(w @ e(history) @ w) == pytest.approx(0.08, rel=1e-12)
    assert math.fsum(w.tolist()) == pytest.approx(s, rel=1e-12)
    assert rule.inner is inner
    assert rule.name == "vt0.08_erc_sample_ewma30"
    rule(history[:300])
    assert len(rule.scales) == 2


def test_vol_target_cap(history):
    inner = rs.equal_weight()
    rule = v2.vol_target(inner, 5.0, v2.sample_estimator(), max_scale=1.5)
    w = rule(history)
    assert rule.scales == [1.5]
    np.testing.assert_allclose(w, 1.5 / history.shape[1])


def test_vol_target_scale_equals_backtest_with_sample_estimator(returns):
    # With the sample estimator the wrapper's scale is run_backtest_net's own.
    inner = rs.inverse_vol()
    res = rs.run_backtest_net(returns, inner, window=500, step=250, vol_target=0.1)
    rule = v2.vol_target(inner, 0.1, v2.sample_estimator())
    for k, t_k in enumerate(res.rebalance_dates):
        w = rule(returns[:t_k])
        assert rule.scales[-1] == pytest.approx(res.scales[k], rel=1e-12)
        np.testing.assert_allclose(w, res.target_weights[k], rtol=1e-12)


def test_vol_target_validation(history):
    inner = rs.equal_weight()
    e = v2.sample_estimator()
    for bad in (0.0, -0.1, math.inf, math.nan):
        with pytest.raises(ValueError, match="target_vol"):
            v2.vol_target(inner, bad, e)
    with pytest.raises(ValueError, match="max_scale"):
        v2.vol_target(inner, 0.1, e, max_scale=0.0)
    with pytest.raises(ValueError, match="inner"):
        v2.vol_target(lambda h: np.ones(2) / 2, 0.1, e)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="CovarianceEstimator"):
        v2.vol_target(inner, 0.1, None)  # type: ignore[arg-type]
    flat = np.zeros((20, 3))
    with pytest.raises(ValueError, match="ex-ante variance"):
        v2.vol_target(inner, 0.1, e)(flat)


def test_backtest_runs_on_new_rules(returns):
    for rule in (
        v2.min_variance_longonly_with(v2.nonlinear_estimator()),
        v2.erc_with(v2.ewma_estimator(60.0)),
        v2.max_sharpe_bayes_stein(rf=0.0, estimator=v2.lw_identity_estimator()),
    ):
        res = rs.run_backtest_net(returns, rule, window=500, step=126, cost_bps=5.0)
        assert np.isfinite(res.net_returns).all()
        assert res.n_trades >= 1
