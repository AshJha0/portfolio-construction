"""Tests for portopt.research.strategies weight functions."""

from __future__ import annotations

import math

import numpy as np
import pytest

import portopt as po
from portopt import research as rs
from portopt.research.strategies import cap_toward


@pytest.fixture(scope="module")
def history(returns) -> np.ndarray:
    return returns[:500]


@pytest.fixture(scope="module")
def bl_inputs(asset_names, views_doc, market_caps):
    p, q, conf = po.views_to_matrices(views_doc["views"], asset_names)
    return {
        "w_mkt": market_caps,
        "p": p,
        "q": q,
        "tau": views_doc["tau"],
        "delta": views_doc["delta"],
        "confidences": conf.tolist(),
    }


def all_rules(bl_inputs):
    return [
        rs.equal_weight(),
        rs.inverse_vol(),
        rs.inverse_vol(shrink=True),
        rs.erc(),
        rs.erc(shrink=True),
        rs.min_variance_longonly(),
        rs.min_variance_longonly(shrink=True),
        rs.max_sharpe_longonly(rf=0.02),
        rs.tangency_capped(rf=0.0),
        rs.tangency_capped(rf=0.0, max_gross=2.0),
        rs.black_litterman(**bl_inputs),
        rs.black_litterman(**bl_inputs, max_gross=1.2),
    ]


def test_every_rule_returns_valid_weights(history, bl_inputs):
    for rule in all_rules(bl_inputs):
        w = rule(history)
        assert w.shape == (8,), rule.name
        assert np.isfinite(w).all(), rule.name
        assert math.fsum(w.tolist()) == pytest.approx(1.0, abs=1e-9), rule.name


def test_rules_match_core_functions(history, bl_inputs):
    s = po.annualize_cov(po.sample_cov(history))
    s_lw = po.annualize_cov(po.shrink_covariance(history).sigma)
    mu = po.annualize_mean(po.sample_mean(history))
    np.testing.assert_array_equal(rs.inverse_vol()(history), po.inverse_vol_weights(s))
    np.testing.assert_array_equal(rs.erc(shrink=True)(history), po.erc_weights(s_lw))
    np.testing.assert_array_equal(rs.min_variance_longonly()(history), po.min_variance_longonly(s))
    np.testing.assert_array_equal(
        rs.tangency_capped(rf=0.0)(history), po.tangency_weights(mu, s, 0.0)
    )
    pi = po.implied_equilibrium_returns(s, bl_inputs["w_mkt"], bl_inputs["delta"])
    w_bl, _ = po.bl_weights(
        s, pi, bl_inputs["p"], bl_inputs["q"], bl_inputs["tau"],
        confidences=bl_inputs["confidences"],
    )  # fmt: skip
    np.testing.assert_array_equal(rs.black_litterman(**bl_inputs)(history), w_bl)


def test_max_sharpe_beats_every_frontier_point(history):
    rf = 0.0
    s = po.annualize_cov(po.sample_cov(history))
    mu = po.annualize_mean(po.sample_mean(history))
    w = rs.max_sharpe_longonly(rf=rf)(history)
    targets, _, var = po.constrained_frontier(mu, s, 25)
    best = (float(mu @ w) - rf) / math.sqrt(float(w @ s @ w))
    assert best >= float(np.max((targets - rf) / np.sqrt(var))) - 1e-12
    assert np.all(w >= 0.0)


def test_tangency_cap_and_fallback(history):
    capped = rs.tangency_capped(rf=0.0, max_gross=2.0)
    w = capped(history)
    assert po.gross_leverage(w) <= 2.0 + 1e-12
    assert po.gross_leverage(rs.tangency_capped(rf=0.0)(history)) > 2.0  # the cap binds
    huge = rs.tangency_capped(rf=10.0)
    s = po.annualize_cov(po.sample_cov(history))
    np.testing.assert_array_equal(huge(history), po.min_variance_weights(s))
    assert huge.fallback_dates == [500]


def test_max_sharpe_falls_back_when_nothing_beats_rf(history):
    rule = rs.max_sharpe_longonly(rf=10.0)
    s = po.annualize_cov(po.sample_cov(history))
    np.testing.assert_array_equal(rule(history), po.min_variance_longonly(s))
    assert rule.fallback_dates == [500]


def test_cap_toward_is_on_the_segment_and_tight():
    w = np.array([3.0, -1.5, -0.5])
    a = np.array([0.4, 0.4, 0.2])
    out = cap_toward(w, a, 2.0)
    assert po.gross_leverage(out) == pytest.approx(2.0, abs=1e-12)
    alpha = (out - a) / (w - a)
    np.testing.assert_allclose(alpha, alpha[0], atol=1e-12)
    assert 0.0 < alpha[0] < 1.0
    np.testing.assert_array_equal(cap_toward(a, w, 2.0), a)  # already within the cap
    with pytest.raises(ValueError, match="anchor"):
        cap_toward(w, w, 2.0)


def test_validation(bl_inputs):
    with pytest.raises(ValueError, match="max_gross"):
        rs.tangency_capped(rf=0.0, max_gross=0.5)
    with pytest.raises(ValueError, match="rf"):
        rs.max_sharpe_longonly(rf=float("nan"))
    with pytest.raises(ValueError, match="n_points"):
        rs.max_sharpe_longonly(rf=0.0, n_points=1)
    with pytest.raises(ValueError, match="w_mkt"):
        rs.black_litterman(**{**bl_inputs, "w_mkt": np.array([-1.0, 2.0])})
    with pytest.raises(ValueError, match="history"):
        rs.equal_weight()(np.zeros(5))


def test_rules_run_in_net_backtest(returns, bl_inputs):
    for rule in all_rules(bl_inputs):
        res = rs.run_backtest_net(returns, rule, cost_bps=10.0, rf=0.02)
        assert np.isfinite(res.sharpe), rule.name
