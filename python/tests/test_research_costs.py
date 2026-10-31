"""Tests for portopt.research.costs (drifted-weight, cost-aware backtest)."""

from __future__ import annotations

import math

import numpy as np
import pytest

import portopt as po
from portopt.research import run_backtest_net

R3 = np.array(
    [
        [0.010, -0.020, 0.005],
        [0.000, 0.010, -0.010],
        [0.020, -0.010, 0.000],
        [-0.010, 0.030, 0.010],
        [0.005, 0.000, -0.020],
        [0.010, 0.010, 0.010],
    ]
)
W3 = np.array([0.5, 0.3, 0.2])


def const(w: np.ndarray):
    return lambda h: w.copy()


def dollar_reference(
    x: np.ndarray, targets: list[np.ndarray], dates: list[int], kappa: float, rf_d: float
) -> tuple[list[float], list[float], list[float]]:
    """Independent dollar-ledger implementation: holdings and cash in currency."""
    t = x.shape[0]
    hold = np.zeros(x.shape[1])
    cash = 1.0
    gross, net, turn = [], [], []
    for k, t_k in enumerate(dates):
        v_start = float(hold.sum() + cash)
        drift_w = hold / v_start
        traded = float(np.abs(targets[k] - drift_w).sum())
        turn.append(traded)
        new_hold = targets[k] * v_start
        cash = cash - float((new_hold - hold).sum())
        hold = new_hold
        cost = kappa * traded * v_start
        end = dates[k + 1] if k + 1 < len(dates) else t
        for row in range(t_k, end):
            v0 = float(hold.sum() + cash)
            hold = hold * (1.0 + x[row])
            cash = cash * (1.0 + rf_d)
            g_v = float(hold.sum() + cash)
            if row == t_k:
                cash -= cost
            v1 = float(hold.sum() + cash)
            gross.append(g_v / v0 - 1.0)
            net.append(v1 / v0 - 1.0)
    return gross, net, turn


def test_buy_and_hold_hand_computed():
    # One rebalance at row 2 (step larger than the sample), no costs.
    res = run_backtest_net(R3, const(W3), window=2, step=100)
    assert res.rebalance_dates == [2]
    growth = np.cumprod(1.0 + R3[2:], axis=0)
    nav = np.concatenate([[1.0], growth @ W3])
    expected = nav[1:] / nav[:-1] - 1.0
    np.testing.assert_allclose(res.net_returns, expected, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(res.gross_returns, res.net_returns)
    # Drift arithmetic on day one, by hand: w_i (1 + r_i) / (1 + w'r).
    r_p = float(W3 @ R3[2])
    assert r_p == pytest.approx(0.5 * 0.02 + 0.3 * -0.01 + 0.2 * 0.0, abs=1e-15)
    assert res.turnovers.tolist() == [pytest.approx(1.0, abs=1e-15)]
    assert res.total_cost == 0.0


def test_drift_turnover_and_costs_match_dollar_ledger():
    res = run_backtest_net(R3, const(W3), window=2, step=2, cost_bps=10.0, rf=0.05)
    assert res.rebalance_dates == [2, 4]
    gross, net, turn = dollar_reference(R3, [W3, W3], [2, 4], 1e-3, 0.05 / 252)
    np.testing.assert_allclose(res.gross_returns, gross, rtol=0, atol=1e-14)
    np.testing.assert_allclose(res.net_returns, net, rtol=0, atol=1e-14)
    np.testing.assert_allclose(res.turnovers, turn, rtol=0, atol=1e-14)
    # Drifted weights at row 4, by hand from the two holding days.
    v = W3 * (1 + R3[2]) * (1 + R3[3])
    cash = -1e-3 * (1 + 0.05 / 252)  # initial build cost, then one day of rf
    drift = v / (v.sum() + cash)
    np.testing.assert_allclose(res.pre_trade_weights[1], drift, rtol=0, atol=1e-14)
    assert res.turnovers[1] == pytest.approx(float(np.abs(W3 - drift).sum()), abs=1e-14)
    assert res.turnovers[1] > 0.0  # drift is charged even though the target is constant


def test_gross_minus_net_equals_costs(returns):
    ew = const(np.full(returns.shape[1], 1.0 / returns.shape[1]))
    res = run_backtest_net(returns, ew, cost_bps=25.0, rf=0.02)
    np.testing.assert_allclose(res.gross_returns - res.net_returns, res.costs, rtol=0, atol=1e-17)
    assert res.total_cost == pytest.approx(math.fsum(res.costs.tolist()), abs=0)
    assert res.total_cost == pytest.approx(25e-4 * float(res.turnovers.sum()), rel=1e-12)
    trade_days = [d - 500 for d in res.rebalance_dates]
    assert np.all(res.costs[np.setdiff1d(np.arange(res.costs.size), trade_days)] == 0.0)
    assert res.ann_cost == pytest.approx(252 * res.total_cost / res.costs.size)
    assert res.sharpe < res.gross_sharpe


def test_zero_cost_net_equals_gross(returns):
    res = run_backtest_net(returns, const(np.full(8, 0.125)))
    np.testing.assert_array_equal(res.net_returns, res.gross_returns)
    assert res.n_trades == len(res.rebalance_dates)


def test_large_band_is_buy_and_hold(returns):
    w = np.full(8, 0.125)
    banded = run_backtest_net(returns, const(w), band=10.0, cost_bps=5.0)
    held = run_backtest_net(returns, const(w), step=10**6, cost_bps=5.0)
    assert banded.executed.tolist() == [True] + [False] * (len(banded.rebalance_dates) - 1)
    assert np.all(banded.turnovers[1:] == 0.0)
    np.testing.assert_allclose(banded.net_returns, held.net_returns, rtol=0, atol=1e-15)
    assert banded.n_trades == 1


def test_band_threshold_is_strict_on_distance():
    res = run_backtest_net(R3, const(W3), window=2, step=2)
    dist = float(np.abs(W3 - res.pre_trade_weights[1]).sum())
    below = run_backtest_net(R3, const(W3), window=2, step=2, band=dist * (1 + 1e-9))
    above = run_backtest_net(R3, const(W3), window=2, step=2, band=dist * (1 - 1e-9))
    assert below.executed.tolist() == [True, False]
    assert above.executed.tolist() == [True, True]


def test_partial_rebalance_moves_fraction_of_gap(returns):
    rng = np.random.default_rng(7)
    targets = {t: v / v.sum() for t, v in ((t, rng.random(8)) for t in range(500, 1250, 63))}
    res = run_backtest_net(returns, lambda h: targets[h.shape[0]], partial=0.25)
    np.testing.assert_allclose(res.post_trade_weights[0], res.target_weights[0], atol=0)
    for k in range(1, len(res.rebalance_dates)):
        gap = res.target_weights[k] - res.pre_trade_weights[k]
        np.testing.assert_allclose(
            res.post_trade_weights[k], res.pre_trade_weights[k] + 0.25 * gap, atol=1e-15
        )
        assert res.turnovers[k] == pytest.approx(0.25 * float(np.abs(gap).sum()), rel=1e-12)


def test_vol_target_scales_to_ex_ante_vol(returns):
    w = np.full(8, 0.125)
    res = run_backtest_net(returns, const(w), vol_target=0.05, rf=0.03)
    for k, t_k in enumerate(res.rebalance_dates):
        sig = po.annualize_cov(po.sample_cov(returns[:t_k]))
        assert res.scales[k] * math.sqrt(w @ sig @ w) == pytest.approx(0.05, rel=1e-12)
        np.testing.assert_allclose(res.target_weights[k], res.scales[k] * w, rtol=1e-15)
    s0 = res.scales[0]
    assert s0 < 1.0
    day0 = s0 * float(w @ returns[500]) + (1.0 - s0) * 0.03 / 252
    assert res.gross_returns[0] == pytest.approx(day0, abs=1e-15)


def test_vol_estimator_default_and_custom(returns):
    w = np.full(8, 0.125)
    base = run_backtest_net(returns, const(w), vol_target=0.05, cost_bps=5)

    def sample_annual(h: np.ndarray) -> np.ndarray:
        return po.annualize_cov(po.sample_cov(h))

    same = run_backtest_net(
        returns, const(w), vol_target=0.05, cost_bps=5, vol_estimator=sample_annual
    )
    np.testing.assert_array_equal(base.scales, same.scales)
    np.testing.assert_array_equal(base.net_returns, same.net_returns)

    def doubled(h: np.ndarray) -> np.ndarray:
        return 2.0 * sample_annual(h)

    dbl = run_backtest_net(returns, const(w), vol_target=0.05, cost_bps=5, vol_estimator=doubled)
    np.testing.assert_allclose(dbl.scales, base.scales / math.sqrt(2.0), rtol=1e-12)
    # The estimator is only consulted when vol_target is set.
    plain = run_backtest_net(returns, const(w), vol_estimator=doubled)
    np.testing.assert_array_equal(plain.scales, np.ones_like(plain.scales))


def test_vol_estimator_validation(returns):
    w = np.full(8, 0.125)
    with pytest.raises(ValueError, match="vol_estimator must be None or callable"):
        run_backtest_net(returns, const(w), vol_target=0.05, vol_estimator=3.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite 8 x 8 matrix"):
        run_backtest_net(returns, const(w), vol_target=0.05, vol_estimator=lambda h: np.eye(3))
    bad = np.full((8, 8), np.nan)
    with pytest.raises(ValueError, match="finite 8 x 8 matrix"):
        run_backtest_net(returns, const(w), vol_target=0.05, vol_estimator=lambda h: bad)


def test_max_gross_caps_scaled_target(returns):
    w = np.zeros(8)
    w[0], w[1] = 2.0, -1.0
    res = run_backtest_net(returns, const(w), max_gross=1.5)
    for k in range(len(res.rebalance_dates)):
        assert po.gross_leverage(res.target_weights[k]) == pytest.approx(1.5, rel=1e-12)
    res2 = run_backtest_net(returns, const(np.full(8, 0.125)), vol_target=0.5, max_gross=1.2)
    assert np.all(res2.target_weights.sum(axis=1) <= 1.2 + 1e-12)


def test_deterministic(returns):
    a = run_backtest_net(returns, const(np.full(8, 0.125)), cost_bps=10, band=0.05)
    b = run_backtest_net(returns, const(np.full(8, 0.125)), cost_bps=10, band=0.05)
    np.testing.assert_array_equal(a.net_returns, b.net_returns)
    np.testing.assert_array_equal(a.turnovers, b.turnovers)


def test_history_is_read_only(returns):
    def bad(h):
        h[0, 0] = 1.0
        return np.full(8, 0.125)

    with pytest.raises(ValueError):
        run_backtest_net(returns, bad)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"cost_bps": -1.0}, "cost_bps"),
        ({"cost_bps": float("nan")}, "cost_bps"),
        ({"band": -0.1}, "band"),
        ({"partial": 0.0}, "partial"),
        ({"partial": 1.5}, "partial"),
        ({"rf": float("inf")}, "rf"),
        ({"vol_target": 0.0}, "vol_target"),
        ({"max_gross": -1.0}, "max_gross"),
        ({"window": 1}, "window"),
        ({"window": 2.5}, "window"),
        ({"step": 0}, "step"),
        ({"window": 5}, "at least 2 holdout days"),
    ],
)
def test_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        run_backtest_net(R3, const(W3), **{"window": 2, **kwargs})


def test_weight_fn_validation():
    with pytest.raises(ValueError, match="invalid weight vector"):
        run_backtest_net(R3, const(np.array([0.5, 0.5])), window=2)
    with pytest.raises(ValueError, match="sum to 1"):
        run_backtest_net(R3, const(np.array([0.5, 0.5, 0.5])), window=2)
    with pytest.raises(ValueError, match="callable"):
        run_backtest_net(R3, None, window=2)  # type: ignore[arg-type]
    bad = R3.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        run_backtest_net(bad, const(W3), window=2)


def test_ruin_raises():
    x = np.array([[0.0, 0.0], [0.0, 0.0], [0.5, -0.5], [0.0, 0.0]])
    with pytest.raises(ValueError, match="net asset value"):
        run_backtest_net(x, const(np.array([-3.0, 4.0])), window=2)
