"""Tests for the quarterly expanding-window backtest and metrics."""
import numpy as np
import pytest

import portopt as po


def equal_fn(h: np.ndarray) -> np.ndarray:
    return np.full(h.shape[1], 1.0 / h.shape[1])


def test_rebalance_schedule_pinned():
    assert po.rebalance_indices(1250, 500, 63) == [
        500, 563, 626, 689, 752, 815, 878, 941, 1004, 1067, 1130, 1193
    ]
    assert po.rebalance_indices(600, 500, 63) == [500, 563]
    with pytest.raises(ValueError, match="exceed"):
        po.rebalance_indices(400, 500, 63)
    with pytest.raises(ValueError, match="step"):
        po.rebalance_indices(600, 500, 0)


def test_metric_hand_values():
    r = np.array([0.01, -0.02, 0.015, 0.0, 0.005])
    assert po.annualized_return(r) == pytest.approx(252 * np.mean(r), abs=1e-15)
    assert po.annualized_vol(r) == pytest.approx(
        np.sqrt(252) * np.std(r, ddof=1), abs=1e-15
    )
    sr = po.sharpe_ratio(r, rf=0.01)
    assert sr == pytest.approx(
        (252 * np.mean(r) - 0.01) / (np.sqrt(252) * np.std(r, ddof=1)), abs=1e-12
    )


def test_max_drawdown_hand_case():
    # wealth: 1.1, 0.99, 1.089, 0.9801 -> peak 1.1, trough 0.9801
    r = np.array([0.10, -0.10, 0.10, -0.10])
    assert po.max_drawdown(r) == pytest.approx(1.0 - 0.9801 / 1.1, abs=1e-12)
    assert po.max_drawdown(np.array([0.01, 0.02])) == 0.0


def test_sharpe_zero_vol_rejected():
    with pytest.raises(ValueError, match="volatility is zero"):
        po.sharpe_ratio(np.full(10, 0.001))


def test_metric_input_validation():
    with pytest.raises(ValueError, match="NaN or infinite"):
        po.annualized_return(np.array([0.01, np.nan]))
    with pytest.raises(ValueError, match="at least 2"):
        po.annualized_vol(np.array([0.01]))


def test_backtest_accounting_identity(returns):
    """The portfolio return series must equal block-wise w'r exactly and
    cover every holdout day exactly once."""
    res = po.run_backtest(returns, equal_fn, rf=0.02)
    t, n = returns.shape
    assert res.portfolio_returns.shape[0] == t - 500
    manual = np.zeros(t - 500)
    dates = res.rebalance_dates
    for k, t_k in enumerate(dates):
        t_end = dates[k + 1] if k + 1 < len(dates) else t
        manual[t_k - 500 : t_end - 500] = returns[t_k:t_end] @ res.weights[k]
    assert np.array_equal(res.portfolio_returns, manual)
    # equal weights every quarter -> zero turnover
    assert np.allclose(res.turnovers, 0.0, atol=0)
    assert res.avg_turnover == 0.0


def test_backtest_metrics_consistent(returns):
    res = po.run_backtest(returns, equal_fn, rf=0.02)
    r = res.portfolio_returns
    assert res.ann_return == pytest.approx(po.annualized_return(r), abs=1e-15)
    assert res.ann_vol == pytest.approx(po.annualized_vol(r), abs=1e-15)
    assert res.sharpe == pytest.approx(po.sharpe_ratio(r, 0.02), abs=1e-15)
    assert res.max_drawdown == pytest.approx(po.max_drawdown(r), abs=1e-15)


def test_backtest_turnover(returns):
    """Alternating weight function gives a known constant turnover."""
    state = {"k": 0}

    def alternating(h):
        n = h.shape[1]
        w = np.zeros(n)
        w[state["k"] % 2] = 1.0
        state["k"] += 1
        return w

    res = po.run_backtest(returns, alternating, rf=0.0)
    assert np.allclose(res.turnovers, 2.0, atol=0)
    assert res.avg_turnover == pytest.approx(2.0, abs=1e-15)


def test_backtest_expanding_window(returns):
    """The weight function must see history [0, t_k) — check lengths."""
    seen = []

    def spy(h):
        seen.append(h.shape[0])
        return equal_fn(h)

    po.run_backtest(returns, spy, rf=0.0)
    assert seen == po.rebalance_indices(returns.shape[0], 500, 63)


def test_backtest_rejects_bad_weight_fn(returns):
    with pytest.raises(ValueError, match="sum to 1"):
        po.run_backtest(returns, lambda h: np.full(h.shape[1], 0.2), rf=0.0)
    with pytest.raises(ValueError, match="invalid weight"):
        po.run_backtest(returns, lambda h: np.full(h.shape[1], np.nan), rf=0.0)


def test_backtest_determinism(returns):
    def erc_fn(h):
        return po.erc_weights(po.sample_cov(h))

    r1 = po.run_backtest(returns, erc_fn, rf=0.02)
    r2 = po.run_backtest(returns, erc_fn, rf=0.02)
    assert np.array_equal(r1.portfolio_returns, r2.portfolio_returns)
    assert np.array_equal(r1.weights, r2.weights)
    assert r1.sharpe == r2.sharpe
