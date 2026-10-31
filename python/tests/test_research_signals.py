"""Tests for :mod:`portopt.research.signals`.

The look-ahead tests run each rule inside ``run_schedule_net`` on a panel,
perturb every row on or after one rebalance row, and assert that every
weight decided at or before that row is identical.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from portopt.markowitz import mean_variance_turnover_solve
from portopt.research import signals as sg
from portopt.research import walkforward as wf


def _panel(t: int = 900, n: int = 6, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    common = rng.normal(0.0003, 0.009, size=(t, 1))
    return common * rng.uniform(0.6, 1.4, size=(1, n)) + rng.normal(0.0002, 0.006, size=(t, n))


def test_momentum_skips_last_month() -> None:
    x = _panel()
    y = x.copy()
    y[-sg.MONTH :] += 0.05  # last month does not enter 12-1 momentum
    np.testing.assert_array_equal(sg.momentum_12_1(x), sg.momentum_12_1(y))
    expect = np.prod(1.0 + x[-252:-21], axis=0) - 1.0
    np.testing.assert_allclose(sg.momentum_12_1(x), expect, rtol=0, atol=1e-15)


def test_reversal_and_low_vol_and_beta_definitions() -> None:
    x = _panel()
    np.testing.assert_allclose(
        sg.short_term_reversal(x), -(np.prod(1.0 + x[-21:], axis=0) - 1.0), atol=1e-15
    )
    np.testing.assert_allclose(
        sg.low_volatility(x), -np.std(x[-252:], axis=0, ddof=1) * math.sqrt(252), atol=1e-15
    )
    w = x[-252:]
    m = w.mean(axis=1)
    beta = [np.polyfit(m, w[:, i], 1)[0] for i in range(w.shape[1])]
    np.testing.assert_allclose(sg.low_beta(x), -np.array(beta), atol=1e-10)


def test_tsmom_sign_and_scaling() -> None:
    x = _panel()
    s = sg.tsmom(x, 0.0)
    vol = np.std(x[-252:], axis=0, ddof=1) * math.sqrt(252)
    np.testing.assert_allclose(np.abs(s), 1.0 / vol)
    huge = sg.tsmom(x, 0.01)  # 1% a day of risk-free: every excess return negative
    assert np.all(huge < 0)


def test_zscore_and_combine() -> None:
    z = sg.zscore(np.array([1.0, 2.0, 3.0]))
    assert abs(z.mean()) < 1e-15 and abs(np.std(z) - 1.0) < 1e-12
    np.testing.assert_array_equal(sg.zscore(np.ones(4)), np.zeros(4))
    x = _panel()
    one = sg.combine(["strev"], x)
    np.testing.assert_allclose(one, sg.zscore(sg.short_term_reversal(x)))
    iv = sg.combine(["mom12_1", "strev"], x, method="inverse_vol")
    assert iv.shape == (6,) and abs(np.std(iv) - 1.0) < 1e-12
    with pytest.raises(ValueError):
        sg.combine([], x)


def test_expected_returns_rule() -> None:
    a = sg.signal_to_expected_returns(np.array([1.0, -2.0]), np.array([0.2, 0.1]), 0.05)
    np.testing.assert_allclose(a, [0.01, -0.01])
    with pytest.raises(ValueError):
        sg.signal_to_expected_returns(np.ones(2), np.ones(2), 1.5)


def test_top_tercile() -> None:
    w = sg.top_fraction_weights(np.array([0.1, 3.0, 2.0, -1.0, 2.0, 0.0]))
    np.testing.assert_allclose(w, [0, 0.5, 0.5, 0, 0, 0])  # tie at 2.0 broken by column order


def _weights_until(fn_factory, x: np.ndarray, rows: list[int]) -> np.ndarray:  # type: ignore[no-untyped-def]
    res = wf.run_schedule_net(x, fn_factory(), rows, cost_bps=10.0)
    return np.asarray(res.post_trade_weights)


RULES = {
    "rank": lambda: sg.rank_top_tercile(signals=["mom12_1", "tsmom", "strev", "lowbeta"]),
    "rank_iv": lambda: sg.rank_top_tercile(signals=["mom12_1", "strev"], method="inverse_vol"),
    "lowvol": lambda: sg.rank_top_tercile(signals=["lowvol"]),
    "bl": lambda: sg.signal_black_litterman(signals=["mom12_1", "strev"], lookback=504),
    "mvtc": lambda: sg.SignalMeanVariance(signals=["mom12_1", "lowbeta"], lookback=504),
}


@pytest.mark.parametrize("name", sorted(RULES))
def test_no_look_ahead(name: str) -> None:
    x = _panel(t=900, n=5)
    rows = list(range(520, 880, 21))
    k = 8
    base = _weights_until(RULES[name], x, rows)
    y = x.copy()
    rng = np.random.default_rng(99)
    y[rows[k] :] = rng.normal(0.0, 0.03, size=y[rows[k] :].shape)  # rewrite the future
    pert = _weights_until(RULES[name], y, rows)
    np.testing.assert_array_equal(base[: k + 1], pert[: k + 1])
    assert not np.array_equal(base[k + 1 :], pert[k + 1 :])


def test_mean_variance_drift_and_solver() -> None:
    x = _panel(t=700, n=5)
    rule = sg.SignalMeanVariance(signals=["mom12_1"], lookback=504, cost_bps=10.0)
    w1 = rule(x[:600])
    w2 = rule(x[:621])
    assert np.all(w1 >= 0) and abs(w1.sum() - 1) < 1e-9 and abs(w2.sum() - 1) < 1e-9
    v = w1 * np.prod(1.0 + x[600:621], axis=0)
    w0 = v / v.sum()
    sig = rule.est(x[621 - 504 : 621])
    alpha = 12.0 * sg.signal_to_expected_returns(
        sg.combine(["mom12_1"], x[:621]), np.sqrt(np.diag(sig) / 12.0), 0.05
    )
    ref = mean_variance_turnover_solve(alpha, sig, w0, 5.0, 12.0 * 10.0 / 1e4 / 5.0)
    np.testing.assert_allclose(w2, ref.weights, atol=1e-9)
    with pytest.raises(ValueError):
        rule(x[:621])  # history must grow


def test_short_history_raises() -> None:
    with pytest.raises(ValueError):
        sg.momentum_12_1(_panel(t=100))
    with pytest.raises(ValueError):
        sg.rank_top_tercile(signals=["nope"])
