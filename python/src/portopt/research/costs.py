"""Cost-aware backtest with drifting weights.

The core :func:`portopt.run_backtest` is pinned to a simple model: target
weights are held constant every day (an implicit daily rebalance whose
trading is never counted) and nothing is charged for trading.  This
module replaces that model with the bookkeeping a desk would use.  It is
part of the Python-only research layer and is not covered by API_SPEC.md.

Schedule.  Rebalance dates are the pinned ``t_k = window + step k`` of
:func:`portopt.rebalance_indices`.  At ``t_k`` the weight function sees
rows ``[0, t_k)`` (a read-only view) and returns target weights summing
to one.  The holdout is rows ``[window, T)``.

State.  We track the risky weights ``w`` (fractions of start-of-day net
asset value) and the cash weight ``c = 1 - sum(w)`` before costs.  Cash
earns ``rf_d = rf / 252`` per day; negative cash is borrowing at the same
rate.

Rebalance at ``t_k`` (executed at the start of the day, before row
``t_k``'s returns):

1. Scaling.  The raw target ``u`` is multiplied by a scalar ``s``.  With
   ``vol_target`` set, ``s = vol_target / sqrt(u' Sigma u)`` where Sigma is
   the annualized sample covariance of the history.  With ``max_gross``
   set, ``s`` is then reduced, if needed, so that ``s * sum|u_i| <=
   max_gross``.  Whatever is not in risky assets sits in cash.
2. Decision.  The first rebalance (the initial build from all cash) is
   always executed in full.  Later, with drifted weights ``v`` and scaled
   target ``u* = s u``, the trade is skipped when
   ``sum_i |u*_i - v_i| < band``; otherwise the new weights are
   ``v + partial (u* - v)``.
3. Cost.  Traded notional is ``tau = sum_i |w_new_i - v_i|``: each unit
   of risky notional bought or sold is counted once (the cash leg is not
   charged).  The cost is ``kappa tau`` with ``kappa = cost_bps / 1e4``,
   as a fraction of start-of-day NAV, paid from cash at the end of the day.

Daily accounting.  For row ``t`` with asset returns ``r_t``::

    gross_t = w' r_t + c rf_d
    net_t   = gross_t - cost_t           (cost_t = 0 except on trade days)
    w      <- w * (1 + r_t) / (1 + net_t)
    c      <- (c (1 + rf_d) - cost_t) / (1 + net_t)

so ``sum(w) + c = 1`` is preserved and ``gross_t - net_t = cost_t``
exactly.  ``gross_t`` is the return of the held portfolio before the
cost of reaching it; it is not the return of a separate cost-free path.
Within a holding block the portfolio is buy and hold, so drift turnover
shows up at the next rebalance.

Metrics use the core pinned formulas (arithmetic annualization by 252,
``ddof=1`` volatility, Sharpe ``(ann_return - rf) / ann_vol``).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from portopt import (
    annualize_cov,
    annualized_return,
    annualized_vol,
    gross_leverage,
    max_drawdown,
    rebalance_indices,
    sample_cov,
    sharpe_ratio,
    validate_returns,
)

__all__ = ["NetBacktestResult", "run_backtest_net"]

ANNUALIZATION = 252


@dataclass(frozen=True)
class NetBacktestResult:
    """Result of :func:`run_backtest_net`.

    Daily arrays cover the holdout rows ``[window, T)``; per-rebalance
    arrays have one entry per date in ``rebalance_dates`` (K entries).

    Attributes
    ----------
    gross_returns, net_returns, costs:
        Daily returns before and after costs, and the daily cost (fraction
        of start-of-day NAV); ``gross_returns - net_returns == costs``.
    rebalance_dates:
        Row indices of the scheduled rebalances.
    target_weights:
        Scaled target risky weights per rebalance (K x n).
    pre_trade_weights, post_trade_weights:
        Risky weights just before and just after each rebalance (K x n);
        equal when the trade was skipped.
    executed:
        Whether each rebalance traded (False when the no-trade band held).
    turnovers:
        Traded notional ``sum_i |w_new_i - v_i|`` per rebalance (index 0
        is the initial build; 0.0 where skipped).
    scales:
        Volatility-target and leverage-cap scalar applied per rebalance.
    total_cost:
        Sum of daily costs (fractions of NAV).
    ann_cost:
        ``252 * total_cost / holdout_days``.
    avg_turnover:
        Mean traded notional over rebalances ``k >= 1`` (skipped ones
        count as zero); 0.0 when there is only one rebalance.
    annual_turnover:
        ``252 * sum(turnovers[1:]) / holdout_days`` (initial build excluded).
    n_trades:
        Number of executed rebalances, initial build included.
    ann_return, ann_vol, sharpe, max_drawdown:
        Net-of-cost metrics (core formulas).
    gross_sharpe:
        Sharpe ratio of the gross daily returns.
    """

    gross_returns: np.ndarray
    net_returns: np.ndarray
    costs: np.ndarray
    rebalance_dates: list[int]
    target_weights: np.ndarray
    pre_trade_weights: np.ndarray
    post_trade_weights: np.ndarray
    executed: np.ndarray
    turnovers: np.ndarray
    scales: np.ndarray
    total_cost: float
    ann_cost: float
    avg_turnover: float
    annual_turnover: float
    n_trades: int
    ann_return: float
    ann_vol: float
    sharpe: float
    gross_sharpe: float
    max_drawdown: float


def _finite_nonneg(name: str, value: float) -> float:
    v = float(value)
    if not (math.isfinite(v) and v >= 0.0):
        raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
    return v


def _optional_positive(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    v = float(value)
    if not (math.isfinite(v) and v > 0.0):
        raise ValueError(f"{name} must be None or finite and > 0, got {value!r}")
    return v


def _int_param(name: str, value: int, minimum: int) -> int:
    if isinstance(value, bool) or int(value) != value:
        raise ValueError(f"{name} must be an integer, got {value!r}")
    v = int(value)
    if v < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {v}")
    return v


def _fsum_dot(a: np.ndarray, b: np.ndarray) -> float:
    return math.fsum((a * b).tolist())


def run_backtest_net(
    returns: np.ndarray,
    weight_fn: Callable[[np.ndarray], np.ndarray],
    *,
    window: int = 500,
    step: int = 63,
    cost_bps: float = 0.0,
    band: float = 0.0,
    partial: float = 1.0,
    rf: float = 0.0,
    vol_target: float | None = None,
    max_gross: float | None = None,
) -> NetBacktestResult:
    """Backtest with drifted weights, transaction costs and trading rules.

    See the module docstring for the exact accounting.

    Parameters
    ----------
    returns:
        T x n simple daily returns (validated as in the core library).
    weight_fn:
        ``weight_fn(history) -> weights``; must return n finite weights
        summing to one within 1e-6.
    window, step:
        Estimation window (>= 2) and rebalance step (>= 1) in rows.
    cost_bps:
        Cost in basis points per unit of traded risky notional (>= 0).
    band:
        No-trade band on the L1 distance between drifted and target risky
        weights (>= 0; 0 means always trade).  Not applied to the initial
        build.
    partial:
        Fraction in (0, 1] of the gap to target closed at each executed
        rebalance after the first.  The initial build is always full.
    rf:
        Annual risk-free rate; cash accrues ``rf / 252`` per day.
    vol_target:
        Annualized ex-ante volatility target (None disables it).
    max_gross:
        Cap on ``sum_i |w_i|`` of the scaled target (None disables it).

    Raises
    ------
    ValueError
        On invalid inputs, an invalid weight vector, a zero ex-ante
        volatility under ``vol_target``, or a day on which net asset
        value would fall to zero or below.
    """
    x = validate_returns(returns, min_rows=3)
    t, n = x.shape
    win = _int_param("window", window, 2)
    stp = _int_param("step", step, 1)
    if t < win + 2:
        raise ValueError(
            f"backtest needs at least 2 holdout days: sample length {t} must be "
            f">= window + 2 = {win + 2}"
        )
    kappa = _finite_nonneg("cost_bps", cost_bps) / 1.0e4
    band_v = _finite_nonneg("band", band)
    part = float(partial)
    if not (math.isfinite(part) and 0.0 < part <= 1.0):
        raise ValueError(f"partial must lie in (0, 1], got {partial!r}")
    rfv = float(rf)
    if not math.isfinite(rfv):
        raise ValueError("rf must be finite")
    vt = _optional_positive("vol_target", vol_target)
    mg = _optional_positive("max_gross", max_gross)
    if not callable(weight_fn):
        raise ValueError("weight_fn must be callable")

    rf_d = rfv / ANNUALIZATION
    dates = rebalance_indices(t, win, stp)
    k_reb = len(dates)
    h_days = t - win
    gross = np.zeros(h_days)
    net = np.zeros(h_days)
    costs = np.zeros(h_days)
    targets = np.zeros((k_reb, n))
    pre = np.zeros((k_reb, n))
    post = np.zeros((k_reb, n))
    executed = np.zeros(k_reb, dtype=bool)
    turnovers = np.zeros(k_reb)
    scales = np.ones(k_reb)

    w = np.zeros(n)
    cash = 1.0
    for k, t_k in enumerate(dates):
        history = x[:t_k].view()
        history.flags.writeable = False
        u = np.asarray(weight_fn(history), dtype=float).reshape(-1)
        if u.shape[0] != n or not np.isfinite(u).all():
            raise ValueError(f"weight_fn returned an invalid weight vector at t={t_k}")
        if abs(math.fsum(u.tolist()) - 1.0) > 1e-6:
            raise ValueError(
                f"weight_fn weights must sum to 1 (got {math.fsum(u.tolist()):.8f}) at t={t_k}"
            )
        s = 1.0
        if vt is not None:
            sig = annualize_cov(sample_cov(x[:t_k]), ANNUALIZATION)
            var = float(u @ sig @ u)
            if not var > 0.0:
                raise ValueError(f"vol_target: ex-ante volatility is zero at t={t_k}")
            s = vt / math.sqrt(var)
        if mg is not None:
            g = s * gross_leverage(u)
            if g > mg:
                s *= mg / g
        target = s * u
        targets[k] = target
        scales[k] = s
        pre[k] = w
        if k == 0:
            new = target.copy()
        elif math.fsum(np.abs(target - w).tolist()) < band_v:
            new = w.copy()
        else:
            new = w + part * (target - w)
        traded = math.fsum(np.abs(new - w).tolist())
        executed[k] = k == 0 or not np.array_equal(new, w)
        turnovers[k] = traded
        post[k] = new
        # Self-financing trade: the cash weight absorbs the change in risky weight.
        cash = cash + math.fsum((w - new).tolist())
        w = new
        cost_today = kappa * traded
        t_end = dates[k + 1] if k + 1 < k_reb else t
        for row in range(t_k, t_end):
            r = x[row]
            g_r = _fsum_dot(w, r) + cash * rf_d
            c_r = cost_today if row == t_k else 0.0
            n_r = g_r - c_r
            if not 1.0 + n_r > 0.0:
                raise ValueError(
                    f"net asset value fell to zero or below at row {row}; reduce leverage or costs"
                )
            gross[row - win] = g_r
            net[row - win] = n_r
            costs[row - win] = c_r
            w = w * (1.0 + r) / (1.0 + n_r)
            cash = (cash * (1.0 + rf_d) - c_r) / (1.0 + n_r)

    total_cost = math.fsum(costs.tolist())
    later = turnovers[1:]
    avg_to = math.fsum(later.tolist()) / later.size if later.size else 0.0
    return NetBacktestResult(
        gross_returns=gross,
        net_returns=net,
        costs=costs,
        rebalance_dates=dates,
        target_weights=targets,
        pre_trade_weights=pre,
        post_trade_weights=post,
        executed=executed,
        turnovers=turnovers,
        scales=scales,
        total_cost=total_cost,
        ann_cost=ANNUALIZATION * total_cost / h_days,
        avg_turnover=avg_to,
        annual_turnover=ANNUALIZATION * math.fsum(later.tolist()) / h_days,
        n_trades=int(np.sum(executed)),
        ann_return=annualized_return(net),
        ann_vol=annualized_vol(net),
        sharpe=sharpe_ratio(net, rfv),
        gross_sharpe=sharpe_ratio(gross, rfv),
        max_drawdown=max_drawdown(net),
    )
