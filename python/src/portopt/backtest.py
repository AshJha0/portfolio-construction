"""Backtest-lite: quarterly expanding-window rebalance loop and metrics.

Pinned schedule (API_SPEC section 6): with T rows of daily returns,
estimation window W (default 500) and rebalance step of 63 trading days,
rebalance dates are the row indices ``t_k = W + 63 k`` for
``k = 0, 1, ...`` while ``t_k < T``.  At each date the weight function
sees the expanding history ``rows [0, t_k)`` and the resulting weights
are applied to rows ``[t_k, min(t_{k+1}, T))``.  Within a holding block
the portfolio is rebalanced back to the target weights daily, so the
daily portfolio return is exactly ``w' r_t``.

Turnover at rebalance k >= 1 (pinned): ``sum_i |w_k[i] - w_{k-1}[i]|``
(two-sided L1 turnover between consecutive target weight vectors; no
turnover is charged for the initial position).

Metric formulas (pinned):

* ``ann_return = 252 * mean(r)`` (arithmetic)
* ``ann_vol    = sqrt(252) * std(r, ddof=1)``
* ``sharpe     = (ann_return - rf) / ann_vol`` (error if ann_vol == 0)
* max drawdown from wealth ``V_0 = 1, V_t = V_{t-1} (1 + r_t)``:
  ``maxdd = max_t (1 - V_t / max_{s <= t} V_s)`` (peak includes V_0);
  reported as a positive fraction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np

from .estimators import validate_returns

__all__ = [
    "annualized_return",
    "annualized_vol",
    "sharpe_ratio",
    "max_drawdown",
    "rebalance_indices",
    "BacktestResult",
    "run_backtest",
]

DEFAULT_WINDOW = 500
DEFAULT_STEP = 63
ANNUALIZATION = 252


def _validate_series(r: np.ndarray, min_len: int = 1) -> np.ndarray:
    x = np.asarray(r, dtype=float).reshape(-1)
    if x.shape[0] < min_len:
        raise ValueError(f"return series must have at least {min_len} observations")
    if not np.isfinite(x).all():
        raise ValueError("return series contains NaN or infinite entries")
    return x


def annualized_return(r: np.ndarray, periods: int = ANNUALIZATION) -> float:
    """Arithmetic annualized return ``periods * mean(r)``."""
    x = _validate_series(r, 1)
    return periods * float(np.sum(x)) / x.shape[0]


def annualized_vol(r: np.ndarray, periods: int = ANNUALIZATION) -> float:
    """Annualized volatility ``sqrt(periods) * std(r, ddof=1)``."""
    x = _validate_series(r, 2)
    mu = float(np.sum(x)) / x.shape[0]
    var = float(np.sum((x - mu) ** 2)) / (x.shape[0] - 1)
    return math.sqrt(periods) * math.sqrt(var)


def sharpe_ratio(r: np.ndarray, rf: float = 0.0, periods: int = ANNUALIZATION) -> float:
    """Annualized Sharpe ``(ann_return - rf) / ann_vol``; error if vol is ~0."""
    vol = annualized_vol(r, periods)
    if vol < 1e-15:
        raise ValueError("Sharpe undefined: portfolio volatility is zero")
    return (annualized_return(r, periods) - float(rf)) / vol


def max_drawdown(r: np.ndarray) -> float:
    """Maximum drawdown of the compounded wealth path (positive fraction)."""
    x = _validate_series(r, 1)
    wealth = 1.0
    peak = 1.0
    mdd = 0.0
    for ret in x:
        wealth *= 1.0 + ret
        if wealth > peak:
            peak = wealth
        dd = 1.0 - wealth / peak
        if dd > mdd:
            mdd = dd
    return mdd


def rebalance_indices(t: int, window: int = DEFAULT_WINDOW, step: int = DEFAULT_STEP) -> List[int]:
    """Pinned rebalance dates ``window, window+step, ...`` strictly below t."""
    if window < 2:
        raise ValueError("estimation window must be >= 2")
    if step < 1:
        raise ValueError("rebalance step must be >= 1")
    if t <= window:
        raise ValueError(
            f"sample length {t} must exceed the estimation window {window}"
        )
    return list(range(window, t, step))


@dataclass
class BacktestResult:
    """Result of :func:`run_backtest`.

    Attributes
    ----------
    portfolio_returns:
        Daily portfolio returns over the holdout ``rows [window, T)``.
    rebalance_dates:
        Row indices at which weights were recomputed.
    weights:
        Target weights per rebalance (K x n).
    turnovers:
        L1 turnover at each rebalance after the first (length K-1).
    ann_return, ann_vol, sharpe, max_drawdown, avg_turnover:
        Summary metrics (see module docstring).
    """

    portfolio_returns: np.ndarray
    rebalance_dates: List[int]
    weights: np.ndarray
    turnovers: np.ndarray
    ann_return: float
    ann_vol: float
    sharpe: float
    max_drawdown: float
    avg_turnover: float = field(default=0.0)


def run_backtest(
    returns: np.ndarray,
    weight_fn: Callable[[np.ndarray], np.ndarray],
    rf: float = 0.0,
    window: int = DEFAULT_WINDOW,
    step: int = DEFAULT_STEP,
) -> BacktestResult:
    """Quarterly expanding-window backtest with the pinned schedule.

    ``weight_fn(history)`` receives the expanding daily-return history
    (rows ``[0, t_k)``) and must return weights of length n summing to ~1
    (validated to 1e-6).
    """
    x = validate_returns(returns, min_rows=3)
    t, n = x.shape
    dates = rebalance_indices(t, window, step)
    k = len(dates)
    weights = np.zeros((k, n), dtype=float)
    port = np.zeros(t - window, dtype=float)
    for idx, t_k in enumerate(dates):
        w = np.asarray(weight_fn(x[:t_k]), dtype=float).reshape(-1)
        if w.shape[0] != n or not np.isfinite(w).all():
            raise ValueError(f"weight_fn returned an invalid weight vector at t={t_k}")
        if abs(float(np.sum(w)) - 1.0) > 1e-6:
            raise ValueError(
                f"weight_fn weights must sum to 1 (got {float(np.sum(w)):.8f}) at t={t_k}"
            )
        weights[idx] = w
        t_end = dates[idx + 1] if idx + 1 < k else t
        block = x[t_k:t_end] @ w
        port[t_k - window : t_end - window] = block
    turnovers = np.array(
        [float(np.sum(np.abs(weights[i] - weights[i - 1]))) for i in range(1, k)],
        dtype=float,
    )
    avg_to = float(np.mean(turnovers)) if turnovers.size else 0.0
    return BacktestResult(
        portfolio_returns=port,
        rebalance_dates=dates,
        weights=weights,
        turnovers=turnovers,
        ann_return=annualized_return(port),
        ann_vol=annualized_vol(port),
        sharpe=sharpe_ratio(port, rf),
        max_drawdown=max_drawdown(port),
        avg_turnover=avg_to,
    )
