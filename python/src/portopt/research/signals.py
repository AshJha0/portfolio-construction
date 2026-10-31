"""Return signals and the weight rules that use them.

Every function here maps a daily return window ``returns`` (T x n, the
rows strictly before the decision date, oldest first) to one value per
asset.  None of them sees a date: a "month" is :data:`MONTH` = 21 trading
days and a "year" :data:`YEAR` = 252, counted back from the last row of
the window.  At a monthly rebalance on the first trading day of a month
the last 21 rows are close to, but not exactly, the previous calendar
month; the approximation is stated, not hidden, and it lets the same
functions run inside ``forward/run_forward.py``, which passes only a
trailing window of total returns.

Signals (sign convention: higher means a higher expected return).

* :func:`momentum_12_1`: compound return over months t-12..t-2, skipping
  the most recent month (Jegadeesh and Titman 1993; for industries,
  Moskowitz and Grinblatt 1999).
* :func:`tsmom`: sign of the trailing 12-month excess return divided by
  the asset's trailing volatility (Moskowitz, Ooi and Pedersen 2012).
* :func:`short_term_reversal`: minus the last month's return (Jegadeesh
  1990; Lehmann 1990).
* :func:`low_volatility`: minus trailing 12-month volatility (Baker,
  Bradley and Wurgler 2011).
* :func:`low_beta`: minus the trailing 12-month beta against a market
  series, by default the equal-weighted mean of the columns (Frazzini and
  Pedersen 2014).

:func:`zscore` standardizes a signal across assets and :func:`combine`
averages z-scores with equal weights or with weights inversely
proportional to each signal's trailing long-short return volatility.  No
weight is fitted to returns.

:func:`signal_to_expected_returns` turns a z-score into an expected
return by Grinold and Kahn's rule ``alpha_i = IC sigma_i z_i``; the
information coefficient ``IC`` is an assumption the caller states.

Weight rules: :func:`rank_top_tercile` (long-only, equal weight on the
top third, the transparent baseline), :class:`SignalMeanVariance`
(long-only mean-variance with an L1 turnover penalty around the drifted
weights, :func:`portopt.markowitz.mean_variance_turnover_solve`) and
:func:`signal_black_litterman` (absolute views ``pi + alpha`` blended with
an equal-weight equilibrium, then long-only mean-variance on the
posterior).  Nothing here shows that a signal has an edge; the studies in
``python/studies/signal_study.py`` measure that, with costs and error
bars, on non-investable index data.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

from ..black_litterman import bl_posterior
from ..markowitz import mean_variance_turnover_solve
from .strategies import WeightFunction
from .strategies_v2 import CovarianceEstimator, lw_constant_corr_estimator

#: Trading days per "month" and per "year" in every window computation.
MONTH = 21
YEAR = 252

SignalFn = Callable[[np.ndarray], np.ndarray]


def _window(returns: np.ndarray, rows: int, name: str) -> np.ndarray:
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2 or x.shape[1] < 1:
        raise ValueError(f"{name}: returns must be a T x n matrix")
    if x.shape[0] < rows:
        raise ValueError(f"{name}: needs at least {rows} rows, got {x.shape[0]}")
    w = x[-rows:]
    if not np.isfinite(w).all():
        raise ValueError(f"{name}: returns must be finite in the last {rows} rows")
    return w


def _compound(x: np.ndarray) -> np.ndarray:
    out: np.ndarray = np.prod(1.0 + x, axis=0) - 1.0
    return out


def _rf_window(rf_daily: float | np.ndarray, rows: int, t: int) -> np.ndarray:
    rf = np.asarray(rf_daily, dtype=float)
    if rf.ndim == 0:
        return np.full(rows, float(rf))
    if rf.shape != (t,):
        raise ValueError(f"rf_daily must be a scalar or a vector of length {t}")
    return rf[-rows:]


# ------------------------------------------------------------------ signals


def momentum_12_1(returns: np.ndarray) -> np.ndarray:
    """Compound return over rows ``[-252, -21)``: months t-12..t-2."""
    w = _window(returns, YEAR, "momentum_12_1")
    return _compound(w[:-MONTH])


def tsmom(returns: np.ndarray, rf_daily: float | np.ndarray = 0.0) -> np.ndarray:
    """``sign(12-month excess return) / annualized 12-month volatility``.

    ``rf_daily`` is a scalar or one value per row of ``returns``.  The
    forward runner passes total returns and no rate, so there the sign is
    that of the total return (``rf_daily = 0``).
    """
    w = _window(returns, YEAR, "tsmom")
    rf = _rf_window(rf_daily, YEAR, np.asarray(returns).shape[0])
    ex = float(np.prod(1.0 + rf)) - 1.0
    vol = np.std(w, axis=0, ddof=1) * math.sqrt(YEAR)
    if np.any(vol <= 0.0):
        raise ValueError("tsmom: zero volatility in the window")
    out: np.ndarray = np.sign(_compound(w) - ex) / vol
    return out


def short_term_reversal(returns: np.ndarray) -> np.ndarray:
    """Minus the compound return of the last 21 rows."""
    w = _window(returns, MONTH, "short_term_reversal")
    return -_compound(w)


def low_volatility(returns: np.ndarray) -> np.ndarray:
    """Minus the annualized volatility of the last 252 rows."""
    w = _window(returns, YEAR, "low_volatility")
    out: np.ndarray = -np.std(w, axis=0, ddof=1) * math.sqrt(YEAR)
    return out


def low_beta(returns: np.ndarray, market: np.ndarray | None = None) -> np.ndarray:
    """Minus the OLS beta of each asset on ``market`` over the last 252 rows.

    ``market`` (one value per row of ``returns``) defaults to the
    equal-weighted mean of the columns.
    """
    w = _window(returns, YEAR, "low_beta")
    if market is None:
        m = w.mean(axis=1)
    else:
        mk = np.asarray(market, dtype=float).reshape(-1)
        if mk.shape[0] != np.asarray(returns).shape[0]:
            raise ValueError("low_beta: market must have one value per row of returns")
        m = mk[-YEAR:]
    mc = m - m.mean()
    var = float(mc @ mc)
    if not var > 0.0:
        raise ValueError("low_beta: zero market variance in the window")
    beta = (w - w.mean(axis=0)).T @ mc / var
    out: np.ndarray = -beta
    return out


SIGNALS: dict[str, SignalFn] = {
    "mom12_1": momentum_12_1,
    "tsmom": tsmom,
    "strev": short_term_reversal,
    "lowvol": low_volatility,
    "lowbeta": low_beta,
}
#: Rows of history each signal reads.
SIGNAL_ROWS = {"mom12_1": YEAR, "tsmom": YEAR, "strev": MONTH, "lowvol": YEAR, "lowbeta": YEAR}


def zscore(s: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score (population standard deviation); zeros when
    the signal does not vary across assets."""
    v = np.asarray(s, dtype=float).reshape(-1)
    if not np.isfinite(v).all():
        raise ValueError("zscore: signal must be finite")
    sd = float(np.std(v))
    if not sd > 1e-15 * max(1.0, float(np.max(np.abs(v)))):
        return np.zeros_like(v)
    out: np.ndarray = (v - v.mean()) / sd
    return out


def _signal(name: str, returns: np.ndarray, rf_daily: float | np.ndarray) -> np.ndarray:
    if name not in SIGNALS:
        raise ValueError(f"unknown signal {name!r}; one of {sorted(SIGNALS)}")
    if name == "tsmom":
        return tsmom(returns, rf_daily)
    return SIGNALS[name](returns)


def factor_returns(
    name: str, returns: np.ndarray, rf_daily: float | np.ndarray = 0.0, months: int = 12
) -> np.ndarray:
    """Daily returns of the z-score-weighted long-short portfolio of one
    signal over the last ``months * 21`` rows of ``returns``.

    The weights for month ``k`` are ``z / sum|z|`` of the signal computed
    on the rows before that month starts, so each month's return uses only
    earlier rows.  Needs ``months * 21 + SIGNAL_ROWS[name]`` rows.
    """
    x = np.asarray(returns, dtype=float)
    span = months * MONTH
    need = span + SIGNAL_ROWS[name]
    _window(x, need, f"factor_returns({name})")
    t = x.shape[0]
    rf = np.asarray(rf_daily, dtype=float)
    out = np.empty(span)
    for k in range(months):
        start = t - span + k * MONTH
        rf_k: float | np.ndarray = rf if rf.ndim == 0 else rf[:start]
        z = zscore(_signal(name, x[:start], rf_k))
        gross = float(np.sum(np.abs(z)))
        wk = z / gross if gross > 0.0 else z
        out[k * MONTH : (k + 1) * MONTH] = x[start : start + MONTH] @ wk
    return out


COMBINE_METHODS = ("equal", "inverse_vol")


def combine(
    names: Sequence[str],
    returns: np.ndarray,
    *,
    method: str = "equal",
    rf_daily: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Average of the cross-sectional z-scores of ``names``, re-z-scored.

    ``method="equal"`` weights each signal equally; ``"inverse_vol"``
    weights it by ``1 / vol`` of its :func:`factor_returns` over the last
    12 months (a risk weight, not a fitted one).  With one name the result
    is that signal's z-score.
    """
    if not names:
        raise ValueError("combine: at least one signal")
    if method not in COMBINE_METHODS:
        raise ValueError(f"combine: method must be one of {COMBINE_METHODS}")
    zs = np.array([zscore(_signal(n, returns, rf_daily)) for n in names])
    if method == "equal" or len(names) == 1:
        wts = np.full(len(names), 1.0 / len(names))
    else:
        vols = np.array([float(np.std(factor_returns(n, returns, rf_daily))) for n in names])
        if np.any(vols <= 0.0):
            raise ValueError("combine: a signal's factor return has zero volatility")
        wts = (1.0 / vols) / float(np.sum(1.0 / vols))
    return zscore(wts @ zs)


def combine_rows(names: Sequence[str], method: str) -> int:
    """Rows of history :func:`combine` reads."""
    base = max(SIGNAL_ROWS[n] for n in names)
    if method == "inverse_vol" and len(names) > 1:
        return max(base, 12 * MONTH + base)
    return base


def signal_to_expected_returns(z: np.ndarray, sigma: np.ndarray, ic: float) -> np.ndarray:
    """Grinold and Kahn's forecast rule ``alpha_i = IC * sigma_i * z_i``.

    ``sigma`` is each asset's volatility over the forecast horizon, so
    ``alpha`` is in the same horizon's return units.  ``ic`` is an assumed
    information coefficient (correlation of forecast and outcome), not an
    estimate.
    """
    zv = np.asarray(z, dtype=float).reshape(-1)
    sv = np.asarray(sigma, dtype=float).reshape(-1)
    if zv.shape != sv.shape or not (np.isfinite(zv).all() and np.isfinite(sv).all()):
        raise ValueError("z and sigma must be finite vectors of the same length")
    if np.any(sv < 0.0):
        raise ValueError("sigma must be non-negative")
    icv = float(ic)
    if not math.isfinite(icv) or abs(icv) > 1.0:
        raise ValueError("ic must lie in [-1, 1]")
    out: np.ndarray = icv * sv * zv
    return out


# ------------------------------------------------------------ weight rules


def _check_names(signals: Sequence[str], method: str) -> tuple[str, ...]:
    names = tuple(signals)
    if not names or any(n not in SIGNALS for n in names):
        raise ValueError(f"signals must be a non-empty list from {sorted(SIGNALS)}")
    if method not in COMBINE_METHODS:
        raise ValueError(f"method must be one of {COMBINE_METHODS}")
    return names


def top_fraction_weights(score: np.ndarray, fraction: float = 1.0 / 3.0) -> np.ndarray:
    """Equal weights on the ``max(1, floor(n * fraction))`` highest scores.

    Ties at the cut are broken by column order (lower index first), so the
    result is deterministic.
    """
    s = np.asarray(score, dtype=float).reshape(-1)
    n = s.shape[0]
    k = max(1, int(math.floor(n * fraction + 1e-12)))
    order = np.lexsort((np.arange(n), -s))
    w = np.zeros(n)
    w[order[:k]] = 1.0 / k
    return w


def rank_top_tercile(
    *, signals: Sequence[str], method: str = "equal", rf: float = 0.0
) -> WeightFunction:
    """Long-only, equal weight on the top third of the combined score.

    ``rf`` is a constant daily rate for :func:`tsmom` (the forward runner
    has no rate series; the studies pass excess returns instead).
    """
    names = _check_names(signals, method)
    rows = combine_rows(names, method)
    rfv = float(rf)

    def fn(h: np.ndarray, log: list[int]) -> np.ndarray:
        if h.shape[0] < rows:
            raise ValueError(f"rank_top_tercile needs {rows} rows of history")
        return top_fraction_weights(combine(names, h, method=method, rf_daily=rfv))

    return WeightFunction(f"rank_top_tercile[{'+'.join(names)}|{method}]", fn)


def _alpha_annual(score: np.ndarray, sigma_annual: np.ndarray, ic: float) -> np.ndarray:
    """Annualized ``alpha`` for a one-month horizon: ``12 IC sigma_month z``."""
    sig_m = np.sqrt(np.diag(sigma_annual) / 12.0)
    return 12.0 * signal_to_expected_returns(score, sig_m, ic)


class SignalMeanVariance:
    """Long-only mean-variance with an L1 turnover penalty::

        max  alpha'w - (gamma/2) w'Sw - gamma * kappa * |w - w0|_1

    ``alpha`` is :func:`signal_to_expected_returns` of the combined score
    (one-month horizon, annualized), ``S`` the annualized covariance of
    the last ``lookback`` rows, ``kappa = 12 * cost / gamma`` (a monthly
    holding period, as in API_SPEC 3.7), solved by
    :func:`portopt.markowitz.mean_variance_turnover_solve`.

    ``w0`` is the previous target drifted by the asset returns since then
    and renormalized to sum to one; the first call starts from cash
    (``w0 = 0``).  The drift is read from the history passed in, which
    therefore must grow from call to call (as in
    :func:`portopt.research.walkforward.run_schedule_net`); a call with a
    history no longer than the previous one raises ValueError.  The rule
    is path dependent and cannot run in the forward runner, which passes a
    fixed-length window.
    """

    def __init__(
        self,
        *,
        signals: Sequence[str],
        method: str = "equal",
        ic: float = 0.05,
        risk_aversion: float = 5.0,
        cost_bps: float = 10.0,
        lookback: int = 504,
        estimator: CovarianceEstimator | None = None,
        rf_daily: np.ndarray | None = None,
    ) -> None:
        self.names = _check_names(signals, method)
        self.method = method
        self.ic = float(ic)
        self.gamma = float(risk_aversion)
        if not self.gamma > 0.0:
            raise ValueError("risk_aversion must be > 0")
        self.kappa = 12.0 * float(cost_bps) / 1.0e4 / self.gamma
        self.lookback = int(lookback)
        self.est = lw_constant_corr_estimator() if estimator is None else estimator
        self.rf = rf_daily
        self.targets: list[np.ndarray] = []
        self.last_t = -1
        self.n_unconverged = 0

    def __call__(self, history: np.ndarray) -> np.ndarray:
        h = np.asarray(history, dtype=float)
        t, n = h.shape
        if t <= self.last_t:
            raise ValueError("SignalMeanVariance: history must grow between calls")
        need = max(self.lookback, combine_rows(self.names, self.method))
        if t < need:
            raise ValueError(f"SignalMeanVariance needs {need} rows of history")
        if not self.targets:
            w0 = np.zeros(n)
        else:
            v = self.targets[-1] * np.prod(1.0 + h[self.last_t : t], axis=0)
            w0 = v / math.fsum(v.tolist())
        rf: float | np.ndarray = 0.0 if self.rf is None else self.rf[:t]
        score = combine(self.names, h, method=self.method, rf_daily=rf)
        sig = self.est(h[t - self.lookback : t])
        alpha = _alpha_annual(score, sig, self.ic)
        res = mean_variance_turnover_solve(alpha, sig, w0, self.gamma, self.kappa)
        if not res.converged:
            self.n_unconverged += 1
        raw = np.clip(np.asarray(res.weights, dtype=float), 0.0, None)
        w: np.ndarray = raw / math.fsum(raw.tolist())
        self.targets.append(w)
        self.last_t = t
        return w


def bl_signal_weights(
    sigma: np.ndarray,
    score: np.ndarray,
    *,
    ic: float,
    delta: float,
    tau: float,
    confidence: float,
) -> np.ndarray:
    """Black-Litterman with one absolute view per asset, then long-only
    mean-variance on the posterior.

    Equilibrium ``pi = delta S w_eq`` with ``w_eq = 1/n`` (the index data
    carry no market capitalizations).  Views ``P = I``,
    ``q = pi + alpha`` with ``alpha`` from :func:`_alpha_annual`; the view
    covariance is the He-Litterman default divided by ``confidence``.
    Weights solve ``max mu_post'w - (delta/2) w' S_post w`` over the
    long-only budget simplex.
    """
    s = np.asarray(sigma, dtype=float)
    n = s.shape[0]
    w_eq = np.full(n, 1.0 / n)
    pi = float(delta) * s @ w_eq
    alpha = _alpha_annual(score, s, ic)
    post = bl_posterior(s, pi, np.eye(n), pi + alpha, tau, confidences=[confidence] * n)
    res = mean_variance_turnover_solve(post.mu_post, post.sigma_post, w_eq, float(delta), 0.0)
    w = np.clip(np.asarray(res.weights, dtype=float), 0.0, None)
    out: np.ndarray = w / math.fsum(w.tolist())
    return out


def signal_black_litterman(
    *,
    signals: Sequence[str],
    method: str = "equal",
    ic: float = 0.05,
    delta: float = 2.5,
    tau: float = 0.05,
    confidence: float = 1.0,
    lookback: int = 504,
    rf: float = 0.0,
) -> WeightFunction:
    """Stateless :func:`bl_signal_weights` on the LW constant-correlation
    covariance of the last ``lookback`` rows."""
    names = _check_names(signals, method)
    rows = max(int(lookback), combine_rows(names, method))
    est = lw_constant_corr_estimator()
    rfv = float(rf)
    for name, v in (("delta", delta), ("tau", tau), ("confidence", confidence)):
        if not (math.isfinite(float(v)) and float(v) > 0.0):
            raise ValueError(f"{name} must be finite and > 0")

    def fn(h: np.ndarray, log: list[int]) -> np.ndarray:
        if h.shape[0] < rows:
            raise ValueError(f"signal_black_litterman needs {rows} rows of history")
        score = combine(names, h, method=method, rf_daily=rfv)
        sig = est(h[-int(lookback) :])
        return bl_signal_weights(sig, score, ic=ic, delta=delta, tau=tau, confidence=confidence)

    return WeightFunction(f"signal_black_litterman[{'+'.join(names)}|{method}]", fn)


__all__ = [
    "COMBINE_METHODS",
    "MONTH",
    "SIGNALS",
    "SIGNAL_ROWS",
    "YEAR",
    "SignalMeanVariance",
    "bl_signal_weights",
    "combine",
    "combine_rows",
    "factor_returns",
    "low_beta",
    "low_volatility",
    "momentum_12_1",
    "rank_top_tercile",
    "short_term_reversal",
    "signal_black_litterman",
    "signal_to_expected_returns",
    "top_fraction_weights",
    "tsmom",
    "zscore",
]
