"""Weight functions for :func:`portopt.research.run_backtest_net`.

Part of the Python-only research layer (not covered by API_SPEC.md).
Every rule is built only from public :mod:`portopt` functions.  Each
factory returns a :class:`WeightFunction`: a callable mapping the
read-only daily-return history (rows ``[0, t)``) to weights summing to
one, which also records the history lengths at which it had to fall back
to a simpler rule.

Moments.  Expected returns are the annualized sample mean
(``annualize_mean(sample_mean(h))``); the covariance is the annualized
sample covariance, or with ``shrink=True`` the annualized Ledoit-Wolf
(2004) constant-correlation estimate ``shrink_covariance(h).sigma``.

Leverage cap.  For rules that can be leveraged (tangency, Black-Litterman)
``max_gross`` caps ``sum_i |w_i|`` by moving along the straight line from
the rule's weights ``w`` toward a low-leverage anchor ``a``: the result is
``a + alpha (w - a)`` with the largest ``alpha`` in [0, 1] whose gross
leverage is within the cap.  Gross leverage is convex in ``alpha``, so
the feasible set is an interval containing 0 when the anchor itself is
within the cap, and bisection finds its end.  Weights still sum to one.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

import portopt
from portopt import (
    TangencyUndefinedError,
    annualize_cov,
    annualize_mean,
    bl_weights,
    constrained_frontier,
    erc_weights,
    gross_leverage,
    implied_equilibrium_returns,
    inverse_vol_weights,
    min_variance_weights,
    sample_cov,
    sample_mean,
    shrink_covariance,
    tangency_weights,
)

__all__ = [
    "WeightFunction",
    "black_litterman",
    "equal_weight",
    "erc",
    "inverse_vol",
    "max_sharpe_longonly",
    "min_variance_longonly",
    "tangency_capped",
]

_BISECT_STEPS = 200


class WeightFunction:
    """A named weight rule ``history -> weights`` with a fallback log.

    ``fallback_dates`` lists ``history.shape[0]`` for every call in which
    the rule fell back (see each factory).  It is the only state the
    object holds; the weights themselves are a pure function of the
    history.
    """

    def __init__(self, name: str, fn: Callable[[np.ndarray, list[int]], np.ndarray]) -> None:
        self.name = name
        self._fn = fn
        self.fallback_dates: list[int] = []

    def __call__(self, history: np.ndarray) -> np.ndarray:
        h = np.asarray(history, dtype=float)
        if h.ndim != 2 or h.shape[0] < 2:
            raise ValueError("history must be a T x n matrix with T >= 2")
        return self._fn(h, self.fallback_dates)

    def __repr__(self) -> str:
        return f"WeightFunction({self.name!r})"


def _cov(h: np.ndarray, shrink: bool) -> np.ndarray:
    s = shrink_covariance(h).sigma if shrink else sample_cov(h)
    return annualize_cov(s)


def _mu(h: np.ndarray) -> np.ndarray:
    return annualize_mean(sample_mean(h))


def _suffix(shrink: bool) -> str:
    return "_lw" if shrink else ""


def _check_rf(rf: float) -> float:
    v = float(rf)
    if not math.isfinite(v):
        raise ValueError("rf must be finite")
    return v


def _check_cap(max_gross: float | None) -> float | None:
    if max_gross is None:
        return None
    v = float(max_gross)
    if not (math.isfinite(v) and v >= 1.0):
        raise ValueError(f"max_gross must be None or finite and >= 1, got {max_gross!r}")
    return v


def cap_toward(w: np.ndarray, anchor: np.ndarray, cap: float) -> np.ndarray:
    """Largest step from ``anchor`` toward ``w`` with gross leverage <= cap.

    Requires ``gross_leverage(anchor) <= cap``.  Returns ``w`` unchanged
    when it already satisfies the cap.
    """
    if gross_leverage(w) <= cap:
        return w
    if gross_leverage(anchor) > cap:
        raise ValueError("cap_toward: the anchor itself exceeds the gross-leverage cap")
    lo, hi = 0.0, 1.0
    d = w - anchor
    for _ in range(_BISECT_STEPS):
        mid = 0.5 * (lo + hi)
        if gross_leverage(anchor + mid * d) <= cap:
            lo = mid
        else:
            hi = mid
    out: np.ndarray = anchor + lo * d
    return out


def equal_weight() -> WeightFunction:
    """``w_i = 1/n``; needs no estimates and never falls back."""

    def fn(h: np.ndarray, _log: list[int]) -> np.ndarray:
        return np.full(h.shape[1], 1.0 / h.shape[1])

    return WeightFunction("equal_weight", fn)


def inverse_vol(*, shrink: bool = False) -> WeightFunction:
    """Inverse-volatility weights (:func:`portopt.inverse_vol_weights`)."""

    def fn(h: np.ndarray, _log: list[int]) -> np.ndarray:
        return inverse_vol_weights(_cov(h, shrink))

    return WeightFunction("inverse_vol" + _suffix(shrink), fn)


def erc(*, shrink: bool = False) -> WeightFunction:
    """Equal-risk-contribution weights (:func:`portopt.erc_weights`)."""

    def fn(h: np.ndarray, _log: list[int]) -> np.ndarray:
        return erc_weights(_cov(h, shrink))

    return WeightFunction("erc" + _suffix(shrink), fn)


def min_variance_longonly(*, shrink: bool = False) -> WeightFunction:
    """Long-only minimum variance (:func:`portopt.min_variance_longonly`)."""

    def fn(h: np.ndarray, _log: list[int]) -> np.ndarray:
        return portopt.min_variance_longonly(_cov(h, shrink))

    return WeightFunction("min_var_lo" + _suffix(shrink), fn)


def max_sharpe_longonly(*, rf: float, shrink: bool = False, n_points: int = 25) -> WeightFunction:
    """Long-only maximum Sharpe, approximated on the constrained frontier.

    Sweeps :func:`portopt.constrained_frontier` (``n_points`` targets from
    the long-only minimum-variance return to the largest attainable
    return) and returns the grid point with the largest
    ``(mu'w - rf) / sqrt(w' Sigma w)`` (ties to the lower-return point).
    The answer is exact only up to the grid spacing.

    Falls back to long-only minimum variance (and logs it) when the grid
    is empty (the minimum-variance portfolio already has the largest
    attainable return) or when no grid point earns more than ``rf``, in
    which case maximizing the Sharpe ratio is not meaningful.
    """
    rfv = _check_rf(rf)
    if isinstance(n_points, bool) or int(n_points) != n_points or int(n_points) < 2:
        raise ValueError("n_points must be an integer >= 2")
    npts = int(n_points)

    def fn(h: np.ndarray, log: list[int]) -> np.ndarray:
        mu = _mu(h)
        sig = _cov(h, shrink)
        try:
            targets, weights, variances = constrained_frontier(mu, sig, npts)
        except ValueError as exc:
            if "frontier grid empty" not in str(exc):
                raise
            log.append(h.shape[0])
            return portopt.min_variance_longonly(sig)
        excess = targets - rfv
        if not np.any(excess > 0.0):
            log.append(h.shape[0])
            return portopt.min_variance_longonly(sig)
        sharpe = excess / np.sqrt(variances)
        best = int(np.argmax(sharpe))
        out: np.ndarray = weights[best].copy()
        return out

    return WeightFunction("max_sharpe_lo" + _suffix(shrink), fn)


def tangency_capped(
    *, rf: float, max_gross: float | None = None, shrink: bool = False
) -> WeightFunction:
    """Unconstrained tangency weights, optionally capped in gross leverage.

    When :func:`portopt.tangency_weights` raises
    :class:`portopt.TangencyUndefinedError` the rule falls back to the
    unconstrained minimum-variance portfolio (logged), as the core demo
    does.  With ``max_gross`` set, the weights are moved toward the
    unconstrained minimum-variance portfolio, or toward the long-only
    minimum-variance portfolio when the former is itself above the cap
    (see the module docstring).
    """
    rfv = _check_rf(rf)
    cap = _check_cap(max_gross)

    def fn(h: np.ndarray, log: list[int]) -> np.ndarray:
        mu = _mu(h)
        sig = _cov(h, shrink)
        gmv = min_variance_weights(sig)
        try:
            w = tangency_weights(mu, sig, rfv)
        except TangencyUndefinedError:
            log.append(h.shape[0])
            w = gmv
        if cap is None:
            return w
        anchor = gmv if gross_leverage(gmv) <= cap else portopt.min_variance_longonly(sig)
        return cap_toward(w, anchor, cap)

    name = "tangency" + (f"_cap{cap:g}" if cap is not None else "") + _suffix(shrink)
    return WeightFunction(name, fn)


def black_litterman(
    *,
    w_mkt: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    tau: float,
    delta: float,
    confidences: Sequence[float] | None = None,
    max_gross: float | None = None,
    shrink: bool = False,
) -> WeightFunction:
    """Black-Litterman posterior tangency weights (:func:`portopt.bl_weights`).

    At each date the equilibrium returns are re-implied from the history
    covariance, ``pi = delta Sigma w_mkt``, and the posterior tangency is
    taken with the pinned ``rf = 0`` (``pi`` are excess returns).  Falls
    back to ``w_mkt`` (logged) when the posterior tangency is undefined.
    With ``max_gross`` set, weights are moved toward ``w_mkt``.
    """
    wm = np.asarray(w_mkt, dtype=float).reshape(-1)
    if wm.size == 0 or not np.isfinite(wm).all() or np.any(wm < 0) or not np.sum(wm) > 0:
        raise ValueError("w_mkt must be a non-empty, finite, non-negative vector with positive sum")
    wm = wm / float(np.sum(wm))
    cap = _check_cap(max_gross)
    pm = np.asarray(p, dtype=float)
    qv = np.asarray(q, dtype=float)
    conf = None if confidences is None else [float(c) for c in confidences]

    def fn(h: np.ndarray, log: list[int]) -> np.ndarray:
        if h.shape[1] != wm.shape[0]:
            raise ValueError(
                f"w_mkt length {wm.shape[0]} does not match the {h.shape[1]} assets in history"
            )
        sig = _cov(h, shrink)
        pi = implied_equilibrium_returns(sig, wm, delta)
        try:
            w, _ = bl_weights(sig, pi, pm, qv, tau, confidences=conf)
        except TangencyUndefinedError:
            log.append(h.shape[0])
            return wm.copy()
        if cap is None:
            return w
        return cap_toward(w, wm, cap)

    name = "black_litterman" + (f"_cap{cap:g}" if cap is not None else "") + _suffix(shrink)
    return WeightFunction(name, fn)
