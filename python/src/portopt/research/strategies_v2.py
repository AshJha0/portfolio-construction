"""Weight functions built on the estimators of :mod:`portopt.research.estimators`.

Part of the Python-only research layer (not covered by API_SPEC.md).  The
conventions follow :mod:`portopt.research.strategies`: each factory
returns a :class:`~portopt.research.strategies.WeightFunction` that maps
the read-only daily-return history (rows ``[0, t)``) to weights and logs
the history lengths at which it fell back to a simpler rule.

Covariance estimators.  A :class:`CovarianceEstimator` maps a daily
history to an *annualized* covariance (the per-period estimate times
252, :func:`portopt.annualize_cov`).  The factories below wrap every
estimator in the research layer plus the two core ones:

==================================  =========================================
factory                             estimator
==================================  =========================================
:func:`sample_estimator`            :func:`portopt.sample_cov` (ddof=1)
:func:`lw_constant_corr_estimator`  :func:`portopt.shrink_covariance`
:func:`lw_identity_estimator`       :func:`~.estimators.ledoit_wolf_identity`
:func:`nonlinear_estimator`         :func:`~.estimators.nonlinear_shrinkage`
:func:`ewma_estimator`              :func:`~.estimators.ewma_covariance`
:func:`ewma_cc_estimator`           :func:`~.estimators.ewma_constant_correlation`
:func:`pca_estimator`               :func:`~.estimators.pca_factor_covariance`
==================================  =========================================

Volatility targeting.  :func:`run_backtest_net` already has a
``vol_target`` argument, and its cash accounting is the one to use.  It
scales the raw target by ``vol_target / sqrt(u' Sigma u)`` with Sigma
fixed to the annualized *sample* covariance, and it requires weight
functions to return weights summing to one.  :func:`vol_target` here
makes the same scaling with any :class:`CovarianceEstimator` and returns
the scaled *risky* weights, which sum to the scale ``s`` (cash is
``1 - s``).  Those weights do not satisfy the sum-to-one check of the
current ``run_backtest_net``, so to backtest them the backtest needs a
covariance-estimator hook for its own ``vol_target`` step.  With the
sample estimator, use ``run_backtest_net(..., vol_target=...)`` directly.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

import portopt
from portopt import (
    annualize_cov,
    annualize_mean,
    constrained_frontier,
    erc_weights,
    sample_cov,
    shrink_covariance,
)

from .estimators import (
    bayes_stein_mean,
    ewma_constant_correlation,
    ewma_covariance,
    ewma_decay,
    ledoit_wolf_identity,
    nonlinear_shrinkage,
    pca_factor_covariance,
)
from .strategies import WeightFunction

__all__ = [
    "CovarianceEstimator",
    "VolTargetedWeights",
    "erc_with",
    "ewma_cc_estimator",
    "ewma_estimator",
    "lw_constant_corr_estimator",
    "lw_identity_estimator",
    "max_sharpe_bayes_stein",
    "min_variance_longonly_with",
    "nonlinear_estimator",
    "pca_estimator",
    "sample_estimator",
    "vol_target",
]


class CovarianceEstimator:
    """A named map ``daily history -> annualized covariance``.

    The wrapped function returns a per-period covariance; this object
    annualizes it by 252.  It holds no state.
    """

    def __init__(self, name: str, fn: Callable[[np.ndarray], np.ndarray]) -> None:
        self.name = name
        self._fn = fn

    def __call__(self, history: np.ndarray) -> np.ndarray:
        h = np.asarray(history, dtype=float)
        if h.ndim != 2 or h.shape[0] < 2:
            raise ValueError("history must be a T x n matrix with T >= 2")
        return annualize_cov(self._fn(h))

    def __repr__(self) -> str:
        return f"CovarianceEstimator({self.name!r})"


def _hl_tag(half_life: float) -> str:
    ewma_decay(half_life)  # validates
    return "inf" if math.isinf(float(half_life)) else f"{float(half_life):g}"


def sample_estimator() -> CovarianceEstimator:
    """ddof=1 sample covariance (:func:`portopt.sample_cov`)."""
    return CovarianceEstimator("sample", sample_cov)


def lw_constant_corr_estimator() -> CovarianceEstimator:
    """Ledoit-Wolf (2004, JPM) constant-correlation shrinkage (core)."""
    return CovarianceEstimator("lwcc", lambda h: shrink_covariance(h).sigma)


def lw_identity_estimator() -> CovarianceEstimator:
    """Ledoit-Wolf (2004, JMVA) linear shrinkage toward ``m I``."""
    return CovarianceEstimator("lwid", lambda h: ledoit_wolf_identity(h).sigma)


def nonlinear_estimator() -> CovarianceEstimator:
    """Ledoit-Wolf (2020) analytical nonlinear shrinkage."""
    return CovarianceEstimator("nls", lambda h: nonlinear_shrinkage(h).sigma)


def ewma_estimator(half_life: float, *, demean: bool = True) -> CovarianceEstimator:
    """EWMA covariance with the given half-life in rows (days)."""
    tag = _hl_tag(half_life)
    hl = float(half_life)
    return CovarianceEstimator(f"ewma{tag}", lambda h: ewma_covariance(h, hl, demean=demean))


def ewma_cc_estimator(half_life: float, *, demean: bool = True) -> CovarianceEstimator:
    """EWMA volatilities with a constant sample correlation."""
    tag = _hl_tag(half_life)
    hl = float(half_life)
    return CovarianceEstimator(
        f"ewmacc{tag}", lambda h: ewma_constant_correlation(h, hl, demean=demean)
    )


def pca_estimator(k: int, *, specific_floor: float = 1e-4) -> CovarianceEstimator:
    """k-factor principal-component model ``B F B' + D``.

    ``k`` and ``specific_floor`` are validated on first use against the
    history's number of assets; ``k`` must be an integer >= 1 here.
    """
    if isinstance(k, bool) or int(k) != k or int(k) < 1:
        raise ValueError(f"k must be an integer >= 1, got {k!r}")
    fl = float(specific_floor)
    if not (math.isfinite(fl) and 0.0 < fl < 1.0):
        raise ValueError(f"specific_floor must lie in (0, 1), got {specific_floor!r}")
    kk = int(k)
    return CovarianceEstimator(
        f"pca{kk}", lambda h: pca_factor_covariance(h, kk, specific_floor=fl).sigma
    )


def _check_estimator(estimator: CovarianceEstimator) -> CovarianceEstimator:
    if not isinstance(estimator, CovarianceEstimator):
        raise ValueError("estimator must be a CovarianceEstimator (see the *_estimator factories)")
    return estimator


def min_variance_longonly_with(estimator: CovarianceEstimator) -> WeightFunction:
    """Long-only minimum variance (:func:`portopt.min_variance_longonly`) on ``estimator``."""
    est = _check_estimator(estimator)

    def fn(h: np.ndarray, _log: list[int]) -> np.ndarray:
        return portopt.min_variance_longonly(est(h))

    return WeightFunction(f"min_var_lo_{est.name}", fn)


def erc_with(estimator: CovarianceEstimator) -> WeightFunction:
    """Equal-risk-contribution weights (:func:`portopt.erc_weights`) on ``estimator``."""
    est = _check_estimator(estimator)

    def fn(h: np.ndarray, _log: list[int]) -> np.ndarray:
        return erc_weights(est(h))

    return WeightFunction(f"erc_{est.name}", fn)


def max_sharpe_bayes_stein(
    *,
    rf: float,
    estimator: CovarianceEstimator | None = None,
    n_points: int = 25,
) -> WeightFunction:
    """Long-only maximum Sharpe with Bayes-Stein means.

    Expected returns are the annualized Bayes-Stein mean of Jorion (1986)
    (:func:`~.estimators.bayes_stein_mean`); the covariance comes from
    ``estimator`` (default: the sample covariance).  The optimisation is
    the frontier sweep of
    :func:`portopt.research.strategies.max_sharpe_longonly`: the grid
    point of :func:`portopt.constrained_frontier` with the largest
    ``(mu'w - rf) / sqrt(w' Sigma w)``, exact only up to the grid
    spacing.

    Falls back (logged) to long-only minimum variance on the same
    covariance when the history is too short for Bayes-Stein
    (``T <= N + 2``), when the frontier grid is empty, or when no grid
    point earns more than ``rf``.  Jorion's predictive covariance
    adjustment is not applied; pick the covariance through ``estimator``.
    """
    rfv = float(rf)
    if not math.isfinite(rfv):
        raise ValueError("rf must be finite")
    if isinstance(n_points, bool) or int(n_points) != n_points or int(n_points) < 2:
        raise ValueError("n_points must be an integer >= 2")
    npts = int(n_points)
    est = sample_estimator() if estimator is None else _check_estimator(estimator)

    def fn(h: np.ndarray, log: list[int]) -> np.ndarray:
        sig = est(h)
        t, n = h.shape
        if t <= n + 2:
            log.append(t)
            return portopt.min_variance_longonly(sig)
        mu = annualize_mean(bayes_stein_mean(h).mean)
        try:
            targets, weights, variances = constrained_frontier(mu, sig, npts)
        except ValueError as exc:
            if "frontier grid empty" not in str(exc):
                raise
            log.append(t)
            return portopt.min_variance_longonly(sig)
        excess = targets - rfv
        if not np.any(excess > 0.0):
            log.append(t)
            return portopt.min_variance_longonly(sig)
        sharpe = excess / np.sqrt(variances)
        out: np.ndarray = weights[int(np.argmax(sharpe))].copy()
        return out

    return WeightFunction(f"max_sharpe_lo_bs_{est.name}", fn)


class VolTargetedWeights(WeightFunction):
    """Risky weights scaled to an ex-ante volatility target; see :func:`vol_target`.

    ``scales`` records the scale ``s`` used at each call (in call order),
    alongside the inherited ``fallback_dates``.
    """

    def __init__(
        self,
        name: str,
        fn: Callable[[np.ndarray, list[int]], np.ndarray],
        inner: WeightFunction,
        scales: list[float],
    ) -> None:
        super().__init__(name, fn)
        self.inner = inner
        self.scales = scales


def vol_target(
    inner: WeightFunction,
    target_vol: float,
    estimator: CovarianceEstimator,
    *,
    max_scale: float | None = None,
) -> VolTargetedWeights:
    """Scale ``inner``'s portfolio to an annualized ex-ante volatility target.

    At each call: ``u = inner(history)`` (summing to one),
    ``Sigma = estimator(history)`` (annualized),
    ``s = target_vol / sqrt(u' Sigma u)``, capped at ``max_scale`` when
    given, and the result is ``s u``.  The risky weights sum to ``s`` and
    the remaining ``1 - s`` is cash (borrowing when ``s > 1``).  This is
    the scaling of :func:`portopt.research.run_backtest_net`'s own
    ``vol_target`` with the covariance estimator made a choice; see the
    module docstring for why it is not a drop-in weight function for that
    backtest.

    Volatility targeting (Moreira and Muir 2017, "Volatility-Managed
    Portfolios", Journal of Finance 72(4), 1611-1644) helps only when
    volatility clusters and the forecast tracks it; a fast estimator such
    as :func:`ewma_estimator` is the natural choice.

    Fallbacks from ``inner`` are logged in ``inner.fallback_dates``.
    Raises ValueError for a non-positive or non-finite ``target_vol`` or
    ``max_scale``, or when the ex-ante variance is not positive.
    """
    if not isinstance(inner, WeightFunction):
        raise ValueError("inner must be a WeightFunction")
    tv = float(target_vol)
    if not (math.isfinite(tv) and tv > 0.0):
        raise ValueError(f"target_vol must be finite and > 0, got {target_vol!r}")
    cap: float | None = None
    if max_scale is not None:
        cap = float(max_scale)
        if not (math.isfinite(cap) and cap > 0.0):
            raise ValueError(f"max_scale must be None or finite and > 0, got {max_scale!r}")
    est = _check_estimator(estimator)
    scales: list[float] = []

    def fn(h: np.ndarray, _log: list[int]) -> np.ndarray:
        u = np.asarray(inner(h), dtype=float).reshape(-1)
        sig = est(h)
        var = float(u @ sig @ u)
        if not (math.isfinite(var) and var > 0.0):
            raise ValueError(f"vol_target: ex-ante variance is not positive at T={h.shape[0]}")
        s = tv / math.sqrt(var)
        if cap is not None:
            s = min(s, cap)
        scales.append(s)
        out: np.ndarray = s * u
        return out

    name = f"vt{tv:g}_{inner.name}_{est.name}"
    return VolTargetedWeights(name, fn, inner, scales)
