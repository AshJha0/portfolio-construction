"""Helpers for ``real_data_study.py``: a scaled calendar-schedule backtest and
a fast long-only minimum-variance solver.

Neither is part of the :mod:`portopt` package.  Both exist because the
library, as of this study, lacks a combination the study needs, and both
are checked against the library every time the study runs.

* :func:`run_schedule_scaled`.  :func:`portopt.research.walkforward.run_schedule_net`
  handles calendar rebalance rows and a per-row risk-free series but has
  no ``vol_target``, ``max_gross`` or ``vol_estimator``;
  :func:`portopt.research.run_backtest_net` has those but only a fixed
  step and a constant risk-free rate.  This function is the loop of
  ``run_schedule_net`` with the scaling step of ``run_backtest_net``
  inserted before the trade decision, line for line.  :func:`self_check`
  verifies that it reproduces ``run_schedule_net`` bit for bit when no
  scaling is asked for, and ``run_backtest_net`` bit for bit on a regular
  schedule with a constant rate when it is.

* :func:`minvar_longonly`.  The pinned Goldfarb-Idnani QP in
  :mod:`portopt.markowitz` is written for n <= 20 with explicit Python
  loops (see ``portopt/linalg.py``); at n = 49 one solve takes seconds,
  which makes a monthly 42-year backtest with twelve estimator settings
  take hours.  This is the primal active-set method of Nocedal and Wright
  (2006, Algorithm 16.3) for ``min 1/2 w' S w`` subject to ``sum w = 1``
  and ``w >= 0``, with numpy solves.  The problem is strictly convex, so
  its minimizer is unique and any correct solver returns it; every result
  here is accepted only with a KKT certificate, and the study also
  compares it with :func:`portopt.min_variance_longonly` on a fixed set of
  dates.

Reference: J. Nocedal and S. J. Wright (2006), "Numerical Optimization",
2nd edition, Springer, section 16.5.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from portopt import annualize_cov, gross_leverage, sample_cov, validate_returns
from portopt.research import run_backtest_net
from portopt.research import walkforward as wf

__all__ = [
    "KKT_TOL",
    "ScaledSchedule",
    "minvar_kkt_residual",
    "minvar_longonly",
    "run_schedule_scaled",
    "self_check",
]

#: Relative KKT tolerance, the value of ``portopt.markowitz.QP_CONV_TOL``.
KKT_TOL = 1.0e-9
_STEP_TOL = 1.0e-14


def minvar_kkt_residual(sigma: np.ndarray, w: np.ndarray) -> float:
    """Largest KKT violation of ``w`` for ``min 1/2 w'Sw, sum w = 1, w >= 0``.

    With ``g = S w`` and ``lam = w' g`` (the budget multiplier at a KKT
    point), the conditions are: ``sum w = 1``; ``w >= 0``;
    ``g_i = lam`` where ``w_i > 0``; ``g_i >= lam`` where ``w_i = 0``.
    The stationarity and dual parts are scaled by ``1 + max|g|``, as in
    the library's certificate.
    """
    s = np.asarray(sigma, dtype=float)
    x = np.asarray(w, dtype=float)
    g = s @ x
    lam = float(x @ g)
    scale = 1.0 + float(np.max(np.abs(g)))
    feas = max(abs(math.fsum(x.tolist()) - 1.0), max(0.0, -float(np.min(x))))
    pos = x > 0.0
    stat = float(np.max(np.abs(g[pos] - lam))) if pos.any() else 0.0
    dual = float(max(0.0, -float(np.min(g[~pos] - lam)))) if (~pos).any() else 0.0
    return max(feas, stat / scale, dual / scale)


def minvar_longonly(sigma: np.ndarray, *, max_iter: int | None = None) -> np.ndarray:
    """Long-only minimum-variance weights by a primal active-set method.

    Start at ``w = 1/n`` with every bound inactive.  At each iteration
    solve the equality-constrained problem on the free set ``F``
    (``w_F = S_FF^-1 1 / 1' S_FF^-1 1``, the rest zero) and move toward it,
    stopping at the first free weight that would turn negative (that
    weight becomes bound at zero).  When the full step is taken, the
    multipliers of the bound weights are ``g_i - lam`` with ``g = S w``
    and ``lam = w' g``; if the most negative is below the tolerance its
    weight is freed, otherwise the point is optimal.

    Raises ValueError for a non-square or non-finite input, when the
    iteration cap (``10 n + 100`` by default) is reached, or when the
    final point fails the KKT certificate at :data:`KKT_TOL`.
    """
    s = np.asarray(sigma, dtype=float)
    if s.ndim != 2 or s.shape[0] != s.shape[1] or not np.isfinite(s).all():
        raise ValueError("sigma must be a finite square matrix")
    n = s.shape[0]
    cap = 10 * n + 100 if max_iter is None else int(max_iter)
    w = np.full(n, 1.0 / n)
    free = np.ones(n, dtype=bool)
    for _ in range(cap):
        idx = np.nonzero(free)[0]
        y = np.linalg.solve(s[np.ix_(idx, idx)], np.ones(idx.size))
        target = np.zeros(n)
        target[idx] = y / math.fsum(y.tolist())
        p = target - w
        if float(np.max(np.abs(p))) <= _STEP_TOL:
            w = target
            g = s @ w
            mult = g - float(w @ g)
            mult[free] = 0.0
            j = int(np.argmin(mult))
            if mult[j] >= -KKT_TOL * (1.0 + float(np.max(np.abs(g)))):
                res = minvar_kkt_residual(s, w)
                if res > KKT_TOL:
                    raise ValueError(f"minvar_longonly: KKT certificate failed ({res:.3e})")
                return w
            free[j] = True
            continue
        alpha = 1.0
        block = -1
        neg = free & (p < 0.0)
        if neg.any():
            cand = np.nonzero(neg)[0]
            ratios = -w[cand] / p[cand]
            k = int(np.argmin(ratios))
            if ratios[k] < 1.0:
                alpha = float(ratios[k])
                block = int(cand[k])
        w = w + alpha * p
        if block >= 0:
            free[block] = False
        w[~free] = 0.0
        w[free & (w < 0.0)] = 0.0
    raise ValueError(f"minvar_longonly: no convergence in {cap} iterations")


@dataclass(frozen=True)
class ScaledSchedule:
    """Result of :func:`run_schedule_scaled`: the schedule result plus the
    scaled targets and the scale applied at each rebalance."""

    result: wf.ScheduleResult
    targets: np.ndarray
    scales: np.ndarray


def _positive_or_none(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    v = float(value)
    if not (math.isfinite(v) and v > 0.0):
        raise ValueError(f"{name} must be None or finite and > 0, got {value!r}")
    return v


def run_schedule_scaled(
    returns: np.ndarray,
    weight_fn: Callable[[np.ndarray], np.ndarray],
    rebalance_rows: Sequence[int],
    *,
    cost_bps: float = 0.0,
    band: float = 0.0,
    partial: float = 1.0,
    rf_daily: float | np.ndarray = 0.0,
    vol_target: float | None = None,
    max_gross: float | None = None,
    vol_estimator: Callable[[np.ndarray], np.ndarray] | None = None,
) -> ScaledSchedule:
    """``run_schedule_net`` with the scaling step of ``run_backtest_net``.

    At rebalance row ``t_k`` the weight function sees rows ``[0, t_k)``
    and returns ``u`` summing to one.  Then, exactly as in
    ``run_backtest_net``: with ``vol_target`` set,
    ``s = vol_target / sqrt(u' Sigma u)`` where Sigma is
    ``vol_estimator(history)`` (or the annualized ddof=1 sample
    covariance of the history when None); with ``max_gross`` set, ``s`` is
    then reduced so that ``s * sum|u_i| <= max_gross``.  The scaled target
    ``s u`` goes through the band and partial-rebalance decision of
    ``run_schedule_net``, and what is not in risky assets sits in cash at
    the day's ``rf_daily`` (borrowing at the same rate when negative).
    All other accounting is that of ``run_schedule_net``.
    """
    x = validate_returns(returns, min_rows=3)
    t, n = x.shape
    rows = [int(r) for r in rebalance_rows]
    if not rows:
        raise ValueError("rebalance_rows is empty")
    if rows[0] < 2 or rows[-1] >= t or any(b <= a for a, b in zip(rows, rows[1:], strict=False)):
        raise ValueError("rebalance_rows must be strictly increasing within [2, T)")
    kappa = float(cost_bps) / 1.0e4
    band_v = float(band)
    part = float(partial)
    if not (math.isfinite(kappa) and kappa >= 0.0 and math.isfinite(band_v) and band_v >= 0.0):
        raise ValueError("cost_bps and band must be finite and >= 0")
    if not (math.isfinite(part) and 0.0 < part <= 1.0):
        raise ValueError(f"partial must lie in (0, 1], got {partial!r}")
    vt = _positive_or_none("vol_target", vol_target)
    mg = _positive_or_none("max_gross", max_gross)
    if vol_estimator is not None and not callable(vol_estimator):
        raise ValueError("vol_estimator must be None or callable")
    rf_arr = np.broadcast_to(np.asarray(rf_daily, dtype=float), (t,)).copy()
    if not np.isfinite(rf_arr).all():
        raise ValueError("rf_daily must be finite")
    start = rows[0]
    h_days = t - start
    gross = np.zeros(h_days)
    net = np.zeros(h_days)
    costs = np.zeros(h_days)
    k_reb = len(rows)
    pre = np.zeros((k_reb, n))
    post = np.zeros((k_reb, n))
    targets = np.zeros((k_reb, n))
    scales = np.ones(k_reb)
    executed = np.zeros(k_reb, dtype=bool)
    turnovers = np.zeros(k_reb)
    w = np.zeros(n)
    cash = 1.0
    for k, t_k in enumerate(rows):
        history = x[:t_k].view()
        history.flags.writeable = False
        u = np.asarray(weight_fn(history), dtype=float).reshape(-1)
        if u.shape[0] != n or not np.isfinite(u).all():
            raise ValueError(f"weight_fn returned an invalid weight vector at row {t_k}")
        if abs(math.fsum(u.tolist()) - 1.0) > 1e-6:
            raise ValueError(f"weight_fn weights must sum to 1 at row {t_k}")
        s = 1.0
        if vt is not None:
            if vol_estimator is None:
                sig = annualize_cov(sample_cov(x[:t_k]), 252)
            else:
                sig = np.asarray(vol_estimator(history), dtype=float)
                if sig.shape != (n, n) or not np.isfinite(sig).all():
                    raise ValueError(
                        f"vol_estimator must return a finite {n} x {n} matrix at row {t_k}"
                    )
            var = float(u @ sig @ u)
            if not var > 0.0:
                raise ValueError(f"vol_target: ex-ante volatility is zero at row {t_k}")
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
        cash = cash + math.fsum((w - new).tolist())
        w = new
        cost_today = kappa * traded
        t_end = rows[k + 1] if k + 1 < k_reb else t
        for row in range(t_k, t_end):
            r = x[row]
            rf_d = float(rf_arr[row])
            g_r = math.fsum((w * r).tolist()) + cash * rf_d
            c_r = cost_today if row == t_k else 0.0
            n_r = g_r - c_r
            if not 1.0 + n_r > 0.0:
                raise ValueError(f"net asset value fell to zero or below at row {row}")
            gross[row - start] = g_r
            net[row - start] = n_r
            costs[row - start] = c_r
            w = w * (1.0 + r) / (1.0 + n_r)
            cash = (cash * (1.0 + rf_d) - c_r) / (1.0 + n_r)
    res = wf.ScheduleResult(start, rows, gross, net, costs, turnovers, executed, pre, post)
    return ScaledSchedule(res, targets, scales)


def _trailing_inverse_vol(lookback: int) -> Callable[[np.ndarray], np.ndarray]:
    def fn(history: np.ndarray) -> np.ndarray:
        sd = history[-lookback:].std(axis=0, ddof=1)
        inv = 1.0 / sd
        out: np.ndarray = inv / math.fsum(inv.tolist())
        return out

    return fn


def self_check(
    returns: np.ndarray,
    rf_daily: np.ndarray,
    vol_estimator: Callable[[np.ndarray], np.ndarray],
    *,
    lookback: int = 252,
) -> list[str]:
    """Verify :func:`run_schedule_scaled` against the library; return a log.

    1. No scaling, monthly-like rows, a per-row rate, cost, band and
       partial rebalancing: identical arrays to ``run_schedule_net``.
    2. ``vol_target``, ``max_gross`` and ``vol_estimator`` on the regular
       schedule ``window + 21 k`` with a constant rate: identical net
       returns, costs and turnovers to ``run_backtest_net``.

    Raises AssertionError on any mismatch.
    """
    x = np.asarray(returns, dtype=float)
    t = x.shape[0]
    fn = _trailing_inverse_vol(lookback)
    rows = list(range(lookback, t - 1, 21))
    log: list[str] = []
    for cost, band, part in ((10.0, 0.0, 1.0), (25.0, 0.05, 1.0), (10.0, 0.0, 0.5)):
        a = wf.run_schedule_net(
            x, fn, rows, cost_bps=cost, band=band, partial=part, rf_daily=rf_daily
        )
        b = run_schedule_scaled(
            x, fn, rows, cost_bps=cost, band=band, partial=part, rf_daily=rf_daily
        ).result
        for field in (
            "gross_returns",
            "net_returns",
            "costs",
            "turnovers",
            "executed",
            "pre_trade_weights",
            "post_trade_weights",
        ):
            assert np.array_equal(getattr(a, field), getattr(b, field)), (
                f"self_check: run_schedule_scaled differs from run_schedule_net in {field} "
                f"(cost {cost}, band {band}, partial {part})"
            )
        log.append(
            f"run_schedule_scaled == run_schedule_net (bit for bit): cost {cost:g} bp, "
            f"band {band:g}, partial {part:g}, {len(rows)} rebalances, {t - rows[0]} days"
        )
    rf_const = float(np.mean(rf_daily)) * 252.0
    for vt, mg, est in ((0.10, 1.5, vol_estimator), (0.10, None, None), (0.15, 1.0, vol_estimator)):
        c = run_backtest_net(
            x,
            fn,
            window=lookback,
            step=21,
            cost_bps=10.0,
            rf=rf_const,
            vol_target=vt,
            max_gross=mg,
            vol_estimator=est,
        )
        d = run_schedule_scaled(
            x,
            fn,
            c.rebalance_dates,
            cost_bps=10.0,
            rf_daily=rf_const / 252.0,
            vol_target=vt,
            max_gross=mg,
            vol_estimator=est,
        )
        assert np.array_equal(c.net_returns, d.result.net_returns), "self_check: net returns differ"
        assert np.array_equal(c.costs, d.result.costs), "self_check: costs differ"
        assert np.array_equal(c.turnovers, d.result.turnovers), "self_check: turnovers differ"
        assert np.array_equal(c.scales, d.scales), "self_check: scales differ"
        log.append(
            f"run_schedule_scaled == run_backtest_net (bit for bit): vol_target {vt:g}, "
            f"max_gross {mg}, vol_estimator {'given' if est is not None else 'sample'}, "
            f"{len(c.rebalance_dates)} rebalances"
        )
    return log
