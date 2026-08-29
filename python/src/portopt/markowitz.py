"""Markowitz mean-variance: closed forms, simplex projection, and PGD.

Closed-form results use the classical two-fund theorem.  With
``x = Sigma^{-1} mu`` and ``y = Sigma^{-1} 1`` (both computed by Cholesky
solves, pinned in linalg.py) define the scalars::

    A = 1' Sigma^{-1} mu = sum(x)      (also = 1' x = mu' y)
    B = mu' Sigma^{-1} mu = mu' x
    C = 1' Sigma^{-1} 1  = sum(y)
    D = B * C - A * A

Derivation: minimizing ``w' Sigma w`` subject to ``1'w = 1`` (and
optionally ``mu'w = m``) with Lagrange multipliers gives
``w = Sigma^{-1}(lam * mu + gam * 1)``; the constraints become the linear
system ``[[B, A], [A, C]] [lam, gam]' = [m, 1]'`` whose determinant is D.
Hence every frontier portfolio is ``w(m) = g + h * m`` with::

    g = (B * y - A * x) / D
    h = (C * x - A * y) / D

Special points: minimum-variance ``w_mv = y / C`` (return A/C, variance
1/C); tangency w.r.t. risk-free rate rf:
``w_tan = Sigma^{-1}(mu - rf 1) / (A - rf C)`` (defined only when
``A - rf C > 0``, i.e. rf below the min-variance return).

Constrained problems (long-only, sum-to-one, optional box bounds) are
solved by projected gradient descent with monotone backtracking and an
exact sort-based projection onto the box-constrained simplex; all steps
are pinned in API_SPEC.md sections 3.3-3.5 so scipy-free ports match.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from .linalg import cholesky, chol_solve

__all__ = [
    "TwoFundScalars",
    "two_fund_scalars",
    "min_variance_weights",
    "tangency_weights",
    "frontier_weights",
    "efficient_frontier",
    "project_simplex_box",
    "pgd_minimize",
    "min_variance_longonly",
    "constrained_frontier_weights",
    "constrained_frontier",
]

FRONTIER_PENALTY = 1.0e4
PGD_TOL = 1.0e-12
PGD_MAX_ITER = 10_000


def _validate_mu_sigma(mu: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.asarray(mu, dtype=float).reshape(-1)
    s = np.asarray(sigma, dtype=float)
    if s.ndim != 2 or s.shape[0] != s.shape[1]:
        raise ValueError(f"sigma must be square, got shape {s.shape}")
    if m.shape[0] != s.shape[0]:
        raise ValueError(
            f"mu length {m.shape[0]} does not match sigma dimension {s.shape[0]}"
        )
    if not np.isfinite(m).all():
        raise ValueError("mu contains NaN or infinite entries")
    return m, s


@dataclass(frozen=True)
class TwoFundScalars:
    """Two-fund theorem scalars A, B, C, D (see module docstring)."""

    a: float
    b: float
    c: float
    d: float


def two_fund_scalars(mu: np.ndarray, sigma: np.ndarray) -> TwoFundScalars:
    """Compute A = 1'S^-1 mu, B = mu'S^-1 mu, C = 1'S^-1 1, D = BC - A^2.

    Raises ValueError if sigma is not symmetric positive definite.
    """
    m, s = _validate_mu_sigma(mu, sigma)
    low = cholesky(s)
    x = chol_solve(low, m)
    y = chol_solve(low, np.ones_like(m))
    a = float(np.sum(x))
    b = float(m @ x)
    c = float(np.sum(y))
    d = b * c - a * a
    return TwoFundScalars(a=a, b=b, c=c, d=d)


def min_variance_weights(sigma: np.ndarray) -> np.ndarray:
    """Unconstrained (sum-to-one) minimum-variance weights ``S^-1 1 / C``.

    For n = 1 returns the trivial [1.0].
    """
    s = np.asarray(sigma, dtype=float)
    if s.ndim != 2 or s.shape[0] != s.shape[1]:
        raise ValueError(f"sigma must be square, got shape {s.shape}")
    if s.shape[0] == 1:
        if s[0, 0] <= 0 or not np.isfinite(s[0, 0]):
            raise ValueError("1x1 covariance must be positive and finite")
        return np.array([1.0])
    y = chol_solve(cholesky(s), np.ones(s.shape[0]))
    return y / float(np.sum(y))


def tangency_weights(mu: np.ndarray, sigma: np.ndarray, rf: float) -> np.ndarray:
    """Tangency (max-Sharpe) weights ``S^-1 (mu - rf 1) / (A - rf C)``.

    Raises
    ------
    ValueError
        If ``A - rf*C <= 0`` — the tangency portfolio is undefined when
        the risk-free rate is at or above the minimum-variance return
        (in particular when rf exceeds every expected return).
    """
    m, s = _validate_mu_sigma(mu, sigma)
    rfv = float(rf)
    if not math.isfinite(rfv):
        raise ValueError("rf must be finite")
    if s.shape[0] == 1:
        if m[0] <= rfv:
            raise ValueError(
                "tangency undefined: risk-free rate >= expected return of the single asset"
            )
        if s[0, 0] <= 0 or not np.isfinite(s[0, 0]):
            raise ValueError("1x1 covariance must be positive and finite")
        return np.array([1.0])
    low = cholesky(s)
    z = chol_solve(low, m - rfv)
    denom = float(np.sum(z))  # = A - rf*C
    if denom <= 0.0:
        raise ValueError(
            "tangency undefined: A - rf*C <= 0 (risk-free rate is at or above "
            "the minimum-variance portfolio return)"
        )
    return z / denom


def frontier_weights(mu: np.ndarray, sigma: np.ndarray, target: float) -> np.ndarray:
    """Efficient-frontier weights for a target return via the two-fund theorem.

    ``w(m) = g + h m`` with g, h as in the module docstring.

    Raises
    ------
    ValueError
        If n = 1 and the target differs from the single asset's mean; if
        D <= 0 (degenerate frontier, e.g. mu proportional to 1); or if
        the target return lies below the minimum-variance return A/C
        (inefficient branch, rejected per spec).
    """
    m, s = _validate_mu_sigma(mu, sigma)
    tgt = float(target)
    if not math.isfinite(tgt):
        raise ValueError("target return must be finite")
    if s.shape[0] == 1:
        if abs(tgt - m[0]) > 1e-12 * (1.0 + abs(m[0])):
            raise ValueError(
                "n=1: target return must equal the single asset's expected return"
            )
        return np.array([1.0])
    low = cholesky(s)
    x = chol_solve(low, m)
    y = chol_solve(low, np.ones_like(m))
    a = float(np.sum(x))
    b = float(m @ x)
    c = float(np.sum(y))
    d = b * c - a * a
    # Pinned relative degeneracy test: by Cauchy-Schwarz D >= 0 with
    # equality iff mu is proportional to ones; rounding turns the exact
    # zero into a tiny residual, so compare against B*C (> 0).
    if d <= 1e-12 * b * c:
        raise ValueError(
            f"degenerate frontier: D = B*C - A^2 = {d:.3e} <= 1e-12*B*C "
            "(expected returns proportional to ones?)"
        )
    mv_ret = a / c
    if tgt < mv_ret - 1e-12 * (1.0 + abs(mv_ret)):
        raise ValueError(
            f"target return {tgt:.6g} lies below the minimum-variance return "
            f"{mv_ret:.6g}; the efficient frontier is defined for targets >= A/C"
        )
    g = (b * y - a * x) / d
    h = (c * x - a * y) / d
    return g + h * tgt


def efficient_frontier(
    mu: np.ndarray, sigma: np.ndarray, n_points: int = 25
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unconstrained frontier sweep.

    Pinned: targets are ``n_points`` equally spaced values from the
    minimum-variance return A/C to ``max(mu)`` inclusive
    (``target_k = mv + k*(max(mu)-mv)/(n_points-1)``).

    Returns (targets, weights [n_points x n], variances).
    """
    m, s = _validate_mu_sigma(mu, sigma)
    if n_points < 2:
        raise ValueError("n_points must be >= 2")
    sc = two_fund_scalars(m, s)
    if sc.d <= 1e-12 * sc.b * sc.c:
        raise ValueError("degenerate frontier: D <= 1e-12*B*C")
    lo = sc.a / sc.c
    hi = float(np.max(m))
    targets = np.array(
        [lo + k * (hi - lo) / (n_points - 1) for k in range(n_points)], dtype=float
    )
    weights = np.zeros((n_points, m.shape[0]), dtype=float)
    variances = np.zeros(n_points, dtype=float)
    for k, tgt in enumerate(targets):
        w = frontier_weights(m, s, float(tgt))
        weights[k] = w
        variances[k] = float(w @ s @ w)
    return targets, weights, variances


def project_simplex_box(
    v: np.ndarray,
    total: float = 1.0,
    lb: Optional[Sequence[float] | float] = None,
    ub: Optional[Sequence[float] | float] = None,
) -> np.ndarray:
    """Exact Euclidean projection onto {w : sum w = total, lb <= w <= ub}.

    Pinned sort-based algorithm (API_SPEC section 3.4).  The KKT solution
    is ``w_i = clip(v_i - t, lb_i, ub_i)`` for the unique shift t making
    the sum equal ``total``; ``g(t) = sum_i clip(v_i - t, lb_i, ub_i)`` is
    piecewise-linear and non-increasing, with kinks (clipping breakpoints)
    at ``v_i - lb_i`` and, for finite bounds, ``v_i - ub_i``.  The
    algorithm sorts the breakpoints ascending, scans them left to right to
    bracket ``g(t) = total``, and finishes with one exact linear
    interpolation on the bracketing segment — a finite, deterministic
    procedure (no iteration tolerance).  With lb = 0, ub = +inf it
    reproduces the classic Duchi et al. sorted simplex projection.

    Defaults: lb = 0, ub = +inf (long-only simplex).

    Raises
    ------
    ValueError
        If bounds are infeasible (``sum(lb) > total`` or
        ``sum(ub) < total`` or any ``lb_i > ub_i``) or inputs are invalid.
    """
    x = np.asarray(v, dtype=float).reshape(-1)
    n = x.shape[0]
    if n == 0:
        raise ValueError("cannot project an empty vector")
    if not np.isfinite(x).all():
        raise ValueError("projection input contains NaN or infinite entries")
    s = float(total)
    if not math.isfinite(s):
        raise ValueError("total must be finite")

    lo = np.zeros(n) if lb is None else np.broadcast_to(
        np.asarray(lb, dtype=float), (n,)
    ).copy()
    hi = np.full(n, np.inf) if ub is None else np.broadcast_to(
        np.asarray(ub, dtype=float), (n,)
    ).copy()
    if np.any(np.isnan(lo)) or np.any(np.isnan(hi)):
        raise ValueError("bounds contain NaN")
    if np.any(lo > hi):
        raise ValueError("infeasible bounds: some lb_i > ub_i")
    lo_sum = float(np.sum(lo))
    hi_sum = float(np.sum(hi))
    feas_tol = 1e-12 * (1.0 + abs(s))
    if lo_sum > s + feas_tol:
        raise ValueError(
            f"infeasible bounds: sum(lb) = {lo_sum:.6g} exceeds total {s:.6g}"
        )
    if hi_sum < s - feas_tol:
        raise ValueError(
            f"infeasible bounds: sum(ub) = {hi_sum:.6g} is below total {s:.6g}"
        )

    def g(t: float) -> float:
        return float(np.sum(np.clip(x - t, lo, hi)))

    bps = [x[i] - lo[i] for i in range(n)]
    for i in range(n):
        if math.isfinite(hi[i]):
            bps.append(x[i] - hi[i])
    bps.sort()

    t_first = bps[0]
    g_first = g(t_first)
    if g_first < s:
        # t* lies left of every breakpoint; there the unclipped (infinite-ub)
        # coordinates give slope -k with k = number of infinite upper bounds.
        k = int(np.sum(~np.isfinite(hi)))
        if k == 0:
            # sum(ub) >= s guarantees g_first >= s up to rounding; snap.
            return np.clip(x - t_first, lo, hi)
        return np.clip(x - (t_first - (s - g_first) / k), lo, hi)

    t_prev, g_prev = t_first, g_first
    for t_cur in bps[1:]:
        g_cur = g(t_cur)
        if g_cur <= s:
            if g_prev == g_cur:
                t_star = t_cur
            else:
                t_star = t_prev + (g_prev - s) * (t_cur - t_prev) / (g_prev - g_cur)
            return np.clip(x - t_star, lo, hi)
        t_prev, g_prev = t_cur, g_cur
    # Right of every breakpoint g(t) = sum(lb) <= s; feasibility means equality.
    return np.clip(x - bps[-1], lo, hi)


def pgd_minimize(
    fun: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    project: Callable[[np.ndarray], np.ndarray],
    w0: np.ndarray,
    tol: float = PGD_TOL,
    max_iter: int = PGD_MAX_ITER,
) -> tuple[np.ndarray, int, bool]:
    """Projected gradient descent with monotone backtracking (pinned).

    Pinned algorithm (API_SPEC section 3.5):

    1. ``w = project(w0)``; step ``t = 1``.
    2. Each iteration: set ``t = 2 t``; compute ``g = grad(w)``;
       ``w+ = project(w - t g)``; while the sufficient-decrease condition
       ``f(w+) <= f(w) + g.(w+ - w) + ||w+ - w||^2 / (2 t)`` fails,
       halve ``t`` and recompute ``w+`` (at most 100 halvings).
    3. Stop when ``max_i |w+_i - w_i| < tol`` (converged) or after
       ``max_iter`` iterations (not converged; returned flag is False —
       reported, not raised).

    Deterministic — no randomness anywhere.  Returns (w, iterations, converged).
    """
    w = project(np.asarray(w0, dtype=float).reshape(-1))
    t = 1.0
    for it in range(1, max_iter + 1):
        t *= 2.0
        gvec = grad(w)
        fw = fun(w)
        for _ in range(100):
            w_new = project(w - t * gvec)
            dw = w_new - w
            if fun(w_new) <= fw + float(gvec @ dw) + float(dw @ dw) / (2.0 * t):
                break
            t *= 0.5
        else:
            w_new = project(w - t * gvec)
            dw = w_new - w
        step = float(np.max(np.abs(dw)))
        w = w_new
        if step < tol:
            return w, it, True
    return w, max_iter, False


def _pgd_setup(sigma: np.ndarray, lb, ub) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray], np.ndarray]:
    s = np.asarray(sigma, dtype=float)
    cholesky(s)  # PSD/validity gate; raises ValueError on bad input
    n = s.shape[0]

    def project(v: np.ndarray) -> np.ndarray:
        return project_simplex_box(v, 1.0, lb=lb, ub=ub)

    w0 = np.full(n, 1.0 / n)
    return s, project, w0


def min_variance_longonly(
    sigma: np.ndarray,
    lb: Optional[Sequence[float] | float] = None,
    ub: Optional[Sequence[float] | float] = None,
    tol: float = PGD_TOL,
    max_iter: int = PGD_MAX_ITER,
) -> np.ndarray:
    """Constrained minimum variance: min 1/2 w'Sw over the box-simplex.

    Defaults to the long-only simplex (lb = 0, ub = +inf).  Solved by the
    pinned PGD of :func:`pgd_minimize` started from projected equal
    weights.  The objective is strictly convex, so the minimizer is unique
    and language ports agree to well below the golden tolerance 1e-8.
    """
    s, project, w0 = _pgd_setup(sigma, lb, ub)

    def fun(w: np.ndarray) -> float:
        return 0.5 * float(w @ s @ w)

    def grad(w: np.ndarray) -> np.ndarray:
        return s @ w

    w, _, _ = pgd_minimize(fun, grad, project, w0, tol=tol, max_iter=max_iter)
    return w


def constrained_frontier_weights(
    mu: np.ndarray,
    sigma: np.ndarray,
    target: float,
    penalty: float = FRONTIER_PENALTY,
    lb: Optional[Sequence[float] | float] = None,
    ub: Optional[Sequence[float] | float] = None,
    tol: float = PGD_TOL,
    max_iter: int = PGD_MAX_ITER,
) -> np.ndarray:
    """Constrained frontier point via the pinned quadratic-penalty PGD.

    Pinned objective (API_SPEC section 3.6)::

        f(w) = 1/2 w' Sigma w + penalty/2 * (mu'w - target)^2

    minimized over {sum w = 1, lb <= w <= ub} with penalty = 1e4 by
    default.  The penalty formulation itself is the contract: the golden
    values are defined as the exact minimizer of THIS objective (return
    deviates from the target by O(1/penalty), which is intentional).
    """
    m, s = _validate_mu_sigma(mu, sigma)
    if penalty <= 0:
        raise ValueError("penalty must be positive")
    tgt = float(target)
    _, project, w0 = _pgd_setup(s, lb, ub)

    def fun(w: np.ndarray) -> float:
        gap = float(m @ w) - tgt
        return 0.5 * float(w @ s @ w) + 0.5 * penalty * gap * gap

    def grad(w: np.ndarray) -> np.ndarray:
        gap = float(m @ w) - tgt
        return s @ w + penalty * gap * m

    w, _, _ = pgd_minimize(fun, grad, project, w0, tol=tol, max_iter=max_iter)
    return w


def constrained_frontier(
    mu: np.ndarray,
    sigma: np.ndarray,
    n_points: int = 25,
    penalty: float = FRONTIER_PENALTY,
    lb: Optional[Sequence[float] | float] = None,
    ub: Optional[Sequence[float] | float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Constrained frontier sweep over the same pinned target grid as
    :func:`efficient_frontier` (min-variance return A/C to max(mu)).

    Returns (targets, weights, variances).
    """
    m, s = _validate_mu_sigma(mu, sigma)
    if n_points < 2:
        raise ValueError("n_points must be >= 2")
    sc = two_fund_scalars(m, s)
    lo = sc.a / sc.c
    hi = float(np.max(m))
    targets = np.array(
        [lo + k * (hi - lo) / (n_points - 1) for k in range(n_points)], dtype=float
    )
    weights = np.zeros((n_points, m.shape[0]), dtype=float)
    variances = np.zeros(n_points, dtype=float)
    for k, tgt in enumerate(targets):
        w = constrained_frontier_weights(m, s, float(tgt), penalty=penalty, lb=lb, ub=ub)
        weights[k] = w
        variances[k] = float(w @ s @ w)
    return targets, weights, variances
