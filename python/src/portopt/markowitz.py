"""Markowitz mean-variance: closed forms, simplex projection, active-set QP.

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
``A - rf C > 0``, i.e. rf below the min-variance return; additionally
rejected as ill-conditioned when ``A - rf C`` is within a relative
``1e-6`` of cancelling, see :func:`tangency_weights`).

Constrained problems (sum-to-one, optional target return, box bounds)
are strictly convex quadratic programs; they are solved by the finite
dual active-set method of Goldfarb & Idnani (1983) whose every linear
solve is the pinned Cholesky (API_SPEC.md section 3.6).  The result is
the exact KKT point of the QP up to rounding, so all language ports agree
to ~1e-13 and the ``converged`` flag is a statement about the KKT
conditions, not about an iterate step size.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .linalg import cholesky, chol_solve

__all__ = [
    "TangencyUndefinedError",
    "TwoFundScalars",
    "QpResult",
    "QP_MAX_ITER",
    "TANGENCY_COND_RTOL",
    "two_fund_scalars",
    "min_variance_weights",
    "tangency_weights",
    "gross_leverage",
    "frontier_weights",
    "efficient_frontier",
    "project_simplex_box",
    "attainable_return_range",
    "min_variance_longonly_solve",
    "min_variance_longonly",
    "constrained_frontier_solve",
    "constrained_frontier_weights",
    "constrained_frontier",
]

#: Pinned iteration cap of the dual active-set QP (each iteration adds or
#: drops one constraint; the method is finite for a strictly convex
#: objective, so this bound is only hit on numerically inconsistent inputs
#: and is then reported via ``converged=False``).
QP_MAX_ITER = 1000
#: Pinned feasibility tolerance of the dual active-set QP: a bound is
#: treated as violated when ``n'w - b < -QP_FEAS_TOL * (1 + |b|)``;
#: rounding-level violations below it are clipped in the final answer.
QP_FEAS_TOL = 1.0e-13
#: Pinned tolerance below which a rounding-level violation of a constraint
#: that is linearly dependent on the active set (primal-degenerate vertex)
#: is accepted as satisfied; ``kkt_residual`` reports the actual value.
QP_DEGENERATE_TOL = 1.0e-9
#: Pinned KKT certificate tolerance: ``converged`` requires every equality
#: residual and bound violation <= QP_CONV_TOL * (1 + max|rhs|) and the
#: stationarity residual |S w - N u| <= QP_CONV_TOL * (1 + scale).
QP_CONV_TOL = 1.0e-9
#: Pinned relative conditioning threshold for the tangency denominator.
TANGENCY_COND_RTOL = 1.0e-6



class TangencyUndefinedError(ValueError):
    """Tangency portfolio undefined or ill-conditioned (``A - rf C`` at or
    within a relative ``TANGENCY_COND_RTOL`` of zero)."""


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


def _validate_1x1(s: np.ndarray) -> None:
    if s[0, 0] <= 0 or not np.isfinite(s[0, 0]):
        raise ValueError("1x1 covariance must be positive and finite")


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
        _validate_1x1(s)
        return np.array([1.0])
    y = chol_solve(cholesky(s), np.ones(s.shape[0]))
    return y / float(np.sum(y))


def tangency_weights(mu: np.ndarray, sigma: np.ndarray, rf: float) -> np.ndarray:
    """Tangency (max-Sharpe) weights ``S^-1 (mu - rf 1) / (A - rf C)``.

    Raises
    ------
    TangencyUndefinedError
        If ``A - rf*C <= TANGENCY_COND_RTOL * max(|A|, |rf|*C)``.  The
        tangency portfolio is undefined when the risk-free rate is at or
        above the minimum-variance return ``A/C`` (in particular when rf
        exceeds every expected return), and it is numerically meaningless
        when rf sits within a relative 1e-6 of ``A/C``: the weights scale
        like ``1/(A - rf C)`` and reach gross leverage of 1e6+ there.
    ValueError
        On malformed inputs or a non-SPD covariance.
    """
    m, s = _validate_mu_sigma(mu, sigma)
    rfv = float(rf)
    if not math.isfinite(rfv):
        raise ValueError("rf must be finite")
    if s.shape[0] == 1:
        _validate_1x1(s)
        if m[0] <= rfv:
            raise TangencyUndefinedError(
                "tangency undefined: risk-free rate >= expected return of the single asset"
            )
        return np.array([1.0])
    low = cholesky(s)
    z = chol_solve(low, m - rfv)
    denom = float(np.sum(z))  # = A - rf*C
    if denom <= 0.0:
        raise TangencyUndefinedError(
            "tangency undefined: A - rf*C <= 0 (risk-free rate is at or above "
            "the minimum-variance portfolio return)"
        )
    # Conditioning guard: A - rf*C is a difference of two O(A) numbers.
    a = float(np.sum(chol_solve(low, m)))
    c = float(np.sum(chol_solve(low, np.ones_like(m))))
    scale = max(abs(a), abs(rfv) * c)
    if denom <= TANGENCY_COND_RTOL * scale:
        raise TangencyUndefinedError(
            f"tangency ill-conditioned: A - rf*C = {denom:.3e} is within a relative "
            f"{TANGENCY_COND_RTOL:g} of cancelling (rf within {TANGENCY_COND_RTOL:g} "
            f"of the minimum-variance return {a / c:.6g}); weights would carry "
            "gross leverage of order 1e6 or more"
        )
    return z / denom


def gross_leverage(w: np.ndarray) -> float:
    """Gross exposure ``sum_i |w_i|`` (1.0 for any long-only portfolio).

    Unconstrained tangency and Black-Litterman weights can carry large
    gross leverage; a desk should cap this number before anything reaches
    an order-management system.
    """
    wv = np.asarray(w, dtype=float).reshape(-1)
    if wv.shape[0] == 0 or not np.isfinite(wv).all():
        raise ValueError("weights must be a non-empty finite vector")
    return float(math.fsum(abs(float(v)) for v in wv))


def _two_fund_gh(m: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Frontier generators g, h and the min-variance return A/C (n >= 2)."""
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
    g = (b * y - a * x) / d
    h = (c * x - a * y) / d
    return g, h, a / c


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
        _validate_1x1(s)
        if abs(tgt - m[0]) > 1e-12 * (1.0 + abs(m[0])):
            raise ValueError(
                "n=1: target return must equal the single asset's expected return"
            )
        return np.array([1.0])
    g, h, mv_ret = _two_fund_gh(m, s)
    if tgt < mv_ret - 1e-12 * (1.0 + abs(mv_ret)):
        raise ValueError(
            f"target return {tgt:.6g} lies below the minimum-variance return "
            f"{mv_ret:.6g}; the efficient frontier is defined for targets >= A/C"
        )
    return g + h * tgt


def _frontier_grid(lo: float, hi: float, n_points: int, what: str) -> np.ndarray:
    if hi <= lo + 1e-12 * (1.0 + abs(lo)):
        raise ValueError(
            f"frontier grid empty: {what} ({hi:.6g}) does not exceed the "
            f"minimum-variance return ({lo:.6g}); supply targets explicitly via "
            "frontier_weights / constrained_frontier_weights"
        )
    targets = [lo + k * (hi - lo) / (n_points - 1) for k in range(n_points)]
    targets[-1] = hi  # pinned: the last grid point is exactly hi
    return np.array(targets, dtype=float)


def efficient_frontier(
    mu: np.ndarray, sigma: np.ndarray, n_points: int = 25
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unconstrained frontier sweep.

    Pinned: targets are ``n_points`` equally spaced values from the
    minimum-variance return A/C to ``max(mu)`` inclusive
    (``target_k = mv + k*(max(mu)-mv)/(n_points-1)``).  The two-fund
    generators g, h are computed once (one Cholesky, two solves) and reused
    for every point.

    Returns (targets, weights [n_points x n], variances).

    Raises ValueError if ``max(mu) <= A/C`` (the GMV portfolio already
    earns at least the best single asset, which happens when it shorts a
    low-return asset) — supply explicit targets to :func:`frontier_weights`.
    """
    m, s = _validate_mu_sigma(mu, sigma)
    if n_points < 2:
        raise ValueError("n_points must be >= 2")
    if s.shape[0] == 1:
        raise ValueError("frontier sweep requires at least 2 assets")
    g, h, lo = _two_fund_gh(m, s)
    targets = _frontier_grid(lo, float(np.max(m)), n_points, "max(mu)")
    weights = np.zeros((n_points, m.shape[0]), dtype=float)
    variances = np.zeros(n_points, dtype=float)
    for k, tgt in enumerate(targets):
        w = g + h * float(tgt)
        weights[k] = w
        variances[k] = float(w @ s @ w)
    return targets, weights, variances


def _broadcast_bounds(
    n: int,
    lb: Optional[Sequence[float] | float],
    ub: Optional[Sequence[float] | float],
    total: float,
) -> tuple[np.ndarray, np.ndarray]:
    lo = np.zeros(n) if lb is None else np.broadcast_to(
        np.asarray(lb, dtype=float), (n,)
    ).copy()
    hi = np.full(n, np.inf) if ub is None else np.broadcast_to(
        np.asarray(ub, dtype=float), (n,)
    ).copy()
    if np.any(np.isnan(lo)) or np.any(np.isnan(hi)):
        raise ValueError("bounds contain NaN")
    if not np.isfinite(lo).all():
        raise ValueError("lower bounds must be finite")
    if np.any(hi == -np.inf):
        raise ValueError("upper bounds must be > -inf")
    if np.any(lo > hi):
        raise ValueError("infeasible bounds: some lb_i > ub_i")
    lo_sum = float(np.sum(lo))
    hi_sum = float(np.sum(hi))
    feas_tol = 1e-12 * (1.0 + abs(total))
    if lo_sum > total + feas_tol:
        raise ValueError(
            f"infeasible bounds: sum(lb) = {lo_sum:.6g} exceeds total {total:.6g}"
        )
    if hi_sum < total - feas_tol:
        raise ValueError(
            f"infeasible bounds: sum(ub) = {hi_sum:.6g} is below total {total:.6g}"
        )
    return lo, hi


def project_simplex_box(
    v: np.ndarray,
    total: float = 1.0,
    lb: Optional[Sequence[float] | float] = None,
    ub: Optional[Sequence[float] | float] = None,
) -> np.ndarray:
    """Exact Euclidean projection onto {w : sum w = total, lb <= w <= ub}.

    Pinned sort-based algorithm (API_SPEC section 3.5).  The KKT solution
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
    lo, hi = _broadcast_bounds(n, lb, ub, s)

    def g(t: float) -> float:
        return float(math.fsum(min(max(float(x[i] - t), float(lo[i])), float(hi[i])) for i in range(n)))

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


# --------------------------------------------------------------------- QP


@dataclass(frozen=True)
class QpResult:
    """Output of the active-set QP solvers.

    Attributes
    ----------
    weights:
        Feasible weights (bounds respected exactly; equality constraints
        to rounding).
    iterations:
        Dual active-set iterations executed (constraint additions/drops).
    converged:
        True iff the KKT conditions are certified at the returned point:
        primal feasibility and stationarity ``S w = N u`` within
        ``QP_CONV_TOL`` (bound multipliers are non-negative by
        construction).  False means the iteration cap was hit or the
        constraint set proved numerically inconsistent — the returned
        weights are then not certified optimal.
    kkt_residual:
        Largest KKT violation at ``weights``: max of the equality
        residuals, bound violations and the stationarity residual
        ``|S w - N u|_inf`` (N = active normals, u = multipliers).
    """

    weights: np.ndarray
    iterations: int
    converged: bool
    kkt_residual: float


def _greedy_extreme(
    m: np.ndarray, lo: np.ndarray, hi: np.ndarray, total: float, descending: bool
) -> np.ndarray:
    """Vertex of {sum w = total, lb <= w <= ub} maximizing (descending) or
    minimizing (ascending) mu'w: fill from lb in mu order (ties by index)."""
    n = m.shape[0]
    order = sorted(range(n), key=(lambda i: (-m[i], i)) if descending else (lambda i: (m[i], i)))
    w = lo.copy()
    remaining = total - float(math.fsum(float(v) for v in lo))
    if remaining < 0.0:
        remaining = 0.0
    for i in order:
        if remaining <= 0.0:
            break
        room = hi[i] - lo[i]
        add = remaining if room > remaining else room
        w[i] = lo[i] + add
        remaining -= add
    return w


def attainable_return_range(
    mu: np.ndarray,
    lb: Optional[Sequence[float] | float] = None,
    ub: Optional[Sequence[float] | float] = None,
) -> tuple[float, float]:
    """Range ``[r_min, r_max]`` of ``mu'w`` over ``{sum w = 1, lb <= w <= ub}``.

    Both extremes are vertices found by the pinned greedy fill (assets in
    ascending / descending ``mu`` order, ties by index).
    """
    m = np.asarray(mu, dtype=float).reshape(-1)
    if m.shape[0] == 0 or not np.isfinite(m).all():
        raise ValueError("mu must be a non-empty finite vector")
    lo, hi = _broadcast_bounds(m.shape[0], lb, ub, 1.0)
    w_min = _greedy_extreme(m, lo, hi, 1.0, descending=False)
    w_max = _greedy_extreme(m, lo, hi, 1.0, descending=True)
    return float(m @ w_min), float(m @ w_max)


def _constraint_normal(n: int, a_eq: np.ndarray, kind: int, idx: int) -> np.ndarray:
    if kind == 0:
        return a_eq[idx].copy()
    e = np.zeros(n)
    e[idx] = -1.0 if kind == 2 else 1.0
    return e


def _dual_active_set_qp(
    s: np.ndarray,
    a_eq: np.ndarray,
    b_eq: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    max_iter: int,
) -> QpResult:
    """Goldfarb-Idnani dual active-set method for
    ``min 1/2 w'Sw`` s.t. ``a_eq w = b_eq``, ``lo <= w <= hi`` (API_SPEC 3.6).

    Constraints are ``n_c'w >= b_c``: the equalities (the rows of ``a_eq``
    followed by every fixed coordinate ``lo_i == hi_i``; added first, in
    order, multipliers unrestricted), then the lower bounds ``e_i'w >= lo_i``
    and the finite upper bounds ``-e_i'w >= -hi_i`` of the non-fixed
    coordinates.  Starting from the
    unconstrained minimizer ``w = 0`` the method adds the most violated
    constraint, taking the largest step that keeps every inequality
    multiplier non-negative (dropping the blocking constraint on a partial
    step), until no constraint is violated.  Every linear solve is the
    pinned Cholesky of ``S`` (factored once) or of the small Gram matrix
    ``N' S^-1 N`` of the active normals.
    """
    n = s.shape[0]
    low = cholesky(s)
    # Constraint list: (kind, idx, rhs); kind 0 = equality row of a_eq,
    # 3 = fixed coordinate (lo_i == hi_i, treated as the equality w_i = lo_i),
    # 1 = lower bound, 2 = upper bound.  Equalities come first.
    cons: list[tuple[int, int, float]] = [(0, j, float(b_eq[j])) for j in range(a_eq.shape[0])]
    cons += [(3, i, float(lo[i])) for i in range(n) if lo[i] == hi[i]]
    meq = len(cons)
    cons += [(1, i, float(lo[i])) for i in range(n) if lo[i] != hi[i]]
    cons += [(2, i, -float(hi[i])) for i in range(n) if lo[i] != hi[i] and math.isfinite(hi[i])]
    normals = [_constraint_normal(n, a_eq, k, i) for (k, i, _) in cons]
    rhs = np.array([c[2] for c in cons])
    n_cons = len(cons)

    w = np.zeros(n)
    active: list[int] = []
    u: list[float] = []
    skipped: list[int] = []
    iterations = 0
    p = -1
    u_p = 0.0
    while iterations < max_iter:
        iterations += 1
        if p < 0:
            # (1) pick the next constraint: equalities first, then the most
            # violated inequality (ties -> lowest index).
            if len(active) < meq:
                p = len(active)
            else:
                worst = 0.0
                for c in range(meq, n_cons):
                    if c in active or c in skipped:
                        continue
                    viol = float(normals[c] @ w) - rhs[c]
                    if viol < -QP_FEAS_TOL * (1.0 + abs(rhs[c])) and viol < worst:
                        worst, p = viol, c
                if p < 0:
                    break  # no violated constraint: dual method finished
            u_p = 0.0
        # (2) primal / dual step directions for constraint p.
        q = len(active)
        ys = [chol_solve(low, normals[c]) for c in active]
        v = chol_solve(low, normals[p])
        r = np.zeros(q)
        if q > 0:
            gram = np.zeros((q, q))
            for i in range(q):
                for j in range(i, q):
                    gram[i, j] = float(normals[active[i]] @ ys[j])
                    gram[j, i] = gram[i, j]
            r = chol_solve(cholesky(gram), np.array([float(normals[c] @ v) for c in active]))
        z = v.copy()
        for i in range(q):
            z = z - r[i] * ys[i]
        zn = float(z @ normals[p])
        vn = float(v @ normals[p])
        s_p = float(normals[p] @ w) - rhs[p]
        # (3) step lengths: t1 keeps inequality multipliers >= 0, t2 satisfies p.
        t1 = math.inf
        j_block = -1
        for i in range(q):
            if active[i] >= meq and r[i] > 0.0:
                cand = u[i] / r[i]
                if cand < t1:
                    t1, j_block = cand, i
        t2 = -s_p / zn if zn > 1e-10 * vn else math.inf
        t = min(t1, t2)
        if not math.isfinite(t):
            # n_p is a combination of the active normals and no multiplier
            # blocks: in exact arithmetic the constraint set would be
            # inconsistent.  For the feasible sets used here this only
            # happens at a primal-degenerate vertex where p is violated at
            # rounding level; accept it as satisfied when the violation is
            # below QP_DEGENERATE_TOL, else give up (reported, not raised).
            if abs(s_p) <= QP_DEGENERATE_TOL * (1.0 + abs(rhs[p])):
                skipped.append(p)
                p = -1
                continue
            break
        if not math.isfinite(t2):
            # Dependent normal: dual step only, drop the blocking constraint.
            u = [u[i] - t * r[i] for i in range(q)]
            u_p += t
            del active[j_block]
            del u[j_block]
            continue
        w = w + t * z
        u = [u[i] - t * r[i] for i in range(q)]
        u_p += t
        if t == t2:
            active.append(p)
            u.append(u_p)
            p = -1
        else:
            del active[j_block]
            del u[j_block]
    # Polish: the minimizer on the final active set in closed form,
    # w = S^-1 N u with (N' S^-1 N) u = b_active — one Cholesky solve —
    # removes the drift accumulated by the steps (the answer is then an
    # exact function of the active set, which is what makes the ports
    # agree to rounding).  Then active bounds sit exactly on their bound
    # and tolerated rounding-level violations of inactive bounds are clipped.
    q = len(active)
    if q > 0:
        ys = [chol_solve(low, normals[c]) for c in active]
        gram = np.zeros((q, q))
        for i in range(q):
            for j in range(i, q):
                gram[i, j] = float(normals[active[i]] @ ys[j])
                gram[j, i] = gram[i, j]
        low_g = cholesky(gram)
        b_act = np.array([rhs[c] for c in active])
        u_arr = chol_solve(low_g, b_act)
        w = np.zeros(n)
        for i in range(q):
            w = w + u_arr[i] * ys[i]
        # One step of iterative refinement on the equality residual.
        resid = b_act - np.array([float(normals[c] @ w) for c in active])
        du = chol_solve(low_g, resid)
        for i in range(q):
            w = w + du[i] * ys[i]
        u = [float(u_arr[i] + du[i]) for i in range(q)]
    for c in active:
        kind, i, _ = cons[c]
        if kind == 1 or kind == 3:
            w[i] = lo[i]
        elif kind == 2:
            w[i] = hi[i]
    # KKT certificate on the polished point: primal feasibility, and
    # stationarity S w = N u with u >= 0 on the inequalities (guaranteed by
    # construction).  ``converged`` is this certificate, independent of how
    # the loop terminated.
    feas = 0.0
    rhs_scale = 0.0
    for c in range(n_cons):
        viol = float(normals[c] @ w) - rhs[c]
        rhs_scale = max(rhs_scale, abs(rhs[c]))
        feas = max(feas, abs(viol) if c < meq else -viol)
    stat = s @ w
    stat_scale = float(np.max(np.abs(stat)))
    for i, c in enumerate(active):
        stat = stat - u[i] * normals[c]
        stat_scale += abs(u[i]) * float(np.max(np.abs(normals[c])))
    stat_res = float(np.max(np.abs(stat)))
    converged = (
        feas <= QP_CONV_TOL * (1.0 + rhs_scale) and stat_res <= QP_CONV_TOL * (1.0 + stat_scale)
    )
    w = np.clip(w, lo, hi)
    return QpResult(w, iterations, converged, max(feas, stat_res))


def _qp_setup(sigma: np.ndarray, lb, ub, max_iter: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = np.asarray(sigma, dtype=float)
    if s.ndim != 2 or s.shape[0] != s.shape[1]:
        raise ValueError(f"sigma must be square, got shape {s.shape}")
    cholesky(s)  # SPD/validity gate; raises ValueError on bad input
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    lo, hi = _broadcast_bounds(s.shape[0], lb, ub, 1.0)
    return s, lo, hi


def min_variance_longonly_solve(
    sigma: np.ndarray,
    lb: Optional[Sequence[float] | float] = None,
    ub: Optional[Sequence[float] | float] = None,
    max_iter: int = QP_MAX_ITER,
) -> QpResult:
    """Constrained minimum variance ``min 1/2 w'Sw`` over the box-simplex.

    Defaults to the long-only simplex (lb = 0, ub = +inf).  Solved by the
    pinned dual active-set QP (Goldfarb-Idnani).  The objective is strictly
    convex, so the minimizer is unique; the result is its KKT point up to
    rounding.
    """
    s, lo, hi = _qp_setup(sigma, lb, ub, max_iter)
    n = s.shape[0]
    return _dual_active_set_qp(s, np.ones((1, n)), np.array([1.0]), lo, hi, max_iter)


def _require_converged(res: QpResult, what: str) -> np.ndarray:
    if not res.converged:
        raise ValueError(
            f"{what}: active-set QP did not converge within {res.iterations} "
            f"iterations (KKT residual {res.kkt_residual:.3e}); the input is "
            "degenerate — use the *_solve variant to inspect the iterate"
        )
    return res.weights


def min_variance_longonly(
    sigma: np.ndarray,
    lb: Optional[Sequence[float] | float] = None,
    ub: Optional[Sequence[float] | float] = None,
    max_iter: int = QP_MAX_ITER,
) -> np.ndarray:
    """Weights of :func:`min_variance_longonly_solve`; raises ValueError if
    the solver did not certify convergence."""
    return _require_converged(
        min_variance_longonly_solve(sigma, lb, ub, max_iter), "min_variance_longonly"
    )


def _frontier_point(
    m: np.ndarray,
    s: np.ndarray,
    tgt: float,
    lo: np.ndarray,
    hi: np.ndarray,
    w_gmv: np.ndarray,
    r_gmv: float,
    r_min: float,
    r_max: float,
    max_iter: int,
) -> QpResult:
    tol = 1e-12 * (1.0 + abs(tgt))
    if tgt < r_gmv - tol:
        raise ValueError(
            f"target return {tgt:.6g} lies below the constrained minimum-variance "
            f"return {r_gmv:.6g}; the constrained efficient frontier is defined for "
            f"targets in [{r_gmv:.6g}, {r_max:.6g}]"
        )
    if tgt > r_max + tol:
        raise ValueError(
            f"target return {tgt:.6g} is not attainable under the bounds "
            f"(attainable range [{r_min:.6g}, {r_max:.6g}])"
        )
    if abs(tgt - r_gmv) <= tol or r_max - r_min <= tol:
        # At the minimum-variance return, or when every feasible portfolio
        # has the same expected return, the target row is redundant.
        return QpResult(w_gmv.copy(), 0, True, 0.0)
    n = m.shape[0]
    if abs(tgt - r_max) <= tol:
        # Maximum-return endpoint: the feasible face is degenerate (more
        # active constraints than coordinates), so reduce it analytically.
        # Any portfolio with mu'w = r_max holds every asset above the greedy
        # cut level at its cap and every asset below it at its floor; only
        # the assets tied at the cut level remain free.  The face problem
        # is then a min-variance QP with the sum constraint only.
        w_max = _greedy_extreme(m, lo, hi, 1.0, descending=True)
        filled = [float(m[i]) for i in range(n) if w_max[i] > lo[i]]
        if not filled:
            return QpResult(w_gmv.copy(), 0, True, 0.0)
        cut = min(filled)
        lo2 = lo.copy()
        hi2 = hi.copy()
        for i in range(n):
            if m[i] > cut:
                lo2[i] = hi[i]
            elif m[i] < cut:
                hi2[i] = lo[i]
        res = _dual_active_set_qp(s, np.ones((1, n)), np.array([1.0]), lo2, hi2, max_iter)
        return QpResult(res.weights, res.iterations, res.converged,
                        max(res.kkt_residual, abs(float(m @ res.weights) - tgt)))
    a_eq = np.vstack([np.ones(n), m])
    b_eq = np.array([1.0, tgt])
    return _dual_active_set_qp(s, a_eq, b_eq, lo, hi, max_iter)


def constrained_frontier_solve(
    mu: np.ndarray,
    sigma: np.ndarray,
    target: float,
    lb: Optional[Sequence[float] | float] = None,
    ub: Optional[Sequence[float] | float] = None,
    max_iter: int = QP_MAX_ITER,
) -> QpResult:
    """Constrained frontier point: ``min 1/2 w'Sw`` subject to
    ``sum w = 1``, ``mu'w = target`` and ``lb <= w <= ub``.

    The target is met exactly (to rounding) — there is no penalty term.
    Solved by the pinned dual active-set QP; the attainable range of
    :func:`attainable_return_range` is checked first.

    Raises ValueError if the target lies below the constrained
    minimum-variance return (inefficient branch) or above the maximum
    attainable return under the bounds.
    """
    m, _ = _validate_mu_sigma(mu, sigma)
    s, lo, hi = _qp_setup(sigma, lb, ub, max_iter)
    tgt = float(target)
    if not math.isfinite(tgt):
        raise ValueError("target return must be finite")
    gmv = min_variance_longonly_solve(s, lo, hi, max_iter)
    w_gmv = _require_converged(gmv, "constrained_frontier_solve (min-variance point)")
    r_min, r_max = attainable_return_range(m, lo, hi)
    return _frontier_point(m, s, tgt, lo, hi, w_gmv, float(m @ w_gmv), r_min, r_max, max_iter)


def constrained_frontier_weights(
    mu: np.ndarray,
    sigma: np.ndarray,
    target: float,
    lb: Optional[Sequence[float] | float] = None,
    ub: Optional[Sequence[float] | float] = None,
    max_iter: int = QP_MAX_ITER,
) -> np.ndarray:
    """Weights of :func:`constrained_frontier_solve`; raises ValueError if
    the solver did not certify convergence."""
    return _require_converged(
        constrained_frontier_solve(mu, sigma, target, lb, ub, max_iter),
        "constrained_frontier_weights",
    )


def constrained_frontier(
    mu: np.ndarray,
    sigma: np.ndarray,
    n_points: int = 25,
    lb: Optional[Sequence[float] | float] = None,
    ub: Optional[Sequence[float] | float] = None,
    max_iter: int = QP_MAX_ITER,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Constrained frontier sweep on the pinned grid from the constrained
    minimum-variance return to the maximum attainable return under the
    bounds (``n_points`` equally spaced targets, both ends inclusive).

    Every point is solved to KKT optimality; a non-converged point raises.
    Returns (targets, weights, variances); ``mu @ weights[k] == targets[k]``
    to rounding.
    """
    m, _ = _validate_mu_sigma(mu, sigma)
    if n_points < 2:
        raise ValueError("n_points must be >= 2")
    s, lo, hi = _qp_setup(sigma, lb, ub, max_iter)
    w_gmv = _require_converged(
        min_variance_longonly_solve(s, lo, hi, max_iter), "constrained_frontier (min-variance point)"
    )
    r_gmv = float(m @ w_gmv)
    r_min, r_max = attainable_return_range(m, lo, hi)
    targets = _frontier_grid(r_gmv, r_max, n_points, "maximum attainable return")
    weights = np.zeros((n_points, m.shape[0]), dtype=float)
    variances = np.zeros(n_points, dtype=float)
    for k, tgt in enumerate(targets):
        res = _frontier_point(m, s, float(tgt), lo, hi, w_gmv, r_gmv, r_min, r_max, max_iter)
        w = _require_converged(res, f"constrained_frontier point {k} (target {tgt:.6g})")
        weights[k] = w
        variances[k] = float(w @ s @ w)
    return targets, weights, variances
