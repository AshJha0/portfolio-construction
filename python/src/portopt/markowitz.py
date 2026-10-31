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
rejected when its gross leverage ``sum |w_i|`` exceeds a cap, default
``TANGENCY_MAX_LEVERAGE = 10``, see :func:`tangency_weights`).

Constrained problems (sum-to-one, optional target return, box bounds)
are strictly convex quadratic programs; they are solved by the finite
dual active-set method of Goldfarb & Idnani (1983) whose every linear
solve is the pinned Cholesky (API_SPEC.md section 3.6).  The result is
the exact KKT point of the QP up to rounding, so all language ports agree
to ~1e-13 and the ``converged`` flag is a statement about the KKT
conditions, not about an iterate step size.

Cost-aware (turnover-penalised) problems ``min 1/2 w'Sw - c'w +
kappa sum|w - w0|`` over the same box-simplex, optionally with a hard
turnover cap, are solved through the same dual active-set method on an
exactly equivalent strictly convex split problem in ``(w, b, s)`` (see
:func:`min_variance_turnover_solve`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .linalg import chol_solve, cholesky

# Box bounds: a scalar, a per-asset sequence or array, or None for the default.
Bounds = Sequence[float] | float | np.ndarray | None

__all__ = [
    "TangencyUndefinedError",
    "TwoFundScalars",
    "QpResult",
    "TurnoverQpResult",
    "QP_MAX_ITER",
    "TANGENCY_MAX_LEVERAGE",
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
    "min_variance_turnover_solve",
    "min_variance_turnover",
    "mean_variance_turnover_solve",
    "mean_variance_turnover",
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
#: stationarity residual |S w - N u| <= QP_CONV_TOL * (1 + scale), and every
#: active bound multiplier u_c >= -QP_CONV_TOL * (1 + scale) (same scale).
QP_CONV_TOL = 1.0e-9
#: Pinned default cap on the gross leverage ``sum |w_i|`` of the
#: unconstrained tangency portfolio; above it the portfolio is rejected
#: as ill-conditioned (API_SPEC 3.3).
TANGENCY_MAX_LEVERAGE = 10.0


class TangencyUndefinedError(ValueError):
    """Tangency portfolio undefined (``A - rf C <= 0``) or ill-conditioned
    (gross leverage above the cap, default ``TANGENCY_MAX_LEVERAGE``)."""


def _validate_mu_sigma(mu: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.asarray(mu, dtype=float).reshape(-1)
    s = np.asarray(sigma, dtype=float)
    if s.ndim != 2 or s.shape[0] != s.shape[1]:
        raise ValueError(f"sigma must be square, got shape {s.shape}")
    if m.shape[0] != s.shape[0]:
        raise ValueError(f"mu length {m.shape[0]} does not match sigma dimension {s.shape[0]}")
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


def tangency_weights(
    mu: np.ndarray,
    sigma: np.ndarray,
    rf: float,
    max_leverage: float = TANGENCY_MAX_LEVERAGE,
) -> np.ndarray:
    """Tangency (max-Sharpe) weights ``S^-1 (mu - rf 1) / (A - rf C)``.

    Parameters
    ----------
    max_leverage:
        Cap on the gross leverage ``sum |w_i|`` of the result (>= 1;
        ``math.inf`` disables it).  Default ``TANGENCY_MAX_LEVERAGE = 10``.

    Raises
    ------
    TangencyUndefinedError
        If ``A - rf*C <= 0``: the tangency portfolio is undefined when the
        risk-free rate is at or above the minimum-variance return ``A/C``
        (in particular when rf exceeds every expected return).  Also when
        the gross leverage of the weights exceeds ``max_leverage``: the
        weights scale like ``1/(A - rf C)``, so as rf approaches ``A/C``
        from below they blow up (94x on the bundled backtest at a gap of
        0.07%), and the number is then an artefact of the estimate of
        ``A/C`` rather than an allocation.
    ValueError
        On malformed inputs, a non-SPD covariance, or ``max_leverage``
        below 1 or NaN.
    """
    m, s = _validate_mu_sigma(mu, sigma)
    rfv = float(rf)
    if not math.isfinite(rfv):
        raise ValueError("rf must be finite")
    cap = float(max_leverage)
    if math.isnan(cap) or cap < 1.0:
        raise ValueError(f"max_leverage must be >= 1 (got {cap})")
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
    w: np.ndarray = z / denom
    lev = gross_leverage(w)
    if lev > cap:
        raise TangencyUndefinedError(
            f"tangency ill-conditioned: gross leverage {lev:.4g} exceeds max_leverage "
            f"{cap:g} (A - rf*C = {denom:.3e}; rf is too close below the "
            "minimum-variance return A/C)"
        )
    return w


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
            raise ValueError("n=1: target return must equal the single asset's expected return")
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
    lb: Bounds,
    ub: Bounds,
    total: float,
) -> tuple[np.ndarray, np.ndarray]:
    lo = np.zeros(n) if lb is None else np.broadcast_to(np.asarray(lb, dtype=float), (n,)).copy()
    hi = (
        np.full(n, np.inf)
        if ub is None
        else np.broadcast_to(np.asarray(ub, dtype=float), (n,)).copy()
    )
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
        raise ValueError(f"infeasible bounds: sum(lb) = {lo_sum:.6g} exceeds total {total:.6g}")
    if hi_sum < total - feas_tol:
        raise ValueError(f"infeasible bounds: sum(ub) = {hi_sum:.6g} is below total {total:.6g}")
    return lo, hi


def project_simplex_box(
    v: np.ndarray,
    total: float = 1.0,
    lb: Bounds = None,
    ub: Bounds = None,
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
        return float(
            math.fsum(min(max(float(x[i] - t), float(lo[i])), float(hi[i])) for i in range(n))
        )

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
            out: np.ndarray = np.clip(x - t_first, lo, hi)
            return out
        out = np.clip(x - (t_first - (s - g_first) / k), lo, hi)
        return out

    t_prev, g_prev = t_first, g_first
    for t_cur in bps[1:]:
        g_cur = g(t_cur)
        if g_cur <= s:
            if g_prev == g_cur:
                t_star = t_cur
            else:
                t_star = t_prev + (g_prev - s) * (t_cur - t_prev) / (g_prev - g_cur)
            out = np.clip(x - t_star, lo, hi)
            return out
        t_prev, g_prev = t_cur, g_cur
    # Right of every breakpoint g(t) = sum(lb) <= s; feasibility means equality.
    out = np.clip(x - bps[-1], lo, hi)
    return out


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
        primal feasibility, stationarity ``S w = N u`` and dual
        feasibility (every active bound multiplier ``>= 0``), each within
        ``QP_CONV_TOL`` (relative; see :func:`_kkt_certificate`).  False means the iteration cap was hit or the
        constraint set proved numerically inconsistent — the returned
        weights are then not certified optimal.
    kkt_residual:
        Largest KKT violation at ``weights``: max of the equality
        residuals, bound violations, the stationarity residual
        ``|S w - N u|_inf`` (N = active normals, u = multipliers) and the
        dual-feasibility violation ``max(0, -min u_bound)``.
    """

    weights: np.ndarray
    iterations: int
    converged: bool
    kkt_residual: float


@dataclass(frozen=True)
class TurnoverQpResult:
    """Output of the cost-aware (turnover-penalised) QP solvers.

    Attributes
    ----------
    weights:
        Feasible weights (bounds exact, budget to rounding).
    iterations:
        Dual active-set iterations on the split problem.
    converged:
        True iff the KKT conditions of the original nonsmooth problem are
        certified at ``weights``: primal feasibility including the turnover
        cap, stationarity with an L1 subgradient, cap multiplier >= 0 and
        complementarity (see :func:`_turnover_certificate`).
    kkt_residual:
        Largest violation found by that certificate.
    turnover:
        ``sum_i |w_i - w0_i|`` of the returned weights (correctly rounded).
    """

    weights: np.ndarray
    iterations: int
    converged: bool
    kkt_residual: float
    turnover: float


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
    lb: Bounds = None,
    ub: Bounds = None,
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


def _constraint_normal(
    n: int, a_eq: np.ndarray, a_in: np.ndarray | None, kind: int, idx: int
) -> np.ndarray:
    if kind == 0:
        row: np.ndarray = a_eq[idx].copy()
        return row
    if kind == 4:
        assert a_in is not None
        row = a_in[idx].copy()
        return row
    e = np.zeros(n)
    e[idx] = -1.0 if kind == 2 else 1.0
    return e


def _kkt_certificate(
    s: np.ndarray,
    normals: Sequence[np.ndarray],
    rhs: np.ndarray,
    meq: int,
    active: Sequence[int],
    u: Sequence[float],
    w: np.ndarray,
) -> tuple[bool, float]:
    """KKT certificate of ``min 1/2 w'Sw`` s.t. ``n_c'w >= b_c`` (the first
    ``meq`` constraints are equalities) at ``w`` with active set ``active``
    and multipliers ``u`` (``u[i]`` belongs to constraint ``active[i]``).

    Checks primal feasibility (equality residuals and inequality
    violations ``<= QP_CONV_TOL * (1 + max|b|)``), stationarity
    ``|S w - N u|_inf <= QP_CONV_TOL * (1 + stat_scale)`` and dual
    feasibility ``u_i >= -QP_CONV_TOL * (1 + stat_scale)`` for every active
    inequality, where ``stat_scale = |S w|_inf + sum_i |u_i| |n_i|_inf`` (a
    multiplier has the units of the gradient, so it shares the stationarity
    scale).  Returns ``(certified, residual)`` with ``residual`` the largest
    of the feasibility, stationarity and dual-feasibility violations.
    """
    feas = 0.0
    rhs_scale = 0.0
    for c in range(len(normals)):
        viol = float(normals[c] @ w) - rhs[c]
        rhs_scale = max(rhs_scale, abs(rhs[c]))
        feas = max(feas, abs(viol) if c < meq else -viol)
    stat = s @ w
    stat_scale = float(np.max(np.abs(stat)))
    dual = 0.0
    for i, c in enumerate(active):
        stat = stat - u[i] * normals[c]
        stat_scale += abs(u[i]) * float(np.max(np.abs(normals[c])))
        if c >= meq:
            dual = max(dual, -u[i])
    stat_res = float(np.max(np.abs(stat)))
    stat_tol = QP_CONV_TOL * (1.0 + stat_scale)
    certified = (
        feas <= QP_CONV_TOL * (1.0 + rhs_scale) and stat_res <= stat_tol and dual <= stat_tol
    )
    return certified, max(feas, stat_res, dual)


@dataclass
class _GiState:
    """Final state of the dual active-set loop (internal)."""

    w: np.ndarray
    active: list[int]
    u: list[float]
    iterations: int
    cons: list[tuple[int, int, float]]
    normals: list[np.ndarray]
    rhs: np.ndarray
    meq: int
    low: np.ndarray


def _gi_loop(
    s: np.ndarray,
    lin: np.ndarray | None,
    a_eq: np.ndarray,
    b_eq: np.ndarray,
    a_in: np.ndarray | None,
    b_in: np.ndarray | None,
    lo: np.ndarray,
    hi: np.ndarray,
    max_iter: int,
) -> _GiState:
    """Goldfarb-Idnani loop for ``min 1/2 x'Sx - lin'x`` subject to
    ``a_eq x = b_eq``, ``a_in x >= b_in`` and ``lo <= x <= hi``.

    ``lin = None`` means no linear term (start at ``x = 0``); with a linear
    term the method starts at the unconstrained minimizer ``S^-1 lin`` and
    is otherwise unchanged (the step directions do not depend on ``lin``).
    General inequality rows come after the bounds in the constraint order.
    """
    n = s.shape[0]
    low = cholesky(s)
    # Constraint list: (kind, idx, rhs); kind 0 = equality row of a_eq,
    # 3 = fixed coordinate (lo_i == hi_i, treated as the equality w_i = lo_i),
    # 1 = lower bound, 2 = upper bound, 4 = inequality row of a_in.
    # Equalities come first.
    cons: list[tuple[int, int, float]] = [(0, j, float(b_eq[j])) for j in range(a_eq.shape[0])]
    cons += [(3, i, float(lo[i])) for i in range(n) if lo[i] == hi[i]]
    meq = len(cons)
    cons += [(1, i, float(lo[i])) for i in range(n) if lo[i] != hi[i]]
    cons += [(2, i, -float(hi[i])) for i in range(n) if lo[i] != hi[i] and math.isfinite(hi[i])]
    if a_in is not None and b_in is not None:
        cons += [(4, j, float(b_in[j])) for j in range(a_in.shape[0])]
    normals = [_constraint_normal(n, a_eq, a_in, k, i) for (k, i, _) in cons]
    rhs = np.array([c[2] for c in cons])
    n_cons = len(cons)

    w = np.zeros(n) if lin is None else chol_solve(low, lin)
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
    return _GiState(w, active, u, iterations, cons, normals, rhs, meq, low)


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
    st = _gi_loop(s, None, a_eq, b_eq, None, None, lo, hi, max_iter)
    n = s.shape[0]
    low, cons, normals, rhs, meq = st.low, st.cons, st.normals, st.rhs, st.meq
    w, active, u, iterations = st.w, st.active, st.u, st.iterations
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
    # KKT certificate on the polished point (``converged`` is this
    # certificate, independent of how the loop terminated).
    converged, kkt_residual = _kkt_certificate(s, normals, rhs, meq, active, u, w)
    w = np.clip(w, lo, hi)
    return QpResult(w, iterations, converged, kkt_residual)


def _qp_setup(
    sigma: np.ndarray, lb: Bounds, ub: Bounds, max_iter: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    lb: Bounds = None,
    ub: Bounds = None,
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
    lb: Bounds = None,
    ub: Bounds = None,
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
        return QpResult(
            res.weights,
            res.iterations,
            res.converged,
            max(res.kkt_residual, abs(float(m @ res.weights) - tgt)),
        )
    a_eq = np.vstack([np.ones(n), m])
    b_eq = np.array([1.0, tgt])
    return _dual_active_set_qp(s, a_eq, b_eq, lo, hi, max_iter)


def constrained_frontier_solve(
    mu: np.ndarray,
    sigma: np.ndarray,
    target: float,
    lb: Bounds = None,
    ub: Bounds = None,
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
    lb: Bounds = None,
    ub: Bounds = None,
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
    lb: Bounds = None,
    ub: Bounds = None,
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
        min_variance_longonly_solve(s, lo, hi, max_iter),
        "constrained_frontier (min-variance point)",
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


# --------------------------------------------------------- cost-aware QP


def _turnover_certificate(
    s: np.ndarray,
    c: np.ndarray,
    w0: np.ndarray,
    kappa: float,
    lo: np.ndarray,
    hi: np.ndarray,
    cap: float,
    nu: float,
    tau: float,
    w: np.ndarray,
) -> tuple[bool, float]:
    """KKT certificate of ``min 1/2 w'Sw - c'w + kappa sum|w - w0|`` over
    ``{sum w = 1, lo <= w <= hi, sum|w - w0| <= cap}`` at ``w`` with budget
    multiplier ``nu`` and cap multiplier ``tau`` (API_SPEC 3.7).

    With ``k = kappa + tau`` and ``g = S w - c - nu 1`` the conditions are:
    ``-g_i`` lies in the subdifferential ``[sL_i, sR_i]`` of
    ``k |w_i - w0_i|`` plus the indicator of ``[lo_i, hi_i]`` at ``w_i``
    (``sL = -inf`` at the lower bound, else ``-k`` at or left of ``w0_i``
    and ``+k`` right of it; ``sR`` symmetrically), ``tau >= 0``, and
    ``tau > 0`` only with the cap binding.  Primal feasibility is checked to
    ``QP_CONV_TOL * (1 + rhs_scale)``, the dual conditions to
    ``QP_CONV_TOL * (1 + |S w|_inf + |c|_inf + |nu| + k)``; a coordinate
    counts as at a breakpoint when within ``QP_CONV_TOL * (1 + |bp|)`` of
    it.  Returns ``(certified, residual)``.
    """
    n = w.shape[0]
    k = kappa + tau
    rhs_scale = 1.0
    for i in range(n):
        rhs_scale = max(rhs_scale, abs(float(w0[i])), abs(float(lo[i])))
        if math.isfinite(hi[i]):
            rhs_scale = max(rhs_scale, abs(float(hi[i])))
    if math.isfinite(cap):
        rhs_scale = max(rhs_scale, cap)
    turnover = math.fsum(abs(float(w[i]) - float(w0[i])) for i in range(n))
    feas = abs(math.fsum(float(v) for v in w) - 1.0)
    for i in range(n):
        feas = max(feas, float(lo[i]) - float(w[i]), float(w[i]) - float(hi[i]))
    if math.isfinite(cap):
        feas = max(feas, turnover - cap)
    sw = s @ w
    stat_scale = abs(nu) + k
    c_max = 0.0
    sw_max = 0.0
    for i in range(n):
        sw_max = max(sw_max, abs(float(sw[i])))
        c_max = max(c_max, abs(float(c[i])))
    stat_scale += sw_max + c_max
    stat_tol = QP_CONV_TOL * (1.0 + stat_scale)
    dual = max(0.0, -tau)
    if tau > stat_tol and math.isfinite(cap):
        # Complementarity: a positive cap multiplier needs a binding cap.
        feas = max(feas, cap - turnover)
    for i in range(n):
        wi = float(w[i])
        d = wi - float(w0[i])
        at_lo = wi <= float(lo[i]) + QP_CONV_TOL * (1.0 + abs(float(lo[i])))
        at_hi = math.isfinite(hi[i]) and wi >= float(hi[i]) - QP_CONV_TOL * (
            1.0 + abs(float(hi[i]))
        )
        at_w0 = abs(d) <= QP_CONV_TOL * (1.0 + abs(float(w0[i])))
        xi = -(float(sw[i]) - float(c[i]) - nu)
        if not at_lo:
            s_left = -k if (at_w0 or d < 0.0) else k
            dual = max(dual, s_left - xi)
        if not at_hi:
            s_right = k if (at_w0 or d > 0.0) else -k
            dual = max(dual, xi - s_right)
    certified = feas <= QP_CONV_TOL * (1.0 + rhs_scale) and dual <= stat_tol
    return certified, max(feas, dual)


def _turnover_qp(
    s: np.ndarray,
    c: np.ndarray,
    w0: np.ndarray,
    kappa: float,
    lo: np.ndarray,
    hi: np.ndarray,
    cap: float,
    max_iter: int,
) -> TurnoverQpResult:
    """Pinned cost-aware QP (API_SPEC 3.7); inputs already validated."""
    n = s.shape[0]
    if n == 1:
        return TurnoverQpResult(np.array([1.0]), 0, True, 0.0, abs(1.0 - float(w0[0])))
    low = cholesky(s)
    # Proximal weight eps = 0.5 / trace(S^-1) <= lambda_min(S) / 2, so
    # S - eps I stays SPD.  The split problem below is exactly equivalent
    # to the original one for any 0 < eps < lambda_min(S).
    tr = 0.0
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        tr += float(chol_solve(low, e)[j])
    eps = 0.5 / tr
    m = 3 * n
    h = np.zeros((m, m))
    h[:n, :n] = s
    for i in range(n):
        h[i, i] = s[i, i] - eps
        h[n + i, n + i] = eps
        h[2 * n + i, 2 * n + i] = eps
    lin = np.zeros(m)
    for i in range(n):
        lin[i] = c[i] - eps * w0[i]
        lin[n + i] = -kappa
        lin[2 * n + i] = -kappa
    a_eq = np.zeros((n + 1, m))
    b_eq = np.zeros(n + 1)
    a_eq[0, :n] = 1.0
    b_eq[0] = 1.0
    for i in range(n):
        a_eq[1 + i, i] = 1.0
        a_eq[1 + i, n + i] = -1.0
        a_eq[1 + i, 2 * n + i] = 1.0
        b_eq[1 + i] = w0[i]
    lo3 = np.concatenate([lo, np.zeros(2 * n)])
    hi3 = np.concatenate([hi, np.full(2 * n, np.inf)])
    a_in: np.ndarray | None = None
    b_in: np.ndarray | None = None
    if math.isfinite(cap):
        a_in = np.zeros((1, m))
        a_in[0, n:] = -1.0
        b_in = np.array([-cap])
    st = _gi_loop(h, lin, a_eq, b_eq, a_in, b_in, lo3, hi3, max_iter)

    # States from the active set: a coordinate is fixed at a bound (bound
    # active, or lo == hi), held at w0 (b >= 0 and s >= 0 both active), or
    # free on the buy side (+1, s >= 0 active) or the sell side (-1,
    # b >= 0 active); with neither active the sign of b - s decides.
    at_lo = [bool(lo[i] == hi[i]) for i in range(n)]
    at_hi = [False] * n
    b_act = [False] * n
    s_act = [False] * n
    cap_act = False
    for cidx in st.active:
        kind, idx, _ = st.cons[cidx]
        if kind == 4:
            cap_act = True
        elif kind == 2:
            at_hi[idx] = True
        elif kind in (1, 3):
            if idx < n:
                at_lo[idx] = True
            elif idx < 2 * n:
                b_act[idx - n] = True
            else:
                s_act[idx - 2 * n] = True
    is_fixed = [False] * n
    fix_val = np.zeros(n)
    sgn = np.zeros(n)
    for i in range(n):
        if at_lo[i]:
            is_fixed[i], fix_val[i] = True, lo[i]
        elif at_hi[i]:
            is_fixed[i], fix_val[i] = True, hi[i]
        elif b_act[i] and s_act[i]:
            is_fixed[i], fix_val[i] = True, w0[i]
        elif s_act[i]:
            sgn[i] = 1.0
        elif b_act[i]:
            sgn[i] = -1.0
        else:
            sgn[i] = 1.0 if st.w[n + i] - st.w[2 * n + i] >= 0.0 else -1.0

    # Exact polish in w alone: minimise 1/2 w'Sw - (c - kappa sgn)'w on the
    # face {sum w = 1, [cap row], w_i = fix_i}, in closed form.
    normals: list[np.ndarray] = [np.ones(n)]
    rhs: list[float] = [1.0]
    if cap_act:
        b_cap = -cap
        for i in range(n):
            if is_fixed[i]:
                b_cap += abs(float(fix_val[i]) - float(w0[i]))
            else:
                b_cap -= float(sgn[i]) * float(w0[i])
        normals.append(-sgn)
        rhs.append(b_cap)
    for i in range(n):
        if is_fixed[i]:
            e = np.zeros(n)
            e[i] = 1.0
            normals.append(e)
            rhs.append(float(fix_val[i]))
    c_eff = c - kappa * sgn
    x0 = chol_solve(low, c_eff)
    q = len(normals)
    ys = [chol_solve(low, nr) for nr in normals]
    gram = np.zeros((q, q))
    for i in range(q):
        for j in range(i, q):
            gram[i, j] = float(normals[i] @ ys[j])
            gram[j, i] = gram[i, j]
    try:
        low_g = cholesky(gram)
    except ValueError:
        # Dependent face rows: the active set cannot be polished, so the
        # loop iterate is returned uncertified.
        w_bad = np.clip(st.w[:n].copy(), lo, hi)
        to_bad = math.fsum(abs(float(w_bad[i]) - float(w0[i])) for i in range(n))
        return TurnoverQpResult(w_bad, st.iterations, False, math.inf, to_bad)
    b_vec = np.array(rhs)
    u_arr = chol_solve(low_g, b_vec - np.array([float(nr @ x0) for nr in normals]))
    w = x0.copy()
    for i in range(q):
        w = w + u_arr[i] * ys[i]
    # One step of iterative refinement on the face residual.
    resid = b_vec - np.array([float(nr @ w) for nr in normals])
    du = chol_solve(low_g, resid)
    for i in range(q):
        w = w + du[i] * ys[i]
    for i in range(n):
        if is_fixed[i]:
            w[i] = fix_val[i]
    nu = float(u_arr[0] + du[0])
    tau = float(u_arr[1] + du[1]) if cap_act else 0.0
    converged, kkt_residual = _turnover_certificate(s, c, w0, kappa, lo, hi, cap, nu, tau, w)
    w = np.clip(w, lo, hi)
    turnover = math.fsum(abs(float(w[i]) - float(w0[i])) for i in range(n))
    return TurnoverQpResult(w, st.iterations, converged, kkt_residual, turnover)


def _turnover_setup(
    sigma: np.ndarray,
    w0: np.ndarray,
    kappa: float,
    lb: Bounds,
    ub: Bounds,
    max_turnover: float | None,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, float]:
    s, lo, hi = _qp_setup(sigma, lb, ub, max_iter)
    n = s.shape[0]
    w0v = np.asarray(w0, dtype=float).reshape(-1)
    if w0v.shape[0] != n:
        raise ValueError(f"w0 length {w0v.shape[0]} does not match sigma dimension {n}")
    if not np.isfinite(w0v).all():
        raise ValueError("w0 contains NaN or infinite entries")
    k = float(kappa)
    if not math.isfinite(k) or k < 0.0:
        raise ValueError(f"kappa must be finite and >= 0 (got {k})")
    cap = math.inf if max_turnover is None else float(max_turnover)
    if math.isnan(cap) or cap < 0.0:
        raise ValueError(f"max_turnover must be >= 0 (got {cap})")
    if math.isfinite(cap):
        # Smallest attainable turnover: clip w0 into the box, then move the
        # budget gap |1 - sum(clip)|; no feasible portfolio does better.
        wc = np.clip(w0v, lo, hi)
        to_min = math.fsum(abs(float(wc[i]) - float(w0v[i])) for i in range(n)) + abs(
            1.0 - math.fsum(float(v) for v in wc)
        )
        if cap < to_min - 1e-12 * (1.0 + to_min):
            raise ValueError(
                f"infeasible turnover cap: max_turnover = {cap:.6g} is below the "
                f"smallest attainable turnover {to_min:.6g}"
            )
    return s, w0v, k, lo, hi, cap


def min_variance_turnover_solve(
    sigma: np.ndarray,
    w0: np.ndarray,
    kappa: float,
    lb: Bounds = None,
    ub: Bounds = None,
    max_turnover: float | None = None,
    max_iter: int = QP_MAX_ITER,
) -> TurnoverQpResult:
    """Cost-aware minimum variance (API_SPEC 3.7)::

        min  1/2 w'Sw + kappa * sum_i |w_i - w0_i|
        s.t. sum w = 1,  lb <= w <= ub,  [sum_i |w_i - w0_i| <= max_turnover]

    ``w0`` is the current portfolio (any finite vector) and ``kappa >= 0``
    the cost per unit of two-sided turnover in the objective's units
    (variance; a cost rate ``c`` in return units against a risk aversion
    ``gamma`` maps to ``kappa = c / gamma``).  ``max_turnover = None`` means
    no cap.  Defaults: long-only (lb = 0, ub = +inf).

    Method: the trade is split as ``w = w0 + b - s`` with ``b, s >= 0`` and
    the dual active-set QP of 3.6 runs on the 3n-variable problem::

        min 1/2 w'(S - eps I)w + eps w0'w - c'w
            + eps/2 (b'b + s's) + kappa 1'(b + s)
        s.t. sum w = 1,  w - b + s = w0,  lb <= w <= ub,  b, s >= 0,
             [1'(b + s) <= max_turnover]

    with ``eps = 0.5 / trace(S^-1)`` (so ``S - eps I`` is SPD).  That
    problem is strictly convex, and minimising it over ``(b, s)`` at fixed
    ``w`` gives exactly ``b = (w - w0)+``, ``s = (w - w0)-`` with
    ``eps/2 (b'b + s's) = eps/2 |w - w0|^2``, which cancels the ``-eps``
    shift, so the minimum over ``(b, s)`` is the original objective and
    the minimiser is the exact solution: the proximal term changes the
    conditioning, not the answer.  The weights
    are then re-polished in ``w`` alone on the identified face and
    certified against the original problem's KKT conditions.

    Raises ValueError on invalid input, infeasible bounds, or a cap below
    the smallest attainable turnover.  With ``kappa = 0`` and no cap the
    result equals :func:`min_variance_longonly` to rounding.
    """
    s, w0v, k, lo, hi, cap = _turnover_setup(sigma, w0, kappa, lb, ub, max_turnover, max_iter)
    return _turnover_qp(s, np.zeros(s.shape[0]), w0v, k, lo, hi, cap, max_iter)


def _require_turnover_converged(res: TurnoverQpResult, what: str) -> np.ndarray:
    if not res.converged:
        raise ValueError(
            f"{what}: cost-aware QP not certified after {res.iterations} iterations "
            f"(KKT residual {res.kkt_residual:.3e}); the input is degenerate, use "
            "the *_solve variant to inspect the iterate"
        )
    return res.weights


def min_variance_turnover(
    sigma: np.ndarray,
    w0: np.ndarray,
    kappa: float,
    lb: Bounds = None,
    ub: Bounds = None,
    max_turnover: float | None = None,
    max_iter: int = QP_MAX_ITER,
) -> np.ndarray:
    """Weights of :func:`min_variance_turnover_solve`; raises ValueError if
    the KKT certificate fails."""
    return _require_turnover_converged(
        min_variance_turnover_solve(sigma, w0, kappa, lb, ub, max_turnover, max_iter),
        "min_variance_turnover",
    )


def mean_variance_turnover_solve(
    mu: np.ndarray,
    sigma: np.ndarray,
    w0: np.ndarray,
    risk_aversion: float,
    kappa: float,
    lb: Bounds = None,
    ub: Bounds = None,
    max_turnover: float | None = None,
    max_iter: int = QP_MAX_ITER,
) -> TurnoverQpResult:
    """Cost-aware mean-variance (API_SPEC 3.7)::

        min  1/2 w'Sw - mu'w / risk_aversion + kappa * sum_i |w_i - w0_i|
        s.t. sum w = 1,  lb <= w <= ub,  [sum_i |w_i - w0_i| <= max_turnover]

    This is ``max mu'w - (gamma/2) w'Sw - gamma kappa TO`` divided by
    ``gamma = risk_aversion > 0``.  Same method and certificate as
    :func:`min_variance_turnover_solve`.
    """
    m, _ = _validate_mu_sigma(mu, sigma)
    ra = float(risk_aversion)
    if not math.isfinite(ra) or ra <= 0.0:
        raise ValueError(f"risk_aversion must be finite and > 0 (got {ra})")
    s, w0v, k, lo, hi, cap = _turnover_setup(sigma, w0, kappa, lb, ub, max_turnover, max_iter)
    return _turnover_qp(s, m / ra, w0v, k, lo, hi, cap, max_iter)


def mean_variance_turnover(
    mu: np.ndarray,
    sigma: np.ndarray,
    w0: np.ndarray,
    risk_aversion: float,
    kappa: float,
    lb: Bounds = None,
    ub: Bounds = None,
    max_turnover: float | None = None,
    max_iter: int = QP_MAX_ITER,
) -> np.ndarray:
    """Weights of :func:`mean_variance_turnover_solve`; raises ValueError if
    the KKT certificate fails."""
    return _require_turnover_converged(
        mean_variance_turnover_solve(
            mu, sigma, w0, risk_aversion, kappa, lb, ub, max_turnover, max_iter
        ),
        "mean_variance_turnover",
    )
