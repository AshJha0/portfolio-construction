"""Risk parity / equal risk contribution via cyclical coordinate descent.

CCD update derivation (Griveau-Billion, Richard & Roncalli 2013 style,
log-barrier formulation; pinned in API_SPEC section 5).  Solve

    min_{y > 0}  F(y) = 1/2 y' Sigma y - sum_i b_i ln y_i

for risk budgets b_i > 0 (sum normalized to 1).  Coordinate-wise, with
``m_i = (Sigma y)_i - Sigma_ii y_i`` (the cross term, held fixed):

    dF/dy_i = Sigma_ii y_i + m_i - b_i / y_i = 0
    =>  Sigma_ii y_i^2 + m_i y_i - b_i = 0
    =>  y_i = ( -m_i + sqrt(m_i^2 + 4 Sigma_ii b_i) ) / (2 Sigma_ii)

taking the positive root of the quadratic (the discriminant is > 0
because Sigma_ii > 0 and b_i > 0, so y_i stays strictly positive).  At a
stationary point, multiplying the first-order condition by y_i gives
``y_i (Sigma y)_i = b_i`` — risk contributions exactly proportional to
the budgets — and the normalized weights ``w = y / sum(y)`` satisfy
``RC_i(w) = b_i``.  F is strictly convex on the positive orthant, so the
solution is unique and CCD converges monotonically.

Sweep order pinned: coordinates i = 0, 1, ..., n-1 in ascending order;
after each single-coordinate update the cached product ``Sigma y`` is
updated before the next coordinate.  Convergence: after a full sweep,
``max_i |y_i - y_i_prev_sweep| < tol`` (default 1e-10); at most 10000
sweeps (non-convergence is reported via a flag, not an exception).
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from .linalg import check_square_symmetric, cholesky

__all__ = [
    "risk_contributions",
    "inverse_vol_weights",
    "risk_budget_solve",
    "risk_budget_weights",
    "erc_weights",
]

CCD_TOL = 1.0e-10
CCD_MAX_SWEEPS = 10_000


def risk_contributions(w: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Normalized risk contributions ``RC_i = w_i (Sigma w)_i / (w' Sigma w)``.

    The contributions sum to exactly 1 (portfolio variance normalized).
    Raises ValueError if the portfolio variance is not strictly positive.
    """
    s = check_square_symmetric(sigma)
    wv = np.asarray(w, dtype=float).reshape(-1)
    if wv.shape[0] != s.shape[0]:
        raise ValueError(
            f"weights length {wv.shape[0]} does not match sigma dimension {s.shape[0]}"
        )
    if not np.isfinite(wv).all():
        raise ValueError("weights contain NaN or infinite entries")
    sw = s @ wv
    var = float(wv @ sw)
    if var <= 0.0:
        raise ValueError("portfolio variance must be strictly positive")
    return wv * sw / var


def inverse_vol_weights(sigma: np.ndarray) -> np.ndarray:
    """Naive inverse-volatility weights ``w_i = (1/sqrt(S_ii)) / sum_j (1/sqrt(S_jj))``."""
    s = check_square_symmetric(sigma)
    d = np.diag(s)
    if np.any(d <= 0):
        raise ValueError("all variances must be strictly positive for inverse-vol")
    inv = 1.0 / np.sqrt(d)
    return inv / float(np.sum(inv))


def _validate_budgets(budgets: Optional[Sequence[float]], n: int) -> np.ndarray:
    if budgets is None:
        return np.full(n, 1.0 / n)
    b = np.asarray(budgets, dtype=float).reshape(-1)
    if b.shape[0] != n:
        raise ValueError(f"budgets must have length {n}")
    if np.any(~np.isfinite(b)) or np.any(b <= 0):
        raise ValueError("risk budgets must be positive and finite")
    return b / float(np.sum(b))


def risk_budget_solve(
    sigma: np.ndarray,
    budgets: Optional[Sequence[float]] = None,
    tol: float = CCD_TOL,
    max_sweeps: int = CCD_MAX_SWEEPS,
) -> tuple[np.ndarray, int, bool]:
    """General risk budgeting via pinned CCD (module docstring).

    Parameters
    ----------
    sigma:
        SPD covariance matrix (validated via Cholesky).
    budgets:
        Target risk budgets ``b_i > 0`` (normalized internally to sum 1);
        ``None`` means equal budgets (ERC).
    tol, max_sweeps:
        Sweep-level convergence tolerance and sweep cap (pinned defaults
        1e-10 / 10000).

    Returns
    -------
    (weights, sweeps, converged):
        Long-only weights on the simplex, number of full sweeps executed,
        and a convergence flag (non-convergence is reported, not raised).

    Notes
    -----
    Initialization is pinned to ``y_i = 1/n``.  An almost-zero-vol asset
    (tiny ``Sigma_ii``) yields a large ``y_i`` and hence a weight near 1
    after normalization — the correct ERC limit; convergence is preserved
    because the coordinate quadratic keeps a positive discriminant.
    """
    s = check_square_symmetric(sigma)
    cholesky(s)  # SPD gate: rejects singular / non-PSD inputs with a clear error
    n = s.shape[0]
    b = _validate_budgets(budgets, n)
    if not (math.isfinite(tol) and tol > 0):
        raise ValueError("tol must be positive")
    if max_sweeps < 1:
        raise ValueError("max_sweeps must be >= 1")

    y = np.full(n, 1.0 / n)
    sy = s @ y  # cached Sigma y, updated incrementally
    for sweep in range(1, max_sweeps + 1):
        y_prev = y.copy()
        for i in range(n):
            m_i = sy[i] - s[i, i] * y[i]
            y_new = (-m_i + math.sqrt(m_i * m_i + 4.0 * s[i, i] * b[i])) / (
                2.0 * s[i, i]
            )
            dy = y_new - y[i]
            if dy != 0.0:
                sy = sy + s[:, i] * dy
                y[i] = y_new
        if float(np.max(np.abs(y - y_prev))) < tol:
            return y / float(np.sum(y)), sweep, True
    return y / float(np.sum(y)), max_sweeps, False


def risk_budget_weights(
    sigma: np.ndarray,
    budgets: Optional[Sequence[float]] = None,
    tol: float = CCD_TOL,
    max_sweeps: int = CCD_MAX_SWEEPS,
) -> np.ndarray:
    """Risk-budget weights (weights only; see :func:`risk_budget_solve`)."""
    w, _, _ = risk_budget_solve(sigma, budgets, tol=tol, max_sweeps=max_sweeps)
    return w


def erc_weights(
    sigma: np.ndarray, tol: float = CCD_TOL, max_sweeps: int = CCD_MAX_SWEEPS
) -> np.ndarray:
    """Equal-risk-contribution weights (equal budgets b_i = 1/n)."""
    return risk_budget_weights(sigma, None, tol=tol, max_sweeps=max_sweeps)
