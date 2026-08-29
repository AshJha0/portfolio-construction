"""Black-Litterman: reverse optimization and the posterior master formula.

Master formula (pinned form, API_SPEC section 4).  With prior
(equilibrium) returns pi, uncertainty scaling tau in (0, 1], pick matrix
P (k x n), view returns Q (k,), and view covariance Omega (k x k):

    A_k     = P (tau Sigma) P' + Omega                      (k x k, SPD)
    mu_post = pi + (tau Sigma) P' A_k^{-1} (Q - P pi)
    M       = tau Sigma - (tau Sigma) P' A_k^{-1} P (tau Sigma)
    S_post  = Sigma + M

This is algebraically identical to the textbook
``[(tau S)^-1 + P' O^-1 P]^-1 [(tau S)^-1 pi + P' O^-1 Q]`` form but
needs only one k x k Cholesky solve, which is the pinned implementation.

Limits (tested): tau -> 0 gives mu_post -> pi and M -> 0; Omega -> inf
(zero confidence) gives mu_post -> pi and M -> tau Sigma.

Omega default (Idzorek-style): ``Omega = diag(P (tau Sigma) P')`` scaled
per view by a confidence c_k > 0: ``Omega_kk = [P tau S P']_kk / c_k``
(c_k -> inf: dogmatic view; c_k -> 0: view ignored).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

from .linalg import check_square_symmetric, cholesky, chol_solve
from .markowitz import tangency_weights

__all__ = [
    "implied_equilibrium_returns",
    "views_to_matrices",
    "default_omega",
    "BLResult",
    "bl_posterior",
    "bl_weights",
]


def implied_equilibrium_returns(
    sigma: np.ndarray, w_mkt: np.ndarray, delta: float
) -> np.ndarray:
    """Reverse optimization: ``pi = delta * Sigma @ w_mkt``.

    ``w_mkt`` must be non-negative with positive sum; it is renormalized
    to sum to one before use.  ``delta`` (risk aversion) must be > 0.
    """
    s = check_square_symmetric(sigma)
    w = np.asarray(w_mkt, dtype=float).reshape(-1)
    if w.shape[0] != s.shape[0]:
        raise ValueError(
            f"w_mkt length {w.shape[0]} does not match sigma dimension {s.shape[0]}"
        )
    if not np.isfinite(w).all():
        raise ValueError("w_mkt contains NaN or infinite entries")
    if np.any(w < 0):
        raise ValueError("market-cap weights must be non-negative")
    tot = float(np.sum(w))
    if tot <= 0:
        raise ValueError("market-cap weights must have positive sum")
    d = float(delta)
    if not (math.isfinite(d) and d > 0):
        raise ValueError("risk aversion delta must be positive and finite")
    return d * (s @ (w / tot))


def views_to_matrices(
    views: Sequence[Mapping], asset_names: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (P, Q, confidences) from a list of view dicts.

    Supported view types (schema of data/views.json):

    * ``{"type": "absolute", "asset": name, "q": x, "confidence": c}`` —
      P row is the unit vector of ``asset``;
    * ``{"type": "relative", "long": name, "short": name, "q": x,
      "confidence": c}`` — P row is +1 on ``long`` and -1 on ``short``.

    ``confidence`` defaults to 1.0 and must be > 0.
    """
    names = list(asset_names)
    if len(views) == 0:
        raise ValueError("views list must be non-empty")
    k, n = len(views), len(names)
    p = np.zeros((k, n), dtype=float)
    q = np.zeros(k, dtype=float)
    conf = np.ones(k, dtype=float)
    for i, view in enumerate(views):
        vtype = view.get("type")
        if vtype == "absolute":
            p[i, names.index(view["asset"])] = 1.0
        elif vtype == "relative":
            p[i, names.index(view["long"])] = 1.0
            p[i, names.index(view["short"])] = -1.0
        else:
            raise ValueError(f"unknown view type: {vtype!r}")
        q[i] = float(view["q"])
        c = float(view.get("confidence", 1.0))
        if not (math.isfinite(c) and c > 0):
            raise ValueError(f"view {i}: confidence must be positive and finite")
        conf[i] = c
    return p, q, conf


def _validate_views(p: np.ndarray, n: int) -> np.ndarray:
    pm = np.asarray(p, dtype=float)
    if pm.ndim != 2 or pm.shape[1] != n:
        raise ValueError(f"P must be k x {n}, got shape {pm.shape}")
    if pm.shape[0] < 1:
        raise ValueError("P must contain at least one view row")
    if not np.isfinite(pm).all():
        raise ValueError("P contains NaN or infinite entries")
    for i in range(pm.shape[0]):
        if float(np.max(np.abs(pm[i]))) == 0.0:
            raise ValueError(f"view row {i} of P is identically zero")
    return pm


def _validate_tau(tau: float) -> float:
    t = float(tau)
    if not (math.isfinite(t) and 0.0 < t <= 1.0):
        raise ValueError(f"tau must lie in (0, 1], got {t}")
    return t


def default_omega(
    sigma: np.ndarray,
    p: np.ndarray,
    tau: float,
    confidences: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Idzorek-style default view covariance.

    ``Omega = diag(P (tau Sigma) P')`` with each diagonal entry divided by
    the corresponding per-view confidence ``c_k > 0`` (default 1).
    """
    s = check_square_symmetric(sigma)
    pm = _validate_views(p, s.shape[0])
    t = _validate_tau(tau)
    k = pm.shape[0]
    conf = np.ones(k) if confidences is None else np.asarray(confidences, dtype=float).reshape(-1)
    if conf.shape[0] != k:
        raise ValueError(f"confidences must have length {k}")
    if np.any(~np.isfinite(conf)) or np.any(conf <= 0):
        raise ValueError("confidences must be positive and finite")
    core = pm @ (t * s) @ pm.T
    omega = np.zeros((k, k), dtype=float)
    for i in range(k):
        omega[i, i] = core[i, i] / conf[i]
    return omega


@dataclass(frozen=True)
class BLResult:
    """Black-Litterman posterior.

    Attributes
    ----------
    mu_post:
        Posterior expected returns.
    sigma_post:
        Posterior return covariance ``Sigma + M``.
    m:
        Posterior covariance of the mean estimate,
        ``tau Sigma - tau Sigma P' A_k^{-1} P tau Sigma``.
    omega:
        The view covariance actually used.
    """

    mu_post: np.ndarray
    sigma_post: np.ndarray
    m: np.ndarray
    omega: np.ndarray


def bl_posterior(
    sigma: np.ndarray,
    pi: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    tau: float,
    omega: Optional[np.ndarray] = None,
    confidences: Optional[Sequence[float]] = None,
) -> BLResult:
    """Black-Litterman posterior via the pinned master formula.

    ``omega=None`` uses :func:`default_omega` (with optional per-view
    confidences).  The k x k system is solved column-wise with one
    Cholesky factorization of ``A_k = P tau Sigma P' + Omega``.
    """
    s = check_square_symmetric(sigma)
    n = s.shape[0]
    pi_v = np.asarray(pi, dtype=float).reshape(-1)
    if pi_v.shape[0] != n or not np.isfinite(pi_v).all():
        raise ValueError(f"pi must be a finite vector of length {n}")
    pm = _validate_views(p, n)
    k = pm.shape[0]
    qv = np.asarray(q, dtype=float).reshape(-1)
    if qv.shape[0] != k or not np.isfinite(qv).all():
        raise ValueError(f"Q must be a finite vector of length {k}")
    t = _validate_tau(tau)
    if omega is None:
        om = default_omega(s, pm, t, confidences)
    else:
        om = check_square_symmetric(omega)
        if om.shape[0] != k:
            raise ValueError(f"Omega must be {k} x {k}, got {om.shape}")

    ts = t * s
    b = pm @ ts  # k x n = P (tau Sigma)
    a_k = b @ pm.T + om
    low = cholesky(a_k)  # raises ValueError if Omega makes A_k non-SPD

    resid = qv - pm @ pi_v
    x = chol_solve(low, resid)
    mu_post = pi_v + b.T @ x

    # M = tau S - B' A_k^{-1} B, computed via k solves against B's rows.
    ainv_b = np.zeros_like(b)
    for j in range(n):
        ainv_b[:, j] = chol_solve(low, b[:, j])
    m_mat = ts - b.T @ ainv_b
    return BLResult(mu_post=mu_post, sigma_post=s + m_mat, m=m_mat, omega=om)


def bl_weights(
    sigma: np.ndarray,
    pi: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    tau: float,
    rf: float = 0.0,
    omega: Optional[np.ndarray] = None,
    confidences: Optional[Sequence[float]] = None,
) -> tuple[np.ndarray, BLResult]:
    """Posterior-based optimal weights.

    Pinned: unconstrained tangency portfolio computed from
    ``(mu_post, sigma_post)``.  Because ``pi = delta Sigma w_mkt`` are
    equilibrium *excess* returns, the pinned default is ``rf = 0`` (a
    non-zero rf would double-count the risk-free rate); with no views the
    tangency on ``(pi, Sigma)`` recovers exactly ``w_mkt``.
    Returns (weights, BLResult).
    """
    res = bl_posterior(sigma, pi, p, q, tau, omega=omega, confidences=confidences)
    w = tangency_weights(res.mu_post, res.sigma_post, rf)
    return w, res
