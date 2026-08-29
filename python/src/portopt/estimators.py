"""Return-moment estimators: sample moments and Ledoit-Wolf shrinkage.

Pinned conventions (see API_SPEC.md section 2):

* sample mean uses divisor T; sample covariance uses divisor T-1 (ddof=1);
* Ledoit-Wolf (2004) constant-correlation shrinkage with the classic
  closed-form intensity; the dispersion sums (pi/rho/gamma) use divisor T
  while S keeps divisor T-1 — this deliberate simplification is pinned so
  that every language port computes the identical number;
* annualization multiplies means and covariances by 252 (configurable).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .linalg import cholesky

__all__ = [
    "validate_returns",
    "sample_mean",
    "sample_cov",
    "ShrinkageResult",
    "shrink_covariance",
    "annualize_mean",
    "annualize_cov",
]


def validate_returns(returns: np.ndarray, min_rows: int = 2) -> np.ndarray:
    """Validate a T x N returns matrix (finite floats, T >= min_rows, N >= 1)."""
    x = np.ascontiguousarray(returns, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"returns must be a 2-D T x N matrix, got ndim={x.ndim}")
    t, n = x.shape
    if n < 1:
        raise ValueError("returns must contain at least one asset column")
    if t < min_rows:
        raise ValueError(f"returns must contain at least {min_rows} rows, got {t}")
    if not np.isfinite(x).all():
        raise ValueError("returns contain NaN or infinite entries")
    return x


def sample_mean(returns: np.ndarray) -> np.ndarray:
    """Per-asset sample mean: mu_i = (1/T) * sum_t X[t, i]."""
    x = validate_returns(returns, min_rows=1)
    return x.sum(axis=0) / x.shape[0]


def sample_cov(returns: np.ndarray) -> np.ndarray:
    """Sample covariance with ddof=1: S = Xc' Xc / (T-1), Xc demeaned columns."""
    x = validate_returns(returns, min_rows=2)
    t = x.shape[0]
    xc = x - sample_mean(x)
    return (xc.T @ xc) / (t - 1)


@dataclass(frozen=True)
class ShrinkageResult:
    """Output of Ledoit-Wolf constant-correlation shrinkage.

    Attributes
    ----------
    sigma:
        Shrunk covariance ``delta * F + (1 - delta) * S``.
    intensity:
        Shrinkage intensity ``delta`` in [0, 1].
    rbar:
        Average sample correlation used by the constant-correlation target.
    sample:
        Sample covariance S (ddof=1).
    target:
        Constant-correlation target F.
    """

    sigma: np.ndarray
    intensity: float
    rbar: float
    sample: np.ndarray
    target: np.ndarray


def shrink_covariance(returns: np.ndarray) -> ShrinkageResult:
    """Ledoit-Wolf (2004) shrinkage towards the constant-correlation target.

    Pinned term-by-term (API_SPEC section 2.2). With demeaned returns
    ``x[t,i]`` and S the ddof=1 sample covariance:

    * ``rbar = (2 / (N(N-1))) * sum_{i<j} S_ij / sqrt(S_ii S_jj)``
    * target ``F_ii = S_ii``, ``F_ij = rbar * sqrt(S_ii S_jj)``
    * ``pi_ij = (1/T) sum_t (x_ti x_tj - S_ij)^2``, ``pi_hat = sum_ij pi_ij``
    * ``theta_ii,ij = (1/T) sum_t (x_ti^2 - S_ii)(x_ti x_tj - S_ij)``
    * ``rho_hat = sum_i pi_ii + sum_{i != j} (rbar/2) *
      (sqrt(S_jj/S_ii) theta_ii,ij + sqrt(S_ii/S_jj) theta_jj,ji)``
    * ``gamma_hat = sum_ij (F_ij - S_ij)^2``
    * ``kappa = (pi_hat - rho_hat) / gamma_hat``;
      ``delta = max(0, min(1, kappa / T))`` (delta = 0 if gamma_hat == 0)

    Requires N >= 2 (rbar undefined for a single asset).
    """
    x = validate_returns(returns, min_rows=2)
    t, n = x.shape
    if n < 2:
        raise ValueError("shrinkage requires at least 2 assets")
    xc = x - sample_mean(x)
    s = (xc.T @ xc) / (t - 1)
    sd = np.sqrt(np.diag(s))
    if np.any(sd <= 0.0):
        raise ValueError("shrinkage requires strictly positive sample variances")

    corr_sum = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            corr_sum += s[i, j] / (sd[i] * sd[j])
    rbar = 2.0 * corr_sum / (n * (n - 1))

    f = np.outer(sd, sd) * rbar
    np.fill_diagonal(f, np.diag(s))

    # Cross products y[t, i, j] = x_ti * x_tj handled column-pair-wise.
    pi_hat = 0.0
    rho_hat = 0.0
    gamma_hat = 0.0
    # pi_mat[i, j] = (1/T) sum_t (x_ti x_tj - S_ij)^2
    for i in range(n):
        for j in range(n):
            prod = xc[:, i] * xc[:, j]
            dev = prod - s[i, j]
            pij = float(dev @ dev) / t
            pi_hat += pij
            if i == j:
                rho_hat += pij
            gamma_hat += (f[i, j] - s[i, j]) ** 2
    # theta terms for i != j
    for i in range(n):
        xi2_dev = xc[:, i] * xc[:, i] - s[i, i]
        for j in range(n):
            if i == j:
                continue
            dev_ij = xc[:, i] * xc[:, j] - s[i, j]
            theta_ii_ij = float(xi2_dev @ dev_ij) / t
            rho_hat += (rbar / 2.0) * math.sqrt(s[j, j] / s[i, i]) * theta_ii_ij
            # The symmetric partner sqrt(S_ii/S_jj)*theta_jj,ij is produced
            # by the (j, i) pass of this same loop, so each unordered pair
            # contributes both terms exactly once.

    if gamma_hat <= 0.0:
        delta = 0.0
    else:
        kappa = (pi_hat - rho_hat) / gamma_hat
        delta = max(0.0, min(1.0, kappa / t))

    sigma = delta * f + (1.0 - delta) * s
    return ShrinkageResult(sigma=sigma, intensity=delta, rbar=rbar, sample=s, target=f)


def annualize_mean(mu: np.ndarray, periods: int = 252) -> np.ndarray:
    """Annualize a per-period mean vector: ``mu_ann = periods * mu``."""
    m = np.asarray(mu, dtype=float)
    if periods <= 0:
        raise ValueError("periods must be positive")
    return periods * m


def annualize_cov(cov: np.ndarray, periods: int = 252) -> np.ndarray:
    """Annualize a per-period covariance: ``Sigma_ann = periods * Sigma``."""
    if periods <= 0:
        raise ValueError("periods must be positive")
    c = np.asarray(cov, dtype=float)
    return periods * c


def _psd_check(sigma: np.ndarray) -> None:
    """Internal helper: raise ValueError if sigma fails Cholesky."""
    cholesky(sigma)
