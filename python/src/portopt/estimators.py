"""Return-moment estimators: sample moments and Ledoit-Wolf shrinkage.

Pinned conventions (see API_SPEC.md section 2):

* sample mean uses divisor T; sample covariance uses divisor T-1 (ddof=1);
* Ledoit-Wolf (2004) constant-correlation shrinkage: the intensity delta
  is computed exactly as in the paper's Appendix B / the authors'
  ``covCor.m`` (divisor T throughout, both theta terms per ordered pair),
  and applied to the ddof=1 sample covariance and its constant-correlation
  target — see :func:`shrink_covariance` for the exact statement;
* annualization multiplies means and covariances by 252 (configurable).

Reductions over the T-length sample use ``math.fsum`` (correctly rounded),
so the reference values do not depend on the BLAS numpy is linked against;
ports using plain sequential or compensated sums agree to ~1e-13 relative.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

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
    """Validate a T x N simple-returns matrix.

    Requires finite entries strictly greater than -1 (a simple return at
    or below -100% is not a valid price move), T >= ``min_rows`` and
    N >= 1.
    """
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
    if np.any(x <= -1.0):
        raise ValueError("returns contain a simple return <= -1 (-100% or worse)")
    return x


def _validate_periods(periods: float) -> float:
    p = float(periods)
    if not (math.isfinite(p) and p > 0.0):
        raise ValueError("periods must be positive and finite")
    return p


def sample_mean(returns: np.ndarray) -> np.ndarray:
    """Per-asset sample mean: mu_i = (1/T) * sum_t X[t, i] (correctly rounded sum)."""
    x = validate_returns(returns, min_rows=1)
    t, n = x.shape
    return np.array([math.fsum(x[:, i].tolist()) / t for i in range(n)])


def _demeaned(x: np.ndarray) -> np.ndarray:
    return x - sample_mean(x)


def _cov_from_demeaned(xc: np.ndarray) -> np.ndarray:
    t, n = xc.shape
    s = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            v = math.fsum((xc[:, i] * xc[:, j]).tolist()) / (t - 1)
            s[i, j] = v
            s[j, i] = v
    return s


def sample_cov(returns: np.ndarray) -> np.ndarray:
    """Sample covariance with ddof=1: S = Xc' Xc / (T-1), Xc demeaned columns."""
    x = validate_returns(returns, min_rows=2)
    return _cov_from_demeaned(_demeaned(x))


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

    Pinned term by term to Ledoit & Wolf (2004), *Journal of Portfolio
    Management* 30(4), Appendix B — equivalently the authors' ``covCor.m``.
    With demeaned returns ``x[t,i]`` and ``S_T = Xc'Xc / T`` (the paper's
    sample covariance, divisor T), ``s_ii = S_T[i,i]``:

    * ``rbar = (2 / (N(N-1))) * sum_{i<j} S_T_ij / sqrt(s_ii s_jj)``
    * ``pi_ij = (1/T) sum_t (x_ti x_tj - S_T_ij)^2``, ``pi_hat = sum_ij pi_ij``
    * ``theta_ii,ij = (1/T) sum_t (x_ti^2 - s_ii)(x_ti x_tj - S_T_ij)``
    * ``rho_hat = sum_i pi_ii + sum_{i != j} rbar * sqrt(s_jj / s_ii) * theta_ii,ij``
      (each unordered pair contributes both
      ``sqrt(s_jj/s_ii) theta_ii,ij`` and ``sqrt(s_ii/s_jj) theta_jj,ij``,
      weighted ``rbar``: the paper's ``(rbar/2)[...]`` summed over ordered
      pairs)
    * ``gamma_hat = sum_ij (F_T_ij - S_T_ij)^2`` with
      ``F_T_ii = s_ii``, ``F_T_ij = rbar sqrt(s_ii s_jj)``
    * ``kappa = (pi_hat - rho_hat) / gamma_hat``;
      ``delta = max(0, min(1, kappa / T))`` (delta = 0 if gamma_hat <= 0)

    The intensity ``delta`` is therefore exactly the paper's.  The returned
    matrices use the project-wide ddof=1 convention (API_SPEC 2.1):
    ``sample = S = T/(T-1) S_T``, ``target = F = T/(T-1) F_T`` and
    ``sigma = delta F + (1 - delta) S``, i.e. the paper's estimator rescaled
    by ``T/(T-1)``; ``rbar`` and ``delta`` are scale-free so they are
    unchanged by the convention.

    Requires N >= 2 (rbar undefined for a single asset) and strictly
    positive sample variances.
    """
    x = validate_returns(returns, min_rows=2)
    t, n = x.shape
    if n < 2:
        raise ValueError("shrinkage requires at least 2 assets")
    xc = _demeaned(x)
    s = _cov_from_demeaned(xc)
    if np.any(np.diag(s) <= 0.0):
        raise ValueError("shrinkage requires strictly positive sample variances")
    # Paper convention (divisor T) for every dispersion quantity.
    s_t = s * ((t - 1) / t)
    var_t = np.diag(s_t)
    sd_t = np.sqrt(var_t)

    corr_sum = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            corr_sum += s_t[i, j] / (sd_t[i] * sd_t[j])
    rbar = 2.0 * corr_sum / (n * (n - 1))

    f_t = np.outer(sd_t, sd_t) * rbar
    np.fill_diagonal(f_t, var_t)

    pi_hat = 0.0
    rho_hat = 0.0
    gamma_hat = 0.0
    for i in range(n):
        xi2_dev = xc[:, i] * xc[:, i] - var_t[i]
        for j in range(n):
            dev_ij = xc[:, i] * xc[:, j] - s_t[i, j]
            pij = math.fsum((dev_ij * dev_ij).tolist()) / t
            pi_hat += pij
            gd = f_t[i, j] - s_t[i, j]
            gamma_hat += gd * gd
            if i == j:
                rho_hat += pij
            else:
                theta_ii_ij = math.fsum((xi2_dev * dev_ij).tolist()) / t
                rho_hat += rbar * math.sqrt(var_t[j] / var_t[i]) * theta_ii_ij

    if gamma_hat <= 0.0:
        delta = 0.0
    else:
        kappa = (pi_hat - rho_hat) / gamma_hat
        delta = max(0.0, min(1.0, kappa / t))

    sd = np.sqrt(np.diag(s))
    f = np.outer(sd, sd) * rbar
    np.fill_diagonal(f, np.diag(s))
    sigma = delta * f + (1.0 - delta) * s
    return ShrinkageResult(sigma=sigma, intensity=delta, rbar=rbar, sample=s, target=f)


def annualize_mean(mu: np.ndarray, periods: float = 252) -> np.ndarray:
    """Annualize a per-period mean vector: ``mu_ann = periods * mu``."""
    p = _validate_periods(periods)
    m = np.asarray(mu, dtype=float)
    if not np.isfinite(m).all():
        raise ValueError("mu contains NaN or infinite entries")
    return p * m


def annualize_cov(cov: np.ndarray, periods: float = 252) -> np.ndarray:
    """Annualize a per-period covariance: ``Sigma_ann = periods * Sigma``."""
    p = _validate_periods(periods)
    c = np.asarray(cov, dtype=float)
    if not np.isfinite(c).all():
        raise ValueError("cov contains NaN or infinite entries")
    return p * c
