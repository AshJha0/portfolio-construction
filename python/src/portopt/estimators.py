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
The one exception is the default :func:`shrink_covariance`, which is
vectorised (numpy pairwise sums, still BLAS-free) and is tested against
the pinned loop version ``_shrink_covariance_reference`` to 1e-13
relative.
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
    out: np.ndarray = x - sample_mean(x)
    return out


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


def _shrink_covariance_reference(returns: np.ndarray) -> ShrinkageResult:
    """Pinned loop implementation of :func:`shrink_covariance`.

    Every T-length reduction is a correctly rounded ``math.fsum`` taken in
    the same order as the ports' loops. It is the specification the
    vectorised default is tested against (1e-13 relative) and is kept
    private only so that it is not mistaken for the fast entry point.
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


def _cov_from_demeaned_rows(xc: np.ndarray) -> np.ndarray:
    """Same values as :func:`_cov_from_demeaned`, bit for bit, with fewer
    Python-level calls: each ``math.fsum`` sees the same element-wise
    products (multiplication is commutative in IEEE 754), and a correctly
    rounded sum does not depend on how the terms are batched."""
    t, n = xc.shape
    xt = np.ascontiguousarray(xc.T)
    s = np.zeros((n, n), dtype=float)
    for i in range(n):
        prods = (xt[i:] * xt[i]).tolist()
        row = np.array([math.fsum(p) for p in prods]) / (t - 1)
        s[i, i:] = row
        s[i:, i] = row
    return s


def _sequential_sum(values: np.ndarray) -> float:
    """Left-to-right sum in row-major order, the order of the reference's
    ``+=`` loops (``np.cumsum`` accumulates sequentially, unlike
    ``np.sum``, which is pairwise)."""
    flat = np.ravel(values)
    if flat.size == 0:
        return 0.0
    return float(np.cumsum(flat)[-1])


# Size, in doubles, of the (i, j, t) tiles evaluated per numpy call. Small
# enough that each temporary (512 KiB) stays in a core's L2 cache: the
# dispersion sums are memory-bound, and tiles of 8 MiB ran three to four
# times slower at N = 50.
_TILE_ELEMS = 1 << 16

# Up to this many assets (the project's design range, see linalg) the
# vectorised shrinkage keeps the correctly rounded sample covariance, so
# its ``sample`` is bit-identical to :func:`sample_cov`. Above it the
# O(N^2 T) ``math.fsum`` work would dominate the run time, and numpy's
# pairwise sums are used instead (about 1e-16 relative to the largest
# entry, still independent of the BLAS).
_EXACT_SAMPLE_MAX_N = 20


def _tiles(n: int, t: int) -> list[tuple[int, int, int, int]]:
    """``(i0, i1, j0, j1)`` tiles covering the N x N pairs, each with at most
    about ``_TILE_ELEMS`` (i, j, t) entries: whole rows grouped when a row
    fits, otherwise one row split into column chunks."""
    cols = max(1, _TILE_ELEMS // t)
    if cols >= n:
        rows = max(1, _TILE_ELEMS // (n * t))
        return [(i0, min(n, i0 + rows), 0, n) for i0 in range(0, n, rows)]
    return [(i, i + 1, j0, min(n, j0 + cols)) for i in range(n) for j0 in range(0, n, cols)]


def _cov_pairwise(xt: np.ndarray) -> np.ndarray:
    """``Xc'Xc / (T-1)`` from the N x T transposed demeaned returns with
    numpy's pairwise sums along T (exactly symmetric: x_i x_j and x_j x_i
    are the same products, reduced in the same order)."""
    n, t = xt.shape
    s = np.empty((n, n), dtype=float)
    for i0, i1, j0, j1 in _tiles(n, t):
        prods = xt[i0:i1, None, :] * xt[None, j0:j1, :]
        s[i0:i1, j0:j1] = np.sum(prods, axis=-1) / (t - 1)
    return s


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

    Implementation note (vectorised default): the T-length dispersion
    sums inside ``pi_ij`` and ``theta_ii,ij`` are numpy pairwise sums
    rather than ``math.fsum``, so ``intensity`` and ``sigma`` agree with
    the pinned loop version ``_shrink_covariance_reference`` to about
    1e-15 relative. For N <= 20 (the bundled data and every golden case)
    ``sample``, ``target`` and ``rbar`` are bit-identical to it, because the
    covariance sums are the same correctly rounded ``math.fsum`` and
    ``rbar``/``gamma_hat`` are summed in the loop order; for larger N the
    covariance sums are pairwise too (about 1e-16 relative). The test suite
    pins the agreement at 1e-13 relative on the bundled data and on random
    problems up to N = 100.
    """
    x = validate_returns(returns, min_rows=2)
    t, n = x.shape
    if n < 2:
        raise ValueError("shrinkage requires at least 2 assets")
    xc = _demeaned(x)
    xt = np.ascontiguousarray(xc.T)
    s = _cov_from_demeaned_rows(xc) if n <= _EXACT_SAMPLE_MAX_N else _cov_pairwise(xt)
    if np.any(np.diag(s) <= 0.0):
        raise ValueError("shrinkage requires strictly positive sample variances")
    # Paper convention (divisor T) for every dispersion quantity.
    s_t = s * ((t - 1) / t)
    var_t = np.diag(s_t).copy()
    sd_t = np.sqrt(var_t)

    iu = np.triu_indices(n, 1)
    rbar = 2.0 * _sequential_sum((s_t / np.outer(sd_t, sd_t))[iu]) / (n * (n - 1))

    f_t = np.outer(sd_t, sd_t) * rbar
    np.fill_diagonal(f_t, var_t)

    # pi_ij and theta_ii,ij for all (i, j), in tiles of the (i, j, t) array
    # dev[i, j, t] = x_ti x_tj - S_T_ij (t is the contiguous axis, so the
    # reductions below are numpy's pairwise sums).
    xi2_dev = xt * xt - var_t[:, None]
    pi = np.empty((n, n), dtype=float)
    theta = np.empty((n, n), dtype=float)
    for i0, i1, j0, j1 in _tiles(n, t):
        dev = xt[i0:i1, None, :] * xt[None, j0:j1, :]
        dev -= s_t[i0:i1, j0:j1, None]
        pi[i0:i1, j0:j1] = np.sum(dev * dev, axis=-1) / t
        dev *= xi2_dev[i0:i1, None, :]
        theta[i0:i1, j0:j1] = np.sum(dev, axis=-1) / t

    pi_hat = _sequential_sum(pi)
    rho_terms = (rbar * np.sqrt(var_t[None, :] / var_t[:, None])) * theta
    np.fill_diagonal(rho_terms, np.diag(pi))
    rho_hat = _sequential_sum(rho_terms)
    gd = f_t - s_t
    gamma_hat = _sequential_sum(gd * gd)

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
