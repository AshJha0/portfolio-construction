"""Pinned dense linear algebra kernels for small SPD systems (n <= 20).

Every routine here is the *reference algorithm* that scipy-free ports in
C++/Rust/Java must reproduce bit-for-bit up to IEEE-754 rounding:

* Cholesky: Cholesky-Banachiewicz, lower-triangular, row by row, with a
  pinned relative pivot threshold (a numerically singular matrix such as
  a duplicated-asset covariance yields a pivot that is a tiny rounding
  residual instead of an exact zero, so the guard must be relative).
* SPD solve: forward substitution with L, then back substitution with L^T.
* SPD inverse: column-by-column Cholesky solves against identity columns.

numpy is used only as an array container; all loops are explicit so the
operation order is pinned.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "check_square_symmetric",
    "cholesky",
    "chol_solve",
    "spd_solve",
    "spd_inverse",
]

_SYM_TOL = 1e-10
# Pinned relative pivot threshold: pivot s at row i must exceed
# PIVOT_RTOL * a[i][i] (a[i][i] > 0 required). Rejects matrices with an
# effective condition number beyond ~1e13 — far outside anything a valid
# n <= 20 covariance produces, but safely catching duplicated assets.
PIVOT_RTOL = 1e-13


def check_square_symmetric(a: np.ndarray) -> np.ndarray:
    """Validate that ``a`` is a finite, square, symmetric matrix.

    Parameters
    ----------
    a:
        Candidate matrix (anything array-like coercible to float).

    Returns
    -------
    numpy.ndarray
        The validated matrix as a C-contiguous float64 array.

    Raises
    ------
    ValueError
        If ``a`` is not 2-D square, contains NaN/inf, or the maximum
        absolute asymmetry exceeds ``1e-10 * (1 + max|a|)``.
    """
    m = np.ascontiguousarray(a, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"matrix must be square 2-D, got shape {m.shape}")
    if m.size == 0:
        raise ValueError("matrix must be non-empty")
    if not np.isfinite(m).all():
        raise ValueError("matrix contains NaN or infinite entries")
    scale = 1.0 + float(np.max(np.abs(m)))
    asym = float(np.max(np.abs(m - m.T)))
    if asym > _SYM_TOL * scale:
        raise ValueError(f"matrix is not symmetric (max asymmetry {asym:.3e} exceeds tolerance)")
    return m


def cholesky(a: np.ndarray) -> np.ndarray:
    """Lower-triangular Cholesky factor L with ``a = L @ L.T``.

    Pinned algorithm (Cholesky-Banachiewicz): for rows i = 0..n-1 and
    columns j = 0..i, accumulate ``s = a[i,j] - sum_{k<j} L[i,k] L[j,k]``;
    the diagonal pivot is ``L[j,j] = sqrt(s)`` and off-diagonals are
    ``L[i,j] = s / L[j,j]``.

    Raises
    ------
    ValueError
        If the matrix fails symmetry validation, any diagonal entry is
        ``<= 0``, or any pivot argument satisfies
        ``s <= PIVOT_RTOL * a[i,i]`` (matrix not positive definite to
        working precision; this is the pinned detector for singular or
        indefinite covariance inputs, e.g. a duplicated asset).
    """
    m = check_square_symmetric(a)
    n = m.shape[0]
    low = np.zeros((n, n), dtype=float)
    for i in range(n):
        if m[i, i] <= 0.0:
            raise ValueError(
                f"matrix is not positive definite (diagonal entry {m[i, i]:.3e} <= 0 at row {i})"
            )
        for j in range(i + 1):
            s = m[i, j]
            for k in range(j):
                s -= low[i, k] * low[j, k]
            if i == j:
                if s <= PIVOT_RTOL * m[i, i]:
                    raise ValueError(
                        "matrix is not positive definite "
                        f"(Cholesky pivot {s:.3e} <= {PIVOT_RTOL:g} * diag at row {i}); "
                        "covariance may be singular (e.g. duplicated asset)"
                    )
                low[i, j] = math.sqrt(s)
            else:
                low[i, j] = s / low[j, j]
    return low


def chol_solve(low: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve ``(L @ L.T) x = b`` given the lower Cholesky factor ``low``.

    Pinned: forward substitution ``L y = b`` (ascending rows), then back
    substitution ``L.T x = y`` (descending rows).
    """
    lmat = np.asarray(low, dtype=float)
    rhs = np.asarray(b, dtype=float)
    n = lmat.shape[0]
    if rhs.shape != (n,):
        raise ValueError(f"rhs must have shape ({n},), got {rhs.shape}")
    y = np.zeros(n, dtype=float)
    for i in range(n):
        s = rhs[i]
        for k in range(i):
            s -= lmat[i, k] * y[k]
        y[i] = s / lmat[i, i]
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        s = y[i]
        for k in range(i + 1, n):
            s -= lmat[k, i] * x[k]
        x[i] = s / lmat[i, i]
    return x


def spd_solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve ``a x = b`` for symmetric positive definite ``a`` via Cholesky."""
    return chol_solve(cholesky(a), np.asarray(b, dtype=float))


def spd_inverse(a: np.ndarray) -> np.ndarray:
    """Inverse of an SPD matrix via column-by-column Cholesky solves.

    Pinned: factor once, then solve ``A x_j = e_j`` for unit vectors
    ``e_j``, j = 0..n-1, stacking the solutions as columns.  No
    post-symmetrization is applied.
    """
    low = cholesky(a)
    n = low.shape[0]
    inv = np.zeros((n, n), dtype=float)
    for j in range(n):
        e = np.zeros(n, dtype=float)
        e[j] = 1.0
        inv[:, j] = chol_solve(low, e)
    return inv
