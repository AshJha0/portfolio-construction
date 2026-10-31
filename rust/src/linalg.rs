//! Pinned dense linear algebra kernels for small SPD systems (n <= 20).
//!
//! Every routine here mirrors the Python reference algorithm exactly
//! (API_SPEC.md section 3):
//!
//! * Cholesky: Cholesky-Banachiewicz, lower triangular, computed row by
//!   row, with a pinned *relative* pivot threshold (a numerically singular
//!   matrix such as a duplicated-asset covariance yields a tiny rounding
//!   residual instead of an exact zero pivot, so the guard must be
//!   relative).
//! * SPD solve: forward substitution with `L`, then back substitution
//!   with `L'`.
//! * SPD inverse: column-by-column Cholesky solves against identity
//!   columns (no post-symmetrization).
//!
//! Beyond the factorization kernels, this module provides [`kahan_sum`]
//! (Neumaier-compensated summation) for every T-length reduction: the
//! Python reference sums with correctly rounded `math.fsum`, and the
//! compensated sum agrees with it to a couple of ulps, so no golden value
//! depends on a BLAS or on the summation order. The Cholesky recurrences
//! use plain left-to-right sums to mirror the Python loops operation for
//! operation.

use crate::error::{PortOptError, Result};

/// Symmetry gate tolerance: `max|A - A'| <= 1e-10 * (1 + max|A|)`.
const SYM_TOL: f64 = 1e-10;

/// Pinned relative pivot threshold: the pivot argument `s` at row `i`
/// must exceed `PIVOT_RTOL * A[i][i]`. Rejects matrices with an effective
/// condition number beyond ~1e13 — far outside anything a valid n <= 20
/// covariance produces, but safely catching duplicated assets.
pub const PIVOT_RTOL: f64 = 1e-13;

/// Dense row-major matrix of `f64` (small: n <= 20 for square inputs,
/// T x N for return panels).
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    /// Zero matrix of the given shape (both dimensions must be > 0).
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Identity matrix of order `n`.
    pub fn identity(n: usize) -> Self {
        let mut m = Matrix::zeros(n, n);
        for i in 0..n {
            m.set(i, i, 1.0);
        }
        m
    }

    /// Build from a slice of equally sized rows.
    ///
    /// # Errors
    /// [`PortOptError::InvalidInput`] on an empty matrix or ragged rows.
    pub fn from_rows(rows: &[Vec<f64>]) -> Result<Self> {
        if rows.is_empty() || rows[0].is_empty() {
            return Err(PortOptError::InvalidInput(
                "matrix must be non-empty".into(),
            ));
        }
        let cols = rows[0].len();
        let mut data = Vec::with_capacity(rows.len() * cols);
        for (i, r) in rows.iter().enumerate() {
            if r.len() != cols {
                return Err(PortOptError::InvalidInput(format!(
                    "ragged matrix: row {i} has {} entries, expected {cols}",
                    r.len()
                )));
            }
            data.extend_from_slice(r);
        }
        Ok(Matrix {
            rows: rows.len(),
            cols,
            data,
        })
    }

    /// Square matrix from a flat row-major slice of length `n*n`.
    pub fn from_flat(n: usize, data: &[f64]) -> Result<Self> {
        if data.len() != n * n {
            return Err(PortOptError::InvalidInput(format!(
                "flat data length {} does not match {n}x{n}",
                data.len()
            )));
        }
        Ok(Matrix {
            rows: n,
            cols: n,
            data: data.to_vec(),
        })
    }

    /// Diagonal matrix from a slice.
    pub fn from_diag(d: &[f64]) -> Self {
        let n = d.len();
        let mut m = Matrix::zeros(n, n);
        for (i, &v) in d.iter().enumerate() {
            m.set(i, i, v);
        }
        m
    }

    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.cols
    }

    /// Entry at `(i, j)` (row, column). Out-of-range indices are a
    /// programming error and panic like slice indexing (debug assertion
    /// plus the underlying bounds check).
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        debug_assert!(i < self.rows && j < self.cols);
        self.data[i * self.cols + j]
    }

    /// Set entry at `(i, j)`.
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        debug_assert!(i < self.rows && j < self.cols);
        self.data[i * self.cols + j] = v;
    }

    /// Row `i` as a slice.
    pub fn row(&self, i: usize) -> &[f64] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    /// Column `j` copied into a new vector.
    pub fn col(&self, j: usize) -> Vec<f64> {
        (0..self.rows).map(|i| self.get(i, j)).collect()
    }

    /// True when every entry is finite.
    pub fn is_finite(&self) -> bool {
        self.data.iter().all(|v| v.is_finite())
    }

    /// Matrix-vector product `A v` with plain left-to-right row sums.
    ///
    /// # Panics
    /// If `v.len() != ncols()` — a shape mismatch is a programming error,
    /// not invalid data; every public API validates shapes before calling
    /// this helper and returns `Err` instead.
    pub fn matvec(&self, v: &[f64]) -> Vec<f64> {
        assert_eq!(v.len(), self.cols, "matvec dimension mismatch");
        (0..self.rows).map(|i| dot(self.row(i), v)).collect()
    }

    /// Scale every entry by `s`, returning a new matrix.
    pub fn scaled(&self, s: f64) -> Matrix {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|v| s * v).collect(),
        }
    }
}

/// Plain left-to-right dot product (used for short n <= 20 vectors,
/// mirroring the reference's explicit loops).
#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0;
    for k in 0..a.len() {
        s += a[k] * b[k];
    }
    s
}

/// Neumaier-compensated sum for long (T-length) accumulations.
///
/// Matches the reference's correctly rounded `math.fsum` to within a
/// couple of ulps (an empty iterator sums to 0).
pub fn kahan_sum<I: IntoIterator<Item = f64>>(xs: I) -> f64 {
    let mut sum = 0.0;
    let mut c = 0.0;
    for x in xs {
        let t = sum + x;
        if sum.abs() >= x.abs() {
            c += (sum - t) + x;
        } else {
            c += (x - t) + sum;
        }
        sum = t;
    }
    sum + c
}

/// Validate that `a` is a finite, square, symmetric matrix.
///
/// # Errors
/// [`PortOptError::InvalidInput`] if `a` is not square, contains NaN/inf,
/// or the maximum absolute asymmetry exceeds `1e-10 * (1 + max|a|)`.
pub fn check_square_symmetric(a: &Matrix) -> Result<()> {
    if a.nrows() != a.ncols() {
        return Err(PortOptError::InvalidInput(format!(
            "matrix must be square, got shape {}x{}",
            a.nrows(),
            a.ncols()
        )));
    }
    if a.nrows() == 0 {
        return Err(PortOptError::InvalidInput(
            "matrix must be non-empty".into(),
        ));
    }
    if !a.is_finite() {
        return Err(PortOptError::InvalidInput(
            "matrix contains NaN or infinite entries".into(),
        ));
    }
    let n = a.nrows();
    let mut max_abs = 0.0_f64;
    let mut asym = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            max_abs = max_abs.max(a.get(i, j).abs());
            asym = asym.max((a.get(i, j) - a.get(j, i)).abs());
        }
    }
    if asym > SYM_TOL * (1.0 + max_abs) {
        return Err(PortOptError::InvalidInput(format!(
            "matrix is not symmetric (max asymmetry {asym:.3e} exceeds tolerance)"
        )));
    }
    Ok(())
}

/// Lower-triangular Cholesky factor `L` with `a = L L'`.
///
/// Pinned algorithm (Cholesky-Banachiewicz): for rows `i = 0..n-1` and
/// columns `j = 0..i`, accumulate `s = a[i][j] - sum_{k<j} L[i][k] L[j][k]`;
/// the diagonal pivot is `L[i][i] = sqrt(s)` and off-diagonals are
/// `L[i][j] = s / L[j][j]`.
///
/// # Errors
/// [`PortOptError::InvalidInput`] if the symmetry gate fails;
/// [`PortOptError::NotPositiveDefinite`] if a diagonal entry is `<= 0` or
/// a pivot argument satisfies `s <= PIVOT_RTOL * a[i][i]` (the pinned
/// detector for singular or indefinite covariance inputs, e.g. a
/// duplicated asset).
pub fn cholesky(a: &Matrix) -> Result<Matrix> {
    check_square_symmetric(a)?;
    let n = a.nrows();
    let mut low = Matrix::zeros(n, n);
    for i in 0..n {
        let aii = a.get(i, i);
        if aii <= 0.0 {
            return Err(PortOptError::NotPositiveDefinite(format!(
                "matrix is not positive definite (diagonal entry {aii:.3e} <= 0 at row {i})"
            )));
        }
        for j in 0..=i {
            let mut s = a.get(i, j);
            for k in 0..j {
                s -= low.get(i, k) * low.get(j, k);
            }
            if i == j {
                if s <= PIVOT_RTOL * aii {
                    return Err(PortOptError::NotPositiveDefinite(format!(
                        "matrix is not positive definite (Cholesky pivot {s:.3e} <= \
                         {PIVOT_RTOL:e} * diag at row {i}); covariance may be singular \
                         (e.g. duplicated asset)"
                    )));
                }
                low.set(i, j, s.sqrt());
            } else {
                low.set(i, j, s / low.get(j, j));
            }
        }
    }
    Ok(low)
}

/// Solve `(L L') x = b` given the lower Cholesky factor `low`.
///
/// Pinned: forward substitution `L y = b` (ascending rows), then back
/// substitution `L' x = y` (descending rows).
///
/// # Errors
/// [`PortOptError::InvalidInput`] if `b` does not match the factor's order.
#[allow(clippy::needless_range_loop)] // index loops mirror the Python reference; kept verbatim so evaluation order stays pinned to the goldens
pub fn chol_solve(low: &Matrix, b: &[f64]) -> Result<Vec<f64>> {
    let n = low.nrows();
    if b.len() != n {
        return Err(PortOptError::InvalidInput(format!(
            "rhs must have length {n}, got {}",
            b.len()
        )));
    }
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= low.get(i, k) * y[k];
        }
        y[i] = s / low.get(i, i);
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in (i + 1)..n {
            s -= low.get(k, i) * x[k];
        }
        x[i] = s / low.get(i, i);
    }
    Ok(x)
}

/// Solve `a x = b` for symmetric positive definite `a` via Cholesky.
pub fn spd_solve(a: &Matrix, b: &[f64]) -> Result<Vec<f64>> {
    let low = cholesky(a)?;
    chol_solve(&low, b)
}

/// Inverse of an SPD matrix via column-by-column Cholesky solves.
///
/// Pinned: factor once, then solve `A x_j = e_j` for unit vectors `e_j`,
/// `j = 0..n-1`, stacking the solutions as columns. No post-symmetrization.
#[allow(clippy::needless_range_loop)] // index loops mirror the Python reference; kept verbatim so evaluation order stays pinned to the goldens
pub fn spd_inverse(a: &Matrix) -> Result<Matrix> {
    let low = cholesky(a)?;
    let n = low.nrows();
    let mut inv = Matrix::zeros(n, n);
    for j in 0..n {
        let mut e = vec![0.0; n];
        e[j] = 1.0;
        let x = chol_solve(&low, &e)?;
        for i in 0..n {
            inv.set(i, j, x[i]);
        }
    }
    Ok(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spd3() -> Matrix {
        Matrix::from_rows(&[
            vec![4.0, 1.0, 0.5],
            vec![1.0, 3.0, 0.2],
            vec![0.5, 0.2, 2.0],
        ])
        .unwrap()
    }

    #[test]
    fn cholesky_reconstructs() {
        let a = spd3();
        let l = cholesky(&a).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let mut s = 0.0;
                for k in 0..3 {
                    s += l.get(i, k) * l.get(j, k);
                }
                assert!((s - a.get(i, j)).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn solve_and_inverse_agree() {
        let a = spd3();
        let b = vec![1.0, -2.0, 3.0];
        let x = spd_solve(&a, &b).unwrap();
        let ax = a.matvec(&x);
        for i in 0..3 {
            assert!((ax[i] - b[i]).abs() < 1e-13);
        }
        let inv = spd_inverse(&a).unwrap();
        let xi = inv.matvec(&b);
        for i in 0..3 {
            assert!((xi[i] - x[i]).abs() < 1e-13);
        }
    }

    #[test]
    fn rejects_indefinite_singular_asymmetric_nan() {
        let indef = Matrix::from_rows(&[vec![1.0, 2.0], vec![2.0, 1.0]]).unwrap();
        assert!(matches!(
            cholesky(&indef),
            Err(PortOptError::NotPositiveDefinite(_))
        ));
        let sing = Matrix::from_rows(&[vec![1.0, 1.0], vec![1.0, 1.0]]).unwrap();
        assert!(matches!(
            cholesky(&sing),
            Err(PortOptError::NotPositiveDefinite(_))
        ));
        let asym = Matrix::from_rows(&[vec![1.0, 0.5], vec![0.0, 1.0]]).unwrap();
        assert!(matches!(
            cholesky(&asym),
            Err(PortOptError::InvalidInput(_))
        ));
        let nan = Matrix::from_rows(&[vec![1.0, f64::NAN], vec![f64::NAN, 1.0]]).unwrap();
        assert!(matches!(cholesky(&nan), Err(PortOptError::InvalidInput(_))));
    }

    #[test]
    fn kahan_matches_exact_small_case() {
        // 1 + 1e-16 * 100 accumulated naively loses mass; Neumaier keeps it.
        let xs: Vec<f64> = std::iter::once(1.0)
            .chain(std::iter::repeat(1e-16).take(100))
            .collect();
        let s = kahan_sum(xs.iter().copied());
        assert!((s - (1.0 + 1e-14)).abs() < 1e-18);
    }
}
