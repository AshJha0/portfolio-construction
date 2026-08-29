/// \file linalg.hpp
/// \brief Pinned dense linear algebra kernels for small SPD systems (n <= 20).
///
/// Every routine reproduces the Python reference algorithm exactly
/// (API_SPEC.md section 3):
///
/// * Cholesky: Cholesky-Banachiewicz, lower triangular, computed row by
///   row, with a pinned relative pivot threshold (a numerically singular
///   matrix such as a duplicated-asset covariance yields a pivot that is a
///   tiny rounding residual instead of an exact zero, so the guard must be
///   relative).
/// * SPD solve: forward substitution with L, then back substitution with
///   L'.
/// * SPD inverse: column-by-column Cholesky solves against identity
///   columns (no post-symmetrization).
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace portopt {

/// Dense column vector of doubles.
using Vector = std::vector<double>;

/// Minimal dense row-major matrix of doubles.
class Matrix {
 public:
  Matrix() = default;

  /// rows x cols matrix filled with \p fill.
  Matrix(std::size_t rows, std::size_t cols, double fill = 0.0)
      : rows_(rows), cols_(cols), data_(rows * cols, fill) {}

  std::size_t rows() const { return rows_; }
  std::size_t cols() const { return cols_; }

  double& operator()(std::size_t i, std::size_t j) { return data_[i * cols_ + j]; }
  double operator()(std::size_t i, std::size_t j) const { return data_[i * cols_ + j]; }

  /// Row \p i copied out as a Vector.
  Vector row(std::size_t i) const {
    return Vector(data_.begin() + static_cast<std::ptrdiff_t>(i * cols_),
                  data_.begin() + static_cast<std::ptrdiff_t>((i + 1) * cols_));
  }

 private:
  std::size_t rows_ = 0;
  std::size_t cols_ = 0;
  std::vector<double> data_;
};

/// Pinned relative Cholesky pivot threshold (API_SPEC 3.1): the pivot at
/// row i must exceed kPivotRtol * a(i,i). Rejects matrices with an
/// effective condition number beyond ~1e13 — far outside anything a valid
/// n <= 20 covariance produces, but safely catching duplicated assets.
constexpr double kPivotRtol = 1e-13;

/// Dot product of two equal-length vectors (plain left-to-right
/// accumulation, mirroring the reference's explicit loops).
double dot(const Vector& a, const Vector& b);

/// Matrix-vector product a * v (plain sequential row sums).
Vector mat_vec(const Matrix& a, const Vector& v);

// ---------------------------------------------------------------------------
// Reference-parity reductions.
//
// The pinned PGD golden cases are ulp-sensitive: the iterate stops a few
// 1e-8 from the exact minimizer along a low-curvature direction, and the
// pinned turnover value (tol 1e-10) is a function of that stopping point.
// Reproducing them therefore requires reproducing the Python reference's
// floating-point path bit for bit, and the reference delegates its inner
// products to numpy/OpenBLAS, whose summation orders differ from plain
// sequential loops. The kernels below replicate those orders exactly
// (verified against the reference BLAS); std::fma is used where the BLAS
// kernels fuse, since it is correctly rounded.
// ---------------------------------------------------------------------------

/// Sequential fused-multiply-add dot product: s = fma(a[k], b[k], s).
///
/// Reproduces the OpenBLAS `ddot` result numpy produces for the short
/// (n <= 8) vector-vector products inside the reference PGD loop; keeping
/// those bits identical keeps the PGD trajectory identical to Python's.
double fma_dot(const Vector& a, const Vector& b);

/// OpenBLAS dgemm k-blocking constant (GEMM_Q) observed for the
/// reference BLAS: long reduction dimensions are processed in blocks.
constexpr std::size_t kGemmQ = 384;

/// Long dot product mirroring the OpenBLAS dgemm accumulation order used
/// by numpy's `xc.T @ xc` (the sample covariance): the k-dimension is cut
/// greedily into blocks (a remainder >= 2*384 takes 384; a remainder in
/// (384, 768) is halved with the first half rounded up), each block is
/// accumulated with sequential FMA, and block partials are added left to
/// right. Bit-exact for every window length the pinned backtest uses.
double blas_dot(const Vector& a, const Vector& b);

/// numpy-compatible pairwise summation (`np.sum` over a contiguous
/// float64 vector): naive below 8 elements, an 8-accumulator unrolled
/// loop with a fixed reduction tree up to 128, and a recursive halving
/// split (rounded down to a multiple of 8) above.
double np_sum(const double* v, std::size_t n);

/// Overload of np_sum taking a Vector.
double np_sum(const Vector& v);

/// Matrix-vector product mirroring the OpenBLAS dgemv row kernel numpy
/// uses for `Sigma @ w` (the PGD gradient): per row, four FMA
/// accumulators striding the columns, reduced as (r0+r2) + (r1+r3).
Vector np_matvec(const Matrix& m, const Vector& v);

/// Vector-matrix product v' M mirroring numpy's `w @ Sigma` (the
/// OpenBLAS dgemv_n SkylakeX kernel): per output column, blocks of four
/// terms are combined as fma(m2, v2, fma(m3, v3, ...)) seeded by
/// fma(m0, v0, m1*v1), and block sums are added one block at a time; a
/// 2/1-column tail follows the driver's 4x2/4x1 kernels.
Vector np_vecmat(const Vector& v, const Matrix& m);

/// Neumaier-compensated sum for long (T-length) accumulations.
///
/// Matches numpy's pairwise summation to within a couple of ulps, keeping
/// the T = 1250 dispersion sums inside the 1e-10 golden tolerances.
double kahan_sum(const Vector& xs);

/// Validate that \p a is a finite, square, symmetric, non-empty matrix.
/// \throws std::invalid_argument if not square/non-empty, contains
///   NaN/inf, or max|a - a'| > 1e-10 * (1 + max|a|).
void check_square_symmetric(const Matrix& a);

/// Lower-triangular Cholesky factor L with a = L L' (Cholesky-Banachiewicz,
/// row by row; API_SPEC 3.1).
/// \throws std::invalid_argument on failed symmetry validation.
/// \throws std::domain_error if any diagonal entry is <= 0 or any pivot
///   argument s satisfies s <= kPivotRtol * a(i,i) (matrix not positive
///   definite to working precision — the pinned detector for singular or
///   indefinite covariance inputs, e.g. a duplicated asset).
Matrix cholesky(const Matrix& a);

/// Solve (L L') x = b given the lower Cholesky factor \p low.
/// Pinned: forward substitution L y = b (ascending rows), then back
/// substitution L' x = y (descending rows).
/// \throws std::invalid_argument if b's length does not match.
Vector chol_solve(const Matrix& low, const Vector& b);

/// Solve a x = b for symmetric positive definite \p a via Cholesky.
Vector spd_solve(const Matrix& a, const Vector& b);

/// Inverse of an SPD matrix via column-by-column Cholesky solves against
/// the identity columns e_0..e_{n-1} (pinned; no post-symmetrization).
Matrix spd_inverse(const Matrix& a);

}  // namespace portopt
