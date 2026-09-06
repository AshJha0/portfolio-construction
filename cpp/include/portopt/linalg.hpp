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
#include <stdexcept>
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

/// Sum of a vector with Neumaier compensation (a Kahan variant that also
/// handles terms larger than the running sum). Used for every T-length
/// reduction (means, covariances, shrinkage dispersion sums, backtest
/// metrics): the Python reference sums with correctly rounded `math.fsum`,
/// and the compensated sum agrees with it to a couple of ulps, so no
/// golden value depends on the BLAS or the summation order.
double kahan_sum(const Vector& xs);

/// Thrown when the tangency portfolio is undefined (A - rf*C <= 0) or
/// ill-conditioned (A - rf*C within a relative kTangencyCondRtol of
/// cancelling). Derives from std::domain_error so generic handlers still
/// work; the demo's pinned fallback catches exactly this type.
struct TangencyUndefinedError : std::domain_error {
  using std::domain_error::domain_error;
};

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
