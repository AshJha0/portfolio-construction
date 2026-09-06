#include "portopt/linalg.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace portopt {

namespace {
constexpr double kSymTol = 1e-10;
}  // namespace

double dot(const Vector& a, const Vector& b) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("dot: vector lengths differ");
  }
  double s = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
  return s;
}

Vector mat_vec(const Matrix& a, const Vector& v) {
  if (a.cols() != v.size()) {
    throw std::invalid_argument("mat_vec: dimension mismatch");
  }
  Vector out(a.rows(), 0.0);
  for (std::size_t i = 0; i < a.rows(); ++i) {
    double s = 0.0;
    for (std::size_t j = 0; j < a.cols(); ++j) s += a(i, j) * v[j];
    out[i] = s;
  }
  return out;
}

double kahan_sum(const Vector& xs) {
  double sum = 0.0;
  double c = 0.0;
  for (double x : xs) {
    const double t = sum + x;
    if (std::fabs(sum) >= std::fabs(x)) {
      c += (sum - t) + x;
    } else {
      c += (x - t) + sum;
    }
    sum = t;
  }
  return sum + c;
}

void check_square_symmetric(const Matrix& a) {
  if (a.rows() != a.cols()) {
    std::ostringstream msg;
    msg << "matrix must be square, got shape (" << a.rows() << ", " << a.cols() << ")";
    throw std::invalid_argument(msg.str());
  }
  if (a.rows() == 0) {
    throw std::invalid_argument("matrix must be non-empty");
  }
  const std::size_t n = a.rows();
  double max_abs = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      const double v = a(i, j);
      if (!std::isfinite(v)) {
        throw std::invalid_argument("matrix contains NaN or infinite entries");
      }
      if (std::fabs(v) > max_abs) max_abs = std::fabs(v);
    }
  }
  const double scale = 1.0 + max_abs;
  double asym = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      const double d = std::fabs(a(i, j) - a(j, i));
      if (d > asym) asym = d;
    }
  }
  if (asym > kSymTol * scale) {
    std::ostringstream msg;
    msg << "matrix is not symmetric (max asymmetry " << asym << " exceeds tolerance)";
    throw std::invalid_argument(msg.str());
  }
}

Matrix cholesky(const Matrix& a) {
  check_square_symmetric(a);
  const std::size_t n = a.rows();
  Matrix low(n, n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    if (a(i, i) <= 0.0) {
      std::ostringstream msg;
      msg << "matrix is not positive definite (diagonal entry " << a(i, i)
          << " <= 0 at row " << i << ")";
      throw std::domain_error(msg.str());
    }
    for (std::size_t j = 0; j <= i; ++j) {
      double s = a(i, j);
      for (std::size_t k = 0; k < j; ++k) s -= low(i, k) * low(j, k);
      if (i == j) {
        if (s <= kPivotRtol * a(i, i)) {
          std::ostringstream msg;
          msg << "matrix is not positive definite (Cholesky pivot " << s << " <= "
              << kPivotRtol << " * diag at row " << i
              << "); covariance may be singular (e.g. duplicated asset)";
          throw std::domain_error(msg.str());
        }
        low(i, j) = std::sqrt(s);
      } else {
        low(i, j) = s / low(j, j);
      }
    }
  }
  return low;
}

Vector chol_solve(const Matrix& low, const Vector& b) {
  const std::size_t n = low.rows();
  if (b.size() != n) {
    std::ostringstream msg;
    msg << "rhs must have length " << n << ", got " << b.size();
    throw std::invalid_argument(msg.str());
  }
  Vector y(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    double s = b[i];
    for (std::size_t k = 0; k < i; ++k) s -= low(i, k) * y[k];
    y[i] = s / low(i, i);
  }
  Vector x(n, 0.0);
  for (std::size_t ip1 = n; ip1 > 0; --ip1) {
    const std::size_t i = ip1 - 1;
    double s = y[i];
    for (std::size_t k = i + 1; k < n; ++k) s -= low(k, i) * x[k];
    x[i] = s / low(i, i);
  }
  return x;
}

Vector spd_solve(const Matrix& a, const Vector& b) { return chol_solve(cholesky(a), b); }

Matrix spd_inverse(const Matrix& a) {
  const Matrix low = cholesky(a);
  const std::size_t n = low.rows();
  Matrix inv(n, n, 0.0);
  for (std::size_t j = 0; j < n; ++j) {
    Vector e(n, 0.0);
    e[j] = 1.0;
    const Vector col = chol_solve(low, e);
    for (std::size_t i = 0; i < n; ++i) inv(i, j) = col[i];
  }
  return inv;
}

}  // namespace portopt
