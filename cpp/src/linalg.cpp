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

double fma_dot(const Vector& a, const Vector& b) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("fma_dot: vector lengths differ");
  }
  double s = 0.0;
  for (std::size_t k = 0; k < a.size(); ++k) s = std::fma(a[k], b[k], s);
  return s;
}

double blas_dot(const Vector& a, const Vector& b) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("blas_dot: vector lengths differ");
  }
  const std::size_t n = a.size();
  bool have_total = false;
  double total = 0.0;
  std::size_t k0 = 0;
  while (k0 < n) {
    const std::size_t rem = n - k0;
    std::size_t blk;
    if (rem >= 2 * kGemmQ) {
      blk = kGemmQ;
    } else if (rem > kGemmQ) {
      blk = (rem + 1) / 2;  // halved, first half rounded up
    } else {
      blk = rem;
    }
    double part = 0.0;
    for (std::size_t k = k0; k < k0 + blk; ++k) part = std::fma(a[k], b[k], part);
    total = have_total ? total + part : part;
    have_total = true;
    k0 += blk;
  }
  return total;
}

double np_sum(const double* v, std::size_t n) {
  if (n == 0) return 0.0;
  if (n < 8) {
    double s = v[0];
    for (std::size_t i = 1; i < n; ++i) s += v[i];
    return s;
  }
  if (n <= 128) {
    double r[8];
    for (std::size_t j = 0; j < 8; ++j) r[j] = v[j];
    std::size_t i = 8;
    for (; i + 8 <= n; i += 8) {
      for (std::size_t j = 0; j < 8; ++j) r[j] += v[i + j];
    }
    double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; ++i) res += v[i];
    return res;
  }
  std::size_t n2 = n / 2;
  n2 -= n2 % 8;
  return np_sum(v, n2) + np_sum(v + n2, n - n2);
}

double np_sum(const Vector& v) { return np_sum(v.data(), v.size()); }

Vector np_matvec(const Matrix& m, const Vector& v) {
  if (m.cols() != v.size()) {
    throw std::invalid_argument("np_matvec: dimension mismatch");
  }
  const std::size_t n = m.cols();
  Vector out(m.rows(), 0.0);
  for (std::size_t i = 0; i < m.rows(); ++i) {
    double r[4] = {0.0, 0.0, 0.0, 0.0};
    for (std::size_t k = 0; k < n; ++k) {
      r[k % 4] = std::fma(m(i, k), v[k], r[k % 4]);
    }
    out[i] = (n >= 4) ? (r[0] + r[2]) + (r[1] + r[3]) : r[0] + r[1] + r[2] + r[3];
  }
  return out;
}

Vector np_vecmat(const Vector& v, const Matrix& m) {
  if (m.rows() != v.size()) {
    throw std::invalid_argument("np_vecmat: dimension mismatch");
  }
  const std::size_t n = m.rows();
  Vector out(m.cols(), 0.0);
  for (std::size_t j = 0; j < m.cols(); ++j) {
    double y = 0.0;
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
      double s = std::fma(m(i, j), v[i], m(i + 1, j) * v[i + 1]);
      s = std::fma(m(i + 2, j), v[i + 2], s);
      s = std::fma(m(i + 3, j), v[i + 3], s);
      y += s;
    }
    if (n - i >= 2) {
      y += std::fma(m(i, j), v[i], m(i + 1, j) * v[i + 1]);
      i += 2;
    }
    if (i < n) y += m(i, j) * v[i];
    out[j] = y;
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
