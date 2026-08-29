#include "portopt/estimators.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace portopt {

void validate_returns(const Matrix& returns, std::size_t min_rows) {
  const std::size_t t = returns.rows();
  const std::size_t n = returns.cols();
  if (n < 1) {
    throw std::invalid_argument("returns must contain at least one asset column");
  }
  if (t < min_rows) {
    std::ostringstream msg;
    msg << "returns must contain at least " << min_rows << " rows, got " << t;
    throw std::invalid_argument(msg.str());
  }
  for (std::size_t r = 0; r < t; ++r) {
    for (std::size_t c = 0; c < n; ++c) {
      if (!std::isfinite(returns(r, c))) {
        throw std::invalid_argument("returns contain NaN or infinite entries");
      }
    }
  }
}

Vector sample_mean(const Matrix& returns) {
  validate_returns(returns, 1);
  const std::size_t t = returns.rows();
  const std::size_t n = returns.cols();
  // Pinned order: accumulate row by row (matches numpy's axis-0 sum), then
  // divide by T once.
  Vector acc(n, 0.0);
  for (std::size_t r = 0; r < t; ++r) {
    for (std::size_t c = 0; c < n; ++c) acc[c] += returns(r, c);
  }
  for (std::size_t c = 0; c < n; ++c) acc[c] /= static_cast<double>(t);
  return acc;
}

namespace {

/// Demeaned columns of the returns panel, stored column-major
/// (`xc[i][t]`) so the T-length products are contiguous.
std::vector<Vector> demeaned_columns(const Matrix& x) {
  const Vector mu = sample_mean(x);
  const std::size_t t = x.rows();
  const std::size_t n = x.cols();
  std::vector<Vector> xc(n, Vector(t, 0.0));
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t r = 0; r < t; ++r) xc[j][r] = x(r, j) - mu[j];
  }
  return xc;
}

/// S[i][j] = (1/(T-1)) sum_t xc[t][i] xc[t][j].
///
/// The T-length products use blas_dot (the reference BLAS dgemm order),
/// making every entry bit-identical to the Python implementation —
/// required because the downstream PGD golden values are ulp-sensitive to
/// the covariance input.
Matrix cov_from_demeaned(const std::vector<Vector>& xc, std::size_t t) {
  const std::size_t n = xc.size();
  Matrix s(n, n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i; j < n; ++j) {
      const double v = blas_dot(xc[i], xc[j]) / static_cast<double>(t - 1);
      s(i, j) = v;
      s(j, i) = v;
    }
  }
  return s;
}

}  // namespace

Matrix sample_cov(const Matrix& returns) {
  validate_returns(returns, 2);
  return cov_from_demeaned(demeaned_columns(returns), returns.rows());
}

ShrinkageResult shrink_covariance(const Matrix& returns) {
  validate_returns(returns, 2);
  const std::size_t t = returns.rows();
  const std::size_t n = returns.cols();
  if (n < 2) {
    throw std::invalid_argument("shrinkage requires at least 2 assets");
  }
  const std::vector<Vector> xc = demeaned_columns(returns);
  const Matrix s = cov_from_demeaned(xc, t);
  Vector sd(n);
  for (std::size_t i = 0; i < n; ++i) {
    if (s(i, i) <= 0.0) {
      throw std::domain_error("shrinkage requires strictly positive sample variances");
    }
    sd[i] = std::sqrt(s(i, i));
  }

  // rbar: average of the upper-triangle sample correlations.
  double corr_sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) corr_sum += s(i, j) / (sd[i] * sd[j]);
  }
  const double rbar =
      2.0 * corr_sum / (static_cast<double>(n) * static_cast<double>(n - 1));

  // Constant-correlation target F.
  Matrix f(n, n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) f(i, j) = rbar * sd[i] * sd[j];
    f(i, i) = s(i, i);
  }

  // Dispersion sums (pinned: divisor T, ordered double loops).
  double pi_hat = 0.0;
  double rho_hat = 0.0;
  double gamma_hat = 0.0;
  Vector terms(t, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      const double sij = s(i, j);
      for (std::size_t r = 0; r < t; ++r) {
        const double dev = xc[i][r] * xc[j][r] - sij;
        terms[r] = dev * dev;
      }
      const double pij = kahan_sum(terms) / static_cast<double>(t);
      pi_hat += pij;
      if (i == j) rho_hat += pij;
      const double gd = f(i, j) - sij;
      gamma_hat += gd * gd;
    }
  }
  // theta terms: each ordered pair (i, j), i != j, contributes
  // (rbar/2) sqrt(S_jj/S_ii) theta_ii,ij; the (j, i) pass supplies the
  // symmetric partner, so each unordered pair gets both classic terms once.
  for (std::size_t i = 0; i < n; ++i) {
    const double sii = s(i, i);
    for (std::size_t j = 0; j < n; ++j) {
      if (i == j) continue;
      const double sij = s(i, j);
      for (std::size_t r = 0; r < t; ++r) {
        const double xi2_dev = xc[i][r] * xc[i][r] - sii;
        const double dev_ij = xc[i][r] * xc[j][r] - sij;
        terms[r] = xi2_dev * dev_ij;
      }
      const double theta = kahan_sum(terms) / static_cast<double>(t);
      rho_hat += (rbar / 2.0) * std::sqrt(s(j, j) / sii) * theta;
    }
  }

  double delta = 0.0;
  if (gamma_hat > 0.0) {
    const double kappa = (pi_hat - rho_hat) / gamma_hat;
    delta = kappa / static_cast<double>(t);
    if (delta < 0.0) delta = 0.0;
    if (delta > 1.0) delta = 1.0;
  }

  Matrix sigma(n, n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      sigma(i, j) = delta * f(i, j) + (1.0 - delta) * s(i, j);
    }
  }
  return ShrinkageResult{sigma, delta, rbar, s, f};
}

Vector annualize_mean(const Vector& mu, int periods) {
  if (periods <= 0) throw std::invalid_argument("periods must be positive");
  Vector out(mu);
  for (double& v : out) v *= static_cast<double>(periods);
  return out;
}

Matrix annualize_cov(const Matrix& cov, int periods) {
  if (periods <= 0) throw std::invalid_argument("periods must be positive");
  Matrix out(cov);
  for (std::size_t i = 0; i < out.rows(); ++i) {
    for (std::size_t j = 0; j < out.cols(); ++j) {
      out(i, j) *= static_cast<double>(periods);
    }
  }
  return out;
}

}  // namespace portopt
