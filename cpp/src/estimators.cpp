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
      if (returns(r, c) <= -1.0) {
        throw std::invalid_argument(
            "returns contain a simple return <= -1 (-100% or worse)");
      }
    }
  }
}

Vector sample_mean(const Matrix& returns) {
  validate_returns(returns, 1);
  const std::size_t t = returns.rows();
  const std::size_t n = returns.cols();
  Vector out(n, 0.0);
  Vector col(t, 0.0);
  for (std::size_t c = 0; c < n; ++c) {
    for (std::size_t r = 0; r < t; ++r) col[r] = returns(r, c);
    out[c] = kahan_sum(col) / static_cast<double>(t);
  }
  return out;
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

/// S[i][j] = (1/(T-1)) sum_t xc[t][i] xc[t][j] with a compensated sum.
Matrix cov_from_demeaned(const std::vector<Vector>& xc, std::size_t t) {
  const std::size_t n = xc.size();
  Matrix s(n, n, 0.0);
  Vector prod(t, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i; j < n; ++j) {
      for (std::size_t r = 0; r < t; ++r) prod[r] = xc[i][r] * xc[j][r];
      const double v = kahan_sum(prod) / static_cast<double>(t - 1);
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
  for (std::size_t i = 0; i < n; ++i) {
    if (s(i, i) <= 0.0) {
      throw std::domain_error("shrinkage requires strictly positive sample variances");
    }
  }
  const double tf = static_cast<double>(t);
  // Paper convention (Ledoit & Wolf 2004, App. B / covCor.m): divisor T
  // for every dispersion quantity.
  Matrix s_t(n, n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) s_t(i, j) = s(i, j) * ((tf - 1.0) / tf);
  }
  Vector var_t(n, 0.0);
  Vector sd_t(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    var_t[i] = s_t(i, i);
    sd_t[i] = std::sqrt(var_t[i]);
  }

  // rbar: average of the upper-triangle sample correlations.
  double corr_sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) corr_sum += s_t(i, j) / (sd_t[i] * sd_t[j]);
  }
  const double rbar =
      2.0 * corr_sum / (static_cast<double>(n) * static_cast<double>(n - 1));

  // Constant-correlation target in the paper convention.
  Matrix f_t(n, n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) f_t(i, j) = rbar * sd_t[i] * sd_t[j];
    f_t(i, i) = var_t[i];
  }

  // Dispersion sums: pi_hat, rho_hat (both theta terms per unordered pair,
  // i.e. weight rbar per ordered pair), gamma_hat.
  double pi_hat = 0.0;
  double rho_hat = 0.0;
  double gamma_hat = 0.0;
  Vector terms(t, 0.0);
  Vector xi2_dev(t, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t r = 0; r < t; ++r) xi2_dev[r] = xc[i][r] * xc[i][r] - var_t[i];
    for (std::size_t j = 0; j < n; ++j) {
      const double sij = s_t(i, j);
      for (std::size_t r = 0; r < t; ++r) {
        const double dev = xc[i][r] * xc[j][r] - sij;
        terms[r] = dev * dev;
      }
      const double pij = kahan_sum(terms) / tf;
      pi_hat += pij;
      const double gd = f_t(i, j) - sij;
      gamma_hat += gd * gd;
      if (i == j) {
        rho_hat += pij;
      } else {
        for (std::size_t r = 0; r < t; ++r) {
          terms[r] = xi2_dev[r] * (xc[i][r] * xc[j][r] - sij);
        }
        const double theta_ii_ij = kahan_sum(terms) / tf;
        rho_hat += rbar * std::sqrt(var_t[j] / var_t[i]) * theta_ii_ij;
      }
    }
  }

  double delta = 0.0;
  if (gamma_hat > 0.0) {
    const double kappa = (pi_hat - rho_hat) / gamma_hat;
    delta = kappa / tf;
    if (delta < 0.0) delta = 0.0;
    if (delta > 1.0) delta = 1.0;
  }

  // Output in the project-wide ddof=1 convention (API_SPEC 2.1/2.2).
  Vector sd(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) sd[i] = std::sqrt(s(i, i));
  Matrix f(n, n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) f(i, j) = rbar * sd[i] * sd[j];
    f(i, i) = s(i, i);
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
  for (double& v : out) {
    if (!std::isfinite(v)) throw std::invalid_argument("mu contains NaN or infinite entries");
    v *= static_cast<double>(periods);
  }
  return out;
}

Matrix annualize_cov(const Matrix& cov, int periods) {
  if (periods <= 0) throw std::invalid_argument("periods must be positive");
  Matrix out(cov);
  for (std::size_t i = 0; i < out.rows(); ++i) {
    for (std::size_t j = 0; j < out.cols(); ++j) {
      if (!std::isfinite(out(i, j))) {
        throw std::invalid_argument("cov contains NaN or infinite entries");
      }
      out(i, j) *= static_cast<double>(periods);
    }
  }
  return out;
}

}  // namespace portopt
