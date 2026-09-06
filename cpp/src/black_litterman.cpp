#include "portopt/black_litterman.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

#include "portopt/markowitz.hpp"

namespace portopt {

namespace {

/// Validate the k x n pick matrix P: at least one row, finite entries, no
/// identically-zero row.
void validate_views_matrix(const Matrix& p, std::size_t n) {
  if (p.cols() != n) {
    std::ostringstream msg;
    msg << "P must be k x " << n << ", got shape (" << p.rows() << ", " << p.cols()
        << ")";
    throw std::invalid_argument(msg.str());
  }
  if (p.rows() < 1) throw std::invalid_argument("P must contain at least one view row");
  for (std::size_t i = 0; i < p.rows(); ++i) {
    double row_max = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
      if (!std::isfinite(p(i, j))) {
        throw std::invalid_argument("P contains NaN or infinite entries");
      }
      row_max = std::max(row_max, std::fabs(p(i, j)));
    }
    if (row_max == 0.0) {
      std::ostringstream msg;
      msg << "view row " << i << " of P is identically zero";
      throw std::invalid_argument(msg.str());
    }
  }
}

double validate_tau(double tau) {
  if (!(std::isfinite(tau) && tau > 0.0 && tau <= 1.0)) {
    std::ostringstream msg;
    msg << "tau must lie in (0, 1], got " << tau;
    throw std::invalid_argument(msg.str());
  }
  return tau;
}

Vector validate_confidences(const Vector& confidences, std::size_t k) {
  if (confidences.empty()) return Vector(k, 1.0);
  if (confidences.size() != k) {
    std::ostringstream msg;
    msg << "confidences must have length " << k;
    throw std::invalid_argument(msg.str());
  }
  for (double c : confidences) {
    if (!(std::isfinite(c) && c > 0.0)) {
      throw std::invalid_argument("confidences must be positive and finite");
    }
  }
  return confidences;
}

std::size_t name_index(const std::vector<std::string>& names, const std::string& name) {
  const auto it = std::find(names.begin(), names.end(), name);
  if (it == names.end()) {
    throw std::invalid_argument("unknown asset name in view: '" + name + "'");
  }
  return static_cast<std::size_t>(it - names.begin());
}

/// B = P (tau Sigma), k x n.
Matrix pick_times_tau_sigma(const Matrix& p, const Matrix& ts) {
  const std::size_t k = p.rows();
  const std::size_t n = ts.rows();
  Matrix b(k, n, 0.0);
  for (std::size_t i = 0; i < k; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      double acc = 0.0;
      for (std::size_t l = 0; l < n; ++l) acc += p(i, l) * ts(l, j);
      b(i, j) = acc;
    }
  }
  return b;
}

}  // namespace

Vector implied_equilibrium_returns(const Matrix& sigma, const Vector& w_mkt,
                                   double delta) {
  check_square_symmetric(sigma);
  const std::size_t n = sigma.rows();
  if (w_mkt.size() != n) {
    std::ostringstream msg;
    msg << "w_mkt length " << w_mkt.size() << " does not match sigma dimension " << n;
    throw std::invalid_argument(msg.str());
  }
  double tot = 0.0;
  for (double w : w_mkt) {
    if (!std::isfinite(w)) {
      throw std::invalid_argument("w_mkt contains NaN or infinite entries");
    }
    if (w < 0.0) throw std::invalid_argument("market-cap weights must be non-negative");
    tot += w;
  }
  if (!(tot > 0.0)) {
    throw std::invalid_argument("market-cap weights must have positive sum");
  }
  if (!(std::isfinite(delta) && delta > 0.0)) {
    throw std::invalid_argument("risk aversion delta must be positive and finite");
  }
  Vector w(w_mkt);
  for (double& v : w) v /= tot;
  Vector pi = mat_vec(sigma, w);
  for (double& v : pi) v *= delta;
  return pi;
}

ViewMatrices views_to_matrices(const std::vector<View>& views,
                               const std::vector<std::string>& asset_names) {
  if (asset_names.empty()) throw std::invalid_argument("asset_names must be non-empty");
  for (std::size_t i = 0; i < asset_names.size(); ++i) {
    for (std::size_t j = i + 1; j < asset_names.size(); ++j) {
      if (asset_names[i] == asset_names[j]) {
        throw std::invalid_argument("asset_names contains duplicates");
      }
    }
  }
  if (views.empty()) throw std::invalid_argument("views list must be non-empty");
  const std::size_t k = views.size();
  const std::size_t n = asset_names.size();
  ViewMatrices out{Matrix(k, n, 0.0), Vector(k, 0.0), Vector(k, 1.0)};
  for (std::size_t i = 0; i < k; ++i) {
    const View& view = views[i];
    if (view.type == "absolute") {
      out.p(i, name_index(asset_names, view.asset)) = 1.0;
    } else if (view.type == "relative") {
      const std::size_t i_long = name_index(asset_names, view.long_name);
      const std::size_t i_short = name_index(asset_names, view.short_name);
      if (i_long == i_short) {
        std::ostringstream msg;
        msg << "view " << i << ": relative view: long and short legs must differ";
        throw std::invalid_argument(msg.str());
      }
      out.p(i, i_long) = 1.0;
      out.p(i, i_short) = -1.0;
    } else {
      throw std::invalid_argument("unknown view type: '" + view.type + "'");
    }
    if (!std::isfinite(view.q)) {
      std::ostringstream msg;
      msg << "view " << i << ": q must be finite";
      throw std::invalid_argument(msg.str());
    }
    out.q[i] = view.q;
    if (!(std::isfinite(view.confidence) && view.confidence > 0.0)) {
      std::ostringstream msg;
      msg << "view " << i << ": confidence must be positive and finite";
      throw std::invalid_argument(msg.str());
    }
    out.confidences[i] = view.confidence;
  }
  return out;
}

Matrix default_omega(const Matrix& sigma, const Matrix& p, double tau,
                     const Vector& confidences) {
  check_square_symmetric(sigma);
  validate_views_matrix(p, sigma.rows());
  const double t = validate_tau(tau);
  const std::size_t k = p.rows();
  const std::size_t n = sigma.rows();
  const Vector conf = validate_confidences(confidences, k);
  Matrix ts(n, n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) ts(i, j) = t * sigma(i, j);
  }
  const Matrix b = pick_times_tau_sigma(p, ts);
  Matrix omega(k, k, 0.0);
  for (std::size_t i = 0; i < k; ++i) {
    // core diagonal entry [P tau S P']_ii = b row i . p row i
    double core = 0.0;
    for (std::size_t j = 0; j < n; ++j) core += b(i, j) * p(i, j);
    omega(i, i) = core / conf[i];
  }
  return omega;
}

namespace {

BLResult bl_posterior_core(const Matrix& sigma, const Vector& pi, const Matrix& p,
                           const Vector& q, double tau, const Matrix* omega,
                           const Vector& confidences) {
  check_square_symmetric(sigma);
  const std::size_t n = sigma.rows();
  if (pi.size() != n) {
    std::ostringstream msg;
    msg << "pi must be a finite vector of length " << n;
    throw std::invalid_argument(msg.str());
  }
  for (double v : pi) {
    if (!std::isfinite(v)) {
      throw std::invalid_argument("pi contains NaN or infinite entries");
    }
  }
  validate_views_matrix(p, n);
  const std::size_t k = p.rows();
  if (q.size() != k) {
    std::ostringstream msg;
    msg << "Q must be a finite vector of length " << k;
    throw std::invalid_argument(msg.str());
  }
  for (double v : q) {
    if (!std::isfinite(v)) {
      throw std::invalid_argument("Q contains NaN or infinite entries");
    }
  }
  const double t = validate_tau(tau);
  Matrix om;
  if (omega == nullptr) {
    om = default_omega(sigma, p, t, confidences);
  } else {
    if (!confidences.empty()) {
      throw std::invalid_argument(
          "confidences are only used with the default Omega; pass one or the other");
    }
    check_square_symmetric(*omega);
    if (omega->rows() != k) {
      std::ostringstream msg;
      msg << "Omega must be " << k << " x " << k << ", got (" << omega->rows() << ", "
          << omega->cols() << ")";
      throw std::invalid_argument(msg.str());
    }
    om = *omega;
  }

  Matrix ts(n, n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) ts(i, j) = t * sigma(i, j);
  }
  const Matrix b = pick_times_tau_sigma(p, ts);  // k x n = P (tau Sigma)
  Matrix a_k(k, k, 0.0);                         // B P' + Omega
  for (std::size_t i = 0; i < k; ++i) {
    for (std::size_t j = 0; j < k; ++j) {
      double acc = 0.0;
      for (std::size_t l = 0; l < n; ++l) acc += b(i, l) * p(j, l);
      a_k(i, j) = acc + om(i, j);
    }
  }
  const Matrix low = cholesky(a_k);  // throws if Omega makes A_k non-SPD

  Vector resid(k, 0.0);  // Q - P pi
  for (std::size_t i = 0; i < k; ++i) {
    double acc = 0.0;
    for (std::size_t j = 0; j < n; ++j) acc += p(i, j) * pi[j];
    resid[i] = q[i] - acc;
  }
  const Vector x = chol_solve(low, resid);
  Vector mu_post(pi);
  for (std::size_t j = 0; j < n; ++j) {
    double acc = 0.0;
    for (std::size_t i = 0; i < k; ++i) acc += b(i, j) * x[i];
    mu_post[j] += acc;
  }

  // M = tau S - B' A_k^-1 B: A_k^-1 B via n solves, one per column of B.
  Matrix ainv_b(k, n, 0.0);
  for (std::size_t j = 0; j < n; ++j) {
    Vector col(k, 0.0);
    for (std::size_t i = 0; i < k; ++i) col[i] = b(i, j);
    const Vector sol = chol_solve(low, col);
    for (std::size_t i = 0; i < k; ++i) ainv_b(i, j) = sol[i];
  }
  Matrix m_mat(n, n, 0.0);
  Matrix sigma_post(n, n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      double acc = 0.0;
      for (std::size_t l = 0; l < k; ++l) acc += b(l, i) * ainv_b(l, j);
      m_mat(i, j) = ts(i, j) - acc;
      sigma_post(i, j) = sigma(i, j) + m_mat(i, j);
    }
  }
  return BLResult{mu_post, sigma_post, m_mat, om};
}

}  // namespace

BLResult bl_posterior(const Matrix& sigma, const Vector& pi, const Matrix& p,
                      const Vector& q, double tau, const Vector& confidences) {
  return bl_posterior_core(sigma, pi, p, q, tau, nullptr, confidences);
}

BLResult bl_posterior_with_omega(const Matrix& sigma, const Vector& pi, const Matrix& p,
                                 const Vector& q, double tau, const Matrix& omega) {
  return bl_posterior_core(sigma, pi, p, q, tau, &omega, Vector());
}

Vector bl_weights(const Matrix& sigma, const Vector& pi, const Matrix& p,
                  const Vector& q, double tau, double rf, const Vector& confidences,
                  BLResult* result) {
  if (!std::isfinite(rf)) throw std::invalid_argument("rf must be finite");
  BLResult res = bl_posterior(sigma, pi, p, q, tau, confidences);
  Vector w = tangency_weights(res.mu_post, res.sigma_post, rf);
  if (result != nullptr) *result = std::move(res);
  return w;
}

Vector bl_weights_with_omega(const Matrix& sigma, const Vector& pi, const Matrix& p,
                             const Vector& q, double tau, const Matrix& omega, double rf,
                             BLResult* result) {
  if (!std::isfinite(rf)) throw std::invalid_argument("rf must be finite");
  BLResult res = bl_posterior_with_omega(sigma, pi, p, q, tau, omega);
  Vector w = tangency_weights(res.mu_post, res.sigma_post, rf);
  if (result != nullptr) *result = std::move(res);
  return w;
}

}  // namespace portopt
