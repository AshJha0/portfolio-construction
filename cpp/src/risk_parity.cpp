#include "portopt/risk_parity.hpp"

#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace portopt {

Vector risk_contributions(const Vector& w, const Matrix& sigma) {
  check_square_symmetric(sigma);
  const std::size_t n = sigma.rows();
  if (w.size() != n) {
    std::ostringstream msg;
    msg << "weights length " << w.size() << " does not match sigma dimension " << n;
    throw std::invalid_argument(msg.str());
  }
  for (double v : w) {
    if (!std::isfinite(v)) {
      throw std::invalid_argument("weights contain NaN or infinite entries");
    }
  }
  const Vector sw = mat_vec(sigma, w);
  const double var = dot(w, sw);
  if (!(var > 0.0)) {
    throw std::domain_error("portfolio variance must be strictly positive");
  }
  Vector rc(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) rc[i] = w[i] * sw[i] / var;
  return rc;
}

Vector inverse_vol_weights(const Matrix& sigma) {
  check_square_symmetric(sigma);
  const std::size_t n = sigma.rows();
  Vector inv(n, 0.0);
  double tot = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    if (!(sigma(i, i) > 0.0)) {
      throw std::domain_error("all variances must be strictly positive for inverse-vol");
    }
    inv[i] = 1.0 / std::sqrt(sigma(i, i));
    tot += inv[i];
  }
  for (double& v : inv) v /= tot;
  return inv;
}

namespace {

Vector validate_budgets(const Vector& budgets, std::size_t n) {
  if (budgets.empty()) return Vector(n, 1.0 / static_cast<double>(n));
  if (budgets.size() != n) {
    std::ostringstream msg;
    msg << "budgets must have length " << n;
    throw std::invalid_argument(msg.str());
  }
  double tot = 0.0;
  for (double b : budgets) {
    if (!(std::isfinite(b) && b > 0.0)) {
      throw std::invalid_argument("risk budgets must be positive and finite");
    }
    tot += b;
  }
  Vector out(budgets);
  for (double& b : out) b /= tot;
  return out;
}

}  // namespace

RiskBudgetResult risk_budget_solve(const Matrix& sigma, const Vector& budgets,
                                   double tol, int max_sweeps) {
  check_square_symmetric(sigma);
  cholesky(sigma);  // SPD gate: rejects singular / non-PSD inputs with a clear error
  const std::size_t n = sigma.rows();
  const Vector b = validate_budgets(budgets, n);
  if (!(std::isfinite(tol) && tol > 0.0)) {
    throw std::invalid_argument("tol must be positive");
  }
  if (max_sweeps < 1) throw std::invalid_argument("max_sweeps must be >= 1");

  Vector y(n, 1.0 / static_cast<double>(n));
  Vector sy = mat_vec(sigma, y);  // cached Sigma y, updated incrementally
  for (int sweep = 1; sweep <= max_sweeps; ++sweep) {
    const Vector y_prev(y);
    for (std::size_t i = 0; i < n; ++i) {
      const double m_i = sy[i] - sigma(i, i) * y[i];
      const double y_new =
          (-m_i + std::sqrt(m_i * m_i + 4.0 * sigma(i, i) * b[i])) / (2.0 * sigma(i, i));
      const double dy = y_new - y[i];
      if (dy != 0.0) {
        for (std::size_t r = 0; r < n; ++r) sy[r] += sigma(r, i) * dy;
        y[i] = y_new;
      }
    }
    double step = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      step = std::max(step, std::fabs(y[i] - y_prev[i]));
    }
    if (step < tol) {
      double tot = 0.0;
      for (double v : y) tot += v;
      Vector w(y);
      for (double& v : w) v /= tot;
      return RiskBudgetResult{w, sweep, true};
    }
  }
  double tot = 0.0;
  for (double v : y) tot += v;
  Vector w(y);
  for (double& v : w) v /= tot;
  return RiskBudgetResult{w, max_sweeps, false};
}

Vector risk_budget_weights(const Matrix& sigma, const Vector& budgets, double tol,
                           int max_sweeps) {
  RiskBudgetResult res = risk_budget_solve(sigma, budgets, tol, max_sweeps);
  if (!res.converged) {
    std::ostringstream msg;
    msg << std::scientific << std::setprecision(3)
        << "risk_budget_weights: CCD did not converge within " << res.sweeps
        << " sweeps (tol " << tol << "); use risk_budget_solve to inspect the iterate";
    throw std::domain_error(msg.str());
  }
  return std::move(res.weights);
}

Vector erc_weights(const Matrix& sigma, double tol, int max_sweeps) {
  return risk_budget_weights(sigma, Vector(), tol, max_sweeps);
}

}  // namespace portopt
