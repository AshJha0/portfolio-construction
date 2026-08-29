#include "portopt/markowitz.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace portopt {

namespace {

/// Shared mu/sigma validation (shapes + finite mu); PSD is checked later
/// by the Cholesky factorization, matching the reference order of checks.
void validate_mu_sigma(const Vector& mu, const Matrix& sigma) {
  if (sigma.rows() != sigma.cols()) {
    std::ostringstream msg;
    msg << "sigma must be square, got shape (" << sigma.rows() << ", " << sigma.cols()
        << ")";
    throw std::invalid_argument(msg.str());
  }
  if (mu.size() != sigma.rows()) {
    std::ostringstream msg;
    msg << "mu length " << mu.size() << " does not match sigma dimension "
        << sigma.rows();
    throw std::invalid_argument(msg.str());
  }
  for (double v : mu) {
    if (!std::isfinite(v)) {
      throw std::invalid_argument("mu contains NaN or infinite entries");
    }
  }
}

double clip(double v, double lo, double hi) { return std::min(std::max(v, lo), hi); }

/// Plain sequential sum, mirroring the reference's explicit loops.
double vec_sum(const Vector& v) {
  double s = 0.0;
  for (double x : v) s += x;
  return s;
}

/// numpy `w @ s @ w` (left-associated: (w @ s) @ w), using the
/// reference-parity BLAS kernels of linalg.hpp — the PGD objective is
/// evaluated through this exact path.
double np_quad(const Vector& w, const Matrix& s) { return fma_dot(np_vecmat(w, s), w); }

}  // namespace

TwoFundScalars two_fund_scalars(const Vector& mu, const Matrix& sigma) {
  validate_mu_sigma(mu, sigma);
  const Matrix low = cholesky(sigma);
  const Vector x = chol_solve(low, mu);
  const Vector y = chol_solve(low, Vector(mu.size(), 1.0));
  const double a = vec_sum(x);
  const double b = dot(mu, x);
  const double c = vec_sum(y);
  return TwoFundScalars{a, b, c, b * c - a * a};
}

Vector min_variance_weights(const Matrix& sigma) {
  if (sigma.rows() != sigma.cols()) {
    std::ostringstream msg;
    msg << "sigma must be square, got shape (" << sigma.rows() << ", " << sigma.cols()
        << ")";
    throw std::invalid_argument(msg.str());
  }
  if (sigma.rows() == 1) {
    if (!(sigma(0, 0) > 0.0) || !std::isfinite(sigma(0, 0))) {
      throw std::invalid_argument("1x1 covariance must be positive and finite");
    }
    return Vector{1.0};
  }
  const Vector y = chol_solve(cholesky(sigma), Vector(sigma.rows(), 1.0));
  const double c = vec_sum(y);
  Vector w(y);
  for (double& v : w) v /= c;
  return w;
}

Vector tangency_weights(const Vector& mu, const Matrix& sigma, double rf) {
  validate_mu_sigma(mu, sigma);
  if (!std::isfinite(rf)) throw std::invalid_argument("rf must be finite");
  if (sigma.rows() == 1) {
    if (mu[0] <= rf) {
      throw std::domain_error(
          "tangency undefined: risk-free rate >= expected return of the single asset");
    }
    if (!(sigma(0, 0) > 0.0) || !std::isfinite(sigma(0, 0))) {
      throw std::invalid_argument("1x1 covariance must be positive and finite");
    }
    return Vector{1.0};
  }
  const Matrix low = cholesky(sigma);
  Vector excess(mu);
  for (double& v : excess) v -= rf;
  const Vector z = chol_solve(low, excess);
  const double denom = vec_sum(z);  // = A - rf*C
  if (denom <= 0.0) {
    throw std::domain_error(
        "tangency undefined: A - rf*C <= 0 (risk-free rate is at or above the "
        "minimum-variance portfolio return)");
  }
  Vector w(z);
  for (double& v : w) v /= denom;
  return w;
}

Vector frontier_weights(const Vector& mu, const Matrix& sigma, double target) {
  validate_mu_sigma(mu, sigma);
  if (!std::isfinite(target)) {
    throw std::invalid_argument("target return must be finite");
  }
  if (sigma.rows() == 1) {
    if (std::fabs(target - mu[0]) > 1e-12 * (1.0 + std::fabs(mu[0]))) {
      throw std::domain_error(
          "n=1: target return must equal the single asset's expected return");
    }
    return Vector{1.0};
  }
  const Matrix low = cholesky(sigma);
  const Vector x = chol_solve(low, mu);
  const Vector y = chol_solve(low, Vector(mu.size(), 1.0));
  const double a = vec_sum(x);
  const double b = dot(mu, x);
  const double c = vec_sum(y);
  const double d = b * c - a * a;
  // Pinned relative degeneracy test: by Cauchy-Schwarz D >= 0 with equality
  // iff mu is proportional to ones; rounding turns the exact zero into a
  // tiny residual, so compare against B*C (> 0).
  if (d <= 1e-12 * b * c) {
    std::ostringstream msg;
    msg << "degenerate frontier: D = B*C - A^2 = " << d
        << " <= 1e-12*B*C (expected returns proportional to ones?)";
    throw std::domain_error(msg.str());
  }
  const double mv_ret = a / c;
  if (target < mv_ret - 1e-12 * (1.0 + std::fabs(mv_ret))) {
    std::ostringstream msg;
    msg << "target return " << target << " lies below the minimum-variance return "
        << mv_ret << "; the efficient frontier is defined for targets >= A/C";
    throw std::domain_error(msg.str());
  }
  const std::size_t n = mu.size();
  Vector w(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    const double g = (b * y[i] - a * x[i]) / d;
    const double h = (c * x[i] - a * y[i]) / d;
    w[i] = g + h * target;
  }
  return w;
}

namespace {

/// Pinned grid: n_points targets from lo to hi inclusive.
Vector frontier_grid(double lo, double hi, int n_points) {
  Vector targets(static_cast<std::size_t>(n_points), 0.0);
  for (int k = 0; k < n_points; ++k) {
    targets[static_cast<std::size_t>(k)] =
        lo + static_cast<double>(k) * (hi - lo) / static_cast<double>(n_points - 1);
  }
  return targets;
}

double quad_form(const Vector& w, const Matrix& s) { return dot(w, mat_vec(s, w)); }

}  // namespace

FrontierResult efficient_frontier(const Vector& mu, const Matrix& sigma, int n_points) {
  validate_mu_sigma(mu, sigma);
  if (n_points < 2) throw std::invalid_argument("n_points must be >= 2");
  const TwoFundScalars sc = two_fund_scalars(mu, sigma);
  if (sc.d <= 1e-12 * sc.b * sc.c) {
    throw std::domain_error("degenerate frontier: D <= 1e-12*B*C");
  }
  const double lo = sc.a / sc.c;
  const double hi = *std::max_element(mu.begin(), mu.end());
  FrontierResult out;
  out.targets = frontier_grid(lo, hi, n_points);
  out.weights = Matrix(static_cast<std::size_t>(n_points), mu.size());
  out.variances = Vector(static_cast<std::size_t>(n_points), 0.0);
  for (std::size_t k = 0; k < out.targets.size(); ++k) {
    const Vector w = frontier_weights(mu, sigma, out.targets[k]);
    for (std::size_t i = 0; i < w.size(); ++i) out.weights(k, i) = w[i];
    out.variances[k] = quad_form(w, sigma);
  }
  return out;
}

Vector project_simplex_box(const Vector& v, double total, const Vector& lb,
                           const Vector& ub) {
  const std::size_t n = v.size();
  if (n == 0) throw std::invalid_argument("cannot project an empty vector");
  for (double x : v) {
    if (!std::isfinite(x)) {
      throw std::invalid_argument("projection input contains NaN or infinite entries");
    }
  }
  if (!std::isfinite(total)) throw std::invalid_argument("total must be finite");

  Vector lo = lb.empty() ? Vector(n, 0.0) : lb;
  Vector hi = ub.empty() ? Vector(n, std::numeric_limits<double>::infinity()) : ub;
  if (lo.size() != n || hi.size() != n) {
    throw std::invalid_argument("bounds must match the vector length");
  }
  for (std::size_t i = 0; i < n; ++i) {
    if (std::isnan(lo[i]) || std::isnan(hi[i])) {
      throw std::invalid_argument("bounds contain NaN");
    }
    if (lo[i] > hi[i]) throw std::invalid_argument("infeasible bounds: some lb_i > ub_i");
  }
  const double lo_sum = np_sum(lo);
  const double hi_sum = np_sum(hi);
  const double feas_tol = 1e-12 * (1.0 + std::fabs(total));
  if (lo_sum > total + feas_tol) {
    std::ostringstream msg;
    msg << "infeasible bounds: sum(lb) = " << lo_sum << " exceeds total " << total;
    throw std::invalid_argument(msg.str());
  }
  if (hi_sum < total - feas_tol) {
    std::ostringstream msg;
    msg << "infeasible bounds: sum(ub) = " << hi_sum << " is below total " << total;
    throw std::invalid_argument(msg.str());
  }

  // g(t) = sum(clip(v - t, lo, hi)), summed in numpy pairwise order to
  // match the reference bit for bit (the PGD path depends on it).
  const auto g = [&](double t) {
    Vector c(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) c[i] = clip(v[i] - t, lo[i], hi[i]);
    return np_sum(c);
  };
  const auto clipped = [&](double t) {
    Vector w(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) w[i] = clip(v[i] - t, lo[i], hi[i]);
    return w;
  };

  // Breakpoints of the piecewise-linear, non-increasing g(t).
  std::vector<double> bps;
  bps.reserve(2 * n);
  for (std::size_t i = 0; i < n; ++i) bps.push_back(v[i] - lo[i]);
  for (std::size_t i = 0; i < n; ++i) {
    if (std::isfinite(hi[i])) bps.push_back(v[i] - hi[i]);
  }
  std::sort(bps.begin(), bps.end());

  const double t_first = bps.front();
  const double g_first = g(t_first);
  if (g_first < total) {
    // t* lies left of every breakpoint; there the unclipped (infinite-ub)
    // coordinates give slope -k with k = number of infinite upper bounds.
    std::size_t k = 0;
    for (std::size_t i = 0; i < n; ++i) {
      if (!std::isfinite(hi[i])) ++k;
    }
    if (k == 0) {
      // sum(ub) >= total guarantees g_first >= total up to rounding; snap.
      return clipped(t_first);
    }
    return clipped(t_first - (total - g_first) / static_cast<double>(k));
  }

  double t_prev = t_first;
  double g_prev = g_first;
  for (std::size_t idx = 1; idx < bps.size(); ++idx) {
    const double t_cur = bps[idx];
    const double g_cur = g(t_cur);
    if (g_cur <= total) {
      const double t_star =
          (g_prev == g_cur)
              ? t_cur
              : t_prev + (g_prev - total) * (t_cur - t_prev) / (g_prev - g_cur);
      return clipped(t_star);
    }
    t_prev = t_cur;
    g_prev = g_cur;
  }
  // Right of every breakpoint g(t) = sum(lb) <= total; feasibility means equality.
  return clipped(bps.back());
}

Vector project_simplex_box(const Vector& v, double total, double lb, double ub) {
  return project_simplex_box(v, total, Vector(v.size(), lb), Vector(v.size(), ub));
}

PgdResult pgd_minimize(const std::function<double(const Vector&)>& fun,
                       const std::function<Vector(const Vector&)>& grad,
                       const std::function<Vector(const Vector&)>& project,
                       const Vector& w0, double tol, int max_iter) {
  Vector w = project(w0);
  const std::size_t n = w.size();
  double t = 1.0;
  for (int it = 1; it <= max_iter; ++it) {
    t *= 2.0;
    const Vector g = grad(w);
    const double fw = fun(w);
    Vector w_new;
    Vector dw(n, 0.0);
    bool accepted = false;
    for (int halving = 0; halving < 100; ++halving) {
      Vector cand(n, 0.0);
      for (std::size_t i = 0; i < n; ++i) cand[i] = w[i] - t * g[i];
      w_new = project(cand);
      for (std::size_t i = 0; i < n; ++i) dw[i] = w_new[i] - w[i];
      if (fun(w_new) <= fw + fma_dot(g, dw) + fma_dot(dw, dw) / (2.0 * t)) {
        accepted = true;
        break;
      }
      t *= 0.5;
    }
    if (!accepted) {
      // Pinned: after 100 halvings accept the candidate at the final step.
      Vector cand(n, 0.0);
      for (std::size_t i = 0; i < n; ++i) cand[i] = w[i] - t * g[i];
      w_new = project(cand);
      for (std::size_t i = 0; i < n; ++i) dw[i] = w_new[i] - w[i];
    }
    double step = 0.0;
    for (std::size_t i = 0; i < n; ++i) step = std::max(step, std::fabs(dw[i]));
    w = w_new;
    if (step < tol) return PgdResult{w, it, true};
  }
  return PgdResult{w, max_iter, false};
}

namespace {

/// Common PGD setup: Cholesky validity gate + projection + equal-weight start.
void pgd_setup(const Matrix& sigma, const Vector& lb, const Vector& ub,
               std::function<Vector(const Vector&)>& project, Vector& w0) {
  cholesky(sigma);  // PSD/validity gate; throws on bad input
  const std::size_t n = sigma.rows();
  project = [lb, ub](const Vector& v) { return project_simplex_box(v, 1.0, lb, ub); };
  w0 = Vector(n, 1.0 / static_cast<double>(n));
}

}  // namespace

Vector min_variance_longonly(const Matrix& sigma, const Vector& lb, const Vector& ub,
                             double tol, int max_iter) {
  std::function<Vector(const Vector&)> project;
  Vector w0;
  pgd_setup(sigma, lb, ub, project, w0);
  const auto fun = [&sigma](const Vector& w) { return 0.5 * np_quad(w, sigma); };
  const auto grad = [&sigma](const Vector& w) { return np_matvec(sigma, w); };
  return pgd_minimize(fun, grad, project, w0, tol, max_iter).w;
}

Vector constrained_frontier_weights(const Vector& mu, const Matrix& sigma, double target,
                                    double penalty, const Vector& lb, const Vector& ub,
                                    double tol, int max_iter) {
  validate_mu_sigma(mu, sigma);
  if (!(penalty > 0.0)) throw std::invalid_argument("penalty must be positive");
  std::function<Vector(const Vector&)> project;
  Vector w0;
  pgd_setup(sigma, lb, ub, project, w0);
  const auto fun = [&](const Vector& w) {
    const double gap = fma_dot(mu, w) - target;
    return 0.5 * np_quad(w, sigma) + 0.5 * penalty * gap * gap;
  };
  const auto grad = [&](const Vector& w) {
    const double gap = fma_dot(mu, w) - target;
    const double pg = penalty * gap;
    Vector g = np_matvec(sigma, w);
    for (std::size_t i = 0; i < g.size(); ++i) g[i] += pg * mu[i];
    return g;
  };
  return pgd_minimize(fun, grad, project, w0, tol, max_iter).w;
}

FrontierResult constrained_frontier(const Vector& mu, const Matrix& sigma, int n_points,
                                    double penalty, const Vector& lb, const Vector& ub) {
  validate_mu_sigma(mu, sigma);
  if (n_points < 2) throw std::invalid_argument("n_points must be >= 2");
  const TwoFundScalars sc = two_fund_scalars(mu, sigma);
  const double lo = sc.a / sc.c;
  const double hi = *std::max_element(mu.begin(), mu.end());
  FrontierResult out;
  out.targets = frontier_grid(lo, hi, n_points);
  out.weights = Matrix(static_cast<std::size_t>(n_points), mu.size());
  out.variances = Vector(static_cast<std::size_t>(n_points), 0.0);
  for (std::size_t k = 0; k < out.targets.size(); ++k) {
    const Vector w =
        constrained_frontier_weights(mu, sigma, out.targets[k], penalty, lb, ub);
    for (std::size_t i = 0; i < w.size(); ++i) out.weights(k, i) = w[i];
    out.variances[k] = quad_form(w, sigma);
  }
  return out;
}

}  // namespace portopt
