#include "portopt/markowitz.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace portopt {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

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

void validate_1x1(const Matrix& sigma) {
  if (!(sigma(0, 0) > 0.0) || !std::isfinite(sigma(0, 0))) {
    throw std::invalid_argument("1x1 covariance must be positive and finite");
  }
}

double clip(double v, double lo, double hi) { return std::min(std::max(v, lo), hi); }

/// Plain sequential sum, mirroring the reference's explicit loops.
double vec_sum(const Vector& v) {
  double s = 0.0;
  for (double x : v) s += x;
  return s;
}

double quad_form(const Vector& w, const Matrix& s) { return dot(w, mat_vec(s, w)); }

/// Frontier generators g, h and the min-variance return A/C (n >= 2).
struct TwoFundGh {
  Vector g;
  Vector h;
  double mv_ret;
};

TwoFundGh two_fund_gh(const Vector& mu, const Matrix& sigma) {
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
  const std::size_t n = mu.size();
  TwoFundGh out{Vector(n, 0.0), Vector(n, 0.0), a / c};
  for (std::size_t i = 0; i < n; ++i) {
    out.g[i] = (b * y[i] - a * x[i]) / d;
    out.h[i] = (c * x[i] - a * y[i]) / d;
  }
  return out;
}

/// Pinned grid: n_points targets from lo to hi inclusive (last exactly hi).
Vector frontier_grid(double lo, double hi, int n_points, const char* what) {
  if (hi <= lo + 1e-12 * (1.0 + std::fabs(lo))) {
    std::ostringstream msg;
    msg << "frontier grid empty: " << what << " (" << hi
        << ") does not exceed the minimum-variance return (" << lo
        << "); supply targets explicitly via frontier_weights / "
           "constrained_frontier_weights";
    throw std::domain_error(msg.str());
  }
  Vector targets(static_cast<std::size_t>(n_points), 0.0);
  for (int k = 0; k < n_points; ++k) {
    targets[static_cast<std::size_t>(k)] =
        lo + static_cast<double>(k) * (hi - lo) / static_cast<double>(n_points - 1);
  }
  targets.back() = hi;
  return targets;
}

/// Broadcast/validate bounds (API_SPEC 3.5 feasibility gate).
void broadcast_bounds(std::size_t n, const Vector& lb, const Vector& ub, double total,
                      Vector& lo, Vector& hi) {
  lo = lb.empty() ? Vector(n, 0.0) : lb;
  hi = ub.empty() ? Vector(n, kInf) : ub;
  if (lo.size() != n || hi.size() != n) {
    throw std::invalid_argument("bounds must match the vector length");
  }
  for (std::size_t i = 0; i < n; ++i) {
    if (std::isnan(lo[i]) || std::isnan(hi[i])) {
      throw std::invalid_argument("bounds contain NaN");
    }
    if (!std::isfinite(lo[i])) throw std::invalid_argument("lower bounds must be finite");
    if (hi[i] == -kInf) throw std::invalid_argument("upper bounds must be > -inf");
    if (lo[i] > hi[i]) throw std::invalid_argument("infeasible bounds: some lb_i > ub_i");
  }
  const double lo_sum = vec_sum(lo);
  const double hi_sum = vec_sum(hi);
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
}

/// Vertex of {sum w = total, lb <= w <= ub} maximizing (descending) or
/// minimizing (ascending) mu'w: fill from lb in mu order (ties by index).
Vector greedy_extreme(const Vector& mu, const Vector& lo, const Vector& hi, double total,
                      bool descending) {
  const std::size_t n = mu.size();
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    if (mu[a] != mu[b]) return descending ? mu[a] > mu[b] : mu[a] < mu[b];
    return a < b;
  });
  Vector w(lo);
  double remaining = total - kahan_sum(lo);
  if (remaining < 0.0) remaining = 0.0;
  for (std::size_t i : order) {
    if (remaining <= 0.0) break;
    const double room = hi[i] - lo[i];
    const double add = room > remaining ? remaining : room;
    w[i] = lo[i] + add;
    remaining -= add;
  }
  return w;
}

/// Constraint kinds of the dual active-set QP.
enum class Kind { kEquality, kLower, kUpper, kFixed };

struct Constraint {
  Kind kind;
  std::size_t idx;
  double rhs;
  Vector normal;
};

/// Goldfarb-Idnani dual active-set method for min 1/2 w'Sw s.t.
/// a_eq w = b_eq, lo <= w <= hi (API_SPEC 3.6). Constraints are
/// n_c'w >= b_c: the equalities (rows of a_eq, then every fixed coordinate
/// lo_i == hi_i; added first, multipliers unrestricted), then the lower
/// bounds and the finite upper bounds of the non-fixed coordinates.
QpResult dual_active_set_qp(const Matrix& s, const std::vector<Vector>& a_eq,
                            const Vector& b_eq, const Vector& lo, const Vector& hi,
                            int max_iter) {
  const std::size_t n = s.rows();
  const Matrix low = cholesky(s);
  std::vector<Constraint> cons;
  for (std::size_t j = 0; j < a_eq.size(); ++j) {
    cons.push_back(Constraint{Kind::kEquality, j, b_eq[j], a_eq[j]});
  }
  for (std::size_t i = 0; i < n; ++i) {
    if (lo[i] == hi[i]) {
      Vector e(n, 0.0);
      e[i] = 1.0;
      cons.push_back(Constraint{Kind::kFixed, i, lo[i], e});
    }
  }
  const std::size_t meq = cons.size();
  for (std::size_t i = 0; i < n; ++i) {
    if (lo[i] != hi[i]) {
      Vector e(n, 0.0);
      e[i] = 1.0;
      cons.push_back(Constraint{Kind::kLower, i, lo[i], e});
    }
  }
  for (std::size_t i = 0; i < n; ++i) {
    if (lo[i] != hi[i] && std::isfinite(hi[i])) {
      Vector e(n, 0.0);
      e[i] = -1.0;
      cons.push_back(Constraint{Kind::kUpper, i, -hi[i], e});
    }
  }
  const std::size_t n_cons = cons.size();

  Vector w(n, 0.0);
  std::vector<std::size_t> active;
  Vector u;
  std::vector<bool> skipped(n_cons, false);
  int iterations = 0;
  long p = -1;
  double u_p = 0.0;
  const auto is_active = [&](std::size_t c) {
    return std::find(active.begin(), active.end(), c) != active.end();
  };
  while (iterations < max_iter) {
    ++iterations;
    if (p < 0) {
      // (1) pick the next constraint: equalities first, then the most
      // violated inequality (ties -> lowest index).
      if (active.size() < meq) {
        p = static_cast<long>(active.size());
      } else {
        double worst = 0.0;
        for (std::size_t c = meq; c < n_cons; ++c) {
          if (skipped[c] || is_active(c)) continue;
          const double viol = dot(cons[c].normal, w) - cons[c].rhs;
          if (viol < -kQpFeasTol * (1.0 + std::fabs(cons[c].rhs)) && viol < worst) {
            worst = viol;
            p = static_cast<long>(c);
          }
        }
        if (p < 0) break;  // no violated constraint: dual method finished
      }
      u_p = 0.0;
    }
    const std::size_t pc = static_cast<std::size_t>(p);
    // (2) primal / dual step directions for constraint p.
    const std::size_t q = active.size();
    std::vector<Vector> ys(q);
    for (std::size_t i = 0; i < q; ++i) ys[i] = chol_solve(low, cons[active[i]].normal);
    const Vector v = chol_solve(low, cons[pc].normal);
    Vector r(q, 0.0);
    if (q > 0) {
      Matrix gram(q, q, 0.0);
      for (std::size_t i = 0; i < q; ++i) {
        for (std::size_t j = i; j < q; ++j) {
          gram(i, j) = dot(cons[active[i]].normal, ys[j]);
          gram(j, i) = gram(i, j);
        }
      }
      Vector nv(q, 0.0);
      for (std::size_t i = 0; i < q; ++i) nv[i] = dot(cons[active[i]].normal, v);
      r = chol_solve(cholesky(gram), nv);
    }
    Vector z(v);
    for (std::size_t i = 0; i < q; ++i) {
      for (std::size_t k = 0; k < n; ++k) z[k] -= r[i] * ys[i][k];
    }
    const double zn = dot(z, cons[pc].normal);
    const double vn = dot(v, cons[pc].normal);
    const double s_p = dot(cons[pc].normal, w) - cons[pc].rhs;
    // (3) step lengths: t1 keeps inequality multipliers >= 0, t2 satisfies p.
    double t1 = kInf;
    long j_block = -1;
    for (std::size_t i = 0; i < q; ++i) {
      if (active[i] >= meq && r[i] > 0.0) {
        const double cand = u[i] / r[i];
        if (cand < t1) {
          t1 = cand;
          j_block = static_cast<long>(i);
        }
      }
    }
    const double t2 = (zn > 1e-10 * vn) ? -s_p / zn : kInf;
    const double t = std::min(t1, t2);
    if (!std::isfinite(t)) {
      // n_p is a combination of the active normals and no multiplier
      // blocks: accept a rounding-level violation at a primal-degenerate
      // vertex as satisfied, else give up (reported via the certificate).
      if (std::fabs(s_p) <= kQpDegenerateTol * (1.0 + std::fabs(cons[pc].rhs))) {
        skipped[pc] = true;
        p = -1;
        continue;
      }
      break;
    }
    if (!std::isfinite(t2)) {
      // Dependent normal: dual step only, drop the blocking constraint.
      for (std::size_t i = 0; i < q; ++i) u[i] -= t * r[i];
      u_p += t;
      active.erase(active.begin() + j_block);
      u.erase(u.begin() + j_block);
      continue;
    }
    for (std::size_t k = 0; k < n; ++k) w[k] += t * z[k];
    for (std::size_t i = 0; i < q; ++i) u[i] -= t * r[i];
    u_p += t;
    if (t == t2) {
      active.push_back(pc);
      u.push_back(u_p);
      p = -1;
    } else {
      active.erase(active.begin() + j_block);
      u.erase(u.begin() + j_block);
    }
  }
  // Polish: minimizer on the final active set in closed form,
  // w = S^-1 N u with (N' S^-1 N) u = b_active, plus one step of iterative
  // refinement on the equality residual.
  const std::size_t q = active.size();
  if (q > 0) {
    std::vector<Vector> ys(q);
    for (std::size_t i = 0; i < q; ++i) ys[i] = chol_solve(low, cons[active[i]].normal);
    Matrix gram(q, q, 0.0);
    for (std::size_t i = 0; i < q; ++i) {
      for (std::size_t j = i; j < q; ++j) {
        gram(i, j) = dot(cons[active[i]].normal, ys[j]);
        gram(j, i) = gram(i, j);
      }
    }
    const Matrix low_g = cholesky(gram);
    Vector b_act(q, 0.0);
    for (std::size_t i = 0; i < q; ++i) b_act[i] = cons[active[i]].rhs;
    const Vector u_arr = chol_solve(low_g, b_act);
    w.assign(n, 0.0);
    for (std::size_t i = 0; i < q; ++i) {
      for (std::size_t k = 0; k < n; ++k) w[k] += u_arr[i] * ys[i][k];
    }
    Vector resid(q, 0.0);
    for (std::size_t i = 0; i < q; ++i) resid[i] = b_act[i] - dot(cons[active[i]].normal, w);
    const Vector du = chol_solve(low_g, resid);
    for (std::size_t i = 0; i < q; ++i) {
      for (std::size_t k = 0; k < n; ++k) w[k] += du[i] * ys[i][k];
    }
    u.assign(q, 0.0);
    for (std::size_t i = 0; i < q; ++i) u[i] = u_arr[i] + du[i];
  }
  for (std::size_t c : active) {
    if (cons[c].kind == Kind::kLower || cons[c].kind == Kind::kFixed) {
      w[cons[c].idx] = lo[cons[c].idx];
    } else if (cons[c].kind == Kind::kUpper) {
      w[cons[c].idx] = hi[cons[c].idx];
    }
  }
  // KKT certificate on the polished point.
  double feas = 0.0;
  double rhs_scale = 0.0;
  for (std::size_t c = 0; c < n_cons; ++c) {
    const double viol = dot(cons[c].normal, w) - cons[c].rhs;
    rhs_scale = std::max(rhs_scale, std::fabs(cons[c].rhs));
    feas = std::max(feas, c < meq ? std::fabs(viol) : -viol);
  }
  Vector stat = mat_vec(s, w);
  double stat_scale = 0.0;
  for (double v : stat) stat_scale = std::max(stat_scale, std::fabs(v));
  for (std::size_t i = 0; i < q; ++i) {
    double nmax = 0.0;
    for (std::size_t k = 0; k < n; ++k) {
      stat[k] -= u[i] * cons[active[i]].normal[k];
      nmax = std::max(nmax, std::fabs(cons[active[i]].normal[k]));
    }
    stat_scale += std::fabs(u[i]) * nmax;
  }
  double stat_res = 0.0;
  for (double v : stat) stat_res = std::max(stat_res, std::fabs(v));
  const bool converged =
      feas <= kQpConvTol * (1.0 + rhs_scale) && stat_res <= kQpConvTol * (1.0 + stat_scale);
  for (std::size_t k = 0; k < n; ++k) w[k] = clip(w[k], lo[k], hi[k]);
  return QpResult{w, iterations, converged, std::max(feas, stat_res)};
}

/// Shared QP setup: square + SPD gate, bound validation, iteration cap.
void qp_setup(const Matrix& sigma, const Vector& lb, const Vector& ub, int max_iter,
              Vector& lo, Vector& hi) {
  if (sigma.rows() != sigma.cols()) {
    std::ostringstream msg;
    msg << "sigma must be square, got shape (" << sigma.rows() << ", " << sigma.cols()
        << ")";
    throw std::invalid_argument(msg.str());
  }
  cholesky(sigma);  // SPD/validity gate; throws on bad input
  if (max_iter < 1) throw std::invalid_argument("max_iter must be >= 1");
  broadcast_bounds(sigma.rows(), lb, ub, 1.0, lo, hi);
}

Vector require_converged(const QpResult& res, const char* what) {
  if (!res.converged) {
    std::ostringstream msg;
    msg << what << ": active-set QP did not converge within " << res.iterations
        << " iterations (KKT residual " << res.kkt_residual
        << "); the input is degenerate — use the *_solve variant to inspect the iterate";
    throw std::domain_error(msg.str());
  }
  return res.weights;
}

QpResult frontier_point(const Vector& mu, const Matrix& sigma, double tgt, const Vector& lo,
                        const Vector& hi, const Vector& w_gmv, double r_gmv, double r_min,
                        double r_max, int max_iter) {
  const double tol = 1e-12 * (1.0 + std::fabs(tgt));
  if (tgt < r_gmv - tol) {
    std::ostringstream msg;
    msg << "target return " << tgt << " lies below the constrained minimum-variance return "
        << r_gmv << "; the constrained efficient frontier is defined for targets in ["
        << r_gmv << ", " << r_max << "]";
    throw std::domain_error(msg.str());
  }
  if (tgt > r_max + tol) {
    std::ostringstream msg;
    msg << "target return " << tgt << " is not attainable under the bounds (attainable range ["
        << r_min << ", " << r_max << "])";
    throw std::domain_error(msg.str());
  }
  if (std::fabs(tgt - r_gmv) <= tol || r_max - r_min <= tol) {
    return QpResult{w_gmv, 0, true, 0.0};
  }
  const std::size_t n = mu.size();
  if (std::fabs(tgt - r_max) <= tol) {
    // Maximum-return endpoint: reduce the degenerate face analytically
    // (assets above the greedy cut level at their caps, below it at their
    // floors, tied assets free) and solve the sum-only min-variance QP.
    const Vector w_max = greedy_extreme(mu, lo, hi, 1.0, true);
    bool any = false;
    double cut = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      if (w_max[i] > lo[i]) {
        cut = any ? std::min(cut, mu[i]) : mu[i];
        any = true;
      }
    }
    if (!any) return QpResult{w_gmv, 0, true, 0.0};
    Vector lo2(lo);
    Vector hi2(hi);
    for (std::size_t i = 0; i < n; ++i) {
      if (mu[i] > cut) {
        lo2[i] = hi[i];
      } else if (mu[i] < cut) {
        hi2[i] = lo[i];
      }
    }
    QpResult res = dual_active_set_qp(sigma, {Vector(n, 1.0)}, Vector{1.0}, lo2, hi2, max_iter);
    res.kkt_residual = std::max(res.kkt_residual, std::fabs(dot(mu, res.weights) - tgt));
    return res;
  }
  return dual_active_set_qp(sigma, {Vector(n, 1.0), mu}, Vector{1.0, tgt}, lo, hi, max_iter);
}

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
    validate_1x1(sigma);
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
    validate_1x1(sigma);
    if (mu[0] <= rf) {
      throw TangencyUndefinedError(
          "tangency undefined: risk-free rate >= expected return of the single asset");
    }
    return Vector{1.0};
  }
  const Matrix low = cholesky(sigma);
  Vector excess(mu);
  for (double& v : excess) v -= rf;
  const Vector z = chol_solve(low, excess);
  const double denom = vec_sum(z);  // = A - rf*C
  if (denom <= 0.0) {
    throw TangencyUndefinedError(
        "tangency undefined: A - rf*C <= 0 (risk-free rate is at or above the "
        "minimum-variance portfolio return)");
  }
  // Conditioning guard: A - rf*C is a difference of two O(A) numbers.
  const double a = vec_sum(chol_solve(low, mu));
  const double c = vec_sum(chol_solve(low, Vector(mu.size(), 1.0)));
  const double scale = std::max(std::fabs(a), std::fabs(rf) * c);
  if (denom <= kTangencyCondRtol * scale) {
    std::ostringstream msg;
    msg << "tangency ill-conditioned: A - rf*C = " << denom << " is within a relative "
        << kTangencyCondRtol << " of cancelling (rf within " << kTangencyCondRtol
        << " of the minimum-variance return " << a / c
        << "); weights would carry gross leverage of order 1e6 or more";
    throw TangencyUndefinedError(msg.str());
  }
  Vector w(z);
  for (double& v : w) v /= denom;
  return w;
}

double gross_leverage(const Vector& w) {
  if (w.empty()) throw std::invalid_argument("weights must be a non-empty finite vector");
  Vector a(w.size(), 0.0);
  for (std::size_t i = 0; i < w.size(); ++i) {
    if (!std::isfinite(w[i])) {
      throw std::invalid_argument("weights must be a non-empty finite vector");
    }
    a[i] = std::fabs(w[i]);
  }
  return kahan_sum(a);
}

Vector frontier_weights(const Vector& mu, const Matrix& sigma, double target) {
  validate_mu_sigma(mu, sigma);
  if (!std::isfinite(target)) {
    throw std::invalid_argument("target return must be finite");
  }
  if (sigma.rows() == 1) {
    validate_1x1(sigma);
    if (std::fabs(target - mu[0]) > 1e-12 * (1.0 + std::fabs(mu[0]))) {
      throw std::domain_error(
          "n=1: target return must equal the single asset's expected return");
    }
    return Vector{1.0};
  }
  const TwoFundGh gh = two_fund_gh(mu, sigma);
  if (target < gh.mv_ret - 1e-12 * (1.0 + std::fabs(gh.mv_ret))) {
    std::ostringstream msg;
    msg << "target return " << target << " lies below the minimum-variance return "
        << gh.mv_ret << "; the efficient frontier is defined for targets >= A/C";
    throw std::domain_error(msg.str());
  }
  const std::size_t n = mu.size();
  Vector w(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) w[i] = gh.g[i] + gh.h[i] * target;
  return w;
}

FrontierResult efficient_frontier(const Vector& mu, const Matrix& sigma, int n_points) {
  validate_mu_sigma(mu, sigma);
  if (n_points < 2) throw std::invalid_argument("n_points must be >= 2");
  if (sigma.rows() == 1) throw std::invalid_argument("frontier sweep requires at least 2 assets");
  const TwoFundGh gh = two_fund_gh(mu, sigma);
  const double hi = *std::max_element(mu.begin(), mu.end());
  FrontierResult out;
  out.targets = frontier_grid(gh.mv_ret, hi, n_points, "max(mu)");
  out.weights = Matrix(static_cast<std::size_t>(n_points), mu.size());
  out.variances = Vector(static_cast<std::size_t>(n_points), 0.0);
  for (std::size_t k = 0; k < out.targets.size(); ++k) {
    Vector w(mu.size(), 0.0);
    for (std::size_t i = 0; i < w.size(); ++i) w[i] = gh.g[i] + gh.h[i] * out.targets[k];
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
  Vector lo;
  Vector hi;
  broadcast_bounds(n, lb, ub, total, lo, hi);

  const auto clipped = [&](double t) {
    Vector w(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) w[i] = clip(v[i] - t, lo[i], hi[i]);
    return w;
  };
  // g(t) = sum(clip(v - t, lo, hi)) with a compensated sum.
  const auto g = [&](double t) { return kahan_sum(clipped(t)); };

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

std::pair<double, double> attainable_return_range(const Vector& mu, const Vector& lb,
                                                  const Vector& ub) {
  if (mu.empty()) throw std::invalid_argument("mu must be a non-empty finite vector");
  for (double v : mu) {
    if (!std::isfinite(v)) throw std::invalid_argument("mu must be a non-empty finite vector");
  }
  Vector lo;
  Vector hi;
  broadcast_bounds(mu.size(), lb, ub, 1.0, lo, hi);
  const Vector w_min = greedy_extreme(mu, lo, hi, 1.0, false);
  const Vector w_max = greedy_extreme(mu, lo, hi, 1.0, true);
  return {dot(mu, w_min), dot(mu, w_max)};
}

QpResult min_variance_longonly_solve(const Matrix& sigma, const Vector& lb, const Vector& ub,
                                     int max_iter) {
  Vector lo;
  Vector hi;
  qp_setup(sigma, lb, ub, max_iter, lo, hi);
  const std::size_t n = sigma.rows();
  return dual_active_set_qp(sigma, {Vector(n, 1.0)}, Vector{1.0}, lo, hi, max_iter);
}

Vector min_variance_longonly(const Matrix& sigma, const Vector& lb, const Vector& ub,
                             int max_iter) {
  return require_converged(min_variance_longonly_solve(sigma, lb, ub, max_iter),
                           "min_variance_longonly");
}

QpResult constrained_frontier_solve(const Vector& mu, const Matrix& sigma, double target,
                                    const Vector& lb, const Vector& ub, int max_iter) {
  validate_mu_sigma(mu, sigma);
  Vector lo;
  Vector hi;
  qp_setup(sigma, lb, ub, max_iter, lo, hi);
  if (!std::isfinite(target)) throw std::invalid_argument("target return must be finite");
  const Vector w_gmv = require_converged(min_variance_longonly_solve(sigma, lo, hi, max_iter),
                                         "constrained_frontier_solve (min-variance point)");
  const auto range = attainable_return_range(mu, lo, hi);
  return frontier_point(mu, sigma, target, lo, hi, w_gmv, dot(mu, w_gmv), range.first,
                        range.second, max_iter);
}

Vector constrained_frontier_weights(const Vector& mu, const Matrix& sigma, double target,
                                    const Vector& lb, const Vector& ub, int max_iter) {
  return require_converged(constrained_frontier_solve(mu, sigma, target, lb, ub, max_iter),
                           "constrained_frontier_weights");
}

FrontierResult constrained_frontier(const Vector& mu, const Matrix& sigma, int n_points,
                                    const Vector& lb, const Vector& ub, int max_iter) {
  validate_mu_sigma(mu, sigma);
  if (n_points < 2) throw std::invalid_argument("n_points must be >= 2");
  Vector lo;
  Vector hi;
  qp_setup(sigma, lb, ub, max_iter, lo, hi);
  const Vector w_gmv = require_converged(min_variance_longonly_solve(sigma, lo, hi, max_iter),
                                         "constrained_frontier (min-variance point)");
  const double r_gmv = dot(mu, w_gmv);
  const auto range = attainable_return_range(mu, lo, hi);
  FrontierResult out;
  out.targets = frontier_grid(r_gmv, range.second, n_points, "maximum attainable return");
  out.weights = Matrix(static_cast<std::size_t>(n_points), mu.size());
  out.variances = Vector(static_cast<std::size_t>(n_points), 0.0);
  for (std::size_t k = 0; k < out.targets.size(); ++k) {
    const QpResult res = frontier_point(mu, sigma, out.targets[k], lo, hi, w_gmv, r_gmv,
                                        range.first, range.second, max_iter);
    std::ostringstream what;
    what << "constrained_frontier point " << k << " (target " << out.targets[k] << ")";
    const Vector w = require_converged(res, what.str().c_str());
    for (std::size_t i = 0; i < w.size(); ++i) out.weights(k, i) = w[i];
    out.variances[k] = quad_form(w, sigma);
  }
  return out;
}

}  // namespace portopt
