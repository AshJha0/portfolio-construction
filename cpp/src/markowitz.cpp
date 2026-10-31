#include "portopt/markowitz.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace portopt {

namespace detail {

KktCertificate kkt_certificate(const Matrix& s, const std::vector<Vector>& normals,
                               const Vector& rhs, std::size_t meq,
                               const std::vector<std::size_t>& active, const Vector& u,
                               const Vector& w) {
  const std::size_t n = s.rows();
  double feas = 0.0;
  double rhs_scale = 0.0;
  for (std::size_t c = 0; c < normals.size(); ++c) {
    const double viol = dot(normals[c], w) - rhs[c];
    rhs_scale = std::max(rhs_scale, std::fabs(rhs[c]));
    feas = std::max(feas, c < meq ? std::fabs(viol) : -viol);
  }
  Vector stat = mat_vec(s, w);
  double stat_scale = 0.0;
  for (double v : stat) stat_scale = std::max(stat_scale, std::fabs(v));
  double dual = 0.0;
  for (std::size_t i = 0; i < active.size(); ++i) {
    const Vector& nc = normals[active[i]];
    double nmax = 0.0;
    for (std::size_t k = 0; k < n; ++k) {
      stat[k] -= u[i] * nc[k];
      nmax = std::max(nmax, std::fabs(nc[k]));
    }
    stat_scale += std::fabs(u[i]) * nmax;
    if (active[i] >= meq) dual = std::max(dual, -u[i]);
  }
  double stat_res = 0.0;
  for (double v : stat) stat_res = std::max(stat_res, std::fabs(v));
  const double stat_tol = kQpConvTol * (1.0 + stat_scale);
  const bool certified = feas <= kQpConvTol * (1.0 + rhs_scale) && stat_res <= stat_tol &&
                         dual <= stat_tol;
  return KktCertificate{certified, std::max({feas, stat_res, dual})};
}

}  // namespace detail

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

/// Constraint kinds of the dual active-set QP (kInequality = a general
/// row a'x >= b, placed after the bounds).
enum class Kind { kEquality, kLower, kUpper, kFixed, kInequality };

struct Constraint {
  Kind kind;
  std::size_t idx;
  double rhs;
  Vector normal;
};

/// Final state of the dual active-set loop.
struct GiState {
  Vector w;
  std::vector<std::size_t> active;
  Vector u;
  int iterations;
  std::vector<Constraint> cons;
  std::size_t meq;
  Matrix low;
};

/// Goldfarb-Idnani loop for min 1/2 x'Sx - lin'x s.t. a_eq x = b_eq,
/// a_in x >= b_in, lo <= x <= hi (API_SPEC 3.6, 3.7). An empty `lin`
/// means no linear term (start at x = 0); otherwise the loop starts at the
/// unconstrained minimizer S^-1 lin and is unchanged (the step directions
/// do not depend on lin). Constraint order: equality rows, fixed
/// coordinates, lower bounds, finite upper bounds, inequality rows.
GiState gi_loop(const Matrix& s, const Vector& lin, const std::vector<Vector>& a_eq,
                const Vector& b_eq, const std::vector<Vector>& a_in, const Vector& b_in,
                const Vector& lo, const Vector& hi, int max_iter) {
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
  for (std::size_t j = 0; j < a_in.size(); ++j) {
    cons.push_back(Constraint{Kind::kInequality, j, b_in[j], a_in[j]});
  }
  const std::size_t n_cons = cons.size();

  Vector w = lin.empty() ? Vector(n, 0.0) : chol_solve(low, lin);
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
  return GiState{w, active, u, iterations, cons, meq, low};
}

/// Goldfarb-Idnani dual active-set method for min 1/2 w'Sw s.t.
/// a_eq w = b_eq, lo <= w <= hi (API_SPEC 3.6). Constraints are
/// n_c'w >= b_c: the equalities (rows of a_eq, then every fixed coordinate
/// lo_i == hi_i; added first, multipliers unrestricted), then the lower
/// bounds and the finite upper bounds of the non-fixed coordinates.
QpResult dual_active_set_qp(const Matrix& s, const std::vector<Vector>& a_eq,
                            const Vector& b_eq, const Vector& lo, const Vector& hi,
                            int max_iter) {
  const std::size_t n = s.rows();
  GiState st = gi_loop(s, Vector(), a_eq, b_eq, {}, Vector(), lo, hi, max_iter);
  const Matrix& low = st.low;
  const std::vector<Constraint>& cons = st.cons;
  const std::size_t meq = st.meq;
  const std::size_t n_cons = cons.size();
  Vector w = st.w;
  const std::vector<std::size_t>& active = st.active;
  Vector u = st.u;
  const int iterations = st.iterations;
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
  // KKT certificate on the polished point (independent of how the loop
  // terminated).
  std::vector<Vector> normals(n_cons);
  Vector rhs(n_cons, 0.0);
  for (std::size_t c = 0; c < n_cons; ++c) {
    normals[c] = cons[c].normal;
    rhs[c] = cons[c].rhs;
  }
  const detail::KktCertificate cert =
      detail::kkt_certificate(s, normals, rhs, meq, active, u, w);
  for (std::size_t k = 0; k < n; ++k) w[k] = clip(w[k], lo[k], hi[k]);
  return QpResult{w, iterations, cert.certified, cert.residual};
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

Vector require_turnover_converged(const TurnoverQpResult& res, const char* what) {
  if (!res.converged) {
    std::ostringstream msg;
    msg << what << ": cost-aware QP not certified after " << res.iterations
        << " iterations (KKT residual " << res.kkt_residual
        << "); the input is degenerate, use the *_solve variant to inspect the iterate";
    throw std::domain_error(msg.str());
  }
  return res.weights;
}

double abs_diff_sum(const Vector& w, const Vector& w0) {
  Vector a(w.size(), 0.0);
  for (std::size_t i = 0; i < w.size(); ++i) a[i] = std::fabs(w[i] - w0[i]);
  return kahan_sum(a);
}

/// Pinned cost-aware QP (API_SPEC 3.7); inputs already validated.
TurnoverQpResult turnover_qp(const Matrix& s, const Vector& c, const Vector& w0, double kappa,
                             const Vector& lo, const Vector& hi, double cap, int max_iter) {
  const std::size_t n = s.rows();
  if (n == 1) return TurnoverQpResult{Vector{1.0}, 0, true, 0.0, std::fabs(1.0 - w0[0])};
  const Matrix low = cholesky(s);
  // eps = 0.5 / trace(S^-1) <= lambda_min(S) / 2 keeps S - eps I SPD; the
  // split problem is exactly equivalent for any 0 < eps < lambda_min(S).
  double tr = 0.0;
  for (std::size_t j = 0; j < n; ++j) {
    Vector e(n, 0.0);
    e[j] = 1.0;
    tr += chol_solve(low, e)[j];
  }
  const double eps = 0.5 / tr;
  const std::size_t m = 3 * n;
  Matrix h(m, m, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) h(i, j) = s(i, j);
  }
  for (std::size_t i = 0; i < n; ++i) {
    h(i, i) = s(i, i) - eps;
    h(n + i, n + i) = eps;
    h(2 * n + i, 2 * n + i) = eps;
  }
  Vector lin(m, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    lin[i] = c[i] - eps * w0[i];
    lin[n + i] = -kappa;
    lin[2 * n + i] = -kappa;
  }
  std::vector<Vector> a_eq(n + 1, Vector(m, 0.0));
  Vector b_eq(n + 1, 0.0);
  for (std::size_t i = 0; i < n; ++i) a_eq[0][i] = 1.0;
  b_eq[0] = 1.0;
  for (std::size_t i = 0; i < n; ++i) {
    a_eq[1 + i][i] = 1.0;
    a_eq[1 + i][n + i] = -1.0;
    a_eq[1 + i][2 * n + i] = 1.0;
    b_eq[1 + i] = w0[i];
  }
  Vector lo3(m, 0.0);
  Vector hi3(m, kInf);
  for (std::size_t i = 0; i < n; ++i) {
    lo3[i] = lo[i];
    hi3[i] = hi[i];
  }
  std::vector<Vector> a_in;
  Vector b_in;
  if (std::isfinite(cap)) {
    Vector row(m, 0.0);
    for (std::size_t k = n; k < m; ++k) row[k] = -1.0;
    a_in.push_back(row);
    b_in.push_back(-cap);
  }
  const GiState st = gi_loop(h, lin, a_eq, b_eq, a_in, b_in, lo3, hi3, max_iter);

  // States from the active set (fixed at a bound, held at w0, or free on
  // the buy / sell side).
  std::vector<bool> at_lo(n, false);
  std::vector<bool> at_hi(n, false);
  std::vector<bool> b_act(n, false);
  std::vector<bool> s_act(n, false);
  bool cap_act = false;
  for (std::size_t i = 0; i < n; ++i) at_lo[i] = lo[i] == hi[i];
  for (std::size_t cidx : st.active) {
    const Constraint& con = st.cons[cidx];
    if (con.kind == Kind::kInequality) {
      cap_act = true;
    } else if (con.kind == Kind::kUpper) {
      at_hi[con.idx] = true;
    } else if (con.kind == Kind::kLower || con.kind == Kind::kFixed) {
      if (con.idx < n) {
        at_lo[con.idx] = true;
      } else if (con.idx < 2 * n) {
        b_act[con.idx - n] = true;
      } else {
        s_act[con.idx - 2 * n] = true;
      }
    }
  }
  std::vector<bool> is_fixed(n, false);
  Vector fix_val(n, 0.0);
  Vector sgn(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    if (at_lo[i]) {
      is_fixed[i] = true;
      fix_val[i] = lo[i];
    } else if (at_hi[i]) {
      is_fixed[i] = true;
      fix_val[i] = hi[i];
    } else if (b_act[i] && s_act[i]) {
      is_fixed[i] = true;
      fix_val[i] = w0[i];
    } else if (s_act[i]) {
      sgn[i] = 1.0;
    } else if (b_act[i]) {
      sgn[i] = -1.0;
    } else {
      sgn[i] = (st.w[n + i] - st.w[2 * n + i] >= 0.0) ? 1.0 : -1.0;
    }
  }

  // Exact polish in w alone on the face {sum w = 1, [cap row], w_i = fix_i}.
  std::vector<Vector> normals{Vector(n, 1.0)};
  Vector rhs{1.0};
  if (cap_act) {
    double b_cap = -cap;
    Vector row(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      row[i] = -sgn[i];
      if (is_fixed[i]) {
        b_cap += std::fabs(fix_val[i] - w0[i]);
      } else {
        b_cap -= sgn[i] * w0[i];
      }
    }
    normals.push_back(row);
    rhs.push_back(b_cap);
  }
  for (std::size_t i = 0; i < n; ++i) {
    if (is_fixed[i]) {
      Vector e(n, 0.0);
      e[i] = 1.0;
      normals.push_back(e);
      rhs.push_back(fix_val[i]);
    }
  }
  Vector c_eff(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) c_eff[i] = c[i] - kappa * sgn[i];
  const Vector x0 = chol_solve(low, c_eff);
  const std::size_t q = normals.size();
  std::vector<Vector> ys(q);
  for (std::size_t i = 0; i < q; ++i) ys[i] = chol_solve(low, normals[i]);
  Matrix gram(q, q, 0.0);
  for (std::size_t i = 0; i < q; ++i) {
    for (std::size_t j = i; j < q; ++j) {
      gram(i, j) = dot(normals[i], ys[j]);
      gram(j, i) = gram(i, j);
    }
  }
  Matrix low_g;
  try {
    low_g = cholesky(gram);
  } catch (const std::exception&) {
    // Dependent face rows: the loop iterate is returned uncertified.
    Vector w_bad(n, 0.0);
    for (std::size_t k = 0; k < n; ++k) w_bad[k] = clip(st.w[k], lo[k], hi[k]);
    return TurnoverQpResult{w_bad, st.iterations, false, kInf, abs_diff_sum(w_bad, w0)};
  }
  Vector rb(q, 0.0);
  for (std::size_t i = 0; i < q; ++i) rb[i] = rhs[i] - dot(normals[i], x0);
  const Vector u_arr = chol_solve(low_g, rb);
  Vector w(x0);
  for (std::size_t i = 0; i < q; ++i) {
    for (std::size_t k = 0; k < n; ++k) w[k] += u_arr[i] * ys[i][k];
  }
  Vector resid(q, 0.0);
  for (std::size_t i = 0; i < q; ++i) resid[i] = rhs[i] - dot(normals[i], w);
  const Vector du = chol_solve(low_g, resid);
  for (std::size_t i = 0; i < q; ++i) {
    for (std::size_t k = 0; k < n; ++k) w[k] += du[i] * ys[i][k];
  }
  for (std::size_t i = 0; i < n; ++i) {
    if (is_fixed[i]) w[i] = fix_val[i];
  }
  const double nu = u_arr[0] + du[0];
  const double tau = cap_act ? u_arr[1] + du[1] : 0.0;
  const detail::KktCertificate cert =
      detail::turnover_certificate(s, c, w0, kappa, lo, hi, cap, nu, tau, w);
  for (std::size_t k = 0; k < n; ++k) w[k] = clip(w[k], lo[k], hi[k]);
  const double turnover = abs_diff_sum(w, w0);
  return TurnoverQpResult{w, st.iterations, cert.certified, cert.residual, turnover};
}

/// Shared validation of the cost-aware entry points (API_SPEC 3.7).
void turnover_setup(const Matrix& sigma, const Vector& w0, double kappa, const Vector& lb,
                    const Vector& ub, double max_turnover, int max_iter, Vector& lo,
                    Vector& hi) {
  qp_setup(sigma, lb, ub, max_iter, lo, hi);
  const std::size_t n = sigma.rows();
  if (w0.size() != n) {
    std::ostringstream msg;
    msg << "w0 length " << w0.size() << " does not match sigma dimension " << n;
    throw std::invalid_argument(msg.str());
  }
  for (double v : w0) {
    if (!std::isfinite(v)) throw std::invalid_argument("w0 contains NaN or infinite entries");
  }
  if (!std::isfinite(kappa) || kappa < 0.0) {
    std::ostringstream msg;
    msg << "kappa must be finite and >= 0 (got " << kappa << ")";
    throw std::invalid_argument(msg.str());
  }
  if (std::isnan(max_turnover) || max_turnover < 0.0) {
    std::ostringstream msg;
    msg << "max_turnover must be >= 0 (got " << max_turnover << ")";
    throw std::invalid_argument(msg.str());
  }
  if (std::isfinite(max_turnover)) {
    // Smallest attainable turnover: clip w0 into the box, then move the
    // budget gap |1 - sum(clip)|; no feasible portfolio does better.
    Vector wc(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) wc[i] = clip(w0[i], lo[i], hi[i]);
    const double to_min = abs_diff_sum(wc, w0) + std::fabs(1.0 - kahan_sum(wc));
    if (max_turnover < to_min - 1e-12 * (1.0 + to_min)) {
      std::ostringstream msg;
      msg << "infeasible turnover cap: max_turnover = " << max_turnover
          << " is below the smallest attainable turnover " << to_min;
      throw std::invalid_argument(msg.str());
    }
  }
}

}  // namespace

namespace detail {

KktCertificate turnover_certificate(const Matrix& s, const Vector& c, const Vector& w0,
                                    double kappa, const Vector& lo, const Vector& hi,
                                    double cap, double nu, double tau, const Vector& w) {
  const std::size_t n = w.size();
  const double k = kappa + tau;
  double rhs_scale = 1.0;
  for (std::size_t i = 0; i < n; ++i) {
    rhs_scale = std::max({rhs_scale, std::fabs(w0[i]), std::fabs(lo[i])});
    if (std::isfinite(hi[i])) rhs_scale = std::max(rhs_scale, std::fabs(hi[i]));
  }
  if (std::isfinite(cap)) rhs_scale = std::max(rhs_scale, cap);
  const double turnover = abs_diff_sum(w, w0);
  double feas = std::fabs(kahan_sum(w) - 1.0);
  for (std::size_t i = 0; i < n; ++i) feas = std::max({feas, lo[i] - w[i], w[i] - hi[i]});
  if (std::isfinite(cap)) feas = std::max(feas, turnover - cap);
  const Vector sw = mat_vec(s, w);
  double sw_max = 0.0;
  double c_max = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    sw_max = std::max(sw_max, std::fabs(sw[i]));
    c_max = std::max(c_max, std::fabs(c[i]));
  }
  const double stat_scale = std::fabs(nu) + k + sw_max + c_max;
  const double stat_tol = kQpConvTol * (1.0 + stat_scale);
  double dual = std::max(0.0, -tau);
  if (tau > stat_tol && std::isfinite(cap)) {
    feas = std::max(feas, cap - turnover);  // complementarity
  }
  for (std::size_t i = 0; i < n; ++i) {
    const double d = w[i] - w0[i];
    const bool at_lo = w[i] <= lo[i] + kQpConvTol * (1.0 + std::fabs(lo[i]));
    const bool at_hi =
        std::isfinite(hi[i]) && w[i] >= hi[i] - kQpConvTol * (1.0 + std::fabs(hi[i]));
    const bool at_w0 = std::fabs(d) <= kQpConvTol * (1.0 + std::fabs(w0[i]));
    const double xi = -(sw[i] - c[i] - nu);
    if (!at_lo) {
      const double s_left = (at_w0 || d < 0.0) ? -k : k;
      dual = std::max(dual, s_left - xi);
    }
    if (!at_hi) {
      const double s_right = (at_w0 || d > 0.0) ? k : -k;
      dual = std::max(dual, xi - s_right);
    }
  }
  const bool certified = feas <= kQpConvTol * (1.0 + rhs_scale) && dual <= stat_tol;
  return KktCertificate{certified, std::max(feas, dual)};
}

}  // namespace detail

TurnoverQpResult min_variance_turnover_solve(const Matrix& sigma, const Vector& w0,
                                             double kappa, const Vector& lb, const Vector& ub,
                                             double max_turnover, int max_iter) {
  Vector lo;
  Vector hi;
  turnover_setup(sigma, w0, kappa, lb, ub, max_turnover, max_iter, lo, hi);
  return turnover_qp(sigma, Vector(sigma.rows(), 0.0), w0, kappa, lo, hi, max_turnover,
                     max_iter);
}

Vector min_variance_turnover(const Matrix& sigma, const Vector& w0, double kappa,
                             const Vector& lb, const Vector& ub, double max_turnover,
                             int max_iter) {
  return require_turnover_converged(
      min_variance_turnover_solve(sigma, w0, kappa, lb, ub, max_turnover, max_iter),
      "min_variance_turnover");
}

TurnoverQpResult mean_variance_turnover_solve(const Vector& mu, const Matrix& sigma,
                                              const Vector& w0, double risk_aversion,
                                              double kappa, const Vector& lb,
                                              const Vector& ub, double max_turnover,
                                              int max_iter) {
  validate_mu_sigma(mu, sigma);
  if (!std::isfinite(risk_aversion) || risk_aversion <= 0.0) {
    std::ostringstream msg;
    msg << "risk_aversion must be finite and > 0 (got " << risk_aversion << ")";
    throw std::invalid_argument(msg.str());
  }
  Vector lo;
  Vector hi;
  turnover_setup(sigma, w0, kappa, lb, ub, max_turnover, max_iter, lo, hi);
  Vector c(mu.size(), 0.0);
  for (std::size_t i = 0; i < mu.size(); ++i) c[i] = mu[i] / risk_aversion;
  return turnover_qp(sigma, c, w0, kappa, lo, hi, max_turnover, max_iter);
}

Vector mean_variance_turnover(const Vector& mu, const Matrix& sigma, const Vector& w0,
                              double risk_aversion, double kappa, const Vector& lb,
                              const Vector& ub, double max_turnover, int max_iter) {
  return require_turnover_converged(
      mean_variance_turnover_solve(mu, sigma, w0, risk_aversion, kappa, lb, ub, max_turnover,
                                   max_iter),
      "mean_variance_turnover");
}

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

Vector tangency_weights(const Vector& mu, const Matrix& sigma, double rf,
                        double max_leverage) {
  validate_mu_sigma(mu, sigma);
  if (!std::isfinite(rf)) throw std::invalid_argument("rf must be finite");
  if (std::isnan(max_leverage) || max_leverage < 1.0) {
    std::ostringstream msg;
    msg << "max_leverage must be >= 1 (got " << max_leverage << ")";
    throw std::invalid_argument(msg.str());
  }
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
  Vector w(z);
  for (double& v : w) v /= denom;
  const double lev = gross_leverage(w);
  if (lev > max_leverage) {
    std::ostringstream msg;
    msg << "tangency ill-conditioned: gross leverage " << lev << " exceeds max_leverage "
        << max_leverage << " (A - rf*C = " << denom
        << "; rf is too close below the minimum-variance return A/C)";
    throw TangencyUndefinedError(msg.str());
  }
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
