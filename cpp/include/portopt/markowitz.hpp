/// \file markowitz.hpp
/// \brief Markowitz mean-variance: closed forms, simplex projection, and
///        the constrained active-set QP.
///
/// Closed-form results use the classical two-fund theorem. With
/// x = Sigma^-1 mu and y = Sigma^-1 1 (both computed by Cholesky solves,
/// pinned in linalg.hpp) define the scalars
///
///     A = 1' Sigma^-1 mu = sum(x)
///     B = mu' Sigma^-1 mu = mu . x
///     C = 1' Sigma^-1 1  = sum(y)
///     D = B * C - A * A
///
/// Every frontier portfolio is w(m) = g + h * m with
/// g = (B y - A x) / D and h = (C x - A y) / D; the minimum-variance
/// portfolio is y / C (return A/C, variance 1/C) and the tangency
/// portfolio is Sigma^-1 (mu - rf 1) / (A - rf C).
///
/// Constrained problems (sum-to-one, optional target return, box bounds)
/// are strictly convex QPs solved by the finite dual active-set method of
/// Goldfarb & Idnani (1983) whose every linear solve is the pinned
/// Cholesky (API_SPEC.md section 3.6). The result is the KKT point of the
/// QP up to rounding, so all ports agree to ~1e-13 and `converged` is a
/// statement about the KKT conditions, not about an iterate step size.
///
/// Cost-aware problems min 1/2 w'Sw - c'w + kappa sum|w - w0| over the
/// same box-simplex, optionally with a turnover cap, run the same dual
/// active-set method on an exactly equivalent strictly convex split
/// problem in (w, b, s) (API_SPEC.md section 3.7).
#pragma once

#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "portopt/linalg.hpp"

namespace portopt {

/// Pinned iteration cap of the dual active-set QP (each iteration adds or
/// drops one constraint; the method is finite for a strictly convex
/// objective, so the cap is only hit on numerically inconsistent inputs
/// and is then reported via `converged = false`).
constexpr int kQpMaxIter = 1000;
/// Pinned feasibility tolerance: a bound counts as violated when
/// n'w - b < -kQpFeasTol * (1 + |b|).
constexpr double kQpFeasTol = 1.0e-13;
/// Pinned tolerance below which a rounding-level violation of a constraint
/// that is linearly dependent on the active set is accepted as satisfied.
constexpr double kQpDegenerateTol = 1.0e-9;
/// Pinned KKT certificate tolerance (relative) for `converged`.
constexpr double kQpConvTol = 1.0e-9;
/// Pinned default cap on the gross leverage sum_i |w_i| of the
/// unconstrained tangency portfolio (API_SPEC 3.3).
constexpr double kTangencyMaxLeverage = 10.0;

/// Two-fund theorem scalars A, B, C, D (see file docstring).
struct TwoFundScalars {
  double a;
  double b;
  double c;
  double d;
};

/// Compute A = 1'S^-1 mu, B = mu'S^-1 mu, C = 1'S^-1 1, D = BC - A^2.
/// \throws std::invalid_argument / std::domain_error if sigma is not
///   symmetric positive definite or dimensions mismatch.
TwoFundScalars two_fund_scalars(const Vector& mu, const Matrix& sigma);

/// Unconstrained (sum-to-one) minimum-variance weights S^-1 1 / C.
/// For n = 1 returns the trivial [1.0].
Vector min_variance_weights(const Matrix& sigma);

/// Tangency (max-Sharpe) weights S^-1 (mu - rf 1) / (A - rf C).
/// \param max_leverage cap on the gross leverage sum_i |w_i| (>= 1;
///   +inf disables it).
/// \throws TangencyUndefinedError if A - rf*C <= 0 (rf at or above the
///   minimum-variance return A/C, in particular above every expected
///   return) or if the gross leverage of the weights exceeds max_leverage
///   (they scale like 1/(A - rf C), so rf just below A/C gives an
///   allocation that is an artefact of the estimate of A/C).
/// \throws std::invalid_argument on malformed inputs (non-finite rf,
///   max_leverage below 1 or NaN, ...).
Vector tangency_weights(const Vector& mu, const Matrix& sigma, double rf,
                        double max_leverage = kTangencyMaxLeverage);

/// Gross exposure sum_i |w_i| (1.0 for any long-only portfolio); a desk
/// should cap it before unconstrained weights reach an OMS.
/// \throws std::invalid_argument on an empty or non-finite vector.
double gross_leverage(const Vector& w);

/// Efficient-frontier weights for a target return via the two-fund theorem.
/// \throws std::domain_error if n = 1 and the target differs from the
///   single asset's mean; if D <= 1e-12*B*C (degenerate frontier, e.g. mu
///   proportional to 1); or if the target lies below the minimum-variance
///   return A/C (inefficient branch, rejected per spec).
Vector frontier_weights(const Vector& mu, const Matrix& sigma, double target);

/// Frontier sweep output: n_points targets, weight rows, variances.
struct FrontierResult {
  Vector targets;    ///< Pinned grid lo + k*(hi-lo)/(n_points-1), last point exactly hi.
  Matrix weights;    ///< n_points x n weight matrix (one row per target).
  Vector variances;  ///< w' Sigma w per target.
};

/// Unconstrained frontier sweep over the pinned grid from the
/// minimum-variance return A/C to max(mu) inclusive (API_SPEC 3.4); the
/// two-fund generators g, h are computed once and reused.
/// \throws std::domain_error if max(mu) <= A/C ("frontier grid empty").
FrontierResult efficient_frontier(const Vector& mu, const Matrix& sigma,
                                  int n_points = 25);

/// Exact Euclidean projection onto {w : sum w = total, lb <= w <= ub}
/// via the pinned sort-based breakpoint algorithm (API_SPEC 3.5): the KKT
/// solution is w_i = clip(v_i - t, lb_i, ub_i) for the unique shift t
/// making the sum equal total; the algorithm sorts the clipping
/// breakpoints, brackets g(t) = total and finishes with one exact linear
/// interpolation — finite and deterministic (no iteration tolerance).
/// Bound vectors must have length n; defaults (empty vectors) mean lb = 0
/// and ub = +inf (long-only simplex). Lower bounds must be finite.
/// \throws std::invalid_argument if bounds are infeasible
///   (sum(lb) > total, sum(ub) < total, or any lb_i > ub_i) or inputs are
///   non-finite/empty.
Vector project_simplex_box(const Vector& v, double total = 1.0,
                           const Vector& lb = Vector(), const Vector& ub = Vector());

/// Convenience overload with uniform scalar bounds.
Vector project_simplex_box(const Vector& v, double total, double lb, double ub);

/// Range [r_min, r_max] of mu'w over {sum w = 1, lb <= w <= ub}; both
/// extremes are vertices found by the pinned greedy fill (assets in
/// ascending / descending mu order, ties by index). Empty bound vectors
/// mean lb = 0, ub = +inf.
std::pair<double, double> attainable_return_range(const Vector& mu,
                                                  const Vector& lb = Vector(),
                                                  const Vector& ub = Vector());

/// Output of the active-set QP solvers.
struct QpResult {
  Vector weights;       ///< Feasible weights (bounds exact, equalities to rounding).
  int iterations;       ///< Dual active-set iterations (constraint adds/drops).
  bool converged;       ///< KKT certificate holds at `weights` (see kQpConvTol).
  double kkt_residual;  ///< Largest KKT violation: equality residuals, bound violations,
                        ///< |S w - N u| and max(0, -min u_bound) (dual feasibility).
};

namespace detail {

/// Result of kkt_certificate.
struct KktCertificate {
  bool certified;   ///< Primal, stationarity and dual feasibility all within tolerance.
  double residual;  ///< Largest of the three violations.
};

/// KKT certificate of min 1/2 w'Sw s.t. n_c'w >= b_c (the first `meq`
/// constraints are equalities) at `w`, with active set `active` and
/// multipliers `u` (u[i] belongs to constraint active[i]). Checks primal
/// feasibility (<= kQpConvTol * (1 + max|b|)), stationarity
/// |S w - N u|_inf <= kQpConvTol * (1 + stat_scale) and dual feasibility
/// u_i >= -kQpConvTol * (1 + stat_scale) for every active inequality, with
/// stat_scale = |S w|_inf + sum_i |u_i| |n_i|_inf. Internal to the QP;
/// exposed for testing.
KktCertificate kkt_certificate(const Matrix& s, const std::vector<Vector>& normals,
                               const Vector& rhs, std::size_t meq,
                               const std::vector<std::size_t>& active, const Vector& u,
                               const Vector& w);

}  // namespace detail

/// Constrained minimum variance min 1/2 w'Sw over {sum w = 1, lb <= w <= ub}
/// (defaults: long-only, lb = 0, ub = +inf), solved by the pinned dual
/// active-set QP. Strictly convex: the minimizer is unique.
QpResult min_variance_longonly_solve(const Matrix& sigma, const Vector& lb = Vector(),
                                     const Vector& ub = Vector(),
                                     int max_iter = kQpMaxIter);

/// Weights of min_variance_longonly_solve.
/// \throws std::domain_error if the solver did not certify convergence.
Vector min_variance_longonly(const Matrix& sigma, const Vector& lb = Vector(),
                             const Vector& ub = Vector(), int max_iter = kQpMaxIter);

/// Constrained frontier point: min 1/2 w'Sw subject to sum w = 1,
/// mu'w = target and lb <= w <= ub. The target is met exactly (to
/// rounding) — there is no penalty term.
/// \throws std::domain_error if the target lies below the constrained
///   minimum-variance return (inefficient branch) or above the maximum
///   attainable return under the bounds.
QpResult constrained_frontier_solve(const Vector& mu, const Matrix& sigma, double target,
                                    const Vector& lb = Vector(), const Vector& ub = Vector(),
                                    int max_iter = kQpMaxIter);

/// Weights of constrained_frontier_solve.
/// \throws std::domain_error if the solver did not certify convergence.
Vector constrained_frontier_weights(const Vector& mu, const Matrix& sigma, double target,
                                    const Vector& lb = Vector(),
                                    const Vector& ub = Vector(), int max_iter = kQpMaxIter);

/// Output of the cost-aware (turnover-penalised) QP solvers.
struct TurnoverQpResult {
  Vector weights;       ///< Feasible weights (bounds exact, budget to rounding).
  int iterations;       ///< Dual active-set iterations on the split problem.
  bool converged;       ///< KKT certificate of the original nonsmooth problem holds.
  double kkt_residual;  ///< Largest violation found by that certificate.
  double turnover;      ///< sum_i |w_i - w0_i| of `weights` (compensated sum).
};

namespace detail {

/// KKT certificate of min 1/2 w'Sw - c'w + kappa sum|w - w0| over
/// {sum w = 1, lo <= w <= hi, sum|w - w0| <= cap} at `w`, with budget
/// multiplier nu and cap multiplier tau (API_SPEC 3.7): with
/// k = kappa + tau and g = S w - c - nu 1, -g_i must lie in the
/// subdifferential of k |w_i - w0_i| plus the box indicator, tau >= 0, and
/// tau > 0 only with a binding cap. Internal; exposed for testing.
KktCertificate turnover_certificate(const Matrix& s, const Vector& c, const Vector& w0,
                                    double kappa, const Vector& lo, const Vector& hi,
                                    double cap, double nu, double tau, const Vector& w);

}  // namespace detail

/// Cost-aware minimum variance (API_SPEC 3.7):
/// min 1/2 w'Sw + kappa sum_i |w_i - w0_i| s.t. sum w = 1, lb <= w <= ub
/// and, when max_turnover is finite, sum_i |w_i - w0_i| <= max_turnover.
/// kappa >= 0 is in the objective's (variance) units; +inf means no cap.
/// Solved by the dual active-set QP on the split problem w = w0 + b - s
/// with Hessian diag(S - eps I, eps I, eps I), eps = 0.5 / trace(S^-1),
/// which is strictly convex and exactly equivalent (no perturbation);
/// then re-polished in w on the identified face and certified.
/// \throws std::invalid_argument on invalid input, infeasible bounds or a
///   cap below the smallest attainable turnover.
TurnoverQpResult min_variance_turnover_solve(
    const Matrix& sigma, const Vector& w0, double kappa, const Vector& lb = Vector(),
    const Vector& ub = Vector(),
    double max_turnover = std::numeric_limits<double>::infinity(),
    int max_iter = kQpMaxIter);

/// Weights of min_variance_turnover_solve.
/// \throws std::domain_error if the KKT certificate fails.
Vector min_variance_turnover(const Matrix& sigma, const Vector& w0, double kappa,
                             const Vector& lb = Vector(), const Vector& ub = Vector(),
                             double max_turnover = std::numeric_limits<double>::infinity(),
                             int max_iter = kQpMaxIter);

/// Cost-aware mean-variance (API_SPEC 3.7):
/// min 1/2 w'Sw - mu'w / risk_aversion + kappa sum_i |w_i - w0_i| over the
/// same feasible set (risk_aversion > 0). Same method and certificate.
TurnoverQpResult mean_variance_turnover_solve(
    const Vector& mu, const Matrix& sigma, const Vector& w0, double risk_aversion,
    double kappa, const Vector& lb = Vector(), const Vector& ub = Vector(),
    double max_turnover = std::numeric_limits<double>::infinity(),
    int max_iter = kQpMaxIter);

/// Weights of mean_variance_turnover_solve.
/// \throws std::domain_error if the KKT certificate fails.
Vector mean_variance_turnover(const Vector& mu, const Matrix& sigma, const Vector& w0,
                              double risk_aversion, double kappa,
                              const Vector& lb = Vector(), const Vector& ub = Vector(),
                              double max_turnover = std::numeric_limits<double>::infinity(),
                              int max_iter = kQpMaxIter);

/// Constrained frontier sweep on the pinned grid from the constrained
/// minimum-variance return to the maximum attainable return under the
/// bounds (n_points equally spaced targets, both ends inclusive). Every
/// point is solved to KKT optimality; a non-converged point throws.
FrontierResult constrained_frontier(const Vector& mu, const Matrix& sigma,
                                    int n_points = 25, const Vector& lb = Vector(),
                                    const Vector& ub = Vector(), int max_iter = kQpMaxIter);

}  // namespace portopt
