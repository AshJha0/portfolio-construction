/// \file markowitz.hpp
/// \brief Markowitz mean-variance: closed forms, simplex projection, PGD.
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
/// Constrained problems (long-only, sum-to-one, optional box bounds) are
/// solved by projected gradient descent with monotone backtracking and an
/// exact sort-based projection onto the box-constrained simplex; all
/// steps are pinned in API_SPEC.md sections 3.3-3.6.
#pragma once

#include <functional>

#include "portopt/linalg.hpp"

namespace portopt {

/// Pinned quadratic-penalty coefficient for constrained frontier points.
constexpr double kFrontierPenalty = 1.0e4;
/// Pinned PGD convergence tolerance (max|w+ - w| per iteration).
constexpr double kPgdTol = 1.0e-12;
/// Pinned PGD iteration cap.
constexpr int kPgdMaxIter = 10000;

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
/// \throws std::domain_error if A - rf*C <= 0 — the tangency portfolio is
///   undefined when the risk-free rate is at or above the minimum-variance
///   return (in particular when rf exceeds every expected return).
Vector tangency_weights(const Vector& mu, const Matrix& sigma, double rf);

/// Efficient-frontier weights for a target return via the two-fund theorem.
/// \throws std::domain_error if n = 1 and the target differs from the
///   single asset's mean; if D <= 1e-12*B*C (degenerate frontier, e.g. mu
///   proportional to 1); or if the target lies below the minimum-variance
///   return A/C (inefficient branch, rejected per spec).
Vector frontier_weights(const Vector& mu, const Matrix& sigma, double target);

/// Frontier sweep output: n_points targets, weight rows, variances.
struct FrontierResult {
  Vector targets;    ///< Pinned grid lo + k*(hi-lo)/(n_points-1).
  Matrix weights;    ///< n_points x n weight matrix (one row per target).
  Vector variances;  ///< w' Sigma w per target.
};

/// Unconstrained frontier sweep over the pinned grid from the
/// minimum-variance return A/C to max(mu) inclusive (API_SPEC 3.4).
FrontierResult efficient_frontier(const Vector& mu, const Matrix& sigma,
                                  int n_points = 25);

/// Exact Euclidean projection onto {w : sum w = total, lb <= w <= ub}
/// via the pinned sort-based breakpoint algorithm (API_SPEC 3.5): the KKT
/// solution is w_i = clip(v_i - t, lb_i, ub_i) for the unique shift t
/// making the sum equal total; the algorithm sorts the clipping
/// breakpoints, brackets g(t) = total and finishes with one exact linear
/// interpolation — finite and deterministic (no iteration tolerance).
/// Bound vectors must have length n; defaults (empty vectors) mean lb = 0
/// and ub = +inf (long-only simplex).
/// \throws std::invalid_argument if bounds are infeasible
///   (sum(lb) > total, sum(ub) < total, or any lb_i > ub_i) or inputs are
///   non-finite/empty.
Vector project_simplex_box(const Vector& v, double total = 1.0,
                           const Vector& lb = Vector(), const Vector& ub = Vector());

/// Convenience overload with uniform scalar bounds.
Vector project_simplex_box(const Vector& v, double total, double lb, double ub);

/// PGD output: minimizer, iterations executed, convergence flag.
struct PgdResult {
  Vector w;
  int iterations;
  bool converged;  ///< false when the iteration cap was hit (reported, not thrown).
};

/// Projected gradient descent with monotone backtracking (pinned,
/// API_SPEC 3.6):
/// 1. w = project(w0); step t = 1.
/// 2. Each iteration: t = 2t; g = grad(w); w+ = project(w - t g); while
///    the sufficient-decrease test
///    f(w+) <= f(w) + g.(w+ - w) + ||w+ - w||^2 / (2t) fails, halve t and
///    recompute w+ (at most 100 halvings, then accept).
/// 3. Stop when max_i |w+_i - w_i| < tol (converged) or after max_iter
///    iterations (flag reported, never thrown).
/// Deterministic — no randomness anywhere.
PgdResult pgd_minimize(const std::function<double(const Vector&)>& fun,
                       const std::function<Vector(const Vector&)>& grad,
                       const std::function<Vector(const Vector&)>& project,
                       const Vector& w0, double tol = kPgdTol,
                       int max_iter = kPgdMaxIter);

/// Constrained minimum variance: min 1/2 w'Sw over the box-simplex
/// (defaults: long-only, lb = 0, ub = +inf), solved by the pinned PGD
/// started from projected equal weights. Strictly convex, so the
/// minimizer is unique and ports agree well below the golden tolerance.
Vector min_variance_longonly(const Matrix& sigma, const Vector& lb = Vector(),
                             const Vector& ub = Vector(), double tol = kPgdTol,
                             int max_iter = kPgdMaxIter);

/// Constrained frontier point via the pinned quadratic-penalty PGD
/// objective f(w) = 1/2 w' Sigma w + penalty/2 (mu.w - target)^2 with
/// penalty = 1e4 by default. The penalty formulation itself is the
/// contract: golden values are the minimizer of THIS objective (achieved
/// return deviates from the target by O(1/penalty) by design).
Vector constrained_frontier_weights(const Vector& mu, const Matrix& sigma,
                                    double target, double penalty = kFrontierPenalty,
                                    const Vector& lb = Vector(),
                                    const Vector& ub = Vector(), double tol = kPgdTol,
                                    int max_iter = kPgdMaxIter);

/// Constrained frontier sweep over the same pinned target grid as
/// efficient_frontier (min-variance return A/C to max(mu)).
FrontierResult constrained_frontier(const Vector& mu, const Matrix& sigma,
                                    int n_points = 25,
                                    double penalty = kFrontierPenalty,
                                    const Vector& lb = Vector(),
                                    const Vector& ub = Vector());

}  // namespace portopt
