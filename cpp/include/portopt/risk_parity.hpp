/// \file risk_parity.hpp
/// \brief Risk parity / equal risk contribution via cyclical coordinate descent.
///
/// CCD update derivation (Griveau-Billion, Richard & Roncalli 2013 style,
/// log-barrier formulation; pinned in API_SPEC section 5). Solve
///
///     min_{y > 0}  F(y) = 1/2 y' Sigma y - sum_i b_i ln y_i
///
/// for risk budgets b_i > 0 (sum normalized to 1). Coordinate-wise, with
/// m_i = (Sigma y)_i - Sigma_ii y_i (the cross term, held fixed):
///
///     dF/dy_i = Sigma_ii y_i + m_i - b_i / y_i = 0
///     =>  y_i = ( -m_i + sqrt(m_i^2 + 4 Sigma_ii b_i) ) / (2 Sigma_ii)
///
/// taking the positive root (the discriminant is > 0 because Sigma_ii > 0
/// and b_i > 0, so y_i stays strictly positive). At a stationary point
/// y_i (Sigma y)_i = b_i exactly, so the normalized weights w = y/sum(y)
/// satisfy RC_i(w) = b_i. F is strictly convex on the positive orthant,
/// so the solution is unique and CCD converges monotonically.
///
/// Sweep order pinned: coordinates i = 0..n-1 ascending; after each
/// single-coordinate update the cached product Sigma y is updated before
/// the next coordinate. Convergence: after a full sweep,
/// max_i |y_i - y_i_prev_sweep| < tol (default 1e-10); at most 10000
/// sweeps (non-convergence is reported via a flag, not thrown).
#pragma once

#include "portopt/linalg.hpp"

namespace portopt {

/// Pinned CCD sweep-level convergence tolerance.
constexpr double kCcdTol = 1.0e-10;
/// Pinned CCD sweep cap.
constexpr int kCcdMaxSweeps = 10000;

/// Normalized risk contributions RC_i = w_i (Sigma w)_i / (w' Sigma w);
/// the contributions sum to exactly 1.
/// \throws std::domain_error if the portfolio variance is not strictly positive.
Vector risk_contributions(const Vector& w, const Matrix& sigma);

/// Naive inverse-volatility weights w_i = (1/sqrt(S_ii)) / sum_j (1/sqrt(S_jj)).
/// \throws std::domain_error if any variance is not strictly positive.
Vector inverse_vol_weights(const Matrix& sigma);

/// CCD output: simplex weights, sweeps executed, convergence flag.
struct RiskBudgetResult {
  Vector weights;
  int sweeps;
  bool converged;  ///< false when the sweep cap was hit (reported, not thrown).
};

/// General risk budgeting via pinned CCD (file docstring). Budgets must
/// be positive and finite (empty vector = equal budgets, i.e. ERC); they
/// are normalized internally to sum 1. Sigma is validated via Cholesky —
/// singular/non-PSD input throws. An almost-zero-vol asset yields a
/// weight near 1 after normalization (the correct ERC limit); the update
/// stays well-defined because Sigma_ii > 0 after the Cholesky gate.
RiskBudgetResult risk_budget_solve(const Matrix& sigma, const Vector& budgets = Vector(),
                                   double tol = kCcdTol, int max_sweeps = kCcdMaxSweeps);

/// Risk-budget weights (weights only; see risk_budget_solve).
Vector risk_budget_weights(const Matrix& sigma, const Vector& budgets = Vector(),
                           double tol = kCcdTol, int max_sweeps = kCcdMaxSweeps);

/// Equal-risk-contribution weights (equal budgets b_i = 1/n).
Vector erc_weights(const Matrix& sigma, double tol = kCcdTol,
                   int max_sweeps = kCcdMaxSweeps);

}  // namespace portopt
