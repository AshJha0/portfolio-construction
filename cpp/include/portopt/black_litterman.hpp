/// \file black_litterman.hpp
/// \brief Black-Litterman: reverse optimization and the posterior master formula.
///
/// Master formula (pinned form, API_SPEC section 4). With prior
/// (equilibrium) returns pi, uncertainty scaling tau in (0, 1], pick
/// matrix P (k x n), view returns Q (k), and view covariance Omega (k x k):
///
///     A_k     = P (tau Sigma) P' + Omega                  (k x k, SPD)
///     mu_post = pi + (tau Sigma) P' A_k^-1 (Q - P pi)
///     M       = tau Sigma - (tau Sigma) P' A_k^-1 P (tau Sigma)
///     S_post  = Sigma + M
///
/// This is algebraically identical to the textbook
/// [(tau S)^-1 + P' O^-1 P]^-1 [(tau S)^-1 pi + P' O^-1 Q] form but needs
/// only one k x k Cholesky factorization, which is the pinned
/// implementation. Limits (tested): tau -> 0 at fixed Omega gives
/// mu_post -> pi and M -> 0; Omega -> inf (zero confidence) gives
/// mu_post -> pi and M -> tau Sigma.
///
/// Omega default (Idzorek-style): Omega = diag(P (tau Sigma) P') scaled
/// per view by a confidence c_k > 0: Omega_kk = [P tau S P']_kk / c_k
/// (c_k -> inf: dogmatic view; c_k -> 0: view ignored).
#pragma once

#include <string>
#include <vector>

#include "portopt/linalg.hpp"

namespace portopt {

/// One investor view (schema of data/views.json).
struct View {
  std::string type;        ///< "absolute" or "relative".
  std::string asset;       ///< Asset name (absolute views).
  std::string long_name;   ///< Long leg (relative views).
  std::string short_name;  ///< Short leg (relative views).
  double q = 0.0;          ///< View return.
  double confidence = 1.0; ///< Per-view confidence (> 0, default 1).
};

/// (P, Q, confidences) built from a view list.
struct ViewMatrices {
  Matrix p;            ///< k x n pick matrix.
  Vector q;            ///< View returns, length k.
  Vector confidences;  ///< Per-view confidences, length k.
};

/// Reverse optimization: pi = delta * Sigma * w_mkt with w_mkt
/// renormalized to sum 1. pi are equilibrium *excess* returns: tangency
/// at rf = 0 on (pi, Sigma) returns exactly w_mkt.
/// \throws std::invalid_argument if delta <= 0 or non-finite, any cap
///   weight is negative, or the cap sum is <= 0.
Vector implied_equilibrium_returns(const Matrix& sigma, const Vector& w_mkt,
                                   double delta);

/// Build (P, Q, confidences) from view structs: an absolute view's P row
/// is the unit vector of its asset; a relative view's P row is +1 on the
/// long leg and -1 on the short leg.
/// \throws std::invalid_argument on empty views, unknown type/asset name,
///   or non-positive/non-finite confidence.
ViewMatrices views_to_matrices(const std::vector<View>& views,
                               const std::vector<std::string>& asset_names);

/// Idzorek-style default view covariance:
/// Omega = diag(P (tau Sigma) P') with each diagonal entry divided by the
/// corresponding per-view confidence c_k > 0 (empty vector = all 1).
Matrix default_omega(const Matrix& sigma, const Matrix& p, double tau,
                     const Vector& confidences = Vector());

/// Black-Litterman posterior.
struct BLResult {
  Vector mu_post;     ///< Posterior expected returns.
  Matrix sigma_post;  ///< Posterior return covariance Sigma + M.
  Matrix m;           ///< Posterior covariance of the mean estimate.
  Matrix omega;       ///< The view covariance actually used.
};

/// Posterior via the pinned master formula with the default Omega
/// (optionally confidence-scaled). The k x k system A_k is factored once
/// by Cholesky and solved column-wise.
/// \throws std::invalid_argument on shape/finiteness/tau violations.
/// \throws std::domain_error if A_k is not SPD.
BLResult bl_posterior(const Matrix& sigma, const Vector& pi, const Matrix& p,
                      const Vector& q, double tau,
                      const Vector& confidences = Vector());

/// Posterior with an explicitly supplied view covariance Omega.
BLResult bl_posterior_with_omega(const Matrix& sigma, const Vector& pi, const Matrix& p,
                                 const Vector& q, double tau, const Matrix& omega);

/// Posterior-based optimal weights: unconstrained tangency on
/// (mu_post, sigma_post). Pinned default rf = 0 because pi are excess
/// returns (a non-zero rf would double-count the risk-free rate); with no
/// binding views the tangency on (pi, Sigma) recovers exactly w_mkt.
/// Returns the weights; \p result (if non-null) receives the posterior.
Vector bl_weights(const Matrix& sigma, const Vector& pi, const Matrix& p,
                  const Vector& q, double tau, double rf = 0.0,
                  const Vector& confidences = Vector(), BLResult* result = nullptr);

}  // namespace portopt
