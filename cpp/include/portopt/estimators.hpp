/// \file estimators.hpp
/// \brief Return-moment estimators: sample moments and Ledoit-Wolf shrinkage.
///
/// Pinned conventions (API_SPEC.md section 2):
///
/// * sample mean uses divisor T; sample covariance uses divisor T-1
///   (ddof = 1);
/// * Ledoit-Wolf (2004) constant-correlation shrinkage: the intensity is
///   computed exactly as in the paper's Appendix B / covCor.m (divisor T
///   throughout, both theta terms per ordered pair) and applied in the
///   ddof = 1 convention — see shrink_covariance;
/// * annualization multiplies means and covariances by 252 (configurable).
///
/// Every T-length reduction uses the compensated kahan_sum, matching the
/// reference's correctly rounded sums to a couple of ulps.
#pragma once

#include "portopt/linalg.hpp"

namespace portopt {

/// Validate a T x N simple-returns matrix: finite entries strictly greater
/// than -1, T >= min_rows, N >= 1.
/// \throws std::invalid_argument on violation.
void validate_returns(const Matrix& returns, std::size_t min_rows = 2);

/// Per-asset sample mean: mu_i = (1/T) * sum_t X(t, i).
Vector sample_mean(const Matrix& returns);

/// Sample covariance with ddof = 1: S = Xc' Xc / (T-1), Xc demeaned columns.
Matrix sample_cov(const Matrix& returns);

/// Output of Ledoit-Wolf constant-correlation shrinkage.
struct ShrinkageResult {
  Matrix sigma;      ///< Shrunk covariance delta * F + (1 - delta) * S.
  double intensity;  ///< Shrinkage intensity delta in [0, 1].
  double rbar;       ///< Average sample correlation of the target.
  Matrix sample;     ///< Sample covariance S (ddof = 1).
  Matrix target;     ///< Constant-correlation target F.
};

/// Ledoit-Wolf (2004) shrinkage towards the constant-correlation target,
/// pinned term by term to the paper's Appendix B (API_SPEC 2.2). With
/// demeaned returns x(t,i), S_T = Xc'Xc / T (divisor T) and s_ii = S_T(i,i):
///
/// * rbar = (2 / (N(N-1))) * sum_{i<j} S_T_ij / sqrt(s_ii s_jj)
/// * pi_ij = (1/T) sum_t (x_ti x_tj - S_T_ij)^2, pi_hat = sum_ij pi_ij
/// * theta_ii,ij = (1/T) sum_t (x_ti^2 - s_ii)(x_ti x_tj - S_T_ij)
/// * rho_hat = sum_i pi_ii + sum over ordered pairs i != j of
///   rbar * sqrt(s_jj / s_ii) * theta_ii,ij
/// * gamma_hat = sum_ij (F_T_ij - S_T_ij)^2, F_T_ii = s_ii,
///   F_T_ij = rbar sqrt(s_ii s_jj)
/// * kappa = (pi_hat - rho_hat) / gamma_hat;
///   delta = max(0, min(1, kappa / T)) (delta = 0 if gamma_hat <= 0)
///
/// The intensity is exactly the paper's; the returned matrices use the
/// project-wide ddof = 1 convention: sample = S = T/(T-1) S_T,
/// target = F = T/(T-1) F_T, sigma = delta F + (1 - delta) S.
///
/// \throws std::invalid_argument if N < 2 or T < 2.
/// \throws std::domain_error if any sample variance is not strictly positive.
ShrinkageResult shrink_covariance(const Matrix& returns);

/// Annualize a per-period mean vector: mu_ann = periods * mu.
/// \throws std::invalid_argument if periods <= 0.
Vector annualize_mean(const Vector& mu, int periods = 252);

/// Annualize a per-period covariance: Sigma_ann = periods * Sigma.
/// \throws std::invalid_argument if periods <= 0.
Matrix annualize_cov(const Matrix& cov, int periods = 252);

}  // namespace portopt
