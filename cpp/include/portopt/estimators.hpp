/// \file estimators.hpp
/// \brief Return-moment estimators: sample moments and Ledoit-Wolf shrinkage.
///
/// Pinned conventions (API_SPEC.md section 2):
///
/// * sample mean uses divisor T; sample covariance uses divisor T-1
///   (ddof = 1);
/// * Ledoit-Wolf (2004) constant-correlation shrinkage with the classic
///   closed-form intensity; the dispersion sums (pi/rho/gamma) use divisor
///   T while S keeps divisor T-1 — this deliberate simplification is
///   pinned so that every language port computes the identical number;
/// * annualization multiplies means and covariances by 252 (configurable).
#pragma once

#include "portopt/linalg.hpp"

namespace portopt {

/// Validate a T x N returns matrix (finite entries, T >= min_rows, N >= 1).
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
/// pinned term by term (API_SPEC 2.2). With demeaned returns x(t,i) and S
/// the ddof=1 sample covariance:
///
/// * rbar = (2 / (N(N-1))) * sum_{i<j} S_ij / sqrt(S_ii S_jj)
/// * target F_ii = S_ii, F_ij = rbar * sqrt(S_ii S_jj)
/// * pi_ij = (1/T) sum_t (x_ti x_tj - S_ij)^2, pi_hat = sum_ij pi_ij
/// * theta_ii,ij = (1/T) sum_t (x_ti^2 - S_ii)(x_ti x_tj - S_ij)
/// * rho_hat = sum_i pi_ii + sum over ordered pairs i != j of
///   (rbar/2) * sqrt(S_jj/S_ii) * theta_ii,ij
/// * gamma_hat = sum_ij (F_ij - S_ij)^2
/// * kappa = (pi_hat - rho_hat) / gamma_hat;
///   delta = max(0, min(1, kappa / T)) (delta = 0 if gamma_hat <= 0)
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
