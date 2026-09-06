/// \file test_estimators.cpp
/// \brief Unit tests for sample moments, Ledoit-Wolf shrinkage, and
///        annualization (properties + error paths).
#include <gtest/gtest.h>

#include <cmath>

#include "portopt/estimators.hpp"
#include "portopt/linalg.hpp"
#include "test_data.hpp"

using portopt::Matrix;
using portopt::Vector;

TEST(Estimators, SampleMeanAndCovHandCase) {
  Matrix x(3, 2, 0.0);
  x(0, 0) = 0.01;  x(0, 1) = 0.02;
  x(1, 0) = -0.01; x(1, 1) = 0.00;
  x(2, 0) = 0.03;  x(2, 1) = 0.01;
  const Vector mu = portopt::sample_mean(x);
  EXPECT_NEAR(mu[0], 0.01, 1e-16);
  EXPECT_NEAR(mu[1], 0.01, 1e-16);
  const Matrix s = portopt::sample_cov(x);
  // ddof = 1 by hand: var_0 = (0 + 4e-4 + 4e-4)/2, cov = (0 + 2e-4 + 0)/2.
  EXPECT_NEAR(s(0, 0), 4e-4, 1e-18);
  EXPECT_NEAR(s(0, 1), 1e-4, 1e-18);
  EXPECT_NEAR(s(1, 0), s(0, 1), 0.0);
  EXPECT_NEAR(s(1, 1), 1e-4, 1e-18);
}

TEST(Estimators, ShrinkageIntensityInUnitIntervalAndPsd) {
  const portopt::ShrinkageResult res =
      portopt::shrink_covariance(testdata::returns_data().returns);
  EXPECT_GE(res.intensity, 0.0);
  EXPECT_LE(res.intensity, 1.0);
  EXPECT_GE(res.rbar, -1.0);
  EXPECT_LE(res.rbar, 1.0);
  // Shrunk matrix stays PSD: the pinned Cholesky must succeed.
  EXPECT_NO_THROW(portopt::cholesky(res.sigma));
  // Diagonal is preserved by the constant-correlation target.
  for (std::size_t i = 0; i < res.sigma.rows(); ++i) {
    EXPECT_NEAR(res.sigma(i, i), res.sample(i, i), 1e-18);
  }
}

TEST(Estimators, ShrinkageConvexCombination) {
  const portopt::ShrinkageResult res =
      portopt::shrink_covariance(testdata::returns_data().returns);
  const double d = res.intensity;
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      EXPECT_NEAR(res.sigma(i, j),
                  d * res.target(i, j) + (1.0 - d) * res.sample(i, j), 1e-18);
    }
  }
}

TEST(Estimators, AnnualizationScales) {
  const Vector mu{0.001, -0.002};
  const Vector mu_a = portopt::annualize_mean(mu);
  EXPECT_DOUBLE_EQ(mu_a[0], 0.252);
  EXPECT_DOUBLE_EQ(mu_a[1], -0.504);
  Matrix c(1, 1, 1e-4);
  EXPECT_DOUBLE_EQ(portopt::annualize_cov(c)(0, 0), 252e-4);
  EXPECT_THROW(portopt::annualize_mean(mu, 0), std::invalid_argument);
  EXPECT_THROW(portopt::annualize_cov(c, -1), std::invalid_argument);
}

TEST(Estimators, ErrorPaths) {
  // Too few rows.
  EXPECT_THROW(portopt::sample_cov(Matrix(1, 2, 0.0)), std::invalid_argument);
  // NaN entry.
  Matrix bad(3, 2, 0.0);
  bad(1, 1) = std::nan("");
  EXPECT_THROW(portopt::sample_mean(bad), std::invalid_argument);
  // Shrinkage needs at least 2 assets and 2 rows.
  EXPECT_THROW(portopt::shrink_covariance(Matrix(10, 1, 0.01)), std::invalid_argument);
  EXPECT_THROW(portopt::shrink_covariance(Matrix(1, 3, 0.01)), std::invalid_argument);
  // Zero-variance column (PT-9).
  Matrix flat(10, 2, 0.004);
  for (std::size_t r = 0; r < 10; ++r) flat(r, 0) = -0.01 + 0.02 * static_cast<double>(r) / 9.0;
  EXPECT_THROW(portopt::shrink_covariance(flat), std::domain_error);
  // A simple return <= -1 is not a valid price move.
  Matrix loss(3, 2, 0.01);
  loss(1, 1) = -1.0;
  EXPECT_THROW(portopt::sample_mean(loss), std::invalid_argument);
  // Non-finite moments cannot be annualized.
  EXPECT_THROW(portopt::annualize_mean(Vector{std::nan("")}), std::invalid_argument);
}

namespace {

/// Direct transcription of Ledoit & Wolf's covCor.m (JPM 2004, App. B):
/// divisor T throughout, both theta terms per ordered pair. Returns
/// (delta, rbar) — dense O(T n^2) loops, test-only.
std::pair<double, double> covcor(const Matrix& x) {
  const std::size_t t = x.rows();
  const std::size_t n = x.cols();
  const double tf = static_cast<double>(t);
  std::vector<Vector> xc(n, Vector(t, 0.0));
  for (std::size_t j = 0; j < n; ++j) {
    double m = 0.0;
    for (std::size_t r = 0; r < t; ++r) m += x(r, j);
    m /= tf;
    for (std::size_t r = 0; r < t; ++r) xc[j][r] = x(r, j) - m;
  }
  Matrix sample(n, n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      double acc = 0.0;
      for (std::size_t r = 0; r < t; ++r) acc += xc[i][r] * xc[j][r];
      sample(i, j) = acc / tf;
    }
  }
  Vector sqrtvar(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) sqrtvar[i] = std::sqrt(sample(i, i));
  double corr = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) corr += sample(i, j) / (sqrtvar[i] * sqrtvar[j]);
  }
  const double rbar = (corr - static_cast<double>(n)) /
                      (static_cast<double>(n) * static_cast<double>(n - 1));
  double phi = 0.0;
  double rho = 0.0;
  double gamma = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      double phi_ij = 0.0;
      double theta_ij = 0.0;
      for (std::size_t r = 0; r < t; ++r) {
        const double d = xc[i][r] * xc[j][r] - sample(i, j);
        phi_ij += d * d;
        theta_ij += (xc[i][r] * xc[i][r] - sample(i, i)) * d;
      }
      phi += phi_ij / tf;
      const double prior = (i == j) ? sample(i, i) : rbar * sqrtvar[i] * sqrtvar[j];
      gamma += (sample(i, j) - prior) * (sample(i, j) - prior);
      if (i == j) {
        rho += phi_ij / tf;
      } else {
        rho += rbar * (sqrtvar[j] / sqrtvar[i]) * theta_ij / tf;
      }
    }
  }
  const double kappa = (phi - rho) / gamma;
  return {std::max(0.0, std::min(1.0, kappa / tf)), rbar};
}

}  // namespace

TEST(Estimators, LwRhoMatchesPaperFormula) {
  // PT-1 (CRIT-1): 12x3 hand panel with unequal vols so that
  // theta_ii,ij != theta_jj,ij and the old halved rho_hat would be wrong.
  const double rows[12][3] = {
      {0.010, 0.020, -0.010}, {-0.020, 0.000, 0.010},  {0.030, -0.010, 0.020},
      {0.005, 0.040, -0.030}, {-0.015, -0.030, 0.000}, {0.025, 0.010, 0.015},
      {-0.010, 0.050, -0.020}, {0.000, -0.020, 0.025}, {0.020, 0.030, 0.005},
      {-0.030, -0.040, -0.010}, {0.015, 0.000, 0.030}, {0.010, 0.060, -0.005}};
  const double scale[3] = {1.0, 3.0, 0.5};
  Matrix x(12, 3, 0.0);
  for (std::size_t r = 0; r < 12; ++r) {
    for (std::size_t c = 0; c < 3; ++c) x(r, c) = rows[r][c] * scale[c];
  }
  const auto paper = covcor(x);
  const portopt::ShrinkageResult res = portopt::shrink_covariance(x);
  EXPECT_NEAR(res.intensity, paper.first, 1e-12);
  EXPECT_NEAR(res.rbar, paper.second, 1e-12);
  // n = 2: target == sample (gamma_hat = 0) -> delta = 0, sigma == sample.
  Matrix x2(12, 2, 0.0);
  for (std::size_t r = 0; r < 12; ++r) {
    x2(r, 0) = rows[r][0];
    x2(r, 1) = rows[r][1];
  }
  const portopt::ShrinkageResult res2 = portopt::shrink_covariance(x2);
  EXPECT_EQ(res2.intensity, 0.0);
  EXPECT_EQ(res2.sigma(0, 1), res2.sample(0, 1));
}

TEST(Estimators, LwBundledDataMatchesPaper) {
  const Matrix& x = testdata::returns_data().returns;
  const auto paper = covcor(x);
  const portopt::ShrinkageResult res = portopt::shrink_covariance(x);
  EXPECT_NEAR(res.intensity, paper.first, 1e-13);
  EXPECT_NEAR(res.intensity, 0.00721176522548, 1e-12);  // paper value, not 0.009430
  EXPECT_NEAR(res.rbar, paper.second, 1e-13);
}

TEST(Estimators, LwIntensitySmallTRepairsSingularSample) {
  // PT-2: T = 5 <= N = 8 — the sample covariance is singular but the shrunk
  // matrix passes the Cholesky gate and 0 < delta <= 1.
  const Matrix& x = testdata::returns_data().returns;
  Matrix head(5, x.cols(), 0.0);
  for (std::size_t r = 0; r < 5; ++r) {
    for (std::size_t c = 0; c < x.cols(); ++c) head(r, c) = x(r, c);
  }
  const portopt::ShrinkageResult res = portopt::shrink_covariance(head);
  EXPECT_GT(res.intensity, 0.0);
  EXPECT_LE(res.intensity, 1.0);
  EXPECT_THROW(portopt::cholesky(res.sample), std::domain_error);
  EXPECT_NO_THROW(portopt::cholesky(res.sigma));
  EXPECT_NEAR(res.intensity, covcor(head).first, 1e-12);
}

TEST(Estimators, Determinism) {
  const Matrix& x = testdata::returns_data().returns;
  const portopt::ShrinkageResult r1 = portopt::shrink_covariance(x);
  const portopt::ShrinkageResult r2 = portopt::shrink_covariance(x);
  EXPECT_EQ(r1.intensity, r2.intensity);  // bit-identical, no RNG anywhere
  for (std::size_t i = 0; i < r1.sigma.rows(); ++i) {
    for (std::size_t j = 0; j < r1.sigma.cols(); ++j) {
      EXPECT_EQ(r1.sigma(i, j), r2.sigma(i, j));
    }
  }
}
