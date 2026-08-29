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
  // Zero-variance column.
  Matrix flat(5, 2, 0.0);
  for (std::size_t r = 0; r < 5; ++r) flat(r, 0) = 0.01 * static_cast<double>(r);
  EXPECT_THROW(portopt::shrink_covariance(flat), std::domain_error);
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
