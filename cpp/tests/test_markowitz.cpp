/// \file test_markowitz.cpp
/// \brief Markowitz unit tests: frontier properties, projection hand
///        cases, PGD vs closed form, n=1 trivia, and error paths.
#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "portopt/markowitz.hpp"
#include "test_data.hpp"

using portopt::Matrix;
using portopt::Vector;

namespace {

double variance(const Vector& w, const Matrix& s) {
  return portopt::dot(w, portopt::mat_vec(s, w));
}

}  // namespace

TEST(Markowitz, FrontierVarianceIdentityAndConvexity) {
  const Vector& mu = testdata::mu_ann();
  const Matrix& sig = testdata::sigma_ann();
  const portopt::TwoFundScalars sc = portopt::two_fund_scalars(mu, sig);
  const portopt::FrontierResult fr = portopt::efficient_frontier(mu, sig, 25);
  ASSERT_EQ(fr.targets.size(), 25u);
  // Identity var(m) = (C m^2 - 2 A m + B) / D on the whole grid.
  for (std::size_t k = 0; k < fr.targets.size(); ++k) {
    const double m = fr.targets[k];
    const double expected = (sc.c * m * m - 2.0 * sc.a * m + sc.b) / sc.d;
    EXPECT_NEAR(fr.variances[k], expected, 1e-12) << k;
  }
  // Convexity: second differences of variance in target are non-negative.
  for (std::size_t k = 2; k < fr.targets.size(); ++k) {
    const double d2 =
        fr.variances[k] - 2.0 * fr.variances[k - 1] + fr.variances[k - 2];
    EXPECT_GE(d2, -1e-12) << k;
  }
  // Grid endpoints: min-variance return to max(mu).
  EXPECT_NEAR(fr.targets.front(), sc.a / sc.c, 1e-15);
  double mu_max = mu[0];
  for (double v : mu) mu_max = std::max(mu_max, v);
  EXPECT_NEAR(fr.targets.back(), mu_max, 1e-15);
}

TEST(Markowitz, TangencyHasMaxSharpeOnFrontierGrid) {
  const Vector& mu = testdata::mu_ann();
  const Matrix& sig = testdata::sigma_ann();
  const double rf = testdata::config().risk_free_rate;
  const Vector w_tan = portopt::tangency_weights(mu, sig, rf);
  const double sharpe_tan =
      (portopt::dot(mu, w_tan) - rf) / std::sqrt(variance(w_tan, sig));
  const portopt::FrontierResult fr = portopt::efficient_frontier(mu, sig, 25);
  for (std::size_t k = 0; k < fr.targets.size(); ++k) {
    const Vector w = fr.weights.row(k);
    const double sharpe = (portopt::dot(mu, w) - rf) / std::sqrt(variance(w, sig));
    EXPECT_LE(sharpe, sharpe_tan + 1e-12) << k;
  }
}

TEST(Markowitz, MinVarIsFrontierMinimum) {
  const Vector& mu = testdata::mu_ann();
  const Matrix& sig = testdata::sigma_ann();
  const Vector w_mv = portopt::min_variance_weights(sig);
  const portopt::FrontierResult fr = portopt::efficient_frontier(mu, sig, 25);
  const double var_mv = variance(w_mv, sig);
  for (double v : fr.variances) EXPECT_GE(v, var_mv - 1e-14);
  // Weights sum to one.
  double s = 0.0;
  for (double v : w_mv) s += v;
  EXPECT_NEAR(s, 1.0, 1e-12);
}

TEST(Markowitz, SimplexProjectionHandCases) {
  // Already on the simplex: projection is the identity.
  const Vector on{0.2, 0.3, 0.5};
  const Vector w1 = portopt::project_simplex_box(on);
  for (std::size_t i = 0; i < 3; ++i) EXPECT_NEAR(w1[i], on[i], 1e-15);
  // Uniform shift: [2, 2] -> [0.5, 0.5].
  const Vector w2 = portopt::project_simplex_box(Vector{2.0, 2.0});
  EXPECT_NEAR(w2[0], 0.5, 1e-15);
  EXPECT_NEAR(w2[1], 0.5, 1e-15);
  // One dominant coordinate clips the rest to zero.
  const Vector w3 = portopt::project_simplex_box(Vector{5.0, 0.0, 0.0});
  EXPECT_NEAR(w3[0], 1.0, 1e-15);
  EXPECT_NEAR(w3[1], 0.0, 1e-15);
  EXPECT_NEAR(w3[2], 0.0, 1e-15);
  // Box bounds active on both sides; sum preserved.
  const Vector w4 = portopt::project_simplex_box(Vector{0.9, 0.1, 0.0}, 1.0, 0.1, 0.6);
  double s = 0.0;
  for (double v : w4) s += v;
  EXPECT_NEAR(s, 1.0, 1e-12);
  for (double v : w4) {
    EXPECT_GE(v, 0.1 - 1e-15);
    EXPECT_LE(v, 0.6 + 1e-15);
  }
}

TEST(Markowitz, SimplexProjectionInfeasibleBounds) {
  const Vector v{0.5, 0.5};
  EXPECT_THROW(portopt::project_simplex_box(v, 1.0, 0.6, 2.0), std::invalid_argument);
  EXPECT_THROW(portopt::project_simplex_box(v, 1.0, 0.0, 0.4), std::invalid_argument);
  EXPECT_THROW(portopt::project_simplex_box(v, 1.0, 0.5, 0.4), std::invalid_argument);
  EXPECT_THROW(portopt::project_simplex_box(Vector{}, 1.0), std::invalid_argument);
  EXPECT_THROW(
      portopt::project_simplex_box(Vector{std::nan(""), 0.0}, 1.0),
      std::invalid_argument);
}

TEST(Markowitz, PgdMatchesClosedFormWhenConstraintsInactive) {
  // With a wide box the sum-to-one constraint is the only active one, so
  // PGD must land on the closed-form minimum-variance portfolio (1e-6).
  const Matrix& sig = testdata::sigma_ann();
  const Vector w_closed = portopt::min_variance_weights(sig);
  const Vector lb(sig.rows(), -5.0);
  const Vector ub(sig.rows(), 5.0);
  const Vector w_pgd = portopt::min_variance_longonly(sig, lb, ub);
  ASSERT_EQ(w_pgd.size(), w_closed.size());
  for (std::size_t i = 0; i < w_pgd.size(); ++i) {
    EXPECT_NEAR(w_pgd[i], w_closed[i], 1e-6) << i;
  }
}

TEST(Markowitz, PgdDeterminism) {
  const Matrix& sig = testdata::sigma_ann();
  const Vector w1 = portopt::min_variance_longonly(sig);
  const Vector w2 = portopt::min_variance_longonly(sig);
  for (std::size_t i = 0; i < w1.size(); ++i) EXPECT_EQ(w1[i], w2[i]);
}

TEST(Markowitz, SingleAssetTrivia) {
  Matrix s1(1, 1, 0.04);
  EXPECT_EQ(portopt::min_variance_weights(s1), Vector{1.0});
  EXPECT_EQ(portopt::tangency_weights(Vector{0.08}, s1, 0.02), Vector{1.0});
  EXPECT_EQ(portopt::frontier_weights(Vector{0.08}, s1, 0.08), Vector{1.0});
  // rf at/above the single asset's mean: tangency undefined.
  EXPECT_THROW(portopt::tangency_weights(Vector{0.08}, s1, 0.08), std::domain_error);
  // Frontier target must equal the single asset's mean.
  EXPECT_THROW(portopt::frontier_weights(Vector{0.08}, s1, 0.09), std::domain_error);
}

TEST(Markowitz, TangencyUndefinedWhenRfTooHigh) {
  const Vector& mu = testdata::mu_ann();
  const Matrix& sig = testdata::sigma_ann();
  // rf above every expected return.
  double mu_max = mu[0];
  for (double v : mu) mu_max = std::max(mu_max, v);
  EXPECT_THROW(portopt::tangency_weights(mu, sig, mu_max + 0.01), std::domain_error);
}

TEST(Markowitz, FrontierErrorPaths) {
  const Vector& mu = testdata::mu_ann();
  const Matrix& sig = testdata::sigma_ann();
  const portopt::TwoFundScalars sc = portopt::two_fund_scalars(mu, sig);
  // Target below the min-variance return (includes negative targets).
  EXPECT_THROW(portopt::frontier_weights(mu, sig, sc.a / sc.c - 0.01),
               std::domain_error);
  EXPECT_THROW(portopt::frontier_weights(mu, sig, -0.5), std::domain_error);
  // Degenerate frontier: mu proportional to ones.
  const Vector flat(mu.size(), 0.05);
  EXPECT_THROW(portopt::frontier_weights(flat, sig, 0.06), std::domain_error);
  // Non-finite target.
  EXPECT_THROW(
      portopt::frontier_weights(mu, sig, std::numeric_limits<double>::infinity()),
      std::invalid_argument);
  // Mismatched mu length.
  EXPECT_THROW(portopt::two_fund_scalars(Vector{0.1, 0.2}, sig), std::invalid_argument);
}

TEST(Markowitz, ConstrainedFrontierPenaltyValidation) {
  const Vector& mu = testdata::mu_ann();
  const Matrix& sig = testdata::sigma_ann();
  EXPECT_THROW(portopt::constrained_frontier_weights(mu, sig, 0.08, 0.0),
               std::invalid_argument);
  EXPECT_THROW(portopt::efficient_frontier(mu, sig, 1), std::invalid_argument);
}
