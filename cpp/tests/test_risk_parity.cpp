/// \file test_risk_parity.cpp
/// \brief Risk-parity unit tests: RC identities, ERC equality, budget
///        matching, near-zero-vol limit, and error paths.
#include <gtest/gtest.h>

#include <cmath>

#include "portopt/risk_parity.hpp"
#include "test_data.hpp"

using portopt::Matrix;
using portopt::Vector;

TEST(RiskParity, RiskContributionsSumToOne) {
  // Sum of normalized RCs == 1 exactly (portfolio variance normalized).
  const Matrix& sig = testdata::sigma_ann();
  const Vector w(sig.rows(), 1.0 / static_cast<double>(sig.rows()));
  const Vector rc = portopt::risk_contributions(w, sig);
  double s = 0.0;
  for (double v : rc) s += v;
  EXPECT_NEAR(s, 1.0, 1e-12);
}

TEST(RiskParity, ErcContributionsEqual) {
  const Matrix& sig = testdata::sigma_ann();
  const Vector w = portopt::erc_weights(sig);
  const Vector rc = portopt::risk_contributions(w, sig);
  const double target = 1.0 / static_cast<double>(sig.rows());
  for (double v : rc) EXPECT_NEAR(v, target, 1e-9);
  // Long-only simplex weights.
  double s = 0.0;
  for (double v : w) {
    EXPECT_GT(v, 0.0);
    s += v;
  }
  EXPECT_NEAR(s, 1.0, 1e-12);
}

TEST(RiskParity, BudgetsAreMatched) {
  const Matrix& sig = testdata::sigma_ann();
  Vector b{0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05};
  const portopt::RiskBudgetResult res = portopt::risk_budget_solve(sig, b);
  EXPECT_TRUE(res.converged);
  EXPECT_GE(res.sweeps, 1);
  const Vector rc = portopt::risk_contributions(res.weights, sig);
  for (std::size_t i = 0; i < rc.size(); ++i) EXPECT_NEAR(rc[i], b[i], 1e-9) << i;
  // Budgets are normalized internally: scaling them changes nothing.
  Vector b2(b);
  for (double& v : b2) v *= 7.0;
  const Vector w2 = portopt::risk_budget_weights(sig, b2);
  for (std::size_t i = 0; i < w2.size(); ++i) EXPECT_NEAR(w2[i], res.weights[i], 1e-12);
}

TEST(RiskParity, InverseVolClosedForm) {
  const Matrix a = testdata::spd3();
  const Vector w = portopt::inverse_vol_weights(a);
  const double i0 = 1.0 / std::sqrt(a(0, 0));
  const double i1 = 1.0 / std::sqrt(a(1, 1));
  const double i2 = 1.0 / std::sqrt(a(2, 2));
  const double tot = i0 + i1 + i2;
  EXPECT_NEAR(w[0], i0 / tot, 1e-15);
  EXPECT_NEAR(w[1], i1 / tot, 1e-15);
  EXPECT_NEAR(w[2], i2 / tot, 1e-15);
}

TEST(RiskParity, AlmostZeroVolAssetConverges) {
  // An almost-zero-vol asset gets weight near 1 — the correct ERC limit.
  Matrix a(3, 3, 0.0);
  a(0, 0) = 0.04;
  a(1, 1) = 0.09;
  a(2, 2) = 1e-10;  // near-zero-vol asset (still PD)
  a(0, 1) = 0.006;
  a(1, 0) = 0.006;
  const portopt::RiskBudgetResult res = portopt::risk_budget_solve(a);
  EXPECT_TRUE(res.converged);
  EXPECT_GT(res.weights[2], 0.99);
  const Vector rc = portopt::risk_contributions(res.weights, a);
  for (double v : rc) EXPECT_NEAR(v, 1.0 / 3.0, 1e-6);
}

TEST(RiskParity, Determinism) {
  const Matrix& sig = testdata::sigma_ann();
  const Vector w1 = portopt::erc_weights(sig);
  const Vector w2 = portopt::erc_weights(sig);
  for (std::size_t i = 0; i < w1.size(); ++i) EXPECT_EQ(w1[i], w2[i]);
}

TEST(RiskParity, ErrorPaths) {
  const Matrix& sig = testdata::sigma_ann();
  // Non-PSD / singular covariance rejected via the Cholesky gate.
  Matrix dup(2, 2, 0.0);
  dup(0, 0) = 0.04;
  dup(0, 1) = 0.04;
  dup(1, 0) = 0.04;
  dup(1, 1) = 0.04;
  EXPECT_THROW(portopt::risk_budget_solve(dup), std::domain_error);
  // Bad budgets: wrong length, non-positive, non-finite.
  EXPECT_THROW(portopt::risk_budget_solve(sig, Vector{0.5, 0.5}),
               std::invalid_argument);
  Vector bad(sig.rows(), 0.125);
  bad[3] = 0.0;
  EXPECT_THROW(portopt::risk_budget_solve(sig, bad), std::invalid_argument);
  bad[3] = std::nan("");
  EXPECT_THROW(portopt::risk_budget_solve(sig, bad), std::invalid_argument);
  // Bad tol / sweep cap.
  EXPECT_THROW(portopt::risk_budget_solve(sig, Vector(), 0.0), std::invalid_argument);
  EXPECT_THROW(portopt::risk_budget_solve(sig, Vector(), 1e-10, 0),
               std::invalid_argument);
  // Zero-weight portfolio has zero variance.
  EXPECT_THROW(portopt::risk_contributions(Vector(sig.rows(), 0.0), sig),
               std::domain_error);
  // Inverse-vol needs strictly positive variances.
  Matrix z(2, 2, 0.0);
  z(0, 0) = 0.0;
  z(1, 1) = 0.01;
  EXPECT_THROW(portopt::inverse_vol_weights(z), std::domain_error);
}
