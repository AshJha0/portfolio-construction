/// \file test_markowitz.cpp
/// \brief Markowitz unit tests: frontier properties, projection hand
///        cases, active-set QP certificates and optimality, n=1 trivia,
///        and error paths.
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "portopt/markowitz.hpp"
#include "test_data.hpp"

using portopt::Matrix;
using portopt::Vector;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

double variance(const Vector& w, const Matrix& s) {
  return portopt::dot(w, portopt::mat_vec(s, w));
}

double vsum(const Vector& v) {
  double s = 0.0;
  for (double x : v) s += x;
  return s;
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
    // The sweep reuses one pair of solves and must equal the point function.
    const Vector w = portopt::frontier_weights(mu, sig, m);
    for (std::size_t i = 0; i < w.size(); ++i) EXPECT_NEAR(fr.weights(k, i), w[i], 1e-15);
  }
  // Convexity: second differences of variance in target are non-negative.
  for (std::size_t k = 2; k < fr.targets.size(); ++k) {
    const double d2 =
        fr.variances[k] - 2.0 * fr.variances[k - 1] + fr.variances[k - 2];
    EXPECT_GE(d2, -1e-12) << k;
  }
  // Grid endpoints: min-variance return to max(mu) (last point exactly hi).
  EXPECT_NEAR(fr.targets.front(), sc.a / sc.c, 1e-15);
  double mu_max = mu[0];
  for (double v : mu) mu_max = std::max(mu_max, v);
  EXPECT_EQ(fr.targets.back(), mu_max);
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
  EXPECT_NEAR(vsum(w_mv), 1.0, 1e-12);
}

TEST(Markowitz, FrontierGridWhenGmvReturnExceedsMaxMu) {
  // GMV shorts the low-return asset: A/C = 0.1185 > max(mu) = 0.10 (MAJ-3).
  const Vector mu{0.10, 0.05};
  Matrix sig(2, 2, 0.0);
  sig(0, 0) = 0.01;
  sig(0, 1) = sig(1, 0) = 0.027;
  sig(1, 1) = 0.09;
  const portopt::TwoFundScalars sc = portopt::two_fund_scalars(mu, sig);
  EXPECT_GT(sc.a / sc.c, 0.10);
  EXPECT_THROW(portopt::efficient_frontier(mu, sig, 25), std::domain_error);
  EXPECT_THROW(portopt::constrained_frontier(mu, sig, 25), std::domain_error);
  const Vector w = portopt::frontier_weights(mu, sig, 0.13);
  EXPECT_NEAR(portopt::dot(mu, w), 0.13, 1e-12);
}

TEST(Markowitz, TangencyNearGmvRejected) {
  // rf within 1e-6 (relative) of A/C: catastrophic cancellation (MAJ-1).
  const Vector& mu = testdata::mu_ann();
  const Matrix& sig = testdata::sigma_ann();
  const portopt::TwoFundScalars sc = portopt::two_fund_scalars(mu, sig);
  const double r_gmv = sc.a / sc.c;
  EXPECT_THROW(portopt::tangency_weights(mu, sig, r_gmv - 1e-13),
               portopt::TangencyUndefinedError);
  EXPECT_THROW(portopt::tangency_weights(mu, sig, r_gmv - 1e-9),
               portopt::TangencyUndefinedError);
  const Vector w = portopt::tangency_weights(mu, sig, r_gmv - 1e-3);
  EXPECT_NEAR(vsum(w), 1.0, 1e-8);
  double l1 = 0.0;
  for (double v : w) l1 += std::fabs(v);
  EXPECT_NEAR(portopt::gross_leverage(w), l1, 1e-12);
  EXPECT_NEAR(portopt::gross_leverage(Vector{0.2, 0.3, 0.5}), 1.0, 1e-15);
  EXPECT_NEAR(portopt::gross_leverage(Vector{1.5, -0.5}), 2.0, 1e-15);
  EXPECT_THROW(portopt::gross_leverage(Vector{}), std::invalid_argument);
  EXPECT_THROW(portopt::gross_leverage(Vector{std::nan("")}), std::invalid_argument);
  // The dedicated type derives from std::domain_error.
  EXPECT_THROW(portopt::tangency_weights(mu, sig, 0.5), std::domain_error);
}

TEST(Markowitz, SimplexProjectionHandCases) {
  // Classic Duchi example and box case (golden inputs).
  const Vector w0 = portopt::project_simplex_box(Vector{0.9, 0.5, -0.2});
  EXPECT_NEAR(w0[0], 0.7, 1e-14);
  EXPECT_NEAR(w0[1], 0.3, 1e-14);
  EXPECT_NEAR(w0[2], 0.0, 1e-14);
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
  EXPECT_NEAR(vsum(w4), 1.0, 1e-12);
  for (double v : w4) {
    EXPECT_GE(v, 0.1 - 1e-15);
    EXPECT_LE(v, 0.6 + 1e-15);
  }
}

TEST(Markowitz, SimplexProjectionIsEuclideanOptimal) {
  std::mt19937_64 rng(5);
  std::normal_distribution<double> nd(0.0, 1.0);
  std::uniform_real_distribution<double> ud(0.0, 1.0);
  for (int trial = 0; trial < 20; ++trial) {
    Vector v(6);
    for (double& x : v) x = nd(rng);
    const Vector w = portopt::project_simplex_box(v, 1.0, 0.0, 0.6);
    EXPECT_NEAR(vsum(w), 1.0, 1e-12);
    double dw = 0.0;
    for (std::size_t i = 0; i < 6; ++i) {
      EXPECT_GE(w[i], -1e-14);
      EXPECT_LE(w[i], 0.6 + 1e-14);
      dw += (w[i] - v[i]) * (w[i] - v[i]);
    }
    for (int c = 0; c < 10; ++c) {
      Vector z(6);
      double zs = 0.0;
      for (double& x : z) {
        x = ud(rng) + 0.05;
        zs += x;
      }
      bool feasible = true;
      double dz = 0.0;
      for (std::size_t i = 0; i < 6; ++i) {
        z[i] /= zs;
        if (z[i] > 0.6) feasible = false;
        dz += (z[i] - v[i]) * (z[i] - v[i]);
      }
      if (feasible) {
        EXPECT_LE(dw, dz + 1e-10);
      }
    }
  }
}

TEST(Markowitz, SimplexProjectionInfeasibleBounds) {
  const Vector v{0.5, 0.5};
  EXPECT_THROW(portopt::project_simplex_box(v, 1.0, 0.6, 2.0), std::invalid_argument);
  EXPECT_THROW(portopt::project_simplex_box(v, 1.0, 0.0, 0.4), std::invalid_argument);
  EXPECT_THROW(portopt::project_simplex_box(v, 1.0, 0.5, 0.4), std::invalid_argument);
  EXPECT_THROW(portopt::project_simplex_box(Vector{}, 1.0), std::invalid_argument);
  EXPECT_THROW(portopt::project_simplex_box(Vector{std::nan(""), 0.0}, 1.0),
               std::invalid_argument);
  EXPECT_THROW(portopt::project_simplex_box(v, 1.0, -kInf, kInf), std::invalid_argument);
  EXPECT_THROW(portopt::project_simplex_box(v, 1.0, Vector{0.0}, Vector()),
               std::invalid_argument);
}

TEST(Markowitz, QpMatchesClosedFormWhenConstraintsInactive) {
  // With a wide box the sum-to-one constraint is the only active one, so
  // the QP must land on the closed-form minimum-variance portfolio exactly.
  const Matrix& sig = testdata::sigma_ann();
  const Vector w_closed = portopt::min_variance_weights(sig);
  const portopt::QpResult res = portopt::min_variance_longonly_solve(
      sig, Vector(sig.rows(), -10.0), Vector(sig.rows(), 10.0));
  EXPECT_TRUE(res.converged);
  EXPECT_LE(res.iterations, 2);
  ASSERT_EQ(res.weights.size(), w_closed.size());
  for (std::size_t i = 0; i < res.weights.size(); ++i) {
    EXPECT_NEAR(res.weights[i], w_closed[i], 1e-14) << i;
  }
}

TEST(Markowitz, QpLongOnlyCertifiedAndKkt) {
  const Matrix& sig = testdata::sigma_ann();
  const portopt::QpResult res = portopt::min_variance_longonly_solve(sig);
  EXPECT_TRUE(res.converged);
  EXPECT_LT(res.kkt_residual, 1e-12);
  const Vector& w = res.weights;
  EXPECT_NEAR(vsum(w), 1.0, 1e-12);
  // KKT for {sum w = 1, w >= 0}: on the support the gradient is constant
  // (= the multiplier lam), off the support it is >= lam.
  const Vector g = portopt::mat_vec(sig, w);
  double lam = kInf;
  for (std::size_t i = 0; i < w.size(); ++i) {
    EXPECT_GE(w[i], 0.0);
    if (w[i] > 1e-14) lam = std::min(lam, g[i]);
  }
  for (std::size_t i = 0; i < w.size(); ++i) {
    if (w[i] > 1e-14) {
      EXPECT_NEAR(g[i], lam, 1e-13);
    } else {
      EXPECT_GE(g[i], lam - 1e-13);
    }
  }
  // Cannot beat the unconstrained optimum, must beat equal weights.
  const Vector w_cf = portopt::min_variance_weights(sig);
  EXPECT_GE(variance(w, sig), variance(w_cf, sig) - 1e-14);
  EXPECT_LE(variance(w, sig), variance(Vector(w.size(), 0.125), sig));
}

TEST(Markowitz, QpBoxBoundsRespectedAndOptimal) {
  // lb = 0.02, ub = 0.40: bounds hold exactly and the solution beats every
  // projected random feasible point (PT-14).
  const Matrix& sig = testdata::sigma_ann();
  const std::size_t n = sig.rows();
  const portopt::QpResult res = portopt::min_variance_longonly_solve(
      sig, Vector(n, 0.02), Vector(n, 0.40));
  EXPECT_TRUE(res.converged);
  EXPECT_LT(res.kkt_residual, 1e-12);
  for (double v : res.weights) {
    EXPECT_GE(v, 0.02);
    EXPECT_LE(v, 0.40);
  }
  EXPECT_NEAR(vsum(res.weights), 1.0, 1e-12);
  const double var = variance(res.weights, sig);
  std::mt19937_64 rng(3);
  std::normal_distribution<double> nd(0.0, 1.0);
  for (int trial = 0; trial < 20; ++trial) {
    Vector v(n);
    for (double& x : v) x = nd(rng);
    const Vector z = portopt::project_simplex_box(v, 1.0, 0.02, 0.40);
    EXPECT_LE(var, variance(z, sig) + 1e-15);
  }
}

TEST(Markowitz, ConstrainedFrontierReportsConvergenceAndHitsTarget) {
  // PT-3: every grid point is certified and meets its target exactly.
  const Vector& mu = testdata::mu_ann();
  const Matrix& sig = testdata::sigma_ann();
  const portopt::QpResult res = portopt::constrained_frontier_solve(mu, sig, 0.08);
  EXPECT_TRUE(res.converged);
  EXPECT_LT(res.kkt_residual, 1e-9);
  EXPECT_NEAR(portopt::dot(mu, res.weights), 0.08, 1e-12);
  EXPECT_NEAR(vsum(res.weights), 1.0, 1e-12);
  for (double v : res.weights) EXPECT_GE(v, 0.0);
  const portopt::FrontierResult fc = portopt::constrained_frontier(mu, sig, 25);
  const Vector w_lo = portopt::min_variance_longonly(sig);
  EXPECT_NEAR(fc.targets.front(), portopt::dot(mu, w_lo), 1e-15);
  double mu_max = mu[0];
  for (double v : mu) mu_max = std::max(mu_max, v);
  EXPECT_EQ(fc.targets.back(), mu_max);
  for (std::size_t k = 0; k < 25; ++k) {
    const Vector w = fc.weights.row(k);
    EXPECT_NEAR(portopt::dot(mu, w), fc.targets[k], 1e-12) << k;
    for (double v : w) EXPECT_GE(v, 0.0);
    EXPECT_NEAR(fc.variances[k], variance(w, sig), 1e-15);
    if (k > 0) {
      EXPECT_GE(fc.variances[k], fc.variances[k - 1] - 1e-14) << k;
    }
  }
  // Max-return endpoint is the single-asset vertex; first point is the GMV.
  for (std::size_t i = 0; i < mu.size(); ++i) {
    EXPECT_NEAR(fc.weights(24, i), mu[i] == mu_max ? 1.0 : 0.0, 1e-15);
    EXPECT_NEAR(fc.weights(0, i), w_lo[i], 1e-15);
  }
}

TEST(Markowitz, ConstrainedFrontierRejectsUnattainableTargets) {
  const Vector& mu = testdata::mu_ann();
  const Matrix& sig = testdata::sigma_ann();
  const Vector w_lo = portopt::min_variance_longonly(sig);
  EXPECT_THROW(portopt::constrained_frontier_weights(mu, sig, portopt::dot(mu, w_lo) - 1e-3),
               std::domain_error);
  double mu_max = mu[0];
  for (double v : mu) mu_max = std::max(mu_max, v);
  EXPECT_THROW(portopt::constrained_frontier_weights(mu, sig, mu_max + 1e-3),
               std::domain_error);
  EXPECT_THROW(portopt::constrained_frontier_weights(mu, sig, 0.09, Vector(8, 0.0),
                                                     Vector(8, 0.4)),
               std::domain_error);
  EXPECT_THROW(portopt::constrained_frontier_weights(mu, sig, kInf), std::invalid_argument);
  EXPECT_THROW(portopt::constrained_frontier_weights(mu, sig, 0.06, Vector(), Vector(), 0),
               std::invalid_argument);
  EXPECT_THROW(portopt::constrained_frontier(mu, sig, 1), std::invalid_argument);
  EXPECT_THROW(portopt::efficient_frontier(mu, sig, 1), std::invalid_argument);
}

TEST(Markowitz, AttainableReturnRange) {
  const Vector& mu = testdata::mu_ann();
  const auto r = portopt::attainable_return_range(mu);
  Vector sorted(mu);
  std::sort(sorted.begin(), sorted.end(), std::greater<double>());
  EXPECT_EQ(r.first, sorted.back());
  EXPECT_EQ(r.second, sorted.front());
  const auto rc = portopt::attainable_return_range(mu, Vector(8, 0.0), Vector(8, 0.4));
  EXPECT_NEAR(rc.second, 0.4 * sorted[0] + 0.4 * sorted[1] + 0.2 * sorted[2], 1e-15);
  EXPECT_NEAR(rc.first, 0.4 * sorted[7] + 0.4 * sorted[6] + 0.2 * sorted[5], 1e-15);
  EXPECT_THROW(portopt::attainable_return_range(mu, Vector(8, 0.2), Vector()),
               std::invalid_argument);
}

TEST(Markowitz, QpDegenerateInputsCertified) {
  // Fixed asset (lb == ub), tied top returns with caps, all-equal returns.
  const Vector& mu = testdata::mu_ann();
  const Matrix& sig = testdata::sigma_ann();
  const std::size_t n = sig.rows();
  Vector lb(n, 0.0);
  Vector ub(n, kInf);
  lb[5] = ub[5] = 0.30;
  const portopt::QpResult fixed = portopt::min_variance_longonly_solve(sig, lb, ub);
  EXPECT_TRUE(fixed.converged);
  EXPECT_EQ(fixed.weights[5], 0.30);
  EXPECT_NEAR(vsum(fixed.weights), 1.0, 1e-12);
  Vector mu2(mu);
  mu2[1] = mu2[0];
  double mu_max = mu2[0];
  for (double v : mu2) mu_max = std::max(mu_max, v);
  const portopt::QpResult tie =
      portopt::constrained_frontier_solve(mu2, sig, mu_max, Vector(n, 0.0), Vector(n, 0.7));
  EXPECT_TRUE(tie.converged);
  EXPECT_LT(tie.kkt_residual, 1e-12);
  EXPECT_NEAR(tie.weights[0] + tie.weights[1], 1.0, 1e-12);
  const portopt::QpResult flat = portopt::constrained_frontier_solve(Vector(n, 0.05), sig, 0.05);
  EXPECT_TRUE(flat.converged);
  const Vector w_lo = portopt::min_variance_longonly(sig);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(flat.weights[i], w_lo[i]);
}

TEST(Markowitz, QpDeterminism) {
  const Matrix& sig = testdata::sigma_ann();
  const Vector w1 = portopt::min_variance_longonly(sig);
  const Vector w2 = portopt::min_variance_longonly(sig);
  for (std::size_t i = 0; i < w1.size(); ++i) EXPECT_EQ(w1[i], w2[i]);
  const portopt::FrontierResult a = portopt::constrained_frontier(testdata::mu_ann(), sig, 7);
  const portopt::FrontierResult b = portopt::constrained_frontier(testdata::mu_ann(), sig, 7);
  for (std::size_t k = 0; k < 7; ++k) {
    for (std::size_t i = 0; i < w1.size(); ++i) EXPECT_EQ(a.weights(k, i), b.weights(k, i));
  }
}

TEST(Markowitz, SingleAssetTrivia) {
  Matrix s1(1, 1, 0.04);
  EXPECT_EQ(portopt::min_variance_weights(s1), Vector{1.0});
  EXPECT_EQ(portopt::tangency_weights(Vector{0.08}, s1, 0.02), Vector{1.0});
  EXPECT_EQ(portopt::frontier_weights(Vector{0.08}, s1, 0.08), Vector{1.0});
  EXPECT_EQ(portopt::min_variance_longonly(s1), Vector{1.0});
  // rf at/above the single asset's mean: tangency undefined.
  EXPECT_THROW(portopt::tangency_weights(Vector{0.08}, s1, 0.08),
               portopt::TangencyUndefinedError);
  // Frontier target must equal the single asset's mean.
  EXPECT_THROW(portopt::frontier_weights(Vector{0.08}, s1, 0.09), std::domain_error);
  // MIN-11: the n = 1 paths validate the covariance too.
  EXPECT_THROW(portopt::frontier_weights(Vector{0.08}, Matrix(1, 1, -1.0), 0.08),
               std::invalid_argument);
  EXPECT_THROW(portopt::tangency_weights(Vector{0.08}, Matrix(1, 1, 0.0), 0.02),
               std::invalid_argument);
  EXPECT_THROW(portopt::efficient_frontier(Vector{0.08}, s1, 5), std::invalid_argument);
}

TEST(Markowitz, TangencyUndefinedWhenRfTooHigh) {
  const Vector& mu = testdata::mu_ann();
  const Matrix& sig = testdata::sigma_ann();
  double mu_max = mu[0];
  for (double v : mu) mu_max = std::max(mu_max, v);
  EXPECT_THROW(portopt::tangency_weights(mu, sig, mu_max + 0.01),
               portopt::TangencyUndefinedError);
  EXPECT_THROW(portopt::tangency_weights(mu, sig, std::nan("")), std::invalid_argument);
}

TEST(Markowitz, FrontierErrorPaths) {
  const Vector& mu = testdata::mu_ann();
  const Matrix& sig = testdata::sigma_ann();
  const portopt::TwoFundScalars sc = portopt::two_fund_scalars(mu, sig);
  // Target below the min-variance return (includes negative targets).
  EXPECT_THROW(portopt::frontier_weights(mu, sig, sc.a / sc.c - 0.01), std::domain_error);
  EXPECT_THROW(portopt::frontier_weights(mu, sig, -0.5), std::domain_error);
  // Degenerate frontier: mu proportional to ones.
  const Vector flat(mu.size(), 0.05);
  EXPECT_THROW(portopt::frontier_weights(flat, sig, 0.06), std::domain_error);
  EXPECT_THROW(portopt::efficient_frontier(flat, sig, 5), std::domain_error);
  // Non-finite target.
  EXPECT_THROW(portopt::frontier_weights(mu, sig, kInf), std::invalid_argument);
  // Mismatched mu length.
  EXPECT_THROW(portopt::two_fund_scalars(Vector{0.1, 0.2}, sig), std::invalid_argument);
  // Singular covariance rejected by the Cholesky gate in the QP too.
  Matrix dup(2, 2, 0.04);
  EXPECT_THROW(portopt::min_variance_longonly(dup), std::domain_error);
}

TEST(Markowitz, KktCertificateChecksDualFeasibility) {
  // min 1/2 |w|^2 s.t. sum w = 1, w >= 0 (S = I, n = 2). The true active
  // set {sum} gives w = (1/2, 1/2), u = 1/2 and is certified. Forcing the
  // wrong active set {sum, w_0 >= 0} gives w = (0, 1): primal feasible and
  // stationary (S w = 1*(1,1) - 1*e_0), but the bound multiplier is -1.
  Matrix s(2, 2, 0.0);
  s(0, 0) = 1.0;
  s(1, 1) = 1.0;
  const std::vector<Vector> normals{Vector{1.0, 1.0}, Vector{1.0, 0.0}, Vector{0.0, 1.0}};
  const Vector rhs{1.0, 0.0, 0.0};
  auto ok = portopt::detail::kkt_certificate(s, normals, rhs, 1, {0}, Vector{0.5},
                                             Vector{0.5, 0.5});
  EXPECT_TRUE(ok.certified);
  EXPECT_EQ(ok.residual, 0.0);
  auto bad = portopt::detail::kkt_certificate(s, normals, rhs, 1, {0, 1}, Vector{1.0, -1.0},
                                              Vector{0.0, 1.0});
  EXPECT_FALSE(bad.certified);
  EXPECT_EQ(bad.residual, 1.0);
  // A rounding-level negative bound multiplier is tolerated (scaled tol).
  auto tiny = portopt::detail::kkt_certificate(s, normals, rhs, 1, {0, 1},
                                               Vector{0.5, -1e-12}, Vector{0.5, 0.5});
  EXPECT_TRUE(tiny.certified);
  EXPECT_NEAR(tiny.residual, 1e-12, 1e-15);
  // An equality multiplier may be negative.
  const std::vector<Vector> eq_only{Vector{1.0, 1.0}};
  auto neg_eq = portopt::detail::kkt_certificate(s, eq_only, Vector{-1.0}, 1, {0},
                                                 Vector{-0.5}, Vector{-0.5, -0.5});
  EXPECT_TRUE(neg_eq.certified);
}

TEST(Markowitz, QpWithActiveBoundsStillCertified) {
  // Dual feasibility holds at the true optimum: problems with several
  // active bounds stay certified with the same residual bound as before.
  const Matrix& sig = testdata::sigma_ann();
  const Vector& mu = testdata::mu_ann();
  const std::size_t n = sig.rows();
  const double bounds[3][2] = {{0.0, kInf}, {0.02, 0.40}, {0.0, 0.25}};
  for (const auto& b : bounds) {
    const portopt::QpResult res =
        portopt::min_variance_longonly_solve(sig, Vector(n, b[0]), Vector(n, b[1]));
    EXPECT_TRUE(res.converged) << b[0] << " " << b[1];
    EXPECT_LT(res.kkt_residual, 1e-12);
    bool any_active = false;
    for (double v : res.weights) any_active = any_active || v == b[0] || v == b[1];
    EXPECT_TRUE(any_active);
  }
  const portopt::QpResult fr =
      portopt::constrained_frontier_solve(mu, sig, 0.07, Vector(n, 0.0), Vector(n, 0.5));
  EXPECT_TRUE(fr.converged);
  EXPECT_LT(fr.kkt_residual, 1e-9);
}
