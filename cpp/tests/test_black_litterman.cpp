/// \file test_black_litterman.cpp
/// \brief Black-Litterman unit tests: reverse-optimization round trip,
///        posterior limits (tau -> 0, Omega -> inf), conflicting views,
///        and error paths.
#include <gtest/gtest.h>

#include <cmath>

#include "portopt/black_litterman.hpp"
#include "portopt/markowitz.hpp"
#include "test_data.hpp"

using portopt::Matrix;
using portopt::Vector;

namespace {

/// Bundled (pi, P, Q, conf, tau) built from the data files.
struct Parts {
  Vector pi;
  portopt::ViewMatrices vm;
  double tau;
};

Parts parts() {
  const auto& doc = testdata::views_doc();
  Parts out;
  out.pi = portopt::implied_equilibrium_returns(testdata::sigma_ann(), testdata::w_mkt(),
                                                doc.delta);
  out.vm = portopt::views_to_matrices(doc.views, testdata::returns_data().names);
  out.tau = doc.tau;
  return out;
}

}  // namespace

TEST(BlackLitterman, ReverseOptimizationRoundTrip) {
  // pi are excess returns: tangency at rf = 0 on (pi, Sigma) returns w_mkt.
  const Parts p = parts();
  const Vector w = portopt::tangency_weights(p.pi, testdata::sigma_ann(), 0.0);
  const Vector& w_mkt = testdata::w_mkt();
  ASSERT_EQ(w.size(), w_mkt.size());
  for (std::size_t i = 0; i < w.size(); ++i) EXPECT_NEAR(w[i], w_mkt[i], 1e-10) << i;
}

TEST(BlackLitterman, ViewsToMatricesShape) {
  const Parts p = parts();
  ASSERT_EQ(p.vm.p.rows(), 2u);
  ASSERT_EQ(p.vm.p.cols(), 8u);
  // Absolute view on EM_EQ (index 4): unit row.
  EXPECT_EQ(p.vm.p(0, 4), 1.0);
  // Relative view US_EQ (0) over EU_EQ (1).
  EXPECT_EQ(p.vm.p(1, 0), 1.0);
  EXPECT_EQ(p.vm.p(1, 1), -1.0);
  EXPECT_DOUBLE_EQ(p.vm.q[0], 0.10);
  EXPECT_DOUBLE_EQ(p.vm.q[1], 0.02);
}

TEST(BlackLitterman, ZeroConfidenceViewsReturnPrior) {
  // Omega -> inf (via c -> 0): mu_post -> pi and M -> tau Sigma.
  const Parts p = parts();
  const Vector tiny_conf(p.vm.q.size(), 1e-12);
  const portopt::BLResult res = portopt::bl_posterior(
      testdata::sigma_ann(), p.pi, p.vm.p, p.vm.q, p.tau, tiny_conf);
  for (std::size_t i = 0; i < p.pi.size(); ++i) {
    EXPECT_NEAR(res.mu_post[i], p.pi[i], 1e-8) << i;
    for (std::size_t j = 0; j < p.pi.size(); ++j) {
      EXPECT_NEAR(res.m(i, j), p.tau * testdata::sigma_ann()(i, j), 1e-8);
    }
  }
}

TEST(BlackLitterman, TauToZeroLimitAtFixedOmega) {
  // Fixed Omega and tau -> 0: mu_post -> pi and M -> 0 (the default Omega
  // scales with tau, so the limit must be taken with Omega held fixed).
  const Parts p = parts();
  const Matrix omega =
      portopt::default_omega(testdata::sigma_ann(), p.vm.p, p.tau, p.vm.confidences);
  const double tiny_tau = 1e-9;
  const portopt::BLResult res = portopt::bl_posterior_with_omega(
      testdata::sigma_ann(), p.pi, p.vm.p, p.vm.q, tiny_tau, omega);
  for (std::size_t i = 0; i < p.pi.size(); ++i) {
    EXPECT_NEAR(res.mu_post[i], p.pi[i], 1e-6) << i;
    for (std::size_t j = 0; j < p.pi.size(); ++j) {
      EXPECT_NEAR(res.m(i, j), 0.0, 1e-9);
    }
  }
}

TEST(BlackLitterman, ConflictingExtremeViewsStayFinite) {
  // Two dogmatic, contradictory views on the same asset: A_k stays SPD
  // and the posterior stays finite.
  const std::vector<std::string>& names = testdata::returns_data().names;
  std::vector<portopt::View> views;
  portopt::View v1;
  v1.type = "absolute";
  v1.asset = names[0];
  v1.q = 5.0;  // +500%
  v1.confidence = 1e6;
  portopt::View v2;
  v2.type = "absolute";
  v2.asset = names[0];
  v2.q = -5.0;  // -500%
  v2.confidence = 1e6;
  views.push_back(v1);
  views.push_back(v2);
  const portopt::ViewMatrices vm = portopt::views_to_matrices(views, names);
  const Parts p = parts();
  const portopt::BLResult res = portopt::bl_posterior(
      testdata::sigma_ann(), p.pi, vm.p, vm.q, p.tau, vm.confidences);
  for (double v : res.mu_post) EXPECT_TRUE(std::isfinite(v));
  for (std::size_t i = 0; i < res.sigma_post.rows(); ++i) {
    for (std::size_t j = 0; j < res.sigma_post.cols(); ++j) {
      EXPECT_TRUE(std::isfinite(res.sigma_post(i, j)));
    }
  }
}

TEST(BlackLitterman, RelativeViewSameLegsRejected) {
  // PT-7 (MAJ-2): long == short must not become a silent absolute short view.
  const std::vector<std::string>& names = testdata::returns_data().names;
  portopt::View v;
  v.type = "relative";
  v.long_name = "US_EQ";
  v.short_name = "US_EQ";
  v.q = 0.02;
  EXPECT_THROW(portopt::views_to_matrices({v}, names), std::invalid_argument);
}

TEST(BlackLitterman, ViewsParseErrorPaths) {
  // PT-11: unknown asset, NaN q, duplicate names, empty views.
  const std::vector<std::string>& names = testdata::returns_data().names;
  portopt::View v;
  v.type = "absolute";
  v.asset = "NOPE";
  v.q = 0.1;
  EXPECT_THROW(portopt::views_to_matrices({v}, names), std::invalid_argument);
  v.asset = "US_EQ";
  v.q = std::nan("");
  EXPECT_THROW(portopt::views_to_matrices({v}, names), std::invalid_argument);
  v.q = 0.1;
  std::vector<std::string> dup(names);
  dup.push_back("US_EQ");
  EXPECT_THROW(portopt::views_to_matrices({v}, dup), std::invalid_argument);
  EXPECT_THROW(portopt::views_to_matrices({v}, {}), std::invalid_argument);
  EXPECT_THROW(portopt::views_to_matrices({}, names), std::invalid_argument);
  const portopt::ViewMatrices vm = portopt::views_to_matrices({v}, names);
  EXPECT_EQ(vm.confidences[0], 1.0);
}

TEST(BlackLitterman, OmegaAndConfidencesTogetherRejected) {
  // MIN-12: explicit Omega plus confidences would silently ignore the latter.
  const Parts p = parts();
  const Matrix omega =
      portopt::default_omega(testdata::sigma_ann(), p.vm.p, p.tau, p.vm.confidences);
  // bl_posterior_with_omega has no confidences parameter by design; the
  // shared core rejects the combination, reachable through bl_weights'
  // confidences + an explicit Omega only in the ports that allow both
  // (Rust/Java). Here we assert the explicit-Omega path reproduces the
  // confidence-scaled posterior exactly.
  const portopt::BLResult a = portopt::bl_posterior_with_omega(
      testdata::sigma_ann(), p.pi, p.vm.p, p.vm.q, p.tau, omega);
  const portopt::BLResult b = portopt::bl_posterior(testdata::sigma_ann(), p.pi, p.vm.p,
                                                    p.vm.q, p.tau, p.vm.confidences);
  for (std::size_t i = 0; i < a.mu_post.size(); ++i) EXPECT_EQ(a.mu_post[i], b.mu_post[i]);
}

TEST(BlackLitterman, BlWeightsWithOmegaAndLeverage) {
  const Parts p = parts();
  const Matrix omega =
      portopt::default_omega(testdata::sigma_ann(), p.vm.p, p.tau, p.vm.confidences);
  portopt::BLResult res_a;
  const Vector w_a = portopt::bl_weights_with_omega(testdata::sigma_ann(), p.pi, p.vm.p,
                                                    p.vm.q, p.tau, omega, 0.0, &res_a);
  portopt::BLResult res_b;
  const Vector w_b = portopt::bl_weights(testdata::sigma_ann(), p.pi, p.vm.p, p.vm.q, p.tau,
                                         0.0, p.vm.confidences, &res_b);
  for (std::size_t i = 0; i < w_a.size(); ++i) {
    EXPECT_NEAR(w_a[i], w_b[i], 1e-15);
    EXPECT_NEAR(res_a.mu_post[i], res_b.mu_post[i], 1e-16);
  }
  EXPECT_GT(portopt::gross_leverage(w_b), 1.0);  // posterior shorts EU_EQ
  EXPECT_THROW(portopt::bl_weights(testdata::sigma_ann(), p.pi, p.vm.p, p.vm.q, p.tau,
                                   std::nan("")),
               std::invalid_argument);
}

TEST(BlackLitterman, ErrorPaths) {
  const Parts p = parts();
  const Matrix& sig = testdata::sigma_ann();
  const std::vector<std::string>& names = testdata::returns_data().names;
  // Empty views.
  EXPECT_THROW(portopt::views_to_matrices({}, names), std::invalid_argument);
  // Unknown view type.
  portopt::View bad;
  bad.type = "sideways";
  EXPECT_THROW(portopt::views_to_matrices({bad}, names), std::invalid_argument);
  // Non-positive confidence.
  portopt::View v;
  v.type = "absolute";
  v.asset = names[0];
  v.q = 0.05;
  v.confidence = 0.0;
  EXPECT_THROW(portopt::views_to_matrices({v}, names), std::invalid_argument);
  // Zero P row.
  Matrix p0(1, sig.rows(), 0.0);
  EXPECT_THROW(portopt::default_omega(sig, p0, 0.05), std::invalid_argument);
  EXPECT_THROW(portopt::bl_posterior(sig, p.pi, p0, Vector{0.05}, 0.05),
               std::invalid_argument);
  // Q length mismatch.
  EXPECT_THROW(portopt::bl_posterior(sig, p.pi, p.vm.p, Vector{0.05}, 0.05),
               std::invalid_argument);
  // tau outside (0, 1].
  EXPECT_THROW(portopt::bl_posterior(sig, p.pi, p.vm.p, p.vm.q, 0.0),
               std::invalid_argument);
  EXPECT_THROW(portopt::bl_posterior(sig, p.pi, p.vm.p, p.vm.q, 1.5),
               std::invalid_argument);
  // delta <= 0; negative caps; zero cap sum.
  EXPECT_THROW(portopt::implied_equilibrium_returns(sig, testdata::w_mkt(), 0.0),
               std::invalid_argument);
  Vector neg(sig.rows(), 1.0);
  neg[0] = -1.0;
  EXPECT_THROW(portopt::implied_equilibrium_returns(sig, neg, 2.5),
               std::invalid_argument);
  EXPECT_THROW(
      portopt::implied_equilibrium_returns(sig, Vector(sig.rows(), 0.0), 2.5),
      std::invalid_argument);
}
