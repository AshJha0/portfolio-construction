/// \file test_golden.cpp
/// \brief Golden-value suite: recompute every case from the bundled data
///        and assert agreement with ../data/golden/golden.json within its
///        tolerance (absolute comparison, per API_SPEC section 8).
#include <gtest/gtest.h>

#include <cmath>

#include "portopt/backtest.hpp"
#include "portopt/black_litterman.hpp"
#include "portopt/estimators.hpp"
#include "portopt/markowitz.hpp"
#include "portopt/risk_parity.hpp"
#include "test_data.hpp"

using portopt::Matrix;
using portopt::Vector;
using testdata::golden_case;
using testdata::golden_vec;

namespace {

void expect_vec_near(const Vector& got, const Vector& want, double tol,
                     const std::string& what) {
  ASSERT_EQ(got.size(), want.size()) << what;
  for (std::size_t i = 0; i < got.size(); ++i) {
    EXPECT_NEAR(got[i], want[i], tol) << what << " index " << i;
  }
}

/// Shared Black-Litterman building blocks from the bundled data.
struct BLParts {
  Vector pi;
  portopt::ViewMatrices vm;
  double tau;
};

BLParts bl_parts() {
  const auto& doc = testdata::views_doc();
  BLParts out;
  out.pi = portopt::implied_equilibrium_returns(testdata::sigma_ann(), testdata::w_mkt(),
                                                doc.delta);
  out.vm = portopt::views_to_matrices(doc.views, testdata::returns_data().names);
  out.tau = doc.tau;
  return out;
}

}  // namespace

TEST(Golden, CaseCount) { EXPECT_GE(testdata::golden().size(), 20u); }

TEST(Golden, SampleMomentsDaily) {
  const auto& c = golden_case("sample_moments_daily");
  const Vector mu = portopt::sample_mean(testdata::returns_data().returns);
  const Matrix cov = portopt::sample_cov(testdata::returns_data().returns);
  EXPECT_NEAR(mu[0], c.expect.at("mean_0"), c.tol);
  EXPECT_NEAR(mu[4], c.expect.at("mean_4"), c.tol);
  EXPECT_NEAR(mu[7], c.expect.at("mean_7"), c.tol);
  EXPECT_NEAR(cov(0, 0), c.expect.at("cov_0_0"), c.tol);
  EXPECT_NEAR(cov(0, 1), c.expect.at("cov_0_1"), c.tol);
  EXPECT_NEAR(cov(4, 4), c.expect.at("cov_4_4"), c.tol);
  EXPECT_NEAR(cov(6, 7), c.expect.at("cov_6_7"), c.tol);
}

TEST(Golden, ShrinkageConstCorrDaily) {
  const auto& c = golden_case("shrinkage_const_corr_daily");
  const portopt::ShrinkageResult res =
      portopt::shrink_covariance(testdata::returns_data().returns);
  EXPECT_NEAR(res.intensity, c.expect.at("intensity"), c.tol);
  EXPECT_NEAR(res.rbar, c.expect.at("rbar"), c.tol);
  EXPECT_NEAR(res.sigma(0, 0), c.expect.at("shrunk_0_0"), c.tol);
  EXPECT_NEAR(res.sigma(0, 1), c.expect.at("shrunk_0_1"), c.tol);
  EXPECT_NEAR(res.sigma(2, 5), c.expect.at("shrunk_2_5"), c.tol);
  EXPECT_NEAR(res.sigma(7, 7), c.expect.at("shrunk_7_7"), c.tol);
}

TEST(Golden, TwoFundScalarsAnnualized) {
  const auto& c = golden_case("two_fund_scalars_annualized");
  const portopt::TwoFundScalars sc =
      portopt::two_fund_scalars(testdata::mu_ann(), testdata::sigma_ann());
  EXPECT_NEAR(sc.a, c.expect.at("a"), c.tol);
  EXPECT_NEAR(sc.b, c.expect.at("b"), c.tol);
  EXPECT_NEAR(sc.c, c.expect.at("c"), c.tol);
  EXPECT_NEAR(sc.d, c.expect.at("d"), c.tol);
}

TEST(Golden, MinVarUnconstrained) {
  const auto& c = golden_case("minvar_unconstrained");
  const Vector w = portopt::min_variance_weights(testdata::sigma_ann());
  expect_vec_near(w, golden_vec(c.expect, "w"), c.tol, "minvar w");
  EXPECT_NEAR(portopt::dot(testdata::mu_ann(), w), c.expect.at("ret"), c.tol);
  const double var = portopt::dot(w, portopt::mat_vec(testdata::sigma_ann(), w));
  EXPECT_NEAR(std::sqrt(var), c.expect.at("vol"), c.tol);
}

TEST(Golden, TangencyUnconstrained) {
  const auto& c = golden_case("tangency_unconstrained");
  const double rf = c.inputs.at("rf");
  const Vector w = portopt::tangency_weights(testdata::mu_ann(), testdata::sigma_ann(), rf);
  expect_vec_near(w, golden_vec(c.expect, "w"), c.tol, "tangency w");
  const double var = portopt::dot(w, portopt::mat_vec(testdata::sigma_ann(), w));
  const double sharpe = (portopt::dot(testdata::mu_ann(), w) - rf) / std::sqrt(var);
  EXPECT_NEAR(sharpe, c.expect.at("sharpe"), c.tol);
}

TEST(Golden, FrontierPointT080) {
  const auto& c = golden_case("frontier_point_t080");
  const Vector w = portopt::frontier_weights(testdata::mu_ann(), testdata::sigma_ann(),
                                             c.inputs.at("target"));
  expect_vec_near(w, golden_vec(c.expect, "w"), c.tol, "frontier w");
  const double var = portopt::dot(w, portopt::mat_vec(testdata::sigma_ann(), w));
  EXPECT_NEAR(var, c.expect.at("variance"), c.tol);
}

TEST(Golden, MinVarLongOnlyPgd) {
  const auto& c = golden_case("minvar_longonly_pgd");
  const Vector w = portopt::min_variance_longonly(testdata::sigma_ann());
  expect_vec_near(w, golden_vec(c.expect, "w"), c.tol, "minvar long-only w");
}

TEST(Golden, FrontierConstrainedT080) {
  const auto& c = golden_case("frontier_constrained_t080");
  const Vector w = portopt::constrained_frontier_weights(
      testdata::mu_ann(), testdata::sigma_ann(), c.inputs.at("target"),
      c.inputs.at("penalty"));
  expect_vec_near(w, golden_vec(c.expect, "w"), c.tol, "constrained frontier w");
}

TEST(Golden, CholeskyAnnualizedCov) {
  const auto& c = golden_case("cholesky_annualized_cov");
  const Matrix low = portopt::cholesky(testdata::sigma_ann());
  EXPECT_NEAR(low(0, 0), c.expect.at("l_0_0"), c.tol);
  EXPECT_NEAR(low(7, 0), c.expect.at("l_7_0"), c.tol);
  EXPECT_NEAR(low(7, 7), c.expect.at("l_7_7"), c.tol);
}

TEST(Golden, SimplexProjectionHand) {
  const auto& c = golden_case("simplex_projection_hand");
  const Vector v = golden_vec(c.inputs, "v");
  const Vector w = portopt::project_simplex_box(v, c.inputs.at("total"));
  expect_vec_near(w, golden_vec(c.expect, "w"), c.tol, "hand projection");
}

TEST(Golden, SimplexProjectionBox) {
  const auto& c = golden_case("simplex_projection_box");
  const Vector v = golden_vec(c.inputs, "v");
  const Vector w = portopt::project_simplex_box(v, c.inputs.at("total"),
                                                c.inputs.at("lb"), c.inputs.at("ub"));
  expect_vec_near(w, golden_vec(c.expect, "w"), c.tol, "box projection");
}

TEST(Golden, ImpliedPi) {
  const auto& c = golden_case("implied_pi");
  const Vector pi = portopt::implied_equilibrium_returns(
      testdata::sigma_ann(), testdata::w_mkt(), c.inputs.at("delta"));
  expect_vec_near(pi, golden_vec(c.expect, "pi"), c.tol, "implied pi");
}

TEST(Golden, BlOmegaDefault) {
  const auto& c = golden_case("bl_omega_default");
  const BLParts parts = bl_parts();
  ASSERT_EQ(parts.vm.confidences.size(), 2u);
  EXPECT_DOUBLE_EQ(parts.vm.confidences[0], c.inputs.at("conf_0"));
  EXPECT_DOUBLE_EQ(parts.vm.confidences[1], c.inputs.at("conf_1"));
  const Matrix om = portopt::default_omega(testdata::sigma_ann(), parts.vm.p,
                                           c.inputs.at("tau"), parts.vm.confidences);
  EXPECT_NEAR(om(0, 0), c.expect.at("omega_0_0"), c.tol);
  EXPECT_NEAR(om(1, 1), c.expect.at("omega_1_1"), c.tol);
}

TEST(Golden, BlPosteriorReturnsAndWeights) {
  const BLParts parts = bl_parts();
  const portopt::BLResult res =
      portopt::bl_posterior(testdata::sigma_ann(), parts.pi, parts.vm.p, parts.vm.q,
                            parts.tau, parts.vm.confidences);
  const auto& cr = golden_case("bl_posterior_returns");
  expect_vec_near(res.mu_post, golden_vec(cr.expect, "mu"), cr.tol, "bl mu_post");

  const auto& cw = golden_case("bl_posterior_weights");
  const Vector w =
      portopt::tangency_weights(res.mu_post, res.sigma_post, cw.inputs.at("rf"));
  expect_vec_near(w, golden_vec(cw.expect, "w"), cw.tol, "bl weights");
}

TEST(Golden, InverseVol) {
  const auto& c = golden_case("inverse_vol");
  const Vector w = portopt::inverse_vol_weights(testdata::sigma_ann());
  expect_vec_near(w, golden_vec(c.expect, "w"), c.tol, "inverse vol");
}

TEST(Golden, ErcWeights) {
  const auto& c = golden_case("erc_weights");
  const Vector w = portopt::erc_weights(testdata::sigma_ann());
  expect_vec_near(w, golden_vec(c.expect, "w"), c.tol, "erc w");
  const Vector rc = portopt::risk_contributions(w, testdata::sigma_ann());
  double lo = rc[0], hi = rc[0];
  for (double v : rc) {
    lo = std::min(lo, v);
    hi = std::max(hi, v);
  }
  EXPECT_NEAR(hi - lo, c.expect.at("rc_spread"), c.tol);
}

TEST(Golden, RiskBudgetPinned) {
  const auto& c = golden_case("risk_budget_pinned");
  const Vector b = golden_vec(c.inputs, "b");
  const Vector w = portopt::risk_budget_weights(testdata::sigma_ann(), b);
  expect_vec_near(w, golden_vec(c.expect, "w"), c.tol, "risk budget w");
}

TEST(Golden, BacktestEqualWeight) {
  const auto& c = golden_case("backtest_equal_weight");
  const auto& cfg = testdata::config();
  const portopt::BacktestResult r = portopt::run_backtest(
      testdata::returns_data().returns,
      [](const Matrix& h) {
        return Vector(h.cols(), 1.0 / static_cast<double>(h.cols()));
      },
      c.inputs.at("rf"), cfg.estimation_window, cfg.rebalance_step);
  EXPECT_NEAR(r.ann_return, c.expect.at("ann_return"), c.tol);
  EXPECT_NEAR(r.ann_vol, c.expect.at("ann_vol"), c.tol);
  EXPECT_NEAR(r.sharpe, c.expect.at("sharpe"), c.tol);
  EXPECT_NEAR(r.max_drawdown, c.expect.at("max_drawdown"), c.tol);
}

TEST(Golden, BacktestErc) {
  const auto& c = golden_case("backtest_erc");
  const auto& cfg = testdata::config();
  const portopt::BacktestResult r = portopt::run_backtest(
      testdata::returns_data().returns,
      [](const Matrix& h) { return portopt::erc_weights(portopt::sample_cov(h)); },
      c.inputs.at("rf"), cfg.estimation_window, cfg.rebalance_step);
  EXPECT_NEAR(r.sharpe, c.expect.at("sharpe"), c.tol);
  EXPECT_NEAR(r.max_drawdown, c.expect.at("max_drawdown"), c.tol);
}

TEST(Golden, TurnoverMinVarLongOnly) {
  const auto& c = golden_case("turnover_minvar_longonly");
  const auto& cfg = testdata::config();
  const portopt::BacktestResult r = portopt::run_backtest(
      testdata::returns_data().returns,
      [](const Matrix& h) {
        return portopt::min_variance_longonly(
            portopt::annualize_cov(portopt::sample_cov(h)));
      },
      cfg.risk_free_rate, cfg.estimation_window, cfg.rebalance_step);
  EXPECT_NEAR(r.avg_turnover, c.expect.at("avg_turnover"), c.tol);
  EXPECT_NEAR(static_cast<double>(r.rebalance_dates.size()),
              c.expect.at("n_rebalances"), c.tol);
}
