/// \file demo.cpp
/// \brief portopt end-to-end demo.
///
/// Run from the cpp/ directory:  ./build/portopt_demo  [data_dir]
/// (data_dir defaults to ../data).
///
/// Prints: estimator table, unconstrained + long-only frontier sweep,
/// Black-Litterman prior vs posterior weights, ERC vs inverse-vol risk
/// contributions, cost-aware (turnover-penalised) long-only min variance,
/// and a 63-day re-estimation backtest across seven methods.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "portopt/backtest.hpp"
#include "portopt/black_litterman.hpp"
#include "portopt/data_io.hpp"
#include "portopt/estimators.hpp"
#include "portopt/markowitz.hpp"
#include "portopt/risk_parity.hpp"

namespace {

using portopt::Matrix;
using portopt::Vector;

void rule(const std::string& title) {
  std::printf("\n=== %s ", title.c_str());
  const int pad = 66 - static_cast<int>(title.size());
  for (int i = 0; i < pad; ++i) std::printf("=");
  std::printf("\n");
}

/// Format a double the way Python prints a float: shortest %g form, but
/// always carrying a decimal point (1 -> "1.0") so the view lines read
/// identically to the reference demo's.
std::string pyfloat(double v) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%g", v);
  std::string out(buf);
  if (out.find('.') == std::string::npos && out.find('e') == std::string::npos &&
      out.find("inf") == std::string::npos && out.find("nan") == std::string::npos) {
    out += ".0";
  }
  return out;
}

void wrow(const std::string& label, const Vector& w) {
  std::printf("%-14s", label.c_str());
  for (double x : w) std::printf("%9.4f", x);
  std::printf("\n");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::string data_dir = argc > 1 ? argv[1] : "../data";
    const portopt::ReturnsData ret = portopt::load_returns_csv(data_dir + "/asset_returns.csv");
    const portopt::Config cfg = portopt::load_config(data_dir + "/config.json");
    const portopt::ViewsDocument views_doc = portopt::load_views(data_dir + "/views.json");
    const portopt::MarketCaps caps = portopt::load_market_caps(data_dir + "/market_caps.csv");
    const Matrix& x = ret.returns;
    const std::vector<std::string>& names = ret.names;
    const double rf = cfg.risk_free_rate;

    std::printf("portopt demo: %zu-asset universe, %zu daily returns\n", x.cols(), x.rows());
    std::printf("assets:");
    for (std::size_t i = 0; i < names.size(); ++i) {
      std::printf("%s %s", i == 0 ? "" : ",", names[i].c_str());
    }
    std::printf("\n");

    // ----------------------------------------------------------- estimators
    rule("Estimators (annualized, Ledoit-Wolf constant-correlation shrinkage)");
    const Vector mu = portopt::annualize_mean(portopt::sample_mean(x));
    const portopt::ShrinkageResult shrink = portopt::shrink_covariance(x);
    const Matrix sig = portopt::annualize_cov(portopt::sample_cov(x));
    const Matrix sig_sh = portopt::annualize_cov(shrink.sigma);
    std::printf("shrinkage intensity delta = %.6f   rbar = %.4f\n", shrink.intensity,
                shrink.rbar);
    std::printf("%-10s%9s%13s%13s\n", "asset", "mean", "vol(sample)", "vol(shrunk)");
    for (std::size_t i = 0; i < names.size(); ++i) {
      std::printf("%-10s%9.4f%13.4f%13.4f\n", names[i].c_str(), mu[i],
                  std::sqrt(sig(i, i)), std::sqrt(sig_sh(i, i)));
    }

    // ------------------------------------------------------------- frontier
    rule("Efficient frontier sweeps (25 points each, showing every 4th)");
    const portopt::FrontierResult fu =
        portopt::efficient_frontier(mu, sig, cfg.frontier_points);
    const portopt::FrontierResult fc =
        portopt::constrained_frontier(mu, sig, cfg.frontier_points);
    std::printf("unconstrained grid: A/C .. max(mu);  long-only grid: constrained GMV .. max(mu)\n");
    std::printf("%8s%11s%10s%15s\n", "target", "vol uncon", "target", "vol long-only");
    for (std::size_t k = 0; k < fu.targets.size(); k += 4) {
      std::printf("%8.4f%11.4f%10.4f%15.4f\n", fu.targets[k], std::sqrt(fu.variances[k]),
                  fc.targets[k], std::sqrt(fc.variances[k]));
    }
    double gap = 0.0;
    for (std::size_t k = 0; k < fc.targets.size(); ++k) {
      gap = std::max(gap, std::fabs(portopt::dot(mu, fc.weights.row(k)) - fc.targets[k]));
    }
    std::printf("long-only targets met exactly: max |mu'w - target| = %.1e\n", gap);
    const Vector w_tan = portopt::tangency_weights(mu, sig, rf);
    const double tan_sharpe = (portopt::dot(mu, w_tan) - rf) /
                              std::sqrt(portopt::dot(w_tan, portopt::mat_vec(sig, w_tan)));
    std::printf("tangency (rf=%.2f%%): sharpe = %.4f, gross leverage = %.2f\n", 100.0 * rf,
                tan_sharpe, portopt::gross_leverage(w_tan));

    // ------------------------------------------------------ black-litterman
    rule("Black-Litterman: prior (equilibrium) vs posterior weights");
    double cap_sum = 0.0;
    for (double c : caps.caps) cap_sum += c;
    Vector w_mkt(caps.caps);
    for (double& v : w_mkt) v /= cap_sum;
    const Vector pi = portopt::implied_equilibrium_returns(sig, w_mkt, views_doc.delta);
    const portopt::ViewMatrices vm = portopt::views_to_matrices(views_doc.views, names);
    for (const portopt::View& v : views_doc.views) {
      if (v.type == "absolute") {
        std::printf("view: %s returns %.1f%% (confidence %s)\n", v.asset.c_str(),
                    100.0 * v.q, pyfloat(v.confidence).c_str());
      } else {
        std::printf("view: %s beats %s by %.1f%% (confidence %s)\n", v.long_name.c_str(),
                    v.short_name.c_str(), 100.0 * v.q, pyfloat(v.confidence).c_str());
      }
    }
    portopt::BLResult bl;
    const Vector w_bl =
        portopt::bl_weights(sig, pi, vm.p, vm.q, views_doc.tau, 0.0, vm.confidences, &bl);
    std::printf("%14s", "");
    for (const std::string& n : names) std::printf("%9.8s", n.c_str());
    std::printf("\n");
    wrow("pi (equil.)", pi);
    wrow("mu posterior", bl.mu_post);
    wrow("w market", w_mkt);
    wrow("w posterior", w_bl);
    std::printf("posterior gross leverage = %.4f\n", portopt::gross_leverage(w_bl));

    // ---------------------------------------------------------- risk parity
    rule("Risk contributions: ERC vs inverse-vol");
    const Vector w_erc = portopt::erc_weights(sig);
    const Vector w_iv = portopt::inverse_vol_weights(sig);
    const Vector rc_erc = portopt::risk_contributions(w_erc, sig);
    const Vector rc_iv = portopt::risk_contributions(w_iv, sig);
    std::printf("%14s", "");
    for (const std::string& n : names) std::printf("%9.8s", n.c_str());
    std::printf("\n");
    wrow("w ERC", w_erc);
    wrow("RC ERC", rc_erc);
    wrow("w inv-vol", w_iv);
    wrow("RC inv-vol", rc_iv);
    const auto spread = [](const Vector& rc) {
      double lo = rc[0], hi = rc[0];
      for (double v : rc) {
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      return hi - lo;
    };
    std::printf("RC spread: ERC %.2e, inv-vol %.2e\n", spread(rc_erc), spread(rc_iv));

    // ------------------------------------------------------------ cost-aware
    rule("Cost-aware rebalancing: long-only min variance from market weights");
    std::printf(
        "objective 1/2 w'Sw + kappa*sum|w - w0|, w0 = market weights, kappa in variance "
        "units\n");
    std::printf("%10s%10s%9s%11s\n", "kappa", "turnover", "vol", "certified");
    for (double kappa : {0.0, 3e-5, 1e-4, 3e-4, 1e-3}) {
      const portopt::TurnoverQpResult r_tc =
          portopt::min_variance_turnover_solve(sig, w_mkt, kappa);
      const double vol_tc = std::sqrt(portopt::dot(r_tc.weights, portopt::mat_vec(sig, r_tc.weights)));
      const char* ok = (r_tc.converged && r_tc.kkt_residual < 1e-12) ? "yes" : "NO";
      std::printf("%10.6f%10.4f%9.4f%11s\n", kappa, r_tc.turnover, vol_tc, ok);
    }
    const portopt::TurnoverQpResult r_cap =
        portopt::min_variance_turnover_solve(sig, w_mkt, 0.0, Vector(), Vector(), 0.25);
    std::printf("kappa 0 with turnover cap 0.25: turnover %.4f, vol %.4f\n", r_cap.turnover,
                std::sqrt(portopt::dot(r_cap.weights, portopt::mat_vec(sig, r_cap.weights))));
    std::printf("%14s", "");
    for (const std::string& n : names) std::printf("%9.8s", n.c_str());
    std::printf("\n");
    wrow("w market", w_mkt);
    wrow("w kappa 1e-4", portopt::min_variance_turnover(sig, w_mkt, 1e-4));
    wrow("w cap 0.25", r_cap.weights);

    // -------------------------------------------------------------- backtest
    rule("Backtest: re-estimated every 63 days, daily constant mix (holdout = last 750d)");
    int fallback_undefined = 0;
    int fallback_leverage = 0;

    const auto equal_weight_fn = [](const Matrix& h) {
      return Vector(h.cols(), 1.0 / static_cast<double>(h.cols()));
    };
    const auto min_var_lo_fn = [](const Matrix& h) {
      return portopt::min_variance_longonly(portopt::annualize_cov(portopt::sample_cov(h)));
    };
    const auto tangency_fn = [&](const Matrix& h) {
      // Pinned backtest method: tangency, falling back to unconstrained
      // min-variance for a rebalance where tangency is undefined (rf at or
      // above A/C) or above the gross-leverage cap, reported, not crashed
      // (API_SPEC 6.3). Any other error (e.g. a singular covariance)
      // propagates.
      const Matrix s = portopt::annualize_cov(portopt::sample_cov(h));
      const Vector m = portopt::annualize_mean(portopt::sample_mean(h));
      try {
        return portopt::tangency_weights(m, s, rf);
      } catch (const portopt::TangencyUndefinedError&) {
        const portopt::TwoFundScalars sc = portopt::two_fund_scalars(m, s);
        if (sc.a - rf * sc.c <= 0.0) {
          ++fallback_undefined;
        } else {
          ++fallback_leverage;
        }
        return portopt::min_variance_weights(s);
      }
    };
    // Long-only min variance with turnover penalty kappa = 3e-5 against the
    // previous target (equal weight before the first rebalance).
    Vector prev_tc;
    const auto min_var_tc_fn = [&](const Matrix& h) {
      const Vector w0 =
          prev_tc.empty() ? Vector(h.cols(), 1.0 / static_cast<double>(h.cols())) : prev_tc;
      Vector w = portopt::min_variance_turnover(portopt::annualize_cov(portopt::sample_cov(h)),
                                                w0, 3e-5);
      prev_tc = w;
      return w;
    };
    const auto bl_fn = [&](const Matrix& h) {
      const Matrix s = portopt::annualize_cov(portopt::sample_cov(h));
      const Vector pi_h = portopt::implied_equilibrium_returns(s, w_mkt, views_doc.delta);
      return portopt::bl_weights(s, pi_h, vm.p, vm.q, views_doc.tau, 0.0, vm.confidences);
    };
    const auto erc_fn = [](const Matrix& h) {
      return portopt::erc_weights(portopt::sample_cov(h));
    };
    const auto inverse_vol_fn = [](const Matrix& h) {
      return portopt::inverse_vol_weights(portopt::sample_cov(h));
    };

    const std::vector<std::pair<std::string, std::function<Vector(const Matrix&)>>>
        methods = {{"equal_weight", equal_weight_fn}, {"min_var_lo", min_var_lo_fn},
                   {"min_var_tc", min_var_tc_fn},     {"tangency", tangency_fn},
                   {"black_litt", bl_fn},             {"erc", erc_fn},
                   {"inverse_vol", inverse_vol_fn}};

    std::printf("%-14s%9s%9s%9s%9s%11s\n", "method", "ann ret", "ann vol", "sharpe",
                "max DD", "avg turno");
    for (const auto& method : methods) {
      const portopt::BacktestResult r = portopt::run_backtest(
          x, method.second, rf, cfg.estimation_window, cfg.rebalance_step);
      std::printf("%-14s%9.4f%9.4f%9.4f%9.4f%11.4f\n", method.first.c_str(),
                  r.ann_return, r.ann_vol, r.sharpe, r.max_drawdown, r.avg_turnover);
    }
    const int n_fb = fallback_undefined + fallback_leverage;
    if (n_fb > 0) {
      std::printf(
          "note: tangency fell back to min-variance at %d rebalance date(s): %d with rf at "
          "or above the min-var return A/C, %d above the %gx gross-leverage cap.\n",
          n_fb, fallback_undefined, fallback_leverage, portopt::kTangencyMaxLeverage);
    }
    std::printf(
        "min_var_tc = min_var_lo with turnover penalty kappa = 3e-5 against the previous "
        "target.\n");

    std::printf("\ndemo complete.\n");
    return 0;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "demo failed: %s\n", ex.what());
    return 1;
  }
}
