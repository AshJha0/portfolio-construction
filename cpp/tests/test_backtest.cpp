/// \file test_backtest.cpp
/// \brief Backtest unit tests: pinned schedule, accounting identity,
///        metric hand cases, and error paths.
#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "portopt/backtest.hpp"
#include "test_data.hpp"

using portopt::Matrix;
using portopt::Vector;

TEST(Backtest, RebalanceScheduleIsPinned) {
  // T = 1250, W = 500, step 63: 12 dates 500, 563, ..., 1193.
  const std::vector<std::size_t> dates = portopt::rebalance_indices(1250, 500, 63);
  ASSERT_EQ(dates.size(), 12u);
  EXPECT_EQ(dates.front(), 500u);
  EXPECT_EQ(dates.back(), 1193u);
  for (std::size_t i = 1; i < dates.size(); ++i) {
    EXPECT_EQ(dates[i] - dates[i - 1], 63u);
  }
}

TEST(Backtest, AccountingIdentity) {
  // The daily portfolio return is exactly w . r_t for the block's target
  // weights, and the holdout covers rows [window, T).
  const Matrix& x = testdata::returns_data().returns;
  const auto& cfg = testdata::config();
  const portopt::BacktestResult r = portopt::run_backtest(
      x,
      [](const Matrix& h) {
        return Vector(h.cols(), 1.0 / static_cast<double>(h.cols()));
      },
      cfg.risk_free_rate, cfg.estimation_window, cfg.rebalance_step);
  ASSERT_EQ(r.portfolio_returns.size(),
            x.rows() - static_cast<std::size_t>(cfg.estimation_window));
  const double n = static_cast<double>(x.cols());
  for (std::size_t t = 0; t < r.portfolio_returns.size(); t += 97) {
    double expected = 0.0;
    const std::size_t row = t + static_cast<std::size_t>(cfg.estimation_window);
    for (std::size_t c = 0; c < x.cols(); ++c) expected += x(row, c) / n;
    EXPECT_NEAR(r.portfolio_returns[t], expected, 1e-15) << t;
  }
  // Constant weights: turnover is exactly zero at every rebalance.
  for (double to : r.turnovers) EXPECT_EQ(to, 0.0);
  EXPECT_EQ(r.avg_turnover, 0.0);
  ASSERT_EQ(r.rebalance_dates.size(), 12u);
  EXPECT_EQ(r.turnovers.size(), 11u);
}

TEST(Backtest, MetricHandCases) {
  // Constant returns: ann_return = 252 * r; vol = 0 -> Sharpe throws.
  const Vector flat(10, 0.001);
  EXPECT_NEAR(portopt::annualized_return(flat), 0.252, 1e-15);
  EXPECT_NEAR(portopt::annualized_vol(flat), 0.0, 1e-15);
  EXPECT_THROW(portopt::sharpe_ratio(flat), std::domain_error);
  // Two-point series: std with ddof=1 is |r1 - r0| / sqrt(2).
  const Vector two{0.01, -0.01};
  EXPECT_NEAR(portopt::annualized_vol(two),
              std::sqrt(252.0) * 0.02 / std::sqrt(2.0), 1e-15);
  // Drawdown: up 10%, down 20%, up 5% -> max DD = 20%.
  const Vector path{0.10, -0.20, 0.05};
  EXPECT_NEAR(portopt::max_drawdown(path), 0.20, 1e-15);
  // Monotonic growth has zero drawdown.
  EXPECT_NEAR(portopt::max_drawdown(Vector(5, 0.01)), 0.0, 0.0);
}

TEST(Backtest, TurnoverBetweenTargets) {
  // Alternate between two fixed target vectors: every turnover is the L1
  // distance between them.
  Matrix x(130, 2, 0.0);
  for (std::size_t r = 0; r < x.rows(); ++r) {
    x(r, 0) = 0.001;
    x(r, 1) = -0.0005;
  }
  int calls = 0;
  const auto fn = [&calls](const Matrix&) {
    ++calls;
    return (calls % 2 == 1) ? Vector{0.7, 0.3} : Vector{0.4, 0.6};
  };
  const portopt::BacktestResult r = portopt::run_backtest(x, fn, 0.0, 50, 20);
  ASSERT_EQ(r.rebalance_dates.size(), 4u);  // 50, 70, 90, 110
  ASSERT_EQ(r.turnovers.size(), 3u);
  for (double to : r.turnovers) EXPECT_NEAR(to, 0.6, 1e-15);
  EXPECT_NEAR(r.avg_turnover, 0.6, 1e-15);
}

TEST(Backtest, ErrorPaths) {
  const Matrix& x = testdata::returns_data().returns;
  const auto equal_fn = [](const Matrix& h) {
    return Vector(h.cols(), 1.0 / static_cast<double>(h.cols()));
  };
  // Sample must exceed the estimation window.
  EXPECT_THROW(portopt::rebalance_indices(500, 500, 63), std::invalid_argument);
  EXPECT_THROW(portopt::rebalance_indices(1250, 1, 63), std::invalid_argument);
  EXPECT_THROW(portopt::rebalance_indices(1250, 500, 0), std::invalid_argument);
  EXPECT_THROW(portopt::run_backtest(x, equal_fn, 0.0, 2000, 63),
               std::invalid_argument);
  // Weights must sum to ~1.
  const auto bad_sum = [](const Matrix& h) { return Vector(h.cols(), 1.0); };
  EXPECT_THROW(portopt::run_backtest(x, bad_sum, 0.0, 500, 63), std::invalid_argument);
  // Weights must be finite and of the right length.
  const auto bad_len = [](const Matrix&) { return Vector{1.0}; };
  EXPECT_THROW(portopt::run_backtest(x, bad_len, 0.0, 500, 63), std::invalid_argument);
  const auto bad_nan = [](const Matrix& h) {
    Vector w(h.cols(), 1.0 / static_cast<double>(h.cols()));
    w[0] = std::nan("");
    return w;
  };
  EXPECT_THROW(portopt::run_backtest(x, bad_nan, 0.0, 500, 63), std::invalid_argument);
  // Metrics on empty / too-short series.
  EXPECT_THROW(portopt::annualized_return(Vector{}), std::invalid_argument);
  EXPECT_THROW(portopt::annualized_vol(Vector{0.01}), std::invalid_argument);
}

TEST(Backtest, MetricsRejectBadRfPeriodsAndTotalLoss) {
  // PT-15 / MIN-13 / MAJ-5: descriptive errors, never NaN.
  const Vector r{0.01, -0.02, 0.015, 0.0, 0.005};
  EXPECT_THROW(portopt::sharpe_ratio(r, std::nan("")), std::invalid_argument);
  EXPECT_THROW(portopt::sharpe_ratio(r, std::numeric_limits<double>::infinity()),
               std::invalid_argument);
  EXPECT_THROW(portopt::annualized_vol(r, -1), std::invalid_argument);
  EXPECT_THROW(portopt::annualized_vol(r, 0), std::invalid_argument);
  EXPECT_THROW(portopt::annualized_return(r, 0), std::invalid_argument);
  EXPECT_THROW(portopt::max_drawdown(Vector{-1.5}), std::invalid_argument);
  EXPECT_THROW(portopt::max_drawdown(Vector{0.1, -1.0, 0.2}), std::invalid_argument);
  EXPECT_THROW(portopt::annualized_return(Vector{0.1, -1.2}), std::invalid_argument);
  EXPECT_NEAR(portopt::max_drawdown(Vector{-0.999}), 0.999, 1e-15);
}

TEST(Backtest, MinimalSample) {
  // PT-10: T = window + 2 is the smallest valid backtest; T = window + 1
  // raises the explicit schedule error although the schedule itself exists.
  const Matrix& x = testdata::returns_data().returns;
  const auto equal_fn = [](const Matrix& h) {
    return Vector(h.cols(), 1.0 / static_cast<double>(h.cols()));
  };
  Matrix x502(502, x.cols());
  for (std::size_t r = 0; r < 502; ++r) {
    for (std::size_t c = 0; c < x.cols(); ++c) x502(r, c) = x(r, c);
  }
  const portopt::BacktestResult r = portopt::run_backtest(x502, equal_fn, 0.02, 500, 63);
  ASSERT_EQ(r.rebalance_dates.size(), 1u);
  EXPECT_EQ(r.portfolio_returns.size(), 2u);
  EXPECT_TRUE(r.turnovers.empty());
  EXPECT_EQ(r.avg_turnover, 0.0);
  EXPECT_TRUE(std::isfinite(r.sharpe) && std::isfinite(r.ann_vol));
  EXPECT_EQ(portopt::rebalance_indices(501, 500, 63), std::vector<std::size_t>{500});
  Matrix x501(501, x.cols());
  for (std::size_t rr = 0; rr < 501; ++rr) {
    for (std::size_t c = 0; c < x.cols(); ++c) x501(rr, c) = x(rr, c);
  }
  EXPECT_THROW(portopt::run_backtest(x501, equal_fn, 0.02, 500, 63), std::invalid_argument);
  EXPECT_THROW(portopt::run_backtest(x502, equal_fn, std::nan(""), 500, 63),
               std::invalid_argument);
}

TEST(Backtest, RebalanceIndicesHugeStepTerminates) {
  // PT-16: a step larger than the sample yields the single date `window`.
  EXPECT_EQ(portopt::rebalance_indices(1250, 500, std::numeric_limits<int>::max()),
            std::vector<std::size_t>{500});
  EXPECT_EQ(portopt::rebalance_indices(1250, 500, 750), std::vector<std::size_t>{500});
  EXPECT_EQ(portopt::rebalance_indices(1251, 500, 750), (std::vector<std::size_t>{500, 1250}));
  EXPECT_EQ(portopt::rebalance_indices(1250, 500, 1).size(), 750u);
}

TEST(Backtest, HistoryViewOverloadMatchesMatrixOverload) {
  // The non-owning view overload and the copying Matrix overload give
  // bit-identical results, and the view sees exactly rows [0, t_k).
  const Matrix& x = testdata::returns_data().returns;
  std::vector<std::size_t> seen_rows;
  const auto by_view = [&seen_rows, &x](const portopt::HistoryView& h) {
    seen_rows.push_back(h.rows());
    EXPECT_EQ(h.cols(), x.cols());
    EXPECT_EQ(h(h.rows() - 1, 0), x(h.rows() - 1, 0));
    // A tilt that depends on the history, so the weights move.
    Vector w(h.cols(), 0.0);
    double total = 0.0;
    for (std::size_t c = 0; c < h.cols(); ++c) {
      w[c] = 1.0 + 10.0 * std::fabs(h(h.rows() - 1, c));
      total += w[c];
    }
    for (double& v : w) v /= total;
    return w;
  };
  const auto by_copy = [&by_view](const Matrix& h) {
    return by_view(portopt::HistoryView(h, h.rows()));
  };
  const portopt::BacktestResult a = portopt::run_backtest(x, by_view, 0.01, 500, 63);
  const portopt::BacktestResult b = portopt::run_backtest(x, by_copy, 0.01, 500, 63);
  EXPECT_EQ(a.portfolio_returns, b.portfolio_returns);
  EXPECT_EQ(a.turnovers, b.turnovers);
  EXPECT_EQ(a.sharpe, b.sharpe);
  EXPECT_EQ(a.max_drawdown, b.max_drawdown);
  EXPECT_GT(a.avg_turnover, 0.0);
  ASSERT_EQ(seen_rows.size(), 2 * a.rebalance_dates.size());
  for (std::size_t i = 0; i < a.rebalance_dates.size(); ++i) {
    EXPECT_EQ(seen_rows[i], a.rebalance_dates[i]);
  }
  // to_matrix() copies exactly the viewed rows; a view longer than the
  // matrix is rejected.
  const Matrix head = portopt::HistoryView(x, 3).to_matrix();
  ASSERT_EQ(head.rows(), 3u);
  EXPECT_EQ(head(2, x.cols() - 1), x(2, x.cols() - 1));
  EXPECT_THROW((void)portopt::HistoryView(x, x.rows() + 1), std::invalid_argument);
}
