/// \file backtest.hpp
/// \brief Backtest-lite: quarterly expanding-window rebalance loop and metrics.
///
/// Pinned schedule (API_SPEC section 6): with T rows of daily returns,
/// estimation window W (default 500) and rebalance step of 63 trading
/// days, rebalance dates are the row indices t_k = W + 63 k for
/// k = 0, 1, ... while t_k < T. At each date the weight function sees the
/// expanding history rows [0, t_k) and the resulting weights apply to
/// rows [t_k, min(t_{k+1}, T)). Within a holding block the portfolio is
/// rebalanced back to target daily, so the daily portfolio return is
/// exactly w' r_t.
///
/// Turnover at rebalance k >= 1 (pinned): sum_i |w_k[i] - w_{k-1}[i]|
/// (two-sided L1 between consecutive target vectors; nothing charged for
/// the initial position).
///
/// Metric formulas (pinned):
/// * ann_return = 252 * mean(r) (arithmetic)
/// * ann_vol    = sqrt(252) * std(r, ddof=1)
/// * sharpe     = (ann_return - rf) / ann_vol (error if ann_vol < 1e-15)
/// * max drawdown from wealth V_0 = 1, V_t = V_{t-1} (1 + r_t):
///   maxdd = max_t (1 - V_t / peak_t), peak includes V_0; positive fraction.
#pragma once

#include <functional>
#include <vector>

#include "portopt/linalg.hpp"

namespace portopt {

/// Pinned defaults (data/config.json).
constexpr int kDefaultWindow = 500;
constexpr int kDefaultStep = 63;
constexpr int kAnnualization = 252;

/// Arithmetic annualized return: periods * mean(r).
double annualized_return(const Vector& r, int periods = kAnnualization);

/// Annualized volatility: sqrt(periods) * std(r, ddof=1). Needs >= 2 points.
double annualized_vol(const Vector& r, int periods = kAnnualization);

/// Annualized Sharpe (ann_return - rf) / ann_vol.
/// \throws std::domain_error if the volatility is below 1e-15.
double sharpe_ratio(const Vector& r, double rf = 0.0, int periods = kAnnualization);

/// Maximum drawdown of the compounded wealth path (positive fraction).
double max_drawdown(const Vector& r);

/// Pinned rebalance dates window, window+step, ... strictly below t.
/// \throws std::invalid_argument if window < 2, step < 1, or t <= window.
std::vector<std::size_t> rebalance_indices(std::size_t t, int window = kDefaultWindow,
                                           int step = kDefaultStep);

/// Result of run_backtest.
struct BacktestResult {
  Vector portfolio_returns;                 ///< Daily returns over rows [window, T).
  std::vector<std::size_t> rebalance_dates; ///< Row indices of the rebalances.
  Matrix weights;                           ///< K x n target weights per rebalance.
  Vector turnovers;                         ///< L1 turnover per rebalance after the first.
  double ann_return;
  double ann_vol;
  double sharpe;
  double max_drawdown;
  double avg_turnover;  ///< Mean of turnovers (0 when fewer than 2 rebalances).
};

/// Quarterly expanding-window backtest with the pinned schedule.
/// weight_fn(history) receives the expanding daily-return history (rows
/// [0, t_k)) and must return finite weights of length n summing to 1
/// within 1e-6 (validated; error otherwise).
BacktestResult run_backtest(const Matrix& returns,
                            const std::function<Vector(const Matrix&)>& weight_fn,
                            double rf = 0.0, int window = kDefaultWindow,
                            int step = kDefaultStep);

}  // namespace portopt
