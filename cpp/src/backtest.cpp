#include "portopt/backtest.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

#include "portopt/estimators.hpp"

namespace portopt {

namespace {

void validate_series(const Vector& r, std::size_t min_len) {
  if (r.size() < min_len) {
    std::ostringstream msg;
    msg << "return series must have at least " << min_len << " observations";
    throw std::invalid_argument(msg.str());
  }
  for (double v : r) {
    if (!std::isfinite(v)) {
      throw std::invalid_argument("return series contains NaN or infinite entries");
    }
    if (v <= -1.0) {
      throw std::invalid_argument(
          "return series contains a simple return <= -1 (-100% or worse)");
    }
  }
}

void validate_periods(int periods) {
  if (periods <= 0) throw std::invalid_argument("periods must be positive and finite");
}

void validate_rf(double rf) {
  if (!std::isfinite(rf)) throw std::invalid_argument("rf must be finite");
}

/// Compensated sum: matches the reference's correctly rounded math.fsum
/// to a couple of ulps.
double series_sum(const Vector& r) { return kahan_sum(r); }

}  // namespace

double annualized_return(const Vector& r, int periods) {
  validate_periods(periods);
  validate_series(r, 1);
  return static_cast<double>(periods) * series_sum(r) / static_cast<double>(r.size());
}

double annualized_vol(const Vector& r, int periods) {
  validate_periods(periods);
  validate_series(r, 2);
  const double mu = series_sum(r) / static_cast<double>(r.size());
  Vector sq(r.size(), 0.0);
  for (std::size_t i = 0; i < r.size(); ++i) sq[i] = (r[i] - mu) * (r[i] - mu);
  const double var = series_sum(sq) / static_cast<double>(r.size() - 1);
  return std::sqrt(static_cast<double>(periods)) * std::sqrt(var);
}

double sharpe_ratio(const Vector& r, double rf, int periods) {
  validate_rf(rf);
  const double vol = annualized_vol(r, periods);
  if (vol < 1e-15) {
    throw std::domain_error("Sharpe undefined: portfolio volatility is zero");
  }
  return (annualized_return(r, periods) - rf) / vol;
}

double max_drawdown(const Vector& r) {
  validate_series(r, 1);
  double wealth = 1.0;
  double peak = 1.0;
  double mdd = 0.0;
  for (double ret : r) {
    wealth *= 1.0 + ret;
    if (wealth > peak) peak = wealth;
    const double dd = 1.0 - wealth / peak;
    if (dd > mdd) mdd = dd;
  }
  return mdd;
}

std::vector<std::size_t> rebalance_indices(std::size_t t, int window, int step) {
  if (window < 2) throw std::invalid_argument("estimation window must be >= 2");
  if (step < 1) throw std::invalid_argument("rebalance step must be >= 1");
  if (t <= static_cast<std::size_t>(window)) {
    std::ostringstream msg;
    msg << "sample length " << t << " must exceed the estimation window " << window;
    throw std::invalid_argument(msg.str());
  }
  std::vector<std::size_t> out;
  std::size_t tk = static_cast<std::size_t>(window);
  while (tk < t) {
    out.push_back(tk);
    // Overflow-safe advance: stop when the next date would be >= t.
    if (t - tk <= static_cast<std::size_t>(step)) break;
    tk += static_cast<std::size_t>(step);
  }
  return out;
}

BacktestResult run_backtest(const Matrix& returns,
                            const std::function<Vector(const Matrix&)>& weight_fn,
                            double rf, int window, int step) {
  validate_returns(returns, 3);
  validate_rf(rf);
  const std::size_t t = returns.rows();
  const std::size_t n = returns.cols();
  if (window < 2) throw std::invalid_argument("estimation window must be >= 2");
  if (t < static_cast<std::size_t>(window) + 2) {
    std::ostringstream msg;
    msg << "backtest needs at least 2 holdout days: sample length " << t
        << " must be >= window + 2 = " << window + 2;
    throw std::invalid_argument(msg.str());
  }
  const std::vector<std::size_t> dates = rebalance_indices(t, window, step);
  const std::size_t k = dates.size();
  Matrix weights(k, n, 0.0);
  Vector port(t - static_cast<std::size_t>(window), 0.0);
  for (std::size_t idx = 0; idx < k; ++idx) {
    const std::size_t t_k = dates[idx];
    // Expanding history rows [0, t_k).
    Matrix history(t_k, n);
    for (std::size_t r = 0; r < t_k; ++r) {
      for (std::size_t c = 0; c < n; ++c) history(r, c) = returns(r, c);
    }
    const Vector w = weight_fn(history);
    if (w.size() != n) {
      std::ostringstream msg;
      msg << "weight_fn returned an invalid weight vector at t=" << t_k;
      throw std::invalid_argument(msg.str());
    }
    double w_sum = 0.0;
    for (double v : w) {
      if (!std::isfinite(v)) {
        std::ostringstream msg;
        msg << "weight_fn returned an invalid weight vector at t=" << t_k;
        throw std::invalid_argument(msg.str());
      }
      w_sum += v;
    }
    if (std::fabs(w_sum - 1.0) > 1e-6) {
      std::ostringstream msg;
      msg << "weight_fn weights must sum to 1 (got " << w_sum << ") at t=" << t_k;
      throw std::invalid_argument(msg.str());
    }
    for (std::size_t c = 0; c < n; ++c) weights(idx, c) = w[c];
    const std::size_t t_end = (idx + 1 < k) ? dates[idx + 1] : t;
    Vector prod(n, 0.0);
    for (std::size_t r = t_k; r < t_end; ++r) {
      for (std::size_t c = 0; c < n; ++c) prod[c] = returns(r, c) * w[c];
      port[r - static_cast<std::size_t>(window)] = kahan_sum(prod);
    }
  }
  Vector turnovers;
  turnovers.reserve(k > 0 ? k - 1 : 0);
  Vector diffs(n, 0.0);
  for (std::size_t i = 1; i < k; ++i) {
    for (std::size_t c = 0; c < n; ++c) {
      diffs[c] = std::fabs(weights(i, c) - weights(i - 1, c));
    }
    turnovers.push_back(kahan_sum(diffs));
  }
  const double avg_to =
      turnovers.empty() ? 0.0 : series_sum(turnovers) / static_cast<double>(turnovers.size());
  BacktestResult out;
  out.portfolio_returns = port;
  out.rebalance_dates = dates;
  out.weights = weights;
  out.turnovers = turnovers;
  out.ann_return = annualized_return(port);
  out.ann_vol = annualized_vol(port);
  out.sharpe = sharpe_ratio(port, rf);
  out.max_drawdown = max_drawdown(port);
  out.avg_turnover = avg_to;
  return out;
}

}  // namespace portopt
