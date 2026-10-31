/// \file bench_kernels.cpp
/// \brief Google Benchmark micro-benchmarks of the portopt kernels.
///
/// Build with -DPORTOPT_BUILD_BENCHMARKS=ON (Release) and run portopt_bench.
/// The problems are the ones every port builds (docs/results/benchmarks.md):
/// SplitMix64 with seed 20261031, a T = 1250 one-factor return panel,
/// sigma = 252 * sample_cov(returns), b = ones(n), and an equal-weight
/// backtest with window 500 and step 63, at n = 8, 20, 50.
#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <vector>

#include "portopt/backtest.hpp"
#include "portopt/estimators.hpp"
#include "portopt/linalg.hpp"
#include "portopt/markowitz.hpp"
#include "portopt/risk_parity.hpp"

namespace {

using portopt::Matrix;
using portopt::Vector;

constexpr std::uint64_t kSeed = 20261031ULL;
constexpr std::size_t kRows = 1250;
constexpr int kWindow = 500;
constexpr int kStep = 63;

/// SplitMix64 (Steele, Lea and Flood 2014); identical in every port.
class SplitMix64 {
 public:
  explicit SplitMix64(std::uint64_t seed) : state_(seed) {}

  std::uint64_t next() {
    state_ += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = state_;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }

  /// Uniform on [0, 1) with 53 random bits.
  double uniform() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }

 private:
  std::uint64_t state_;
};

/// x(t, i) = 0.02 (1 + 0.25 (i mod 5)) (0.6 f_t + e_ti), f_t and e_ti
/// uniform on [-0.5, 0.5), drawn row by row (f_t first).
Matrix synthetic_returns(std::size_t n) {
  SplitMix64 rng(kSeed);
  Matrix x(kRows, n);
  for (std::size_t r = 0; r < kRows; ++r) {
    const double f = rng.uniform() - 0.5;
    for (std::size_t i = 0; i < n; ++i) {
      const double e = rng.uniform() - 0.5;
      x(r, i) = 0.02 * (1.0 + 0.25 * static_cast<double>(i % 5)) * (0.6 * f + e);
    }
  }
  return x;
}

std::size_t size_arg(const benchmark::State& state) {
  return static_cast<std::size_t>(state.range(0));
}

void BM_CholeskySolve(benchmark::State& state) {
  const std::size_t n = size_arg(state);
  const Matrix sigma = portopt::annualize_cov(portopt::sample_cov(synthetic_returns(n)));
  const Vector b(n, 1.0);
  for (auto _ : state) {
    Vector x = portopt::chol_solve(portopt::cholesky(sigma), b);
    benchmark::DoNotOptimize(x.data());
  }
}

void BM_ShrinkCovariance(benchmark::State& state) {
  const Matrix x = synthetic_returns(size_arg(state));
  for (auto _ : state) {
    portopt::ShrinkageResult res = portopt::shrink_covariance(x);
    benchmark::DoNotOptimize(res.intensity);
  }
}

void BM_MinVarianceLongOnly(benchmark::State& state) {
  const std::size_t n = size_arg(state);
  const Matrix sigma = portopt::annualize_cov(portopt::sample_cov(synthetic_returns(n)));
  for (auto _ : state) {
    Vector w = portopt::min_variance_longonly(sigma);
    benchmark::DoNotOptimize(w.data());
  }
}

void BM_ErcWeights(benchmark::State& state) {
  const std::size_t n = size_arg(state);
  const Matrix sigma = portopt::annualize_cov(portopt::sample_cov(synthetic_returns(n)));
  for (auto _ : state) {
    Vector w = portopt::erc_weights(sigma);
    benchmark::DoNotOptimize(w.data());
  }
}

/// The weight function sees a non-owning HistoryView (no copy).
void BM_RunBacktest(benchmark::State& state) {
  const std::size_t n = size_arg(state);
  const Matrix x = synthetic_returns(n);
  const Vector equal(n, 1.0 / static_cast<double>(n));
  const auto fn = [&equal](const portopt::HistoryView&) { return equal; };
  for (auto _ : state) {
    portopt::BacktestResult r = portopt::run_backtest(x, fn, 0.0, kWindow, kStep);
    benchmark::DoNotOptimize(r.sharpe);
  }
}

/// The source-compatible Matrix overload: one history copy per rebalance.
void BM_RunBacktestMatrixCopy(benchmark::State& state) {
  const std::size_t n = size_arg(state);
  const Matrix x = synthetic_returns(n);
  const Vector equal(n, 1.0 / static_cast<double>(n));
  const auto fn = [&equal](const Matrix&) { return equal; };
  for (auto _ : state) {
    portopt::BacktestResult r = portopt::run_backtest(x, fn, 0.0, kWindow, kStep);
    benchmark::DoNotOptimize(r.sharpe);
  }
}

void sizes(benchmark::internal::Benchmark* b) {
  for (const int n : {8, 20, 50}) b->Arg(n);
}

}  // namespace

BENCHMARK(BM_CholeskySolve)->Apply(sizes);
BENCHMARK(BM_ShrinkCovariance)->Apply(sizes)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_MinVarianceLongOnly)->Apply(sizes)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_ErcWeights)->Apply(sizes)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_RunBacktest)->Apply(sizes)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_RunBacktestMatrixCopy)->Apply(sizes)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
