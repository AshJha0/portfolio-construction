# Kernel benchmarks

**These are micro-benchmarks from one laptop with other work running on it.** They tell you roughly what each kernel costs at small n and how the four implementations compare at the order-of-magnitude level. They are not a performance claim for any production setting, and differences of less than about 2x between languages should not be read as meaningful (see [What the numbers do and do not show](#what-the-numbers-do-and-do-not-show)).

Every figure on this page was measured on 2026-10-31 on the machine described below, from the benchmark code in this repository: `python/benchmarks/bench_kernels.py`, `cpp/benchmarks/bench_kernels.cpp`, `rust/benches/kernels.rs` and `java/src/bench/java/com/quant/portopt/bench/KernelBench.java`. The library code measured is the v1.2.0 tree plus the two changes described in the last two sections (the vectorised Python Ledoit-Wolf and the C++ history view).

## How to run

| language | command (from the repository root) | statistic reported here |
|---|---|---|
| Python | `pip install -e python`, then `python python/benchmarks/bench_kernels.py` (add `--lw` for the Ledoit-Wolf comparison) | median of 7 samples; a sample is enough back-to-back calls to fill 50 ms |
| C++ | `cmake -S cpp -B build-bench -DCMAKE_BUILD_TYPE=Release -DPORTOPT_BUILD_BENCHMARKS=ON`, `cmake --build build-bench --config Release`, then `build-bench/portopt_bench --benchmark_min_time=0.3s --benchmark_repetitions=5 --benchmark_report_aggregates_only=true` | median of 5 repetitions (Google Benchmark 1.9.1, wall time) |
| Rust | `cd rust && cargo bench` | criterion 0.5.1 median estimate (30 samples, 2 s measurement, 0.5 s warm-up) |
| Java | `cd java && mvn -B -Pbench package && java -jar target/benchmarks.jar` | median of the 5 one-second measurement iterations (JMH 1.37, average time, 5 two-second warm-up iterations, 1 fork) |

The C++ benchmarks are opt-in (`PORTOPT_BUILD_BENCHMARKS`, default OFF) and fetch Google Benchmark at a pinned URL and SHA-256. The Java benchmarks live in the `bench` Maven profile and are not part of `mvn verify`. The Rust benchmarks use criterion as a dev-dependency only, with default features off so that the dependency tree still builds on the MSRV, Rust 1.78. Python needs nothing beyond numpy.

## The problems

Every language builds the same inputs from the same generator, so a row of any table below is the same computation in four languages.

* Generator: SplitMix64 (Steele, Lea and Flood 2014) with seed 20261031; a uniform draw is `(next() >> 11) * 2^-53`.
* Returns: T = 1250 rows and n columns, drawn row by row. For row t, first `f = u - 0.5`, then for each column i, `e = u - 0.5` and `x[t][i] = 0.02 * (1 + 0.25 * (i mod 5)) * (0.6 f + e)`. This is a one-factor panel with five volatility levels, so the long-only minimum-variance portfolio has active bounds: 1, 9 and 35 zero weights at n = 8, 20 and 50.
* Covariance: `sigma = 252 * sample_cov(returns)`, built once outside the timed region; `b = ones(n)`.
* Sizes: n = 8 (the bundled data's width), 20 and 50.
* Check: copies of the four generators produce the same panel to the bit (the first and last entries and their sequential sum, compared as IEEE 754 bit patterns at n = 50).

What one call of each kernel does:

| kernel | one call |
|---|---|
| `cholesky_solve` | `cholesky(sigma)` (including its symmetry check) and `chol_solve(L, b)` |
| `shrink_covariance` | Ledoit-Wolf constant-correlation shrinkage of the T x n returns, sample covariance included |
| `min_variance_longonly` | the dual active-set QP for the fully invested long-only minimum-variance portfolio |
| `erc_weights` | equal risk contribution weights by cyclical coordinate descent to the pinned tolerance 1e-10 |
| `run_backtest` | the pinned walk-forward schedule (window 500, step 63: 12 rebalances, 750 holdout days) with a weight function that returns precomputed equal weights, so it times the history hand-off and the accounting, not an estimator |

## Machine, toolchains and conditions

| item | value |
|---|---|
| CPU | Intel Core 5 120U: 2 performance cores with two threads each and 8 efficient cores, 12 logical processors; L2 1280 KiB and L3 12 MiB as reported by Google Benchmark |
| memory, OS | 15.6 GB; Windows 11 Home, build 26200 |
| Python | CPython 3.12.10, numpy 2.4.6 |
| C++ | MSVC 19.51.36260 (Visual Studio 18 Build Tools), `/O2` Release build through CMake 4.4.3 with NMake; Google Benchmark 1.9.1 |
| Rust | rustc 1.99.1, release profile; criterion 0.5.1 |
| Java | Oracle HotSpot 27.0.1 (sources compiled for release 25), Maven 3.9.16; JMH 1.37 |

Conditions, which matter more than usual here:

* **The machine was not idle.** A four-process Python job from another session ran throughout. The machine-wide CPU load, sampled every 3 s during each suite, averaged 40% (range 14 to 71%) for C++, 26% (5 to 37%) for Rust, 45% (12 to 99%) for Java and 28 to 33% for Python.
* To limit the damage, every benchmark process ran at High priority with its affinity fixed to logical processors 2 and 3, which Windows reports as one performance core (`GetSystemCpuSetInformation`, efficiency class 1). The JMH fork was raised to the same priority and affinity right after it started. Nothing else was pinned, so other work could still share that core's second hardware thread.
* No frequency locking, no turbo or power-plan control, no core isolation; it is a laptop on its default power settings.
* Every figure is wall-clock time per call.
* The full set was run twice, between a few minutes and about half an hour apart per suite. The tables show the second pass. Of the 63 cells, 48 agreed between passes within 1.25x; the worst were 1.6 to 1.8x for some n = 8 and n = 20 cells in every language, 2.2x for the Java QP at n = 50 and 2.8x for the Java backtest at n = 8. Short calls and the JVM are the least stable.

## Results

### `cholesky_solve`

| language | n = 8 | n = 20 | n = 50 |
|---|---:|---:|---:|
| Python | 81.5 us | 654 us | 9.18 ms |
| C++ | 1.0 us | 4.9 us | 37.1 us |
| Rust | 0.62 us | 4.4 us | 40.2 us |
| Java | 1.8 us | 5.3 us | 47.6 us |

### `shrink_covariance` (T = 1250)

| language | n = 8 | n = 20 | n = 50 |
|---|---:|---:|---:|
| Python (vectorised default) | 3.52 ms | 12.9 ms | 21.8 ms |
| C++ | 0.77 ms | 4.58 ms | 23.3 ms |
| Rust | 0.62 ms | 3.67 ms | 20.7 ms |
| Java | 0.76 ms | 5.10 ms | 31.8 ms |

The Python time barely grows from n = 20 to n = 50 because up to n = 20 the vectorised path still forms the sample covariance with correctly rounded `math.fsum` sums (so that it is bit-identical to `sample_cov`), and that part dominates at n = 20. Above n = 20 it switches to numpy's pairwise sums. See the Ledoit-Wolf section below.

### `min_variance_longonly`

| language | n = 8 | n = 20 | n = 50 |
|---|---:|---:|---:|
| Python | 460 us | 10.3 ms | 830 ms |
| C++ | 8.6 us | 75.5 us | 2.85 ms |
| Rust | 8.3 us | 85.0 us | 3.10 ms |
| Java | 7.9 us | 84.1 us | 5.70 ms |

### `erc_weights`

| language | n = 8 | n = 20 | n = 50 |
|---|---:|---:|---:|
| Python | 417 us | 2.04 ms | 12.1 ms |
| C++ | 4.5 us | 19.1 us | 82.5 us |
| Rust | 4.9 us | 15.8 us | 102 us |
| Java | 5.4 us | 21.1 us | 120 us |

### `run_backtest` (equal-weight function)

| language | n = 8 | n = 20 | n = 50 |
|---|---:|---:|---:|
| Python (history passed as a read-only view) | 1.50 ms | 2.46 ms | 3.50 ms |
| C++, `HistoryView` overload (no copy) | 62.3 us | 133 us | 296 us |
| C++, `Matrix` overload (one copy per rebalance) | 172 us | 461 us | 1.51 ms |
| Rust (copies the history per rebalance) | 1.02 ms | 1.25 ms | 2.02 ms |
| Java (copies the history per rebalance) | 279 us | 687 us | 859 us |

### C++ compilers on Linux (WSL2, same laptop, not pinned)

For a sense of how much the compiler alone moves these numbers, the same C++ benchmark built with g++ 15.2 and clang 21.1 (`-O3` through CMake's Release type, `-ffp-contract=off`) in Ubuntu under WSL2 on the same laptop, without priority or affinity control:

| kernel, n = 8 / 50 | MSVC 19.51 (table above) | g++ 15.2 | clang 21.1 |
|---|---:|---:|---:|
| `shrink_covariance` | 0.77 / 23.3 ms | 0.35 / 14.2 ms | 0.76 / 27.2 ms |
| `min_variance_longonly` | 8.6 / 2850 us | 21.4 / 4590 us | 6.5 / 3100 us |
| `erc_weights` | 4.5 / 82.5 us | 11.0 / 128 us | 6.2 / 129 us |
| `run_backtest`, view | 62 / 296 us | 50 / 228 us | 59 / 259 us |

The same source differs by more than 3x across compilers on single kernels (the QP at n = 8), which is another reason not to rank the languages on small gaps.

## Python Ledoit-Wolf: pinned loops and the vectorised default

**The plain idea.** The Python reference computed every T-length sum in Ledoit-Wolf with an explicit loop over the N^2 asset pairs and `math.fsum`, so that its operation order matched the ports'. That costs O(N^2) interpreter round trips of length T. The default `shrink_covariance` now evaluates the same formulas on (i, j, t) tiles with numpy, and the old code is kept, unchanged, as `portopt.estimators._shrink_covariance_reference`.

**The formal statement.** In the vectorised default, the dispersion sums inside `pi_ij` and `theta_ii,ij` (API_SPEC 2.2) are numpy pairwise sums over t instead of `math.fsum`. Every other reduction is taken in the reference's order. For N <= 20 the sample covariance is the same correctly rounded `math.fsum` sum, so `sample`, `target` and `rbar` are bit-identical to the reference and `sample` is bit-identical to `sample_cov`. Above N = 20 the covariance sums are pairwise as well. `python/tests/test_estimators.py` requires `intensity` and `rbar` to agree with the reference to 1e-13 relative, and `sigma`, `sample` and `target` to 1e-13 relative to their largest entry, on the bundled data (including T <= N and the degenerate N = 2 case) and on random one-factor panels with N up to 100 and T of 60 and 400. It also checks that the tile size changes nothing, to the bit. In exploratory runs on the same kinds of problems the relative differences were at most about 5e-16. The golden shrinkage case (tolerance 1e-12) passes unchanged.

**The measurement.** T = 1250, the same generator as above, the two implementations sampled in turn (7 samples each, median and, in brackets, the fastest sample):

| n | loop reference | vectorised default | speed-up of the medians |
|---:|---:|---:|---:|
| 8 | 8.27 ms (7.04 ms) | 3.46 ms (2.75 ms) | 2.4x |
| 20 | 66.1 ms (51.8 ms) | 16.3 ms (14.4 ms) | 4.1x |
| 50 | 438 ms (354 ms) | 22.7 ms (20.8 ms) | 19.3x |
| 100 | 1.85 s (1.61 s) | 85.3 ms (70.6 ms) | 21.7x |

An earlier interleaved run gave 2.5x, 3.7x, 18.3x and 21.1x. Tile size matters on this machine: in one side-by-side run at N = 50, tiles of 2 MiB and 8 MiB took 63 and 70 ms against 17 to 19 ms for tiles of 0.5 MiB (the size now used), because the sums are limited by memory traffic once the temporaries no longer fit in L2.

## C++ backtest: the history view

**The plain idea.** `run_backtest` used to copy rows [0, t_k) of the returns into a new `Matrix` at every rebalance and hand that copy to the weight function. It now hands over a `portopt::HistoryView`, a non-owning, read-only view of those rows (a pointer to the returns plus a row count) that is valid for the duration of the call.

**The formal statement.** `cpp/include/portopt/backtest.hpp` declares two overloads. The new one takes `std::function<Vector(const HistoryView&)>` and copies nothing. The existing one, taking `std::function<Vector(const Matrix&)>`, is kept for source compatibility and gives identical results; it still materialises the rows with `HistoryView::to_matrix()` once per rebalance, because the estimators take `const Matrix&`. There is no implicit conversion between the two, so a lambda that names its parameter type selects exactly one overload; a generic `[](const auto&)` lambda is ambiguous and has to name the type. `cpp/tests/test_backtest.cpp` checks that the two overloads return bit-identical results and that the view sees exactly rows [0, t_k).

**The measurement.** With an equal-weight function, which does no estimation, the copy was 64% (n = 8) to 80% (n = 50) of the whole backtest's cost: 172 us against 62 us at n = 8 and 1.51 ms against 0.30 ms at n = 50. With a weight function that estimates a covariance, the copy is O(t n) against the estimator's O(t n^2), so the saving is a much smaller share of the total, and code that calls the estimators still has to make the copy itself (`to_matrix()`).

## What the numbers do and do not show

They show:

* the order of magnitude of each kernel at n = 8, 20 and 50, and how it grows with n;
* that the compiled ports are roughly 45 to 300 times faster than the Python reference on the scalar kernels (Cholesky, QP, ERC), whose Python code is written as explicit loops to pin the operation order, not for speed;
* the effect of the two changes above, measured as like-for-like comparisons on the same machine within a few minutes.

They do not show:

* a ranking of C++, Rust and Java. The ports are line-by-line transcriptions of the pinned reference (sequential and compensated sums, row-by-row Cholesky, no BLAS, no SIMD-friendly restructuring), the compiler alone moves single kernels by more than 3x, and the pass-to-pass spread on this machine reached 1.8x (2.8x for one Java cell).
* that vectorised Python beats the compiled ports at Ledoit-Wolf for n = 50 in any general sense. The ports' dispersion sums are scalar loops with a compensated (Neumaier) sum per pair, which the compilers cannot vectorise without reassociating floating-point additions; numpy's pairwise sum is vectorised. Restructuring the ports the same way would probably narrow or reverse the gap; it has not been tried.
* anything about large n. The design range of the linear algebra is n <= 20; n = 50 is already outside it, and nothing here says how the QP or the O(n^3) kernels behave at n = 500.
* performance on native Linux, macOS or other hardware, on an idle or isolated machine, or with frequency control. Hybrid laptop CPUs change clock speed with temperature and load, and the background job ran throughout.
* steady-state behaviour of the JVM beyond 10 s of warm-up per benchmark, or the cost of the first calls (JIT compilation), which the Java figures exclude by design.
