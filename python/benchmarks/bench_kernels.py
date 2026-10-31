"""Micro-benchmarks of the Python reference kernels (no extra dependency).

Run from the repository root after ``pip install -e python``::

    python python/benchmarks/bench_kernels.py            # n = 8, 20, 50
    python python/benchmarks/bench_kernels.py --lw       # Ledoit-Wolf fast vs loop, n up to 100
    python python/benchmarks/bench_kernels.py --quick    # fewer repeats

Every language's benchmark builds the same problems from the same seeded
generator (see ``docs/results/benchmarks.md``): SplitMix64 with seed
20261031, a T = 1250 one-factor return panel, ``sigma = 252 *
sample_cov(returns)``, ``b = ones(n)``, and an equal-weight backtest with
window 500 and step 63. Each figure is the median over ``--repeats``
samples of the per-call time, a sample being as many back-to-back calls as
fill about ``--min-sample-ms`` milliseconds.
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from functools import partial

import numpy as np

import portopt as po
from portopt.estimators import _shrink_covariance_reference

SEED = 20261031
T_ROWS = 1250
WINDOW = 500
STEP = 63
MASK64 = (1 << 64) - 1


class SplitMix64:
    """SplitMix64 (Steele, Lea and Flood 2014); identical in every port."""

    def __init__(self, seed: int) -> None:
        self.state = seed & MASK64

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def uniform(self) -> float:
        """Uniform on [0, 1) with 53 random bits."""
        return (self.next_u64() >> 11) * 2.0**-53


def synthetic_returns(n: int, t: int = T_ROWS, seed: int = SEED) -> np.ndarray:
    """x[t, i] = 0.02 * (1 + 0.25 * (i % 5)) * (0.6 * f_t + e_ti), with f_t
    and e_ti uniform on [-0.5, 0.5), drawn row by row (f_t first)."""
    rng = SplitMix64(seed)
    x = np.empty((t, n))
    for r in range(t):
        f = rng.uniform() - 0.5
        for i in range(n):
            e = rng.uniform() - 0.5
            x[r, i] = 0.02 * (1.0 + 0.25 * (i % 5)) * (0.6 * f + e)
    return x


def time_call(fn: Callable[[], object], repeats: int, min_sample_ms: float) -> float:
    """Median seconds per call."""
    fn()  # warm-up
    t0 = time.perf_counter()
    fn()
    one = max(time.perf_counter() - t0, 1e-9)
    inner = max(1, int(min_sample_ms / 1000.0 / one))
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        samples.append((time.perf_counter() - t0) / inner)
    return statistics.median(samples)


def time_interleaved(
    fns: list[Callable[[], object]], repeats: int, min_sample_ms: float
) -> list[tuple[float, float]]:
    """(median, min) seconds per call for each function, with the samples
    of the functions taken in turn so that a change in machine load hits
    all of them alike."""
    inners = []
    for fn in fns:
        fn()  # warm-up
        t0 = time.perf_counter()
        fn()
        inners.append(max(1, int(min_sample_ms / 1000.0 / max(time.perf_counter() - t0, 1e-9))))
    samples: list[list[float]] = [[] for _ in fns]
    for _ in range(repeats):
        for k, fn in enumerate(fns):
            t0 = time.perf_counter()
            for _ in range(inners[k]):
                fn()
            samples[k].append((time.perf_counter() - t0) / inners[k])
    return [(statistics.median(s), min(s)) for s in samples]


def fmt(seconds: float) -> str:
    if seconds >= 1.0:
        return f"{seconds:.3f} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.3f} ms"
    return f"{seconds * 1e6:.1f} us"


def kernels(n: int) -> dict[str, Callable[[], object]]:
    x = synthetic_returns(n)
    sigma = po.annualize_cov(po.sample_cov(x))
    b = np.ones(n)
    equal = np.full(n, 1.0 / n)
    return {
        "cholesky_solve": lambda: po.chol_solve(po.cholesky(sigma), b),
        "shrink_covariance": lambda: po.shrink_covariance(x),
        "min_variance_longonly": lambda: po.min_variance_longonly(sigma),
        "erc_weights": lambda: po.erc_weights(sigma),
        "run_backtest": lambda: po.run_backtest(x, lambda h: equal, 0.0, WINDOW, STEP),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sizes", type=int, nargs="+", default=[8, 20, 50])
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--min-sample-ms", type=float, default=50.0)
    ap.add_argument("--quick", action="store_true", help="3 repeats, 10 ms samples")
    ap.add_argument(
        "--lw", action="store_true", help="Ledoit-Wolf vectorised vs pinned loop, n = 8..100"
    )
    args = ap.parse_args()
    repeats, min_ms = (3, 10.0) if args.quick else (args.repeats, args.min_sample_ms)

    if args.lw:
        print(f"Ledoit-Wolf, T = {T_ROWS}, median (min) of {repeats}, interleaved")
        print("| n | loop reference | vectorised default | speed-up (medians) |")
        print("|---:|---:|---:|---:|")
        for n in [8, 20, 50, 100] if args.sizes == [8, 20, 50] else args.sizes:
            x = synthetic_returns(n)
            (ref, ref_min), (fast, fast_min) = time_interleaved(
                [partial(_shrink_covariance_reference, x), partial(po.shrink_covariance, x)],
                repeats,
                min_ms,
            )
            print(
                f"| {n} | {fmt(ref)} ({fmt(ref_min)}) | {fmt(fast)} ({fmt(fast_min)}) "
                f"| {ref / fast:.1f}x |"
            )
        return

    print(f"Python reference kernels, T = {T_ROWS}, median of {repeats}")
    print("| kernel | " + " | ".join(f"n={n}" for n in args.sizes) + " |")
    print("|---|" + "---:|" * len(args.sizes))
    table: dict[str, list[str]] = {}
    for n in args.sizes:
        for name, fn in kernels(n).items():
            table.setdefault(name, []).append(fmt(time_call(fn, repeats, min_ms)))
    for name, cells in table.items():
        print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
