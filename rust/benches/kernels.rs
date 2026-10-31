//! Criterion micro-benchmarks of the portopt kernels (`cargo bench`).
//!
//! The problems are the ones every port builds (docs/results/benchmarks.md):
//! SplitMix64 with seed 20261031, a T = 1250 one-factor return panel,
//! `sigma = 252 * sample_cov(returns)`, `b = ones(n)`, and an equal-weight
//! backtest with window 500 and step 63, at n = 8, 20, 50.

use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use portopt::{
    annualize_cov, chol_solve, cholesky, erc_weights, min_variance_longonly, run_backtest,
    sample_cov, shrink_covariance, Matrix,
};

const SEED: u64 = 20_261_031;
const ROWS: usize = 1250;
const WINDOW: usize = 500;
const STEP: usize = 63;
const SIZES: [usize; 3] = [8, 20, 50];

/// SplitMix64 (Steele, Lea and Flood 2014); identical in every port.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform on [0, 1) with 53 random bits.
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// `x[t][i] = 0.02 (1 + 0.25 (i mod 5)) (0.6 f_t + e_ti)`, `f_t` and `e_ti`
/// uniform on [-0.5, 0.5), drawn row by row (`f_t` first).
fn synthetic_returns(n: usize) -> Matrix {
    let mut rng = SplitMix64(SEED);
    let rows: Vec<Vec<f64>> = (0..ROWS)
        .map(|_| {
            let f = rng.uniform() - 0.5;
            (0..n)
                .map(|i| {
                    let e = rng.uniform() - 0.5;
                    0.02 * (1.0 + 0.25 * (i % 5) as f64) * (0.6 * f + e)
                })
                .collect()
        })
        .collect();
    Matrix::from_rows(&rows).expect("synthetic panel is rectangular")
}

fn sigma_for(x: &Matrix) -> Matrix {
    annualize_cov(&sample_cov(x).expect("sample_cov"), 252.0).expect("annualize_cov")
}

fn kernels(c: &mut Criterion) {
    for &n in &SIZES {
        let x = synthetic_returns(n);
        let sigma = sigma_for(&x);
        let b = vec![1.0; n];
        let equal = vec![1.0 / n as f64; n];

        c.bench_with_input(BenchmarkId::new("cholesky_solve", n), &n, |bench, _| {
            bench.iter(|| {
                let low = cholesky(black_box(&sigma)).expect("cholesky");
                chol_solve(&low, black_box(&b)).expect("chol_solve")
            })
        });
        c.bench_with_input(BenchmarkId::new("shrink_covariance", n), &n, |bench, _| {
            bench.iter(|| shrink_covariance(black_box(&x)).expect("shrink_covariance"))
        });
        c.bench_with_input(
            BenchmarkId::new("min_variance_longonly", n),
            &n,
            |bench, _| {
                bench.iter(|| min_variance_longonly(black_box(&sigma), None, None).expect("qp"))
            },
        );
        c.bench_with_input(BenchmarkId::new("erc_weights", n), &n, |bench, _| {
            bench.iter(|| erc_weights(black_box(&sigma)).expect("erc"))
        });
        c.bench_with_input(BenchmarkId::new("run_backtest", n), &n, |bench, _| {
            bench.iter(|| {
                run_backtest(black_box(&x), |_h| Ok(equal.clone()), 0.0, WINDOW, STEP)
                    .expect("backtest")
            })
        });
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2))
        .sample_size(30);
    targets = kernels
}
criterion_main!(benches);
