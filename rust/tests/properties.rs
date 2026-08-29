//! Property, limit, and edge-case tests from the P10 spec's
//! "tests must also cover" list, on the bundled 8-asset data.

use std::path::{Path, PathBuf};

use portopt as po;
use portopt::linalg::dot;
use portopt::{Matrix, PortOptError};

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("data")
}

struct Ctx {
    names: Vec<String>,
    returns: Matrix,
    mu: Vec<f64>,
    sig: Matrix,
    w_mkt: Vec<f64>,
    views: po::ViewsDoc,
}

fn ctx() -> Ctx {
    let dir = data_dir();
    let (names, returns) = po::load_returns_csv(&dir.join("asset_returns.csv")).unwrap();
    let (_, caps) = po::load_market_caps(&dir.join("market_caps.csv")).unwrap();
    let views = po::load_views(&dir.join("views.json")).unwrap();
    let mu = po::annualize_mean(&po::sample_mean(&returns).unwrap(), 252.0).unwrap();
    let sig = po::annualize_cov(&po::sample_cov(&returns).unwrap(), 252.0).unwrap();
    let cap_sum: f64 = caps.iter().sum();
    let w_mkt = caps.iter().map(|c| c / cap_sum).collect();
    Ctx { names, returns, mu, sig, w_mkt, views }
}

const RF: f64 = 0.02;

// ------------------------------------------------------------------ frontier

#[test]
fn frontier_variance_convex_and_analytic() {
    let c = ctx();
    let sweep = po::efficient_frontier(&c.mu, &c.sig, 25).unwrap();
    let sc = po::two_fund_scalars(&c.mu, &c.sig).unwrap();
    // Convexity on the grid: non-negative second differences.
    for k in 1..24 {
        let second =
            sweep.variances[k + 1] - 2.0 * sweep.variances[k] + sweep.variances[k - 1];
        assert!(second > -1e-14, "second difference {second:.3e} at k={k}");
    }
    // Analytic identity var(m) = (C m^2 - 2 A m + B) / D.
    for k in 0..25 {
        let m = sweep.targets[k];
        let analytic = (sc.c * m * m - 2.0 * sc.a * m + sc.b) / sc.d;
        assert!((sweep.variances[k] - analytic).abs() < 1e-12);
    }
    // Min-variance portfolio has the smallest variance on the grid.
    let w_mv = po::min_variance_weights(&c.sig).unwrap();
    let v_mv = dot(&w_mv, &c.sig.matvec(&w_mv));
    for &v in &sweep.variances {
        assert!(v >= v_mv - 1e-14);
    }
}

#[test]
fn tangency_has_max_sharpe_on_frontier_grid() {
    let c = ctx();
    let w_tan = po::tangency_weights(&c.mu, &c.sig, RF).unwrap();
    let sharpe_tan =
        (dot(&c.mu, &w_tan) - RF) / dot(&w_tan, &c.sig.matvec(&w_tan)).sqrt();
    let sweep = po::efficient_frontier(&c.mu, &c.sig, 25).unwrap();
    for k in 0..25 {
        let w = &sweep.weights[k];
        let sharpe_k = (dot(&c.mu, w) - RF) / sweep.variances[k].sqrt();
        assert!(sharpe_k <= sharpe_tan + 1e-12);
    }
}

#[test]
fn frontier_weights_hit_target_and_reject_bad_targets() {
    let c = ctx();
    let w = po::frontier_weights(&c.mu, &c.sig, 0.08).unwrap();
    assert!((dot(&c.mu, &w) - 0.08).abs() < 1e-12);
    assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    let sc = po::two_fund_scalars(&c.mu, &c.sig).unwrap();
    assert!(matches!(
        po::frontier_weights(&c.mu, &c.sig, sc.a / sc.c - 0.01),
        Err(PortOptError::Undefined(_))
    ));
    assert!(matches!(
        po::frontier_weights(&c.mu, &c.sig, -0.05),
        Err(PortOptError::Undefined(_))
    ));
}

#[test]
fn two_fund_scalars_match_spd_inverse() {
    let c = ctx();
    let sc = po::two_fund_scalars(&c.mu, &c.sig).unwrap();
    let inv = po::spd_inverse(&c.sig).unwrap();
    let ones = vec![1.0; 8];
    let a = dot(&ones, &inv.matvec(&c.mu));
    let b = dot(&c.mu, &inv.matvec(&c.mu));
    let cc = dot(&ones, &inv.matvec(&ones));
    assert!((sc.a - a).abs() < 1e-10);
    assert!((sc.b - b).abs() < 1e-10);
    assert!((sc.c - cc).abs() < 1e-8);
    assert!(sc.b > 0.0 && sc.c > 0.0 && sc.d > 0.0);
}

#[test]
fn tangency_rf_above_all_returns_rejected() {
    let c = ctx();
    let too_high = c.mu.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 0.01;
    assert!(matches!(
        po::tangency_weights(&c.mu, &c.sig, too_high),
        Err(PortOptError::Undefined(_))
    ));
}

#[test]
fn singular_covariance_from_duplicated_asset_rejected() {
    let c = ctx();
    // Duplicate column 0 -> exactly singular covariance.
    let rows: Vec<Vec<f64>> = (0..c.returns.nrows())
        .map(|t| {
            vec![
                c.returns.get(t, 0),
                c.returns.get(t, 0),
                c.returns.get(t, 1),
            ]
        })
        .collect();
    let dup = Matrix::from_rows(&rows).unwrap();
    let sig = po::sample_cov(&dup).unwrap();
    assert!(matches!(
        po::min_variance_weights(&sig),
        Err(PortOptError::NotPositiveDefinite(_))
    ));
}

// ----------------------------------------------------------------------- PGD

#[test]
fn pgd_matches_closed_form_when_constraints_inactive() {
    let c = ctx();
    let w_cf = po::min_variance_weights(&c.sig).unwrap();
    let lb = vec![-10.0; 8];
    let ub = vec![10.0; 8];
    let w_pgd = po::min_variance_longonly(&c.sig, Some(&lb), Some(&ub)).unwrap();
    for i in 0..8 {
        assert!((w_pgd[i] - w_cf[i]).abs() < 1e-6);
    }
}

#[test]
fn pgd_longonly_feasible_and_deterministic() {
    let c = ctx();
    let w1 = po::min_variance_longonly(&c.sig, None, None).unwrap();
    let w2 = po::min_variance_longonly(&c.sig, None, None).unwrap();
    assert_eq!(w1, w2, "PGD must be bit-deterministic");
    assert!(w1.iter().all(|&v| v >= -1e-14));
    assert!((w1.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    // Long-only optimum cannot beat the unconstrained optimum.
    let w_cf = po::min_variance_weights(&c.sig).unwrap();
    let v_lo = dot(&w1, &c.sig.matvec(&w1));
    let v_cf = dot(&w_cf, &c.sig.matvec(&w_cf));
    assert!(v_lo >= v_cf - 1e-14);
}

#[test]
fn constrained_frontier_penalty_return_near_target() {
    let c = ctx();
    let w = po::constrained_frontier_weights(&c.mu, &c.sig, 0.08, 1e4, None, None).unwrap();
    assert!(w.iter().all(|&v| v >= -1e-14));
    assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    // Penalty formulation: achieved return within O(1/penalty) of target.
    assert!((dot(&c.mu, &w) - 0.08).abs() < 1e-3);
}

#[test]
fn simplex_projection_euclidean_optimality_grid() {
    // Deterministic pseudo-random inputs (LCG), checked against a grid of
    // feasible competitors: the projection must be the closest point.
    let mut state = 123456789u64;
    let mut next = move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    for _ in 0..20 {
        let v: Vec<f64> = (0..6).map(|_| next() * 2.0).collect();
        let ub = vec![0.6; 6];
        let w = po::project_simplex_box(&v, 1.0, None, Some(&ub)).unwrap();
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(w.iter().all(|&x| (-1e-14..=0.6 + 1e-14).contains(&x)));
        let d_w: f64 = (0..6).map(|i| (w[i] - v[i]) * (w[i] - v[i])).sum();
        for _ in 0..10 {
            // Feasible competitor: normalized clipped positive vector.
            let mut z: Vec<f64> = (0..6).map(|_| next().abs() + 0.05).collect();
            let zs: f64 = z.iter().sum();
            z.iter_mut().for_each(|x| *x /= zs);
            if z.iter().any(|&x| x > 0.6) {
                continue;
            }
            let d_z: f64 = (0..6).map(|i| (z[i] - v[i]) * (z[i] - v[i])).sum();
            assert!(d_w <= d_z + 1e-10);
        }
    }
}

// ------------------------------------------------------------------ shrinkage

#[test]
fn shrinkage_intensity_in_unit_interval_and_psd() {
    let c = ctx();
    let sh = po::shrink_covariance(&c.returns).unwrap();
    assert!((0.0..=1.0).contains(&sh.intensity));
    assert!((-1.0..=1.0).contains(&sh.rbar));
    // Shrunk matrix stays PSD: Cholesky must succeed.
    po::cholesky(&sh.sigma).expect("shrunk covariance must be SPD");
    // Convex combination: entry-wise between S and F (diagonal preserved).
    for i in 0..8 {
        assert!((sh.sigma.get(i, i) - sh.sample.get(i, i)).abs() < 1e-18);
        for j in 0..8 {
            let lo = sh.sample.get(i, j).min(sh.target.get(i, j)) - 1e-18;
            let hi = sh.sample.get(i, j).max(sh.target.get(i, j)) + 1e-18;
            assert!(sh.sigma.get(i, j) >= lo && sh.sigma.get(i, j) <= hi);
        }
    }
    // Target structure: F_ij = rbar * sd_i * sd_j off the diagonal.
    for i in 0..8 {
        for j in 0..8 {
            if i != j {
                let expect = sh.sample.get(i, i).sqrt() * sh.sample.get(j, j).sqrt() * sh.rbar;
                assert!((sh.target.get(i, j) - expect).abs() < 1e-15);
            }
        }
    }
}

// ------------------------------------------------------------------ risk parity

#[test]
fn risk_contribution_identities() {
    let c = ctx();
    let w_erc = po::erc_weights(&c.sig).unwrap();
    let rc = po::risk_contributions(&w_erc, &c.sig).unwrap();
    // Normalized contributions sum to exactly 1 (identity, 1e-12).
    assert!((rc.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    // ERC: contributions equal within tolerance.
    for &r in &rc {
        assert!((r - 0.125).abs() < 1e-9);
    }
    // General budgets are met at the CCD fixed point.
    let budgets = [0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05];
    let w_rb = po::risk_budget_weights(&c.sig, Some(&budgets)).unwrap();
    let rc_rb = po::risk_contributions(&w_rb, &c.sig).unwrap();
    for i in 0..8 {
        assert!((rc_rb[i] - budgets[i]).abs() < 1e-9);
    }
    // Budget normalization: scaled budgets give the same weights.
    let scaled: Vec<f64> = budgets.iter().map(|b| 7.0 * b).collect();
    let w_rb2 = po::risk_budget_weights(&c.sig, Some(&scaled)).unwrap();
    for i in 0..8 {
        assert!((w_rb[i] - w_rb2[i]).abs() < 1e-12);
    }
}

#[test]
fn ccd_determinism() {
    let c = ctx();
    let w1 = po::erc_weights(&c.sig).unwrap();
    let w2 = po::erc_weights(&c.sig).unwrap();
    assert_eq!(w1, w2, "CCD must be bit-deterministic");
}

// ------------------------------------------------------------- black-litterman

#[test]
fn implied_pi_reverse_roundtrip() {
    // Tangency at rf = 0 on (pi, Sigma) recovers exactly w_mkt.
    let c = ctx();
    let pi = po::implied_equilibrium_returns(&c.sig, &c.w_mkt, 2.5).unwrap();
    let w = po::tangency_weights(&pi, &c.sig, 0.0).unwrap();
    for i in 0..8 {
        assert!((w[i] - c.w_mkt[i]).abs() < 1e-10);
    }
}

#[test]
fn bl_limits_and_conflicting_views() {
    let c = ctx();
    let pi = po::implied_equilibrium_returns(&c.sig, &c.w_mkt, c.views.delta).unwrap();
    let (p, q, conf) = po::views_to_matrices(&c.views.views, &c.names).unwrap();
    let omega = po::default_omega(&c.sig, &p, c.views.tau, Some(&conf)).unwrap();

    // tau -> 0 at fixed Omega: posterior -> prior, M -> 0.
    let small = po::bl_posterior(&c.sig, &pi, &p, &q, 1e-9, Some(&omega), None).unwrap();
    for i in 0..8 {
        assert!((small.mu_post[i] - pi[i]).abs() < 1e-8);
        for j in 0..8 {
            assert!(small.m.get(i, j).abs() < 1e-9);
        }
    }

    // Omega -> inf (zero confidence): posterior -> prior, M -> tau*Sigma.
    let huge = Matrix::from_diag(&[1e12, 1e12]);
    let nc = po::bl_posterior(&c.sig, &pi, &p, &q, c.views.tau, Some(&huge), None).unwrap();
    for i in 0..8 {
        assert!((nc.mu_post[i] - pi[i]).abs() < 1e-10);
        for j in 0..8 {
            assert!((nc.m.get(i, j) - c.views.tau * c.sig.get(i, j)).abs() < 1e-10);
        }
    }

    // Conflicting extreme views: posterior stays finite.
    let p2 = Matrix::from_rows(&[
        {
            let mut r = vec![0.0; 8];
            r[0] = 1.0;
            r
        },
        {
            let mut r = vec![0.0; 8];
            r[0] = 1.0;
            r
        },
    ])
    .unwrap();
    let q2 = [5.0, -5.0];
    let res = po::bl_posterior(&c.sig, &pi, &p2, &q2, c.views.tau, None, None).unwrap();
    assert!(res.mu_post.iter().all(|v| v.is_finite()));
    assert!(res.sigma_post.is_finite());

    // Default Omega confidence scaling: c = 0.5 doubles the variance.
    let om1 = po::default_omega(&c.sig, &p, c.views.tau, None).unwrap();
    let om2 = po::default_omega(&c.sig, &p, c.views.tau, Some(&[0.5, 0.5])).unwrap();
    for k in 0..2 {
        assert!((om2.get(k, k) - 2.0 * om1.get(k, k)).abs() < 1e-15);
    }
}

#[test]
fn bl_error_paths() {
    let c = ctx();
    let pi = po::implied_equilibrium_returns(&c.sig, &c.w_mkt, 2.5).unwrap();
    let (p, q, _) = po::views_to_matrices(&c.views.views, &c.names).unwrap();
    // tau out of (0, 1].
    assert!(po::bl_posterior(&c.sig, &pi, &p, &q, 0.0, None, None).is_err());
    assert!(po::bl_posterior(&c.sig, &pi, &p, &q, 1.0000001, None, None).is_err());
    // delta <= 0; negative caps.
    assert!(po::implied_equilibrium_returns(&c.sig, &c.w_mkt, 0.0).is_err());
    assert!(po::implied_equilibrium_returns(&c.sig, &[-1.0; 8], 2.5).is_err());
    // Q length mismatch and zero P row.
    assert!(po::bl_posterior(&c.sig, &pi, &p, &[0.1], 0.05, None, None).is_err());
    let zero_p = Matrix::zeros(1, 8);
    assert!(po::bl_posterior(&c.sig, &pi, &zero_p, &[0.1], 0.05, None, None).is_err());
    // Non-positive confidence.
    assert!(po::default_omega(&c.sig, &p, 0.05, Some(&[1.0, 0.0])).is_err());
}

// ------------------------------------------------------------------- backtest

#[test]
fn backtest_accounting_identity() {
    // Portfolio returns must equal w . r_t with the pinned weight blocks.
    let c = ctx();
    let r = po::run_backtest(
        &c.returns,
        |h: &Matrix| Ok(vec![1.0 / h.ncols() as f64; h.ncols()]),
        RF,
        500,
        63,
    )
    .unwrap();
    assert_eq!(r.portfolio_returns.len(), c.returns.nrows() - 500);
    assert_eq!(r.rebalance_dates.len(), 12);
    let w = vec![0.125; 8];
    for (k, row) in (500..c.returns.nrows()).enumerate() {
        let expect = dot(c.returns.row(row), &w);
        assert!((r.portfolio_returns[k] - expect).abs() < 1e-16);
    }
    // Equal-weight turnover is exactly zero at every rebalance.
    assert!(r.turnovers.iter().all(|&t| t == 0.0));
    assert_eq!(r.avg_turnover, 0.0);
    // Metric consistency with the standalone functions.
    assert_eq!(
        r.ann_return,
        po::annualized_return(&r.portfolio_returns, 252.0).unwrap()
    );
    assert_eq!(r.max_drawdown, po::max_drawdown(&r.portfolio_returns).unwrap());
}

#[test]
fn backtest_expanding_window_and_validation() {
    let c = ctx();
    // The weight function must see strictly growing histories of the
    // pinned lengths 500, 563, ...
    let mut seen: Vec<usize> = Vec::new();
    let r = po::run_backtest(
        &c.returns,
        |h: &Matrix| {
            seen.push(h.nrows());
            Ok(vec![1.0 / h.ncols() as f64; h.ncols()])
        },
        RF,
        500,
        63,
    );
    assert!(r.is_ok());
    let expect: Vec<usize> = (0..12).map(|k| 500 + 63 * k).collect();
    assert_eq!(seen, expect);

    // T <= window is rejected.
    let short = Matrix::from_rows(
        &(0..100).map(|t| c.returns.row(t).to_vec()).collect::<Vec<_>>(),
    )
    .unwrap();
    assert!(po::run_backtest(
        &short,
        |h: &Matrix| Ok(vec![1.0 / h.ncols() as f64; h.ncols()]),
        RF,
        500,
        63
    )
    .is_err());

    // Weights that do not sum to 1 are rejected.
    let bad = po::run_backtest(
        &c.returns,
        |h: &Matrix| Ok(vec![0.5 / h.ncols() as f64; h.ncols()]),
        RF,
        500,
        63,
    );
    assert!(matches!(bad, Err(PortOptError::InvalidInput(_))));

    // A weight-function error propagates.
    let failing = po::run_backtest(
        &c.returns,
        |_: &Matrix| Err(PortOptError::Undefined("boom".into())),
        RF,
        500,
        63,
    );
    assert!(matches!(failing, Err(PortOptError::Undefined(_))));
}

#[test]
fn backtest_determinism() {
    let c = ctx();
    let go = || {
        po::run_backtest(
            &c.returns,
            |h: &Matrix| po::erc_weights(&po::sample_cov(h)?),
            RF,
            500,
            63,
        )
        .unwrap()
    };
    let a = go();
    let b = go();
    assert_eq!(a.portfolio_returns, b.portfolio_returns);
    assert_eq!(a.sharpe, b.sharpe);
    assert_eq!(a.weights, b.weights);
}

// -------------------------------------------------------------------- linalg

#[test]
fn cholesky_reconstruction_grid() {
    // Property loop: A = L L' for a family of SPD matrices
    // A(rho) = rho * ones + (1 - rho) * I scaled by a vol pattern.
    for step in 0..8 {
        let rho = 0.1 * step as f64;
        let n = 5usize;
        let mut a = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let vi = 0.1 + 0.05 * i as f64;
                let vj = 0.1 + 0.05 * j as f64;
                let corr = if i == j { 1.0 } else { rho };
                a.set(i, j, corr * vi * vj);
            }
        }
        let l = po::cholesky(&a).unwrap();
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += l.get(i, k) * l.get(j, k);
                }
                assert!((s - a.get(i, j)).abs() < 1e-14);
                if j > i {
                    assert_eq!(l.get(i, j), 0.0);
                }
            }
        }
        let x = po::spd_solve(&a, &vec![1.0; n]).unwrap();
        let ax = a.matvec(&x);
        for i in 0..n {
            assert!((ax[i] - 1.0).abs() < 1e-11);
        }
    }
}
