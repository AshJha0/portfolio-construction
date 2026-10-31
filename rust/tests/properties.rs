//! Property, limit, and edge-case tests on the bundled 8-asset data.

use std::path::{Path, PathBuf};

use portopt as po;
use portopt::linalg::dot;
use portopt::{Matrix, PortOptError};

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("data")
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
    Ctx {
        names,
        returns,
        mu,
        sig,
        w_mkt,
        views,
    }
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
        let second = sweep.variances[k + 1] - 2.0 * sweep.variances[k] + sweep.variances[k - 1];
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
    let sharpe_tan = (dot(&c.mu, &w_tan) - RF) / dot(&w_tan, &c.sig.matvec(&w_tan)).sqrt();
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
        Err(PortOptError::TangencyUndefined(_))
    ));
    assert!(matches!(
        po::tangency_weights(&c.mu, &c.sig, f64::NAN),
        Err(PortOptError::InvalidInput(_))
    ));
}

#[test]
fn tangency_near_gmv_rejected() {
    // MAJ-1: rf within 1e-6 (relative) of A/C cancels A - rf*C catastrophically.
    let c = ctx();
    let sc = po::two_fund_scalars(&c.mu, &c.sig).unwrap();
    let r_gmv = sc.a / sc.c;
    for eps in [1e-13, 1e-9] {
        assert!(matches!(
            po::tangency_weights(&c.mu, &c.sig, r_gmv - eps),
            Err(PortOptError::TangencyUndefined(_))
        ));
    }
    let w = po::tangency_weights(&c.mu, &c.sig, r_gmv - 1e-3).unwrap();
    assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-8);
    let l1: f64 = w.iter().map(|v| v.abs()).sum();
    assert!((po::gross_leverage(&w).unwrap() - l1).abs() < 1e-12);
    assert!((po::gross_leverage(&[0.2, 0.3, 0.5]).unwrap() - 1.0).abs() < 1e-15);
    assert!((po::gross_leverage(&[1.5, -0.5]).unwrap() - 2.0).abs() < 1e-15);
    assert!(po::gross_leverage(&[]).is_err());
    assert!(po::gross_leverage(&[f64::NAN]).is_err());
}

#[test]
fn frontier_grid_when_gmv_return_exceeds_max_mu() {
    // MAJ-3: GMV shorts the low-return asset, A/C = 0.1185 > max(mu) = 0.10.
    let mu = [0.10, 0.05];
    let sig = Matrix::from_rows(&[vec![0.01, 0.027], vec![0.027, 0.09]]).unwrap();
    let sc = po::two_fund_scalars(&mu, &sig).unwrap();
    assert!(sc.a / sc.c > 0.10);
    assert!(matches!(
        po::efficient_frontier(&mu, &sig, 25),
        Err(PortOptError::Undefined(_))
    ));
    assert!(matches!(
        po::constrained_frontier(&mu, &sig, 25, None, None),
        Err(PortOptError::Undefined(_))
    ));
    let w = po::frontier_weights(&mu, &sig, 0.13).unwrap();
    assert!((dot(&mu, &w) - 0.13).abs() < 1e-12);
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

// ------------------------------------------------------------------------ QP

#[test]
#[allow(clippy::needless_range_loop)] // index-style checks mirror the Python reference formulas
fn qp_matches_closed_form_when_constraints_inactive() {
    let c = ctx();
    let w_cf = po::min_variance_weights(&c.sig).unwrap();
    let lb = vec![-10.0; 8];
    let ub = vec![10.0; 8];
    let res =
        po::min_variance_longonly_solve(&c.sig, Some(&lb), Some(&ub), po::QP_MAX_ITER).unwrap();
    assert!(res.converged && res.iterations <= 2);
    for i in 0..8 {
        assert!((res.weights[i] - w_cf[i]).abs() < 1e-14);
    }
}

#[test]
fn qp_longonly_certified_kkt_and_deterministic() {
    let c = ctx();
    let res = po::min_variance_longonly_solve(&c.sig, None, None, po::QP_MAX_ITER).unwrap();
    assert!(res.converged);
    assert!(res.kkt_residual < 1e-12);
    let w1 = res.weights.clone();
    let w2 = po::min_variance_longonly(&c.sig, None, None).unwrap();
    assert_eq!(w1, w2, "QP must be bit-deterministic");
    assert!(w1.iter().all(|&v| v >= 0.0));
    assert!((w1.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    // KKT for {sum w = 1, w >= 0}: constant gradient on the support, >= off it.
    let g = c.sig.matvec(&w1);
    let lam = (0..8)
        .filter(|&i| w1[i] > 1e-14)
        .map(|i| g[i])
        .fold(f64::INFINITY, f64::min);
    for i in 0..8 {
        if w1[i] > 1e-14 {
            assert!((g[i] - lam).abs() < 1e-13);
        } else {
            assert!(g[i] >= lam - 1e-13);
        }
    }
    // Long-only optimum cannot beat the unconstrained optimum.
    let w_cf = po::min_variance_weights(&c.sig).unwrap();
    let v_lo = dot(&w1, &c.sig.matvec(&w1));
    let v_cf = dot(&w_cf, &c.sig.matvec(&w_cf));
    assert!(v_lo >= v_cf - 1e-14);
}

#[test]
fn qp_box_bounds_respected_and_optimal() {
    // PT-14: lb = 0.02, ub = 0.40 hold exactly; beats projected random points.
    let c = ctx();
    let lb = vec![0.02; 8];
    let ub = vec![0.40; 8];
    let res =
        po::min_variance_longonly_solve(&c.sig, Some(&lb), Some(&ub), po::QP_MAX_ITER).unwrap();
    assert!(res.converged && res.kkt_residual < 1e-12);
    assert!(res.weights.iter().all(|&v| (0.02..=0.40).contains(&v)));
    assert!((res.weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    let var = dot(&res.weights, &c.sig.matvec(&res.weights));
    let mut state = 987654321u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    for _ in 0..20 {
        let v: Vec<f64> = (0..8).map(|_| next()).collect();
        let z = po::project_simplex_box(&v, 1.0, Some(&lb), Some(&ub)).unwrap();
        assert!(var <= dot(&z, &c.sig.matvec(&z)) + 1e-15);
    }
}

#[test]
fn qp_with_active_bounds_still_certified() {
    // Dual feasibility holds at the true optimum: problems with several
    // active bounds stay certified with the same residual bound as before.
    let c = ctx();
    let n = c.sig.nrows();
    for (b0, b1) in [(0.0, f64::INFINITY), (0.02, 0.40), (0.0, 0.25)] {
        let (lb, ub) = (vec![b0; n], vec![b1; n]);
        let res =
            po::min_variance_longonly_solve(&c.sig, Some(&lb), Some(&ub), po::QP_MAX_ITER).unwrap();
        assert!(res.converged, "{b0} {b1}");
        assert!(res.kkt_residual < 1e-12, "{b0} {b1}");
        assert!(res.weights.iter().any(|&v| v == b0 || v == b1), "{b0} {b1}");
    }
    let (lb, ub) = (vec![0.0; n], vec![0.5; n]);
    let fr =
        po::constrained_frontier_solve(&c.mu, &c.sig, 0.07, Some(&lb), Some(&ub), po::QP_MAX_ITER)
            .unwrap();
    assert!(fr.converged && fr.kkt_residual < 1e-9);
}

#[test]
#[allow(clippy::needless_range_loop)] // index-style checks mirror the Python reference formulas
fn constrained_frontier_reports_convergence_and_hits_target() {
    // PT-3: every point certified, target met exactly, variance monotone.
    let c = ctx();
    let res =
        po::constrained_frontier_solve(&c.mu, &c.sig, 0.08, None, None, po::QP_MAX_ITER).unwrap();
    assert!(res.converged && res.kkt_residual < 1e-9);
    assert!(res.weights.iter().all(|&v| v >= 0.0));
    assert!((res.weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    assert!((dot(&c.mu, &res.weights) - 0.08).abs() < 1e-12);
    let sweep = po::constrained_frontier(&c.mu, &c.sig, 25, None, None).unwrap();
    let w_lo = po::min_variance_longonly(&c.sig, None, None).unwrap();
    let mu_max = c.mu.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!((sweep.targets[0] - dot(&c.mu, &w_lo)).abs() < 1e-15);
    assert_eq!(sweep.targets[24], mu_max);
    for k in 0..25 {
        let w = &sweep.weights[k];
        assert!((dot(&c.mu, w) - sweep.targets[k]).abs() < 1e-12);
        assert!(w.iter().all(|&v| v >= 0.0));
        assert!((sweep.variances[k] - dot(w, &c.sig.matvec(w))).abs() < 1e-15);
        if k > 0 {
            assert!(sweep.variances[k] >= sweep.variances[k - 1] - 1e-14);
        }
    }
    for i in 0..8 {
        assert!((sweep.weights[24][i] - if c.mu[i] == mu_max { 1.0 } else { 0.0 }).abs() < 1e-15);
        assert!((sweep.weights[0][i] - w_lo[i]).abs() < 1e-15);
    }
    let a = po::constrained_frontier(&c.mu, &c.sig, 7, None, None).unwrap();
    let b = po::constrained_frontier(&c.mu, &c.sig, 7, None, None).unwrap();
    assert_eq!(a.weights, b.weights, "sweep must be bit-deterministic");
}

#[test]
fn constrained_frontier_rejects_unattainable_targets() {
    let c = ctx();
    let w_lo = po::min_variance_longonly(&c.sig, None, None).unwrap();
    let mu_max = c.mu.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(matches!(
        po::constrained_frontier_weights(&c.mu, &c.sig, dot(&c.mu, &w_lo) - 1e-3, None, None),
        Err(PortOptError::Undefined(_))
    ));
    assert!(matches!(
        po::constrained_frontier_weights(&c.mu, &c.sig, mu_max + 1e-3, None, None),
        Err(PortOptError::Undefined(_))
    ));
    let ub = vec![0.4; 8];
    assert!(po::constrained_frontier_weights(&c.mu, &c.sig, 0.09, None, Some(&ub)).is_err());
    assert!(po::constrained_frontier_weights(&c.mu, &c.sig, f64::INFINITY, None, None).is_err());
    assert!(po::constrained_frontier_solve(&c.mu, &c.sig, 0.06, None, None, 0).is_err());
    assert!(po::constrained_frontier(&c.mu, &c.sig, 1, None, None).is_err());
}

#[test]
fn attainable_return_range_and_degenerate_inputs() {
    let c = ctx();
    let (r_min, r_max) = po::attainable_return_range(&c.mu, None, None).unwrap();
    let mut sorted = c.mu.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    assert_eq!(r_max, sorted[0]);
    assert_eq!(r_min, sorted[7]);
    let lb = vec![0.0; 8];
    let ub = vec![0.4; 8];
    let (rc_min, rc_max) = po::attainable_return_range(&c.mu, Some(&lb), Some(&ub)).unwrap();
    assert!((rc_max - (0.4 * sorted[0] + 0.4 * sorted[1] + 0.2 * sorted[2])).abs() < 1e-15);
    assert!((rc_min - (0.4 * sorted[7] + 0.4 * sorted[6] + 0.2 * sorted[5])).abs() < 1e-15);
    assert!(po::attainable_return_range(&c.mu, Some(&[0.2; 8]), None).is_err());
    // Fixed asset (lb == ub), tied top returns with caps, all-equal returns.
    let mut lb2 = vec![0.0; 8];
    let mut ub2 = vec![f64::INFINITY; 8];
    lb2[5] = 0.30;
    ub2[5] = 0.30;
    let fixed =
        po::min_variance_longonly_solve(&c.sig, Some(&lb2), Some(&ub2), po::QP_MAX_ITER).unwrap();
    assert!(fixed.converged && fixed.weights[5] == 0.30);
    let mut mu2 = c.mu.clone();
    mu2[1] = mu2[0];
    let mu2_max = mu2.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let cap = vec![0.7; 8];
    let tie = po::constrained_frontier_solve(
        &mu2,
        &c.sig,
        mu2_max,
        Some(&lb),
        Some(&cap),
        po::QP_MAX_ITER,
    )
    .unwrap();
    assert!(tie.converged && tie.kkt_residual < 1e-12);
    assert!((tie.weights[0] + tie.weights[1] - 1.0).abs() < 1e-12);
    let flat =
        po::constrained_frontier_solve(&[0.05; 8], &c.sig, 0.05, None, None, po::QP_MAX_ITER)
            .unwrap();
    assert!(flat.converged);
    assert_eq!(
        flat.weights,
        po::min_variance_longonly(&c.sig, None, None).unwrap()
    );
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

/// Direct transcription of Ledoit & Wolf's covCor.m (JPM 2004, App. B):
/// divisor T throughout, both theta terms per ordered pair. Returns (delta, rbar).
#[allow(clippy::needless_range_loop)] // index-style checks mirror the Python reference formulas
fn covcor(x: &Matrix) -> (f64, f64) {
    let (t, n) = (x.nrows(), x.ncols());
    let tf = t as f64;
    let xc: Vec<Vec<f64>> = (0..n)
        .map(|j| {
            let m = (0..t).map(|r| x.get(r, j)).sum::<f64>() / tf;
            (0..t).map(|r| x.get(r, j) - m).collect()
        })
        .collect();
    let sample = |i: usize, j: usize| (0..t).map(|r| xc[i][r] * xc[j][r]).sum::<f64>() / tf;
    let sqrtvar: Vec<f64> = (0..n).map(|i| sample(i, i).sqrt()).collect();
    let mut corr = 0.0;
    for i in 0..n {
        for j in 0..n {
            corr += sample(i, j) / (sqrtvar[i] * sqrtvar[j]);
        }
    }
    let rbar = (corr - n as f64) / (n as f64 * (n - 1) as f64);
    let (mut phi, mut rho, mut gamma) = (0.0, 0.0, 0.0);
    for i in 0..n {
        for j in 0..n {
            let sij = sample(i, j);
            let (mut phi_ij, mut theta_ij) = (0.0, 0.0);
            for r in 0..t {
                let d = xc[i][r] * xc[j][r] - sij;
                phi_ij += d * d;
                theta_ij += (xc[i][r] * xc[i][r] - sample(i, i)) * d;
            }
            phi += phi_ij / tf;
            let prior = if i == j {
                sample(i, i)
            } else {
                rbar * sqrtvar[i] * sqrtvar[j]
            };
            gamma += (sij - prior) * (sij - prior);
            if i == j {
                rho += phi_ij / tf;
            } else {
                rho += rbar * (sqrtvar[j] / sqrtvar[i]) * theta_ij / tf;
            }
        }
    }
    (((phi - rho) / gamma / tf).clamp(0.0, 1.0), rbar)
}

#[test]
fn lw_rho_matches_paper_formula() {
    // PT-1 (CRIT-1): 12x3 hand panel with unequal vols.
    let rows = [
        [0.010, 0.020, -0.010],
        [-0.020, 0.000, 0.010],
        [0.030, -0.010, 0.020],
        [0.005, 0.040, -0.030],
        [-0.015, -0.030, 0.000],
        [0.025, 0.010, 0.015],
        [-0.010, 0.050, -0.020],
        [0.000, -0.020, 0.025],
        [0.020, 0.030, 0.005],
        [-0.030, -0.040, -0.010],
        [0.015, 0.000, 0.030],
        [0.010, 0.060, -0.005],
    ];
    let scale = [1.0, 3.0, 0.5];
    let x = Matrix::from_rows(
        &rows
            .iter()
            .map(|r| (0..3).map(|c| r[c] * scale[c]).collect())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let (delta, rbar) = covcor(&x);
    let res = po::shrink_covariance(&x).unwrap();
    assert!((res.intensity - delta).abs() < 1e-12);
    assert!((res.rbar - rbar).abs() < 1e-12);
    // n = 2: target == sample (gamma_hat = 0) -> delta = 0, sigma == sample.
    let x2 = Matrix::from_rows(&rows.iter().map(|r| vec![r[0], r[1]]).collect::<Vec<_>>()).unwrap();
    let res2 = po::shrink_covariance(&x2).unwrap();
    assert_eq!(res2.intensity, 0.0);
    assert_eq!(res2.sigma, res2.sample);
}

#[test]
fn lw_bundled_data_matches_paper_and_small_t_repairs_singular_sample() {
    let c = ctx();
    let (delta, rbar) = covcor(&c.returns);
    let sh = po::shrink_covariance(&c.returns).unwrap();
    assert!((sh.intensity - delta).abs() < 1e-13);
    assert!((sh.intensity - 0.00721176522548).abs() < 1e-12); // paper value, not 0.009430
    assert!((sh.rbar - rbar).abs() < 1e-13);
    // PT-2: T = 5 <= N = 8, sample singular, shrunk SPD.
    let head = Matrix::from_rows(
        &(0..5)
            .map(|r| c.returns.row(r).to_vec())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let res = po::shrink_covariance(&head).unwrap();
    assert!(res.intensity > 0.0 && res.intensity <= 1.0);
    assert!(matches!(
        po::cholesky(&res.sample),
        Err(PortOptError::NotPositiveDefinite(_))
    ));
    po::cholesky(&res.sigma).expect("shrunk covariance must be SPD");
    assert!((res.intensity - covcor(&head).0).abs() < 1e-12);
}

#[test]
fn shrinkage_zero_variance_rejected() {
    // PT-9: a constant column has zero sample variance.
    let rows: Vec<Vec<f64>> = (0..10)
        .map(|r| vec![-0.01 + 0.02 * r as f64 / 9.0, 0.004])
        .collect();
    let x = Matrix::from_rows(&rows).unwrap();
    assert!(matches!(
        po::shrink_covariance(&x),
        Err(PortOptError::InvalidInput(_))
    ));
}

// ------------------------------------------------------------------ risk parity

#[test]
fn erc_permutation_invariance() {
    // PT-12: erc(P S P') == P erc(S).
    let c = ctx();
    let perm = [5usize, 2, 7, 0, 3, 6, 1, 4];
    let ps = Matrix::from_rows(
        &(0..8)
            .map(|i| (0..8).map(|j| c.sig.get(perm[i], perm[j])).collect())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let w = po::erc_weights(&c.sig).unwrap();
    let wp = po::erc_weights(&ps).unwrap();
    for i in 0..8 {
        assert!((wp[i] - w[perm[i]]).abs() < 1e-10);
    }
    let b = [0.20, 0.20, 0.15, 0.15, 0.10, 0.10, 0.05, 0.05];
    let bp: Vec<f64> = (0..8).map(|i| b[perm[i]]).collect();
    let wb = po::risk_budget_weights(&c.sig, Some(&b)).unwrap();
    let wbp = po::risk_budget_weights(&ps, Some(&bp)).unwrap();
    for i in 0..8 {
        assert!((wbp[i] - wb[perm[i]]).abs() < 1e-10);
    }
}

#[test]
fn erc_scale_invariance_and_cap_reported() {
    // PT-13: daily vs annualized covariance give the same ERC weights.
    let c = ctx();
    let sig_d = po::sample_cov(&c.returns).unwrap();
    let d = po::risk_budget_solve(&sig_d, None, po::CCD_TOL, po::CCD_MAX_SWEEPS).unwrap();
    let a =
        po::risk_budget_solve(&sig_d.scaled(252.0), None, po::CCD_TOL, po::CCD_MAX_SWEEPS).unwrap();
    let k =
        po::risk_budget_solve(&sig_d.scaled(1e-6), None, po::CCD_TOL, po::CCD_MAX_SWEEPS).unwrap();
    assert!(d.converged && a.converged && k.converged);
    for i in 0..8 {
        assert!((d.weights[i] - a.weights[i]).abs() < 1e-10);
        assert!((d.weights[i] - k.weights[i]).abs() < 1e-10);
    }
    let capped = po::risk_budget_solve(&c.sig, None, po::CCD_TOL, 1).unwrap();
    assert!(!capped.converged && capped.sweeps == 1);
    assert!((capped.weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
}

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
#[allow(clippy::needless_range_loop)] // index-style checks mirror the Python reference formulas
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
#[allow(clippy::needless_range_loop)] // index-style checks mirror the Python reference formulas
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
fn relative_view_same_legs_and_parse_errors() {
    // PT-7 (MAJ-2) and PT-11.
    let c = ctx();
    let same = vec![po::View::Relative {
        long: "US_EQ".into(),
        short: "US_EQ".into(),
        q: 0.02,
        confidence: 1.0,
    }];
    assert!(matches!(
        po::views_to_matrices(&same, &c.names),
        Err(PortOptError::InvalidInput(_))
    ));
    let unknown = vec![po::View::Absolute {
        asset: "NOPE".into(),
        q: 0.1,
        confidence: 1.0,
    }];
    assert!(po::views_to_matrices(&unknown, &c.names).is_err());
    let nan_q = vec![po::View::Absolute {
        asset: "US_EQ".into(),
        q: f64::NAN,
        confidence: 1.0,
    }];
    assert!(po::views_to_matrices(&nan_q, &c.names).is_err());
    let mut dup = c.names.clone();
    dup.push("US_EQ".into());
    let ok = vec![po::View::Absolute {
        asset: "US_EQ".into(),
        q: 0.1,
        confidence: 1.0,
    }];
    assert!(po::views_to_matrices(&ok, &dup).is_err());
    assert!(po::views_to_matrices(&ok, &[]).is_err());
    let zero_conf = vec![po::View::Absolute {
        asset: "US_EQ".into(),
        q: 0.1,
        confidence: 0.0,
    }];
    assert!(po::views_to_matrices(&zero_conf, &c.names).is_err());
}

#[test]
fn omega_and_confidences_together_rejected_and_leverage() {
    // MIN-12: explicit Omega plus confidences would silently ignore the latter.
    let c = ctx();
    let pi = po::implied_equilibrium_returns(&c.sig, &c.w_mkt, c.views.delta).unwrap();
    let (p, q, conf) = po::views_to_matrices(&c.views.views, &c.names).unwrap();
    let omega = po::default_omega(&c.sig, &p, c.views.tau, Some(&conf)).unwrap();
    assert!(matches!(
        po::bl_posterior(&c.sig, &pi, &p, &q, c.views.tau, Some(&omega), Some(&conf)),
        Err(PortOptError::InvalidInput(_))
    ));
    assert!(po::bl_weights(
        &c.sig,
        &pi,
        &p,
        &q,
        c.views.tau,
        0.0,
        Some(&omega),
        Some(&conf)
    )
    .is_err());
    assert!(po::bl_weights(
        &c.sig,
        &pi,
        &p,
        &q,
        c.views.tau,
        f64::NAN,
        None,
        Some(&conf)
    )
    .is_err());
    let a = po::bl_posterior(&c.sig, &pi, &p, &q, c.views.tau, Some(&omega), None).unwrap();
    let b = po::bl_posterior(&c.sig, &pi, &p, &q, c.views.tau, None, Some(&conf)).unwrap();
    for i in 0..8 {
        assert!((a.mu_post[i] - b.mu_post[i]).abs() < 1e-16);
    }
    let (w, _) = po::bl_weights(&c.sig, &pi, &p, &q, c.views.tau, 0.0, None, Some(&conf)).unwrap();
    assert!(po::gross_leverage(&w).unwrap() > 1.0); // posterior shorts EU_EQ
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
    assert_eq!(
        r.max_drawdown,
        po::max_drawdown(&r.portfolio_returns).unwrap()
    );
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
        &(0..100)
            .map(|t| c.returns.row(t).to_vec())
            .collect::<Vec<_>>(),
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
fn metrics_reject_bad_rf_periods_and_total_loss() {
    // PT-15 / MIN-13 / MAJ-5.
    let r = [0.01, -0.02, 0.015, 0.0, 0.005];
    assert!(po::sharpe_ratio(&r, f64::NAN, 252.0).is_err());
    assert!(po::sharpe_ratio(&r, f64::INFINITY, 252.0).is_err());
    for bad in [-1.0, 0.0, f64::NAN, f64::INFINITY] {
        assert!(po::annualized_vol(&r, bad).is_err());
        assert!(po::annualized_return(&r, bad).is_err());
    }
    assert!(po::max_drawdown(&[-1.5]).is_err());
    assert!(po::max_drawdown(&[0.1, -1.0, 0.2]).is_err());
    assert!(po::annualized_return(&[0.1, -1.2], 252.0).is_err());
    assert!((po::max_drawdown(&[-0.999]).unwrap() - 0.999).abs() < 1e-15);
}

#[test]
fn backtest_minimal_sample() {
    // PT-10: T = window + 2 is the smallest valid backtest; T = window + 1 errors.
    let c = ctx();
    let take = |t: usize| {
        Matrix::from_rows(
            &(0..t)
                .map(|r| c.returns.row(r).to_vec())
                .collect::<Vec<_>>(),
        )
        .unwrap()
    };
    let eq = |h: &Matrix| Ok(vec![1.0 / h.ncols() as f64; h.ncols()]);
    let r = po::run_backtest(&take(502), eq, 0.02, 500, 63).unwrap();
    assert_eq!(r.rebalance_dates, vec![500]);
    assert_eq!(r.portfolio_returns.len(), 2);
    assert!(r.turnovers.is_empty() && r.avg_turnover == 0.0);
    assert!(r.sharpe.is_finite() && r.ann_vol.is_finite());
    assert_eq!(po::rebalance_indices(501, 500, 63).unwrap(), vec![500]);
    assert!(matches!(
        po::run_backtest(&take(501), eq, 0.02, 500, 63),
        Err(PortOptError::InvalidInput(_))
    ));
    assert!(po::run_backtest(&take(502), eq, f64::NAN, 500, 63).is_err());
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
#[allow(clippy::needless_range_loop)] // index-style checks mirror the Python reference formulas
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
