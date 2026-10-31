//! `portopt` end-to-end demo.
//!
//! Run:  `cd rust && cargo run --bin demo`
//!
//! Prints: estimator table, unconstrained + long-only frontier sweep,
//! Black-Litterman prior vs posterior weights, ERC vs inverse-vol risk
//! contributions, cost-aware (turnover-penalised) long-only min variance,
//! and a 63-day re-estimation backtest across seven methods (mirrors
//! `python/demo.py`).

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use portopt as po;
use portopt::{Matrix, PortOptError, View};

/// Scientific notation with a sign and at least two exponent digits
/// (`1.20e-01`), matching the Python, C++ and Java demos.
fn sci(x: f64, prec: usize) -> String {
    let s = format!("{x:.prec$e}");
    match s.split_once('e') {
        Some((mant, exp)) => {
            let e: i32 = exp.parse().unwrap_or(0);
            let sign = if e < 0 { '-' } else { '+' };
            format!("{mant}e{sign}{:02}", e.abs())
        }
        None => s,
    }
}

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("data")
}

fn rule(title: &str) {
    let pad = 66usize.saturating_sub(title.len());
    println!("\n=== {} {}", title, "=".repeat(pad));
}

fn wrow(label: &str, w: &[f64]) -> String {
    let mut s = format!("{label:<14}");
    for x in w {
        s.push_str(&format!("{x:>9.4}"));
    }
    s
}

fn run() -> po::Result<()> {
    let dir = data_dir();
    let (names, x) = po::load_returns_csv(&dir.join("asset_returns.csv"))?;
    let cfg = po::load_config(&dir.join("config.json"))?;
    let views_doc = po::load_views(&dir.join("views.json"))?;
    let (_, caps) = po::load_market_caps(&dir.join("market_caps.csv"))?;
    let rf = cfg.risk_free_rate;
    let ann = cfg.annualization_factor;

    println!("portopt demo: 8-asset universe, 1250 daily returns");
    println!("assets: {}", names.join(", "));

    // ------------------------------------------------------------ estimators
    rule("Estimators (annualized, Ledoit-Wolf constant-correlation shrinkage)");
    let mu = po::annualize_mean(&po::sample_mean(&x)?, ann)?;
    let shrink = po::shrink_covariance(&x)?;
    let sig = po::annualize_cov(&po::sample_cov(&x)?, ann)?;
    let sig_sh = po::annualize_cov(&shrink.sigma, ann)?;
    println!(
        "shrinkage intensity delta = {:.6}   rbar = {:.4}",
        shrink.intensity, shrink.rbar
    );
    println!(
        "{:<10}{:>9}{:>13}{:>13}",
        "asset", "mean", "vol(sample)", "vol(shrunk)"
    );
    for (i, name) in names.iter().enumerate() {
        println!(
            "{:<10}{:>9.4}{:>13.4}{:>13.4}",
            name,
            mu[i],
            sig.get(i, i).sqrt(),
            sig_sh.get(i, i).sqrt()
        );
    }

    // -------------------------------------------------------------- frontier
    rule("Efficient frontier sweeps (25 points each, showing every 4th)");
    let sweep_u = po::efficient_frontier(&mu, &sig, cfg.frontier_points)?;
    let sweep_c = po::constrained_frontier(&mu, &sig, cfg.frontier_points, None, None)?;
    println!("unconstrained grid: A/C .. max(mu);  long-only grid: constrained GMV .. max(mu)");
    println!(
        "{:>8}{:>11}{:>10}{:>15}",
        "target", "vol uncon", "target", "vol long-only"
    );
    for k in (0..sweep_u.targets.len()).step_by(4) {
        println!(
            "{:>8.4}{:>11.4}{:>10.4}{:>15.4}",
            sweep_u.targets[k],
            sweep_u.variances[k].sqrt(),
            sweep_c.targets[k],
            sweep_c.variances[k].sqrt()
        );
    }
    let gap = (0..sweep_c.targets.len())
        .map(|k| (po::linalg::dot(&mu, &sweep_c.weights[k]) - sweep_c.targets[k]).abs())
        .fold(0.0_f64, f64::max);
    println!(
        "long-only targets met exactly: max |mu'w - target| = {}",
        sci(gap, 1)
    );
    let w_tan = po::tangency_weights(&mu, &sig, rf)?;
    let tan_ret = po::linalg::dot(&mu, &w_tan);
    let tan_vol = po::linalg::dot(&w_tan, &sig.matvec(&w_tan)).sqrt();
    println!(
        "tangency (rf={:.2}%): sharpe = {:.4}, gross leverage = {:.2}",
        100.0 * rf,
        (tan_ret - rf) / tan_vol,
        po::gross_leverage(&w_tan)?
    );

    // ------------------------------------------------------- black-litterman
    rule("Black-Litterman: prior (equilibrium) vs posterior weights");
    let cap_sum: f64 = caps.iter().sum();
    let w_mkt: Vec<f64> = caps.iter().map(|c| c / cap_sum).collect();
    let pi = po::implied_equilibrium_returns(&sig, &w_mkt, views_doc.delta)?;
    let (p, q, conf) = po::views_to_matrices(&views_doc.views, &names)?;
    for v in &views_doc.views {
        match v {
            View::Absolute {
                asset,
                q,
                confidence,
            } => {
                println!(
                    "view: {asset} returns {:.1}% (confidence {confidence:?})",
                    100.0 * q
                );
            }
            View::Relative {
                long,
                short,
                q,
                confidence,
            } => {
                println!(
                    "view: {long} beats {short} by {:.1}% (confidence {confidence:?})",
                    100.0 * q
                );
            }
        }
    }
    let (w_bl, res) = po::bl_weights(&sig, &pi, &p, &q, views_doc.tau, 0.0, None, Some(&conf))?;
    let mut header = format!("{:<14}", "");
    for n in &names {
        let short: String = n.chars().take(8).collect();
        header.push_str(&format!("{short:>9}"));
    }
    println!("{header}");
    println!("{}", wrow("pi (equil.)", &pi));
    println!("{}", wrow("mu posterior", &res.mu_post));
    println!("{}", wrow("w market", &w_mkt));
    println!("{}", wrow("w posterior", &w_bl));
    println!(
        "posterior gross leverage = {:.4}",
        po::gross_leverage(&w_bl)?
    );

    // ----------------------------------------------------------- risk parity
    rule("Risk contributions: ERC vs inverse-vol");
    let w_erc = po::erc_weights(&sig)?;
    let w_iv = po::inverse_vol_weights(&sig)?;
    let rc_erc = po::risk_contributions(&w_erc, &sig)?;
    let rc_iv = po::risk_contributions(&w_iv, &sig)?;
    println!("{header}");
    println!("{}", wrow("w ERC", &w_erc));
    println!("{}", wrow("RC ERC", &rc_erc));
    println!("{}", wrow("w inv-vol", &w_iv));
    println!("{}", wrow("RC inv-vol", &rc_iv));
    let spread = |rc: &[f64]| {
        rc.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - rc.iter().cloned().fold(f64::INFINITY, f64::min)
    };
    println!(
        "RC spread: ERC {}, inv-vol {}",
        sci(spread(&rc_erc), 2),
        sci(spread(&rc_iv), 2)
    );

    // ------------------------------------------------------------ cost-aware
    rule("Cost-aware rebalancing: long-only min variance from market weights");
    println!(
        "objective 1/2 w'Sw + kappa*sum|w - w0|, w0 = market weights, kappa in variance units"
    );
    println!(
        "{:>10}{:>10}{:>9}{:>11}",
        "kappa", "turnover", "vol", "certified"
    );
    let inf = f64::INFINITY;
    for kappa in [0.0, 3e-5, 1e-4, 3e-4, 1e-3] {
        let r_tc =
            po::min_variance_turnover_solve(&sig, &w_mkt, kappa, None, None, inf, po::QP_MAX_ITER)?;
        let vol_tc = po::linalg::dot(&r_tc.weights, &sig.matvec(&r_tc.weights)).sqrt();
        let ok = if r_tc.converged && r_tc.kkt_residual < 1e-12 {
            "yes"
        } else {
            "NO"
        };
        println!(
            "{:>10.6}{:>10.4}{:>9.4}{:>11}",
            kappa, r_tc.turnover, vol_tc, ok
        );
    }
    let r_cap =
        po::min_variance_turnover_solve(&sig, &w_mkt, 0.0, None, None, 0.25, po::QP_MAX_ITER)?;
    println!(
        "kappa 0 with turnover cap 0.25: turnover {:.4}, vol {:.4}",
        r_cap.turnover,
        po::linalg::dot(&r_cap.weights, &sig.matvec(&r_cap.weights)).sqrt()
    );
    println!("{header}");
    println!("{}", wrow("w market", &w_mkt));
    println!(
        "{}",
        wrow(
            "w kappa 1e-4",
            &po::min_variance_turnover(&sig, &w_mkt, 1e-4, None, None, inf)?
        )
    );
    println!("{}", wrow("w cap 0.25", &r_cap.weights));

    // -------------------------------------------------------------- backtest
    rule("Backtest: re-estimated every 63 days, daily constant mix (holdout = last 750d)");
    let mut fallback_undefined = 0usize;
    let mut fallback_leverage = 0usize;
    let mut prev_tc: Option<Vec<f64>> = None;
    let window = cfg.estimation_window;
    let step = cfg.rebalance_step;

    type WeightFn<'a> = Box<dyn FnMut(&Matrix) -> po::Result<Vec<f64>> + 'a>;
    let methods: Vec<(&str, WeightFn)> = vec![
        (
            "equal_weight",
            Box::new(|h: &Matrix| Ok(vec![1.0 / h.ncols() as f64; h.ncols()])),
        ),
        (
            "min_var_lo",
            Box::new(move |h: &Matrix| {
                let s = po::annualize_cov(&po::sample_cov(h)?, ann)?;
                po::min_variance_longonly(&s, None, None)
            }),
        ),
        (
            "min_var_tc",
            Box::new(|h: &Matrix| {
                // Long-only min variance with turnover penalty kappa = 3e-5
                // against the previous target (equal weight before the first
                // rebalance).
                let s = po::annualize_cov(&po::sample_cov(h)?, ann)?;
                let w0 = prev_tc
                    .clone()
                    .unwrap_or_else(|| vec![1.0 / h.ncols() as f64; h.ncols()]);
                let w = po::min_variance_turnover(&s, &w0, 3e-5, None, None, f64::INFINITY)?;
                prev_tc = Some(w.clone());
                Ok(w)
            }),
        ),
        (
            "tangency",
            Box::new(|h: &Matrix| {
                // Pinned backtest method (API_SPEC 6.3): tangency, falling
                // back to unconstrained min-variance for a rebalance where
                // tangency is undefined (rf at or above A/C) or above the
                // gross-leverage cap, reported, not crashed. Any other error
                // (e.g. singular covariance) propagates.
                let s = po::annualize_cov(&po::sample_cov(h)?, ann)?;
                let m = po::annualize_mean(&po::sample_mean(h)?, ann)?;
                match po::tangency_weights(&m, &s, rf) {
                    Ok(w) => Ok(w),
                    Err(PortOptError::TangencyUndefined(_)) => {
                        let sc = po::two_fund_scalars(&m, &s)?;
                        if sc.a - rf * sc.c <= 0.0 {
                            fallback_undefined += 1;
                        } else {
                            fallback_leverage += 1;
                        }
                        po::min_variance_weights(&s)
                    }
                    Err(e) => Err(e),
                }
            }),
        ),
        (
            "black_litt",
            Box::new(|h: &Matrix| {
                let s = po::annualize_cov(&po::sample_cov(h)?, ann)?;
                let pi_h = po::implied_equilibrium_returns(&s, &w_mkt, views_doc.delta)?;
                let (w, _) =
                    po::bl_weights(&s, &pi_h, &p, &q, views_doc.tau, 0.0, None, Some(&conf))?;
                Ok(w)
            }),
        ),
        (
            "erc",
            Box::new(|h: &Matrix| po::erc_weights(&po::sample_cov(h)?)),
        ),
        (
            "inverse_vol",
            Box::new(|h: &Matrix| po::inverse_vol_weights(&po::sample_cov(h)?)),
        ),
    ];

    println!(
        "{:<14}{:>9}{:>9}{:>9}{:>9}{:>11}",
        "method", "ann ret", "ann vol", "sharpe", "max DD", "avg turno"
    );
    for (name, f) in methods {
        let r = po::run_backtest(&x, f, rf, window, step)?;
        println!(
            "{:<14}{:>9.4}{:>9.4}{:>9.4}{:>9.4}{:>11.4}",
            name, r.ann_return, r.ann_vol, r.sharpe, r.max_drawdown, r.avg_turnover
        );
    }
    let n_fb = fallback_undefined + fallback_leverage;
    if n_fb > 0 {
        println!(
            "note: tangency fell back to min-variance at {n_fb} rebalance date(s): \
             {fallback_undefined} with rf at or above the min-var return A/C, \
             {fallback_leverage} above the {}x gross-leverage cap.",
            po::TANGENCY_MAX_LEVERAGE
        );
    }
    println!(
        "min_var_tc = min_var_lo with turnover penalty kappa = 3e-5 against the previous target."
    );

    println!("\ndemo complete.");
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("demo failed: {e}");
            ExitCode::FAILURE
        }
    }
}
