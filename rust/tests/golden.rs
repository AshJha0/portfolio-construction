//! Golden-value suite: loads `../data/golden/golden.json` (relative to
//! the rust/ directory) and asserts every case within its tolerance.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use portopt as po;
use portopt::linalg::dot;
use portopt::Matrix;

#[derive(Debug, Deserialize)]
struct GoldenFile {
    cases: Vec<GoldenCase>,
}

#[derive(Debug, Deserialize)]
struct GoldenCase {
    name: String,
    #[serde(default)]
    inputs: HashMap<String, serde_json::Value>,
    expect: HashMap<String, f64>,
    tol: f64,
}

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("data")
}

struct Ctx {
    names: Vec<String>,
    returns: Matrix,
    mu_d: Vec<f64>,
    cov_d: Matrix,
    mu: Vec<f64>,
    sig: Matrix,
    caps: Vec<f64>,
    views: po::ViewsDoc,
}

fn ctx() -> Ctx {
    let dir = data_dir();
    let (names, returns) = po::load_returns_csv(&dir.join("asset_returns.csv")).unwrap();
    let (_, caps) = po::load_market_caps(&dir.join("market_caps.csv")).unwrap();
    let views = po::load_views(&dir.join("views.json")).unwrap();
    let mu_d = po::sample_mean(&returns).unwrap();
    let cov_d = po::sample_cov(&returns).unwrap();
    let mu = po::annualize_mean(&mu_d, 252.0).unwrap();
    let sig = po::annualize_cov(&cov_d, 252.0).unwrap();
    Ctx { names, returns, mu_d, cov_d, mu, sig, caps, views }
}

fn golden() -> HashMap<String, GoldenCase> {
    let text = std::fs::read_to_string(data_dir().join("golden").join("golden.json")).unwrap();
    let file: GoldenFile = serde_json::from_str(&text).unwrap();
    file.cases.into_iter().map(|c| (c.name.clone(), c)).collect()
}

fn check(case: &GoldenCase, key: &str, got: f64) {
    let expect = *case
        .expect
        .get(key)
        .unwrap_or_else(|| panic!("case {}: missing expect key {key}", case.name));
    assert!(
        (got - expect).abs() <= case.tol,
        "case {} key {key}: got {got:.17e}, expected {expect:.17e} (tol {})",
        case.name,
        case.tol
    );
}

fn check_vec(case: &GoldenCase, prefix: &str, got: &[f64]) {
    for (i, &g) in got.iter().enumerate() {
        check(case, &format!("{prefix}_{i}"), g);
    }
}

fn input_f64(case: &GoldenCase, key: &str) -> f64 {
    case.inputs[key].as_f64().unwrap()
}

#[test]
fn golden_all_cases() {
    let g = golden();
    assert_eq!(g.len(), 21, "expected 21 golden cases");
    let c = ctx();
    let mut covered = 0usize;

    // ------------------------------------------------------ sample_moments_daily
    {
        let case = &g["sample_moments_daily"];
        check(case, "mean_0", c.mu_d[0]);
        check(case, "mean_4", c.mu_d[4]);
        check(case, "mean_7", c.mu_d[7]);
        check(case, "cov_0_0", c.cov_d.get(0, 0));
        check(case, "cov_0_1", c.cov_d.get(0, 1));
        check(case, "cov_4_4", c.cov_d.get(4, 4));
        check(case, "cov_6_7", c.cov_d.get(6, 7));
        covered += 1;
    }

    // ------------------------------------------------- shrinkage_const_corr_daily
    {
        let case = &g["shrinkage_const_corr_daily"];
        let sh = po::shrink_covariance(&c.returns).unwrap();
        check(case, "intensity", sh.intensity);
        check(case, "rbar", sh.rbar);
        check(case, "shrunk_0_0", sh.sigma.get(0, 0));
        check(case, "shrunk_0_1", sh.sigma.get(0, 1));
        check(case, "shrunk_2_5", sh.sigma.get(2, 5));
        check(case, "shrunk_7_7", sh.sigma.get(7, 7));
        covered += 1;
    }

    // -------------------------------------------------- two_fund_scalars_annualized
    {
        let case = &g["two_fund_scalars_annualized"];
        let sc = po::two_fund_scalars(&c.mu, &c.sig).unwrap();
        check(case, "a", sc.a);
        check(case, "b", sc.b);
        check(case, "c", sc.c);
        check(case, "d", sc.d);
        covered += 1;
    }

    // ----------------------------------------------------- minvar_unconstrained
    {
        let case = &g["minvar_unconstrained"];
        let w = po::min_variance_weights(&c.sig).unwrap();
        check_vec(case, "w", &w);
        check(case, "ret", dot(&c.mu, &w));
        check(case, "vol", dot(&w, &c.sig.matvec(&w)).sqrt());
        covered += 1;
    }

    // ---------------------------------------------------- tangency_unconstrained
    {
        let case = &g["tangency_unconstrained"];
        let rf = input_f64(case, "rf");
        let w = po::tangency_weights(&c.mu, &c.sig, rf).unwrap();
        check_vec(case, "w", &w);
        let sharpe = (dot(&c.mu, &w) - rf) / dot(&w, &c.sig.matvec(&w)).sqrt();
        check(case, "sharpe", sharpe);
        covered += 1;
    }

    // ------------------------------------------------------- frontier_point_t080
    {
        let case = &g["frontier_point_t080"];
        let target = input_f64(case, "target");
        let w = po::frontier_weights(&c.mu, &c.sig, target).unwrap();
        check_vec(case, "w", &w);
        check(case, "variance", dot(&w, &c.sig.matvec(&w)));
        covered += 1;
    }

    // ------------------------------------------------------- minvar_longonly_pgd
    {
        let case = &g["minvar_longonly_pgd"];
        let w = po::min_variance_longonly(&c.sig, None, None).unwrap();
        check_vec(case, "w", &w);
        covered += 1;
    }

    // -------------------------------------------------- frontier_constrained_t080
    {
        let case = &g["frontier_constrained_t080"];
        let target = input_f64(case, "target");
        let penalty = input_f64(case, "penalty");
        let w =
            po::constrained_frontier_weights(&c.mu, &c.sig, target, penalty, None, None).unwrap();
        check_vec(case, "w", &w);
        covered += 1;
    }

    // ------------------------------------------------------ cholesky_annualized_cov
    {
        let case = &g["cholesky_annualized_cov"];
        let low = po::cholesky(&c.sig).unwrap();
        check(case, "l_0_0", low.get(0, 0));
        check(case, "l_7_0", low.get(7, 0));
        check(case, "l_7_7", low.get(7, 7));
        covered += 1;
    }

    // ----------------------------------------------------- simplex_projection_hand
    {
        let case = &g["simplex_projection_hand"];
        let v = [
            input_f64(case, "v_0"),
            input_f64(case, "v_1"),
            input_f64(case, "v_2"),
        ];
        let total = input_f64(case, "total");
        let w = po::project_simplex_box(&v, total, None, None).unwrap();
        check_vec(case, "w", &w);
        covered += 1;
    }

    // ------------------------------------------------------ simplex_projection_box
    {
        let case = &g["simplex_projection_box"];
        let v = [
            input_f64(case, "v_0"),
            input_f64(case, "v_1"),
            input_f64(case, "v_2"),
            input_f64(case, "v_3"),
        ];
        let total = input_f64(case, "total");
        let lb = vec![input_f64(case, "lb"); 4];
        let ub = vec![input_f64(case, "ub"); 4];
        let w = po::project_simplex_box(&v, total, Some(&lb), Some(&ub)).unwrap();
        check_vec(case, "w", &w);
        covered += 1;
    }

    // --------------------------------------------------------------- implied_pi
    let w_mkt: Vec<f64> = {
        let cap_sum: f64 = c.caps.iter().sum();
        c.caps.iter().map(|v| v / cap_sum).collect()
    };
    {
        let case = &g["implied_pi"];
        let delta = input_f64(case, "delta");
        let pi = po::implied_equilibrium_returns(&c.sig, &w_mkt, delta).unwrap();
        check_vec(case, "pi", &pi);
        covered += 1;
    }

    // ---------------------------------------------------------- bl_omega_default
    let (p, q, conf) = po::views_to_matrices(&c.views.views, &c.names).unwrap();
    {
        let case = &g["bl_omega_default"];
        let tau = input_f64(case, "tau");
        assert_eq!(input_f64(case, "conf_0"), conf[0]);
        assert_eq!(input_f64(case, "conf_1"), conf[1]);
        let omega = po::default_omega(&c.sig, &p, tau, Some(&conf)).unwrap();
        check(case, "omega_0_0", omega.get(0, 0));
        check(case, "omega_1_1", omega.get(1, 1));
        covered += 1;
    }

    // ---------------------------------------- bl_posterior_returns / bl_posterior_weights
    {
        let case = &g["bl_posterior_returns"];
        let tau = input_f64(case, "tau");
        let pi = po::implied_equilibrium_returns(&c.sig, &w_mkt, c.views.delta).unwrap();
        let res = po::bl_posterior(&c.sig, &pi, &p, &q, tau, None, Some(&conf)).unwrap();
        check_vec(case, "mu", &res.mu_post);
        covered += 1;

        let case_w = &g["bl_posterior_weights"];
        let rf = input_f64(case_w, "rf");
        let w = po::tangency_weights(&res.mu_post, &res.sigma_post, rf).unwrap();
        check_vec(case_w, "w", &w);
        covered += 1;
    }

    // -------------------------------------------------------------- inverse_vol
    {
        let case = &g["inverse_vol"];
        let w = po::inverse_vol_weights(&c.sig).unwrap();
        check_vec(case, "w", &w);
        covered += 1;
    }

    // -------------------------------------------------------------- erc_weights
    {
        let case = &g["erc_weights"];
        let res = po::risk_budget_solve(&c.sig, None, po::CCD_TOL, po::CCD_MAX_SWEEPS).unwrap();
        assert!(res.converged, "ERC CCD did not converge");
        check_vec(case, "w", &res.weights);
        let rc = po::risk_contributions(&res.weights, &c.sig).unwrap();
        let spread = rc.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - rc.iter().cloned().fold(f64::INFINITY, f64::min);
        check(case, "rc_spread", spread);
        covered += 1;
    }

    // -------------------------------------------------------- risk_budget_pinned
    {
        let case = &g["risk_budget_pinned"];
        let budgets: Vec<f64> = (0..8).map(|i| input_f64(case, &format!("b_{i}"))).collect();
        let w = po::risk_budget_weights(&c.sig, Some(&budgets)).unwrap();
        check_vec(case, "w", &w);
        covered += 1;
    }

    // ------------------------------------------------------ backtest_equal_weight
    {
        let case = &g["backtest_equal_weight"];
        let rf = input_f64(case, "rf");
        let window = input_f64(case, "window") as usize;
        let step = input_f64(case, "step") as usize;
        let r = po::run_backtest(
            &c.returns,
            |h: &Matrix| Ok(vec![1.0 / h.ncols() as f64; h.ncols()]),
            rf,
            window,
            step,
        )
        .unwrap();
        check(case, "ann_return", r.ann_return);
        check(case, "ann_vol", r.ann_vol);
        check(case, "sharpe", r.sharpe);
        check(case, "max_drawdown", r.max_drawdown);
        covered += 1;
    }

    // -------------------------------------------------------------- backtest_erc
    {
        let case = &g["backtest_erc"];
        let rf = input_f64(case, "rf");
        let window = input_f64(case, "window") as usize;
        let step = input_f64(case, "step") as usize;
        let r = po::run_backtest(
            &c.returns,
            |h: &Matrix| po::erc_weights(&po::sample_cov(h)?),
            rf,
            window,
            step,
        )
        .unwrap();
        check(case, "sharpe", r.sharpe);
        check(case, "max_drawdown", r.max_drawdown);
        covered += 1;
    }

    // ---------------------------------------------------- turnover_minvar_longonly
    {
        let case = &g["turnover_minvar_longonly"];
        let window = input_f64(case, "window") as usize;
        let step = input_f64(case, "step") as usize;
        let r = po::run_backtest(
            &c.returns,
            |h: &Matrix| {
                let s = po::annualize_cov(&po::sample_cov(h)?, 252.0)?;
                po::min_variance_longonly(&s, None, None)
            },
            0.0,
            window,
            step,
        )
        .unwrap();
        check(case, "avg_turnover", r.avg_turnover);
        check(case, "n_rebalances", r.rebalance_dates.len() as f64);
        covered += 1;
    }

    assert_eq!(covered, 21, "every golden case must be asserted");
}
