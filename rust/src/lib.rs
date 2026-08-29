//! `portopt` — portfolio construction (P10 Rust port).
//!
//! Markowitz mean-variance (closed forms + constrained PGD), Ledoit-Wolf
//! constant-correlation shrinkage, Black-Litterman, risk parity / risk
//! budgeting via CCD, and a quarterly expanding-window backtest. Every
//! numerical algorithm is pinned in the project's `API_SPEC.md` so that
//! this scipy-free port reproduces the Python reference and the golden
//! values in `data/golden/golden.json`.
//!
//! Determinism: no RNG anywhere in the library; repeated calls are
//! bit-identical. Bad input is reported via [`PortOptError`] — the
//! library never panics on invalid data.

#![warn(missing_docs)]

pub mod backtest;
pub mod black_litterman;
pub mod data;
pub mod error;
pub mod estimators;
pub mod linalg;
pub mod markowitz;
pub mod risk_parity;

pub use backtest::{
    annualized_return, annualized_vol, max_drawdown, rebalance_indices, run_backtest,
    sharpe_ratio, BacktestResult, ANNUALIZATION, DEFAULT_STEP, DEFAULT_WINDOW,
};
pub use black_litterman::{
    bl_posterior, bl_weights, default_omega, implied_equilibrium_returns, views_to_matrices,
    BlResult, View,
};
pub use data::{load_config, load_market_caps, load_returns_csv, load_views, Config, ViewsDoc};
pub use error::{PortOptError, Result};
pub use estimators::{
    annualize_cov, annualize_mean, sample_cov, sample_mean, shrink_covariance, validate_returns,
    ShrinkageResult,
};
pub use linalg::{
    check_square_symmetric, chol_solve, cholesky, spd_inverse, spd_solve, Matrix, PIVOT_RTOL,
};
pub use markowitz::{
    constrained_frontier, constrained_frontier_weights, efficient_frontier, frontier_weights,
    min_variance_longonly, min_variance_weights, pgd_minimize, project_simplex_box,
    tangency_weights, two_fund_scalars, FrontierSweep, PgdResult, TwoFundScalars,
    FRONTIER_PENALTY, PGD_MAX_ITER, PGD_TOL,
};
pub use risk_parity::{
    erc_weights, inverse_vol_weights, risk_budget_solve, risk_budget_weights,
    risk_contributions, CcdResult, CCD_MAX_SWEEPS, CCD_TOL,
};
