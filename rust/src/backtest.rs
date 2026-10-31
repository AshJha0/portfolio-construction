//! Backtest-lite: quarterly expanding-window rebalance loop and metrics.
//!
//! Pinned schedule (API_SPEC.md section 6): with `T` rows of daily
//! returns, estimation window `W` (default 500) and rebalance step of 63
//! trading days, rebalance dates are the row indices `t_k = W + 63 k`
//! for `k = 0, 1, ...` while `t_k < T`. At each date the weight function
//! sees the expanding history rows `[0, t_k)` and the resulting weights
//! apply to rows `[t_k, min(t_{k+1}, T))`. Within a holding block the
//! portfolio is rebalanced back to target daily, so the daily portfolio
//! return is exactly `w . r_t`.
//!
//! Turnover at rebalance `k >= 1` (pinned): `sum_i |w_k[i] - w_{k-1}[i]|`
//! (two-sided L1 between consecutive *target* vectors; nothing charged
//! for the initial position).
//!
//! Metric formulas (pinned):
//!
//! * `ann_return = 252 * mean(r)` (arithmetic)
//! * `ann_vol    = sqrt(252) * std(r, ddof=1)`
//! * `sharpe     = (ann_return - rf) / ann_vol` (error if vol ~ 0)
//! * max drawdown from wealth `V_0 = 1`, `V_t = V_{t-1} (1 + r_t)`:
//!   `maxdd = max_t (1 - V_t / max_{s<=t} V_s)` (peak includes `V_0`),
//!   reported as a positive fraction.
//!
//! Every metric rejects NaN/inf, simple returns `<= -1`, non-positive
//! `periods` and a non-finite `rf`. Series sums use the compensated
//! [`crate::linalg::kahan_sum`] (matches the reference's `math.fsum`).

use crate::error::{PortOptError, Result};
use crate::estimators::validate_returns;
use crate::linalg::{kahan_sum, Matrix};

/// Default expanding estimation window (trading days).
pub const DEFAULT_WINDOW: usize = 500;
/// Default rebalance step (trading days, ~quarterly).
pub const DEFAULT_STEP: usize = 63;
/// Trading days per year for annualization.
pub const ANNUALIZATION: f64 = 252.0;

fn validate_series(r: &[f64], min_len: usize) -> Result<()> {
    if r.len() < min_len {
        return Err(PortOptError::InvalidInput(format!(
            "return series must have at least {min_len} observations"
        )));
    }
    if !r.iter().all(|v| v.is_finite()) {
        return Err(PortOptError::InvalidInput(
            "return series contains NaN or infinite entries".into(),
        ));
    }
    if r.iter().any(|&v| v <= -1.0) {
        return Err(PortOptError::InvalidInput(
            "return series contains a simple return <= -1 (-100% or worse)".into(),
        ));
    }
    Ok(())
}

#[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(x > 0.0)` is deliberate: it rejects NaN as well as non-positive values
fn validate_periods(periods: f64) -> Result<()> {
    if !(periods > 0.0) || !periods.is_finite() {
        return Err(PortOptError::InvalidInput(
            "periods must be positive and finite".into(),
        ));
    }
    Ok(())
}

fn validate_rf(rf: f64) -> Result<()> {
    if !rf.is_finite() {
        return Err(PortOptError::InvalidInput("rf must be finite".into()));
    }
    Ok(())
}

/// Arithmetic annualized return `periods * mean(r)` (compensated sum).
///
/// # Errors
/// [`PortOptError::InvalidInput`] on an empty/invalid series or a
/// non-positive `periods`.
pub fn annualized_return(r: &[f64], periods: f64) -> Result<f64> {
    validate_periods(periods)?;
    validate_series(r, 1)?;
    Ok(periods * kahan_sum(r.iter().copied()) / r.len() as f64)
}

/// Annualized volatility `sqrt(periods) * std(r, ddof=1)`.
///
/// # Errors
/// [`PortOptError::InvalidInput`] on an invalid series or a non-positive
/// `periods`.
pub fn annualized_vol(r: &[f64], periods: f64) -> Result<f64> {
    validate_periods(periods)?;
    validate_series(r, 2)?;
    let n = r.len() as f64;
    let mu = kahan_sum(r.iter().copied()) / n;
    let var = kahan_sum(r.iter().map(|v| (v - mu) * (v - mu))) / (n - 1.0);
    Ok(periods.sqrt() * var.sqrt())
}

/// Annualized Sharpe `(ann_return - rf) / ann_vol`.
///
/// # Errors
/// [`PortOptError::InvalidInput`] when `rf` is not finite;
/// [`PortOptError::Undefined`] when the annualized vol is below 1e-15.
pub fn sharpe_ratio(r: &[f64], rf: f64, periods: f64) -> Result<f64> {
    validate_rf(rf)?;
    let vol = annualized_vol(r, periods)?;
    if vol < 1e-15 {
        return Err(PortOptError::Undefined(
            "Sharpe undefined: portfolio volatility is zero".into(),
        ));
    }
    Ok((annualized_return(r, periods)? - rf) / vol)
}

/// Maximum drawdown of the compounded wealth path (positive fraction).
///
/// # Errors
/// [`PortOptError::InvalidInput`] on NaN/inf or a simple return `<= -1`.
pub fn max_drawdown(r: &[f64]) -> Result<f64> {
    validate_series(r, 1)?;
    let mut wealth = 1.0_f64;
    let mut peak = 1.0_f64;
    let mut mdd = 0.0_f64;
    for &ret in r {
        wealth *= 1.0 + ret;
        if wealth > peak {
            peak = wealth;
        }
        let dd = 1.0 - wealth / peak;
        if dd > mdd {
            mdd = dd;
        }
    }
    Ok(mdd)
}

/// Pinned rebalance dates `window, window+step, ...` strictly below `t`
/// (`ceil((t - window) / step)` dates; any `step >= 1` is safe).
///
/// # Errors
/// [`PortOptError::InvalidInput`] for `window < 2`, `step < 1`, or
/// `t <= window` (sample not longer than the estimation window).
pub fn rebalance_indices(t: usize, window: usize, step: usize) -> Result<Vec<usize>> {
    if window < 2 {
        return Err(PortOptError::InvalidInput(
            "estimation window must be >= 2".into(),
        ));
    }
    if step < 1 {
        return Err(PortOptError::InvalidInput(
            "rebalance step must be >= 1".into(),
        ));
    }
    if t <= window {
        return Err(PortOptError::InvalidInput(format!(
            "sample length {t} must exceed the estimation window {window}"
        )));
    }
    Ok((window..t).step_by(step).collect())
}

/// Result of [`run_backtest`].
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Daily portfolio returns over the holdout rows `[window, T)`.
    pub portfolio_returns: Vec<f64>,
    /// Row indices at which weights were recomputed.
    pub rebalance_dates: Vec<usize>,
    /// Target weights per rebalance (K rows).
    pub weights: Vec<Vec<f64>>,
    /// L1 turnover at each rebalance after the first (length K-1).
    pub turnovers: Vec<f64>,
    /// `252 * mean(r)` over the holdout.
    pub ann_return: f64,
    /// `sqrt(252) * std(r, ddof=1)` over the holdout.
    pub ann_vol: f64,
    /// `(ann_return - rf) / ann_vol`.
    pub sharpe: f64,
    /// Maximum drawdown (positive fraction).
    pub max_drawdown: f64,
    /// Mean of `turnovers` (0 when fewer than 2 rebalances).
    pub avg_turnover: f64,
}

/// Quarterly expanding-window backtest with the pinned schedule.
///
/// `weight_fn(history)` receives a copy of the expanding daily-return
/// history (rows `[0, t_k)`) and must return weights of length `n`
/// summing to 1 within 1e-6. A fallible weight function lets callers
/// implement the pinned tangency fallback (API_SPEC 6.3) themselves.
/// Requires `T >= window + 2` (two holdout days, so that the ddof = 1
/// volatility exists) and a finite `rf`.
///
/// # Errors
/// Schedule/shape validation errors, a propagated `weight_fn` error, or
/// invalid weights (wrong length, non-finite, sum away from 1).
pub fn run_backtest<F>(
    returns: &Matrix,
    mut weight_fn: F,
    rf: f64,
    window: usize,
    step: usize,
) -> Result<BacktestResult>
where
    F: FnMut(&Matrix) -> Result<Vec<f64>>,
{
    validate_returns(returns, 3)?;
    validate_rf(rf)?;
    let (t, n) = (returns.nrows(), returns.ncols());
    if window < 2 {
        return Err(PortOptError::InvalidInput(
            "estimation window must be >= 2".into(),
        ));
    }
    if t < window + 2 {
        return Err(PortOptError::InvalidInput(format!(
            "backtest needs at least 2 holdout days: sample length {t} must be >= window + 2 = {}",
            window + 2
        )));
    }
    let dates = rebalance_indices(t, window, step)?;
    let k = dates.len();
    let mut weights: Vec<Vec<f64>> = Vec::with_capacity(k);
    let mut port = vec![0.0; t - window];
    for (idx, &t_k) in dates.iter().enumerate() {
        let history = Matrix::from_rows(
            &(0..t_k)
                .map(|r| returns.row(r).to_vec())
                .collect::<Vec<_>>(),
        )?;
        let w = weight_fn(&history)?;
        if w.len() != n || !w.iter().all(|v| v.is_finite()) {
            return Err(PortOptError::InvalidInput(format!(
                "weight_fn returned an invalid weight vector at t={t_k}"
            )));
        }
        let wsum: f64 = w.iter().sum();
        if (wsum - 1.0).abs() > 1e-6 {
            return Err(PortOptError::InvalidInput(format!(
                "weight_fn weights must sum to 1 (got {wsum:.8}) at t={t_k}"
            )));
        }
        let t_end = if idx + 1 < k { dates[idx + 1] } else { t };
        for row in t_k..t_end {
            let r = returns.row(row);
            port[row - window] = kahan_sum((0..n).map(|j| r[j] * w[j]));
        }
        weights.push(w);
    }
    let turnovers: Vec<f64> = (1..k)
        .map(|i| kahan_sum((0..n).map(|j| (weights[i][j] - weights[i - 1][j]).abs())))
        .collect();
    let avg_turnover = if turnovers.is_empty() {
        0.0
    } else {
        kahan_sum(turnovers.iter().copied()) / turnovers.len() as f64
    };
    Ok(BacktestResult {
        ann_return: annualized_return(&port, ANNUALIZATION)?,
        ann_vol: annualized_vol(&port, ANNUALIZATION)?,
        sharpe: sharpe_ratio(&port, rf, ANNUALIZATION)?,
        max_drawdown: max_drawdown(&port)?,
        portfolio_returns: port,
        rebalance_dates: dates,
        weights,
        turnovers,
        avg_turnover,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_pinned() {
        let d = rebalance_indices(1250, 500, 63).unwrap();
        assert_eq!(d.len(), 12);
        assert_eq!(d[0], 500);
        assert_eq!(d[11], 1193);
        assert!(rebalance_indices(500, 500, 63).is_err());
        assert!(rebalance_indices(1250, 1, 63).is_err());
        assert!(rebalance_indices(1250, 500, 0).is_err());
    }

    #[test]
    fn metric_hand_values() {
        let r = [0.01, -0.02, 0.015, 0.005];
        let mean = r.iter().sum::<f64>() / 4.0;
        assert!((annualized_return(&r, 252.0).unwrap() - 252.0 * mean).abs() < 1e-15);
        let var = r.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / 3.0;
        assert!(
            (annualized_vol(&r, 252.0).unwrap() - (252.0_f64).sqrt() * var.sqrt()).abs() < 1e-15
        );
    }

    #[test]
    fn max_drawdown_hand_case() {
        // Wealth: 1.1, 0.88, 0.968 -> peak 1.1, trough 0.88 -> dd = 0.2
        let mdd = max_drawdown(&[0.10, -0.20, 0.10]).unwrap();
        assert!((mdd - 0.2).abs() < 1e-12);
        assert_eq!(max_drawdown(&[0.01, 0.02]).unwrap(), 0.0);
    }

    #[test]
    fn zero_vol_sharpe_rejected() {
        let r = [0.01; 10];
        assert!(matches!(
            sharpe_ratio(&r, 0.0, 252.0),
            Err(PortOptError::Undefined(_))
        ));
    }

    #[test]
    fn metric_input_validation() {
        assert!(annualized_return(&[], 252.0).is_err());
        assert!(annualized_vol(&[0.01], 252.0).is_err());
        assert!(max_drawdown(&[f64::NAN]).is_err());
        assert!(max_drawdown(&[-1.5]).is_err());
        assert!(annualized_vol(&[0.01, 0.02], -252.0).is_err());
        assert!(sharpe_ratio(&[0.01, 0.02], f64::NAN, 252.0).is_err());
    }

    #[test]
    fn rebalance_indices_huge_step_terminates() {
        assert_eq!(rebalance_indices(1250, 500, usize::MAX).unwrap(), vec![500]);
        assert_eq!(rebalance_indices(1251, 500, 750).unwrap(), vec![500, 1250]);
    }
}
