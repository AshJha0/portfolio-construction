//! Return-moment estimators: sample moments and Ledoit-Wolf shrinkage.
//!
//! Pinned conventions (API_SPEC.md section 2):
//!
//! * sample mean uses divisor `T`; sample covariance uses divisor `T-1`
//!   (ddof = 1);
//! * Ledoit-Wolf (2004) constant-correlation shrinkage: the intensity is
//!   computed exactly as in the paper's Appendix B / `covCor.m` (divisor
//!   T throughout, both theta terms per ordered pair) and applied in the
//!   ddof = 1 convention — see [`shrink_covariance`];
//! * annualization multiplies means and covariances by 252 (configurable).
//!
//! Every T-length reduction uses the compensated [`crate::linalg::kahan_sum`],
//! which matches the reference's correctly rounded `math.fsum` to a couple
//! of ulps; no golden value depends on a BLAS or a summation order.

use crate::error::{PortOptError, Result};
use crate::linalg::{kahan_sum, Matrix};

/// Validate a `T x N` simple-returns matrix: finite entries strictly
/// greater than -1 (a simple return at or below -100% is not a valid price
/// move), `T >= min_rows`, `N >= 1`.
///
/// # Errors
/// [`PortOptError::InvalidInput`] on an empty panel, too few rows,
/// NaN/inf entries, or a return `<= -1`.
pub fn validate_returns(returns: &Matrix, min_rows: usize) -> Result<()> {
    if returns.ncols() < 1 {
        return Err(PortOptError::InvalidInput(
            "returns must contain at least one asset column".into(),
        ));
    }
    if returns.nrows() < min_rows {
        return Err(PortOptError::InvalidInput(format!(
            "returns must contain at least {min_rows} rows, got {}",
            returns.nrows()
        )));
    }
    if !returns.is_finite() {
        return Err(PortOptError::InvalidInput(
            "returns contain NaN or infinite entries".into(),
        ));
    }
    for r in 0..returns.nrows() {
        if returns.row(r).iter().any(|&v| v <= -1.0) {
            return Err(PortOptError::InvalidInput(
                "returns contain a simple return <= -1 (-100% or worse)".into(),
            ));
        }
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

/// Per-asset sample mean: `mu[i] = (1/T) * sum_t X[t][i]` (compensated sum).
pub fn sample_mean(returns: &Matrix) -> Result<Vec<f64>> {
    validate_returns(returns, 1)?;
    let (t, n) = (returns.nrows(), returns.ncols());
    Ok((0..n)
        .map(|j| kahan_sum((0..t).map(|r| returns.get(r, j))) / t as f64)
        .collect())
}

/// Demeaned columns of the returns panel: `xc[t][i] = X[t][i] - mu[i]`,
/// stored column-major (`xc[i][t]`) for the T-length reductions.
fn demeaned_columns(returns: &Matrix) -> Result<Vec<Vec<f64>>> {
    let mu = sample_mean(returns)?;
    let (t, n) = (returns.nrows(), returns.ncols());
    Ok((0..n)
        .map(|j| (0..t).map(|r| returns.get(r, j) - mu[j]).collect())
        .collect())
}

/// Sample covariance with ddof = 1: `S[i][j] = (1/(T-1)) sum_t xc[t][i] xc[t][j]`.
pub fn sample_cov(returns: &Matrix) -> Result<Matrix> {
    validate_returns(returns, 2)?;
    let xc = demeaned_columns(returns)?;
    Ok(cov_from_demeaned(&xc, returns.nrows()))
}

/// Shared ddof = 1 covariance kernel over precomputed demeaned columns.
fn cov_from_demeaned(xc: &[Vec<f64>], t: usize) -> Matrix {
    let n = xc.len();
    let mut s = Matrix::zeros(n, n);
    for i in 0..n {
        for j in i..n {
            let v = kahan_sum((0..t).map(|r| xc[i][r] * xc[j][r])) / (t - 1) as f64;
            s.set(i, j, v);
            s.set(j, i, v);
        }
    }
    s
}

/// Output of Ledoit-Wolf constant-correlation shrinkage.
#[derive(Debug, Clone)]
pub struct ShrinkageResult {
    /// Shrunk covariance `delta * F + (1 - delta) * S`.
    pub sigma: Matrix,
    /// Shrinkage intensity `delta` in `[0, 1]`.
    pub intensity: f64,
    /// Average sample correlation used by the constant-correlation target.
    pub rbar: f64,
    /// Sample covariance `S` (ddof = 1).
    pub sample: Matrix,
    /// Constant-correlation target `F`.
    pub target: Matrix,
}

/// Ledoit-Wolf (2004) shrinkage towards the constant-correlation target.
///
/// Pinned term by term to Ledoit & Wolf (2004), *JPM* 30(4), Appendix B /
/// the authors' `covCor.m` (API_SPEC.md section 2.2). With demeaned
/// returns `x[t][i]`, `S_T = Xc'Xc / T` (divisor T) and `s_ii = S_T[i][i]`:
///
/// * `rbar = (2 / (N(N-1))) * sum_{i<j} S_T_ij / sqrt(s_ii s_jj)`
/// * `pi_ij = (1/T) sum_t (x_ti x_tj - S_T_ij)^2`, `pi_hat = sum_ij pi_ij`
/// * `theta_ii,ij = (1/T) sum_t (x_ti^2 - s_ii)(x_ti x_tj - S_T_ij)`
/// * `rho_hat = sum_i pi_ii + sum over ordered pairs i != j of
///   rbar * sqrt(s_jj / s_ii) * theta_ii,ij`
/// * `gamma_hat = sum_ij (F_T_ij - S_T_ij)^2`, `F_T_ii = s_ii`,
///   `F_T_ij = rbar sqrt(s_ii s_jj)`
/// * `kappa = (pi_hat - rho_hat) / gamma_hat`;
///   `delta = max(0, min(1, kappa / T))` (`delta = 0` if `gamma_hat <= 0`)
///
/// The intensity is exactly the paper's; the returned matrices use the
/// project-wide ddof = 1 convention: `sample = S = T/(T-1) S_T`,
/// `target = F = T/(T-1) F_T`, `sigma = delta F + (1 - delta) S`.
///
/// # Errors
/// [`PortOptError::InvalidInput`] for `N < 2`, `T < 2`, non-finite input,
/// or a non-positive sample variance.
#[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(x > 0.0)` is deliberate: it rejects NaN as well as non-positive values
pub fn shrink_covariance(returns: &Matrix) -> Result<ShrinkageResult> {
    validate_returns(returns, 2)?;
    let (t, n) = (returns.nrows(), returns.ncols());
    if n < 2 {
        return Err(PortOptError::InvalidInput(
            "shrinkage requires at least 2 assets".into(),
        ));
    }
    let xc = demeaned_columns(returns)?;
    let s = cov_from_demeaned(&xc, t);
    if (0..n).any(|i| !(s.get(i, i) > 0.0)) {
        return Err(PortOptError::InvalidInput(
            "shrinkage requires strictly positive sample variances".into(),
        ));
    }
    let tf = t as f64;
    // Paper convention (divisor T) for every dispersion quantity.
    let s_t = s.scaled((tf - 1.0) / tf);
    let var_t: Vec<f64> = (0..n).map(|i| s_t.get(i, i)).collect();
    let sd_t: Vec<f64> = var_t.iter().map(|v| v.sqrt()).collect();

    let mut corr_sum = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            corr_sum += s_t.get(i, j) / (sd_t[i] * sd_t[j]);
        }
    }
    let rbar = 2.0 * corr_sum / (n * (n - 1)) as f64;

    let mut f_t = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            f_t.set(
                i,
                j,
                if i == j {
                    var_t[i]
                } else {
                    rbar * sd_t[i] * sd_t[j]
                },
            );
        }
    }

    let mut pi_hat = 0.0;
    let mut rho_hat = 0.0;
    let mut gamma_hat = 0.0;
    for i in 0..n {
        let xi2_dev: Vec<f64> = (0..t).map(|r| xc[i][r] * xc[i][r] - var_t[i]).collect();
        for j in 0..n {
            let sij = s_t.get(i, j);
            let pij = kahan_sum((0..t).map(|r| {
                let dev = xc[i][r] * xc[j][r] - sij;
                dev * dev
            })) / tf;
            pi_hat += pij;
            let gd = f_t.get(i, j) - sij;
            gamma_hat += gd * gd;
            if i == j {
                rho_hat += pij;
            } else {
                let theta_ii_ij =
                    kahan_sum((0..t).map(|r| xi2_dev[r] * (xc[i][r] * xc[j][r] - sij))) / tf;
                rho_hat += rbar * (var_t[j] / var_t[i]).sqrt() * theta_ii_ij;
            }
        }
    }

    let intensity = if gamma_hat <= 0.0 {
        0.0
    } else {
        let kappa = (pi_hat - rho_hat) / gamma_hat;
        (kappa / tf).clamp(0.0, 1.0)
    };

    // Output in the project-wide ddof = 1 convention.
    let sd: Vec<f64> = (0..n).map(|i| s.get(i, i).sqrt()).collect();
    let mut f = Matrix::zeros(n, n);
    let mut sigma = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let fij = if i == j {
                s.get(i, i)
            } else {
                rbar * sd[i] * sd[j]
            };
            f.set(i, j, fij);
            sigma.set(i, j, intensity * fij + (1.0 - intensity) * s.get(i, j));
        }
    }
    Ok(ShrinkageResult {
        sigma,
        intensity,
        rbar,
        sample: s,
        target: f,
    })
}

/// Annualize a per-period mean vector: `mu_ann = periods * mu`.
///
/// # Errors
/// [`PortOptError::InvalidInput`] if `periods` is not positive and finite
/// or `mu` has NaN/inf entries.
pub fn annualize_mean(mu: &[f64], periods: f64) -> Result<Vec<f64>> {
    validate_periods(periods)?;
    if !mu.iter().all(|v| v.is_finite()) {
        return Err(PortOptError::InvalidInput(
            "mu contains NaN or infinite entries".into(),
        ));
    }
    Ok(mu.iter().map(|v| periods * v).collect())
}

/// Annualize a per-period covariance: `Sigma_ann = periods * Sigma`.
///
/// # Errors
/// [`PortOptError::InvalidInput`] if `periods` is not positive and finite
/// or `cov` has NaN/inf entries.
pub fn annualize_cov(cov: &Matrix, periods: f64) -> Result<Matrix> {
    validate_periods(periods)?;
    if !cov.is_finite() {
        return Err(PortOptError::InvalidInput(
            "cov contains NaN or infinite entries".into(),
        ));
    }
    Ok(cov.scaled(periods))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_panel() -> Matrix {
        Matrix::from_rows(&[
            vec![0.01, 0.02],
            vec![-0.01, 0.00],
            vec![0.02, 0.01],
            vec![0.00, -0.02],
        ])
        .unwrap()
    }

    #[test]
    fn mean_and_cov_hand_values() {
        let x = small_panel();
        let mu = sample_mean(&x).unwrap();
        assert!((mu[0] - 0.005).abs() < 1e-16);
        assert!((mu[1] - 0.0025).abs() < 1e-16);
        let s = sample_cov(&x).unwrap();
        // ddof=1 hand computation for column 0: deviations 0.005,-0.015,0.015,-0.005
        let v0 = (0.005f64.powi(2) + 0.015f64.powi(2) + 0.015f64.powi(2) + 0.005f64.powi(2)) / 3.0;
        assert!((s.get(0, 0) - v0).abs() < 1e-18);
        assert_eq!(s.get(0, 1), s.get(1, 0));
    }

    #[test]
    fn shrinkage_validation_errors() {
        let one_col = Matrix::from_rows(&[vec![0.01], vec![0.02], vec![0.03]]).unwrap();
        assert!(shrink_covariance(&one_col).is_err());
        let one_row = Matrix::from_rows(&[vec![0.01, 0.02]]).unwrap();
        assert!(shrink_covariance(&one_row).is_err());
        let constant = Matrix::from_rows(&[vec![0.01, 0.02], vec![0.01, 0.03]]).unwrap();
        assert!(shrink_covariance(&constant).is_err()); // zero variance col 0
        let loss = Matrix::from_rows(&[vec![0.01, -1.0], vec![0.01, 0.03]]).unwrap();
        assert!(sample_mean(&loss).is_err()); // simple return <= -1
    }

    #[test]
    fn annualize_scales() {
        let mu = annualize_mean(&[0.001], 252.0).unwrap();
        assert!((mu[0] - 0.252).abs() < 1e-15);
        assert!(annualize_mean(&[0.1], 0.0).is_err());
        assert!(annualize_mean(&[0.1], f64::NAN).is_err());
        assert!(annualize_mean(&[f64::NAN], 252.0).is_err());
        assert!(annualize_cov(&Matrix::identity(2), -1.0).is_err());
    }
}
