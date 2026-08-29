//! Return-moment estimators: sample moments and Ledoit-Wolf shrinkage.
//!
//! Pinned conventions (API_SPEC.md section 2):
//!
//! * sample mean uses divisor `T`; sample covariance uses divisor `T-1`
//!   (ddof = 1);
//! * Ledoit-Wolf (2004) constant-correlation shrinkage with the classic
//!   closed-form intensity; the dispersion sums (`pi`/`rho`/`gamma`) use
//!   divisor `T` while `S` keeps divisor `T-1` — this deliberate
//!   simplification is pinned so that every language port computes the
//!   identical number;
//! * annualization multiplies means and covariances by 252 (configurable).
//!
//! The moment sums mirror the reference implementation's floating point
//! exactly: means accumulate row by row (numpy `sum(axis=0)`) and the
//! covariance products use the OpenBLAS dgemm accumulation order
//! ([`crate::linalg::blas_dot`]), because downstream PGD golden values
//! are ulp-sensitive to these inputs. The shrinkage dispersion sums use
//! Neumaier compensation ([`crate::linalg::kahan_sum`]), which is well
//! inside their 1e-10 tolerance.

use crate::error::{PortOptError, Result};
use crate::linalg::{kahan_sum, Matrix};

/// Validate a `T x N` returns matrix (finite entries, `T >= min_rows`).
///
/// # Errors
/// [`PortOptError::InvalidInput`] on an empty panel, too few rows, or
/// NaN/inf entries.
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
    Ok(())
}

/// Per-asset sample mean: `mu[i] = (1/T) * sum_t X[t][i]`.
///
/// Accumulated row by row (numpy's `x.sum(axis=0)` order) so the values
/// are bit-identical to the reference implementation's.
pub fn sample_mean(returns: &Matrix) -> Result<Vec<f64>> {
    validate_returns(returns, 1)?;
    let (t, n) = (returns.nrows(), returns.ncols());
    let mut acc = vec![0.0_f64; n];
    for r in 0..t {
        let row = returns.row(r);
        for j in 0..n {
            acc[j] += row[j];
        }
    }
    Ok(acc.into_iter().map(|s| s / t as f64).collect())
}

/// Demeaned columns of the returns panel: `xc[t][i] = X[t][i] - mu[i]`,
/// stored column-major (`xc[i][t]`) for the T-length dot products.
fn demeaned_columns(returns: &Matrix) -> Result<Vec<Vec<f64>>> {
    let mu = sample_mean(returns)?;
    let (t, n) = (returns.nrows(), returns.ncols());
    Ok((0..n)
        .map(|j| (0..t).map(|r| returns.get(r, j) - mu[j]).collect())
        .collect())
}

/// Sample covariance with ddof = 1: `S[i][j] = (1/(T-1)) sum_t xc[t][i] xc[t][j]`.
///
/// The T-length products are accumulated with [`crate::linalg::blas_dot`]
/// (the reference BLAS dgemm order), making every entry bit-identical to
/// the Python implementation — required because downstream PGD golden
/// values are ulp-sensitive to the covariance input.
pub fn sample_cov(returns: &Matrix) -> Result<Matrix> {
    validate_returns(returns, 2)?;
    let xc = demeaned_columns(returns)?;
    cov_from_demeaned(&xc, returns.nrows())
}

/// Shared ddof = 1 covariance kernel over precomputed demeaned columns.
fn cov_from_demeaned(xc: &[Vec<f64>], t: usize) -> Result<Matrix> {
    let n = xc.len();
    let mut s = Matrix::zeros(n, n);
    for i in 0..n {
        for j in i..n {
            let v = crate::linalg::blas_dot(&xc[i], &xc[j]) / (t - 1) as f64;
            s.set(i, j, v);
            s.set(j, i, v);
        }
    }
    Ok(s)
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
/// Pinned term by term (API_SPEC.md section 2.2). With demeaned returns
/// `xc[t][i]` and `S` the ddof = 1 sample covariance:
///
/// * `rbar = (2 / (N(N-1))) * sum_{i<j} S_ij / sqrt(S_ii S_jj)`
/// * target `F_ii = S_ii`, `F_ij = rbar * sqrt(S_ii S_jj)`
/// * `pi_ij = (1/T) sum_t (xc_ti xc_tj - S_ij)^2`, `pi_hat = sum_ij pi_ij`
/// * `theta_ii,ij = (1/T) sum_t (xc_ti^2 - S_ii)(xc_ti xc_tj - S_ij)`
/// * `rho_hat = sum_i pi_ii + sum over ordered pairs i != j of
///   (rbar/2) sqrt(S_jj/S_ii) theta_ii,ij` (the ordered double sum
///   supplies both symmetric terms of the classic formula exactly once)
/// * `gamma_hat = sum_ij (F_ij - S_ij)^2`
/// * `kappa = (pi_hat - rho_hat) / gamma_hat`;
///   `delta = max(0, min(1, kappa / T))` (`delta = 0` if `gamma_hat <= 0`)
///
/// # Errors
/// [`PortOptError::InvalidInput`] for `N < 2`, `T < 2`, non-finite input,
/// or a non-positive sample variance.
pub fn shrink_covariance(returns: &Matrix) -> Result<ShrinkageResult> {
    validate_returns(returns, 2)?;
    let (t, n) = (returns.nrows(), returns.ncols());
    if n < 2 {
        return Err(PortOptError::InvalidInput(
            "shrinkage requires at least 2 assets".into(),
        ));
    }
    let xc = demeaned_columns(returns)?;
    let s = cov_from_demeaned(&xc, t)?;
    let sd: Vec<f64> = (0..n).map(|i| s.get(i, i).sqrt()).collect();
    if sd.iter().any(|&v| !(v > 0.0)) {
        return Err(PortOptError::InvalidInput(
            "shrinkage requires strictly positive sample variances".into(),
        ));
    }

    let mut corr_sum = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            corr_sum += s.get(i, j) / (sd[i] * sd[j]);
        }
    }
    let rbar = 2.0 * corr_sum / (n * (n - 1)) as f64;

    // Constant-correlation target: F_ii = S_ii, F_ij = rbar * sd_i * sd_j.
    let mut f = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            if i == j {
                f.set(i, j, s.get(i, i));
            } else {
                f.set(i, j, sd[i] * sd[j] * rbar);
            }
        }
    }

    let tf = t as f64;
    let mut pi_hat = 0.0;
    let mut rho_hat = 0.0;
    let mut gamma_hat = 0.0;
    for i in 0..n {
        for j in 0..n {
            let sij = s.get(i, j);
            let pij = kahan_sum((0..t).map(|r| {
                let dev = xc[i][r] * xc[j][r] - sij;
                dev * dev
            })) / tf;
            pi_hat += pij;
            if i == j {
                rho_hat += pij;
            }
            let df = f.get(i, j) - sij;
            gamma_hat += df * df;
        }
    }
    // theta terms over ordered pairs i != j: each unordered pair {i, j}
    // contributes both symmetric halves of the classic rho formula, once
    // from the (i, j) pass and once from the (j, i) pass.
    for i in 0..n {
        let sii = s.get(i, i);
        for j in 0..n {
            if i == j {
                continue;
            }
            let sij = s.get(i, j);
            let theta_ii_ij = kahan_sum(
                (0..t).map(|r| (xc[i][r] * xc[i][r] - sii) * (xc[i][r] * xc[j][r] - sij)),
            ) / tf;
            rho_hat += (rbar / 2.0) * (s.get(j, j) / sii).sqrt() * theta_ii_ij;
        }
    }

    let intensity = if gamma_hat <= 0.0 {
        0.0
    } else {
        let kappa = (pi_hat - rho_hat) / gamma_hat;
        (kappa / tf).clamp(0.0, 1.0)
    };

    let mut sigma = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            sigma.set(i, j, intensity * f.get(i, j) + (1.0 - intensity) * s.get(i, j));
        }
    }
    Ok(ShrinkageResult { sigma, intensity, rbar, sample: s, target: f })
}

/// Annualize a per-period mean vector: `mu_ann = periods * mu`.
pub fn annualize_mean(mu: &[f64], periods: f64) -> Result<Vec<f64>> {
    if !(periods > 0.0) || !periods.is_finite() {
        return Err(PortOptError::InvalidInput("periods must be positive".into()));
    }
    Ok(mu.iter().map(|v| periods * v).collect())
}

/// Annualize a per-period covariance: `Sigma_ann = periods * Sigma`.
pub fn annualize_cov(cov: &Matrix, periods: f64) -> Result<Matrix> {
    if !(periods > 0.0) || !periods.is_finite() {
        return Err(PortOptError::InvalidInput("periods must be positive".into()));
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
    }

    #[test]
    fn annualize_scales() {
        let mu = annualize_mean(&[0.001], 252.0).unwrap();
        assert!((mu[0] - 0.252).abs() < 1e-15);
        assert!(annualize_mean(&[0.1], 0.0).is_err());
        assert!(annualize_cov(&Matrix::identity(2), -1.0).is_err());
    }
}
