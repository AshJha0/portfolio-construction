//! Risk parity / equal risk contribution via cyclical coordinate descent.
//!
//! CCD update derivation (Griveau-Billion, Richard & Roncalli 2013 style,
//! log-barrier formulation; pinned in API_SPEC.md section 5). Solve
//!
//! ```text
//! min_{y > 0}  F(y) = 1/2 y' Sigma y - sum_i b_i ln y_i
//! ```
//!
//! for risk budgets `b_i > 0` (normalized to sum 1). Coordinate-wise,
//! with `m_i = (Sigma y)_i - Sigma_ii y_i` (the cross term, held fixed):
//!
//! ```text
//! dF/dy_i = Sigma_ii y_i + m_i - b_i / y_i = 0
//! =>  y_i = ( -m_i + sqrt(m_i^2 + 4 Sigma_ii b_i) ) / (2 Sigma_ii)
//! ```
//!
//! taking the positive root of the quadratic (the discriminant is > 0
//! because `Sigma_ii > 0` and `b_i > 0`, so `y_i` stays strictly
//! positive). At a stationary point `y_i (Sigma y)_i = b_i`, so the
//! normalized weights `w = y / sum(y)` have risk contributions exactly
//! equal to the budgets. `F` is strictly convex on the positive orthant,
//! so the solution is unique and CCD converges monotonically.
//!
//! Sweep order pinned: coordinates `i = 0..n-1` ascending; the cached
//! product `Sigma y` is updated incrementally after each coordinate
//! change. Convergence: after each full sweep,
//! `max_i |y_i - y_i_prev_sweep| < tol` (default 1e-10); at most 10000
//! sweeps (non-convergence is reported via a flag, never an error).

use crate::error::{PortOptError, Result};
use crate::linalg::{check_square_symmetric, cholesky, dot, Matrix};

/// Pinned CCD sweep-level convergence tolerance.
pub const CCD_TOL: f64 = 1.0e-10;
/// Pinned CCD sweep cap.
pub const CCD_MAX_SWEEPS: usize = 10_000;

/// Normalized risk contributions `RC_i = w_i (Sigma w)_i / (w' Sigma w)`.
///
/// The contributions sum to exactly 1 (portfolio variance normalized).
///
/// # Errors
/// [`PortOptError::InvalidInput`] on shape/finiteness violations;
/// [`PortOptError::Undefined`] if the portfolio variance is not strictly
/// positive.
pub fn risk_contributions(w: &[f64], sigma: &Matrix) -> Result<Vec<f64>> {
    check_square_symmetric(sigma)?;
    if w.len() != sigma.nrows() {
        return Err(PortOptError::InvalidInput(format!(
            "weights length {} does not match sigma dimension {}",
            w.len(),
            sigma.nrows()
        )));
    }
    if !w.iter().all(|v| v.is_finite()) {
        return Err(PortOptError::InvalidInput(
            "weights contain NaN or infinite entries".into(),
        ));
    }
    let sw = sigma.matvec(w);
    let var = dot(w, &sw);
    if var <= 0.0 {
        return Err(PortOptError::Undefined(
            "portfolio variance must be strictly positive".into(),
        ));
    }
    Ok((0..w.len()).map(|i| w[i] * sw[i] / var).collect())
}

/// Naive inverse-volatility weights
/// `w_i = (1/sqrt(S_ii)) / sum_j (1/sqrt(S_jj))`.
pub fn inverse_vol_weights(sigma: &Matrix) -> Result<Vec<f64>> {
    check_square_symmetric(sigma)?;
    let n = sigma.nrows();
    if (0..n).any(|i| sigma.get(i, i) <= 0.0) {
        return Err(PortOptError::InvalidInput(
            "all variances must be strictly positive for inverse-vol".into(),
        ));
    }
    let inv: Vec<f64> = (0..n).map(|i| 1.0 / sigma.get(i, i).sqrt()).collect();
    let tot: f64 = inv.iter().sum();
    Ok(inv.iter().map(|v| v / tot).collect())
}

fn validate_budgets(budgets: Option<&[f64]>, n: usize) -> Result<Vec<f64>> {
    match budgets {
        None => Ok(vec![1.0 / n as f64; n]),
        Some(b) => {
            if b.len() != n {
                return Err(PortOptError::InvalidInput(format!(
                    "budgets must have length {n}, got {}",
                    b.len()
                )));
            }
            if b.iter().any(|v| !v.is_finite() || *v <= 0.0) {
                return Err(PortOptError::InvalidInput(
                    "risk budgets must be positive and finite".into(),
                ));
            }
            let tot: f64 = b.iter().sum();
            Ok(b.iter().map(|v| v / tot).collect())
        }
    }
}

/// Result of [`risk_budget_solve`].
#[derive(Debug, Clone)]
pub struct CcdResult {
    /// Long-only weights on the simplex.
    pub weights: Vec<f64>,
    /// Number of full sweeps executed.
    pub sweeps: usize,
    /// True when the sweep-level tolerance was met; `false` reports the
    /// sweep cap (never an error).
    pub converged: bool,
}

/// General risk budgeting via pinned CCD (see module docs).
///
/// `budgets = None` means equal budgets (ERC). Initialization is pinned
/// to `y_i = 1/n`. An almost-zero-vol asset (tiny `Sigma_ii`) yields a
/// large `y_i` and hence a weight near 1 after normalization — the
/// correct ERC limit; the coordinate quadratic keeps a positive
/// discriminant because `Sigma_ii > 0` after the Cholesky gate.
///
/// # Errors
/// [`PortOptError::NotPositiveDefinite`] for singular / non-PSD `sigma`;
/// [`PortOptError::InvalidInput`] for bad budgets or parameters.
#[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(x > 0.0)` is deliberate: it rejects NaN as well as non-positive values
#[allow(clippy::needless_range_loop)] // index loops mirror the Python reference; kept verbatim so evaluation order stays pinned to the goldens
pub fn risk_budget_solve(
    sigma: &Matrix,
    budgets: Option<&[f64]>,
    tol: f64,
    max_sweeps: usize,
) -> Result<CcdResult> {
    check_square_symmetric(sigma)?;
    cholesky(sigma)?; // SPD gate: rejects singular / non-PSD inputs
    let n = sigma.nrows();
    let b = validate_budgets(budgets, n)?;
    if !(tol > 0.0) || !tol.is_finite() {
        return Err(PortOptError::InvalidInput("tol must be positive".into()));
    }
    if max_sweeps < 1 {
        return Err(PortOptError::InvalidInput("max_sweeps must be >= 1".into()));
    }

    let mut y = vec![1.0 / n as f64; n];
    let mut sy = sigma.matvec(&y); // cached Sigma y, updated incrementally
    for sweep in 1..=max_sweeps {
        let y_prev = y.clone();
        for i in 0..n {
            let sii = sigma.get(i, i);
            let m_i = sy[i] - sii * y[i];
            let y_new = (-m_i + (m_i * m_i + 4.0 * sii * b[i]).sqrt()) / (2.0 * sii);
            let dy = y_new - y[i];
            if dy != 0.0 {
                for r in 0..n {
                    sy[r] += sigma.get(r, i) * dy;
                }
                y[i] = y_new;
            }
        }
        let delta = (0..n).fold(0.0_f64, |m, i| m.max((y[i] - y_prev[i]).abs()));
        if delta < tol {
            let tot: f64 = y.iter().sum();
            return Ok(CcdResult {
                weights: y.iter().map(|v| v / tot).collect(),
                sweeps: sweep,
                converged: true,
            });
        }
    }
    let tot: f64 = y.iter().sum();
    Ok(CcdResult {
        weights: y.iter().map(|v| v / tot).collect(),
        sweeps: max_sweeps,
        converged: false,
    })
}

/// Risk-budget weights (weights only; see [`risk_budget_solve`]) at the
/// pinned [`CCD_TOL`] and [`CCD_MAX_SWEEPS`].
///
/// # Errors
/// As [`risk_budget_weights_with`].
pub fn risk_budget_weights(sigma: &Matrix, budgets: Option<&[f64]>) -> Result<Vec<f64>> {
    risk_budget_weights_with(sigma, budgets, CCD_TOL, CCD_MAX_SWEEPS)
}

/// Risk-budget weights with an explicit tolerance and sweep cap.
///
/// # Errors
/// The errors of [`risk_budget_solve`], plus [`PortOptError::Undefined`]
/// if the CCD hits the sweep cap before the sweep-level step falls below
/// `tol`; use [`risk_budget_solve`] to inspect the unconverged iterate.
pub fn risk_budget_weights_with(
    sigma: &Matrix,
    budgets: Option<&[f64]>,
    tol: f64,
    max_sweeps: usize,
) -> Result<Vec<f64>> {
    let res = risk_budget_solve(sigma, budgets, tol, max_sweeps)?;
    if !res.converged {
        return Err(PortOptError::Undefined(format!(
            "risk_budget_weights: CCD did not converge within {} sweeps (tol {:.3e});              use risk_budget_solve to inspect the iterate",
            res.sweeps, tol
        )));
    }
    Ok(res.weights)
}

/// Equal-risk-contribution weights (equal budgets `b_i = 1/n`) at the
/// pinned [`CCD_TOL`] and [`CCD_MAX_SWEEPS`].
///
/// # Errors
/// As [`risk_budget_weights_with`].
pub fn erc_weights(sigma: &Matrix) -> Result<Vec<f64>> {
    erc_weights_with(sigma, CCD_TOL, CCD_MAX_SWEEPS)
}

/// Equal-risk-contribution weights with an explicit tolerance and sweep cap.
///
/// # Errors
/// As [`risk_budget_weights_with`].
pub fn erc_weights_with(sigma: &Matrix, tol: f64, max_sweeps: usize) -> Result<Vec<f64>> {
    risk_budget_weights_with(sigma, None, tol, max_sweeps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_reject_unconverged_ccd() {
        let sig = Matrix::from_rows(&[
            vec![0.04, 0.006, 0.002],
            vec![0.006, 0.09, 0.009],
            vec![0.002, 0.009, 0.16],
        ])
        .unwrap();
        let err = erc_weights_with(&sig, CCD_TOL, 2).unwrap_err();
        assert!(matches!(err, PortOptError::Undefined(_)));
        let msg = err.to_string();
        assert!(msg.contains("within 2 sweeps"), "{msg}");
        assert!(msg.contains("tol 1.000e-10"), "{msg}");
        // The pinned defaults converge on the same input.
        assert!(erc_weights(&sig).is_ok());
    }

    #[test]
    fn erc_diagonal_equals_inverse_vol() {
        let sig = Matrix::from_diag(&[0.04, 0.09, 0.16]);
        let w_erc = erc_weights(&sig).unwrap();
        let w_iv = inverse_vol_weights(&sig).unwrap();
        for i in 0..3 {
            assert!((w_erc[i] - w_iv[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn two_asset_analytic() {
        // For n=2 ERC weights are inverse-vol regardless of correlation.
        let s1 = 0.2_f64;
        let s2 = 0.1_f64;
        let rho = 0.5;
        let sig = Matrix::from_rows(&[vec![s1 * s1, rho * s1 * s2], vec![rho * s1 * s2, s2 * s2]])
            .unwrap();
        let w = erc_weights(&sig).unwrap();
        let expect0 = (1.0 / s1) / (1.0 / s1 + 1.0 / s2);
        assert!((w[0] - expect0).abs() < 1e-9);
    }

    #[test]
    fn almost_zero_vol_asset_converges() {
        let sig = Matrix::from_rows(&[
            vec![1e-10, 0.0, 0.0],
            vec![0.0, 0.04, 0.001],
            vec![0.0, 0.001, 0.09],
        ])
        .unwrap();
        let res = risk_budget_solve(&sig, None, CCD_TOL, CCD_MAX_SWEEPS).unwrap();
        assert!(res.converged);
        assert!(res.weights[0] > 0.99); // near-zero-vol asset dominates
        let rc = risk_contributions(&res.weights, &sig).unwrap();
        let spread = rc.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - rc.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(spread < 1e-8);
    }

    #[test]
    fn error_paths() {
        let sig = Matrix::from_diag(&[0.04, 0.09]);
        assert!(risk_budget_solve(&sig, Some(&[0.5, -0.5]), CCD_TOL, 10).is_err());
        assert!(risk_budget_solve(&sig, Some(&[0.5]), CCD_TOL, 10).is_err());
        assert!(risk_budget_solve(&sig, None, -1.0, 10).is_err());
        assert!(risk_budget_solve(&sig, None, CCD_TOL, 0).is_err());
        let sing = Matrix::from_rows(&[vec![1.0, 1.0], vec![1.0, 1.0]]).unwrap();
        assert!(matches!(
            risk_budget_solve(&sing, None, CCD_TOL, 10),
            Err(PortOptError::NotPositiveDefinite(_))
        ));
    }
}
