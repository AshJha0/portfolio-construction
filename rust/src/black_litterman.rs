//! Black-Litterman: reverse optimization and the posterior master formula.
//!
//! Master formula (pinned form, API_SPEC.md section 4). With prior
//! (equilibrium) returns `pi`, uncertainty scaling `tau` in `(0, 1]`,
//! pick matrix `P` (k x n), view returns `Q` (k,), and view covariance
//! `Omega` (k x k):
//!
//! ```text
//! A_k     = P (tau Sigma) P' + Omega                (k x k, SPD)
//! mu_post = pi + (tau Sigma) P' A_k^-1 (Q - P pi)
//! M       = tau Sigma - (tau Sigma) P' A_k^-1 P (tau Sigma)
//! S_post  = Sigma + M
//! ```
//!
//! This is algebraically identical to the textbook
//! `[(tau S)^-1 + P' O^-1 P]^-1 [(tau S)^-1 pi + P' O^-1 Q]` form but
//! needs only one k x k Cholesky factorization, which is the pinned
//! implementation.
//!
//! Limits (tested): fixed `Omega` and `tau -> 0` gives `mu_post -> pi`
//! and `M -> 0`; `Omega -> inf` (zero confidence) gives `mu_post -> pi`
//! and `M -> tau Sigma`. Note the default Omega scales with tau, making
//! the posterior tau-invariant — the tau limit must be taken at fixed
//! Omega.
//!
//! Omega default (He & Litterman 1999, scaled by a confidence divisor):
//! `Omega = diag(P (tau Sigma) P')` with each entry divided by a per-view
//! confidence `c_k > 0`: `Omega_kk = [P tau S P']_kk / c_k` (`c_k -> inf`:
//! dogmatic view; `c_k -> 0`: view ignored). This is *not* Idzorek's
//! (2005) calibration, which solves for Omega such that the posterior tilt
//! is a stated fraction of the full-confidence tilt.

use serde::Deserialize;

use crate::error::{PortOptError, Result};
use crate::linalg::{check_square_symmetric, chol_solve, cholesky, dot, Matrix};
use crate::markowitz::tangency_weights;

/// A single investor view (schema of `data/views.json`).
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum View {
    /// `{"type": "absolute", "asset": A, "q": x, "confidence": c}` —
    /// P row is the unit vector of `asset`.
    Absolute {
        /// Asset name the view is about.
        asset: String,
        /// Expected (excess) return asserted by the view.
        q: f64,
        /// Per-view confidence (> 0; defaults to 1).
        #[serde(default = "default_confidence")]
        confidence: f64,
    },
    /// `{"type": "relative", "long": A, "short": B, "q": x, "confidence": c}`
    /// — P row is +1 at `long` and -1 at `short`.
    Relative {
        /// Outperforming asset.
        long: String,
        /// Underperforming asset.
        short: String,
        /// Expected outperformance.
        q: f64,
        /// Per-view confidence (> 0; defaults to 1).
        #[serde(default = "default_confidence")]
        confidence: f64,
    },
}

fn default_confidence() -> f64 {
    1.0
}

/// Reverse optimization: `pi = delta * Sigma * w_mkt`.
///
/// `w_mkt` must be non-negative with positive sum; it is renormalized to
/// sum to one before use. `pi` are *excess* returns: tangency at
/// `rf = 0` on `(pi, Sigma)` returns exactly `w_mkt`.
///
/// # Errors
/// [`PortOptError::InvalidInput`] for shape mismatch, negative cap
/// weights, non-positive cap sum, or `delta <= 0` / non-finite.
pub fn implied_equilibrium_returns(sigma: &Matrix, w_mkt: &[f64], delta: f64) -> Result<Vec<f64>> {
    check_square_symmetric(sigma)?;
    if w_mkt.len() != sigma.nrows() {
        return Err(PortOptError::InvalidInput(format!(
            "w_mkt length {} does not match sigma dimension {}",
            w_mkt.len(),
            sigma.nrows()
        )));
    }
    if !w_mkt.iter().all(|v| v.is_finite()) {
        return Err(PortOptError::InvalidInput(
            "w_mkt contains NaN or infinite entries".into(),
        ));
    }
    if w_mkt.iter().any(|v| *v < 0.0) {
        return Err(PortOptError::InvalidInput(
            "market-cap weights must be non-negative".into(),
        ));
    }
    let tot: f64 = w_mkt.iter().sum();
    if tot <= 0.0 {
        return Err(PortOptError::InvalidInput(
            "market-cap weights must have positive sum".into(),
        ));
    }
    if !delta.is_finite() || delta <= 0.0 {
        return Err(PortOptError::InvalidInput(
            "risk aversion delta must be positive and finite".into(),
        ));
    }
    let w: Vec<f64> = w_mkt.iter().map(|v| v / tot).collect();
    Ok(sigma.matvec(&w).iter().map(|v| delta * v).collect())
}

/// Build `(P, Q, confidences)` from a list of views.
///
/// # Errors
/// [`PortOptError::InvalidInput`] on an empty view list or asset list,
/// duplicate asset names, an unknown asset name, a relative view whose
/// legs coincide, a non-finite `q`, or a non-positive/non-finite
/// confidence.
pub fn views_to_matrices(
    views: &[View],
    asset_names: &[String],
) -> Result<(Matrix, Vec<f64>, Vec<f64>)> {
    if asset_names.is_empty() {
        return Err(PortOptError::InvalidInput(
            "asset_names must be non-empty".into(),
        ));
    }
    for i in 0..asset_names.len() {
        if asset_names[i + 1..].contains(&asset_names[i]) {
            return Err(PortOptError::InvalidInput(
                "asset_names contains duplicates".into(),
            ));
        }
    }
    if views.is_empty() {
        return Err(PortOptError::InvalidInput(
            "views list must be non-empty".into(),
        ));
    }
    let index_of = |name: &str| -> Result<usize> {
        asset_names
            .iter()
            .position(|a| a == name)
            .ok_or_else(|| PortOptError::InvalidInput(format!("unknown asset in view: {name:?}")))
    };
    let (k, n) = (views.len(), asset_names.len());
    let mut p = Matrix::zeros(k, n);
    let mut q = vec![0.0; k];
    let mut conf = vec![1.0; k];
    for (i, view) in views.iter().enumerate() {
        let (qi, ci) = match view {
            View::Absolute {
                asset,
                q,
                confidence,
            } => {
                p.set(i, index_of(asset)?, 1.0);
                (*q, *confidence)
            }
            View::Relative {
                long,
                short,
                q,
                confidence,
            } => {
                let i_long = index_of(long)?;
                let i_short = index_of(short)?;
                if i_long == i_short {
                    return Err(PortOptError::InvalidInput(format!(
                        "view {i}: relative view: long and short legs must differ"
                    )));
                }
                p.set(i, i_long, 1.0);
                p.set(i, i_short, -1.0);
                (*q, *confidence)
            }
        };
        if !qi.is_finite() {
            return Err(PortOptError::InvalidInput(format!(
                "view {i}: q must be finite"
            )));
        }
        if !ci.is_finite() || ci <= 0.0 {
            return Err(PortOptError::InvalidInput(format!(
                "view {i}: confidence must be positive and finite"
            )));
        }
        q[i] = qi;
        conf[i] = ci;
    }
    Ok((p, q, conf))
}

fn validate_views(p: &Matrix, n: usize) -> Result<()> {
    if p.ncols() != n {
        return Err(PortOptError::InvalidInput(format!(
            "P must be k x {n}, got {}x{}",
            p.nrows(),
            p.ncols()
        )));
    }
    if p.nrows() < 1 {
        return Err(PortOptError::InvalidInput(
            "P must contain at least one view row".into(),
        ));
    }
    if !p.is_finite() {
        return Err(PortOptError::InvalidInput(
            "P contains NaN or infinite entries".into(),
        ));
    }
    for i in 0..p.nrows() {
        if p.row(i).iter().all(|v| *v == 0.0) {
            return Err(PortOptError::InvalidInput(format!(
                "view row {i} of P is identically zero"
            )));
        }
    }
    Ok(())
}

fn validate_tau(tau: f64) -> Result<()> {
    if !tau.is_finite() || !(tau > 0.0 && tau <= 1.0) {
        return Err(PortOptError::InvalidInput(format!(
            "tau must lie in (0, 1], got {tau}"
        )));
    }
    Ok(())
}

fn validate_confidences(conf: Option<&[f64]>, k: usize) -> Result<Vec<f64>> {
    match conf {
        None => Ok(vec![1.0; k]),
        Some(c) => {
            if c.len() != k {
                return Err(PortOptError::InvalidInput(format!(
                    "confidences must have length {k}, got {}",
                    c.len()
                )));
            }
            if c.iter().any(|v| !v.is_finite() || *v <= 0.0) {
                return Err(PortOptError::InvalidInput(
                    "confidences must be positive and finite".into(),
                ));
            }
            Ok(c.to_vec())
        }
    }
}

/// `B = P (tau Sigma)`, shape k x n.
fn p_tau_sigma(p: &Matrix, ts: &Matrix) -> Matrix {
    let (k, n) = (p.nrows(), ts.nrows());
    let mut b = Matrix::zeros(k, n);
    for i in 0..k {
        for j in 0..n {
            b.set(i, j, dot(p.row(i), &ts.col(j)));
        }
    }
    b
}

/// He-Litterman default view covariance scaled by a confidence divisor:
/// `Omega = diag(P (tau Sigma) P')` with each diagonal entry divided by
/// the corresponding per-view confidence `c_k > 0` (default 1).
#[allow(clippy::needless_range_loop)] // index loops mirror the Python reference; kept verbatim so evaluation order stays pinned to the goldens
pub fn default_omega(
    sigma: &Matrix,
    p: &Matrix,
    tau: f64,
    confidences: Option<&[f64]>,
) -> Result<Matrix> {
    check_square_symmetric(sigma)?;
    validate_views(p, sigma.nrows())?;
    validate_tau(tau)?;
    let k = p.nrows();
    let conf = validate_confidences(confidences, k)?;
    let ts = sigma.scaled(tau);
    let b = p_tau_sigma(p, &ts);
    let mut omega = Matrix::zeros(k, k);
    for i in 0..k {
        omega.set(i, i, dot(b.row(i), p.row(i)) / conf[i]);
    }
    Ok(omega)
}

/// Black-Litterman posterior.
#[derive(Debug, Clone)]
pub struct BlResult {
    /// Posterior expected returns.
    pub mu_post: Vec<f64>,
    /// Posterior return covariance `Sigma + M`.
    pub sigma_post: Matrix,
    /// Posterior covariance of the mean estimate,
    /// `tau Sigma - tau Sigma P' A_k^-1 P tau Sigma`.
    pub m: Matrix,
    /// The view covariance actually used.
    pub omega: Matrix,
}

/// Black-Litterman posterior via the pinned master formula.
///
/// `omega = None` uses [`default_omega`] (with optional per-view
/// confidences); passing both `omega` and `confidences` is an error (the
/// confidences only parametrize the default Omega). The k x k system
/// `A_k = P tau Sigma P' + Omega` is factored once by Cholesky and solved
/// column-wise.
///
/// # Errors
/// Validation errors for shapes, tau, views, or confidences;
/// [`PortOptError::NotPositiveDefinite`] if `A_k` is not SPD.
#[allow(clippy::needless_range_loop)] // index loops mirror the Python reference; kept verbatim so evaluation order stays pinned to the goldens
pub fn bl_posterior(
    sigma: &Matrix,
    pi: &[f64],
    p: &Matrix,
    q: &[f64],
    tau: f64,
    omega: Option<&Matrix>,
    confidences: Option<&[f64]>,
) -> Result<BlResult> {
    check_square_symmetric(sigma)?;
    let n = sigma.nrows();
    if pi.len() != n || !pi.iter().all(|v| v.is_finite()) {
        return Err(PortOptError::InvalidInput(format!(
            "pi must be a finite vector of length {n}"
        )));
    }
    validate_views(p, n)?;
    let k = p.nrows();
    if q.len() != k || !q.iter().all(|v| v.is_finite()) {
        return Err(PortOptError::InvalidInput(format!(
            "Q must be a finite vector of length {k}"
        )));
    }
    validate_tau(tau)?;
    let om = match omega {
        None => default_omega(sigma, p, tau, confidences)?,
        Some(o) => {
            if confidences.is_some() {
                return Err(PortOptError::InvalidInput(
                    "confidences are only used with the default Omega; pass one or the other"
                        .into(),
                ));
            }
            check_square_symmetric(o)?;
            if o.nrows() != k {
                return Err(PortOptError::InvalidInput(format!(
                    "Omega must be {k} x {k}, got {}x{}",
                    o.nrows(),
                    o.ncols()
                )));
            }
            o.clone()
        }
    };

    let ts = sigma.scaled(tau);
    let b = p_tau_sigma(p, &ts); // k x n = P (tau Sigma)
    let mut a_k = Matrix::zeros(k, k);
    for i in 0..k {
        for j in 0..k {
            a_k.set(i, j, dot(b.row(i), p.row(j)) + om.get(i, j));
        }
    }
    let low = cholesky(&a_k)?; // errors if Omega makes A_k non-SPD

    let resid: Vec<f64> = (0..k).map(|i| q[i] - dot(p.row(i), pi)).collect();
    let x = chol_solve(&low, &resid)?;
    let mu_post: Vec<f64> = (0..n).map(|j| pi[j] + dot(&b.col(j), &x)).collect();

    // M = tau S - B' A_k^-1 B: A_k^-1 B via n solves, one per column of B.
    let mut ainv_b = Matrix::zeros(k, n);
    for j in 0..n {
        let col = chol_solve(&low, &b.col(j))?;
        for i in 0..k {
            ainv_b.set(i, j, col[i]);
        }
    }
    let mut m = Matrix::zeros(n, n);
    let mut sigma_post = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let v = ts.get(i, j) - dot(&b.col(i), &ainv_b.col(j));
            m.set(i, j, v);
            sigma_post.set(i, j, sigma.get(i, j) + v);
        }
    }
    Ok(BlResult {
        mu_post,
        sigma_post,
        m,
        omega: om,
    })
}

/// Posterior-based optimal weights.
///
/// Pinned: unconstrained tangency portfolio computed from
/// `(mu_post, sigma_post)`. Because `pi = delta Sigma w_mkt` are
/// equilibrium *excess* returns, the pinned rate is `rf = 0` (a non-zero
/// rf would double-count the risk-free rate); with no informative views
/// the tangency on `(pi, Sigma)` recovers exactly `w_mkt`.
#[allow(clippy::too_many_arguments)] // public signature mirrors the Python bl_weights keyword arguments
pub fn bl_weights(
    sigma: &Matrix,
    pi: &[f64],
    p: &Matrix,
    q: &[f64],
    tau: f64,
    rf: f64,
    omega: Option<&Matrix>,
    confidences: Option<&[f64]>,
) -> Result<(Vec<f64>, BlResult)> {
    if !rf.is_finite() {
        return Err(PortOptError::InvalidInput("rf must be finite".into()));
    }
    let res = bl_posterior(sigma, pi, p, q, tau, omega, confidences)?;
    let w = tangency_weights(&res.mu_post, &res.sigma_post, rf)?;
    Ok((w, res))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sigma2() -> Matrix {
        Matrix::from_rows(&[vec![0.04, 0.01], vec![0.01, 0.09]]).unwrap()
    }

    #[test]
    fn tau_and_view_validation() {
        let sig = sigma2();
        let p = Matrix::from_rows(&[vec![1.0, 0.0]]).unwrap();
        let q = [0.05];
        assert!(bl_posterior(&sig, &[0.02, 0.03], &p, &q, 0.0, None, None).is_err());
        assert!(bl_posterior(&sig, &[0.02, 0.03], &p, &q, 1.5, None, None).is_err());
        let zero_row = Matrix::from_rows(&[vec![0.0, 0.0]]).unwrap();
        assert!(bl_posterior(&sig, &[0.02, 0.03], &zero_row, &q, 0.05, None, None).is_err());
        assert!(bl_posterior(&sig, &[0.02, 0.03], &p, &[0.05, 0.01], 0.05, None, None).is_err());
        assert!(default_omega(&sig, &p, 0.05, Some(&[-1.0])).is_err());
    }

    #[test]
    fn views_to_matrices_builds_rows() {
        let names: Vec<String> = vec!["A".into(), "B".into()];
        let views = vec![
            View::Absolute {
                asset: "B".into(),
                q: 0.1,
                confidence: 1.0,
            },
            View::Relative {
                long: "A".into(),
                short: "B".into(),
                q: 0.02,
                confidence: 0.5,
            },
        ];
        let (p, q, c) = views_to_matrices(&views, &names).unwrap();
        assert_eq!(p.row(0), &[0.0, 1.0]);
        assert_eq!(p.row(1), &[1.0, -1.0]);
        assert_eq!(q, vec![0.1, 0.02]);
        assert_eq!(c, vec![1.0, 0.5]);
        assert!(views_to_matrices(&[], &names).is_err());
        let bad = vec![View::Absolute {
            asset: "Z".into(),
            q: 0.1,
            confidence: 1.0,
        }];
        assert!(views_to_matrices(&bad, &names).is_err());
        let nan_q = vec![View::Absolute {
            asset: "A".into(),
            q: f64::NAN,
            confidence: 1.0,
        }];
        assert!(views_to_matrices(&nan_q, &names).is_err());
        let same = vec![View::Relative {
            long: "A".into(),
            short: "A".into(),
            q: 0.02,
            confidence: 1.0,
        }];
        assert!(views_to_matrices(&same, &names).is_err());
        let dup: Vec<String> = vec!["A".into(), "B".into(), "A".into()];
        assert!(views_to_matrices(&views, &dup).is_err());
    }
}
