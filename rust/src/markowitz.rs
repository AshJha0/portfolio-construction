//! Markowitz mean-variance: closed forms, simplex projection, and PGD.
//!
//! Closed-form results use the classical two-fund theorem. With
//! `x = Sigma^-1 mu` and `y = Sigma^-1 1` (both computed by Cholesky
//! solves, pinned in [`crate::linalg`]) define the scalars
//!
//! ```text
//! A = 1' Sigma^-1 mu = sum(x)
//! B = mu' Sigma^-1 mu = mu . x
//! C = 1' Sigma^-1 1  = sum(y)
//! D = B*C - A*A
//! ```
//!
//! Every frontier portfolio is `w(m) = g + h*m` with
//! `g = (B*y - A*x)/D` and `h = (C*x - A*y)/D`; the minimum-variance
//! portfolio is `y/C` (return `A/C`, variance `1/C`) and the tangency
//! portfolio w.r.t. rate `rf` is `Sigma^-1 (mu - rf 1) / (A - rf C)`.
//!
//! Constrained problems (long-only, sum-to-one, optional box bounds) are
//! solved by projected gradient descent with monotone backtracking and an
//! exact sort-based projection onto the box-constrained simplex; all
//! steps are pinned in API_SPEC.md sections 3.3-3.6.

use crate::error::{PortOptError, Result};
use crate::linalg::{chol_solve, cholesky, dot, fma_dot, np_matvec, np_sum, np_vecmat, Matrix};

/// Pinned quadratic-penalty coefficient for the constrained frontier.
pub const FRONTIER_PENALTY: f64 = 1.0e4;
/// Pinned PGD convergence tolerance (max-norm of the iterate step).
pub const PGD_TOL: f64 = 1.0e-12;
/// Pinned PGD iteration cap.
pub const PGD_MAX_ITER: usize = 10_000;

fn validate_mu_sigma(mu: &[f64], sigma: &Matrix) -> Result<()> {
    if sigma.nrows() != sigma.ncols() {
        return Err(PortOptError::InvalidInput(format!(
            "sigma must be square, got shape {}x{}",
            sigma.nrows(),
            sigma.ncols()
        )));
    }
    if mu.len() != sigma.nrows() {
        return Err(PortOptError::InvalidInput(format!(
            "mu length {} does not match sigma dimension {}",
            mu.len(),
            sigma.nrows()
        )));
    }
    if !mu.iter().all(|v| v.is_finite()) {
        return Err(PortOptError::InvalidInput(
            "mu contains NaN or infinite entries".into(),
        ));
    }
    Ok(())
}

/// Two-fund theorem scalars `A`, `B`, `C`, `D` (see module docs).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoFundScalars {
    /// `A = 1' Sigma^-1 mu`.
    pub a: f64,
    /// `B = mu' Sigma^-1 mu`.
    pub b: f64,
    /// `C = 1' Sigma^-1 1`.
    pub c: f64,
    /// `D = B*C - A^2`.
    pub d: f64,
}

/// Compute the two-fund scalars via two Cholesky solves.
///
/// # Errors
/// Propagates validation/Cholesky errors for a non-SPD `sigma`.
pub fn two_fund_scalars(mu: &[f64], sigma: &Matrix) -> Result<TwoFundScalars> {
    validate_mu_sigma(mu, sigma)?;
    let low = cholesky(sigma)?;
    let x = chol_solve(&low, mu)?;
    let ones = vec![1.0; mu.len()];
    let y = chol_solve(&low, &ones)?;
    let a: f64 = x.iter().sum();
    let b = dot(mu, &x);
    let c: f64 = y.iter().sum();
    Ok(TwoFundScalars { a, b, c, d: b * c - a * a })
}

/// Unconstrained (sum-to-one) minimum-variance weights `Sigma^-1 1 / C`.
///
/// For `n = 1` returns the trivial `[1.0]`.
pub fn min_variance_weights(sigma: &Matrix) -> Result<Vec<f64>> {
    if sigma.nrows() != sigma.ncols() {
        return Err(PortOptError::InvalidInput(format!(
            "sigma must be square, got shape {}x{}",
            sigma.nrows(),
            sigma.ncols()
        )));
    }
    if sigma.nrows() == 1 {
        let v = sigma.get(0, 0);
        if !(v > 0.0) || !v.is_finite() {
            return Err(PortOptError::InvalidInput(
                "1x1 covariance must be positive and finite".into(),
            ));
        }
        return Ok(vec![1.0]);
    }
    let low = cholesky(sigma)?;
    let ones = vec![1.0; sigma.nrows()];
    let y = chol_solve(&low, &ones)?;
    let c: f64 = y.iter().sum();
    Ok(y.iter().map(|v| v / c).collect())
}

/// Tangency (max-Sharpe) weights `Sigma^-1 (mu - rf 1) / (A - rf C)`.
///
/// # Errors
/// [`PortOptError::Undefined`] if `A - rf*C <= 0` — the tangency
/// portfolio is undefined when the risk-free rate is at or above the
/// minimum-variance return (in particular when `rf` exceeds every
/// expected return).
pub fn tangency_weights(mu: &[f64], sigma: &Matrix, rf: f64) -> Result<Vec<f64>> {
    validate_mu_sigma(mu, sigma)?;
    if !rf.is_finite() {
        return Err(PortOptError::InvalidInput("rf must be finite".into()));
    }
    if sigma.nrows() == 1 {
        if mu[0] <= rf {
            return Err(PortOptError::Undefined(
                "tangency undefined: risk-free rate >= expected return of the single asset"
                    .into(),
            ));
        }
        let v = sigma.get(0, 0);
        if !(v > 0.0) || !v.is_finite() {
            return Err(PortOptError::InvalidInput(
                "1x1 covariance must be positive and finite".into(),
            ));
        }
        return Ok(vec![1.0]);
    }
    let low = cholesky(sigma)?;
    let excess: Vec<f64> = mu.iter().map(|v| v - rf).collect();
    let z = chol_solve(&low, &excess)?;
    let denom: f64 = z.iter().sum(); // = A - rf*C
    if denom <= 0.0 {
        return Err(PortOptError::Undefined(
            "tangency undefined: A - rf*C <= 0 (risk-free rate is at or above the \
             minimum-variance portfolio return)"
                .into(),
        ));
    }
    Ok(z.iter().map(|v| v / denom).collect())
}

/// Efficient-frontier weights for a target return via the two-fund theorem.
///
/// `w(m) = g + h*m` with `g = (B*y - A*x)/D`, `h = (C*x - A*y)/D`.
///
/// # Errors
/// [`PortOptError::Undefined`] if `n = 1` and the target differs from the
/// single asset's mean, if `D <= 1e-12*B*C` (degenerate frontier, e.g.
/// `mu` proportional to ones), or if the target lies below the
/// minimum-variance return `A/C` (includes negative targets).
pub fn frontier_weights(mu: &[f64], sigma: &Matrix, target: f64) -> Result<Vec<f64>> {
    validate_mu_sigma(mu, sigma)?;
    if !target.is_finite() {
        return Err(PortOptError::InvalidInput("target return must be finite".into()));
    }
    if sigma.nrows() == 1 {
        if (target - mu[0]).abs() > 1e-12 * (1.0 + mu[0].abs()) {
            return Err(PortOptError::Undefined(
                "n=1: target return must equal the single asset's expected return".into(),
            ));
        }
        return Ok(vec![1.0]);
    }
    let low = cholesky(sigma)?;
    let x = chol_solve(&low, mu)?;
    let ones = vec![1.0; mu.len()];
    let y = chol_solve(&low, &ones)?;
    let a: f64 = x.iter().sum();
    let b = dot(mu, &x);
    let c: f64 = y.iter().sum();
    let d = b * c - a * a;
    // Pinned relative degeneracy test: by Cauchy-Schwarz D >= 0 with
    // equality iff mu is proportional to ones; rounding turns the exact
    // zero into a tiny residual, so compare against B*C (> 0).
    if d <= 1e-12 * b * c {
        return Err(PortOptError::Undefined(format!(
            "degenerate frontier: D = B*C - A^2 = {d:.3e} <= 1e-12*B*C \
             (expected returns proportional to ones?)"
        )));
    }
    let mv_ret = a / c;
    if target < mv_ret - 1e-12 * (1.0 + mv_ret.abs()) {
        return Err(PortOptError::Undefined(format!(
            "target return {target:.6} lies below the minimum-variance return \
             {mv_ret:.6}; the efficient frontier is defined for targets >= A/C"
        )));
    }
    Ok((0..mu.len())
        .map(|i| {
            let g = (b * y[i] - a * x[i]) / d;
            let h = (c * x[i] - a * y[i]) / d;
            g + h * target
        })
        .collect())
}

/// One frontier sweep: pinned target grid plus per-point weights/variances.
#[derive(Debug, Clone)]
pub struct FrontierSweep {
    /// Target returns `lo + k*(hi - lo)/(n_points - 1)`, `k = 0..n_points-1`.
    pub targets: Vec<f64>,
    /// Frontier weights per target (`n_points` rows).
    pub weights: Vec<Vec<f64>>,
    /// Portfolio variances `w' Sigma w` per target.
    pub variances: Vec<f64>,
}

fn frontier_targets(sc: &TwoFundScalars, mu: &[f64], n_points: usize) -> Vec<f64> {
    let lo = sc.a / sc.c;
    let hi = mu.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (0..n_points)
        .map(|k| lo + k as f64 * (hi - lo) / (n_points - 1) as f64)
        .collect()
}

/// Unconstrained frontier sweep over the pinned grid: `n_points` equally
/// spaced targets from the minimum-variance return `A/C` to `max(mu)`.
pub fn efficient_frontier(mu: &[f64], sigma: &Matrix, n_points: usize) -> Result<FrontierSweep> {
    validate_mu_sigma(mu, sigma)?;
    if n_points < 2 {
        return Err(PortOptError::InvalidInput("n_points must be >= 2".into()));
    }
    let sc = two_fund_scalars(mu, sigma)?;
    if sc.d <= 1e-12 * sc.b * sc.c {
        return Err(PortOptError::Undefined(
            "degenerate frontier: D <= 1e-12*B*C".into(),
        ));
    }
    let targets = frontier_targets(&sc, mu, n_points);
    let mut weights = Vec::with_capacity(n_points);
    let mut variances = Vec::with_capacity(n_points);
    for &tgt in &targets {
        let w = frontier_weights(mu, sigma, tgt)?;
        variances.push(dot(&w, &sigma.matvec(&w)));
        weights.push(w);
    }
    Ok(FrontierSweep { targets, weights, variances })
}

fn clip(v: f64, lo: f64, hi: f64) -> f64 {
    v.max(lo).min(hi)
}

/// Exact Euclidean projection onto `{w : sum w = total, lb <= w <= ub}`.
///
/// Pinned sort-based algorithm (API_SPEC.md section 3.5). The KKT
/// solution is `w[i] = clip(v[i] - t, lb[i], ub[i])` for the unique shift
/// `t` making the sum equal `total`; `g(t) = sum_i clip(v[i] - t, ...)`
/// is piecewise linear and non-increasing with kinks at `v[i] - lb[i]`
/// and, for finite bounds, `v[i] - ub[i]`. The algorithm sorts the
/// breakpoints ascending, scans left to right to bracket `g(t) = total`,
/// and finishes with one exact linear interpolation on the bracketing
/// segment — finite and deterministic, no iteration tolerance. With
/// `lb = 0, ub = +inf` it reproduces the classic Duchi et al. sorted
/// simplex projection.
///
/// `lb = None` means 0 for every coordinate; `ub = None` means `+inf`.
///
/// # Errors
/// [`PortOptError::Infeasible`] when `sum(lb) > total`, `sum(ub) < total`
/// or some `lb[i] > ub[i]`; [`PortOptError::InvalidInput`] on NaN/inf in
/// `v` or NaN bounds.
pub fn project_simplex_box(
    v: &[f64],
    total: f64,
    lb: Option<&[f64]>,
    ub: Option<&[f64]>,
) -> Result<Vec<f64>> {
    let n = v.len();
    if n == 0 {
        return Err(PortOptError::InvalidInput("cannot project an empty vector".into()));
    }
    if !v.iter().all(|x| x.is_finite()) {
        return Err(PortOptError::InvalidInput(
            "projection input contains NaN or infinite entries".into(),
        ));
    }
    if !total.is_finite() {
        return Err(PortOptError::InvalidInput("total must be finite".into()));
    }
    let lo: Vec<f64> = match lb {
        None => vec![0.0; n],
        Some(b) => {
            if b.len() != n {
                return Err(PortOptError::InvalidInput(format!(
                    "lb length {} does not match vector length {n}",
                    b.len()
                )));
            }
            b.to_vec()
        }
    };
    let hi: Vec<f64> = match ub {
        None => vec![f64::INFINITY; n],
        Some(b) => {
            if b.len() != n {
                return Err(PortOptError::InvalidInput(format!(
                    "ub length {} does not match vector length {n}",
                    b.len()
                )));
            }
            b.to_vec()
        }
    };
    if lo.iter().any(|x| x.is_nan()) || hi.iter().any(|x| x.is_nan()) {
        return Err(PortOptError::InvalidInput("bounds contain NaN".into()));
    }
    if lo.iter().zip(&hi).any(|(l, u)| l > u) {
        return Err(PortOptError::Infeasible("some lb_i > ub_i".into()));
    }
    let lo_sum = np_sum(&lo);
    let hi_sum = np_sum(&hi);
    let feas_tol = 1e-12 * (1.0 + total.abs());
    if lo_sum > total + feas_tol {
        return Err(PortOptError::Infeasible(format!(
            "sum(lb) = {lo_sum:.6} exceeds total {total:.6}"
        )));
    }
    if hi_sum < total - feas_tol {
        return Err(PortOptError::Infeasible(format!(
            "sum(ub) = {hi_sum:.6} is below total {total:.6}"
        )));
    }

    let clipped = |t: f64| -> Vec<f64> {
        (0..n).map(|i| clip(v[i] - t, lo[i], hi[i])).collect()
    };
    let g = |t: f64| -> f64 { np_sum(&clipped(t)) };

    // Breakpoints: all v[i] - lb[i], plus v[i] - ub[i] for finite ub.
    let mut bps: Vec<f64> = (0..n).map(|i| v[i] - lo[i]).collect();
    for i in 0..n {
        if hi[i].is_finite() {
            bps.push(v[i] - hi[i]);
        }
    }
    bps.sort_by(f64::total_cmp);

    let t_first = bps[0];
    let g_first = g(t_first);
    if g_first < total {
        // t* lies left of every breakpoint; there the unclipped
        // (infinite-ub) coordinates give slope -k with k = number of
        // infinite upper bounds.
        let k = hi.iter().filter(|u| !u.is_finite()).count();
        if k == 0 {
            // sum(ub) >= total guarantees g_first >= total up to rounding; snap.
            return Ok(clipped(t_first));
        }
        return Ok(clipped(t_first - (total - g_first) / k as f64));
    }

    let mut t_prev = t_first;
    let mut g_prev = g_first;
    for &t_cur in &bps[1..] {
        let g_cur = g(t_cur);
        if g_cur <= total {
            let t_star = if g_prev == g_cur {
                t_cur
            } else {
                t_prev + (g_prev - total) * (t_cur - t_prev) / (g_prev - g_cur)
            };
            return Ok(clipped(t_star));
        }
        t_prev = t_cur;
        g_prev = g_cur;
    }
    // Right of every breakpoint g(t) = sum(lb) <= total; feasibility
    // makes the residual pure rounding.
    Ok(clipped(bps[bps.len() - 1]))
}

/// Result of [`pgd_minimize`].
#[derive(Debug, Clone)]
pub struct PgdResult {
    /// Final (feasible) iterate.
    pub weights: Vec<f64>,
    /// Number of iterations executed.
    pub iterations: usize,
    /// True when the max-norm step fell below the tolerance; a `false`
    /// value reports hitting the iteration cap (never an error).
    pub converged: bool,
}

/// Projected gradient descent with monotone backtracking (pinned).
///
/// Pinned algorithm (API_SPEC.md section 3.6):
///
/// 1. `w = project(w0)`; step `t = 1`.
/// 2. Each iteration: `t = 2t`; `g = grad(w)`; `w+ = project(w - t g)`;
///    while the sufficient-decrease condition
///    `f(w+) <= f(w) + g.(w+ - w) + ||w+ - w||^2 / (2t)` fails, halve `t`
///    and recompute `w+` (at most 100 halvings, then accept).
/// 3. Stop when `max_i |w+_i - w_i| < tol` (converged) or after
///    `max_iter` iterations (non-convergence reported via the flag).
///
/// Deterministic — no randomness anywhere.
pub fn pgd_minimize<F, G, P>(
    fun: F,
    grad: G,
    project: P,
    w0: &[f64],
    tol: f64,
    max_iter: usize,
) -> Result<PgdResult>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
    P: Fn(&[f64]) -> Result<Vec<f64>>,
{
    let mut w = project(w0)?;
    let n = w.len();
    let mut t = 1.0_f64;
    for it in 1..=max_iter {
        t *= 2.0;
        let gvec = grad(&w);
        let fw = fun(&w);
        let candidate = |step: f64| -> Result<Vec<f64>> {
            let trial: Vec<f64> = (0..n).map(|i| w[i] - step * gvec[i]).collect();
            project(&trial)
        };
        let mut w_new = candidate(t)?;
        let mut dw: Vec<f64> = (0..n).map(|i| w_new[i] - w[i]).collect();
        let mut accepted = false;
        for _ in 0..100 {
            let decrease = fw + fma_dot(&gvec, &dw) + fma_dot(&dw, &dw) / (2.0 * t);
            if fun(&w_new) <= decrease {
                accepted = true;
                break;
            }
            t *= 0.5;
            w_new = candidate(t)?;
            dw = (0..n).map(|i| w_new[i] - w[i]).collect();
        }
        if !accepted {
            // Mirror the Python for-else: accept the final halved step.
            w_new = candidate(t)?;
            dw = (0..n).map(|i| w_new[i] - w[i]).collect();
        }
        let step_norm = dw.iter().fold(0.0_f64, |m, d| m.max(d.abs()));
        w = w_new;
        if step_norm < tol {
            return Ok(PgdResult { weights: w, iterations: it, converged: true });
        }
    }
    Ok(PgdResult { weights: w, iterations: max_iter, converged: false })
}

/// Validate sigma via Cholesky and return the pinned PGD starting point
/// (equal weights; [`pgd_minimize`] projects it onto the feasible set).
fn pgd_start(sigma: &Matrix) -> Result<Vec<f64>> {
    cholesky(sigma)?; // PSD/validity gate
    let n = sigma.nrows();
    Ok(vec![1.0 / n as f64; n])
}

/// Constrained minimum variance: `min 1/2 w' Sigma w` over the
/// box-constrained simplex (defaults: long-only, `lb = 0`, `ub = +inf`).
///
/// Solved by the pinned PGD started from projected equal weights. The
/// objective is strictly convex, so the minimizer is unique and language
/// ports agree to well below the golden tolerance 1e-8.
pub fn min_variance_longonly(
    sigma: &Matrix,
    lb: Option<&[f64]>,
    ub: Option<&[f64]>,
) -> Result<Vec<f64>> {
    let w0 = pgd_start(sigma)?;
    // The objective/gradient mirror the reference's numpy arithmetic
    // (`w @ Sigma @ w` and `Sigma @ w`) operation for operation so the
    // deterministic PGD trajectory is reproduced bit for bit.
    let res = pgd_minimize(
        |w| 0.5 * fma_dot(&np_vecmat(w, sigma), w),
        |w| np_matvec(sigma, w),
        |v| project_simplex_box(v, 1.0, lb, ub),
        &w0,
        PGD_TOL,
        PGD_MAX_ITER,
    )?;
    Ok(res.weights)
}

/// Constrained frontier point via the pinned quadratic-penalty PGD.
///
/// Pinned objective (API_SPEC.md section 3.6):
///
/// ```text
/// f(w) = 1/2 w' Sigma w + penalty/2 * (mu'w - target)^2
/// ```
///
/// minimized over `{sum w = 1, lb <= w <= ub}` with `penalty = 1e4` by
/// default. The penalty formulation itself is the contract: the golden
/// values are the exact minimizer of THIS objective (the achieved return
/// deviates from the target by `O(1/penalty)` by design).
pub fn constrained_frontier_weights(
    mu: &[f64],
    sigma: &Matrix,
    target: f64,
    penalty: f64,
    lb: Option<&[f64]>,
    ub: Option<&[f64]>,
) -> Result<Vec<f64>> {
    validate_mu_sigma(mu, sigma)?;
    if !(penalty > 0.0) {
        return Err(PortOptError::InvalidInput("penalty must be positive".into()));
    }
    let w0 = pgd_start(sigma)?;
    let res = pgd_minimize(
        |w| {
            let gap = fma_dot(mu, w) - target;
            0.5 * fma_dot(&np_vecmat(w, sigma), w) + 0.5 * penalty * gap * gap
        },
        |w| {
            let gap = fma_dot(mu, w) - target;
            let sw = np_matvec(sigma, w);
            (0..w.len()).map(|i| sw[i] + penalty * gap * mu[i]).collect()
        },
        |v| project_simplex_box(v, 1.0, lb, ub),
        &w0,
        PGD_TOL,
        PGD_MAX_ITER,
    )?;
    Ok(res.weights)
}

/// Constrained frontier sweep over the same pinned target grid as
/// [`efficient_frontier`] (min-variance return `A/C` to `max(mu)`).
pub fn constrained_frontier(
    mu: &[f64],
    sigma: &Matrix,
    n_points: usize,
    penalty: f64,
    lb: Option<&[f64]>,
    ub: Option<&[f64]>,
) -> Result<FrontierSweep> {
    validate_mu_sigma(mu, sigma)?;
    if n_points < 2 {
        return Err(PortOptError::InvalidInput("n_points must be >= 2".into()));
    }
    let sc = two_fund_scalars(mu, sigma)?;
    let targets = frontier_targets(&sc, mu, n_points);
    let mut weights = Vec::with_capacity(n_points);
    let mut variances = Vec::with_capacity(n_points);
    for &tgt in &targets {
        let w = constrained_frontier_weights(mu, sigma, tgt, penalty, lb, ub)?;
        variances.push(dot(&w, &sigma.matvec(&w)));
        weights.push(w);
    }
    Ok(FrontierSweep { targets, weights, variances })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diag_sigma() -> Matrix {
        Matrix::from_diag(&[0.04, 0.09, 0.01])
    }

    #[test]
    fn degenerate_frontier_rejected() {
        let mu = [0.05, 0.05, 0.05];
        let err = frontier_weights(&mu, &diag_sigma(), 0.05).unwrap_err();
        assert!(err.to_string().contains("degenerate frontier"));
    }

    #[test]
    fn n1_trivial_and_errors() {
        let sig = Matrix::from_rows(&[vec![0.04]]).unwrap();
        assert_eq!(min_variance_weights(&sig).unwrap(), vec![1.0]);
        assert_eq!(tangency_weights(&[0.05], &sig, 0.01).unwrap(), vec![1.0]);
        assert_eq!(frontier_weights(&[0.05], &sig, 0.05).unwrap(), vec![1.0]);
        assert!(frontier_weights(&[0.05], &sig, 0.06).is_err());
        assert!(tangency_weights(&[0.05], &sig, 0.10).is_err());
    }

    #[test]
    fn projection_hand_cases() {
        let w = project_simplex_box(&[0.9, 0.5, -0.2], 1.0, None, None).unwrap();
        assert!((w[0] - 0.7).abs() < 1e-14 && (w[1] - 0.3).abs() < 1e-14 && w[2].abs() < 1e-14);
        // Fixed point on the simplex.
        let v = [0.2, 0.3, 0.5];
        let w = project_simplex_box(&v, 1.0, None, None).unwrap();
        for i in 0..3 {
            assert!((w[i] - v[i]).abs() < 1e-14);
        }
        // Uniform shift only.
        let w = project_simplex_box(&[1.0, 1.0], 1.0, None, None).unwrap();
        assert!((w[0] - 0.5).abs() < 1e-14 && (w[1] - 0.5).abs() < 1e-14);
        // Single coordinate.
        let w = project_simplex_box(&[5.0], 1.0, None, None).unwrap();
        assert!((w[0] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn projection_infeasible_bounds() {
        let z = [0.0; 4];
        assert!(matches!(
            project_simplex_box(&z, 1.0, Some(&[0.3; 4]), None),
            Err(PortOptError::Infeasible(_))
        ));
        assert!(matches!(
            project_simplex_box(&z, 1.0, None, Some(&[0.2; 4])),
            Err(PortOptError::Infeasible(_))
        ));
        assert!(matches!(
            project_simplex_box(&[0.0, 0.0], 1.0, Some(&[0.6, 0.0]), Some(&[0.5, 1.0])),
            Err(PortOptError::Infeasible(_))
        ));
        assert!(matches!(
            project_simplex_box(&[f64::NAN, 0.0], 1.0, None, None),
            Err(PortOptError::InvalidInput(_))
        ));
    }
}
