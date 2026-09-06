//! Markowitz mean-variance: closed forms, simplex projection, active-set QP.
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
//! Constrained problems (sum-to-one, optional target return, box bounds)
//! are strictly convex QPs solved by the finite dual active-set method of
//! Goldfarb & Idnani (1983) whose every linear solve is the pinned
//! Cholesky (API_SPEC.md section 3.6). The result is the KKT point of the
//! QP up to rounding, so all ports agree to ~1e-13 and `converged` is a
//! statement about the KKT conditions, not about an iterate step size.

use crate::error::{PortOptError, Result};
use crate::linalg::{chol_solve, cholesky, dot, kahan_sum, Matrix};

/// Pinned iteration cap of the dual active-set QP (each iteration adds or
/// drops one constraint; the method is finite for a strictly convex
/// objective, so the cap is only hit on numerically inconsistent inputs
/// and is then reported via `converged = false`).
pub const QP_MAX_ITER: usize = 1000;
/// Pinned feasibility tolerance: a bound counts as violated when
/// `n'w - b < -QP_FEAS_TOL * (1 + |b|)`.
pub const QP_FEAS_TOL: f64 = 1.0e-13;
/// Pinned tolerance below which a rounding-level violation of a constraint
/// that is linearly dependent on the active set is accepted as satisfied.
pub const QP_DEGENERATE_TOL: f64 = 1.0e-9;
/// Pinned KKT certificate tolerance (relative) for `converged`.
pub const QP_CONV_TOL: f64 = 1.0e-9;
/// Pinned relative conditioning threshold for the tangency denominator.
pub const TANGENCY_COND_RTOL: f64 = 1.0e-6;

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

fn validate_1x1(sigma: &Matrix) -> Result<()> {
    let v = sigma.get(0, 0);
    if !(v > 0.0) || !v.is_finite() {
        return Err(PortOptError::InvalidInput(
            "1x1 covariance must be positive and finite".into(),
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
        validate_1x1(sigma)?;
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
/// [`PortOptError::TangencyUndefined`] if
/// `A - rf*C <= TANGENCY_COND_RTOL * max(|A|, |rf|*C)`: the tangency
/// portfolio is undefined when the risk-free rate is at or above the
/// minimum-variance return `A/C` (in particular when `rf` exceeds every
/// expected return) and numerically meaningless when `rf` sits within a
/// relative 1e-6 of `A/C` (the weights scale like `1/(A - rf C)` and reach
/// gross leverage of 1e6+). [`PortOptError::InvalidInput`] on malformed
/// inputs (non-finite `rf`, ...).
pub fn tangency_weights(mu: &[f64], sigma: &Matrix, rf: f64) -> Result<Vec<f64>> {
    validate_mu_sigma(mu, sigma)?;
    if !rf.is_finite() {
        return Err(PortOptError::InvalidInput("rf must be finite".into()));
    }
    if sigma.nrows() == 1 {
        validate_1x1(sigma)?;
        if mu[0] <= rf {
            return Err(PortOptError::TangencyUndefined(
                "tangency undefined: risk-free rate >= expected return of the single asset"
                    .into(),
            ));
        }
        return Ok(vec![1.0]);
    }
    let low = cholesky(sigma)?;
    let excess: Vec<f64> = mu.iter().map(|v| v - rf).collect();
    let z = chol_solve(&low, &excess)?;
    let denom: f64 = z.iter().sum(); // = A - rf*C
    if denom <= 0.0 {
        return Err(PortOptError::TangencyUndefined(
            "tangency undefined: A - rf*C <= 0 (risk-free rate is at or above the \
             minimum-variance portfolio return)"
                .into(),
        ));
    }
    // Conditioning guard: A - rf*C is a difference of two O(A) numbers.
    let a: f64 = chol_solve(&low, mu)?.iter().sum();
    let c: f64 = chol_solve(&low, &vec![1.0; mu.len()])?.iter().sum();
    let scale = a.abs().max(rf.abs() * c);
    if denom <= TANGENCY_COND_RTOL * scale {
        return Err(PortOptError::TangencyUndefined(format!(
            "tangency ill-conditioned: A - rf*C = {denom:.3e} is within a relative \
             {TANGENCY_COND_RTOL:e} of cancelling (rf within {TANGENCY_COND_RTOL:e} of the \
             minimum-variance return {:.6}); weights would carry gross leverage of order \
             1e6 or more",
            a / c
        )));
    }
    Ok(z.iter().map(|v| v / denom).collect())
}

/// Gross exposure `sum_i |w_i|` (1.0 for any long-only portfolio); a desk
/// should cap it before unconstrained weights reach an OMS.
///
/// # Errors
/// [`PortOptError::InvalidInput`] on an empty or non-finite vector.
pub fn gross_leverage(w: &[f64]) -> Result<f64> {
    if w.is_empty() || !w.iter().all(|v| v.is_finite()) {
        return Err(PortOptError::InvalidInput(
            "weights must be a non-empty finite vector".into(),
        ));
    }
    Ok(kahan_sum(w.iter().map(|v| v.abs())))
}

/// Frontier generators `g`, `h` and the min-variance return `A/C` (n >= 2).
fn two_fund_gh(mu: &[f64], sigma: &Matrix) -> Result<(Vec<f64>, Vec<f64>, f64)> {
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
    let g = (0..mu.len()).map(|i| (b * y[i] - a * x[i]) / d).collect();
    let h = (0..mu.len()).map(|i| (c * x[i] - a * y[i]) / d).collect();
    Ok((g, h, a / c))
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
        validate_1x1(sigma)?;
        if (target - mu[0]).abs() > 1e-12 * (1.0 + mu[0].abs()) {
            return Err(PortOptError::Undefined(
                "n=1: target return must equal the single asset's expected return".into(),
            ));
        }
        return Ok(vec![1.0]);
    }
    let (g, h, mv_ret) = two_fund_gh(mu, sigma)?;
    if target < mv_ret - 1e-12 * (1.0 + mv_ret.abs()) {
        return Err(PortOptError::Undefined(format!(
            "target return {target:.6} lies below the minimum-variance return \
             {mv_ret:.6}; the efficient frontier is defined for targets >= A/C"
        )));
    }
    Ok((0..mu.len()).map(|i| g[i] + h[i] * target).collect())
}

/// One frontier sweep: pinned target grid plus per-point weights/variances.
#[derive(Debug, Clone)]
pub struct FrontierSweep {
    /// Target returns `lo + k*(hi - lo)/(n_points - 1)`, last point exactly `hi`.
    pub targets: Vec<f64>,
    /// Frontier weights per target (`n_points` rows).
    pub weights: Vec<Vec<f64>>,
    /// Portfolio variances `w' Sigma w` per target.
    pub variances: Vec<f64>,
}

fn frontier_grid(lo: f64, hi: f64, n_points: usize, what: &str) -> Result<Vec<f64>> {
    if hi <= lo + 1e-12 * (1.0 + lo.abs()) {
        return Err(PortOptError::Undefined(format!(
            "frontier grid empty: {what} ({hi:.6}) does not exceed the minimum-variance \
             return ({lo:.6}); supply targets explicitly via frontier_weights / \
             constrained_frontier_weights"
        )));
    }
    let mut targets: Vec<f64> = (0..n_points)
        .map(|k| lo + k as f64 * (hi - lo) / (n_points - 1) as f64)
        .collect();
    targets[n_points - 1] = hi;
    Ok(targets)
}

/// Unconstrained frontier sweep over the pinned grid: `n_points` equally
/// spaced targets from the minimum-variance return `A/C` to `max(mu)`;
/// the two-fund generators are computed once and reused.
///
/// # Errors
/// [`PortOptError::Undefined`] if `max(mu) <= A/C` (frontier grid empty).
pub fn efficient_frontier(mu: &[f64], sigma: &Matrix, n_points: usize) -> Result<FrontierSweep> {
    validate_mu_sigma(mu, sigma)?;
    if n_points < 2 {
        return Err(PortOptError::InvalidInput("n_points must be >= 2".into()));
    }
    if sigma.nrows() == 1 {
        return Err(PortOptError::InvalidInput(
            "frontier sweep requires at least 2 assets".into(),
        ));
    }
    let (g, h, lo) = two_fund_gh(mu, sigma)?;
    let hi = mu.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let targets = frontier_grid(lo, hi, n_points, "max(mu)")?;
    let mut weights = Vec::with_capacity(n_points);
    let mut variances = Vec::with_capacity(n_points);
    for &tgt in &targets {
        let w: Vec<f64> = (0..mu.len()).map(|i| g[i] + h[i] * tgt).collect();
        variances.push(dot(&w, &sigma.matvec(&w)));
        weights.push(w);
    }
    Ok(FrontierSweep { targets, weights, variances })
}

fn clip(v: f64, lo: f64, hi: f64) -> f64 {
    v.max(lo).min(hi)
}

/// Broadcast/validate bounds (API_SPEC 3.5 feasibility gate).
fn broadcast_bounds(
    n: usize,
    lb: Option<&[f64]>,
    ub: Option<&[f64]>,
    total: f64,
) -> Result<(Vec<f64>, Vec<f64>)> {
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
    if !lo.iter().all(|x| x.is_finite()) {
        return Err(PortOptError::InvalidInput("lower bounds must be finite".into()));
    }
    if hi.iter().any(|&x| x == f64::NEG_INFINITY) {
        return Err(PortOptError::InvalidInput("upper bounds must be > -inf".into()));
    }
    if lo.iter().zip(&hi).any(|(l, u)| l > u) {
        return Err(PortOptError::Infeasible("some lb_i > ub_i".into()));
    }
    let lo_sum = kahan_sum(lo.iter().copied());
    let hi_sum = kahan_sum(hi.iter().copied());
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
    Ok((lo, hi))
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
/// Lower bounds must be finite.
///
/// # Errors
/// [`PortOptError::Infeasible`] when `sum(lb) > total`, `sum(ub) < total`
/// or some `lb[i] > ub[i]`; [`PortOptError::InvalidInput`] on NaN/inf in
/// `v` or invalid bounds.
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
    let (lo, hi) = broadcast_bounds(n, lb, ub, total)?;

    let clipped = |t: f64| -> Vec<f64> {
        (0..n).map(|i| clip(v[i] - t, lo[i], hi[i])).collect()
    };
    let g = |t: f64| -> f64 { kahan_sum(clipped(t).into_iter()) };

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

/// Vertex of `{sum w = total, lb <= w <= ub}` maximizing (descending) or
/// minimizing (ascending) `mu'w`: fill from `lb` in `mu` order (ties by index).
fn greedy_extreme(mu: &[f64], lo: &[f64], hi: &[f64], total: f64, descending: bool) -> Vec<f64> {
    let n = mu.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        let ord = if descending {
            mu[b].partial_cmp(&mu[a]).unwrap()
        } else {
            mu[a].partial_cmp(&mu[b]).unwrap()
        };
        ord.then(a.cmp(&b))
    });
    let mut w = lo.to_vec();
    let mut remaining = total - kahan_sum(lo.iter().copied());
    if remaining < 0.0 {
        remaining = 0.0;
    }
    for &i in &order {
        if remaining <= 0.0 {
            break;
        }
        let room = hi[i] - lo[i];
        let add = if room > remaining { remaining } else { room };
        w[i] = lo[i] + add;
        remaining -= add;
    }
    w
}

/// Range `[r_min, r_max]` of `mu'w` over `{sum w = 1, lb <= w <= ub}`;
/// both extremes are vertices found by the pinned greedy fill (assets in
/// ascending / descending `mu` order, ties by index).
pub fn attainable_return_range(
    mu: &[f64],
    lb: Option<&[f64]>,
    ub: Option<&[f64]>,
) -> Result<(f64, f64)> {
    if mu.is_empty() || !mu.iter().all(|v| v.is_finite()) {
        return Err(PortOptError::InvalidInput(
            "mu must be a non-empty finite vector".into(),
        ));
    }
    let (lo, hi) = broadcast_bounds(mu.len(), lb, ub, 1.0)?;
    let w_min = greedy_extreme(mu, &lo, &hi, 1.0, false);
    let w_max = greedy_extreme(mu, &lo, &hi, 1.0, true);
    Ok((dot(mu, &w_min), dot(mu, &w_max)))
}

/// Output of the active-set QP solvers.
#[derive(Debug, Clone)]
pub struct QpResult {
    /// Feasible weights (bounds exact, equalities to rounding).
    pub weights: Vec<f64>,
    /// Dual active-set iterations (constraint additions/drops).
    pub iterations: usize,
    /// True iff the KKT conditions are certified at `weights` (primal
    /// feasibility and stationarity `S w = N u` within [`QP_CONV_TOL`];
    /// bound multipliers are non-negative by construction). False means
    /// the iteration cap was hit or the constraint set proved numerically
    /// inconsistent — the weights are then not certified optimal.
    pub converged: bool,
    /// Largest KKT violation at `weights`: equality residuals, bound
    /// violations and the stationarity residual `|S w - N u|_inf`.
    pub kkt_residual: f64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Equality,
    Lower,
    Upper,
    Fixed,
}

struct Constraint {
    kind: Kind,
    idx: usize,
    rhs: f64,
    normal: Vec<f64>,
}

/// Goldfarb-Idnani dual active-set method for `min 1/2 w'Sw` s.t.
/// `a_eq w = b_eq`, `lo <= w <= hi` (API_SPEC 3.6). Constraints are
/// `n_c'w >= b_c`: the equalities (rows of `a_eq`, then every fixed
/// coordinate `lo_i == hi_i`; added first, multipliers unrestricted), then
/// the lower bounds and the finite upper bounds of the non-fixed
/// coordinates.
fn dual_active_set_qp(
    s: &Matrix,
    a_eq: &[Vec<f64>],
    b_eq: &[f64],
    lo: &[f64],
    hi: &[f64],
    max_iter: usize,
) -> Result<QpResult> {
    let n = s.nrows();
    let low = cholesky(s)?;
    let unit = |i: usize, sign: f64| {
        let mut e = vec![0.0; n];
        e[i] = sign;
        e
    };
    let mut cons: Vec<Constraint> = Vec::new();
    for (j, row) in a_eq.iter().enumerate() {
        cons.push(Constraint { kind: Kind::Equality, idx: j, rhs: b_eq[j], normal: row.clone() });
    }
    for i in 0..n {
        if lo[i] == hi[i] {
            cons.push(Constraint { kind: Kind::Fixed, idx: i, rhs: lo[i], normal: unit(i, 1.0) });
        }
    }
    let meq = cons.len();
    for i in 0..n {
        if lo[i] != hi[i] {
            cons.push(Constraint { kind: Kind::Lower, idx: i, rhs: lo[i], normal: unit(i, 1.0) });
        }
    }
    for i in 0..n {
        if lo[i] != hi[i] && hi[i].is_finite() {
            cons.push(Constraint { kind: Kind::Upper, idx: i, rhs: -hi[i], normal: unit(i, -1.0) });
        }
    }
    let n_cons = cons.len();

    let mut w = vec![0.0; n];
    let mut active: Vec<usize> = Vec::new();
    let mut u: Vec<f64> = Vec::new();
    let mut skipped = vec![false; n_cons];
    let mut iterations = 0usize;
    let mut p: Option<usize> = None;
    let mut u_p = 0.0;
    while iterations < max_iter {
        iterations += 1;
        if p.is_none() {
            // (1) pick the next constraint: equalities first, then the most
            // violated inequality (ties -> lowest index).
            if active.len() < meq {
                p = Some(active.len());
            } else {
                let mut worst = 0.0;
                for c in meq..n_cons {
                    if skipped[c] || active.contains(&c) {
                        continue;
                    }
                    let viol = dot(&cons[c].normal, &w) - cons[c].rhs;
                    if viol < -QP_FEAS_TOL * (1.0 + cons[c].rhs.abs()) && viol < worst {
                        worst = viol;
                        p = Some(c);
                    }
                }
                if p.is_none() {
                    break; // no violated constraint: dual method finished
                }
            }
            u_p = 0.0;
        }
        let pc = p.unwrap();
        // (2) primal / dual step directions for constraint p.
        let q = active.len();
        let mut ys: Vec<Vec<f64>> = Vec::with_capacity(q);
        for &c in &active {
            ys.push(chol_solve(&low, &cons[c].normal)?);
        }
        let v = chol_solve(&low, &cons[pc].normal)?;
        let mut r = vec![0.0; q];
        if q > 0 {
            let mut gram = Matrix::zeros(q, q);
            for i in 0..q {
                for j in i..q {
                    let gij = dot(&cons[active[i]].normal, &ys[j]);
                    gram.set(i, j, gij);
                    gram.set(j, i, gij);
                }
            }
            let nv: Vec<f64> = active.iter().map(|&c| dot(&cons[c].normal, &v)).collect();
            r = chol_solve(&cholesky(&gram)?, &nv)?;
        }
        let mut z = v.clone();
        for i in 0..q {
            for k in 0..n {
                z[k] -= r[i] * ys[i][k];
            }
        }
        let zn = dot(&z, &cons[pc].normal);
        let vn = dot(&v, &cons[pc].normal);
        let s_p = dot(&cons[pc].normal, &w) - cons[pc].rhs;
        // (3) step lengths: t1 keeps inequality multipliers >= 0, t2 satisfies p.
        let mut t1 = f64::INFINITY;
        let mut j_block: Option<usize> = None;
        for i in 0..q {
            if active[i] >= meq && r[i] > 0.0 {
                let cand = u[i] / r[i];
                if cand < t1 {
                    t1 = cand;
                    j_block = Some(i);
                }
            }
        }
        let t2 = if zn > 1e-10 * vn { -s_p / zn } else { f64::INFINITY };
        let t = t1.min(t2);
        if !t.is_finite() {
            // n_p is a combination of the active normals and no multiplier
            // blocks: accept a rounding-level violation at a primal-degenerate
            // vertex as satisfied, else give up (reported via the certificate).
            if s_p.abs() <= QP_DEGENERATE_TOL * (1.0 + cons[pc].rhs.abs()) {
                skipped[pc] = true;
                p = None;
                continue;
            }
            break;
        }
        if !t2.is_finite() {
            // Dependent normal: dual step only, drop the blocking constraint.
            for i in 0..q {
                u[i] -= t * r[i];
            }
            u_p += t;
            let j = j_block.unwrap();
            active.remove(j);
            u.remove(j);
            continue;
        }
        for k in 0..n {
            w[k] += t * z[k];
        }
        for i in 0..q {
            u[i] -= t * r[i];
        }
        u_p += t;
        if t == t2 {
            active.push(pc);
            u.push(u_p);
            p = None;
        } else {
            let j = j_block.unwrap();
            active.remove(j);
            u.remove(j);
        }
    }
    // Polish: minimizer on the final active set in closed form,
    // w = S^-1 N u with (N' S^-1 N) u = b_active, plus one step of iterative
    // refinement on the equality residual.
    let q = active.len();
    if q > 0 {
        let mut ys: Vec<Vec<f64>> = Vec::with_capacity(q);
        for &c in &active {
            ys.push(chol_solve(&low, &cons[c].normal)?);
        }
        let mut gram = Matrix::zeros(q, q);
        for i in 0..q {
            for j in i..q {
                let gij = dot(&cons[active[i]].normal, &ys[j]);
                gram.set(i, j, gij);
                gram.set(j, i, gij);
            }
        }
        let low_g = cholesky(&gram)?;
        let b_act: Vec<f64> = active.iter().map(|&c| cons[c].rhs).collect();
        let u_arr = chol_solve(&low_g, &b_act)?;
        w = vec![0.0; n];
        for i in 0..q {
            for k in 0..n {
                w[k] += u_arr[i] * ys[i][k];
            }
        }
        let resid: Vec<f64> = (0..q)
            .map(|i| b_act[i] - dot(&cons[active[i]].normal, &w))
            .collect();
        let du = chol_solve(&low_g, &resid)?;
        for i in 0..q {
            for k in 0..n {
                w[k] += du[i] * ys[i][k];
            }
        }
        u = (0..q).map(|i| u_arr[i] + du[i]).collect();
    }
    for &c in &active {
        match cons[c].kind {
            Kind::Lower | Kind::Fixed => w[cons[c].idx] = lo[cons[c].idx],
            Kind::Upper => w[cons[c].idx] = hi[cons[c].idx],
            Kind::Equality => {}
        }
    }
    // KKT certificate on the polished point.
    let mut feas = 0.0_f64;
    let mut rhs_scale = 0.0_f64;
    for (c, con) in cons.iter().enumerate() {
        let viol = dot(&con.normal, &w) - con.rhs;
        rhs_scale = rhs_scale.max(con.rhs.abs());
        feas = feas.max(if c < meq { viol.abs() } else { -viol });
    }
    let mut stat = s.matvec(&w);
    let mut stat_scale = stat.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    for i in 0..q {
        let con = &cons[active[i]];
        for k in 0..n {
            stat[k] -= u[i] * con.normal[k];
        }
        stat_scale += u[i].abs() * con.normal.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    }
    let stat_res = stat.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let converged =
        feas <= QP_CONV_TOL * (1.0 + rhs_scale) && stat_res <= QP_CONV_TOL * (1.0 + stat_scale);
    for k in 0..n {
        w[k] = clip(w[k], lo[k], hi[k]);
    }
    Ok(QpResult { weights: w, iterations, converged, kkt_residual: feas.max(stat_res) })
}

/// Shared QP setup: square + SPD gate, bound validation, iteration cap.
fn qp_setup(
    sigma: &Matrix,
    lb: Option<&[f64]>,
    ub: Option<&[f64]>,
    max_iter: usize,
) -> Result<(Vec<f64>, Vec<f64>)> {
    if sigma.nrows() != sigma.ncols() {
        return Err(PortOptError::InvalidInput(format!(
            "sigma must be square, got shape {}x{}",
            sigma.nrows(),
            sigma.ncols()
        )));
    }
    cholesky(sigma)?; // SPD/validity gate
    if max_iter < 1 {
        return Err(PortOptError::InvalidInput("max_iter must be >= 1".into()));
    }
    broadcast_bounds(sigma.nrows(), lb, ub, 1.0)
}

fn require_converged(res: QpResult, what: &str) -> Result<Vec<f64>> {
    if !res.converged {
        return Err(PortOptError::Undefined(format!(
            "{what}: active-set QP did not converge within {} iterations (KKT residual \
             {:.3e}); the input is degenerate — use the *_solve variant to inspect the iterate",
            res.iterations, res.kkt_residual
        )));
    }
    Ok(res.weights)
}

/// Constrained minimum variance `min 1/2 w' Sigma w` over
/// `{sum w = 1, lb <= w <= ub}` (defaults: long-only, `lb = 0`,
/// `ub = +inf`), solved by the pinned dual active-set QP. Strictly convex:
/// the minimizer is unique and the result is its KKT point up to rounding.
pub fn min_variance_longonly_solve(
    sigma: &Matrix,
    lb: Option<&[f64]>,
    ub: Option<&[f64]>,
    max_iter: usize,
) -> Result<QpResult> {
    let (lo, hi) = qp_setup(sigma, lb, ub, max_iter)?;
    let n = sigma.nrows();
    dual_active_set_qp(sigma, &[vec![1.0; n]], &[1.0], &lo, &hi, max_iter)
}

/// Weights of [`min_variance_longonly_solve`] with the pinned iteration cap.
///
/// # Errors
/// [`PortOptError::Undefined`] if the solver did not certify convergence.
pub fn min_variance_longonly(
    sigma: &Matrix,
    lb: Option<&[f64]>,
    ub: Option<&[f64]>,
) -> Result<Vec<f64>> {
    require_converged(
        min_variance_longonly_solve(sigma, lb, ub, QP_MAX_ITER)?,
        "min_variance_longonly",
    )
}

#[allow(clippy::too_many_arguments)]
fn frontier_point(
    mu: &[f64],
    sigma: &Matrix,
    tgt: f64,
    lo: &[f64],
    hi: &[f64],
    w_gmv: &[f64],
    r_gmv: f64,
    r_min: f64,
    r_max: f64,
    max_iter: usize,
) -> Result<QpResult> {
    let tol = 1e-12 * (1.0 + tgt.abs());
    if tgt < r_gmv - tol {
        return Err(PortOptError::Undefined(format!(
            "target return {tgt:.6} lies below the constrained minimum-variance return \
             {r_gmv:.6}; the constrained efficient frontier is defined for targets in \
             [{r_gmv:.6}, {r_max:.6}]"
        )));
    }
    if tgt > r_max + tol {
        return Err(PortOptError::Undefined(format!(
            "target return {tgt:.6} is not attainable under the bounds (attainable range \
             [{r_min:.6}, {r_max:.6}])"
        )));
    }
    if (tgt - r_gmv).abs() <= tol || r_max - r_min <= tol {
        return Ok(QpResult { weights: w_gmv.to_vec(), iterations: 0, converged: true, kkt_residual: 0.0 });
    }
    let n = mu.len();
    if (tgt - r_max).abs() <= tol {
        // Maximum-return endpoint: reduce the degenerate face analytically
        // (assets above the greedy cut level at their caps, below it at their
        // floors, tied assets free) and solve the sum-only min-variance QP.
        let w_max = greedy_extreme(mu, lo, hi, 1.0, true);
        let cut = (0..n)
            .filter(|&i| w_max[i] > lo[i])
            .map(|i| mu[i])
            .fold(f64::INFINITY, f64::min);
        if !cut.is_finite() {
            return Ok(QpResult { weights: w_gmv.to_vec(), iterations: 0, converged: true, kkt_residual: 0.0 });
        }
        let mut lo2 = lo.to_vec();
        let mut hi2 = hi.to_vec();
        for i in 0..n {
            if mu[i] > cut {
                lo2[i] = hi[i];
            } else if mu[i] < cut {
                hi2[i] = lo[i];
            }
        }
        let mut res = dual_active_set_qp(sigma, &[vec![1.0; n]], &[1.0], &lo2, &hi2, max_iter)?;
        res.kkt_residual = res.kkt_residual.max((dot(mu, &res.weights) - tgt).abs());
        return Ok(res);
    }
    dual_active_set_qp(sigma, &[vec![1.0; n], mu.to_vec()], &[1.0, tgt], lo, hi, max_iter)
}

/// Constrained frontier point: `min 1/2 w' Sigma w` subject to
/// `sum w = 1`, `mu'w = target` and `lb <= w <= ub`. The target is met
/// exactly (to rounding) — there is no penalty term.
///
/// # Errors
/// [`PortOptError::Undefined`] if the target lies below the constrained
/// minimum-variance return (inefficient branch) or above the maximum
/// attainable return under the bounds.
pub fn constrained_frontier_solve(
    mu: &[f64],
    sigma: &Matrix,
    target: f64,
    lb: Option<&[f64]>,
    ub: Option<&[f64]>,
    max_iter: usize,
) -> Result<QpResult> {
    validate_mu_sigma(mu, sigma)?;
    let (lo, hi) = qp_setup(sigma, lb, ub, max_iter)?;
    if !target.is_finite() {
        return Err(PortOptError::InvalidInput("target return must be finite".into()));
    }
    let w_gmv = require_converged(
        min_variance_longonly_solve(sigma, Some(&lo), Some(&hi), max_iter)?,
        "constrained_frontier_solve (min-variance point)",
    )?;
    let (r_min, r_max) = attainable_return_range(mu, Some(&lo), Some(&hi))?;
    frontier_point(mu, sigma, target, &lo, &hi, &w_gmv, dot(mu, &w_gmv), r_min, r_max, max_iter)
}

/// Weights of [`constrained_frontier_solve`] with the pinned iteration cap.
///
/// # Errors
/// [`PortOptError::Undefined`] if the solver did not certify convergence.
pub fn constrained_frontier_weights(
    mu: &[f64],
    sigma: &Matrix,
    target: f64,
    lb: Option<&[f64]>,
    ub: Option<&[f64]>,
) -> Result<Vec<f64>> {
    require_converged(
        constrained_frontier_solve(mu, sigma, target, lb, ub, QP_MAX_ITER)?,
        "constrained_frontier_weights",
    )
}

/// Constrained frontier sweep on the pinned grid from the constrained
/// minimum-variance return to the maximum attainable return under the
/// bounds (`n_points` equally spaced targets, both ends inclusive). Every
/// point is solved to KKT optimality; a non-converged point is an error.
pub fn constrained_frontier(
    mu: &[f64],
    sigma: &Matrix,
    n_points: usize,
    lb: Option<&[f64]>,
    ub: Option<&[f64]>,
) -> Result<FrontierSweep> {
    validate_mu_sigma(mu, sigma)?;
    if n_points < 2 {
        return Err(PortOptError::InvalidInput("n_points must be >= 2".into()));
    }
    let (lo, hi) = qp_setup(sigma, lb, ub, QP_MAX_ITER)?;
    let w_gmv = require_converged(
        min_variance_longonly_solve(sigma, Some(&lo), Some(&hi), QP_MAX_ITER)?,
        "constrained_frontier (min-variance point)",
    )?;
    let r_gmv = dot(mu, &w_gmv);
    let (r_min, r_max) = attainable_return_range(mu, Some(&lo), Some(&hi))?;
    let targets = frontier_grid(r_gmv, r_max, n_points, "maximum attainable return")?;
    let mut weights = Vec::with_capacity(n_points);
    let mut variances = Vec::with_capacity(n_points);
    for (k, &tgt) in targets.iter().enumerate() {
        let res = frontier_point(mu, sigma, tgt, &lo, &hi, &w_gmv, r_gmv, r_min, r_max, QP_MAX_ITER)?;
        let w = require_converged(res, &format!("constrained_frontier point {k} (target {tgt:.6})"))?;
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
        assert_eq!(min_variance_longonly(&sig, None, None).unwrap(), vec![1.0]);
        assert!(frontier_weights(&[0.05], &sig, 0.06).is_err());
        assert!(matches!(
            tangency_weights(&[0.05], &sig, 0.10),
            Err(PortOptError::TangencyUndefined(_))
        ));
        let bad = Matrix::from_rows(&[vec![-1.0]]).unwrap();
        assert!(frontier_weights(&[0.05], &bad, 0.05).is_err());
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
        assert!(matches!(
            project_simplex_box(&[0.0, 0.0], 1.0, Some(&[f64::NEG_INFINITY, 0.0]), None),
            Err(PortOptError::InvalidInput(_))
        ));
    }
}
