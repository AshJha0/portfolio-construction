# portopt API_SPEC — language-neutral numerical contract

This document pins every algorithm so that scipy/LAPACK-free ports
(C++ `portopt`, Rust `portopt`, Java `com.quant.portopt`) reproduce the
Python reference and the golden values in `data/golden/golden.json`.
All arithmetic is IEEE-754 double precision. Indices below are 0-based.
Unless stated otherwise, "error" means: Python `ValueError`,
C++ `std::invalid_argument`/`std::domain_error`, Rust `Err(PortOptError)`,
Java `IllegalArgumentException`, each with a descriptive message.

Notation: `T` = number of return rows, `N`/`n` = number of assets,
`1` = vector of ones, `'` = transpose.

**Reductions.** The Python reference sums every T-length quantity (means,
covariances, shrinkage dispersion sums, backtest metrics, turnover) with
correctly rounded `math.fsum`; the ports use a Neumaier-compensated sum.
The two agree to a couple of ulps, so no golden value depends on a BLAS
or on a summation order. Short (n ≤ 20) products use plain sequential
loops everywhere. Every golden quantity is either a closed form or the
exact solution of a finite algorithm, so cross-language agreement is a
rounding question (~1e-13), never an iteration-path question.

---

## 1. Data files

- `data/asset_returns.csv` — header row of 8 asset names
  (`US_EQ,EU_EQ,JP_EQ,UK_EQ,EM_EQ,US_BOND,IG_CORP,FX_CARRY`), then 1250
  rows of 8 comma-separated daily simple returns printed with `%.17g`
  (exact double round-trip). No date column. Golden values are defined on
  the file **as parsed**, not on the generator's in-memory values.
  Parsers must be locale-independent and tolerate CRLF line endings.
- `data/market_caps.csv` — header `asset,market_cap`, 8 rows.
  `w_mkt[i] = cap[i] / sum(cap)`.
- `data/views.json` — `{"tau": 0.05, "delta": 2.5, "views": [...]}` with
  view objects per section 4.2.
- `data/config.json` — constants: `annualization_factor = 252`,
  `risk_free_rate = 0.02`, `risk_aversion_delta = 2.5`, `tau = 0.05`,
  `estimation_window = 500`, `rebalance_step = 63`,
  `frontier_points = 25`.

Annualization (pinned): `mu_ann = 252 * mu_daily`,
`Sigma_ann = 252 * Sigma_daily`. `periods` must be positive and finite
(error otherwise). All portfolio-level golden cases use the
**annualized sample** moments unless the case name says `daily`.

Return validation (every entry point that takes a returns panel or a
return series): entries must be finite and **strictly greater than -1**
(a simple return at or below -100% is not a valid price move); `T >= 1`
for means, `T >= 2` for covariances; `N >= 1`.

---

## 2. Estimators

### 2.1 Sample moments
- Mean: `mu[i] = (1/T) * sum_t X[t][i]` (T >= 1).
- Covariance (ddof = 1, T >= 2): demean columns
  (`xc[t][i] = X[t][i] - mu[i]`), then
  `S[i][j] = (1/(T-1)) * sum_t xc[t][i] * xc[t][j]`.

### 2.2 Ledoit-Wolf constant-correlation shrinkage (2004), pinned term by term
Requires N >= 2 and `S[i][i] > 0` for all i. The intensity is computed
**exactly as in Ledoit & Wolf (2004), Appendix B (the authors' `covCor.m`)**,
i.e. with the paper's divisor-T sample covariance. With `xc` from 2.1 and
`S_T = (T-1)/T * S` (equivalently `S_T[i][j] = (1/T) sum_t xc[t][i] xc[t][j]`),
`s_ii = S_T[i][i]`:

1. Average correlation:
   `rbar = (2 / (N*(N-1))) * sum_{i<j} S_T[i][j] / sqrt(s_ii * s_jj)`.
2. Paper target: `F_T[i][i] = s_ii`; `F_T[i][j] = rbar * sqrt(s_ii * s_jj)`.
3. `pi_ij = (1/T) * sum_t (xc[t][i]*xc[t][j] - S_T[i][j])^2`;
   `pi_hat = sum over all ordered pairs (i,j) of pi_ij` (diagonal included).
4. `theta_ii,ij = (1/T) * sum_t (xc[t][i]^2 - s_ii) * (xc[t][i]*xc[t][j] - S_T[i][j])`.
5. `rho_hat = sum_i pi_ii + sum over ordered pairs i != j of
   rbar * sqrt(s_jj / s_ii) * theta_ii,ij`.
   This is the paper's `sum_{i≠j} (rbar/2) [sqrt(s_jj/s_ii) theta_ii,ij +
   sqrt(s_ii/s_jj) theta_jj,ij]`: each unordered pair contributes **both**
   theta terms, so the weight per ordered pair is `rbar`, not `rbar/2`.
6. `gamma_hat = sum over all ordered pairs (i,j) of (F_T[i][j] - S_T[i][j])^2`.
7. `kappa = (pi_hat - rho_hat) / gamma_hat`;
   `delta = max(0, min(1, kappa / T))`; if `gamma_hat <= 0` then `delta = 0`
   (for N = 2 the target equals the sample covariance, so `delta = 0`).
8. Output in the project-wide ddof = 1 convention: `sample = S`,
   `target = F` with `F[i][i] = S[i][i]`, `F[i][j] = rbar * sqrt(S[i][i] S[j][j])`
   (= `T/(T-1) F_T`), and `Sigma_shrunk = delta * F + (1 - delta) * S`
   (= `T/(T-1)` times the paper's estimator). `rbar` and `delta` are
   scale-free, hence identical to the paper's.

Validated against a direct transcription of `covCor.m` to 1e-14 on the
bundled panel (`delta = 0.0072117652`) and on hand panels (PT-1).

---

## 3. Linear algebra and Markowitz

### 3.1 Cholesky (the only factorization used anywhere)
Variant: **Cholesky-Banachiewicz, lower triangular, computed row by row**.
For `i = 0..n-1`, first require `A[i][i] > 0`; for `j = 0..i`:
`s = A[i][j] - sum_{k<j} L[i][k]*L[j][k]`; if `i == j`:
require `s > PIVOT_RTOL * A[i][i]` with pinned `PIVOT_RTOL = 1e-13`
(else error "not positive definite" — this is the detector for singular
covariance, e.g. a duplicated asset, and for indefinite inputs), then
`L[i][i] = sqrt(s)`; else `L[i][j] = s / L[j][j]`.
Symmetry gate first: error if `max|A - A'| > 1e-10 * (1 + max|A|)`;
error on NaN/inf or non-square input.

### 3.2 SPD solve and inverse
`solve(A, b)`: factor `A = L L'`; forward substitution `L y = b` with
rows ascending; back substitution `L' x = y` with rows descending.
**Every matrix inverse in this project is realized as Cholesky solves**;
`inverse(A)` = solve against the identity columns `e_0..e_{n-1}` stacked
as columns (no post-symmetrization).

### 3.3 Two-fund theorem (closed forms)
With `x = solve(Sigma, mu)` and `y = solve(Sigma, 1)`:

```
A = sum(x)        ( = 1' Sigma^-1 mu )
B = mu . x        ( = mu' Sigma^-1 mu )
C = sum(y)        ( = 1' Sigma^-1 1 )
D = B*C - A*A
```

- Min-variance: `w_mv = y / C`; return `A/C`, variance `1/C`.
  For `n == 1`: `w = [1.0]` (requires `Sigma[0][0] > 0` and finite).
- Tangency: `z = solve(Sigma, mu - rf*1)`, `w_tan = z / sum(z)` where
  `sum(z) = A - rf*C`. `rf` must be finite. **Tangency error** (a dedicated
  type: Python `TangencyUndefinedError(ValueError)`, C++
  `TangencyUndefinedError : std::domain_error`, Rust
  `PortOptError::TangencyUndefined`, Java
  `Markowitz.TangencyUndefinedException extends IllegalArgumentException`)
  if `A - rf*C <= 0` ("tangency undefined"; covers rf above every expected
  return) **or** if `A - rf*C <= TANGENCY_COND_RTOL * max(|A|, |rf|*C)`
  with pinned `TANGENCY_COND_RTOL = 1e-6` ("tangency ill-conditioned": rf
  within one part per million of the min-variance return `A/C`, where the
  weights scale like `1/(A - rf C)` and carry gross leverage of 1e6+).
  For `n == 1`: `[1.0]` if `mu[0] > rf`, else tangency error.
- Gross leverage diagnostic: `gross_leverage(w) = sum_i |w_i|` (error on
  empty/non-finite input). Unconstrained tangency and BL weights should be
  capped on this number before they reach an OMS.
- Frontier at target m: error if `D <= 1e-12 * B * C` ("degenerate
  frontier", mu proportional to 1); error if
  `m < A/C - 1e-12*(1+|A/C|)` (target below the min-variance return,
  includes negative targets); else
  `g = (B*y - A*x)/D`, `h = (C*x - A*y)/D`, `w(m) = g + h*m`.
  For `n == 1`: `[1.0]` if `|m - mu[0]| <= 1e-12*(1+|mu[0]|)` and
  `Sigma[0][0] > 0`, else error.
- Frontier variance identity (used by tests):
  `var(m) = (C*m^2 - 2*A*m + B) / D`.

### 3.4 Frontier sweeps (pinned grids)
`n_points = 25` targets: `target_k = lo + k*(hi - lo)/(n_points - 1)`,
`k = 0..24`, and **`target_24 = hi` exactly** (not the rounded expression).
Error "frontier grid empty" if `hi <= lo + 1e-12*(1+|lo|)`.

- Unconstrained sweep: `lo = A/C`, `hi = max(mu)`. `g`, `h` are computed
  **once** (one Cholesky, two solves) and reused for every point. The grid
  is empty when the GMV portfolio shorts a low-return asset so that
  `A/C >= max(mu)`; explicit targets above `A/C` still work through the
  point function.
- Constrained sweep (3.6): `lo = mu'w_gmv` with `w_gmv` the constrained
  minimum-variance portfolio under the same bounds, `hi = r_max` the
  maximum attainable return under the bounds (3.5b). Every point is the
  exact minimizer at its target (equality, no penalty); a point whose QP
  certificate fails is an error, never a silent iterate.

### 3.5 Projection onto the box-constrained simplex (exact, sort-based)
Project `v` onto `{w : sum(w) = total, lb <= w <= ub}`; defaults
`lb = 0`, `ub = +inf`, `total = 1`.
Feasibility gate (errors): any `lb[i] > ub[i]`;
`sum(lb) > total + 1e-12*(1+|total|)`; `sum(ub) < total - 1e-12*(1+|total|)`;
NaN/inf in `v`; NaN in bounds; non-finite `lb` (lower bounds must be
finite; upper bounds may be `+inf`).

The solution is `w[i] = clip(v[i] - t, lb[i], ub[i])` for the unique
shift `t` with `g(t) = sum_i clip(v[i] - t, lb[i], ub[i]) = total`;
`g` is piecewise linear and non-increasing. Pinned finite algorithm:

1. Collect breakpoints: all `v[i] - lb[i]`, plus `v[i] - ub[i]` for every
   finite `ub[i]`. Sort ascending -> `t_1 <= ... <= t_m`.
2. If `g(t_1) < total` (t* left of all breakpoints): let `k` = number of
   infinite upper bounds. If `k == 0`, take `t = t_1` (feasibility makes
   the gap pure rounding). Else `t = t_1 - (total - g(t_1)) / k`.
3. Otherwise scan `t_2..t_m` for the first `t_c` with `g(t_c) <= total`,
   keeping the predecessor `t_p`. Interpolate on the linear segment:
   `t = t_p + (g(t_p) - total) * (t_c - t_p) / (g(t_p) - g(t_c))`
   (if `g(t_p) == g(t_c)`, take `t = t_c`).
4. If the scan exhausts (only possible when `sum(lb) = total` up to
   rounding), take `t = t_m`.
5. Return `clip(v - t, lb, ub)`.

Convergence: the algorithm is **finite and exact** (one sort + one linear
interpolation; no iteration tolerance). With `lb=0, ub=+inf` it agrees
with the classic Duchi et al. sorted simplex projection (golden case
`simplex_projection_hand`: v = [0.9, 0.5, -0.2] -> [0.7, 0.3, 0.0]).

#### 3.5b Attainable return range (pinned greedy fill)
`attainable_return_range(mu, lb, ub) = [r_min, r_max]` over
`{sum w = 1, lb <= w <= ub}`. `r_max = mu'w_max` where `w_max` starts at
`lb`, then visits assets in **descending `mu` order (ties by ascending
index)** adding `min(remaining, ub_i - lb_i)` to each until the remaining
budget `1 - sum(lb)` is exhausted; `r_min` symmetrically in ascending
order. Both extremes are vertices of the feasible polytope.

### 3.6 Constrained QP: dual active-set method (finite, deterministic)
Used for constrained min-variance and every constrained frontier point:

```
min  1/2 w' Sigma w
s.t. 1'w = 1,  [mu'w = m for a frontier point],  lb <= w <= ub
```

Algorithm (pinned): the dual active-set method of Goldfarb & Idnani
(1983). Constraints are written `n_c'w >= b_c` and ordered as:
(i) the equality rows (`1'w = 1`, then `mu'w = m`), (ii) every fixed
coordinate `lb_i == ub_i` as the equality `w_i = lb_i`, (iii) the lower
bounds `w_i >= lb_i` of non-fixed coordinates in index order, (iv) the
finite upper bounds `-w_i >= -ub_i` in index order. Equalities have
unrestricted multipliers and are added first, in order.

1. Factor `Sigma = L L'` once (3.1 — the SPD gate). Start at the
   unconstrained minimizer `w = 0`, empty active set, multipliers `u = ()`.
2. Pick the next constraint `p`: the next equality if any remains; else
   the most violated inequality, `n_p'w - b_p < -1e-13*(1+|b_p|)`
   (ties -> lowest index); if none, stop.
3. With active normals `N` (columns), compute `Y = Sigma^-1 N` (solves),
   the Gram matrix `G = N'Y` (symmetrized), `v = Sigma^-1 n_p`,
   `r = G^-1 (N'v)` (Cholesky solve of the small SPD `G`), the primal
   direction `z = v - Y r`, and `s_p = n_p'w - b_p`.
4. Step lengths: `t1 = min over active inequalities with r_j > 0 of
   u_j / r_j` (+inf if none); `t2 = -s_p / (z'n_p)` if
   `z'n_p > 1e-10 * v'n_p`, else `+inf` (n_p dependent on N).
   `t = min(t1, t2)`.
   - `t = +inf`: n_p is a combination of the active normals and nothing
     blocks. In exact arithmetic the constraint set would be inconsistent;
     for the feasible sets used here this only happens at a
     primal-degenerate vertex with a rounding-level violation. If
     `|s_p| <= 1e-9*(1+|b_p|)` mark p satisfied and go to 2, else stop.
   - `t2 = +inf` (dual step): `u -= t r`, `u_p += t`, drop the blocking
     constraint, repeat 3 with the same p.
   - else `w += t z`, `u -= t r`, `u_p += t`; if `t == t2` add p with
     multiplier `u_p` and go to 2, otherwise drop the blocking constraint
     and repeat 3 with the same p.
5. Iteration cap `QP_MAX_ITER = 1000` (each iteration adds or drops one
   constraint; the method is finite for a strictly convex objective, so the
   cap is only reachable on numerically inconsistent inputs).
6. **Polish**: on the final active set solve the equality-constrained
   problem in closed form, `u = G^-1 b_active`, `w = Y u`, plus one step of
   iterative refinement on the equality residual; set active bounds
   exactly to their bound values; clip rounding-level violations of
   inactive bounds. The answer is therefore an exact function of the
   active set, which is what makes the ports agree to rounding.
7. **Certificate**: `converged` is true iff every equality residual and
   bound violation is `<= 1e-9*(1+max|b|)` and the stationarity residual
   `|Sigma w - N u|_inf <= 1e-9*(1 + |Sigma w|_inf + sum |u_c| |n_c|_inf)`
   (bound multipliers are non-negative by construction). `kkt_residual`
   reports the largest of those violations. The weights-only wrappers
   (`min_variance_longonly`, `constrained_frontier_weights`, the sweep)
   **raise** when the certificate fails; the `*_solve` variants return the
   flag and residual.

Frontier point pre-checks (before the QP): `m` finite; error if
`m < r_gmv - 1e-12*(1+|m|)` (below the constrained min-variance return,
inefficient branch) or `m > r_max + 1e-12*(1+|m|)` (not attainable);
if `|m - r_gmv|` is within that tolerance, or `r_max - r_min` is, the
answer is `w_gmv` (target row redundant); if `|m - r_max|` is within it,
the degenerate max-return face is reduced analytically — assets with
`mu_i` above the greedy cut level (the smallest `mu` among assets that
received budget in `w_max`) are fixed at `ub_i`, assets below it at
`lb_i`, tied assets stay free — and the sum-only QP is solved on that face.

Both objectives are strictly convex with a unique minimizer, and the
returned point is that minimizer up to rounding: ports agree to ~1e-13
(golden tolerance 1e-10), and scaling the input returns by `1 + 2^-52`
moves the golden values by < 1e-13 (tested in all four languages).

---

## 4. Black-Litterman

### 4.1 Reverse optimization
`pi = delta * Sigma * w_mkt` with `w_mkt` renormalized to sum 1.
Errors: `delta <= 0` or non-finite; negative cap weights; cap sum <= 0.
`pi` are **excess** returns: tangency at `rf = 0` on `(pi, Sigma)`
returns exactly `w_mkt`.

### 4.2 Views
`views.json` schema — each view is either
`{"type": "absolute", "asset": A, "q": x, "confidence": c}`
(P row = unit vector of A) or
`{"type": "relative", "long": A, "short": B, "q": x, "confidence": c}`
(P row = +1 at A, -1 at B). `confidence` defaults to 1 and must be > 0.
Errors (always the language's invalid-input error, never a key/index
error): empty view list; empty or duplicate `asset_names`; unknown type;
missing `asset`/`long`/`short`/`q`; unknown asset name; a relative view
whose legs coincide (`long == short`); non-finite `q`; any P row
identically zero; Q length != number of rows of P.

### 4.3 Omega default (He-Litterman, scaled by a confidence divisor)
`Omega` is diagonal with
`Omega[k][k] = [P (tau*Sigma) P'][k][k] / c_k` (`c_k` = per-view
confidence, > 0). `c -> inf`: dogmatic view; `c -> 0`: view ignored.
This is the He & Litterman (1999) default `Omega = diag(P tau Sigma P')`
with an ad-hoc per-view divisor — it is **not** Idzorek's (2005)
calibration (which solves for Omega such that the posterior tilt is a
stated fraction of the full-confidence tilt).

### 4.4 Posterior (master formula, pinned computational form)
`tau` must satisfy `0 < tau <= 1` (error otherwise). With
`TS = tau*Sigma`, `Bm = P*TS` (k x n), `A_k = Bm*P' + Omega`:

```
mu_post = pi + Bm' * solve(A_k, Q - P*pi)
M       = TS - Bm' * A_k^-1 * Bm      (A_k^-1 * Bm via n Cholesky solves, one per column)
S_post  = Sigma + M
```

`A_k` is factored **once** by the Cholesky of 3.1 (error if not SPD).
An explicit `Omega` and `confidences` may **not** be combined (error): the
confidences only parametrize the default Omega of 4.3.
This is algebraically the textbook
`[(tau S)^-1 + P' O^-1 P]^-1 [(tau S)^-1 pi + P' O^-1 Q]` but is the
pinned implementation. Limits (tested, not golden): fixed Omega and
`tau -> 0` gives `mu_post -> pi`, `M -> 0`; `Omega -> inf` (or `c -> 0`)
gives `mu_post -> pi`, `M -> tau*Sigma`. Note the default Omega of 4.3
scales with tau, making the posterior tau-invariant — the tau limit must
be taken at fixed Omega. Conflicting extreme views keep the posterior
finite (A_k stays SPD).

### 4.5 BL optimal weights
Pinned: unconstrained tangency (3.3) on `(mu_post, S_post)` with
`rf = 0` (posterior returns are excess returns; a non-zero rf would
double-count the risk-free rate). `rf` must be finite; the tangency
conditioning guard of 3.3 applies. Every port exposes the same surface:
`bl_weights(sigma, pi, P, Q, tau, rf=0, omega=None, confidences=None)`
(C++: `bl_weights(..., rf, confidences, result*)` and
`bl_weights_with_omega(..., omega, rf, result*)`).

---

## 5. Risk parity / risk budgeting (CCD)

Budgets `b` must be positive and finite (error otherwise); they are
normalized internally to sum 1; `b = (1/n, ..., 1/n)` is ERC.
Sigma is validated via Cholesky (3.1) — singular/non-PSD input errors.

Log-barrier formulation (Spinu 2013): minimize over `y > 0`
`F(y) = 0.5 * y' Sigma y - sum_i b[i] * ln y[i]`.
Coordinate stationarity with `m_i = (Sigma y)[i] - Sigma[i][i]*y[i]`
gives `Sigma[i][i]*y_i^2 + m_i*y_i - b_i = 0`, hence the pinned update
(positive root; discriminant > 0 always):

```
y[i] = ( -m_i + sqrt(m_i^2 + 4*Sigma[i][i]*b[i]) ) / (2*Sigma[i][i])
```

Pinned iteration: initialize `y[i] = 1/n`; cache `sy = Sigma y` and
update it incrementally after each coordinate change
(`sy += Sigma[:,i] * dy`). Sweep coordinates in ascending order
`i = 0..n-1`. After each full sweep, converged when
`max_i |y[i] - y_prev_sweep[i]| < 1e-10`; at most `10000` sweeps;
non-convergence is reported via a flag, never raised. `tol > 0` and
`max_sweeps >= 1` are validated.
Output weights: `w = y / sum(y)`.

At the fixed point `y_i * (Sigma y)_i = b_i` exactly, so the normalized
risk contributions `RC_i = w_i (Sigma w)_i / (w' Sigma w)` equal `b_i`
(this identity is a golden case: `erc_weights.rc_spread`). The weights
are invariant to the scale of Sigma and equivariant under asset
permutation (both tested). Edge case: an almost-zero-vol asset gets
weight near 1 (correct ERC limit); the update stays well-defined because
`Sigma[i][i] > 0` after the Cholesky gate. Inverse-vol comparison
portfolio: `w[i] = (1/sqrt(Sigma[i][i])) / sum_j (1/sqrt(Sigma[j][j]))`.

---

## 6. Backtest-lite

### 6.1 Schedule (pinned)
Estimation window `W = 500`, step `63` trading days. Rebalance row
indices: `t_k = W + 63*k` for `k = 0, 1, ...` while `t_k < T` (for
T = 1250: 12 dates, 500..1193). `rebalance_indices` requires `W >= 2`,
`step >= 1`, `T > W`; the advance is overflow-safe for any step (a step
larger than the sample yields the single date `W`). `run_backtest`
additionally requires **`T >= W + 2`** (at least two holdout days, so the
ddof = 1 volatility and the Sharpe ratio exist) and a finite `rf`. At `t_k`
the weight function sees the **expanding** history rows `[0, t_k)` (a
read-only view in Python, a copy in the ports — a mutating strategy
cannot corrupt the panel); its weights apply to rows
`[t_k, min(t_{k+1}, T))`. Within a block the portfolio is rebalanced to
target daily, so the daily portfolio return is exactly `w . r_t`.
Weight vectors must be finite and sum to 1 within 1e-6 (error otherwise).

### 6.2 Metrics (pinned formulas)
Every metric rejects NaN/inf, a simple return `<= -1`, `periods <= 0`
(or non-finite) and a non-finite `rf` with a descriptive error.
- `ann_return = 252 * mean(r)` (arithmetic).
- `ann_vol = sqrt(252) * std(r, ddof=1)`.
- `sharpe = (ann_return - rf) / ann_vol`; error if `ann_vol < 1e-15`.
- Max drawdown: wealth `V_0 = 1`, `V_t = V_{t-1}*(1 + r_t)`; peak is the
  running max including `V_0`; `maxdd = max_t (1 - V_t / peak_t)`,
  reported as a positive fraction.
- Turnover at rebalance `k >= 1`: `sum_i |w_k[i] - w_{k-1}[i]|`
  (two-sided L1 between consecutive **target** vectors; nothing charged
  for the initial position). `avg_turnover` = mean over those K-1 values
  (0 if K < 2). Turnover is reported, never charged.

### 6.3 Comparison methods (pinned definitions, demo + golden)
Each maps history `h` (daily returns, rows `[0, t_k)`) to weights:
- `equal_weight`: `1/n` each.
- `min_var_lo`: QP long-only min-variance (3.6) on
  `252 * sample_cov(h)`.
- `tangency`: unconstrained tangency (3.3) on
  `(252*mean(h), 252*sample_cov(h))` with `rf = 0.02`; **on the tangency
  error only** (undefined or ill-conditioned per 3.3) fall back to
  unconstrained min-variance for that rebalance (reported, not crashed);
  any other error (e.g. a singular covariance) propagates.
- `black_litt`: BL weights (4.5) with the full-sample `w_mkt`, the
  bundled views, `tau = 0.05`, `delta = 2.5`, on `252 * sample_cov(h)`.
- `erc`: CCD ERC (5) on `sample_cov(h)` (scale-invariant).
- `inverse_vol`: inverse-vol (5) on `sample_cov(h)`.

---

## 7. Error behavior summary

| Condition | Result |
|---|---|
| non-square / asymmetric / NaN / inf matrix | error (3.1) |
| singular or non-PSD covariance (dup. asset) | error via Cholesky pivot (3.1) |
| returns / series with NaN, inf or a value <= -1 | error (1, 6.2) |
| `periods <= 0`, non-finite `rf` | error (1, 6.2) |
| n = 1 | trivial `[1.0]` weights (3.3, 3.6); frontier target must equal mu[0]; covariance still validated |
| rf >= min-variance return, or within 1e-6 (rel.) of it | tangency error, dedicated type (3.3) |
| target return below min-var return (incl. negative) | frontier error (3.3) |
| constrained target below constrained GMV return / above r_max | frontier error (3.6) |
| mu proportional to 1 | degenerate-frontier error (3.3) |
| `max(mu) <= A/C` (or `r_max <= r_gmv`) | "frontier grid empty" error (3.4) |
| box bounds infeasible (sum lb > 1, sum ub < 1, lb > ub, lb = -inf) | projection/QP error (3.5) |
| QP certificate fails (numerically inconsistent input) | `converged=False` from `*_solve`; weights-only wrappers raise (3.6) |
| CCD hits sweep cap | flag reported, no exception (5) |
| empty views, empty/duplicate names, unknown name/type, missing key, `long == short`, NaN q, zero P row, bad Q length | error (4.2) |
| explicit Omega together with confidences | error (4.4) |
| tau outside (0, 1]; confidence <= 0; delta <= 0 | error (4.1, 4.3, 4.4) |
| conflicting extreme views | finite posterior (no error) |
| T <= estimation window; T < window + 2; zero-vol Sharpe | backtest error (6) |
| shrinkage with n < 2 or T < 2 or zero variance | error (2.2) |

Determinism: no RNG anywhere in the library; repeated calls are
bit-identical. The data generator uses `numpy default_rng(seed=42)` only.

---

## 8. Golden cases (`data/golden/golden.json`)

Schema: `{"cases": [{"name", "inputs": {flat scalars}, "expect":
{flat scalars}, "tol"}]}`; vectors are flattened as `w_0..w_7`,
`pi_0..pi_7`, etc.; matrix entries as `cov_i_j`. Comparisons are
absolute: `|got - expect| <= tol`. Unless marked daily, cases use the
annualized sample moments of `asset_returns.csv` (section 1). There are
exactly **23** cases and every suite asserts that count.

| case | contents | tol |
|---|---|---|
| `sample_moments_daily` | mean_0/4/7, cov_0_0, cov_0_1, cov_4_4, cov_6_7 (daily) | 1e-12 |
| `shrinkage_const_corr_daily` | intensity, rbar, shrunk_0_0, 0_1, 2_5, 7_7 (daily) | 1e-10 |
| `two_fund_scalars_annualized` | a, b, c, d | 1e-10 |
| `minvar_unconstrained` | w_0..w_7, ret, vol | 1e-10 |
| `tangency_unconstrained` | w_0..w_7, sharpe, gross_leverage (rf = 0.02) | 1e-10 |
| `frontier_point_t080` | w_0..w_7, variance at target 0.08 | 1e-10 |
| `minvar_longonly_qp` | w_0..w_7, variance (QP, long-only) | 1e-10 |
| `minvar_box_qp` | w_0..w_7, variance (QP, lb 0.02, ub 0.40) | 1e-10 |
| `frontier_constrained_t080` | w_0..w_7, variance (QP, long-only, target 0.08 met exactly) | 1e-10 |
| `frontier_constrained_box_t060` | w_0..w_7, variance (QP, lb 0.02, ub 0.40, target 0.06) | 1e-10 |
| `cholesky_annualized_cov` | l_0_0, l_7_0, l_7_7 of L | 1e-12 |
| `simplex_projection_hand` | 3-vector projection, no bounds | 1e-12 |
| `simplex_projection_box` | 4-vector, lb 0.05, ub 0.5 | 1e-12 |
| `implied_pi` | pi_0..pi_7 (delta = 2.5, market caps) | 1e-10 |
| `bl_omega_default` | omega_0_0, omega_1_1 (tau 0.05, conf 1.0/0.5) | 1e-12 |
| `bl_posterior_returns` | mu_0..mu_7 | 1e-10 |
| `bl_posterior_weights` | w_0..w_7 (tangency at rf = 0) | 1e-10 |
| `inverse_vol` | w_0..w_7 | 1e-12 |
| `erc_weights` | w_0..w_7, rc_spread (max RC - min RC) | 1e-8 |
| `risk_budget_pinned` | w_0..w_7 for budgets b_0..b_7 in inputs | 1e-8 |
| `backtest_equal_weight` | ann_return, ann_vol, sharpe, max_drawdown | 1e-8 |
| `backtest_erc` | sharpe, max_drawdown | 1e-8 |
| `turnover_minvar_longonly` | avg_turnover, n_rebalances (=12) | 1e-9 |

Regeneration: `python3 data/generate_data.py` (also revalidates the
closed forms against `numpy.linalg.lstsq`, the Ledoit-Wolf intensity
against a transcription of `covCor.m`, the four QP cases against scipy
SLSQP with KKT residual < 1e-12, the ERC equal-contribution identity, and
both BL limits before writing).
