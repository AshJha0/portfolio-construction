# portopt API_SPEC — language-neutral numerical contract (P10)

This document pins every algorithm so that scipy/LAPACK-free ports
(C++ `portopt`, Rust `portopt`, Java `com.quant.portopt`) reproduce the
Python reference and the golden values in `data/golden/golden.json`.
All arithmetic is IEEE-754 double precision. Indices below are 0-based.
Unless stated otherwise, "error" means: Python `ValueError`,
C++ `std::invalid_argument`/`std::domain_error`, Rust `Err(PortOptError)`,
Java `IllegalArgumentException`, each with a descriptive message.

Notation: `T` = number of return rows, `N`/`n` = number of assets,
`1` = vector of ones, `'` = transpose.

---

## 1. Data files

- `data/asset_returns.csv` — header row of 8 asset names
  (`US_EQ,EU_EQ,JP_EQ,UK_EQ,EM_EQ,US_BOND,IG_CORP,FX_CARRY`), then 1250
  rows of 8 comma-separated daily simple returns printed with `%.17g`
  (exact double round-trip). No date column. Golden values are defined on
  the file **as parsed**, not on the generator's in-memory values.
- `data/market_caps.csv` — header `asset,market_cap`, 8 rows.
  `w_mkt[i] = cap[i] / sum(cap)`.
- `data/views.json` — `{"tau": 0.05, "delta": 2.5, "views": [...]}` with
  view objects per section 4.2.
- `data/config.json` — constants: `annualization_factor = 252`,
  `risk_free_rate = 0.02`, `risk_aversion_delta = 2.5`, `tau = 0.05`,
  `estimation_window = 500`, `rebalance_step = 63`,
  `frontier_points = 25`, `frontier_penalty = 1e4`.

Annualization (pinned): `mu_ann = 252 * mu_daily`,
`Sigma_ann = 252 * Sigma_daily`. All portfolio-level golden cases use the
**annualized sample** moments unless the case name says `daily`.

---

## 2. Estimators

### 2.1 Sample moments
- Mean: `mu[i] = (1/T) * sum_t X[t][i]` (T >= 1).
- Covariance (ddof = 1, T >= 2): demean columns
  (`xc[t][i] = X[t][i] - mu[i]`), then
  `S[i][j] = (1/(T-1)) * sum_t xc[t][i] * xc[t][j]`.

### 2.2 Ledoit-Wolf constant-correlation shrinkage (2004), pinned term by term
Requires N >= 2 and `S[i][i] > 0` for all i. With `xc` and `S` from 2.1:

1. Average correlation:
   `rbar = (2 / (N*(N-1))) * sum_{i<j} S[i][j] / sqrt(S[i][i]*S[j][j])`.
2. Target: `F[i][i] = S[i][i]`;
   `F[i][j] = rbar * sqrt(S[i][i]*S[j][j])` for `i != j`.
3. `pi_ij = (1/T) * sum_t (xc[t][i]*xc[t][j] - S[i][j])^2`;
   `pi_hat = sum over all ordered pairs (i,j) of pi_ij` (diagonal included).
4. `theta_ii,ij = (1/T) * sum_t (xc[t][i]^2 - S[i][i]) * (xc[t][i]*xc[t][j] - S[i][j])`.
5. `rho_hat = sum_i pi_ii
   + sum over ordered pairs i != j of (rbar/2) * sqrt(S[j][j]/S[i][i]) * theta_ii,ij`
   (the ordered double sum supplies both symmetric terms of the classic
   formula exactly once each).
6. `gamma_hat = sum over all ordered pairs (i,j) of (F[i][j] - S[i][j])^2`.
7. `kappa = (pi_hat - rho_hat) / gamma_hat`;
   `delta = max(0, min(1, kappa / T))`; if `gamma_hat <= 0` then `delta = 0`.
8. `Sigma_shrunk = delta * F + (1 - delta) * S`.

**Pinned simplification** (deviation from the original paper, deliberate):
`S` keeps divisor `T-1` (consistency with 2.1) while the dispersion sums
pi/theta use divisor `T`. Every port must copy this exactly.

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
  For `n == 1`: `w = [1.0]` (trivial; only requires `Sigma[0][0] > 0`).
- Tangency: `z = solve(Sigma, mu - rf*1)`, `w_tan = z / sum(z)` where
  `sum(z) = A - rf*C`. Error if `A - rf*C <= 0` ("tangency undefined";
  covers rf above every expected return). For `n == 1`: `[1.0]` if
  `mu[0] > rf`, else error.
- Frontier at target m: error if `D <= 1e-12 * B * C` ("degenerate
  frontier", mu proportional to 1); error if
  `m < A/C - 1e-12*(1+|A/C|)` (target below the min-variance return,
  includes negative targets); else
  `g = (B*y - A*x)/D`, `h = (C*x - A*y)/D`, `w(m) = g + h*m`.
  For `n == 1`: `[1.0]` if `|m - mu[0]| <= 1e-12*(1+|mu[0]|)`, else error.
- Frontier variance identity (used by tests):
  `var(m) = (C*m^2 - 2*A*m + B) / D`.

### 3.4 Frontier sweep (pinned grid)
`n_points = 25` targets: `target_k = lo + k*(hi - lo)/(n_points - 1)`,
`k = 0..24`, with `lo = A/C` (min-variance return) and `hi = max(mu)`.
Same grid for the unconstrained sweep (closed form per 3.3) and the
constrained sweep (penalty PGD per 3.6).

### 3.5 Projection onto the box-constrained simplex (exact, sort-based)
Project `v` onto `{w : sum(w) = total, lb <= w <= ub}`; defaults
`lb = 0`, `ub = +inf`, `total = 1`.
Feasibility gate (errors): any `lb[i] > ub[i]`;
`sum(lb) > total + 1e-12*(1+|total|)`; `sum(ub) < total - 1e-12*(1+|total|)`;
NaN/inf in `v`; NaN in bounds.

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

### 3.6 Projected gradient descent (deterministic, no RNG)
Used for constrained min-variance and the constrained frontier.
Objectives:

- min-variance: `f(w) = 0.5 * w' Sigma w`, `grad = Sigma w`.
- constrained frontier point at target m (quadratic-penalty form, pinned
  `penalty = 1e4`): `f(w) = 0.5 * w' Sigma w + 0.5*penalty*(mu.w - m)^2`,
  `grad = Sigma w + penalty*(mu.w - m)*mu`. The golden values are defined
  as the minimizer of THIS penalized objective — the achieved return
  deviates from m by O(1/penalty) by design.

Algorithm (pinned):
1. Validate Sigma via Cholesky (3.1). Start `w = project(1/n, ..., 1/n)`
   with the projection of 3.5; step `t = 1`.
2. Per iteration: `t = 2*t`; `g = grad(w)`; `fw = f(w)`;
   candidate `w+ = project(w - t*g)`, `d = w+ - w`; while the
   sufficient-decrease test `f(w+) <= fw + g.d + (d.d)/(2*t)` fails,
   halve `t` and recompute `w+`, `d` (at most 100 halvings, then accept).
3. Stop when `max_i |d[i]| < 1e-12` (converged) or after 10000
   iterations (non-convergence is reported via a flag, never raised).

Both objectives are convex with a unique minimizer on the feasible set
(strictly convex in the min-variance case), so ports agree to well below
the golden tolerance 1e-8 regardless of tiny rounding differences in the
iteration path.

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
Errors: empty view list; unknown type; any P row identically zero;
Q length != number of rows of P; non-finite entries.

### 4.3 Omega default (Idzorek-style) and confidence scaling
`Omega` is diagonal with
`Omega[k][k] = [P (tau*Sigma) P'][k][k] / c_k` (`c_k` = per-view
confidence, > 0). `c -> inf`: dogmatic view; `c -> 0`: view ignored.

### 4.4 Posterior (master formula, pinned computational form)
`tau` must satisfy `0 < tau <= 1` (error otherwise). With
`TS = tau*Sigma`, `Bm = P*TS` (k x n), `A_k = Bm*P' + Omega`:

```
mu_post = pi + Bm' * solve(A_k, Q - P*pi)
M       = TS - Bm' * A_k^-1 * Bm      (A_k^-1 * Bm via k Cholesky solves)
S_post  = Sigma + M
```

`A_k` is factored **once** by the Cholesky of 3.1 (error if not SPD).
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
double-count the risk-free rate).

---

## 5. Risk parity / risk budgeting (CCD)

Budgets `b` must be positive and finite (error otherwise); they are
normalized internally to sum 1; `b = (1/n, ..., 1/n)` is ERC.
Sigma is validated via Cholesky (3.1) — singular/non-PSD input errors.

Log-barrier formulation: minimize over `y > 0`
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
non-convergence is reported via a flag, never raised.
Output weights: `w = y / sum(y)`.

At the fixed point `y_i * (Sigma y)_i = b_i` exactly, so the normalized
risk contributions `RC_i = w_i (Sigma w)_i / (w' Sigma w)` equal `b_i`
(this identity is a golden case: `erc_weights.rc_spread`).
Edge case: an almost-zero-vol asset gets weight near 1 (correct ERC
limit); the update stays well-defined because `Sigma[i][i] > 0` after
the Cholesky gate. Inverse-vol comparison portfolio:
`w[i] = (1/sqrt(Sigma[i][i])) / sum_j (1/sqrt(Sigma[j][j]))`.

---

## 6. Backtest-lite

### 6.1 Schedule (pinned)
Estimation window `W = 500`, step `63` trading days. Rebalance row
indices: `t_k = W + 63*k` for `k = 0, 1, ...` while `t_k < T` (for
T = 1250: 12 dates, 500..1193). Error if `T <= W`. At `t_k` the weight
function sees the **expanding** history rows `[0, t_k)`; its weights
apply to rows `[t_k, min(t_{k+1}, T))`. Within a block the portfolio is
rebalanced to target daily, so the daily portfolio return is exactly
`w . r_t`. Weight vectors must be finite and sum to 1 within 1e-6
(error otherwise).

### 6.2 Metrics (pinned formulas)
- `ann_return = 252 * mean(r)` (arithmetic).
- `ann_vol = sqrt(252) * std(r, ddof=1)`.
- `sharpe = (ann_return - rf) / ann_vol`; error if `ann_vol < 1e-15`.
- Max drawdown: wealth `V_0 = 1`, `V_t = V_{t-1}*(1 + r_t)`; peak is the
  running max including `V_0`; `maxdd = max_t (1 - V_t / peak_t)`,
  reported as a positive fraction.
- Turnover at rebalance `k >= 1`: `sum_i |w_k[i] - w_{k-1}[i]|`
  (two-sided L1 between consecutive **target** vectors; nothing charged
  for the initial position). `avg_turnover` = mean over those K-1 values
  (0 if K < 2).

### 6.3 Comparison methods (pinned definitions, demo + golden)
Each maps history `h` (daily returns, rows `[0, t_k)`) to weights:
- `equal_weight`: `1/n` each.
- `min_var_lo`: PGD long-only min-variance (3.6) on
  `252 * sample_cov(h)`.
- `tangency`: unconstrained tangency (3.3) on
  `(252*mean(h), 252*sample_cov(h))` with `rf = 0.02`; **if undefined**
  (`A - rf*C <= 0`) fall back to unconstrained min-variance for that
  rebalance (reported, not crashed).
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
| n = 1 | trivial `[1.0]` weights (3.3); frontier target must equal mu[0] |
| rf >= min-variance return (incl. rf above all mu) | tangency error (3.3) |
| target return below min-var return (incl. negative) | frontier error (3.3) |
| mu proportional to 1 | degenerate-frontier error (3.3) |
| box bounds infeasible (sum lb > 1, sum ub < 1, lb > ub) | projection error (3.5) |
| PGD / CCD hits iteration cap | flag reported, no exception (3.6, 5) |
| empty views, zero P row, bad Q length | error (4.2) |
| tau outside (0, 1]; confidence <= 0; delta <= 0 | error (4.1, 4.3, 4.4) |
| conflicting extreme views | finite posterior (no error) |
| T <= estimation window; zero-vol Sharpe | backtest error (6) |
| shrinkage with n < 2 or T < 2 or zero variance | error (2.2) |

Determinism: no RNG anywhere in the library; repeated calls are
bit-identical. The data generator uses `numpy default_rng(seed=42)` only.

---

## 8. Golden cases (`data/golden/golden.json`)

Schema: `{"cases": [{"name", "inputs": {flat scalars}, "expect":
{flat scalars}, "tol"}]}`; vectors are flattened as `w_0..w_7`,
`pi_0..pi_7`, etc.; matrix entries as `cov_i_j`. Comparisons are
absolute: `|got - expect| <= tol`. Unless marked daily, cases use the
annualized sample moments of `asset_returns.csv` (section 1).

| case | contents | tol |
|---|---|---|
| `sample_moments_daily` | mean_0/4/7, cov_0_0, cov_0_1, cov_4_4, cov_6_7 (daily) | 1e-12 |
| `shrinkage_const_corr_daily` | intensity, rbar, shrunk_0_0, 0_1, 2_5, 7_7 (daily) | 1e-10 |
| `two_fund_scalars_annualized` | a, b, c, d | 1e-10 |
| `minvar_unconstrained` | w_0..w_7, ret, vol | 1e-10 |
| `tangency_unconstrained` | w_0..w_7, sharpe (rf = 0.02) | 1e-10 |
| `frontier_point_t080` | w_0..w_7, variance at target 0.08 | 1e-10 |
| `minvar_longonly_pgd` | w_0..w_7 (PGD, long-only) | 1e-8 |
| `frontier_constrained_t080` | w_0..w_7 (penalty PGD, target 0.08) | 1e-8 |
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
| `turnover_minvar_longonly` | avg_turnover, n_rebalances (=12) | 1e-10 |

Regeneration: `python3 data/generate_data.py` (also revalidates the
closed forms against `numpy.linalg.lstsq`, the ERC equal-contribution
identity, and both BL limits before writing).
