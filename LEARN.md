# LEARN — Portfolio Construction from Markowitz to Risk Parity

This document teaches the theory implemented in `portopt`: mean-variance
optimization and its closed forms, why the raw recipe fails on estimated
inputs, covariance shrinkage, constrained optimization, Black-Litterman,
and risk parity — ending with the practical workflow (rebalancing,
turnover, costs) and an honest scorecard of every method. All worked
numbers reproduce entries of `data/golden/golden.json`, computed on the
bundled 8-asset universe (`US_EQ, EU_EQ, JP_EQ, UK_EQ, EM_EQ, US_BOND,
IG_CORP, FX_CARRY`; 1250 daily returns, annualization factor 252).

---

## 1. Mean-variance foundations

### 1.1 The utility view

An investor holding weights `w` in assets with expected returns `mu` and
covariance `Sigma` earns a portfolio return with mean `mu'w` and variance
`w' Sigma w`. Markowitz (1952) proposed judging portfolios by exactly
these two numbers. The standard formalization is quadratic utility (or,
equivalently, exact for normal returns under CARA utility):

```
U(w) = mu'w - (delta/2) * w' Sigma w
```

`delta` is risk aversion: how many units of variance the investor will
pay for one unit of expected return. Maximizing U without constraints
gives the first key formula of the whole subject:

```
dU/dw = mu - delta * Sigma w = 0   =>   w* = (1/delta) * Sigma^{-1} mu
```

Everything else in this project is a variation on `Sigma^{-1} mu`:
Markowitz computes it, Black-Litterman runs it backwards, shrinkage
repairs the `Sigma` that goes into it, and risk parity refuses to use
`mu` at all.

Two readings of the same optimum are worth keeping in mind:

* **Trade-off reading**: each asset's weight is its expected return
  scaled down by risk and correlation with everything else.
* **Hedging reading**: `Sigma^{-1}` mixes assets so that each position
  is partly a bet and partly a hedge of the other bets. This is why
  optimal weights can be violently long one asset and short a highly
  correlated sibling — a point that becomes the central pathology in
  section 2.

### 1.2 The efficient frontier and the two-fund theorem

Fix the budget constraint `1'w = 1` and ask: for a target mean `m`, what
is the minimum achievable variance? Form the Lagrangian

$$
\mathcal{L}(w,\lambda,\gamma)
= \tfrac{1}{2} w^\top \Sigma w
- \lambda\,(\mu^\top w - m) - \gamma\,(\mathbf{1}^\top w - 1),
\qquad
\nabla_w \mathcal{L} = \Sigma w - \lambda \mu - \gamma \mathbf{1} = 0,
$$

which gives `w = Sigma^{-1}(lam*mu + gam*1)`; substituting into the two
constraints leaves the 2x2 linear system

$$
\begin{pmatrix} B & A \\ A & C \end{pmatrix}
\begin{pmatrix} \lambda \\ \gamma \end{pmatrix}
= \begin{pmatrix} m \\ 1 \end{pmatrix},
\qquad \det = BC - A^2 = D .
$$ Its coefficients are the four
scalars the code calls A, B, C, D (`two_fund_scalars`):

```
A = 1' Sigma^{-1} mu     B = mu' Sigma^{-1} mu
C = 1' Sigma^{-1} 1      D = B*C - A^2
```

`B > 0` and `C > 0` because `Sigma^{-1}` is positive definite, and by
Cauchy-Schwarz `D >= 0`, with equality only when `mu` is proportional to
`1` (all assets equally attractive — the frontier degenerates to a
point; the code rejects this with a "degenerate frontier" error).
Solving for the multipliers yields the frontier in closed form:

```
w(m) = g + h*m,   g = (B*y - A*x)/D,   h = (C*x - A*y)/D
```

where `x = Sigma^{-1} mu`, `y = Sigma^{-1} 1`. This is the **two-fund
theorem**: every frontier portfolio is an affine function of the target,
i.e. a mix of just two fixed portfolios (for example `g + h*m1` and
`g + h*m2` for any two distinct targets). An entire n-asset frontier
collapses to a line in portfolio space.

Substituting back gives the frontier variance — a parabola in `m`:

```
var(m) = (C*m^2 - 2*A*m + B) / D
```

Its vertex is the **global minimum-variance (GMV) portfolio**:

```
w_mv = Sigma^{-1} 1 / C,   return A/C,   variance 1/C
```

Note that `w_mv` does not involve `mu` at all — a fact that makes the
GMV portfolio far more robust in practice (section 2).

**Worked example (golden `two_fund_scalars_annualized`,
`minvar_unconstrained`, `frontier_point_t080`).** On the annualized
sample moments of the bundled data:

```
A = 20.1810855643   B = 1.3427105101
C = 626.0618354614  D = B*C - A^2 = 433.3435918688
```

* GMV return `A/C = 20.1810855643 / 626.0618354614 = 0.0322349717`
  (3.22% p.a.) and vol `sqrt(1/C) = 0.0399660645` (4.00%) — matching
  `minvar_unconstrained.ret/vol`. The GMV weights put 70.1% in US_BOND
  and short EM_EQ by 4.6%: minimum variance loves low-vol assets and
  will short whatever hedges best.
* Frontier variance at target `m = 0.08`:
  `(626.0618*0.0064 - 2*20.1810856*0.08 + 1.3427105)/433.3435919
  = 2.1205325/433.3435919 = 0.0048934208` — matching
  `frontier_point_t080.variance` (vol 7.0%).

### 1.3 Adding a risk-free asset: the tangency portfolio

With a risk-free rate `rf`, combinations of cash and any risky portfolio
plot on a straight line in (vol, mean) space. The steepest such line
touches the frontier at the **tangency (maximum Sharpe) portfolio**:

```
w_tan = Sigma^{-1}(mu - rf*1) / (A - rf*C)
```

The denominator `A - rf*C = 1'Sigma^{-1}(mu - rf*1)` must be positive:
`rf < A/C`, i.e. the risk-free rate must sit below the GMV return.
Otherwise no risky portfolio on the upper frontier is tangent to a line
through `(0, rf)` and the code raises "tangency undefined" — this is a
real failure mode, not a corner case: it fires at 8 of 12 rebalance
dates in the bundled backtest, whenever the estimated GMV return of the
expanding window dips below rf = 2%.

The tangency Sharpe ratio has its own closed form,
`sharpe_tan = sqrt(B - 2*rf*A + rf^2*C)`. Worked:
`sqrt(1.3427105 - 0.04*20.1810856 + 0.0004*626.0618355)
= sqrt(0.7858918) = 0.8865054` — matching
`tangency_unconstrained.sharpe` at rf = 0.02.

With a risk-free asset the two-fund theorem becomes **one-fund**: every
investor holds cash plus the same tangency portfolio, scaled by risk
appetite. That equilibrium logic is exactly what Black-Litterman
exploits in reverse (section 5).

---

## 2. Why raw Markowitz fails: estimation error maximization

Everything above assumed `mu` and `Sigma` are known. They never are.
With T observations of N assets:

* the sample mean of an asset with 16% annual vol over 5 years has a
  standard error of `0.16/sqrt(5) ≈ 7%` — as large as any plausible
  equity risk premium. Return estimates are mostly noise at any horizon
  an investor can wait for.
* the sample covariance has `N(N+1)/2` free parameters estimated from
  `T*N` numbers; for N large relative to T its smallest eigenvalues are
  biased severely downward (and it is singular for `T <= N`).

The optimizer does not average this noise away — it **amplifies** it.
`w* ∝ Sigma^{-1} mu` allocates aggressively toward whatever looks best,
and the things that look best are disproportionately the things with the
largest upward estimation error. Michaud's name for this is **error
maximization**: mean-variance optimization is a machine for finding and
leveraging the errors in its inputs. The mechanism is worst along
directions where `Sigma` has small eigenvalues: `Sigma^{-1}` blows those
directions up, so a tiny spurious return difference between two highly
correlated assets produces a huge long-short position "arbitraging" a
signal that does not exist.

You can see all of it in the bundled golden values:

* `tangency_unconstrained` weights: +135.6% US_EQ, −57.6% EU_EQ, −41.2%
  UK_EQ. Three correlated equity indices, and the optimizer bets big on
  their tiny (noise-dominated) sample-mean differences.
* The backtest: raw tangency turns over 18.2 (L1) per quarter — it
  re-bets on fresh noise every rebalance — with 57% realized vol and a
  47% drawdown, while long-only min-variance turns over 0.027.
* Sensitivity is one-sided in a telling way: the GMV portfolio (no `mu`)
  is stable across rebalances; everything that uses sample means is not.
  This asymmetry — means are much harder to estimate than covariances,
  and the optimizer is much more sensitive to them — is the single most
  useful empirical fact in portfolio construction.

The remedies implemented here attack different parts of the pipeline:

| Remedy | Attacks | Section |
|---|---|---|
| Shrink the covariance | noise in `Sigma` | 3 |
| Constrain the weights | noise amplification | 4 |
| Replace `mu` with equilibrium + views | noise in `mu` | 5 |
| Drop `mu` entirely | noise in `mu` | 6 |

---

## 3. Covariance shrinkage (Ledoit-Wolf)

### 3.1 Intuition

The sample covariance `S` is unbiased but high-variance; a structured
target `F` (here: every pair has the *same* correlation `rbar`) is
biased but has very few parameters and hence almost no estimation
variance. A convex combination

```
Sigma_shrunk = delta * F + (1 - delta) * S,    delta in [0, 1]
```

can beat both, exactly as in classical bias-variance trade-offs (it is
the covariance analogue of James-Stein shrinkage of means). Ledoit and
Wolf (2004) made this operational by deriving the `delta` that minimizes
expected Frobenius loss, estimable from the data itself:

```
delta* ≈ (1/T) * (pi_hat - rho_hat) / gamma_hat
```

with the interpretation:

* `pi_hat` — total sampling variance of the entries of `S` ("how noisy
  is the thing I might shrink"): sum over (i,j) of
  `(1/T) sum_t (x_ti x_tj - S_ij)^2`;
* `rho_hat` — a correction for the covariance between the noise in `S`
  and the noise in `F` (the target is itself built from the same data);
* `gamma_hat` — squared distance `sum_ij (F_ij - S_ij)^2` between target
  and sample ("how much bias would full shrinkage introduce").

Much noise and little bias distance ⇒ shrink a lot; the reverse ⇒
barely shrink. The estimate is clipped to [0, 1] (`delta = 0` if
`gamma_hat <= 0`), and shrinkage toward a positive-definite target also
repairs near-singular sample matrices — the shrunk matrix passes the
Cholesky gate whenever `delta > 0` and `F` is PD.

### 3.2 The pinned estimator, and a worked number

`portopt` implements the constant-correlation estimator of Ledoit-Wolf
(2004) with one pinned simplification, stated in API_SPEC section 2.2:
`S` uses divisor `T-1` (consistent with `sample_cov`) while the
dispersion sums `pi/theta/gamma` use divisor `T`. The off-diagonal
`rho_hat` terms use only `theta_ii,ij = (1/T) sum_t (x_ti^2 - S_ii)
(x_ti x_tj - S_ij)`, summed over ordered pairs `i != j` with weight
`(rbar/2) sqrt(S_jj/S_ii)` — the ordered double sum supplies both
symmetric halves of the classic formula exactly once each. Ports must
copy this exactly; "roughly Ledoit-Wolf" will fail the 1e-10 golden
tolerance.

On the bundled data (golden `shrinkage_const_corr_daily`):

```
rbar  = 0.4171937171   (average pairwise correlation)
delta = 0.0094304924   (intensity)
```

Diagonals are untouched by construction (`F_ii = S_ii`, so
`shrunk_0_0 = cov_0_0 = 9.354067081e-05`), and off-diagonals move
slightly toward the constant-correlation value: `S_01 = 7.746918e-05`
shrinks to `7.711134e-05`. The intensity is small — under 1% — because
T = 1250 is large relative to N = 8, so `S` is already well estimated.
Shrinkage matters most when `N/T` is not small (think 500 stocks on 2
years of daily data), where optimal intensities of 0.5+ are routine.
The honest summary: shrinkage reliably improves out-of-sample
minimum-variance portfolios; it cannot fix expected returns.

---

## 4. Constrained optimization: why long-only changes everything

### 4.1 No more closed form

Add `w >= 0` (no shorting) and the Lagrangian solution dies: which
constraints bind is unknown in advance, and with box bounds the active
set can be any of `3^n` sign patterns. The problem is still a convex QP
with a unique minimizer (for PD `Sigma`), but it must be solved
iteratively. `portopt` uses **projected gradient descent** (PGD):
gradient step on the objective, then exact Euclidean projection back
onto the feasible set

```
{ w : sum(w) = 1, lb <= w <= ub }.
```

The projection has a beautiful structure: the KKT conditions say the
projection of `v` is `w_i = clip(v_i - t, lb_i, ub_i)` for a single
scalar shift `t` chosen so the weights sum to 1. `g(t) = sum_i clip(v_i
- t, ...)` is piecewise linear and non-increasing, with kinks only at
the 2n breakpoints `v_i - lb_i`, `v_i - ub_i` — so sorting the
breakpoints and interpolating on the bracketing segment solves the
projection **exactly in finite time** (no tolerance, no iteration).
Golden hand case: project `[0.9, 0.5, -0.2]` onto the simplex — the
third coordinate clips to 0, so solve `(0.9-t) + (0.5-t) = 1`, giving
`t = 0.2` and `w = [0.7, 0.3, 0.0]` (`simplex_projection_hand`). PGD itself is pinned to a deterministic
doubling/halving backtracking line search (API_SPEC 3.6); with a convex
objective it converges without any randomness, so ports agree to well
below 1e-8.

Constrained frontier points use a pinned quadratic penalty
`(penalty/2)(mu'w - m)^2` with `penalty = 1e4` rather than an equality
constraint — simpler to project, and the O(1/penalty) target miss is
part of the contract.

### 4.2 Why practitioners *want* the constraint

Long-only is not just a regulatory nuisance — it is a statistical
regularizer. Jagannathan and Ma (2003) proved the point elegantly:
imposing `w >= 0` on minimum-variance optimization is equivalent to
running the *unconstrained* problem on a modified covariance matrix in
which the covariances of the assets that would have been shorted are
shrunk downward. Constraining weights *is* shrinking the covariance.
Since the shorts in an unconstrained solution are dominated by the
error-maximization mechanism of section 2, banning them removes mostly
noise. Empirically, long-only GMV portfolios routinely beat
unconstrained ones out of sample even though they are worse in sample.

Compare the golden weights on identical inputs:

```
                      US_EQ   EU_EQ   JP_EQ   UK_EQ   EM_EQ  US_BOND IG_CORP FX_CARRY
minvar_unconstrained  0.055   0.009   0.022   0.032  -0.046   0.701   0.039   0.187
minvar_longonly_pgd   0.037   0.000   0.016   0.021   0.000   0.732   0.015   0.179
```

The constrained solution zeroes the short (EM_EQ) and the
near-zero long (EU_EQ) and redistributes — a small in-sample variance
sacrifice for a large reduction in estimation leverage.

---

## 5. Black-Litterman

### 5.1 Reverse optimization: start from a defensible prior

Black and Litterman's diagnosis: the optimizer's inputs are the problem,
so pick expected returns that would make the optimizer *not* embarrass
you by default. If the market portfolio `w_mkt` (cap weights) is optimal
for a representative investor with risk aversion `delta`, then inverting
`w* = (1/delta) Sigma^{-1} mu` gives the **implied equilibrium excess
returns**:

```
pi = delta * Sigma * w_mkt
```

Feed `pi` back into the optimizer and you recover exactly `w_mkt` — the
round trip is a golden-tested identity (tangency at rf = 0 on
`(pi, Sigma)` returns the cap weights). The genius is that `pi` inherits
the covariance structure: highly correlated assets get near-identical
implied returns, so the optimizer has no phantom long-short bets to
make. On the bundled data (`implied_pi`, delta = 2.5): equities imply
2.8–3.9%, US_BOND implies just 0.014% — low vol, low correlation with
the cap-weighted portfolio, so equilibrium demands almost nothing.

### 5.2 Views, tau, Omega

Views are statements `P mu = Q + noise`, one row per view: an
**absolute** view ("EM_EQ will return 10%") is a unit row; a
**relative** view ("US_EQ beats EU_EQ by 2%") is a +1/−1 row. Treating
the prior as `mu ~ N(pi, tau*Sigma)` and the views as independent noisy
observations with covariance `Omega`, Bayes gives the posterior mean —
the BL **master formula**. Textbook precision-weighted form:

$$
\mu_{\text{post}}
= \left[(\tau\Sigma)^{-1} + P^\top \Omega^{-1} P\right]^{-1}
  \left[(\tau\Sigma)^{-1}\pi + P^\top \Omega^{-1} Q\right],
$$

algebraically identical to the numerically convenient form the code pins
(one k×k Cholesky solve, k = number of views):

$$
\mu_{\text{post}}
= \pi + \tau\Sigma P^\top
  \left(P\,\tau\Sigma\,P^\top + \Omega\right)^{-1} (Q - P\pi).
$$

Read it right to left: `Q - P*pi` is the *surprise* — how much your view
disagrees with equilibrium; the middle matrix converts surprise into an
update, weighted by relative confidence; `tau*Sigma*P'` propagates the
update to **every** asset through the covariance (a view on EM_EQ moves
correlated equities too — no view, no problem: assets you say nothing
about still adjust coherently). The posterior covariance is
`Sigma_post = Sigma + M` with `M = tau*Sigma - tau*Sigma P' (...)^{-1} P
tau*Sigma`, the remaining uncertainty about the mean itself.

The two scaling knobs confuse everyone, so pin the semantics:

* **tau** scales prior uncertainty about the *mean* relative to return
  volatility. Convention: small (0.01–0.1; here 0.05), on the logic that
  the mean is estimated from T observations, `tau ≈ 1/T`. The code
  enforces `0 < tau <= 1`.
* **Omega** is the view-noise covariance. `portopt` uses the
  Idzorek-style default `Omega = diag(P (tau Sigma) P')`, each entry
  divided by a per-view confidence `c_k`: `c → ∞` makes the view
  dogmatic, `c → 0` makes it ignored. Golden `bl_omega_default`: with
  tau = 0.05 and confidences (1.0, 0.5), `Omega_00 = 2.0637e-3`,
  `Omega_11 = 8.7083e-4`.

A subtlety worth knowing for interviews: with the default Omega, both
sides of the update scale with tau, so **the posterior mean is
tau-invariant** — tau only matters when Omega is specified exogenously.
The limit tests in the suite therefore fix Omega before sending
`tau → 0` (posterior → pi) or `Omega → ∞` (posterior → pi, M → tau*Sigma).

### 5.3 Worked posterior and why practitioners like it

Bundled views: EM_EQ returns 10% (confidence 1.0); US_EQ beats EU_EQ by
2% (confidence 0.5). Equilibrium says EM_EQ earns 3.85%; the view says
10%; the posterior lands at 6.94% (`bl_posterior_returns.mu_4`) — a
confidence-weighted compromise, and every other asset moved too (US_EQ
3.26% → 5.31%, part view-spillover through correlation, part the
relative view). The posterior tangency weights (`bl_posterior_weights`)
tilt from the market: EM_EQ 5.8% → 28.2%, US_EQ 24.8% → 41.7%, EU_EQ
16.5% → −10.2%. Tilts, not the wild ±100% swings of raw Markowitz.

Why the street likes it:

1. **Sane default**: no views ⇒ hold the market. Every deviation is
   traceable to a stated view with a stated confidence.
2. **Views in tradeable language**: "A beats B by 2%" is how PMs think;
   the machinery translates it into weights consistently.
3. **Internal consistency**: views propagate through correlations, so
   you never get the incoherent partial-input behavior of hand-tweaking
   `mu` one asset at a time.
4. **Numerical robustness**: conflicting extreme views keep `A_k =
   P tau Sigma P' + Omega` SPD, so the posterior stays finite (tested).

What it does *not* fix: `Sigma` still comes from estimation, delta and
tau are judgment calls, and the output is only as good as the views —
BL is a disciplined way to *hold* opinions, not a source of them.

---

## 6. Risk parity and equal risk contribution

### 6.1 Risk contributions

Portfolio vol is homogeneous of degree 1 in `w`, so Euler's theorem
splits it exactly into per-asset contributions. In normalized form:

```
RC_i = w_i * (Sigma w)_i / (w' Sigma w),    sum_i RC_i = 1
```

This decomposition (an identity, tested at 1e-12) reveals how
concentrated "diversified" portfolios really are: in a classic 60/40,
equities contribute ~90% of the risk. Golden `inverse_vol` vs
`erc_weights` makes the same point on our data — inverse-vol weighting
still leaves US_BOND contributing only 3.2% of risk while each equity
contributes ~15%.

**Equal risk contribution (ERC)** demands `RC_i = 1/n` for all i;
general **risk budgeting** demands `RC_i = b_i` for chosen budgets. No
expected returns anywhere — this is the "drop mu entirely" cell of the
remedy table. ERC sits provably between GMV and equal weight: it
diversifies like 1/n but in risk space rather than capital space.

### 6.2 The CCD algorithm (pinned)

ERC weights have no closed form for general Sigma. The pinned solver
(Griveau-Billion/Richard/Roncalli style) minimizes the log-barrier
objective over `y > 0`:

```
F(y) = (1/2) y' Sigma y - sum_i b_i * ln(y_i)
```

At the stationary point `y_i (Sigma y)_i = b_i` exactly — risk
contributions proportional to budgets — and `w = y / sum(y)`. Cyclical
coordinate descent updates one coordinate at a time: holding the cross
term `m_i = (Sigma y)_i - Sigma_ii y_i` fixed, `dF/dy_i = 0` is a
quadratic with a unique positive root,

```
y_i = ( -m_i + sqrt(m_i^2 + 4*Sigma_ii*b_i) ) / (2*Sigma_ii)
```

Strict convexity on the positive orthant gives a unique solution and
monotone convergence; the sweep order (ascending i), the incremental
`Sigma y` cache, tolerance 1e-10 and the 10000-sweep cap are all pinned
so ports match to 1e-8. Golden `erc_weights` achieves an RC spread
(max RC − min RC) of 1.3e-12; `risk_budget_pinned` hits budgets
(0.20, 0.20, 0.15, 0.15, 0.10, 0.10, 0.05, 0.05) the same way. Edge
case: an almost-zero-vol asset legitimately takes weight → 1 (its risk
contribution can only match the others if it dominates the portfolio) —
documentable behavior, not a bug, and a reason practitioners cap weights.

### 6.3 The leverage debate and the all-weather context

The uncomfortable arithmetic of ERC: because it loads up low-vol assets
(our ERC puts 40% in US_BOND, 17.6% in IG_CORP), an unlevered ERC
portfolio has bond-like volatility (5.2% in the backtest) and bond-like
expected return. Risk-parity funds in the Bridgewater "All Weather"
tradition therefore **lever** the portfolio (typically to ~10% vol),
arguing that a balanced, levered portfolio beats an unlevered
concentrated one at equal risk — implicitly a bet that low-risk assets
offer better risk-adjusted returns than the CAPM says (the "betting
against beta" / leverage-aversion story of Frazzini-Pedersen, plus,
historically, a four-decade bond bull market). The counterargument:
leverage introduces financing cost, path dependence, and forced
deleveraging exactly when correlations spike — 2022, when stocks and
bonds fell together, was the canonical stress test. `portopt`
deliberately implements only the unlevered ERC; the leverage decision is
policy, not optimization.

---

## 7. The practical workflow

The backtest module encodes a realistic loop, pinned so all languages
reproduce it: estimation window 500 days, rebalance every 63 trading
days (quarterly), **expanding** window, weights from history `[0, t_k)`
applied to `[t_k, t_{k+1})` — no look-ahead by construction. Metrics are
pinned too: arithmetic annualization `252 * mean`, vol
`sqrt(252) * std(ddof=1)`, Sharpe `(ann_ret - rf)/ann_vol`, max drawdown
on the compounded wealth path, and L1 turnover
`sum_i |w_k,i - w_{k-1},i|` between consecutive *target* vectors.

Reading the comparison table (README) as a practitioner would:

* **Turnover is a cost you pay with certainty; alpha is a hope.** At 10
  bps one-way costs, tangency's 18.2 turnover burns ~36 bps per quarter;
  ERC's 0.005 burns nothing. Net of costs the ranking tightens further
  against estimation-hungry methods.
* **Sharpe differences of 0.1–0.2 on a 750-day holdout are noise.** The
  standard error of a Sharpe estimate over 3 years is roughly 0.6. What
  is significant is the vol and drawdown ordering, which follows
  directly from construction.
* **Failure handling is part of the method.** Tangency was undefined at
  8 of 12 rebalances (estimated GMV return below rf); the pinned policy
  falls back to unconstrained min-variance and reports it. Production
  systems need such policies for every error path in API_SPEC section 7.

Market-convention notes (equity vs FX), since the universe mixes both:

* 252 trading days is the equity convention used throughout; FX and
  rates desks often use 260 or 365 for carry-style annualization — a
  1.5% relative difference that must be pinned (here: 252 everywhere).
* Returns are daily **simple** returns; portfolio return is the
  weighted sum (exact for simple returns under daily rebalancing to
  target — the pinned within-block convention). Log returns aggregate
  over time instead, and mixing the two silently biases long-horizon
  numbers.
* `FX_CARRY` is a *funded excess-return proxy* (long high-yield, short
  low-yield currencies). FX positions are naturally self-funding, so
  their "returns" are already excess returns; equity index returns are
  total returns from which rf must be subtracted for Sharpe. BL sidesteps
  the mismatch by working entirely in excess-return space (`pi` is an
  excess return; posterior tangency uses rf = 0 — a non-zero rf there
  would double-count cash, which is why `bl_weights` pins rf = 0).
* Cap weights for reverse optimization need a decision for non-cap
  assets (bond proxies, FX carry): here the bundled `market_caps.csv`
  simply asserts notional weights — in practice this choice materially
  moves `pi`.

Common pitfalls and numerical issues, all of which the code turns into
explicit behavior:

1. **Singular covariance** (duplicated asset, T < N): caught by the
   pinned Cholesky pivot test `s <= 1e-13 * a_ii`, not by a mysterious
   NaN downstream.
2. **Near-singular but not singular**: the frontier and tangency weights
   explode long before Cholesky fails — inspect `D` and condition, or
   shrink. The degenerate-frontier guard `D <= 1e-12*B*C` catches the
   mu-proportional-to-1 case exactly.
3. **Asymmetric inputs** from round-tripping through files: the symmetry
   gate `max|A - A'| <= 1e-10 (1 + max|A|)` rejects them early.
4. **Explicit matrix inversion**: never formed here — every `Sigma^{-1}x`
   is a Cholesky solve (backward-stable, half the cost).
5. **Annualize-then-optimize vs optimize-then-annualize**: closed-form
   weights are scale-invariant in Sigma but *not* jointly in (mu, Sigma)
   with a nonzero rf; the golden cases pin annualized inputs.
6. **Comparing iterative results across languages**: only pinned
   algorithms (fixed sweep order, fixed backtracking) make 1e-8
   agreement possible; "any convex solver" would not.

---

## 8. Honest limitations — a scorecard

| Method | What it assumes | Where it breaks |
|---|---|---|
| Unconstrained Markowitz | mu, Sigma known; quadratic utility | Error maximization (sec. 2); extreme, unstable, costly weights; useless mu estimates at feasible T |
| GMV | Sigma known | Robust to mu by construction, but concentrates in low-vol assets and still inherits Sigma noise |
| Long-only PGD | Convexity; bounds are meaningful | Corner solutions (zeros) look arbitrary; implicit covariance shrinkage may over-shrink true hedges |
| LW shrinkage | Constant-correlation target is a sane prior | Wrong target biases pairs with genuinely unusual correlation; does nothing for mu; assumes iid returns (real returns have volatility clustering) |
| Black-Litterman | Market is near equilibrium; views are honest, calibrated | delta, tau, Omega are judgment; cap-weight prior is ill-defined for bonds/FX; garbage views in, coherent garbage out |
| ERC / risk budgeting | Risk = covariance; budgets are meaningful | Ignores expected returns entirely; unlevered version is bond-heavy; levered version bets on stable stock-bond correlation and cheap financing (2022) |
| Backtest-lite | Daily rebalance to target; no costs, no slippage | Turnover reported but not charged; single 750-day holdout ⇒ wide Sharpe error bars; expanding window blends regimes |

Deeper limits shared by everything above: covariance is symmetric, but
investors fear downside, not variance; correlations are conditional and
spike in crises exactly when diversification is needed; and all methods
here are single-period — multi-period objectives (drawdown control,
liability matching, tax) need different machinery.

---

## 9. Interview-style Q&A

**Q1. Derive the minimum-variance portfolio and explain why it doesn't
depend on expected returns.**
Minimize `(1/2)w'Sigma w` s.t. `1'w = 1`: `w = gam Sigma^{-1} 1`; the
constraint gives `gam = 1/C`, so `w_mv = Sigma^{-1}1 / (1'Sigma^{-1}1)`.
`mu` never enters because the objective is variance alone — which is
precisely why GMV is the most robust point on the frontier: it is immune
to the noisiest input.

**Q2. What is the two-fund theorem and why does it matter
computationally?**
Every frontier portfolio is `w(m) = g + h*m` — an affine family spanned
by two fixed portfolios. Computationally, one pair of Cholesky solves
(`Sigma^{-1}mu`, `Sigma^{-1}1`) prices the entire frontier; the 25-point
sweep in `efficient_frontier` costs no extra factorizations.

**Q3. Your optimizer says short EU_EQ 58% and lever US_EQ 136%. What do
you tell your PM?**
That this is estimation error, not information: the two indices are ~80%
correlated, so `Sigma^{-1}` turns their tiny sample-mean difference —
statistically indistinguishable from zero — into a huge relative-value
bet. Fixes: shrink or structure the inputs (LW, BL), constrain the
weights (long-only kills the trade), or drop mu (ERC). Also check the
holding-period turnover: if the position flips each rebalance it is
chasing noise.

**Q4. Why does the long-only constraint improve out-of-sample
performance if it can only hurt the in-sample optimum?**
Jagannathan-Ma: the nonnegativity constraint is equivalent to shrinking
covariances of would-be-shorted assets in an unconstrained problem —
i.e., it is implicit covariance shrinkage. In-sample the feasible set is
smaller, so measured variance rises; out-of-sample the removed positions
were mostly estimation noise, so realized risk falls.

**Q5. In Black-Litterman, what exactly are tau and Omega, and what
happens in the limits?**
`tau*Sigma` is the covariance of your uncertainty about the *mean*
(small, ~1/T); `Omega` is the noise covariance of the views. Fixed
Omega, `tau → 0`: prior becomes dogmatic, posterior → pi. `Omega → ∞`
(confidence → 0): views ignored, posterior mean → pi, while the
posterior mean-uncertainty M → tau*Sigma. With the Idzorek default
`Omega = diag(P tau Sigma P')/c`, tau cancels and only the confidences
c matter.

**Q6. Why does BL use rf = 0 when computing posterior weights?**
Because `pi = delta Sigma w_mkt` are equilibrium *excess* returns, and
the posterior blends pi with views, so `mu_post` is an excess return
too. Subtracting rf again in the tangency formula would double-count
the risk-free rate; with no views, tangency at rf = 0 on `(pi, Sigma)`
returns exactly the market weights — the sanity identity the tests pin.

**Q7. Show that risk contributions sum to portfolio variance, and state
the ERC fixed-point identity.**
`sum_i w_i (Sigma w)_i = w' Sigma w` — Euler's theorem for the
degree-2-homogeneous variance (so normalized RCs sum to 1 identically).
The CCD solver's stationarity condition is `y_i (Sigma y)_i = b_i`
exactly; after normalizing `w = y/sum(y)`, `RC_i = b_i` — with equal
budgets, every asset contributes variance equally (golden rc_spread
≈ 1.3e-12).

**Q8. Risk parity beat 60/40 for decades. Give the case for and against
attributing that to the method.**
For: balanced risk exposure captures the term premium efficiently, and
low-beta assets have historically offered better Sharpe than the CAPM
line (leverage-aversion story), so levering bonds was structurally
smart. Against: the sample coincides with a 40-year bond bull market
and falling financing costs; when inflation drove stock-bond
correlation positive (2022), the diversification premise and the
leverage both hurt simultaneously. The method's real, defensible claim
is risk balance — not a return forecast.

**Q9. Your tangency portfolio throws "tangency undefined". What
happened and what do you do?**
`A - rf*C <= 0`: the risk-free rate is at or above the GMV return, so
no line through (0, rf) is tangent to the upper frontier — the
"tangency" would be on the inefficient lower branch. It happens
routinely with estimated means (8 of 12 rebalances in the bundled
backtest). Policy options: fall back to GMV (the pinned choice), use
excess-return BL inputs, or treat it as a signal that the estimated
frontier is too flat to bet on.

**Q10. Why does this project pin algorithms (Cholesky variant, sweep
order, backtracking rule) instead of just pinning tolerances?**
Because different correct algorithms produce different last-bit
rounding, and iterative methods amplify path differences: "any QP
solver" could put a different vertex of the same near-degenerate face
within 1e-4, not 1e-8. Pinning the operation sequence makes
cross-language agreement a rounding question (1e-10..1e-12 for closed
forms) instead of an algorithm question — that is what makes
`golden.json` enforceable in four languages.

---

## 10. Further reading

* H. Markowitz (1952), *Portfolio Selection*, Journal of Finance — the
  original mean-variance paper.
* R. Merton (1972), *An Analytic Derivation of the Efficient Portfolio
  Frontier* — the A/B/C/D closed-form treatment used here.
* R. Michaud (1989), *The Markowitz Optimization Enigma: Is "Optimized"
  Optimal?* — error maximization.
* V. Chopra, W. Ziemba (1993), *The Effect of Errors in Means,
  Variances, and Covariances on Optimal Portfolio Choice* — the classic
  errors-in-means-hurt-most result.
* O. Ledoit, M. Wolf (2004), *Honey, I Shrunk the Sample Covariance
  Matrix* — the constant-correlation shrinkage implemented here.
* R. Jagannathan, T. Ma (2003), *Risk Reduction in Large Portfolios: Why
  Imposing the Wrong Constraints Helps*.
* F. Black, R. Litterman (1992), *Global Portfolio Optimization*;
  T. Idzorek (2005), *A Step-by-Step Guide to the Black-Litterman
  Model* — the confidence-based Omega.
* S. Maillard, T. Roncalli, J. Teiletche (2010), *The Properties of
  Equally Weighted Risk Contribution Portfolios*; T. Griveau-Billion,
  J.-C. Richard, T. Roncalli (2013), *A Fast Algorithm for Computing
  High-dimensional Risk Parity Portfolios* — the CCD solver.
* A. Frazzini, L. Pedersen (2014), *Betting Against Beta* — the
  leverage-aversion foundation of the risk-parity trade.
* V. DeMiguel, L. Garlappi, R. Uppal (2009), *Optimal Versus Naive
  Diversification: How Inefficient Is the 1/N Portfolio Strategy?* — the
  humbling benchmark every method here must beat.
