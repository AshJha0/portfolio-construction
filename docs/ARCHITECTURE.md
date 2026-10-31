# ARCHITECTURE: portopt 1.3.0

Design notes for the four parallel implementations: the Python
reference (`python/src/portopt`), the C++26 library (`portopt::`, in
`cpp/`), the Rust crate `portopt` (`rust/`) and the Java 25 package
`com.quant.portopt` (`java/`). The binding contract is
[API_SPEC.md](../API_SPEC.md): every algorithm is pinned to the operation
level. This document explains why the pieces are shaped the way they are;
[LEARN.md](../LEARN.md) explains the mathematics.

This document describes release 1.3.0; the version numbers live in
`python/pyproject.toml`, `cpp/CMakeLists.txt`, `rust/Cargo.toml` and
`java/pom.xml`. The bundled
data is synthetic (`data/generate_data.py`, seed 42); performance and
accuracy figures below describe the code on that data.

## 1. Component responsibilities

| Component (Python module) | Responsibility | Depends on |
|---|---|---|
| `linalg` | Pinned Cholesky-Banachiewicz, SPD solve and inverse, symmetry and finiteness validation; compensated summation in the ports | nothing |
| `estimators` | Sample mean and covariance (ddof=1), Ledoit-Wolf constant-correlation shrinkage (paper-exact intensity), annualisation, returns validation | nothing (pure reductions, no factorisation) |
| `markowitz` | Two-fund closed forms, frontier sweeps, exact box-simplex projection, attainable-return range, Goldfarb-Idnani dual active-set QP with KKT certificate | `linalg` |
| `black_litterman` | Reverse optimisation, view parsing and validation, default Omega, posterior master formula, posterior tangency | `linalg`, `markowitz` |
| `risk_parity` | Risk contributions, CCD risk budgeting and ERC, inverse volatility | `linalg` |
| `backtest` | Pinned rebalance schedule, portfolio return accounting, metrics, target turnover | `estimators` (validation) |
| `research` (Python only) | `costs`: net-of-cost backtest on drifted weights, band, partial rebalance, volatility targeting with a chosen estimator; `stats`: Sharpe standard errors, PSR, DSR, bootstrap, paired test; `estimators`: EWMA, Ledoit-Wolf identity, nonlinear shrinkage, PCA factors, Bayes-Stein and James-Stein means; `strategies`, `strategies_v2`: weight rules and covariance-estimator objects; `data`: real-data loader with vintages and the holdout gate; `walkforward`: trial registry and calendar walk-forward | the modules above |
| `forward/run_forward.py` (outside the package) | Monthly paper log of frozen rule files against new data | `research` |
| `demo` | End-to-end report on the bundled data | all pinned modules |

The pinned modules form a strict DAG with `linalg` at the root of every
optimiser. Cross-language agreement of everything downstream therefore
reduces to agreement of one Cholesky factorisation and two triangular
substitutions. `estimators` does not depend on `linalg`: shrinkage is a
set of reductions, and the SPD gate belongs to the consumer of the
covariance (every optimiser calls `cholesky` first).

**The research layer.** `python/src/portopt/research/` is a Python-only
layer for studies that the pinned contract does not cover. It is outside
API_SPEC and `golden.json`, has no C++, Rust or Java counterpart, and may
change between minor versions [derived: `research/__init__.py`]. Its
modules are listed in the table above. Results:
[`cost_aware_allocation.md`](results/cost_aware_allocation.md) (synthetic),
[`real_data_study_dev.md`](results/real_data_study_dev.md) (development),
[`real_data_study.md`](results/real_data_study.md) (holdout).

**Real data, registry and holdout.** Daily industry portfolios and
factors from the Kenneth French Data Library are downloaded at run time
by `scripts/fetch_data.py` into the git-ignored `data/real/raw/`. Each
download is recorded as a vintage in `data/real/manifest.json` with the
SHA-256 of the zip and of the parsed arrays, and `load_returns` refuses
data that does not hash to a recorded vintage. Every configuration is
entered in `data/real/registry.json` before it runs, so the trial count
used by the deflated Sharpe ratio is the count of what was actually
tried. Rows from 2016-01-01 on are the holdout: `load_returns` raises
`HoldoutAccessError` for them unless `study_id` names a registered study
[derived: `research/data.py`; `data/real/README.md`]. Source:
`diagrams/real_data_holdout.mmd`.

```mermaid
flowchart TD
    FR["Kenneth French Data Library (zip files)"] -->|"scripts/fetch_data.py, at run time"| RAW["data/real/raw/ (git-ignored)"]
    RAW --> MAN["data/real/manifest.json: zip and parsed SHA-256 per vintage"]
    REG["data/real/registry.json: studies and configurations, registered before running"] --> LR
    MAN --> LR["research.data.load_returns(dataset, period, study_id)"]
    RAW --> LR
    LR -->|"period = development (before 2016-01-01)"| DEV["python/studies/real_data_study.py --phase dev"]
    LR -->|"period = holdout, only with a registered study_id"| HO["phase 2: run once on r2-holdout-v1"]
    LR -->|"holdout without study_id"| ERR["HoldoutAccessError"]
    DEV --> DEVR["docs/results/real_data_study_dev.md"]
    DEVR --> PRE["docs/results/real_data_preregistration.md: H1-H6, Holm, frozen forward rules"]
    PRE --> HO
    HO --> HOR["docs/results/real_data_study.md"]
```

**The forward ledger job.** `forward/rules/*.json` holds rule files
frozen at registration (study `r2-forward-v1`). Once a month the
workflow `.github/workflows/forward.yml` fetches the latest vintage and
runs `forward/run_forward.py`, which validates each rule against its
schema and the `STRATEGIES` table, computes weights only from data
before the month, and appends any newly completed month to
`forward/ledger.csv`. It commits only the ledger, `forward/results.md`
and the manifest [derived: `forward.yml`; `forward/README.md`]. Source:
`diagrams/forward_ledger.mmd`.

```mermaid
flowchart TD
    CRON[".github/workflows/forward.yml: cron 06:00 UTC on the 8th, or manual"] --> RULES{"forward/rules/*.json present?"}
    RULES -->|no| STOP["exit, nothing to log"]
    RULES -->|yes| FETCH["python scripts/fetch_data.py: new vintage recorded in manifest.json"]
    FETCH --> RUN["python forward/run_forward.py: validate each rule against its schema and STRATEGIES; weights from data before the month"]
    RUN --> NEW{"new complete month?"}
    NEW -->|no| STOP2["no change"]
    NEW -->|yes| APP["append rows to forward/ledger.csv; rewrite forward/results.md"]
    APP --> COMMIT["commit ledger.csv, results.md, manifest.json only"]
```

## 2. Data flow

```mermaid
flowchart TD
    subgraph DATA["data/ (synthetic, generate_data.py, seed 42)"]
        RET["asset_returns.csv<br/>1250 x 8 daily returns"]
        CAPS[market_caps.csv]
        VIEWS[views.json]
        CFG["config.json<br/>252, rf, tau, delta, W, step"]
    end

    subgraph EST["estimators"]
        MU["sample_mean, then annualize (x252)"]
        COV["sample_cov (ddof=1), then annualize"]
        LW["shrink_covariance<br/>delta*F + (1-delta)*S"]
    end

    subgraph LA["linalg (pinned kernel)"]
        CHOL["cholesky: Banachiewicz,<br/>pivot > 1e-13 * diag"]
        SOLVE["chol_solve: forward + back<br/>substitution"]
    end

    subgraph OPT["portfolio constructors"]
        TF["two_fund_scalars A,B,C,D<br/>min-var, tangency, frontier"]
        QP["dual active-set QP + KKT certificate<br/>box min-var, exact-target frontier"]
        BL["black_litterman<br/>pi = delta*Sigma*w_mkt, posterior"]
        RP["risk_parity CCD<br/>y_i quadratic update"]
    end

    BT["backtest: t_k = 500 + 63k,<br/>expanding window, L1 target turnover"]
    RES["research (Python only, outside the contract)<br/>net-of-cost backtest, Sharpe statistics"]
    OUTG["golden.json (26 cases)"]
    OUTD["demo report"]

    RET --> MU
    RET --> COV
    RET --> LW
    CFG --> MU
    CFG --> COV
    COV --> TF
    COV --> QP
    COV --> BL
    COV --> RP
    MU --> TF
    MU --> QP
    MU --> BL
    CAPS --> BL
    VIEWS --> BL
    CHOL --> SOLVE
    LA -. "every Sigma^-1 x" .-> OPT
    TF --> BT
    QP --> BT
    BL --> BT
    RP --> BT
    RET --> BT
    OPT --> RES
    BT --> OUTD
    EST --> OUTG
    OPT --> OUTG
    BT --> OUTG
    OPT --> OUTD
```

Flow invariants:

* Golden values are defined on `asset_returns.csv` as parsed, not on the
  generator's in-memory arrays. The CSV is written with `%.17g`, so
  doubles round-trip exactly and all four parsers see identical bits.
* Every T-length reduction is correctly rounded in Python (`math.fsum`)
  and Neumaier-compensated in the ports, so no golden value depends on the
  BLAS numpy links against.
* Shrinkage runs on daily returns (its T-scaling is meaningful there);
  optimisers run on annualised moments; scale-invariant methods (ERC,
  inverse volatility) accept either.
* Black-Litterman lives in excess-return space: `pi` comes out of reverse
  optimisation as an excess return, so posterior tangency is pinned at
  `rf = 0`.

## 3. Module structure

```mermaid
classDiagram
    class linalg {
        +cholesky(A) L
        +chol_solve(L, b) x
        +spd_solve(A, b) x
        +spd_inverse(A) Ainv
        +check_square_symmetric(A) A
        PIVOT_RTOL = 1e-13
    }
    class estimators {
        +validate_returns(X) X
        +sample_mean(X) mu
        +sample_cov(X) S
        +shrink_covariance(X) ShrinkageResult
        +annualize_mean(mu, periods) mu
        +annualize_cov(S, periods) S
    }
    class markowitz {
        +two_fund_scalars(mu, S) TwoFundScalars
        +min_variance_weights(S) w
        +tangency_weights(mu, S, rf) w
        +gross_leverage(w) float
        +frontier_weights(mu, S, m) w
        +efficient_frontier(mu, S, n_points) tuple
        +project_simplex_box(v, total, lb, ub) w
        +attainable_return_range(mu, lb, ub) tuple
        +min_variance_longonly_solve(S, lb, ub) QpResult
        +constrained_frontier_solve(mu, S, m, lb, ub) QpResult
        +constrained_frontier(mu, S, n_points, lb, ub) tuple
    }
    class black_litterman {
        +implied_equilibrium_returns(S, w_mkt, delta) pi
        +views_to_matrices(views, names) tuple
        +default_omega(S, P, tau, conf) Omega
        +bl_posterior(S, pi, P, Q, tau, Omega, conf) BLResult
        +bl_weights(S, pi, P, Q, tau, rf, Omega, conf) tuple
    }
    class risk_parity {
        +risk_contributions(w, S) rc
        +risk_budget_solve(S, b, tol, max_sweeps) tuple
        +risk_budget_weights(S, b) w
        +erc_weights(S) w
        +inverse_vol_weights(S) w
    }
    class backtest {
        +rebalance_indices(T, window, step) list
        +run_backtest(X, weight_fn, rf, window, step) BacktestResult
        +annualized_return(r) float
        +annualized_vol(r) float
        +sharpe_ratio(r, rf) float
        +max_drawdown(r) float
    }
    class TwoFundScalars {
        a
        b
        c
        d
    }
    class QpResult {
        weights
        iterations
        converged
        kkt_residual
    }
    class ShrinkageResult {
        sigma
        intensity
        rbar
        sample
        target
    }
    class BLResult {
        mu_post
        sigma_post
        m
        omega
    }
    class BacktestResult {
        portfolio_returns
        rebalance_dates
        weights
        turnovers
        ann_return
        ann_vol
        sharpe
        max_drawdown
        avg_turnover
    }

    markowitz --> linalg : all Sigma^-1 x
    black_litterman --> linalg : k x k solve
    black_litterman --> markowitz : posterior tangency
    risk_parity --> linalg : SPD gate
    backtest --> estimators : validation
    markowitz ..> TwoFundScalars
    markowitz ..> QpResult
    estimators ..> ShrinkageResult
    black_litterman ..> BLResult
    backtest ..> BacktestResult
```

The backtest engine is generic: `run_backtest` takes a
`weight_fn(history) -> weights` callback (Python callable, C++
`std::function`, Rust closure, Java functional interface), so the six
comparison methods of API_SPEC 6.3 are thin adapters in the demo. The
history handed to the callback is a read-only view (Python) or a copy
(ports), so a mutating strategy cannot corrupt later rebalances. Within
each 63-day block the daily portfolio return is `w_k' r_t`, a
daily-rebalanced constant mix; turnover is measured between consecutive
target vectors and no costs are charged. The rebalance loop is shown as
a sequence because its indexing is the part ports most often get wrong:

```mermaid
sequenceDiagram
    participant D as Demo or caller
    participant B as run_backtest
    participant W as weight_fn
    participant M as metrics

    D->>B: X (1250 x 8), weight_fn, rf, W=500, step=63
    B->>B: validate T >= W+2 and rf finite
    B->>B: dates = [500, 563, ..., 1193] (12)
    loop each rebalance t_k
        B->>W: history = X[0 : t_k] (expanding, read-only, no look-ahead)
        W-->>B: w_k (finite, sum = 1 within 1e-6, else error)
        B->>B: port[t] = X[t] . w_k for t in [t_k, t_next), daily constant mix
    end
    B->>B: turnover_k = sum_i abs(w_k - w_(k-1)) for k >= 1, no costs charged
    B->>M: port (750 daily returns)
    M-->>B: ann_return (252 x mean), ann_vol, sharpe, maxDD
    B-->>D: BacktestResult
```

## 4. Numerical design decisions and trade-offs

**One factorisation, pinned to the operation.** All linear systems (the
two-fund solves, the Black-Litterman k x k system, SPD inverses, the QP's
Gram systems) go through the same Cholesky-Banachiewicz code with a fixed
accumulation order. We give up BLAS-blocked speed and pivoted rescue
strategies and gain bit-level reproducibility across four languages.
Every inverse is realised as solves against identity columns.

**Relative Cholesky pivot gate (`s <= 1e-13 * A[i][i]`).** A duplicated
asset makes the pivot a tiny rounding residual rather than an exact zero,
so an absolute test would depend on scale. The relative gate rejects
condition numbers beyond about 1e13 and doubles as the project-wide
positive-definiteness check: the QP, CCD and Black-Litterman all call it
before iterating. It is permissive by design; a desk that needs to refuse
condition numbers above 1e8 should shrink first (LEARN 3) or add its own
eigenvalue diagnostic.

**Reductions that do not depend on the BLAS.** The first version of this
project pinned the 10,000th iterate of a projected-gradient loop as a
golden value, and that iterate was sensitive to the last bit of the
sample covariance. Every golden quantity is now a closed form or the
exact solution of a finite algorithm, and every T-length sum is correctly
rounded in the reference and compensated in the ports. The suites check
that scaling the input returns by `1 + 2^-52` moves the QP goldens by
less than 1e-13 (`python/tests/test_golden.py`).

**Exact projection.** The box-simplex projection exploits the piecewise
linearity of `g(t) = sum_i clip(v_i - t, lb_i, ub_i)`: one sort of at
most 2n breakpoints plus one interpolation gives the exact shift. It is a
public utility for repairing rounded weights and the feasibility gate for
bound vectors.

**Dual active-set QP.** Constrained min-variance and the constrained
frontier are strictly convex QPs. The Goldfarb-Idnani dual method needs
no feasible starting point, treats the target-return equality exactly
(no penalty bias), is finite, and ends with a closed-form solve on the
final active set, which makes the answer a function of the active set.
Two degenerate cases are handled explicitly: a fixed coordinate
(`lb = ub`) enters as an equality, and the maximum-return endpoint of the
frontier is reduced to a sum-only QP over the assets tied at the greedy
cut level. A rounding-level violation of a constraint that is dependent
on the active set is accepted when below 1e-9 and reported in
`kkt_residual`. The loop and its exits:

```mermaid
flowchart TD
    S0["Input: Sigma (SPD), equality rows a_eq and b_eq, bounds lb and ub"] --> S1["Cholesky of Sigma once; constraint list:<br/>equalities and fixed coordinates first, then lower and finite upper bounds"]
    S1 --> S2["w = 0 (unconstrained minimiser), active set empty"]
    S2 --> P{"Equalities left to add?"}
    P -- yes --> PE["p = next equality"]
    P -- no --> PV{"Inequality violated by more than<br/>1e-13 (1 + abs b)?"}
    PV -- none --> POL
    PV -- found --> PI["p = most violated, ties to lowest index"]
    PE --> DIR
    PI --> DIR
    DIR["Step directions z (primal) and r (dual)<br/>from Cholesky of Sigma and of Gram N' Sigma^-1 N"] --> T["t1 = largest step keeping inequality multipliers >= 0<br/>t2 = step that satisfies p"]
    T --> TC{"t = min(t1, t2) finite?"}
    TC -- "no, violation at most 1e-9" --> SKIP["skip p as degenerate"]
    SKIP --> P
    TC -- "no, larger violation" --> POL
    TC -- "yes, t2 infinite" --> DUAL["dual step only, drop blocking constraint"]
    DUAL --> DIR
    TC -- "yes, t = t2" --> ADD["full step, add p to the active set"]
    TC -- "yes, t = t1 below t2" --> DROP["partial step, drop blocking constraint"]
    ADD --> IT{"iterations below 1000?"}
    DROP --> IT
    IT -- yes --> P
    IT -- no --> POL
    POL["Polish: Gram solve on the final active set, w = Sigma^-1 N u,<br/>one refinement step, active bounds set exactly"] --> CERT["Certificate: primal feasibility, stationarity<br/>and active bound multipliers, tolerance 1e-9"]
    CERT --> CLIP["clip w to lb and ub"]
    CLIP --> OUT["QpResult: weights, iterations, converged, kkt_residual"]
```

**Deterministic CCD for risk parity.** CCD with ascending sweep order, an
incremental `Sigma y` cache and a `1e-10` sweep tolerance converges to the
unique minimiser of a strictly convex log-barrier objective. Its fixed
point is unique, so the ports agree well inside the 1e-8 golden
tolerance. The stopping rule is absolute in the unnormalised iterate, so
the achieved accuracy depends on the scale of `Sigma` (LEARN 6.3):

```mermaid
flowchart TD
    IN["Input: Sigma (SPD gate via Cholesky), budgets b > 0 normalised to sum 1"] --> INIT["y_i = 1/n, cache s = Sigma y"]
    INIT --> SW["next sweep, keep y_prev = y"]
    SW --> UPD["for i = 0 .. n-1 ascending:<br/>m_i = s_i - Sigma_ii y_i<br/>y_new = (-m_i + sqrt(m_i^2 + 4 Sigma_ii b_i)) / (2 Sigma_ii)"]
    UPD --> CACHE["s = s + Sigma[:, i] * (y_new - y_i), then y_i = y_new"]
    CACHE --> MORE{"more coordinates in this sweep?"}
    MORE -- yes --> UPD
    MORE -- no --> STOP{"max_i abs(y_i - y_prev_i) below tol<br/>(1e-10, absolute)?"}
    STOP -- yes --> OK["risk_budget_solve returns w = y / sum(y), sweeps, converged = true"]
    STOP -- no --> CAP{"sweeps below max_sweeps (10000)?"}
    CAP -- yes --> SW
    CAP -- no --> NC["risk_budget_solve returns w = y / sum(y), max_sweeps, converged = false"]
    OK --> WR["risk_budget_weights and erc_weights return the weights"]
    NC --> RAISE["risk_budget_weights and erc_weights raise ValueError"]
```

**Black-Litterman master formula in solve form.** The textbook posterior
`[(tau S)^-1 + P' O^-1 P]^-1 [...]` inverts an n x n matrix and needs an
invertible Omega. The pinned form factors only the k x k matrix
`A_k = P tau Sigma P' + Omega` (k views, usually far fewer than n) once
and reuses it for the mean update (one solve) and the posterior
covariance (n solves, one per column of `P tau Sigma`). It stays finite
for conflicting extreme views because `A_k` remains SPD:

```mermaid
flowchart TD
    SIG["Sigma (annualised covariance)"] --> PI["Reverse optimisation<br/>pi = delta * Sigma * w_mkt"]
    CAPS["market caps, normalised to w_mkt"] --> PI
    DELTA["risk aversion delta = 2.5"] --> PI
    VJSON["views.json: absolute and relative views"] --> VM["views_to_matrices<br/>P (k x n), Q (k), confidences c"]
    TAU["tau = 0.05"] --> OM
    VM --> OM{"Omega supplied?"}
    OM -- no --> OMD["default_omega<br/>Omega_kk = (P tau Sigma P')_kk / c_k"]
    OM -- yes --> OMX["explicit Omega, confidences not allowed"]
    OMD --> AK
    OMX --> AK
    SIG --> AK["A_k = P tau Sigma P' + Omega<br/>one k x k Cholesky"]
    PI --> POST
    AK --> POST["mu_post = pi + tau Sigma P' A_k^-1 (Q - P pi)<br/>M = tau Sigma - tau Sigma P' A_k^-1 P tau Sigma<br/>Sigma_post = Sigma + M"]
    POST --> W["bl_weights: tangency on mu_post and Sigma_post at rf = 0,<br/>normalised to sum 1"]
    W --> ERR{"A - rf C > 0 on the posterior?"}
    ERR -- yes --> OUT["weights and BLResult"]
    ERR -- no --> TU["TangencyUndefinedError"]
```

**Tangency conditioning guard.** `A - rf C` is a difference of two
numbers of order A. When rf is within one part per million of the GMV
return the weights scale like `1/(A - rf C)` and reach gross leverage of
1e6 or more while still summing to 1. The guard rejects that region with
a dedicated exception type, so the demo's fallback can match exactly
"tangency undefined or ill-conditioned" and let every other error
propagate. Outside the guard leverage can still be large: the demo
backtest reaches gross leverage 93.6 at `t = 1004` (LEARN 7.3).

**Turnover on target weights.** Turnover compares consecutive target
vectors, not drifted end-of-block holdings, because the within-block
convention is daily rebalancing to target, which makes the accounting
identity `port_t = w_k' r_t` testable. The cost of that convention is
that the daily constant-mix trades are invisible (LEARN 7.3 measures
them; the research layer charges them).

### Tangency leverage cap and the cost-aware QP (1.3.0)

**Tangency.** `tangency_weights` raises `TangencyUndefinedError` when
`A - rf*C <= 0` or when the gross leverage of the result exceeds
`max_leverage` (default `TANGENCY_MAX_LEVERAGE = 10`). The 1.2.0 rule
tested a relative 1e-6 cancellation of `A - rf*C` and accepted 93.6x
gross leverage on the bundled backtest. Testing leverage directly is
dimensionless and subsumes the old test [derived: API_SPEC 3.3]. Every
port implements it; Rust adds `tangency_weights_capped`.

**Turnover QP.** `min_variance_turnover[_solve]` and
`mean_variance_turnover[_solve]` minimise variance (or mean-variance)
plus `kappa * sum|w - w0|`, with an optional turnover cap. The trade is
split as `w = w0 + b - s`, `b, s >= 0`, which gives a 3n-variable QP
that the existing dual active-set solver handles once a proximal term
with `eps = 0.5 / trace(Sigma^-1)` makes it strictly convex without
changing the minimiser. The result is certified against the KKT
conditions of the original nonsmooth problem (L1 subgradient, cap
multiplier, bounds) [derived: API_SPEC 3.7]. Three golden cases pin it:
`minvar_turnover_l1`, `minvar_turnover_capped`, `meanvar_turnover_box`.
At n = 49 the split problem took about 600 s per solve in the Python
reference, which is why the real-data study ran it on 10 industries
only [measured: `docs/results/real_data_study_dev.md`].

## 5. Error handling and convergence reporting

### 5.1 Policy

Three rules apply in every language:

1. **Bad input fails fast.** Shape, finiteness and symmetry checks run
   before any arithmetic (`check_square_symmetric`, `validate_returns`
   including the `r > -1` rule, the view, budget and bound validators,
   finite `rf`, positive `periods`), and the Cholesky pivot supplies the
   positive-definiteness check. Error messages carry the offending
   numbers (pivot value, row index, bound sums, attainable return range).
2. **Diagnostics variants report, wrappers raise.** Each iterative
   solver has a `*_solve` variant that returns its convergence evidence
   and a weights-only wrapper that raises when that evidence fails:

   | Solver | Diagnostics variant returns | Weights-only wrapper |
   |---|---|---|
   | Dual active-set QP | `QpResult(weights, iterations, converged, kkt_residual)` from `min_variance_longonly_solve`, `constrained_frontier_solve` | `min_variance_longonly`, `constrained_frontier_weights`, `constrained_frontier` raise `ValueError` when `converged` is false |
   | CCD risk budgeting | `(weights, sweeps, converged)` from `risk_budget_solve` | `risk_budget_weights`, `erc_weights` raise `ValueError` when `converged` is false (1.2.0) |

3. **No wrong answer leaves silently.** An uncertified QP point or an
   unconverged CCD iterate is available only through the diagnostics
   variant, where the flag travels with it.

### 5.2 What the convergence flags mean

* **QP.** `converged` is a KKT certificate evaluated on the polished
  point, independent of how the loop ended: primal feasibility (every
  equality residual and bound violation at most `1e-9 * (1 + max|b|)`),
  stationarity (`|Sigma w - N u|_inf` at most `1e-9 * (1 + scale)`), and,
  since 1.2.0, dual feasibility (every polished multiplier on an active
  inequality at least `-1e-9 * (1 + scale)`, the stationarity scale;
  `_kkt_certificate` in `markowitz.py`). Complementary slackness holds by construction, because only
  active constraints carry multipliers and active bounds are set exactly.
  `kkt_residual` is the largest of these violations. The certificate is
  computed before the final clip of `w` to `[lb, ub]`. LEARN 4.2 gives
  the conditions and a worked check.
* **CCD.** `converged` means `max_i |y_i - y_i_prev| < tol` after a full
  sweep, within `max_sweeps` (default 10000). The tolerance is absolute,
  so it is not a statement about the risk-contribution error; check
  `risk_contributions` when the scale of `Sigma` is unusual.

### 5.3 Mechanics per language

Identical conditions, idiomatic mechanics (API_SPEC section 7 has the
full condition table):

| Language | Invalid input | Tangency undefined or ill-conditioned | Non-convergence |
|---|---|---|---|
| Python | `ValueError` | `TangencyUndefinedError(ValueError)` | flag in the `*_solve` result; wrappers raise `ValueError` |
| C++ | `std::invalid_argument` (shape, NaN, asymmetry) or `std::domain_error` (not positive definite, degenerate) | `TangencyUndefinedError : std::domain_error` | flag in `QpResult` / `RiskBudgetResult`; wrappers throw `std::domain_error` |
| Rust | `Err(PortOptError::InvalidInput / NotPositiveDefinite / Undefined / Infeasible)`, no panics on data | `Err(PortOptError::TangencyUndefined)` | flag in `QpResult` / `CcdResult`; wrappers return `Err(Undefined)` |
| Java | `IllegalArgumentException` | `Markowitz.TangencyUndefinedException extends IllegalArgumentException` | flag in `QpResult` / `Solution`; wrappers throw |

## 6. Testing strategy

### 6.1 Golden cases and their tolerances

`data/golden/golden.json` holds 26 cases in a flat
name / inputs / expect / tol schema, read by numpy-free parsers in each
port; every suite asserts the case count. Each value is checked with an
absolute tolerance, `abs(got - expect) <= tol`
(`python/tests/test_golden.py::check`). Tolerances per case family, read
from the file:

| Family | Cases | tol |
|---|---|---|
| Sample moments | `sample_moments_daily` | 1e-12 |
| Ledoit-Wolf shrinkage | `shrinkage_const_corr_daily` | 1e-10 |
| Closed forms | `two_fund_scalars_annualized`, `minvar_unconstrained`, `tangency_unconstrained`, `frontier_point_t080` | 1e-10 |
| Dual active-set QP | `minvar_longonly_qp`, `minvar_box_qp`, `frontier_constrained_t080`, `frontier_constrained_box_t060` | 1e-10 |
| Cholesky | `cholesky_annualized_cov` | 1e-12 |
| Simplex projection | `simplex_projection_hand`, `simplex_projection_box` | 1e-12 |
| Black-Litterman | `implied_pi`, `bl_posterior_returns`, `bl_posterior_weights` | 1e-10 |
| Black-Litterman Omega | `bl_omega_default` | 1e-12 |
| Inverse volatility | `inverse_vol` | 1e-12 |
| CCD risk parity | `erc_weights`, `risk_budget_pinned` | 1e-8 |
| Backtest metrics | `backtest_equal_weight`, `backtest_erc` | 1e-8 |
| Backtest turnover | `turnover_minvar_longonly` | 1e-9 |

```mermaid
flowchart TD
    GEN["data/generate_data.py (seed 42)"] --> CSV["asset_returns.csv written with %.17g,<br/>then re-read from disk"]
    CSV --> REF["Python reference portopt computes each case"]
    REF --> VAL{"Validation gate"}
    VAL --> V1["closed forms vs numpy.linalg.lstsq"]
    VAL --> V2["Ledoit-Wolf intensity vs covCor.m transcription, 1e-14"]
    VAL --> V3["QP vs scipy SLSQP: objective not worse, weights within 1e-6"]
    VAL --> V4["ERC equal contributions, risk budgets met, BL limits"]
    V1 --> GJ
    V2 --> GJ
    V3 --> GJ
    V4 --> GJ
    GJ["golden.json: 26 cases, name, inputs, expect, tol"]
    GJ --> T1["Python: python/tests/test_golden.py"]
    GJ --> T2["C++: cpp/tests/test_golden.cpp"]
    GJ --> T3["Rust: rust/tests/golden.rs"]
    GJ --> T4["Java: GoldenTest.java"]
    T1 --> CHK["each value: abs(got - expect) at most tol"]
    T2 --> CHK
    T3 --> CHK
    T4 --> CHK
```

### 6.2 What is cross-checked against outside references

`data/generate_data.py` refuses to write golden values unless these
checks pass (read from the script):

* closed-form min-variance and tangency weights against
  `numpy.linalg.lstsq` solutions (`atol=1e-11`), and A, B, C against
  direct sums (1e-10);
* the four QP cases against scipy SLSQP: the active-set objective must
  not exceed SLSQP's by more than 1e-12, weights must agree within 1e-6,
  frontier targets must be met within 1e-12, and the KKT residual must be
  below 1e-12;
* the Ledoit-Wolf intensity against a direct transcription of the
  authors' `covCor.m`, within 1e-14;
* ERC risk contributions equal within 1e-9 and risk budgets met within
  1e-9;
* the Black-Litterman limits: `tau -> 0` with fixed Omega returns pi
  within 1e-8, and `Omega = 1e12 * I` returns pi within 1e-10.

The Python suite also compares ERC and the QP with scipy
(`test_erc_matches_scipy`, the SLSQP comparisons in `test_markowitz.py`).

### 6.3 Other test layers

1. **Identity and property checks** (grid-style loops): frontier
   variance matches `(Cm^2 - 2Am + B)/D`; tangency has the highest Sharpe
   on a frontier grid; risk contributions sum to 1 at 1e-12; ERC is
   permutation-equivariant and scale-invariant; the QP equals the closed
   form when no bound binds, satisfies the long-only KKT conditions, and
   beats projected random feasible points; shrinkage repairs a singular
   T <= N sample; the backtest accounting identity holds.
2. **Limit and stability tests**: Black-Litterman with huge Omega or
   fixed Omega and `tau -> 0` returns the prior; ERC with an almost-zero
   volatility asset converges; the one-ulp perturbation test above.
3. **Error-path tests**: each row of the API_SPEC section 7 table has a
   test asserting the right exception or variant, plus determinism (two
   runs, bit-identical results).

The Python suite has 117 tests (`cd python && python -m pytest -q`,
measured for this revision). COOKBOOK.md's eighteen Python recipes were
extracted and run for this revision; the C++, Rust and Java fragments
were not re-compiled.

### 6.4 Known gaps

From the project review of 2026-10-31 (findings 15 and 16), still open
in this revision (the review's third item, a missing length check in the
Rust `check_vec` helper, is fixed: `rust/tests/golden.rs` now counts the
expected keys):

* Matrices are spot-checked rather than compared in full: 3 of 36
  Cholesky entries, 4 shrinkage entries, the Omega diagonal only.
* All tolerances are absolute; the ERC tolerance of 1e-8 on weights near
  0.1 is loose next to the agreement the ports achieve.
* Tests are example-based. There is no hypothesis (Python), proptest
  (Rust) or jqwik (Java) suite, no randomised QP-versus-SLSQP sweep in
  the suite, no fuzzing of the three hand-written parsers, and no
  independent reference for the backtest metrics.

## 7. Continuous integration

`.github/workflows/ci.yml` runs on every push to `main`, every pull
request and on manual dispatch. Each language job runs its golden suite
against the same `golden.json`; the parity job builds all four demos and
`scripts/demo_parity.py` requires every line of the C++, Rust and Java
output to match the Python demo after normalising line endings, trailing
whitespace and the title's dash. CodeQL scanning runs separately
(`.github/workflows/codeql.yml`).

```mermaid
flowchart TD
    TRIG["push to main, pull request, manual dispatch"] --> PY
    TRIG --> CPP
    TRIG --> SAN
    TRIG --> RS
    TRIG --> MSRV
    TRIG --> JV
    TRIG --> PAR
    PY["python: 3.11 and 3.13 on Ubuntu, Windows, macOS<br/>ruff, mypy, pytest, demo"]
    CPP["cpp: C++26 on Ubuntu (GCC 14), Windows (MSVC), macOS<br/>configure with PORTOPT_WERROR, build, ctest, demo"]
    SAN["cpp-sanitizers: Clang ASan and UBSan, Debug, ctest"]
    RS["rust: stable on three OS<br/>fmt, clippy, test, doc, demo"]
    MSRV["rust-msrv: cargo test on 1.78"]
    JV["java: 25 and 27 on three OS<br/>mvn verify, demo"]
    PAR["demo-parity: build and run all four demos on Ubuntu"]
    PAR --> DIFF["scripts/demo_parity.py: every line must match the Python demo"]
    GOLD["data/golden/golden.json"] -.-> PY
    GOLD -.-> CPP
    GOLD -.-> RS
    GOLD -.-> JV
```

Build entry points per language: `pip install -e "python[dev]"` (Python
>= 3.11, `python/pyproject.toml`); `cmake -S cpp -B cpp/build` and
`cmake --build cpp/build --config Release` (CMake >= 3.25, C++26 via
`CMAKE_CXX_STANDARD 26`, `/std:c++latest` on MSVC, GoogleTest fetched
with `FetchContent` when not installed); `cd rust && cargo test
--release` (`rust-version = 1.78`); `cd java && mvn -B verify` (Java 25
via `maven.compiler.release`).

## 8. Performance

The implementations are written for reproducibility, not speed, and are
dense throughout.

* **Cholesky** is the textbook O(n^3) dense factorisation (about n^3/3
  flops). Every solve is O(n^2). There is no blocking and no BLAS.
* **The QP** stores each bound constraint as a dense n-vector. In every
  iteration it recomputes `Sigma^{-1} n_c` for every active constraint
  (q solves of O(n^2) each), rebuilds the q x q Gram matrix and
  refactors it from scratch (O(q^3)), and allocates new arrays. With q up
  to about n active constraints that is roughly O(q n^2 + q^3) per
  iteration, and the iteration count grows with n, so the total is about
  O(n^4) [derived: `_dual_active_set_qp`]. On the bundled 8-asset data
  the long-only GMV takes 4 iterations, the box problem 8 and the 8%
  frontier point 9 (COOKBOOK 17).
* **Ledoit-Wolf** is O(T n^2) with explicit Python loops over asset pairs
  and correctly rounded `math.fsum` reductions over T.
* **CCD** costs O(n^2) per sweep thanks to the incremental `Sigma y`
  cache; it takes 14 sweeps on the annualised bundled covariance.

Kernel benchmarks in all four languages are in
[`docs/results/benchmarks.md`](results/benchmarks.md), with the harnesses
`python/benchmarks/bench_kernels.py`, `cpp/benchmarks/bench_kernels.cpp`
(opt-in, `PORTOPT_BUILD_BENCHMARKS`), `rust/benches/kernels.rs` and the
Java `bench` Maven profile; CI builds and smoke-runs each. On one
non-idle laptop the long-only QP took 460 us at n = 8 and 830 ms at
n = 50 in Python, against about 8 us and 2.9 to 5.7 ms in the ports
[measured]. In 1.3.0 the default Python `shrink_covariance` evaluates
its sums on numpy tiles (2.4x to 22x faster than the loop reference,
agreeing to 1e-13 relative), and the C++ backtest hands the weight
function a non-owning `HistoryView` instead of a copy. The QP still
refactors its working set each iteration; incremental updates remain
open. The Python suite has 349 tests at this revision
(`pytest --collect-only`).

## 9. Thread safety

None of the four implementations holds global mutable state. We checked
by grep for this revision: the Python modules define only constants and
`__all__` at module level; the C++ sources contain no `static` variables
(namespace-scope values are `constexpr`); the Rust crate has no
`static mut`, `lazy_static`, `OnceLock`, `RefCell`, `Mutex` or atomics
and only `const` items; the Java classes have no non-final static
fields. Every function computes from its arguments and returns new
results.

Concurrent calls on different inputs are therefore safe. Two caveats:
the inputs themselves are shared, so a caller must not mutate an array
or matrix while another thread is reading it; and the backtest callback
`weight_fn` is the caller's code, so any state it keeps (the demo's
fallback counter, for example) is the caller's responsibility. No test
exercises concurrent use.

## 10. Diagram sources

The Mermaid sources live in [`diagrams/`](diagrams/): `dataflow.mmd`,
`modules.mmd`, `backtest_sequence.mmd`, `qp_active_set.mmd`,
`risk_parity_ccd.mmd`, `black_litterman_flow.mmd`, `golden_parity.mmd`,
`ci_pipeline.mmd`, `real_data_holdout.mmd` and `forward_ledger.mmd`. The `.mmd` files are the sources; this document
carries inline copies in `mermaid` code blocks, which GitHub renders. No
automated check renders or validates them, so edit the `.mmd` file and
the inline copy together.
