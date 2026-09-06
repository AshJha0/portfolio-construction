# Portfolio Construction (Markowitz, Black-Litterman, Risk Parity)

A four-language reference implementation of classical portfolio construction:
sample and Ledoit-Wolf shrinkage estimators, closed-form and box-constrained
Markowitz mean-variance optimization (a finite dual active-set QP),
Black-Litterman posterior blending, risk parity / risk budgeting via cyclical
coordinate descent, and a quarterly expanding-window backtest — all on a
bundled 8-asset universe (5 equity indices including one EM, 2 bond proxies,
1 FX carry proxy) with 1250 days of deterministic daily returns.

Every numerical algorithm is **pinned** in [API_SPEC.md](API_SPEC.md) —
Cholesky variant, simplex-projection breakpoint scan, the dual active-set
QP and its KKT certificate, the CCD update, Ledoit-Wolf sums term by term —
so the scipy/LAPACK-free ports agree with the Python reference to the
tolerances in `data/golden/golden.json`.

| Language | Package / namespace | Location |
|---|---|---|
| Python 3.11 | `portopt` (src layout) | `python/` |
| C++17 | `portopt::` | `cpp/` |
| Rust (edition 2021) | crate `portopt` | `rust/` |
| Java 21 | `com.quant.portopt` | `java/` |

## Feature matrix

| Capability | Functions (Python names) | Notes |
|---|---|---|
| Sample moments | `sample_mean`, `sample_cov` | mean divisor T, covariance ddof=1; correctly rounded sums |
| Ledoit-Wolf shrinkage | `shrink_covariance` | constant-correlation target; intensity exactly per Ledoit & Wolf (2004) App. B / `covCor.m`, applied in the ddof=1 convention |
| Annualization | `annualize_mean`, `annualize_cov` | multiply by 252 (`periods > 0` validated) |
| Pinned linear algebra | `cholesky`, `chol_solve`, `spd_solve`, `spd_inverse` | Cholesky-Banachiewicz, relative pivot gate `1e-13`; every inverse is Cholesky solves |
| Two-fund closed forms | `two_fund_scalars`, `min_variance_weights`, `tangency_weights`, `gross_leverage`, `frontier_weights`, `efficient_frontier` | A/B/C/D scalars; tangency conditioning guard (rf within 1e-6 of A/C raises `TangencyUndefinedError`); 25-point grid, g/h computed once |
| Constrained optimization | `project_simplex_box`, `attainable_return_range`, `min_variance_longonly[_solve]`, `constrained_frontier_solve`, `constrained_frontier_weights`, `constrained_frontier` | exact sort-based box-simplex projection; Goldfarb-Idnani dual active-set QP with closed-form polish and a KKT certificate (`QpResult.converged`, `kkt_residual`); target return met exactly (no penalty) |
| Black-Litterman | `implied_equilibrium_returns`, `views_to_matrices`, `default_omega`, `bl_posterior`, `bl_weights` | reverse optimization, absolute + relative views (validated), He-Litterman Omega scaled by per-view confidence, one k×k Cholesky solve |
| Risk parity / budgeting | `risk_contributions`, `erc_weights`, `risk_budget_solve`, `risk_budget_weights`, `inverse_vol_weights` | CCD log-barrier update (Spinu 2013), tol 1e-10, ≤10000 sweeps; scale-invariant, permutation-equivariant |
| Backtest-lite | `rebalance_indices`, `run_backtest`, `annualized_return`, `annualized_vol`, `sharpe_ratio`, `max_drawdown` | expanding window 500, step 63, L1 turnover between target vectors; `T >= window + 2`, `r > -1`, finite `rf` validated |

Edge cases handled with clear errors: singular covariance (duplicated
asset) caught by the Cholesky pivot, n=1 trivial weights, tangency
undefined or ill-conditioned when rf sits at/near the min-variance return,
degenerate frontier (mu ∝ 1), target below the min-variance return, empty
frontier grid, unattainable constrained target, infeasible box bounds,
empty/zero-row/same-leg/unknown-asset views, tau outside (0, 1], zero-vol
Sharpe, returns ≤ −1, non-finite rf. The CCD sweep cap is reported via a
flag; the QP returns a KKT certificate and the weights-only wrappers raise
when it fails — a non-certified solution is never returned as if converged.

## Directory layout

```
portfolio-construction/
  README.md            this file
  LEARN.md             theory: mean-variance, shrinkage, BL, risk parity, desk usage
  COOKBOOK.md          task-oriented recipes in all four languages (compile-checked)
  API_SPEC.md          the pinned numerical contract all ports follow
  docs/
    ARCHITECTURE.md    component design, data flow, numerics, testing
    GITHUB_PAGES.md    publishing guide for the docs site: https://ashjha0.github.io/portfolio-construction/
    diagrams/*.mmd     raw Mermaid sources
  data/
    asset_returns.csv  1250 x 8 daily simple returns (deterministic)
    market_caps.csv    market caps -> w_mkt for reverse optimization
    views.json         example absolute + relative BL views
    config.json        shared constants (252, rf=0.02, tau=0.05, ...)
    generate_data.py   regenerates everything (numpy default_rng(42))
    golden/golden.json 23 cross-language golden cases
  python/  src/portopt/*.py, tests/, demo.py
  cpp/     CMakeLists.txt, include/portopt/*.hpp, src/, tests/, build.sh
  rust/    Cargo.toml, src/lib.rs, src/bin/demo.rs, tests/
  java/    src/main|test/java/com/quant/portopt/, build.sh, test.sh, demo.sh
```

## Build, run, test

All commands below work on a fresh Linux clone (Python 3.11 + numpy;
pytest and scipy for the Python tests only; CMake ≥ 3.16, a C++17
compiler and GoogleTest; cargo; JDK 21 with JUnit 4 at
`/usr/share/java/junit4.jar` and Hamcrest at `/usr/share/java/hamcrest.jar`).

### Python

```bash
cd python
PYTHONPATH=src python3 demo.py           # end-to-end demo (~0.3 s)
PYTHONPATH=src python3 -m pytest -q      # 114 tests incl. the golden suite (~1 s)
```

### C++ (C++17, CMake ≥ 3.16, GoogleTest)

```bash
cd cpp
bash build.sh                            # cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j2
./build/portopt_demo                     # demo (data resolved at ../data)
ctest --test-dir build --output-on-failure   # 90 tests
```

### Rust (edition 2021)

```bash
cd rust
cargo build --release                    # zero warnings
cargo run --release --bin demo
cargo test --release                     # 58 tests (23 unit + 2 golden + 33 property)
```

### Java (21, JUnit 4 + Hamcrest — no Maven/Gradle)

```bash
cd java
bash build.sh    # javac -Xlint:all -Werror everything into out/
bash demo.sh     # runs the demo main
bash test.sh     # JUnitCore on all *Test classes: 108 tests
```

## Demo output (Python reference, truncated)

```
portopt demo — 8-asset universe, 1250 daily returns
assets: US_EQ, EU_EQ, JP_EQ, UK_EQ, EM_EQ, US_BOND, IG_CORP, FX_CARRY

=== Estimators (annualized, Ledoit-Wolf constant-correlation shrinkage)
shrinkage intensity delta = 0.007212   rbar = 0.4172
asset          mean  vol(sample)  vol(shrunk)
US_EQ        0.0940       0.1535       0.1535
EU_EQ        0.0341       0.1555       0.1555
...
US_BOND      0.0248       0.0481       0.0481

=== Efficient frontier sweeps (25 points each, showing every 4th) =====
unconstrained grid: A/C .. max(mu);  long-only grid: constrained GMV .. max(mu)
  target  vol uncon    target  vol long-only
  0.0322     0.0400    0.0304         0.0404
  0.0631     0.0545    0.0622         0.0826
  0.0940     0.0843    0.0940         0.1535
long-only targets met exactly: max |mu'w - target| = 2.8e-17
tangency (rf=2.00%): sharpe = 0.8865, gross leverage = 4.14

=== Black-Litterman: prior (equilibrium) vs posterior weights =========
view: EM_EQ returns 10.0% (confidence 1.0)
view: US_EQ beats EU_EQ by 2.0% (confidence 0.5)
                  US_EQ    EU_EQ    JP_EQ    UK_EQ    EM_EQ  US_BOND  IG_CORP FX_CARRY
pi (equil.)      0.0326   0.0321   0.0279   0.0281   0.0385   0.0001   0.0064   0.0097
mu posterior     0.0531   0.0459   0.0430   0.0439   0.0694  -0.0007   0.0093   0.0149
w market         0.2479   0.1653   0.0661   0.0496   0.0579   0.2066   0.1240   0.0826
w posterior      0.4172  -0.1024   0.0504   0.0378   0.2824   0.1574   0.0944   0.0629
posterior gross leverage = 1.2049

=== Risk contributions: ERC vs inverse-vol ============================
w ERC            0.0579   0.0568   0.0620   0.0617   0.0444   0.4037   0.1756   0.1379
RC ERC           0.1250   0.1250   0.1250   0.1250   0.1250   0.1250   0.1250   0.1250
RC spread: ERC 1.30e-12, inv-vol 1.20e-01

=== Quarterly-rebalance backtest, expanding window (holdout = last 750d)
method          ann ret  ann vol   sharpe   max DD  avg turno
equal_weight     0.0614   0.0930   0.4454   0.1868     0.0000
min_var_lo       0.0479   0.0402   0.6948   0.0589     0.0273
tangency         0.4835   0.5749   0.8062   0.4700    18.1968
black_litt       0.0701   0.1070   0.4687   0.2331     0.0174
erc              0.0528   0.0516   0.6365   0.1019     0.0051
inverse_vol      0.0564   0.0630   0.5783   0.1235     0.0043
note: tangency undefined/ill-conditioned at 8 rebalance date(s) (rf at or above the min-var return); fell back to min-variance there.
```

The C++, Rust and Java demos print the same numbers (differences are
limited to the 1e-17 residual line and float formatting). The
raw-tangency turnover of 18.2 per rebalance versus 0.005 for ERC is the
whole estimation-error story in one line; see LEARN.md.

## Golden values — cross-language testing

`data/golden/golden.json` holds 23 named cases with a flat schema
(`{"cases": [{"name", "inputs", "expect", "tol"}]}`); vectors are
flattened as `w_0..w_7`, matrix entries as `cov_i_j`. All values were
produced by the Python reference on `data/asset_returns.csv` **as
parsed** (the CSV round-trips doubles exactly via `%.17g`). Before writing,
the generator revalidates the closed forms against `numpy.linalg.lstsq`,
the Ledoit-Wolf intensity against a transcription of the authors'
`covCor.m`, the four QP cases against scipy SLSQP (with KKT residual
< 1e-12), the ERC equal-contribution identity, and both Black-Litterman
limits.

Every language's test suite loads the same file, asserts the exact case
count, and checks `|got - expect| <= tol` per entry — analytic quantities
at 1e-10 to 1e-12, the QP cases at 1e-10, CCD and backtest metrics at
1e-8/1e-9. Every golden quantity is a closed form or the exact solution
of a finite algorithm (the QP result is polished on its final active set),
and all T-length sums are correctly rounded (Python) or compensated
(ports); agreement is therefore a rounding question, not an
iteration-path question, and the suites verify that scaling the input by
`1 + 2^-52` moves the QP goldens by less than 1e-13. Regenerate the
dataset and golden values with:

```bash
python3 data/generate_data.py
```

Determinism: the library contains no RNG anywhere; repeated runs are
bit-identical. Only the data generator uses randomness
(`numpy default_rng(seed=42)`).

## Real-world usage notes

What a practitioner is getting, precisely:

- **Units and conventions.** Inputs are daily *simple* returns as
  decimals (0.01 = 1%); `annualize_*` multiplies by 252 (equity
  convention — FX/rates desks using 260 or 365 must pass `periods`
  explicitly). Covariance is ddof=1. Weights sum to 1; a negative weight
  is a short. `rf` and all returns are annual decimals. Black-Litterman
  works entirely in excess-return space (`pi = delta Sigma w_mkt`), so
  its tangency is pinned at `rf = 0`.
- **What is validated at every public entry point.** NaN/inf anywhere;
  simple returns ≤ −1; shape mismatches; asymmetric or non-SPD covariance
  (Cholesky pivot gate, relative 1e-13); `periods <= 0`; non-finite `rf`;
  infeasible or infinite lower bounds; unattainable target returns;
  degenerate frontiers; malformed views (unknown asset, missing key,
  `long == short`, NaN q, non-positive confidence); `tau` outside (0, 1];
  `delta <= 0`; empty schedules. Errors are the language's standard
  invalid-input type with a message that names the offending quantity.
- **Convergence reporting.** The QP solvers return a KKT certificate
  (`converged`, `kkt_residual`, `iterations`); weights-only wrappers raise
  rather than return an uncertified point. CCD returns `(weights, sweeps,
  converged)`; the sweep cap (10000) is reported, not raised.
- **Parameter bounds you should respect.** n ≤ ~20 assets (dense O(n³)
  Cholesky per solve; nothing is blocked or sparse); T ≥ 2 rows (T ≥ N+1
  for a non-singular sample covariance — Ledoit-Wolf repairs T ≤ N);
  bounds `lb` finite, `ub` may be +∞, `sum(lb) ≤ 1 ≤ sum(ub)`; `tau ∈ (0,1]`;
  confidences > 0. Unconstrained tangency/BL weights are unbounded in
  principle: check `gross_leverage` (the demo tangency carries 4.1×, and
  the backtest tangency turns over 18× NAV per quarter).
- **Deliberately out of scope.** No calendars or day-count conventions
  (the backtest is index-based: rows are trading days, rebalances at
  fixed row offsets); no transaction-cost model (turnover is reported,
  never charged); no missing-data handling (panels must be complete); no
  leverage or turnover constraints inside the optimizer beyond the box;
  no Idzorek confidence calibration (the BL Omega is He-Litterman's
  diagonal scaled by a per-view divisor); no condition-number diagnostics
  beyond the Cholesky pivot gate (κ up to ~1e13 passes — shrink first if
  your covariance is that ill-conditioned); single-period objectives only.
- **What is verified.** Closed forms against `numpy.linalg.lstsq`; the
  Ledoit-Wolf intensity against `covCor.m`; the QP against scipy SLSQP,
  brute-force active-set enumeration (n = 4) and ~1500 random problems
  with n ≤ 20, κ ≤ 1e5, tied returns, tight caps and fixed assets (no
  failures); ERC against the equal-contribution identity and SLSQP;
  cross-language agreement on 23 golden cases; determinism and
  input-perturbation stability. This is a verified reference
  implementation of the published algorithms — not a production
  portfolio system (see "out of scope").

## References

Papers and books the code implements or the documentation relies on:

- H. Markowitz (1952), "Portfolio Selection", *The Journal of Finance*
  7(1), 77–91. doi:10.2307/2975974
- R. C. Merton (1972), "An Analytic Derivation of the Efficient Portfolio
  Frontier", *Journal of Financial and Quantitative Analysis* 7(4),
  1851–1872. doi:10.2307/2329621
- O. Ledoit, M. Wolf (2004), "Honey, I Shrunk the Sample Covariance
  Matrix", *The Journal of Portfolio Management* 30(4), 110–119.
  doi:10.3905/jpm.2004.110 (constant-correlation estimator, Appendix B;
  reference code `covCor.m` at https://www.econ.uzh.ch/en/people/faculty/wolf/publications.html)
- F. Black, R. Litterman (1992), "Global Portfolio Optimization",
  *Financial Analysts Journal* 48(5), 28–43. doi:10.2469/faj.v48.n5.28
- G. He, R. Litterman (1999), "The Intuition Behind Black-Litterman Model
  Portfolios", Goldman Sachs Investment Management Research; SSRN 334304.
  https://ssrn.com/abstract=334304 (default `Omega = diag(P tau Sigma P')`)
- T. M. Idzorek (2005), "A Step-by-Step Guide to the Black-Litterman
  Model", Zephyr Associates working paper (cited only to state that its
  confidence calibration is *not* what this code implements).
- S. Maillard, T. Roncalli, J. Teiletche (2010), "The Properties of Equally
  Weighted Risk Contribution Portfolios", *The Journal of Portfolio
  Management* 36(4), 60–70. doi:10.3905/jpm.2010.36.4.060
- F. Spinu (2013), "An Algorithm for Computing Risk Parity Weights",
  SSRN 2297383. https://ssrn.com/abstract=2297383 (log-barrier
  formulation and the positive-root coordinate update used here)
- T. Griveau-Billion, J.-C. Richard, T. Roncalli (2013), "A Fast Algorithm
  for Computing High-dimensional Risk Parity Portfolios", SSRN 2325255.
  https://ssrn.com/abstract=2325255 (cyclical coordinate descent for risk
  parity)
- D. Goldfarb, A. Idnani (1983), "A numerically stable dual method for
  solving strictly convex quadratic programs", *Mathematical Programming*
  27, 1–33. doi:10.1007/BF02591962 (the constrained QP solver)
- J. Duchi, S. Shalev-Shwartz, Y. Singer, T. Chandra (2008), "Efficient
  Projections onto the l1-Ball for Learning in High Dimensions",
  *Proceedings of the 25th International Conference on Machine Learning
  (ICML)*, 272–279. doi:10.1145/1390156.1390191 (sorted simplex projection)
- K. C. Kiwiel (2008), "Breakpoint searching algorithms for the continuous
  quadratic knapsack problem", *Mathematical Programming* 112, 473–491.
  doi:10.1007/s10107-006-0050-z (box-constrained breakpoint variant)
- R. Jagannathan, T. Ma (2003), "Risk Reduction in Large Portfolios: Why
  Imposing the Wrong Constraints Helps", *The Journal of Finance* 58(4),
  1651–1683. doi:10.1111/1540-6261.00580 (LEARN.md only)
- R. O. Michaud (1989), "The Markowitz Optimization Enigma: Is 'Optimized'
  Optimal?", *Financial Analysts Journal* 45(1), 31–42.
  doi:10.2469/faj.v45.n1.31 (LEARN.md only)
- V. K. Chopra, W. T. Ziemba (1993), "The Effect of Errors in Means,
  Variances, and Covariances on Optimal Portfolio Choice", *The Journal of
  Portfolio Management* 19(2), 6–11. doi:10.3905/jpm.1993.409440 (LEARN.md only)
- A. W. Lo (2002), "The Statistics of Sharpe Ratios", *Financial Analysts
  Journal* 58(4), 36–52. doi:10.2469/faj.v58.n4.2453 (Sharpe annualization
  and standard error, LEARN.md)
- A. Frazzini, L. H. Pedersen (2014), "Betting Against Beta", *Journal of
  Financial Economics* 111(1), 1–25. doi:10.1016/j.jfineco.2013.10.005
  (LEARN.md only)
- V. DeMiguel, L. Garlappi, R. Uppal (2009), "Optimal Versus Naive
  Diversification: How Inefficient is the 1/N Portfolio Strategy?", *The
  Review of Financial Studies* 22(5), 1915–1953. doi:10.1093/rfs/hhm075
  (LEARN.md only)
- A. Neumaier (1974), "Rundungsfehleranalyse einiger Verfahren zur
  Summation endlicher Summen", *ZAMM* 54(1), 39–51.
  doi:10.1002/zamm.19740540106 (the compensated summation used by the ports)

## License

MIT License — see [LICENSE](LICENSE). Provided "as is", without warranty
of any kind; not investment advice.
