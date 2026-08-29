# Portfolio Construction (Markowitz, Black-Litterman, Risk Parity)

A four-language reference implementation of classical portfolio construction:
sample and Ledoit-Wolf shrinkage estimators, closed-form and constrained
Markowitz mean-variance optimization, Black-Litterman posterior blending,
risk parity / risk budgeting via cyclical coordinate descent, and a
quarterly expanding-window backtest — all on a bundled 8-asset universe
(5 equity indices including one EM, 2 bond proxies, 1 FX carry proxy)
with 1250 days of deterministic daily returns.

Every numerical algorithm is **pinned** in [API_SPEC.md](API_SPEC.md) —
Cholesky variant, simplex-projection breakpoint scan, PGD backtracking
rule, CCD update, Ledoit-Wolf sums term by term — so the scipy/LAPACK-free
ports agree with the Python reference to the tolerances in
`data/golden/golden.json`.

| Language | Package / namespace | Location |
|---|---|---|
| Python 3.11 | `portopt` (src layout) | `python/` |
| C++17 | `portopt::` | `cpp/` |
| Rust (edition 2021) | crate `portopt` | `rust/` |
| Java 21 | `com.quant.portopt` | `java/` |

## Feature matrix

| Capability | Functions (Python names) | Notes |
|---|---|---|
| Sample moments | `sample_mean`, `sample_cov` | mean divisor T, covariance ddof=1 |
| Ledoit-Wolf shrinkage | `shrink_covariance` | constant-correlation target, closed-form intensity (LW 2004), pinned term by term |
| Annualization | `annualize_mean`, `annualize_cov` | multiply by 252 |
| Pinned linear algebra | `cholesky`, `chol_solve`, `spd_solve`, `spd_inverse` | Cholesky-Banachiewicz, relative pivot gate `1e-13`; every inverse is Cholesky solves |
| Two-fund closed forms | `two_fund_scalars`, `min_variance_weights`, `tangency_weights`, `frontier_weights`, `efficient_frontier` | A/B/C/D scalars; 25-point pinned target grid |
| Constrained optimization | `project_simplex_box`, `pgd_minimize`, `min_variance_longonly`, `constrained_frontier_weights`, `constrained_frontier` | exact sort-based box-simplex projection; deterministic PGD, no RNG |
| Black-Litterman | `implied_equilibrium_returns`, `views_to_matrices`, `default_omega`, `bl_posterior`, `bl_weights` | reverse optimization, absolute + relative views, Idzorek-style Omega with per-view confidence, one k×k Cholesky solve |
| Risk parity / budgeting | `risk_contributions`, `erc_weights`, `risk_budget_solve`, `risk_budget_weights`, `inverse_vol_weights` | CCD log-barrier update, tol 1e-10, ≤10000 sweeps |
| Backtest-lite | `rebalance_indices`, `run_backtest`, `annualized_return`, `annualized_vol`, `sharpe_ratio`, `max_drawdown` | expanding window 500, step 63, L1 turnover between target vectors |

Edge cases handled with clear errors: singular covariance (duplicated
asset) caught by the Cholesky pivot, n=1 trivial weights, tangency
undefined when rf sits above the min-variance return, degenerate frontier
(mu ∝ 1), target below the min-variance return, infeasible box bounds,
empty/zero-row view matrices, tau outside (0, 1], zero-vol Sharpe.
Iteration caps in PGD/CCD are reported via flags, never raised.

## Directory layout

```
10-portfolio-construction/
  README.md            this file
  LEARN.md             theory: mean-variance, shrinkage, BL, risk parity
  COOKBOOK.md          task-oriented recipes in all four languages
  API_SPEC.md          the pinned numerical contract all ports follow
  docs/
    ARCHITECTURE.md    component design, data flow, numerics, testing
    diagrams/*.mmd     raw Mermaid sources
  data/
    asset_returns.csv  1250 x 8 daily simple returns (deterministic)
    market_caps.csv    market caps -> w_mkt for reverse optimization
    views.json         example absolute + relative BL views
    config.json        shared constants (252, rf=0.02, tau=0.05, ...)
    generate_data.py   regenerates everything (numpy default_rng(42))
    golden/golden.json 21 cross-language golden cases
  python/  src/portopt/*.py, tests/, demo.py
  cpp/     CMakeLists.txt, include/portopt/*.hpp, src/, tests/, build.sh
  rust/    Cargo.toml, src/lib.rs, src/bin/demo.rs, tests/
  java/    src/main|test/java/com/quant/portopt/, build.sh, test.sh, demo.sh
```

## Build, run, test

### Python

```bash
cd python
PYTHONPATH=src python3 demo.py       # end-to-end demo
PYTHONPATH=src pytest -q             # 85 tests incl. golden suite
```

### C++ (C++17, CMake, GoogleTest)

```bash
cd cpp
./build.sh    # cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j2
./build/demo
ctest --test-dir build --output-on-failure
```

### Rust (edition 2021)

```bash
cd rust
cargo build --release
cargo run --release --bin demo
cargo test --release
```

### Java (21, JUnit 4 at /usr/share/java/junit4.jar — no Maven/Gradle)

```bash
cd java
./build.sh    # javac everything into out/
./demo.sh     # runs the demo main
./test.sh     # JUnitCore on all *Test classes
```

## Demo output (Python reference, truncated)

```
portopt demo — 8-asset universe, 1250 daily returns
assets: US_EQ, EU_EQ, JP_EQ, UK_EQ, EM_EQ, US_BOND, IG_CORP, FX_CARRY

=== Estimators (annualized, Ledoit-Wolf constant-correlation shrinkage)
shrinkage intensity delta = 0.009430   rbar = 0.4172
asset          mean  vol(sample)  vol(shrunk)
US_EQ        0.0940       0.1535       0.1535
EU_EQ        0.0341       0.1555       0.1555
...
US_BOND      0.0248       0.0481       0.0481

=== Efficient frontier sweep (25 points, showing every 4th) ===========
  target  vol uncon  vol long-only
  0.0322     0.0400         0.0407
  0.0631     0.0545         0.0845
  0.0940     0.0843         0.1534
tangency (rf=2.00%): sharpe = 0.8865

=== Black-Litterman: prior (equilibrium) vs posterior weights =========
view: EM_EQ returns 10.0% (confidence 1.0)
view: US_EQ beats EU_EQ by 2.0% (confidence 0.5)
                  US_EQ    EU_EQ    JP_EQ    UK_EQ    EM_EQ  US_BOND  IG_CORP FX_CARRY
pi (equil.)      0.0326   0.0321   0.0279   0.0281   0.0385   0.0001   0.0064   0.0097
mu posterior     0.0531   0.0459   0.0430   0.0439   0.0694  -0.0007   0.0093   0.0149
w market         0.2479   0.1653   0.0661   0.0496   0.0579   0.2066   0.1240   0.0826
w posterior      0.4172  -0.1024   0.0504   0.0378   0.2824   0.1574   0.0944   0.0629

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
note: tangency undefined at 8 rebalance date(s) (rf above min-var return); fell back to min-variance there.
```

(The raw-tangency turnover of 18.2 per rebalance versus 0.005 for ERC is
the whole estimation-error story in one line; see LEARN.md.)

## Golden values — cross-language testing

`data/golden/golden.json` holds 21 named cases with a flat schema
(`{"cases": [{"name", "inputs", "expect", "tol"}]}`); vectors are
flattened as `w_0..w_7`, matrix entries as `cov_i_j`. All values were
produced by the Python reference on `data/asset_returns.csv` **as
parsed** (the CSV round-trips doubles exactly via `%.17g`), and the
generator revalidates the closed forms against `numpy.linalg.lstsq`, the
ERC equal-contribution identity, and both Black-Litterman limits before
writing.

Every language's test suite loads the same file and asserts
`|got - expect| <= tol` per entry — analytic quantities at 1e-10 to
1e-12, iterative solvers (PGD, CCD) at 1e-8. Because the algorithms are
pinned to the operation level (same factorization, same sweep order,
same backtracking rule), agreement is limited only by IEEE-754 rounding,
not by algorithmic drift. Regenerate the dataset and golden values with:

```bash
python3 data/generate_data.py
```

Determinism: the library contains no RNG anywhere; repeated runs are
bit-identical. Only the data generator uses randomness
(`numpy default_rng(seed=42)`).

## License

Educational use. No warranty; not investment advice.
