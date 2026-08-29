# COOKBOOK — portopt recipes

Task-oriented recipes in all four languages. Python snippets run as-is
from the repo root with `cd python && PYTHONPATH=src python3 <file>`
(data paths are relative to `python/`). C++/Rust/Java snippets follow
the API_SPEC naming contract (`portopt::` namespace, `portopt` crate
with `Result<_, PortOptError>`, `com.quant.portopt` package); matrices
are `std::vector<std::vector<double>>` in C++, `Vec<Vec<double>>`-style
`Vec<Vec<f64>>` in Rust, `double[][]` in Java, row-major `T x N`
returns everywhere. Error behavior per language: Python `ValueError`,
C++ `std::invalid_argument`/`std::domain_error`, Rust
`Err(PortOptError)`, Java `IllegalArgumentException`.

---

## 1. How do I load the bundled data and compute annualized moments?

**Python**

```python
import numpy as np
import portopt as po

x = np.loadtxt("../data/asset_returns.csv", delimiter=",", skiprows=1)
mu = po.annualize_mean(po.sample_mean(x))      # 252 * daily mean
sigma = po.annualize_cov(po.sample_cov(x))     # 252 * ddof=1 covariance
print(mu[0])                                   # US_EQ: 0.09397769...
print(np.sqrt(sigma[0, 0]))                    # US_EQ vol: 0.15353...
```

**C++**

```cpp
#include <portopt/estimators.hpp>
#include <portopt/io.hpp>

auto x = portopt::load_returns_csv("data/asset_returns.csv"); // T x N
auto mu = portopt::annualize_mean(portopt::sample_mean(x));
auto sigma = portopt::annualize_cov(portopt::sample_cov(x));
```

**Rust**

```rust
use portopt::{annualize_cov, annualize_mean, load_returns_csv, sample_cov, sample_mean};

let x = load_returns_csv("data/asset_returns.csv")?; // Vec<Vec<f64>>, T x N
let mu = annualize_mean(&sample_mean(&x)?);
let sigma = annualize_cov(&sample_cov(&x)?);
```

**Java**

```java
import com.quant.portopt.Estimators;
import com.quant.portopt.DataIo;

double[][] x = DataIo.loadReturnsCsv("data/asset_returns.csv");
double[] mu = Estimators.annualizeMean(Estimators.sampleMean(x));
double[][] sigma = Estimators.annualizeCov(Estimators.sampleCov(x));
```

---

## 2. How do I shrink the covariance matrix (Ledoit-Wolf)?

Shrinkage is computed on the **daily** returns; annualize the result.

**Python**

```python
res = po.shrink_covariance(x)
print(res.intensity)        # 0.009430492389752111 (golden, 1e-10)
print(res.rbar)             # 0.41719371710084957
sigma_shrunk = po.annualize_cov(res.sigma)  # use in place of sample cov
```

**C++**

```cpp
portopt::ShrinkageResult res = portopt::shrink_covariance(x);
// res.sigma, res.intensity, res.rbar, res.sample, res.target
auto sigma_shrunk = portopt::annualize_cov(res.sigma);
```

**Rust**

```rust
let res = portopt::shrink_covariance(&x)?;
// res.sigma, res.intensity, res.rbar, res.sample, res.target
let sigma_shrunk = portopt::annualize_cov(&res.sigma);
```

**Java**

```java
ShrinkageResult res = Estimators.shrinkCovariance(x);
double[][] sigmaShrunk = Estimators.annualizeCov(res.sigma);
System.out.println(res.intensity);  // 0.00943049...
```

---

## 3. How do I compute minimum-variance and tangency portfolios?

**Python**

```python
w_mv = po.min_variance_weights(sigma)         # S^-1 1 / C
w_tan = po.tangency_weights(mu, sigma, 0.02)  # raises if rf >= A/C
sc = po.two_fund_scalars(mu, sigma)
print(sc.a / sc.c)  # min-variance return 0.032234971725...
```

**C++**

```cpp
#include <portopt/markowitz.hpp>

auto w_mv = portopt::min_variance_weights(sigma);
auto w_tan = portopt::tangency_weights(mu, sigma, 0.02); // throws std::domain_error if undefined
portopt::TwoFundScalars sc = portopt::two_fund_scalars(mu, sigma); // sc.a .b .c .d
```

**Rust**

```rust
let w_mv = portopt::min_variance_weights(&sigma)?;
let w_tan = portopt::tangency_weights(&mu, &sigma, 0.02)?; // Err if A - rf*C <= 0
let sc = portopt::two_fund_scalars(&mu, &sigma)?;          // sc.a, sc.b, sc.c, sc.d
```

**Java**

```java
import com.quant.portopt.Markowitz;

double[] wMv = Markowitz.minVarianceWeights(sigma);
double[] wTan = Markowitz.tangencyWeights(mu, sigma, 0.02);
TwoFundScalars sc = Markowitz.twoFundScalars(mu, sigma);
```

---

## 4. How do I sweep the efficient frontier?

Pinned grid: 25 equally spaced targets from the min-variance return
`A/C` to `max(mu)`, for both the unconstrained (closed-form) and
constrained (penalty-PGD, penalty 1e4) sweeps.

**Python**

```python
targets, weights, variances = po.efficient_frontier(mu, sigma, 25)
t_c, w_c, v_c = po.constrained_frontier(mu, sigma, 25)   # long-only
for t, v, vc in zip(targets[::6], variances[::6], v_c[::6]):
    print(f"{t:.4f}  {v**0.5:.4f}  {vc**0.5:.4f}")
```

**C++**

```cpp
portopt::FrontierResult fr = portopt::efficient_frontier(mu, sigma, 25);
portopt::FrontierResult fc = portopt::constrained_frontier(mu, sigma, 25);
// fr.targets[k], fr.weights[k], fr.variances[k]
```

**Rust**

```rust
let fr = portopt::efficient_frontier(&mu, &sigma, 25)?;
let fc = portopt::constrained_frontier(&mu, &sigma, 25)?;
// fr.targets, fr.weights, fr.variances
```

**Java**

```java
FrontierResult fr = Markowitz.efficientFrontier(mu, sigma, 25);
FrontierResult fc = Markowitz.constrainedFrontier(mu, sigma, 25);
```

---

## 5. How do I solve a long-only min-variance with box bounds?

**Python**

```python
w_lo = po.min_variance_longonly(sigma)                 # lb=0, ub=inf
w_box = po.min_variance_longonly(sigma, lb=0.02, ub=0.40)
print(w_lo)  # golden minvar_longonly_pgd: 73.2% US_BOND, zeros on EU_EQ/EM_EQ
```

**C++**

```cpp
auto w_lo = portopt::min_variance_longonly(sigma);           // defaults
auto w_box = portopt::min_variance_longonly(sigma, 0.02, 0.40);
```

**Rust**

```rust
let w_lo = portopt::min_variance_longonly(&sigma, None, None)?;
let w_box = portopt::min_variance_longonly(&sigma, Some(0.02), Some(0.40))?;
```

**Java**

```java
double[] wLo = Markowitz.minVarianceLongOnly(sigma);
double[] wBox = Markowitz.minVarianceLongOnly(sigma, 0.02, 0.40);
```

Infeasible bounds (`sum(lb) > 1`, `sum(ub) < 1`, or `lb_i > ub_i`) fail
fast in the projection with a descriptive error.

---

## 6. How do I repair a weight vector onto the simplex?

The exact sort-based projection is exposed directly — useful for
renormalizing screened or rounded weights.

**Python**

```python
w = po.project_simplex_box([0.9, 0.5, -0.2])          # -> [0.7, 0.3, 0.0]
w2 = po.project_simplex_box([0.8, 0.6, 0.4, -0.3], lb=0.05, ub=0.5)
# -> [0.5, 0.325, 0.125, 0.05]  (golden simplex_projection_box)
```

**C++**

```cpp
auto w = portopt::project_simplex_box({0.9, 0.5, -0.2}, 1.0);
auto w2 = portopt::project_simplex_box({0.8, 0.6, 0.4, -0.3}, 1.0, 0.05, 0.5);
```

**Rust**

```rust
let w = portopt::project_simplex_box(&[0.9, 0.5, -0.2], 1.0, None, None)?;
let w2 = portopt::project_simplex_box(&[0.8, 0.6, 0.4, -0.3], 1.0,
                                      Some(0.05), Some(0.5))?;
```

**Java**

```java
double[] w = Markowitz.projectSimplexBox(new double[]{0.9, 0.5, -0.2}, 1.0);
double[] w2 = Markowitz.projectSimplexBox(
    new double[]{0.8, 0.6, 0.4, -0.3}, 1.0, 0.05, 0.5);
```

---

## 7. How do I compute implied equilibrium returns and run Black-Litterman?

**Python**

```python
import json

caps = np.loadtxt("../data/market_caps.csv", delimiter=",", skiprows=1, usecols=1)
with open("../data/views.json") as fh:
    doc = json.load(fh)
names = open("../data/asset_returns.csv").readline().strip().split(",")

w_mkt = caps / caps.sum()
pi = po.implied_equilibrium_returns(sigma, w_mkt, doc["delta"])  # delta*Sigma*w
p, q, conf = po.views_to_matrices(doc["views"], names)
w_bl, res = po.bl_weights(sigma, pi, p, q, doc["tau"], confidences=conf)
print(res.mu_post[4])   # EM_EQ posterior 0.069397813... (golden 1e-10)
print(w_bl[0])          # 0.417177201... (golden bl_posterior_weights)
```

**C++**

```cpp
#include <portopt/black_litterman.hpp>

auto pi = portopt::implied_equilibrium_returns(sigma, w_mkt, 2.5);
auto [p, q, conf] = portopt::views_to_matrices(views, names);
auto [w_bl, res] = portopt::bl_weights(sigma, pi, p, q, 0.05, 0.0, conf);
// res.mu_post, res.sigma_post, res.m, res.omega
```

**Rust**

```rust
let pi = portopt::implied_equilibrium_returns(&sigma, &w_mkt, 2.5)?;
let (p, q, conf) = portopt::views_to_matrices(&views, &names)?;
let (w_bl, res) = portopt::bl_weights(&sigma, &pi, &p, &q, 0.05, 0.0,
                                      None, Some(&conf))?;
```

**Java**

```java
import com.quant.portopt.BlackLitterman;

double[] pi = BlackLitterman.impliedEquilibriumReturns(sigma, wMkt, 2.5);
ViewMatrices vm = BlackLitterman.viewsToMatrices(views, names);
BLWeights bl = BlackLitterman.blWeights(sigma, pi, vm.p, vm.q, 0.05, 0.0, vm.confidences);
```

Note: `bl_weights` pins `rf = 0` because pi and the posterior are
*excess* returns; sanity check — with views that restate equilibrium
(or near-zero confidence) the weights collapse back to `w_mkt`.

---

## 8. How do I control the confidence of a Black-Litterman view?

Confidence scales the Idzorek-style default
`Omega_kk = [P tau Sigma P']_kk / c_k`: large `c` ⇒ view taken almost
literally; tiny `c` ⇒ view ignored.

**Python**

```python
for c in (0.01, 1.0, 100.0):
    w, res = po.bl_weights(sigma, pi, p, q, 0.05, confidences=[c, c])
    print(f"c={c:>6}: EM_EQ mu_post={res.mu_post[4]:.4f}  w={w[4]:.4f}")
# c=0.01 -> near equilibrium; c=100 -> EM_EQ posterior near the 10% view
```

**C++**

```cpp
std::vector<double> conf = {100.0, 100.0};       // near-dogmatic views
auto omega = portopt::default_omega(sigma, p, 0.05, conf);
auto res = portopt::bl_posterior(sigma, pi, p, q, 0.05, omega);
```

**Rust**

```rust
let omega = portopt::default_omega(&sigma, &p, 0.05, Some(&[100.0, 100.0]))?;
let res = portopt::bl_posterior(&sigma, &pi, &p, &q, 0.05, Some(&omega), None)?;
```

**Java**

```java
double[][] omega = BlackLitterman.defaultOmega(sigma, vm.p, 0.05,
                                               new double[]{100.0, 100.0});
BLResult res = BlackLitterman.blPosterior(sigma, pi, vm.p, vm.q, 0.05, omega);
```

---

## 9. How do I compute ERC / risk-budget weights and verify contributions?

**Python**

```python
w_erc = po.erc_weights(sigma)
rc = po.risk_contributions(w_erc, sigma)
print(rc)                          # all ~0.125; spread ~1e-12
w_rb = po.risk_budget_weights(sigma, [0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05])
w_iv = po.inverse_vol_weights(sigma)
w, sweeps, converged = po.risk_budget_solve(sigma)   # diagnostics variant
```

**C++**

```cpp
#include <portopt/risk_parity.hpp>

auto w_erc = portopt::erc_weights(sigma);
auto rc = portopt::risk_contributions(w_erc, sigma);
std::vector<double> b = {0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05};
auto w_rb = portopt::risk_budget_weights(sigma, b);
portopt::CcdResult r = portopt::risk_budget_solve(sigma, b); // w, sweeps, converged
```

**Rust**

```rust
let w_erc = portopt::erc_weights(&sigma)?;
let rc = portopt::risk_contributions(&w_erc, &sigma)?;
let b = [0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05];
let r = portopt::risk_budget_solve(&sigma, Some(&b), 1e-10, 10_000)?;
// r.weights, r.sweeps, r.converged
```

**Java**

```java
import com.quant.portopt.RiskParity;

double[] wErc = RiskParity.ercWeights(sigma);
double[] rc = RiskParity.riskContributions(wErc, sigma);
double[] b = {0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05};
double[] wRb = RiskParity.riskBudgetWeights(sigma, b);
```

Budgets are normalized internally; non-convergence within 10000 sweeps
is reported via the `converged` flag, never thrown.

---

## 10. How do I run the quarterly backtest and read the turnover?

**Python**

```python
res = po.run_backtest(
    x,
    lambda h: po.erc_weights(po.sample_cov(h)),  # weight_fn sees rows [0, t_k)
    rf=0.02, window=500, step=63,
)
print(res.sharpe, res.max_drawdown)   # 0.63652214, 0.10188233 (golden 1e-8)
print(res.rebalance_dates)            # [500, 563, ..., 1193] — 12 dates
print(res.avg_turnover)               # mean L1 distance between target vectors
```

**C++**

```cpp
#include <portopt/backtest.hpp>

auto res = portopt::run_backtest(x, [](const portopt::Matrix& h) {
    return portopt::erc_weights(portopt::sample_cov(h));
}, 0.02, 500, 63);
// res.ann_return, res.ann_vol, res.sharpe, res.max_drawdown, res.avg_turnover
```

**Rust**

```rust
let res = portopt::run_backtest(&x, |h| {
    portopt::erc_weights(&portopt::sample_cov(h)?)
}, 0.02, 500, 63)?;
```

**Java**

```java
import com.quant.portopt.Backtest;

BacktestResult res = Backtest.runBacktest(
    x, h -> RiskParity.ercWeights(Estimators.sampleCov(h)), 0.02, 500, 63);
```

For a tangency strategy, wrap the weight function with a fallback to
min-variance when tangency is undefined (API_SPEC 6.3) — see
`python/demo.py::tangency_fn` for the pinned pattern.

---

## 11. How do I handle the standard failure modes?

Duplicated asset (singular covariance) and rf above every mean:

**Python**

```python
bad = np.column_stack([x, x[:, 0]])            # duplicate US_EQ
try:
    po.sample_cov(bad) @ np.ones(9)            # cov itself is fine...
    po.min_variance_weights(po.sample_cov(bad))  # ...Cholesky pivot catches it
except ValueError as e:
    print(e)   # "matrix is not positive definite (... duplicated asset)"

try:
    po.tangency_weights(mu, sigma, rf=0.50)    # rf above min-var return
except ValueError as e:
    print(e)   # "tangency undefined: A - rf*C <= 0 ..."
```

**C++**

```cpp
try {
    auto w = portopt::min_variance_weights(bad_cov);
} catch (const std::invalid_argument& e) { /* singular covariance */ }
try {
    auto w = portopt::tangency_weights(mu, sigma, 0.50);
} catch (const std::domain_error& e) { /* tangency undefined */ }
```

**Rust**

```rust
match portopt::min_variance_weights(&bad_cov) {
    Err(PortOptError::NotPositiveDefinite(msg)) => eprintln!("{msg}"),
    other => { let _ = other?; }
}
```

**Java**

```java
try {
    Markowitz.tangencyWeights(mu, sigma, 0.50);
} catch (IllegalArgumentException e) {
    System.out.println(e.getMessage()); // tangency undefined
}
```

---

## 12. How do I verify my build against the golden values?

Every port ships a golden test suite, but you can spot-check by hand.

**Python**

```python
import json
gold = json.load(open("../data/golden/golden.json"))
case = next(c for c in gold["cases"] if c["name"] == "erc_weights")
w = po.erc_weights(po.annualize_cov(po.sample_cov(x)))
for i in range(8):
    assert abs(w[i] - case["expect"][f"w_{i}"]) <= case["tol"]
```

**C++** — `ctest --test-dir build --output-on-failure` runs the golden
suite (`tests/`, flat-schema JSON reader).

**Rust** — `cargo test --release` includes `tests/golden.rs` via
`serde_json`.

**Java** — `./test.sh` runs all `*Test` classes including the golden
suite (bundled minimal JSON parser).

To regenerate data + golden values after an intentional change:
`python3 data/generate_data.py` (revalidates closed forms, the ERC
identity, and both BL limits before writing).
