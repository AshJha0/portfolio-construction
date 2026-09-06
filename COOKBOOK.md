# COOKBOOK — portopt recipes

Task-oriented recipes in all four languages. Python snippets run as-is
from the repo root with `cd python && PYTHONPATH=src python3 <file>`
(data paths are relative to `python/`). The C++/Rust/Java snippets are
fragments of a single program per language (the preamble in recipe 1
declares `x`, `mu`, `sigma`, `names`, `w_mkt`, `views`); every fragment
has been extracted, compiled against the shipped library and run.
Types: C++ `portopt::Matrix` (row-major, `m(i, j)`) and
`portopt::Vector = std::vector<double>`; Rust `portopt::Matrix` (`get`,
`set`, `row`, `matvec`) and `Vec<f64>`; Java `double[][]` / `double[]`;
row-major `T x N` returns everywhere. Error behavior per language:
Python `ValueError` (tangency: `TangencyUndefinedError`), C++
`std::invalid_argument`/`std::domain_error` (tangency:
`portopt::TangencyUndefinedError`), Rust `Err(PortOptError::...)`, Java
`IllegalArgumentException` (tangency: `Markowitz.TangencyUndefinedException`).

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
#include <portopt/backtest.hpp>
#include <portopt/black_litterman.hpp>
#include <portopt/data_io.hpp>
#include <portopt/estimators.hpp>
#include <portopt/markowitz.hpp>
#include <portopt/risk_parity.hpp>

#include <cstdio>

using portopt::Matrix;
using portopt::Vector;

int main() {
  const portopt::ReturnsData ret = portopt::load_returns_csv("../data/asset_returns.csv");
  const Matrix& x = ret.returns;                       // T x N
  const std::vector<std::string>& names = ret.names;
  const Vector mu = portopt::annualize_mean(portopt::sample_mean(x));   // periods = 252
  const Matrix sigma = portopt::annualize_cov(portopt::sample_cov(x));
  const portopt::MarketCaps caps = portopt::load_market_caps("../data/market_caps.csv");
  Vector w_mkt(caps.caps);
  double cap_sum = 0.0;
  for (double c : caps.caps) cap_sum += c;
  for (double& v : w_mkt) v /= cap_sum;
  const portopt::ViewsDocument views_doc = portopt::load_views("../data/views.json");
  const std::vector<portopt::View>& views = views_doc.views;
  // ... recipes 2-11 go here ...
  return 0;
}
```

**Rust**

```rust
use std::path::Path;

use portopt::{Matrix, PortOptError};

fn main() -> Result<(), PortOptError> {
    let (names, x) = portopt::load_returns_csv(Path::new("../data/asset_returns.csv"))?;
    let mu = portopt::annualize_mean(&portopt::sample_mean(&x)?, 252.0)?;
    let sigma = portopt::annualize_cov(&portopt::sample_cov(&x)?, 252.0)?;
    let (_, caps) = portopt::load_market_caps(Path::new("../data/market_caps.csv"))?;
    let cap_sum: f64 = caps.iter().sum();
    let w_mkt: Vec<f64> = caps.iter().map(|c| c / cap_sum).collect();
    let views_doc = portopt::load_views(Path::new("../data/views.json"))?;
    let views = &views_doc.views;
    // ... recipes 2-11 go here ...
    Ok(())
}
```

**Java** (the loaders `DataFiles`/`MiniJson` are package-private helpers
used by the demo; a client parses its own files — here with plain Java)

```java
import com.quant.portopt.Backtest;
import com.quant.portopt.BlackLitterman;
import com.quant.portopt.Estimators;
import com.quant.portopt.Linalg;
import com.quant.portopt.Markowitz;
import com.quant.portopt.RiskParity;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

public class Cookbook {
    public static void main(String[] args) throws Exception {
        List<String> lines = Files.readAllLines(Paths.get("../data/asset_returns.csv"));
        List<String> names = List.of(lines.get(0).trim().split(","));
        double[][] x = new double[lines.size() - 1][];
        for (int r = 1; r < lines.size(); r++) {
            String[] cells = lines.get(r).trim().split(",");
            x[r - 1] = new double[cells.length];
            for (int c = 0; c < cells.length; c++) {
                x[r - 1][c] = Double.parseDouble(cells[c]);
            }
        }
        double[] mu = Estimators.annualizeMean(Estimators.sampleMean(x), 252);
        double[][] sigma = Estimators.annualizeCov(Estimators.sampleCov(x), 252);
        double[] caps = {30, 20, 8, 6, 7, 25, 15, 10};   // market_caps.csv
        double[] wMkt = new double[caps.length];
        for (int i = 0; i < caps.length; i++) {
            wMkt[i] = caps[i] / 121.0;
        }
        List<BlackLitterman.View> views = List.of(     // views.json
                BlackLitterman.View.absolute("EM_EQ", 0.10, 1.0),
                BlackLitterman.View.relative("US_EQ", "EU_EQ", 0.02, 0.5));
        // ... recipes 2-11 go here ...
    }
}
```

---

## 2. How do I shrink the covariance matrix (Ledoit-Wolf)?

Shrinkage is computed on the **daily** returns; annualize the result.
The intensity is exactly the Ledoit & Wolf (2004) estimator; the matrices
are returned in the ddof=1 convention.

**Python**

```python
res = po.shrink_covariance(x)
print(res.intensity)        # 0.007211765225479597 (golden, 1e-10)
print(res.rbar)             # 0.4171937171008496
sigma_shrunk = po.annualize_cov(res.sigma)  # use in place of sample cov
```

**C++**

```cpp
  const portopt::ShrinkageResult shrink = portopt::shrink_covariance(x);
  // shrink.sigma, shrink.intensity, shrink.rbar, shrink.sample, shrink.target
  const Matrix sigma_shrunk = portopt::annualize_cov(shrink.sigma);
```

**Rust**

```rust
    let res = portopt::shrink_covariance(&x)?;
    // res.sigma, res.intensity, res.rbar, res.sample, res.target
    let sigma_shrunk = portopt::annualize_cov(&res.sigma, 252.0)?;
```

**Java**

```java
        Estimators.ShrinkageResult shrink = Estimators.shrinkCovariance(x);
        double[][] sigmaShrunk = Estimators.annualizeCov(shrink.sigma(), 252);
        System.out.println(shrink.intensity());  // 0.00721176...
```

---

## 3. How do I compute minimum-variance and tangency portfolios?

**Python**

```python
w_mv = po.min_variance_weights(sigma)         # S^-1 1 / C
w_tan = po.tangency_weights(mu, sigma, 0.02)  # TangencyUndefinedError if rf >= A/C (or within 1e-6 of it)
print(po.gross_leverage(w_tan))               # 4.14 — cap this before it reaches an OMS
sc = po.two_fund_scalars(mu, sigma)
print(sc.a / sc.c)  # min-variance return 0.032234971725...
```

**C++**

```cpp
  const Vector w_mv = portopt::min_variance_weights(sigma);
  const Vector w_tan = portopt::tangency_weights(mu, sigma, 0.02);  // throws portopt::TangencyUndefinedError
  const double leverage = portopt::gross_leverage(w_tan);           // 4.14
  const portopt::TwoFundScalars sc = portopt::two_fund_scalars(mu, sigma);  // sc.a .b .c .d
  std::printf("A/C = %.6f, tangency leverage %.2f\n", sc.a / sc.c, leverage);
```

**Rust**

```rust
    let w_mv = portopt::min_variance_weights(&sigma)?;
    let w_tan = portopt::tangency_weights(&mu, &sigma, 0.02)?; // Err(TangencyUndefined) if A - rf*C <= 0
    let leverage = portopt::gross_leverage(&w_tan)?;            // 4.14
    let sc = portopt::two_fund_scalars(&mu, &sigma)?;           // sc.a, sc.b, sc.c, sc.d
```

**Java**

```java
        double[] wMv = Markowitz.minVarianceWeights(sigma);
        double[] wTan = Markowitz.tangencyWeights(mu, sigma, 0.02);
        double leverage = Markowitz.grossLeverage(wTan);            // 4.14
        Markowitz.TwoFundScalars sc = Markowitz.twoFundScalars(mu, sigma);
```

---

## 4. How do I sweep the efficient frontier?

Pinned grids: 25 equally spaced targets from the min-variance return
`A/C` to `max(mu)` for the unconstrained (closed-form) sweep, and from the
constrained min-variance return to the maximum attainable return for the
constrained (active-set QP) sweep. Each constrained point meets its target
exactly and is certified optimal (a failed certificate raises).

**Python**

```python
targets, weights, variances = po.efficient_frontier(mu, sigma, 25)
t_c, w_c, v_c = po.constrained_frontier(mu, sigma, 25)   # long-only
for t, v, tc, vc in zip(targets[::6], variances[::6], t_c[::6], v_c[::6]):
    print(f"{t:.4f}  {v**0.5:.4f}    {tc:.4f}  {vc**0.5:.4f}")
```

**C++**

```cpp
  const portopt::FrontierResult fr = portopt::efficient_frontier(mu, sigma, 25);
  const portopt::FrontierResult fc = portopt::constrained_frontier(mu, sigma, 25);
  // fr.targets[k], fr.weights(k, i) / fr.weights.row(k), fr.variances[k]
```

**Rust**

```rust
    let fr = portopt::efficient_frontier(&mu, &sigma, 25)?;
    let fc = portopt::constrained_frontier(&mu, &sigma, 25, None, None)?;
    // fr.targets, fr.weights (Vec<Vec<f64>>), fr.variances
```

**Java**

```java
        Markowitz.Frontier fr = Markowitz.efficientFrontier(mu, sigma, 25);
        Markowitz.Frontier fc = Markowitz.constrainedFrontier(mu, sigma, 25);
        // fr.targets(), fr.weights()[k], fr.variances()
```

---

## 5. How do I solve a long-only min-variance with box bounds?

**Python**

```python
w_lo = po.min_variance_longonly(sigma)                 # lb=0, ub=inf
w_box = po.min_variance_longonly(sigma, lb=0.02, ub=0.40)
res = po.min_variance_longonly_solve(sigma, lb=0.02, ub=0.40)   # diagnostics
print(res.converged, res.iterations, res.kkt_residual)          # True, ~5, ~1e-17
print(w_lo)  # golden minvar_longonly_qp: 73.2% US_BOND, zeros on EU_EQ/EM_EQ
```

**C++** (bound vectors have length n; empty means the default)

```cpp
  const std::size_t n = sigma.rows();
  const Vector w_lo = portopt::min_variance_longonly(sigma);
  const Vector w_box = portopt::min_variance_longonly(sigma, Vector(n, 0.02), Vector(n, 0.40));
  const portopt::QpResult qp = portopt::min_variance_longonly_solve(sigma, Vector(n, 0.02), Vector(n, 0.40));
  // qp.weights, qp.iterations, qp.converged, qp.kkt_residual
```

**Rust**

```rust
    let n = sigma.nrows();
    let w_lo = portopt::min_variance_longonly(&sigma, None, None)?;
    let w_box = portopt::min_variance_longonly(&sigma, Some(&vec![0.02; n]), Some(&vec![0.40; n]))?;
    let qp = portopt::min_variance_longonly_solve(&sigma, Some(&vec![0.02; n]), Some(&vec![0.40; n]), portopt::QP_MAX_ITER)?;
    // qp.weights, qp.iterations, qp.converged, qp.kkt_residual
```

**Java** (a length-1 bound array broadcasts to every asset)

```java
        double[] wLo = Markowitz.minVarianceLongOnly(sigma);
        double[] wBox = Markowitz.minVarianceLongOnly(sigma, new double[] {0.02}, new double[] {0.40});
        Markowitz.QpResult qp = Markowitz.minVarianceLongOnlySolve(
                sigma, new double[] {0.02}, new double[] {0.40}, Markowitz.QP_MAX_ITER);
        // qp.weights(), qp.iterations(), qp.converged(), qp.kktResidual()
```

Infeasible bounds (`sum(lb) > 1`, `sum(ub) < 1`, `lb_i > ub_i`, or an
infinite lower bound) fail fast with a descriptive error. A frontier
point at an explicit target uses the same bounds:
`constrained_frontier_weights(mu, sigma, 0.06, lb=0.02, ub=0.40)` (raises
if the target is below the constrained min-variance return or above the
attainable maximum — check `attainable_return_range(mu, lb, ub)` first).

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
  const Vector w = portopt::project_simplex_box(Vector{0.9, 0.5, -0.2});
  const Vector w2 = portopt::project_simplex_box(Vector{0.8, 0.6, 0.4, -0.3}, 1.0, 0.05, 0.5);
```

**Rust**

```rust
    let w = portopt::project_simplex_box(&[0.9, 0.5, -0.2], 1.0, None, None)?;
    let w2 = portopt::project_simplex_box(&[0.8, 0.6, 0.4, -0.3], 1.0,
                                          Some(&[0.05; 4]), Some(&[0.5; 4]))?;
```

**Java**

```java
        double[] w = Markowitz.projectSimplexBox(new double[] {0.9, 0.5, -0.2}, 1.0);
        double[] w2 = Markowitz.projectSimplexBox(
                new double[] {0.8, 0.6, 0.4, -0.3}, 1.0, new double[] {0.05}, new double[] {0.5});
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
  const Vector pi = portopt::implied_equilibrium_returns(sigma, w_mkt, 2.5);
  const portopt::ViewMatrices vm = portopt::views_to_matrices(views, names);
  portopt::BLResult bl;
  const Vector w_bl = portopt::bl_weights(sigma, pi, vm.p, vm.q, 0.05, 0.0, vm.confidences, &bl);
  // bl.mu_post, bl.sigma_post, bl.m, bl.omega
```

**Rust**

```rust
    let pi = portopt::implied_equilibrium_returns(&sigma, &w_mkt, 2.5)?;
    let (p, q, conf) = portopt::views_to_matrices(views, &names)?;
    let (w_bl, res) = portopt::bl_weights(&sigma, &pi, &p, &q, 0.05, 0.0, None, Some(&conf))?;
    // res.mu_post, res.sigma_post, res.m, res.omega
```

**Java**

```java
        double[] pi = BlackLitterman.impliedEquilibriumReturns(sigma, wMkt, 2.5);
        BlackLitterman.ViewMatrices vm = BlackLitterman.viewsToMatrices(views, names);
        double[] wBl = BlackLitterman.blWeights(sigma, pi, vm.p(), vm.q(), 0.05, vm.confidences());
        BlackLitterman.Result bl = BlackLitterman.blPosterior(sigma, pi, vm.p(), vm.q(), 0.05, vm.confidences());
        // bl.muPost(), bl.sigmaPost(), bl.m(), bl.omega()
```

Note: `bl_weights` pins `rf = 0` because pi and the posterior are
*excess* returns; sanity check — with views that restate equilibrium
(or near-zero confidence) the weights collapse back to `w_mkt`. Views
are validated: an unknown asset, a missing key, `long == short`, or a
NaN `q` is an error, never a silent row.

---

## 8. How do I control the confidence of a Black-Litterman view?

Confidence scales the He-Litterman default
`Omega_kk = [P tau Sigma P']_kk / c_k`: large `c` ⇒ view taken almost
literally; tiny `c` ⇒ view ignored. An explicit `Omega` replaces the
default entirely and cannot be combined with confidences.

**Python**

```python
for c in (0.01, 1.0, 100.0):
    w, res = po.bl_weights(sigma, pi, p, q, 0.05, confidences=[c, c])
    print(f"c={c:>6}: EM_EQ mu_post={res.mu_post[4]:.4f}  w={w[4]:.4f}")
# c=0.01 -> near equilibrium; c=100 -> EM_EQ posterior near the 10% view
omega = po.default_omega(sigma, p, 0.05, [100.0, 100.0])
res = po.bl_posterior(sigma, pi, p, q, 0.05, omega=omega)   # explicit Omega
```

**C++**

```cpp
  const Vector conf = {100.0, 100.0};       // near-dogmatic views
  const Matrix omega = portopt::default_omega(sigma, vm.p, 0.05, conf);
  const portopt::BLResult dogmatic = portopt::bl_posterior_with_omega(sigma, pi, vm.p, vm.q, 0.05, omega);
  const Vector w_dogmatic = portopt::bl_weights_with_omega(sigma, pi, vm.p, vm.q, 0.05, omega);
```

**Rust**

```rust
    let omega = portopt::default_omega(&sigma, &p, 0.05, Some(&[100.0, 100.0]))?;
    let dogmatic = portopt::bl_posterior(&sigma, &pi, &p, &q, 0.05, Some(&omega), None)?;
```

**Java**

```java
        double[][] omega = BlackLitterman.defaultOmega(sigma, vm.p(), 0.05,
                                                       new double[] {100.0, 100.0});
        BlackLitterman.Result dogmatic = BlackLitterman.blPosterior(sigma, pi, vm.p(), vm.q(), 0.05, omega, null);
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
  const Vector w_erc = portopt::erc_weights(sigma);
  const Vector rc = portopt::risk_contributions(w_erc, sigma);
  const Vector b = {0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05};
  const Vector w_rb = portopt::risk_budget_weights(sigma, b);
  const portopt::RiskBudgetResult r = portopt::risk_budget_solve(sigma, b);  // r.weights, r.sweeps, r.converged
```

**Rust**

```rust
    let w_erc = portopt::erc_weights(&sigma)?;
    let rc = portopt::risk_contributions(&w_erc, &sigma)?;
    let b = [0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05];
    let w_rb = portopt::risk_budget_weights(&sigma, Some(&b))?;
    let r = portopt::risk_budget_solve(&sigma, Some(&b), portopt::CCD_TOL, portopt::CCD_MAX_SWEEPS)?;
    // r.weights, r.sweeps, r.converged
```

**Java**

```java
        double[] wErc = RiskParity.ercWeights(sigma);
        double[] rc = RiskParity.riskContributions(wErc, sigma);
        double[] b = {0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05};
        double[] wRb = RiskParity.riskBudgetWeights(sigma, b);
        RiskParity.Solution sol = RiskParity.riskBudgetSolve(sigma, b, RiskParity.CCD_TOL, RiskParity.CCD_MAX_SWEEPS);
```

Budgets are normalized internally; non-convergence within 10000 sweeps
is reported via the `converged` flag, never thrown.

---

## 10. How do I run the quarterly backtest and read the turnover?

**Python**

```python
res = po.run_backtest(
    x,
    lambda h: po.erc_weights(po.sample_cov(h)),  # weight_fn sees rows [0, t_k) (read-only)
    rf=0.02, window=500, step=63,
)
print(res.sharpe, res.max_drawdown)   # 0.63652214, 0.10188233 (golden 1e-8)
print(res.rebalance_dates)            # [500, 563, ..., 1193] — 12 dates
print(res.avg_turnover)               # mean L1 distance between target vectors
```

**C++**

```cpp
  const portopt::BacktestResult bt = portopt::run_backtest(
      x, [](const Matrix& h) { return portopt::erc_weights(portopt::sample_cov(h)); },
      0.02, 500, 63);
  // bt.ann_return, bt.ann_vol, bt.sharpe, bt.max_drawdown, bt.avg_turnover
```

**Rust** (the weight function is fallible, so `?` works inside it)

```rust
    let bt = portopt::run_backtest(
        &x,
        |h: &Matrix| portopt::erc_weights(&portopt::sample_cov(h)?),
        0.02, 500, 63,
    )?;
    // bt.ann_return, bt.ann_vol, bt.sharpe, bt.max_drawdown, bt.avg_turnover
```

**Java**

```java
        Backtest.Result bt = Backtest.runBacktest(
                x, h -> RiskParity.ercWeights(Estimators.sampleCov(h)), 0.02, 500, 63);
        // bt.annReturn(), bt.annVol(), bt.sharpe(), bt.maxDrawdown(), bt.avgTurnover()
```

For a tangency strategy, wrap the weight function with a fallback to
min-variance **only** on the tangency error type (API_SPEC 6.3) — see
the demos (`python/demo.py::tangency_fn`, `cpp/src/demo.cpp`,
`rust/src/bin/demo.rs`, `java/.../Demo.java`) for the pinned pattern.

---

## 11. How do I handle the standard failure modes?

Duplicated asset (singular covariance) and rf at/above the min-variance
return:

**Python**

```python
bad = np.column_stack([x, x[:, 0]])            # duplicate US_EQ
try:
    po.min_variance_weights(po.sample_cov(bad))  # Cholesky pivot catches it
except ValueError as e:
    print(e)   # "matrix is not positive definite (... duplicated asset)"

try:
    po.tangency_weights(mu, sigma, rf=0.50)    # rf above min-var return
except po.TangencyUndefinedError as e:
    print(e)   # "tangency undefined: A - rf*C <= 0 ..."
```

**C++** (the singular pivot throws `std::domain_error`; the tangency
error is the derived `portopt::TangencyUndefinedError`)

```cpp
  Matrix bad_cov(2, 2, 0.04);   // duplicated asset: exactly singular
  try {
    const Vector w = portopt::min_variance_weights(bad_cov);
  } catch (const std::domain_error& e) { /* not positive definite */ }
  try {
    const Vector w = portopt::tangency_weights(mu, sigma, 0.50);
  } catch (const portopt::TangencyUndefinedError& e) { /* tangency undefined */ }
```

**Rust**

```rust
    let bad_cov = Matrix::from_rows(&[vec![0.04, 0.04], vec![0.04, 0.04]])?;
    match portopt::min_variance_weights(&bad_cov) {
        Err(PortOptError::NotPositiveDefinite(msg)) => eprintln!("{msg}"),
        other => { let _ = other?; }
    }
    match portopt::tangency_weights(&mu, &sigma, 0.50) {
        Err(PortOptError::TangencyUndefined(msg)) => eprintln!("{msg}"),
        other => { let _ = other?; }
    }
```

**Java**

```java
        try {
            Markowitz.tangencyWeights(mu, sigma, 0.50);
        } catch (Markowitz.TangencyUndefinedException e) {
            System.out.println(e.getMessage()); // tangency undefined
        }
        try {
            Markowitz.minVarianceWeights(new double[][] {{0.04, 0.04}, {0.04, 0.04}});
        } catch (IllegalArgumentException e) {
            System.out.println(e.getMessage()); // not positive definite
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

**Java** — `bash test.sh` runs all `*Test` classes including the golden
suite (bundled minimal JSON parser).

To regenerate data + golden values after an intentional change:
`python3 data/generate_data.py` (revalidates closed forms, the
Ledoit-Wolf intensity against `covCor.m`, the QP against SLSQP, the ERC
identity, and both BL limits before writing).
