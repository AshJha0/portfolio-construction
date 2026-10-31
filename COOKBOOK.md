# COOKBOOK: portopt recipes

Task-oriented recipes. Recipes 1 to 12 are given in all four languages;
recipes 13 to 23 are Python only.

**Python.** Every Python block is standalone: it starts with its own
imports and data loading and runs as written from the `python/`
directory. Install once from the repository root with
`pip install -e "python[dev]"`, then `cd python` and run a block with
`python recipe.py`; without installing, use `PYTHONPATH=src python
recipe.py` from `python/`. Data paths are relative to `python/`. All
Python blocks except recipe 23 (which needs a network fetch) were
extracted and run this way for this revision; the expected values in
the comments are what they printed.

**C++, Rust, Java.** These snippets are fragments of one program per
language: the preamble in recipe 1 declares `x`, `mu`, `sigma`, `names`,
`w_mkt` and `views`, and the later fragments go where the preamble says
"recipes 2-11 go here". Paths are relative to the language directory
(`cpp/`, `rust/`, `java/`). The fragments use only functions that exist
in `cpp/include/portopt/`, `rust/src/` and `java/src/main/`; they were
not re-compiled in this revision.

Build commands per language:

| Language | Build and test | Demo |
|---|---|---|
| Python (>= 3.11) | `pip install -e "python[dev]"`, then `cd python && python -m pytest -q` | `cd python && python demo.py` |
| C++26 (CMake >= 3.25; GoogleTest is fetched automatically if not installed) | `cmake -S cpp -B cpp/build` then `cmake --build cpp/build --config Release` then `ctest --test-dir cpp/build -C Release --output-on-failure` | `cpp/build/portopt_demo` (multi-config generators: `cpp/build/Release/portopt_demo`) |
| Rust | `cd rust && cargo test --release` | `cd rust && cargo run --release --bin demo` |
| Java 25 (Maven) | `cd java && mvn -B verify` | `cd java && mvn -B -q exec:java` |

Types: C++ `portopt::Matrix` (row-major, `m(i, j)`) and
`portopt::Vector = std::vector<double>`; Rust `portopt::Matrix` (`get`,
`set`, `row`, `matvec`) and `Vec<f64>`; Java `double[][]` and
`double[]`; row-major `T x N` returns everywhere. Errors per language:
Python `ValueError` (tangency: `TangencyUndefinedError`), C++
`std::invalid_argument` or `std::domain_error` (tangency:
`portopt::TangencyUndefinedError`), Rust `Err(PortOptError::...)`, Java
`IllegalArgumentException` (tangency:
`Markowitz.TangencyUndefinedException`).

The bundled data is synthetic (`data/generate_data.py`: three Gaussian
factors plus Student-t noise, seed 42). The numbers below describe the
code, not any market.

---

## 1. How do I load the bundled data and compute annualised moments?

**Python**

```python
import numpy as np
import portopt as po

x = np.loadtxt("../data/asset_returns.csv", delimiter=",", skiprows=1)
mu = po.annualize_mean(po.sample_mean(x))      # 252 * daily mean
sigma = po.annualize_cov(po.sample_cov(x))     # 252 * ddof=1 covariance
print(mu[0])                                   # US_EQ: 0.0939776...
print(np.sqrt(sigma[0, 0]))                    # US_EQ vol: 0.153532...
corr = sigma / np.sqrt(np.outer(np.diag(sigma), np.diag(sigma)))
print(round(corr[0, 1], 4))                    # US_EQ / EU_EQ: 0.8177
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

**Java** (the loaders `DataFiles` and `MiniJson` are package-private
helpers used by the demo; a client parses its own files, here with plain
Java)

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

Shrinkage is computed on the daily returns; annualise the result. The
intensity is the Ledoit and Wolf (2004) estimator; the matrices are
returned in the ddof=1 convention.

**Python**

```python
import numpy as np
import portopt as po

x = np.loadtxt("../data/asset_returns.csv", delimiter=",", skiprows=1)
res = po.shrink_covariance(x)
print(res.intensity)        # 0.0072117652... (golden, 1e-10)
print(res.rbar)             # 0.4171937171...
sigma_shrunk = po.annualize_cov(res.sigma)  # use in place of the sample covariance
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
import numpy as np
import portopt as po

x = np.loadtxt("../data/asset_returns.csv", delimiter=",", skiprows=1)
mu = po.annualize_mean(po.sample_mean(x))
sigma = po.annualize_cov(po.sample_cov(x))

w_mv = po.min_variance_weights(sigma)         # S^-1 1 / C
w_tan = po.tangency_weights(mu, sigma, 0.02)  # TangencyUndefinedError if rf >= A/C or gross leverage > 10 (recipe 20)
print(round(po.gross_leverage(w_tan), 2))     # 4.14: cap this before it reaches an order system
sc = po.two_fund_scalars(mu, sigma)
print(sc.a / sc.c)                            # min-variance return 0.0322349717...
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
constrained (active-set QP) sweep. Each constrained point meets its
target exactly and is certified (a failed certificate raises).

**Python**

```python
import numpy as np
import portopt as po

x = np.loadtxt("../data/asset_returns.csv", delimiter=",", skiprows=1)
mu = po.annualize_mean(po.sample_mean(x))
sigma = po.annualize_cov(po.sample_cov(x))

targets, weights, variances = po.efficient_frontier(mu, sigma, 25)
t_c, w_c, v_c = po.constrained_frontier(mu, sigma, 25)   # long-only
for t, v, tc, vc in zip(targets[::6], variances[::6], t_c[::6], v_c[::6]):
    print(f"{t:.4f}  {v**0.5:.4f}    {tc:.4f}  {vc**0.5:.4f}")
# first row: 0.0322  0.0400    0.0304  0.0404
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
import numpy as np
import portopt as po

x = np.loadtxt("../data/asset_returns.csv", delimiter=",", skiprows=1)
sigma = po.annualize_cov(po.sample_cov(x))

w_lo = po.min_variance_longonly(sigma)                 # lb=0, ub=inf
w_box = po.min_variance_longonly(sigma, lb=0.02, ub=0.40)
res = po.min_variance_longonly_solve(sigma, lb=0.02, ub=0.40)   # diagnostics
print(res.converged, res.iterations, res.kkt_residual)          # True 8 ~2e-16
print(np.round(w_lo, 4))  # golden minvar_longonly_qp: 73.2% US_BOND, zeros on EU_EQ and EM_EQ
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
`constrained_frontier_weights(mu, sigma, 0.06, lb=0.02, ub=0.40)`. It
raises if the target is below the constrained min-variance return or
above the attainable maximum, so check
`attainable_return_range(mu, lb, ub)` first.

---

## 6. How do I repair a weight vector onto the simplex?

The exact sort-based projection is exposed directly, which is useful for
renormalising screened or rounded weights.

**Python**

```python
import portopt as po

w = po.project_simplex_box([0.9, 0.5, -0.2])          # -> [0.7, 0.3, 0.0]
w2 = po.project_simplex_box([0.8, 0.6, 0.4, -0.3], lb=0.05, ub=0.5)
print(w, w2)  # w2 -> [0.5, 0.325, 0.125, 0.05] (golden simplex_projection_box)
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

import numpy as np
import portopt as po

x = np.loadtxt("../data/asset_returns.csv", delimiter=",", skiprows=1)
sigma = po.annualize_cov(po.sample_cov(x))
names = open("../data/asset_returns.csv").readline().strip().split(",")
caps = np.loadtxt("../data/market_caps.csv", delimiter=",", skiprows=1, usecols=1)
with open("../data/views.json") as fh:
    doc = json.load(fh)

w_mkt = caps / caps.sum()
pi = po.implied_equilibrium_returns(sigma, w_mkt, doc["delta"])  # delta * Sigma * w
p, q, conf = po.views_to_matrices(doc["views"], names)
w_bl, res = po.bl_weights(sigma, pi, p, q, doc["tau"], confidences=conf)
print(res.mu_post[4])   # EM_EQ posterior 0.0693978136... (golden, 1e-10)
print(w_bl[0])          # 0.4171772018... (golden bl_posterior_weights)
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

`bl_weights` pins `rf = 0` because pi and the posterior are excess
returns. With views that restate equilibrium, or near-zero confidence,
the weights return to `w_mkt`. Views are validated: an unknown asset, a
missing key, `long == short`, or a NaN `q` is an error, never a silent
row. The weights are the tangency portfolio of the posterior, normalised
to sum to 1 (recipe 16 shows the unnormalised sum).

---

## 8. How do I control the confidence of a Black-Litterman view?

Confidence scales the He-Litterman default
`Omega_kk = [P tau Sigma P']_kk / c_k`: a large `c` makes the view nearly
binding, a small `c` makes it nearly ignored. An explicit `Omega`
replaces the default and cannot be combined with confidences.

**Python**

```python
import json

import numpy as np
import portopt as po

x = np.loadtxt("../data/asset_returns.csv", delimiter=",", skiprows=1)
sigma = po.annualize_cov(po.sample_cov(x))
names = open("../data/asset_returns.csv").readline().strip().split(",")
caps = np.loadtxt("../data/market_caps.csv", delimiter=",", skiprows=1, usecols=1)
with open("../data/views.json") as fh:
    doc = json.load(fh)
pi = po.implied_equilibrium_returns(sigma, caps / caps.sum(), doc["delta"])
p, q, _ = po.views_to_matrices(doc["views"], names)

for c in (0.01, 1.0, 100.0):
    w, res = po.bl_weights(sigma, pi, p, q, 0.05, confidences=[c, c])
    print(f"c={c:>6}: EM_EQ mu_post={res.mu_post[4]:.4f}  w={w[4]:.4f}")
# c=0.01 -> 0.0391 (near equilibrium 0.0385); c=100 -> 0.0994 (near the 10% view)
omega = po.default_omega(sigma, p, 0.05, [100.0, 100.0])
res = po.bl_posterior(sigma, pi, p, q, 0.05, omega=omega)   # explicit Omega
print(round(res.mu_post[4], 4))                             # 0.0994
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

## 9. How do I compute ERC and risk-budget weights and verify contributions?

**Python**

```python
import numpy as np
import portopt as po

x = np.loadtxt("../data/asset_returns.csv", delimiter=",", skiprows=1)
sigma = po.annualize_cov(po.sample_cov(x))

w_erc = po.erc_weights(sigma)
rc = po.risk_contributions(w_erc, sigma)
print(rc.max() - rc.min())                      # ~1.3e-12
w_rb = po.risk_budget_weights(sigma, [0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05])
w_iv = po.inverse_vol_weights(sigma)

# Diagnostics variant: returns (weights, sweeps, converged).
for label, s in (("daily", po.sample_cov(x)), ("annual", sigma), ("annual x 1e4", sigma * 1e4)):
    w, sweeps, ok = po.risk_budget_solve(s)
    rcs = po.risk_contributions(w, s)
    print(f"{label:>13}: {sweeps} sweeps, converged={ok}, RC spread {rcs.max() - rcs.min():.1e}")
# daily 16 sweeps 2.4e-14; annual 14 sweeps 1.3e-12; annual x 1e4 12 sweeps 1.6e-10

w2, sweeps2, ok2 = po.risk_budget_solve(sigma, max_sweeps=2)
rc2 = po.risk_contributions(w2, sigma)
print(sweeps2, ok2, f"{rc2.max() - rc2.min():.1e}")    # 2 False 1.0e-02
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

Budgets are normalised internally. `risk_budget_solve` reports
non-convergence through its `converged` flag and never raises for it;
since 1.2.0 `risk_budget_weights` and `erc_weights` raise `ValueError`
when that flag is false.
The stopping rule is an absolute tolerance on the unnormalised iterate,
which is why the RC spread above depends on the scale of `Sigma`.

---

## 10. How do I run the quarterly backtest and read the turnover?

**Python**

```python
import numpy as np
import portopt as po

x = np.loadtxt("../data/asset_returns.csv", delimiter=",", skiprows=1)
res = po.run_backtest(
    x,
    lambda h: po.erc_weights(po.sample_cov(h)),  # weight_fn sees rows [0, t_k), read-only
    rf=0.02, window=500, step=63,
)
print(res.sharpe, res.max_drawdown)   # 0.63652214..., 0.10188233... (golden, 1e-8)
print(res.rebalance_dates)            # [500, 563, ..., 1193]: 12 dates
print(res.avg_turnover)               # mean L1 distance between consecutive target vectors
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

Within each 63-day block the backtest rebalances to the target weights
every day, charges no costs, and counts turnover only between target
vectors (recipe 18 measures what that leaves out). For a tangency
strategy, wrap the weight function with a fallback to min-variance only
on the tangency error type (API_SPEC 6.3); the demos
(`python/demo.py::tangency_fn`, `cpp/src/demo.cpp`,
`rust/src/bin/demo.rs`, `java/.../Demo.java`) show the pinned pattern.

---

## 11. How do I handle the standard failure modes?

Duplicated asset (singular covariance) and rf at or above the
min-variance return:

**Python**

```python
import numpy as np
import portopt as po

x = np.loadtxt("../data/asset_returns.csv", delimiter=",", skiprows=1)
mu = po.annualize_mean(po.sample_mean(x))
sigma = po.annualize_cov(po.sample_cov(x))

bad = np.column_stack([x, x[:, 0]])              # duplicate US_EQ
try:
    po.min_variance_weights(po.sample_cov(bad))  # the Cholesky pivot catches it
except ValueError as e:
    print(e)   # "matrix is not positive definite (... duplicated asset)"

try:
    po.tangency_weights(mu, sigma, rf=0.50)      # rf above the min-variance return
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

Every port ships a golden test suite; a spot check by hand looks like
this.

**Python**

```python
import json

import numpy as np
import portopt as po

x = np.loadtxt("../data/asset_returns.csv", delimiter=",", skiprows=1)
with open("../data/golden/golden.json") as fh:
    gold = json.load(fh)
case = next(c for c in gold["cases"] if c["name"] == "erc_weights")
w = po.erc_weights(po.annualize_cov(po.sample_cov(x)))
for i in range(8):
    assert abs(w[i] - case["expect"][f"w_{i}"]) <= case["tol"]
print("erc_weights matches golden within", case["tol"])
```

**C++**: `ctest --test-dir cpp/build -C Release --output-on-failure` runs
the golden suite (`cpp/tests/`, flat-schema JSON reader) after the CMake
build in the table at the top.

**Rust**: `cd rust && cargo test --release` includes `tests/golden.rs`
via `serde_json`.

**Java**: `cd java && mvn -B verify` runs all `*Test` classes, including
the golden suite (bundled minimal JSON parser).

To regenerate data and golden values after an intentional change, run
`python data/generate_data.py`. It revalidates the closed forms against
`numpy.linalg.lstsq`, the Ledoit-Wolf intensity against `covCor.m`, the
QP against scipy SLSQP, the ERC identity and both BL limits before
writing.

---

## 13. How sensitive are tangency weights to a 1% change in one mean?

Python only. Shift one expected return by one percentage point and
measure the L1 change in weights, for the unconstrained tangency
portfolio and for the long-only frontier point at 6%.

```python
import numpy as np
import portopt as po

x = np.loadtxt("../data/asset_returns.csv", delimiter=",", skiprows=1)
mu = po.annualize_mean(po.sample_mean(x))
sigma = po.annualize_cov(po.sample_cov(x))

w_tan = po.tangency_weights(mu, sigma, 0.02)
w_lo = po.constrained_frontier_weights(mu, sigma, 0.06)
for i, name in ((0, "US_EQ"), (1, "EU_EQ")):
    shifted = mu.copy()
    shifted[i] += 0.01
    d_tan = np.abs(po.tangency_weights(shifted, sigma, 0.02) - w_tan).sum()
    d_lo = np.abs(po.constrained_frontier_weights(shifted, sigma, 0.06) - w_lo).sum()
    print(f"{name} +1%: tangency L1 change {d_tan:.3f}, long-only 6% point {d_lo:.3f}")
# US_EQ +1%: tangency L1 change 0.580, long-only 6% point 0.129
# EU_EQ +1%: tangency L1 change 0.608, ...
```

A one-point change is well inside the 7% to 9% standard error of a
five-year sample mean (LEARN section 2), so weights that move by 58% of
NAV for it are tracking noise.

---

## 14. How does the shrinkage intensity depend on the sample length?

Python only. Run Ledoit-Wolf on the first T rows of the same panel.

```python
import numpy as np
import portopt as po

x = np.loadtxt("../data/asset_returns.csv", delimiter=",", skiprows=1)
for t in (5, 10, 20, 60, 120, 250, 500, 1250):
    res = po.shrink_covariance(x[:t])
    po.cholesky(res.sigma)              # passes even at T = 5 < N = 8
    print(f"T={t:>5}: delta={res.intensity:.4f}  rbar={res.rbar:.4f}")
# T=5: 0.3469, T=10: 0.4393, T=20: 0.2574, T=60: 0.1623, T=120: 0.0739,
# T=250: 0.0370, T=500: 0.0176, T=1250: 0.0072
```

At T = 5 the sample covariance is singular (`po.cholesky(po.sample_cov(x[:5]))`
raises), but the shrunk matrix factors. The intensity falls roughly like
`1/T` as the sample covariance becomes more precise.

---

## 15. How do I set risk budgets of 60% equity and 40% bonds?

Python only. Compare the risk split of a 60/40 capital allocation with
the capital split of a 60/40 risk allocation. Assets 0 to 4 are the
equity lines, 5 to 7 the bond and carry lines.

```python
import numpy as np
import portopt as po

x = np.loadtxt("../data/asset_returns.csv", delimiter=",", skiprows=1)
sigma = po.annualize_cov(po.sample_cov(x))

w_cap = np.array([0.6 / 5] * 5 + [0.4 / 3] * 3)          # 60/40 in capital
print(f"60/40 capital: equity risk share {po.risk_contributions(w_cap, sigma)[:5].sum():.3f}")
# 0.897

budgets = [0.6 / 5] * 5 + [0.4 / 3] * 3                   # 60/40 in risk
w_rb, sweeps, ok = po.risk_budget_solve(sigma, budgets)
rc = po.risk_contributions(w_rb, sigma)
print(ok, f"equity capital {w_rb[:5].sum():.3f}, equity risk {rc[:5].sum():.3f}")
# True equity capital 0.270, equity risk 0.600
```

---

## 16. How do the Black-Litterman weights respond to view confidence?

Python only. Sweep both confidences together, check that the posterior
mean does not depend on tau under the default Omega, and compute the sum
of the unnormalised mean-variance weights.

```python
import json

import numpy as np
import portopt as po

x = np.loadtxt("../data/asset_returns.csv", delimiter=",", skiprows=1)
sigma = po.annualize_cov(po.sample_cov(x))
names = open("../data/asset_returns.csv").readline().strip().split(",")
caps = np.loadtxt("../data/market_caps.csv", delimiter=",", skiprows=1, usecols=1)
with open("../data/views.json") as fh:
    doc = json.load(fh)
pi = po.implied_equilibrium_returns(sigma, caps / caps.sum(), doc["delta"])
p, q, conf = po.views_to_matrices(doc["views"], names)

for c in (0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0):
    w, res = po.bl_weights(sigma, pi, p, q, doc["tau"], confidences=[c, c])
    print(f"c={c:>6}: EM_EQ mu={res.mu_post[4]:.4f} w={w[4]:.4f} gross={po.gross_leverage(w):.3f}")
# c=0.01: mu 0.0391 w 0.0635 gross 1.000 ... c=100: mu 0.0994 w 0.4218 gross 1.909

for tau in (0.01, 0.05, 0.5):                              # default Omega: tau cancels
    print(tau, po.bl_posterior(sigma, pi, p, q, tau, confidences=conf).mu_post[4])
# 0.0693978136842196 for all three, to the last digit or two

res = po.bl_posterior(sigma, pi, p, q, doc["tau"], confidences=conf)
w_raw = po.spd_solve(res.sigma_post, res.mu_post) / doc["delta"]
print(f"unnormalised weights sum to {w_raw.sum():.4f}")   # 1.2504; bl_weights rescales to 1
```

---

## 17. How do I check a QP certificate by hand, including dual feasibility?

Python only. For `min (1/2) w' Sigma w` with `1'w = 1` and box bounds,
stationarity reads `Sigma w = gamma 1 + u_lo - u_hi`. On the free assets
`(Sigma w)_i = gamma`; on an asset at its lower bound
`u_lo = (Sigma w)_i - gamma` must be `>= 0`; on an asset at its upper
bound `u_hi = gamma - (Sigma w)_i` must be `>= 0`.

```python
import numpy as np
import portopt as po

x = np.loadtxt("../data/asset_returns.csv", delimiter=",", skiprows=1)
mu = po.annualize_mean(po.sample_mean(x))
sigma = po.annualize_cov(po.sample_cov(x))


def check(res, lb, ub, tol=1e-12):
    w = res.weights
    g = sigma @ w
    at_lo = np.isclose(w, lb, rtol=0.0, atol=tol)
    at_hi = np.isclose(w, ub, rtol=0.0, atol=tol)
    free = ~(at_lo | at_hi)
    gamma = g[free].mean()
    u_lo = g[at_lo] - gamma
    u_hi = gamma - g[at_hi]
    print(f"iterations {res.iterations}, converged {res.converged}, kkt_residual {res.kkt_residual:.1e}")
    print(f"gamma {gamma:.10f}, variance {w @ sigma @ w:.10f}")
    print("stationarity on free assets:", np.max(np.abs(g[free] - gamma)))
    print("lower-bound multipliers:", u_lo, "upper-bound multipliers:", u_hi)
    assert np.all(u_lo >= -1e-12) and np.all(u_hi >= -1e-12), "dual feasibility fails"


check(po.min_variance_longonly_solve(sigma), 0.0, np.inf)
# iterations 4; gamma 0.0016283139 = variance; u_lo for EU_EQ 1.03e-05, EM_EQ 6.77e-04
check(po.min_variance_longonly_solve(sigma, lb=0.02, ub=0.40), 0.02, 0.40)
# iterations 8; US_BOND at 0.40 with u_hi = 0.00203075 - 0.00137135 = 6.59e-04
print(po.constrained_frontier_solve(mu, sigma, 0.08).iterations)   # 9
```

Since 1.2.0 `converged` itself includes this sign check (LEARN 4.2); the
recipe shows what it verifies. In the long-only case `gamma` equals the
portfolio variance: multiply
stationarity by `w'` and use `1'w = 1` and `u'w = 0` (LEARN 4.2). The
same check applies to `constrained_frontier_solve`, with a second
equality multiplier on `mu`; solve for `(gamma, lambda)` by least squares
on the free assets.

---

## 18. How do I put an error bar and a cost charge on a backtest?

Python only. Three numbers the backtest does not report: the Lo (2002)
standard error of the annualised Sharpe ratio (iid case), the annual
cost of the target-to-target turnover at 5 bp per unit, and the turnover
of the daily trades that keep the constant mix, which `run_backtest`
assumes but does not count.

```python
import math

import numpy as np
import portopt as po

x = np.loadtxt("../data/asset_returns.csv", delimiter=",", skiprows=1)
rf, cost = 0.02, 0.0005                                # 5 bp per unit of turnover
log = []


def tangency_fn(h):
    s = po.annualize_cov(po.sample_cov(h))
    m = po.annualize_mean(po.sample_mean(h))
    sc = po.two_fund_scalars(m, s)
    try:
        w = po.tangency_weights(m, s, rf, max_leverage=math.inf)  # the 1.2.0 behaviour
        log.append((h.shape[0], "tangency", sc.a / sc.c, po.gross_leverage(w)))
        return w
    except po.TangencyUndefinedError:
        w = po.min_variance_weights(s)
        log.append((h.shape[0], "fallback", sc.a / sc.c, po.gross_leverage(w)))
        return w


methods = {
    "equal_weight": lambda h: np.full(h.shape[1], 1.0 / h.shape[1]),
    "min_var_lo": lambda h: po.min_variance_longonly(po.annualize_cov(po.sample_cov(h))),
    "tangency": tangency_fn,
    "erc": lambda h: po.erc_weights(po.sample_cov(h)),
}
for name, fn in methods.items():
    res = po.run_backtest(x, fn, rf=rf, window=500, step=63)
    n_days = res.portfolio_returns.size
    years = n_days / 252
    sr_p = res.sharpe / math.sqrt(252)
    se = math.sqrt(252) * math.sqrt((1 + 0.5 * sr_p**2) / n_days)
    drag = cost * res.turnovers.sum() / years
    drift = 0.0                                          # daily constant-mix trades
    dates = res.rebalance_dates + [x.shape[0]]
    for k, w in enumerate(res.weights):
        for row in range(dates[k], dates[k + 1] - 1):    # the last day trades to the next target
            r = x[row]
            drift += np.abs(w - w * (1 + r) / (1 + w @ r)).sum()
    print(f"{name:<13} SR {res.sharpe:.3f} +/- {se:.3f}  target TO {res.turnovers.sum():7.2f}"
          f"  drag {drag:.4%}/yr  drift TO {drift / years:7.2f}/yr  drag {cost * drift / years:.4%}/yr")

for t_k, kind, gmv_ret, lev in log:
    print(t_k, kind, f"A/C={gmv_ret:.5f}", f"gross={lev:.1f}")
# equal_weight  SR 0.445 +/- 0.580  target TO    0.00  drag 0.0000%/yr  drift TO    1.09/yr  drag 0.0544%/yr
# min_var_lo    SR 0.695 +/- 0.580  target TO    0.30  drag 0.0051%/yr  drift TO    0.53/yr  drag 0.0263%/yr
# tangency      SR 0.806 +/- 0.580  target TO  200.17  drag 3.3628%/yr  drift TO  222.43/yr  drag 11.1213%/yr
# erc           SR 0.637 +/- 0.580  target TO    0.06  drag 0.0009%/yr  drift TO    0.90/yr  drag 0.0448%/yr
# 500 .. 941: fallback (A/C below 0.02); 1004 tangency A/C=0.02073 gross=93.6; 1067 gross=28.8; ...
```

The tangency rule here passes `max_leverage=math.inf`, which reproduces
the 1.2.0 behaviour; with the 1.3.0 default cap of 10 it falls back to
min variance at all 12 dates, as in the demo. The uncapped run is kept
because it shows what the cap prevents: 93.6x gross leverage at the 1004
rebalance and 200 units of target turnover in 11 rebalances.

The standard error is about `sqrt(252 / 750) = 0.58` for every method,
so no Sharpe difference in the demo table is distinguishable from zero.
The linear cost model is an assumption; LEARN section 7.3 discusses what
it leaves out. For a backtest that charges costs on drifted holdings,
with no-trade bands, volatility targeting, PSR, DSR and a block
bootstrap, use the Python-only research layer `portopt.research`
(`run_backtest_net`, `research/stats.py`); its study is in
`docs/results/cost_aware_allocation.md`.

---

## 19. How do I rebalance with a turnover penalty or a turnover cap?

Python only (the solver exists in all four ports, API_SPEC 3.7). The
plain idea: when moving from the current holdings `w0` to a new
minimum-variance portfolio costs money, add the cost to the objective,
`1/2 w'Sw + kappa * sum|w - w0|`, or bound the trade outright. The
solver splits the trade into buys and sells, `w = w0 + b - s`, solves
the resulting smooth QP exactly, and certifies the KKT conditions of the
original nonsmooth problem [derived: API_SPEC 3.7]. `kappa` is in
variance units; a cost rate `c` against a risk aversion `gamma` maps to
`kappa = c / gamma`.

```python
import numpy as np
import portopt as po

x = np.loadtxt("../data/asset_returns.csv", delimiter=",", skiprows=1)
sigma = po.annualize_cov(po.sample_cov(x))
w0 = np.loadtxt("../data/market_caps.csv", delimiter=",", skiprows=1, usecols=1)
w0 = w0 / w0.sum()                                  # current holdings: market weights

for kappa in (0.0, 1e-4, 1e-3):
    res = po.min_variance_turnover_solve(sigma, w0, kappa)
    vol = float(np.sqrt(res.weights @ sigma @ res.weights))
    print(f"kappa {kappa:g}: turnover {res.turnover:.4f} vol {vol:.4f} "
          f"certified {res.converged} kkt {res.kkt_residual:.1e}")

capped = po.min_variance_turnover_solve(sigma, w0, 0.0, max_turnover=0.25)
print(f"cap 0.25: turnover {capped.turnover:.4f}")
# kappa 0: turnover 1.2430 vol 0.0404 certified True kkt 2.2e-19
# kappa 0.0001: turnover 1.0379 vol 0.0405 certified True kkt 6.9e-19
# kappa 0.001: turnover 0.8253 vol 0.0432 certified True kkt 4.3e-19
# cap 0.25: turnover 0.2500
```

The same figures appear in the demo's cost-aware section [measured:
`python python/demo.py`]. `min_variance_turnover` and
`mean_variance_turnover` return the weights only and raise when the
certificate fails; the `_solve` forms return `TurnoverQpResult` with
`turnover`, `converged` and `kkt_residual`. A cap below the smallest
attainable turnover raises.

---

## 20. How does the tangency leverage cap work?

Python only. The tangency weights are `S^-1 (mu - rf 1) / (A - rf C)`,
so they grow without bound as rf approaches the min-variance return
`A/C` from below. Since 1.3.0 `tangency_weights` raises
`TangencyUndefinedError` when the gross leverage of the result exceeds
`max_leverage`, default `TANGENCY_MAX_LEVERAGE = 10`; `math.inf` turns
the cap off [derived: API_SPEC 3.3]. The history below is the one the
backtest sees at its 1004 rebalance.

```python
import math

import numpy as np
import portopt as po

x = np.loadtxt("../data/asset_returns.csv", delimiter=",", skiprows=1)
h = x[:1004]                                        # the history at the 1004 rebalance
mu = po.annualize_mean(po.sample_mean(h))
sigma = po.annualize_cov(po.sample_cov(h))
sc = po.two_fund_scalars(mu, sigma)
print(f"A/C = {sc.a / sc.c:.5f}, rf = 0.02")

w_raw = po.tangency_weights(mu, sigma, 0.02, max_leverage=math.inf)
print(f"uncapped gross leverage {po.gross_leverage(w_raw):.1f}")
try:
    po.tangency_weights(mu, sigma, 0.02)            # default cap TANGENCY_MAX_LEVERAGE = 10
except po.TangencyUndefinedError as e:
    print("default cap:", str(e)[:60])
print(po.TANGENCY_MAX_LEVERAGE)
# A/C = 0.02073, rf = 0.02
# uncapped gross leverage 93.6
# default cap: tangency ill-conditioned: gross leverage 93.56 exceeds max_l
# 10.0
```

A gap of 0.07 percentage points between rf and `A/C` produced 93.6x
gross leverage [measured: this recipe]. Version 1.2.0 accepted it,
because its guard only rejected gaps below one part per million.
Callers that cap leverage themselves (for example
`portopt.research.tangency_capped`, which blends toward minimum
variance) pass `max_leverage=math.inf` and apply their own rule.

---

## 21. How do EWMA and sample covariances compare for long-only min variance?

Python only, research layer. Both rules are long-only minimum variance
with the expanding window and 63-day step of the pinned schedule, net of
10 bp per unit of traded notional, with weights drifting between
rebalances (`run_backtest_net`). The EWMA estimator uses a half-life of
22.8 days (decay 0.97).

```python
import numpy as np
import portopt as po
from portopt.research import (
    ewma_estimator,
    min_variance_longonly_with,
    run_backtest_net,
    sample_estimator,
)

x = np.loadtxt("../data/asset_returns.csv", delimiter=",", skiprows=1)
for est in (sample_estimator(), ewma_estimator(half_life=22.8)):
    res = run_backtest_net(x, min_variance_longonly_with(est), cost_bps=10, rf=0.02)
    print(f"{est.name:<12} SR {res.sharpe:.3f} vol {res.ann_vol:.4f} "
          f"turnover/yr {res.annual_turnover:.2f} cost/yr {res.ann_cost:.4%}")
# sample       SR 0.682 vol 0.0402 turnover/yr 0.15 cost/yr 0.0484%
# ewma22.8     SR 0.582 vol 0.0408 turnover/yr 1.11 cost/yr 0.1448%
```

EWMA trades about seven times as much and does not lower volatility
[measured: this recipe, synthetic data]. The synthetic returns have
constant volatility, so a fast estimator has nothing to track. On the
real industry data EWMA also lost, by 0.06 to 0.13 of Sharpe against
the sample covariance [measured: `docs/results/real_data_study_dev.md`,
covariance estimators table]. With a Sharpe standard error of about 0.58
on 750 days (recipe 18), the synthetic difference is noise.

---

## 22. How do I target volatility with a chosen covariance estimator?

Python only, research layer. `run_backtest_net(vol_target=...)` scales
the target at each rebalance by `s = target / sqrt(u' Sigma u)`, caps
`s * sum|u|` at `max_gross`, and holds the rest in cash. Sigma is the
annualized sample covariance of the history unless `vol_estimator`
supplies another one.

```python
import numpy as np
import portopt as po
from portopt.research import ewma_estimator, run_backtest_net

x = np.loadtxt("../data/asset_returns.csv", delimiter=",", skiprows=1)
ew = lambda h: np.full(h.shape[1], 1.0 / h.shape[1])

plain = run_backtest_net(x, ew, cost_bps=10, rf=0.02)
print(f"1/N            SR {plain.sharpe:.3f} vol {plain.ann_vol:.4f}")
for label, est in (("sample", None), ("EWMA hl 22.8", ewma_estimator(half_life=22.8))):
    res = run_backtest_net(x, ew, cost_bps=10, rf=0.02, vol_target=0.10,
                           max_gross=1.5, vol_estimator=est)
    print(f"1/N @10% {label:<13} SR {res.sharpe:.3f} vol {res.ann_vol:.4f} "
          f"scales {np.round(res.scales[:3], 3)}")
# 1/N            SR 0.450 vol 0.0931
# 1/N @10% sample        SR 0.492 vol 0.0923 scales [0.951 0.958 0.968]
# 1/N @10% EWMA hl 22.8  SR 0.559 vol 0.1005 scales [0.871 0.987 1.063]
```

On synthetic data with constant volatility none of these differences is
distinguishable from zero [measured: this recipe; derived: standard
error 0.58]. On the real 10-industry data, 1/N scaled to 10% with the
sample covariance gained 0.13 of Sharpe over plain 1/N (p = 0.02), almost
all of it in 1928-1969 [measured: `docs/results/real_data_study_dev.md`].

---

## 23. How do I load the real Kenneth French data?

Python only, research layer. **Needs a network fetch and is not run in
CI.** The data are research index portfolios from the Kenneth R. French
Data Library; they are downloaded at run time, checked against the
SHA-256 vintages in `data/real/manifest.json`, and never committed
(`data/real/README.md`). The series are not investable.

First fetch from the repository root:

```bash
python scripts/fetch_data.py ind10_vw_daily ff3_daily
```

Then, from `python/`:

```python
import numpy as np
from portopt.research import data, min_variance_longonly_with, run_backtest_net
from portopt.research import lw_constant_corr_estimator

panel = data.load_returns("ind10_vw_daily")        # period="development": before 2016-01-01
print(panel.columns, panel.dates[0], panel.dates[-1], panel.values.shape)
print(panel.vintage_id, panel.zip_sha256[:16])

rule = min_variance_longonly_with(lw_constant_corr_estimator())
res = run_backtest_net(panel.values, rule, window=504, step=21, cost_bps=10)
print(res.sharpe, res.annual_turnover)

try:
    data.load_returns("ind10_vw_daily", period="holdout")  # no registered study
except data.HoldoutAccessError as e:
    print(e)
```

`period="holdout"` and `period="all"` raise `HoldoutAccessError` unless
`study_id` names a study in `data/real/registry.json`
[derived: `research/data.py::load_returns`]. This block was not run for
this revision, so it carries no expected output; the study script
`python/studies/real_data_study.py` uses a calendar schedule rather than
the 21-row step shown here, so its numbers differ.
