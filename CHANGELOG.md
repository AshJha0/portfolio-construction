# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
uses [semantic versioning](https://semver.org/). Dates are UTC.

## [Unreleased]

Nothing yet.

## [1.3.0] - 2026-10-31

Round-2 release: a tangency leverage cap, a cost-aware turnover QP in all
four languages, an extended research layer, real data with a
pre-registered holdout, a forward paper log and kernel benchmarks.

**Breaking change in default behaviour.** `tangency_weights` now raises
`TangencyUndefinedError` when the gross leverage of the result exceeds
`max_leverage`, default `TANGENCY_MAX_LEVERAGE = 10` (C++
`kTangencyMaxLeverage`). Code that relied on 1.2.0 accepting highly
levered tangency portfolios must pass `max_leverage=math.inf` (Rust:
`tangency_weights_capped(..., f64::INFINITY)`). On the bundled demo the
tangency strategy now falls back to min variance at all 12 rebalances,
so its row equals unconstrained min variance.

### Added

- Cost-aware QP (API_SPEC 3.7) in Python, C++, Rust and Java:
  `min_variance_turnover[_solve]` and `mean_variance_turnover[_solve]`,
  objective plus `kappa * sum|w - w0|` with an optional turnover cap,
  solved exactly through the split `w = w0 + b - s` on the dual
  active-set QP, with a KKT certificate of the nonsmooth problem
  (`TurnoverQpResult`). Three golden cases (`minvar_turnover_l1`,
  `minvar_turnover_capped`, `meanvar_turnover_box`), 26 in all.
- `max_leverage` argument to `tangency_weights` (Python, C++, Java) and
  `tangency_weights_capped` (Rust).
- Demo: a cost-aware rebalancing section and a `min_var_tc` backtest row
  (turnover penalty `kappa = 3e-5`); the tangency fallback note now
  counts both reasons.
- Research layer (`portopt.research`, Python only): `estimators` (EWMA,
  EWMA constant correlation, Ledoit-Wolf toward the identity, Ledoit-Wolf
  2020 nonlinear shrinkage, PCA factor model, Bayes-Stein and James-Stein
  means); `strategies_v2` (covariance-estimator objects, weight rules on
  them, `vol_target`); `data` (Kenneth French loader with SHA-256
  vintages and a holdout gate); `walkforward` (trial registry,
  calendar schedule, `walk_forward`); `run_backtest_net(vol_estimator=...)`.
- Real data: `scripts/fetch_data.py`, `data/real/README.md`,
  `manifest.json` and `registry.json`. Third-party files are fetched at
  run time and never committed (`data/real/raw/` is git-ignored).
- Studies: development results (`docs/results/real_data_study_dev.md`,
  206 registered configurations), the pre-registration
  (`docs/results/real_data_preregistration.md`) and the holdout result
  (`docs/results/real_data_study.md`). The holdout is a negative result:
  none of the six hypotheses is supported (every Holm-adjusted p-value
  1.00, all six differences with the wrong sign), with power of 0.04 to
  0.28.
- Forward paper log: `forward/run_forward.py`, two frozen rule files in
  `forward/rules/`, `forward/README.md` and the monthly workflow
  `.github/workflows/forward.yml`. Hypothetical, non-investable indices,
  no real money.
- Kernel benchmarks in all four languages (`python/benchmarks/`,
  `cpp/benchmarks/`, `rust/benches/`, Java `bench` profile), a CI smoke
  job, and `docs/results/benchmarks.md`.
- Docs: LEARN section 9 (where portfolio-construction P&L comes from),
  COOKBOOK recipes 19 to 23, ARCHITECTURE diagrams
  `real_data_holdout.mmd` and `forward_ledger.mmd`.

### Changed

- Python `shrink_covariance` evaluates its sums on numpy tiles by
  default (2.4x to 22x faster; agrees with the loop reference, kept as
  `_shrink_covariance_reference`, to 1e-13 relative). The golden case is
  unchanged.
- C++ `run_backtest` passes the weight function a non-owning
  `HistoryView` instead of copying the history at each rebalance; the
  `Matrix` overload remains.
- `portopt.research.tangency_capped` passes `max_leverage=math.inf`,
  since it applies its own cap.
- COOKBOOK 18 passes `max_leverage=math.inf` so that it still shows what
  the uncapped rule did.

### Fixed

- The 1.2.0 tangency guard (reject only when `A - rf*C` is within a
  relative 1e-6 of cancelling) accepted 93.6x, 28.8x, 42.0x and 15.1x
  gross leverage on the bundled backtest, which produced the demo's
  18.2 average turnover. The leverage cap replaces it.
- Stale documentation: test and golden counts, the tangency guard and
  the turnover narrative in README, LEARN, COOKBOOK, the Pages site,
  ARCHITECTURE and the Rust `PortOptError::TangencyUndefined` doc
  comment.

## [1.2.0] - 2026-10-31

Build, CI, correctness and documentation release. Every golden case in
`data/golden/golden.json` is unchanged, and all four demos print the same
numbers as 1.1.0; only two demo headings were reworded.

### Added

- Continuous integration in `.github/workflows/ci.yml`: Python 3.11 and
  3.13 on Ubuntu, Windows and macOS; C++ with GCC 14, Clang and MSVC; Rust
  stable plus a minimum-supported-version job; Java 25 and 27; and a job
  that diffs the four demo outputs against each other. A Clang
  AddressSanitizer and UBSan job runs the C++ tests.
- CodeQL scanning for all four languages (`.github/workflows/codeql.yml`)
  and Dependabot updates (`.github/dependabot.yml`).
- Python packaging: `python/pyproject.toml` (package `portopt`, extras
  `dev` and `docs`), `py.typed`, ruff and mypy configuration.
- Java Maven build: `java/pom.xml` (`mvn -B verify`, demo through
  `mvn -B -q exec:java`), compiled with `--release 25`.
- C++: GoogleTest is fetched with CMake `FetchContent` when it is not
  installed; MSVC is supported through `/std:c++latest`.
- Figures: `docs/make_figures.py` draws `docs/img/frontier.png`,
  `docs/img/bl_weights.png` and `docs/img/risk_contributions.png` from the
  bundled data.
- `CONTRIBUTING.md`, `SECURITY.md`, issue templates and a pull request
  template.
- `portopt.research`, a Python-only research layer outside the pinned
  cross-language contract: `run_backtest_net` (weights drift between
  rebalances, costs in basis points on traded notional, no-trade band,
  partial rebalancing, volatility target, gross cap), Sharpe standard
  errors (Lo 2002, Mertens 2002), PSR and DSR (Bailey and Lopez de Prado),
  stationary and circular block bootstrap, and the Ledoit-Wolf (2008)
  paired Sharpe test; 49 tests.
- A cost-aware allocation study, `python/studies/cost_aware_allocation.py`,
  with results in `docs/results/cost_aware_allocation.md` and `.csv`
  (144 configurations on the synthetic data; byte-identical on rerun).
- `scripts/demo_parity.py`, used by CI to require the four demo outputs
  to match the Python reference line for line.
- Five new diagrams in `docs/diagrams/` (QP active set, Black-Litterman,
  risk-parity CCD, CI pipeline, golden parity), inlined in ARCHITECTURE.md.

### Fixed

- Risk-parity weight wrappers (`risk_budget_weights`, `erc_weights` and
  their ports) now raise when the coordinate-descent sweep cap is hit,
  matching the QP wrappers. Before, they returned unconverged weights
  silently. The `*_solve` functions still return the flag. Java and Rust
  gain overloads that take the tolerance and sweep cap.
- The QP KKT certificate now checks dual feasibility: no active bound
  multiplier may be below `-1e-9 (1 + scale)`. Before, a wrong active set
  could in principle be certified. `kkt_residual` is now the largest of
  the primal, stationarity and dual residuals. All four languages.
- Rust golden helper `check_vec` now checks the result length.
- C++ builds on MSVC and without a system GoogleTest; the Java build no
  longer depends on Debian jar paths.

### Changed

- Language baselines: C++26 (was C++17) with CMake 3.25 or later, Java 25
  (was 21), Rust with a declared `rust-version`.
- Version 1.2.0 everywhere. Before this release the Python package
  reported 2.0.0 and the Rust crate 1.0.0.
- README rewritten: plain-language opening, a statement that the bundled
  data is synthetic and how it is generated, quickstart per language,
  figures, a corrected description of the backtest (a daily-rebalanced
  constant mix re-estimated every 63 days, turnover target to target, no
  costs), and a "what this is not" section.
- `docs/index.html`: version, CI badge, synthetic-data statement, figures,
  new quickstart; the review notes moved here (see 1.1.0).
- Python sources and the data generator reformatted with ruff; the
  generator docstring now says 23 golden cases, matching the file.
- LEARN.md rewritten to the plain idea, formal statement, worked example
  pattern, with evidence labels; new sections on the KKT certificate,
  backtest limits and costs, and moving to a real desk; 8 more
  knowledge-check questions. The Q&A section was retitled (commit 77448c0).
- COOKBOOK.md: every Python recipe is standalone and runs (18 recipes,
  6 new). ARCHITECTURE.md brought up to date with testing, convergence,
  performance and thread-safety sections.
- Demo headings: the backtest section now reads "Backtest: re-estimated
  every 63 days, daily constant mix", and the title no longer uses a
  dash character that Java printed differently.

## [1.1.0] - 2026-09-06

Correctness review (commit d98c3dd, "Review 2026-09-06: correctness fixes,
edge-case tests, references, docs site").

The `v1.1.0` tag points at commit 935a9ce, not at d98c3dd. The two commits
have the same message, author date and tree (`git diff d98c3dd 935a9ce` is
empty); 935a9ce is a rewritten copy that is not an ancestor of `main`. The
content released as 1.1.0 is therefore exactly d98c3dd.

### Fixed

- Ledoit-Wolf intensity. The constant-correlation correction put a weight
  of `rbar/2` on each ordered pair, half of what the paper specifies, and
  over-shrank by 31% on the bundled data (0.00943 instead of 0.00721). The
  sums now follow Ledoit and Wolf (2004) Appendix B and `covCor.m` term by
  term, and the generator checks them against a transcription of
  `covCor.m`.
- Constrained frontier. Projected gradient descent on a quadratic penalty
  (Hessian condition number about 7e5) could stop at its 10,000-iteration
  cap 1.7e-3 away from the minimiser without saying so. It is replaced by a
  Goldfarb-Idnani dual active-set QP that meets the target-return equality
  exactly and returns a KKT certificate (`QpResult.converged`,
  `kkt_residual`). Weights-only wrappers raise instead of returning an
  uncertified point.

### Added

- Edge-case tests in all four languages: singular covariance, n = 1,
  tangency undefined or ill-conditioned, degenerate frontier, unattainable
  targets, infeasible bounds, malformed views, returns at or below -1.
- QP validation in the generator against scipy SLSQP; `golden.json`
  regenerated with 23 cases and all three ports re-pinned.
- References section in the README; the GitHub Pages landing page
  `docs/index.html` and `docs/GITHUB_PAGES.md`.

### Changed

- API_SPEC.md, LEARN.md, COOKBOOK.md and docs/ARCHITECTURE.md updated to
  the new solver and estimator.

## [1.0.0] - 2026-08-29

Initial port (commit ed814fc, "Add portfolio-construction (Python, C++,
Rust, Java)").

### Added

- Python reference package `portopt`: sample and Ledoit-Wolf estimators,
  closed-form Markowitz, a box-constrained frontier solved by projected
  gradient descent (`pgd_minimize`, replaced in 1.1.0), Black-Litterman,
  risk parity by cyclical coordinate descent, and a quarterly
  expanding-window backtest.
- Ports in C++17, Rust and Java 21, built with shell scripts.
- The pinned contract `API_SPEC.md`, the guide `LEARN.md`, `COOKBOOK.md`,
  `docs/ARCHITECTURE.md` and Mermaid diagrams.
- The synthetic data generator `data/generate_data.py` and the shared
  golden file `data/golden/golden.json` (21 cases).

[Unreleased]: https://github.com/AshJha0/portfolio-construction/compare/v1.3.0...HEAD
[1.3.0]: https://github.com/AshJha0/portfolio-construction/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/AshJha0/portfolio-construction/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/AshJha0/portfolio-construction/compare/ed814fc...v1.1.0
[1.0.0]: https://github.com/AshJha0/portfolio-construction/commit/ed814fc
