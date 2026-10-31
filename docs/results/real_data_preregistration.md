# Pre-registration: holdout test and forward rules (round 2)

Written and registered on 2026-10-31 before any row dated 2016-01-01 or later was read by this study. The registry entries in `data/real/registry.json` carry the SHA-256 of the registered configuration lists; the holdout study's description carries the SHA-256 of the hypothesis list below. Phase 2 (`python python/studies/real_data_study.py --phase holdout`) refuses to run if either differs from the code, records `git rev-parse HEAD`, and refuses a second run.

**Code commit of record:** the commit that adds this file (`git log --diff-filter=A -- docs/results/real_data_preregistration.md`). It contains `python/studies/`, the registry entries and the forward rule files.

## 1. Question

Which portfolio-construction levers improve net-of-cost risk-adjusted performance on real data, and by how much? The development results (`docs/results/real_data_study_dev.md`) give effect sizes on 1926-2015. The holdout tests a small number of them, chosen from development evidence, on 2016-01 to the last complete month of the pinned vintage.

## 2. Data

| dataset | vintage | CRSP | zip SHA-256 | parsed SHA-256 |
|---|---:|---|---|---|
| `ff3_daily` | 1 | 202609 | `96c23b88e651cbfff3740a53f8225e59ee39893371e96385a1d8125ec2cd9c89` | `236360039226914bba94740c484c39f21ff8f7bf2a756a5320db60c1d333e395` |
| `ind10_vw_daily` | 1 | 202609 | `4c79bad80b6f4eee9d08e649ce223d1c51aadd65043b65baca50b54981d38d25` | `d3ad5d200f89b64a3b7f459be61b4feb338de8be161a2f195ce154149dda6e62` |
| `ind49_vw_daily` | 1 | 202609 | `b9d36d571e968d71f7b03207e7fed6398554f953719b8c1bf3ee3ad8d00e7520` | `0a52ff3f96abd8c0f861701877f9c4ef145af91a0f5d7046172974bef0e30016` |

Phase 2 loads vintage 1 (pinned in `real_data_core.DATA_VINTAGE`) with `period="all"` and the study id `r2-holdout-v1`, and evaluates rows from 2016-01-01 to the last complete month of that vintage (2026-09-30, 2,701 trading days). Each configuration starts from cash on the first trading day of 2016-01; its estimation windows reach back into the development data.

## 3. Method (frozen)

Exactly the method of the development page: monthly rebalancing on the first trading day of each month, `run_schedule_net` accounting (the scaled variant in `real_data_engine.py` for volatility targeting), cost on traded risky notional, cash and borrowing at the daily `RF`, the same estimators, windows and parameters. Costs are assumptions (10 bp base, 25 bp stress) for non-investable index series.

## 4. Hypotheses

Each hypothesis is one-sided: `H1: SR(test) > SR(comparator)`, metric the difference of annualized net Sharpe ratios at 10 bp on the holdout, test the Ledoit and Wolf (2008) HAC delta-method test (`portopt.research.sharpe_difference_test`, Bartlett kernel, `floor(4 (T/100)^(2/9))` lags), one-sided p-value `1 - Phi(z)`. The 6 p-values are adjusted by Holm's (1979) step-down method; a hypothesis is supported when its adjusted p-value is at most 0.05 (family-wise error rate 0.05).

| id | lever | test | comparator | statement |
|---|---|---|---|---|
| H1 | vol_target | `ind10/equal_weight/L252/vt0.1-sample-max1.5/10bp` | `ind10/equal_weight/10bp` | Volatility targeting (10% target, sample covariance of the last 252 days, gross capped at 1.5) raises the net Sharpe ratio of 1/N on 10 industries. |
| H2 | cost_aware | `ind10/min_var_lo.lwcc/L252/partial0.5/10bp` | `ind10/min_var_lo.lwcc/L252/10bp` | Closing half the gap to target at each rebalance raises the net Sharpe ratio of long-only minimum variance (LW constant-correlation, 252 days) on 10 industries. |
| H3 | estimator | `ind49/min_var_lo.lwcc/L504/10bp` | `ind49/equal_weight/10bp` | Long-only minimum variance (LW constant-correlation, 504 days) has a higher net Sharpe ratio than 1/N on 49 industries. |
| H4 | estimator | `ind49/min_var_lo.sample/L504/10bp` | `ind49/min_var_lo.ewma/L504/10bp` | For long-only minimum variance on 49 industries, the sample covariance over 504 days gives a higher net Sharpe ratio than EWMA with decay 0.97 (the EWMA version trades about five times as much). |
| H5 | leverage | `ind10/tangency_cap2.sample/L504/10bp` | `ind10/equal_weight/10bp` | The tangency portfolio of 504-day sample moments with gross leverage capped at 2 has a higher net Sharpe ratio than 1/N on 10 industries (uncapped, the same rule lost all its capital in 1929 in development, so it cannot serve as comparator). |
| H6 | mean_shrinkage | `ind10/max_sharpe_lo_bs.sample/L252/10bp` | `ind10/max_sharpe_lo.sample/L252/10bp` | Bayes-Stein means raise the net Sharpe ratio of the long-only maximum-Sharpe portfolio on 10 industries (252 days). |

Development evidence and power. Power is approximate: the holdout standard error is the development one scaled by `sqrt(T_dev / T_holdout)`, and the true effect is taken to equal the development estimate (which, chosen as the best of several, is more likely too large than too small).

| id | dev dSR | dev SE | dev days | holdout SE (approx.) | power, alpha 0.05 | power, alpha 0.05/6 | effect for 80% power |
|---|---:|---:|---:|---:|---:|---:|---:|
| H1 | 0.13 | 0.056 | 23,111 | 0.164 | 0.20 | 0.06 | 0.41 |
| H2 | 0.01 | 0.005 | 23,111 | 0.015 | 0.20 | 0.06 | 0.04 |
| H3 | 0.15 | 0.101 | 10,575 | 0.200 | 0.18 | 0.05 | 0.50 |
| H4 | 0.13 | 0.061 | 10,575 | 0.120 | 0.28 | 0.09 | 0.30 |
| H5 | 0.13 | 0.078 | 23,111 | 0.228 | 0.14 | 0.03 | 0.57 |
| H6 | 0.00 | 0.025 | 23,111 | 0.072 | 0.04 | 0.01 | 0.18 |

Several hypotheses have little power; we register them anyway because a lever that cannot be shown to work on ten years of data is itself a result, and the effect sizes with intervals will be reported whatever the tests say.

## 5. Registered holdout configurations

Study `r2-holdout-v1`, registered 2026-10-31T17:19:04Z, 26 configurations, SHA-256 `d3dd8b8a038b9191f4b7474095db502439942193dce2492317e95fb01577604e`: every test and comparator configuration above, the two forward-rule configurations, and 1/N and the market for each universe, each at 10 bp and at the 25 bp stress level (descriptive; no test at 25 bp). Hypotheses SHA-256 `8a12088547f6357f949006757f85e5e6773f60164042c065797e270d5254d9da`. Phase 2 also re-evaluates these configurations on the development period and stops if any development Sharpe ratio differs from `real_data_study_dev.csv`.

`ind10/equal_weight/10bp`, `ind10/equal_weight/25bp`, `ind10/market`, `ind49/equal_weight/10bp`, `ind49/equal_weight/25bp`, `ind49/market`, `ind10/equal_weight/L252/vt0.1-sample-max1.5/10bp`, `ind10/equal_weight/L252/vt0.1-sample-max1.5/25bp`, `ind10/min_var_lo.lwcc/L252/partial0.5/10bp`, `ind10/min_var_lo.lwcc/L252/partial0.5/25bp`, `ind10/min_var_lo.lwcc/L252/10bp`, `ind10/min_var_lo.lwcc/L252/25bp`, `ind49/min_var_lo.lwcc/L504/10bp`, `ind49/min_var_lo.lwcc/L504/25bp`, `ind49/min_var_lo.sample/L504/10bp`, `ind49/min_var_lo.sample/L504/25bp`, `ind49/min_var_lo.ewma/L504/10bp`, `ind49/min_var_lo.ewma/L504/25bp`, `ind10/tangency_cap2.sample/L504/10bp`, `ind10/tangency_cap2.sample/L504/25bp`, `ind10/max_sharpe_lo_bs.sample/L252/10bp`, `ind10/max_sharpe_lo_bs.sample/L252/25bp`, `ind10/max_sharpe_lo.sample/L252/10bp`, `ind10/max_sharpe_lo.sample/L252/25bp`, `ind10/min_var_lo.lwcc/L504/10bp`, `ind10/min_var_lo.lwcc/L504/25bp`

## 6. Forward rules

Two rules are frozen now, from development evidence only, and will run whatever the holdout shows: they are registered as the study `r2-forward-v1` (2026-10-31T17:19:04Z), whose configuration list is the two rule files as written. A separate study keeps the forward log's trial count (two rules) apart from the holdout test's, and its data access (months from 2026-11 on) apart from the holdout's (2016-2026).

* `forward/rules/r2-minvar-lwcc-ind49.json`: Long-only minimum variance, Ledoit-Wolf constant-correlation covariance of the last 504 days, 49 industries, monthly, 10 bp; development configuration ind49/min_var_lo.lwcc/L504/10bp (hypothesis H3). Hypothetical research log on non-investable Kenneth R. French index series; no real money, not investment advice.
* `forward/rules/r2-minvar-lwcc-ind10.json`: Long-only minimum variance, Ledoit-Wolf constant-correlation covariance of the last 504 days, 10 industries, monthly, 10 bp, against the market (Mkt-RF + RF, no cost); development configuration ind10/min_var_lo.lwcc/L504/10bp. Hypothetical research log on non-investable Kenneth R. French index series; no real money, not investment advice.

Why these. The forward runner (`forward/run_forward.py`) can express fully invested weight rules with a band, not volatility targeting or partial rebalancing, so the largest development effect (volatility targeting of 1/N) cannot be logged without a runner change, and that effect sat mostly in the 1930s. Among the expressible rules, long-only minimum variance with Ledoit-Wolf constant-correlation shrinkage had the largest Sharpe gain over 1/N on 49 industries at about two thirds of the volatility, a turnover of about 1.3 a year, and an estimator in the pinned core. The 10-industry version is logged against the market, the benchmark an index investor would hold. Frozen at 2026-11-01 (first logged month 2026-11, published once the library's November data appears, early 2027), git tag `forward-v1`, cost 10 bp, no band.

## 7. What we will not do

* Re-run the holdout with changed configurations, windows, costs or tests.
* Add hypotheses after seeing holdout data. Anything else computed on the holdout will be labelled exploratory.
* Edit a forward rule. A changed rule is a new rule id with a new registration.
* If a bug is found after phase 2: report it, describe the fix, rerun once with the fix, and publish both results.

## 8. Limits stated in advance

The series are not investable and costs are assumed; one data vintage; the holdout is short relative to the effects (section 4); the construction rules tested were published before 2016, so the holdout is clean with respect to this repository's choices, not to the literature.
