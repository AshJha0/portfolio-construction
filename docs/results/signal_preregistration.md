# Pre-registration: return signals, seen period and forward rules (round 2)

Written on 2026-10-31 after the development run of `python/studies/signal_study.py --phase dev` and before any row dated 2016-01-01 or later was read by the signal study. Hypotheses, configurations and forward rules are fixed in `python/studies/signal_prereg.py`; `--register` records them in `data/real/registry.json` as `r2-signal-v1` (configurations, with the SHA-256 of the hypothesis list in the description) and `r2-forward-signals-v1` (the two rule files as written). `--phase seen` refuses to run if either differs from the code, records `git rev-parse HEAD`, and refuses a second run.

## 1. What this test can and cannot show

The 2016-01 to 2026-09 period is **a seen period, not a clean holdout.** The construction study (`real_data_study.md`) was run on it and its results were read before these signals were chosen; in particular we knew that 1/N and the market did well and minimum variance lagged. Signals were not computed on 2016+ before this file, but the choice of what to test could still be shaped by that knowledge. Results on 2016-2026 are therefore labelled "seen period" and are weaker evidence than a holdout. The clean evidence is the forward log from 2026-11 (section 5).

The signals themselves were published between 1972 and 2012 by people who had seen much of the development history, so development results are in-sample for the literature (McLean and Pontiff 2016 report a decline of about a third after publication for anomalies in stock returns).

## 2. Data and method (frozen)

Vintage 1 of `ff3_daily`, `ind10_vw_daily` and `ind49_vw_daily` (CRSP 202609; hashes in `real_data_preregistration.md` section 2), 10 industries from 1926-07 and 49 industries from 1972-01. Monthly rebalancing on the first trading day of each month, `run_schedule_net` accounting, cost on traded notional (10 bp base, 25 bp stress), cash at the daily `RF`. Each signal reads only rows before the rebalance row; a month is 21 trading days (`portopt.research.signals`, look-ahead test in `python/tests/test_research_signals.py`). Configurations: `python/studies/signal_core.py` (`IC = 0.05`, risk aversion 5, BL `delta = 2.5`, `tau = 0.05`, confidence 1, 1/N equilibrium, LW constant-correlation covariance of 504 days). These parameters are assumptions set before the development run, not tuned.

Development (study `r2-signal-dev`, 66 configurations, all registered before the run): 5 signals and 2 combinations (equal and inverse-volatility weighted z-scores of momentum, time-series momentum, reversal and low beta) under three rules (top tercile on both universes; turnover-penalised mean-variance and Black-Litterman on 10 industries only, because the Python QP takes about 0.4 to 0.8 s per solve at n = 10 and minutes at n = 49), at 10 and 25 bp, plus 1/N, minimum variance and the market. Together with the 206 construction trials (`r2-dev-exploration` 198, `r2-dev-exploration-tc` 8) this analyst has run 272 development trials on this data.

## 3. Hypotheses

Each is one-sided, `SR(test) > SR(comparator)`, on annualized net Sharpe ratios at 10 bp over 2016-01 to 2026-09, tested by the Ledoit and Wolf (2008) HAC test (`sharpe_difference_test`), p = 1 - Phi(z). The 4 p-values are Holm-adjusted; supported when the adjusted p is at most 0.05. The 25 bp results are reported, not tested.

| id | test | comparator | statement |
|---|---|---|---|
| S1 | `ind10/bl.mom12_1/10bp` | `ind10/equal_weight/10bp` | Black-Litterman with 12-1 momentum views (IC 0.05) has a higher net Sharpe ratio than 1/N on 10 industries. |
| S2 | `ind49/rank.mom12_1/10bp` | `ind49/equal_weight/10bp` | The top tercile by 12-1 momentum has a higher net Sharpe ratio than 1/N on 49 industries. |
| S3 | `ind10/mvtc.mom12_1/10bp` | `ind10/minvar.lwcc/L504/10bp` | Turnover-penalised mean-variance with 12-1 momentum alphas has a higher net Sharpe ratio than long-only minimum variance (LW constant-correlation, 504 days) on 10 industries. |
| S4 | `ind49/rank.lowvol/10bp` | `ind49/equal_weight/10bp` | The top tercile by low 12-month volatility has a higher net Sharpe ratio than 1/N on 49 industries. |

Why these. Momentum was the only signal with a positive difference against 1/N in every development sub-period on both universes, and the only one whose Black-Litterman and mean-variance versions also gained. Low volatility on 49 industries was the strongest of the low-risk signals there. Short-term reversal lost heavily (about -0.3 a year in Sharpe against 1/N, with turnover of 15 times NAV), consistent with Moskowitz and Grinblatt (1999), who found one-month industry momentum rather than reversal; the two combinations inherit that loss and are not tested.

Development evidence and approximate power (holdout SE = development SE x sqrt(T_dev / 2,701); true effect taken equal to the development estimate, which, as the best of several, is more likely too large):

| id | dev dSR | dev SE | dev days | seen SE (approx.) | power, alpha 0.05 | power, alpha 0.05/4 | effect for 80% power |
|---|---:|---:|---:|---:|---:|---:|---:|
| S1 | 0.10 | 0.044 | 23,111 | 0.129 | 0.19 | 0.07 | 0.32 |
| S2 | 0.11 | 0.058 | 10,575 | 0.115 | 0.24 | 0.10 | 0.29 |
| S3 | 0.06 | 0.065 | 23,111 | 0.189 | 0.09 | 0.03 | 0.47 |
| S4 | 0.09 | 0.048 | 10,575 | 0.094 | 0.24 | 0.09 | 0.23 |

Power is low; "not supported" will not mean "no effect".

## 4. Seen-period configurations

Study `r2-signal-v1`, registered 2026-10-31T18:58:16Z, 17 configurations, SHA-256 `593721e2914c...` (full value in the registry), hypotheses SHA-256 `ac447b6cba3716e4611e66a7518c53c3e762da9e567c70dcbc634522b6ecbeaa`: the test and comparator of each hypothesis at 10 and 25 bp, 1/N and minimum variance at 10 bp and the market on each universe. Nothing else will be run on 2016+ under this study.

## 5. Forward rules

Registered as `r2-forward-signals-v1` (2026-10-31T18:58:16Z), frozen at 2026-11-01, git tag `forward-v1`, 10 bp, no band, monthly, benchmark 1/N at the same cost. Both are stateless (the forward runner passes a fixed trailing window, so the path-dependent turnover-penalised rule cannot run there) and both are exactly the study configurations named.

* `forward/rules/r2-signal-mom-bl-ind10.json`: `signal_black_litterman`, momentum views, 10 industries, 504 days (S1).
* `forward/rules/r2-signal-mom-rank-ind49.json`: `rank_top_tercile`, momentum, 49 industries, 252 days (S2).

## 6. What we will not do

Re-run the seen period with changed configurations; add hypotheses after seeing it (anything else computed there is exploratory); edit a forward rule (a change is a new rule id and a new registration). A bug found after the run is reported, fixed, rerun once, and both results published.
