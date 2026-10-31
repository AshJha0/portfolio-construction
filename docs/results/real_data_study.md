# Portfolio-construction levers on real data: pre-registered holdout result

Phase 2 of the protocol in `real_data_preregistration.md`, run once at commit `2941d7df494f6501f05f454166ef1a92562d9d89` (pre-registration commit 2941d7d), on study `r2-holdout-v1` (26 configurations). Non-investable index series; assumed costs; not investment advice.

## Pre-registered tests (10 bp, one-sided, Holm-adjusted)

| id | hypothesis | dev dSR (p one-sided) | holdout dSR [95% CI] | holdout p | Holm p | decision |
|---|---|---|---:|---:|---:|---|
| H1 | Volatility targeting (10% target, sample covariance of the last 252 days, gross capped at 1.5) raises the net Sharpe ratio of 1/N on 10 industries. | 0.13 (0.01) | -0.19 [-0.38, 0.00] | 0.978 | 1.000 | not supported |
| H2 | Closing half the gap to target at each rebalance raises the net Sharpe ratio of long-only minimum variance (LW constant-correlation, 252 days) on 10 industries. | 0.01 (0.01) | -0.01 [-0.05, 0.02] | 0.734 | 1.000 | not supported |
| H3 | Long-only minimum variance (LW constant-correlation, 504 days) has a higher net Sharpe ratio than 1/N on 49 industries. | 0.15 (0.08) | -0.15 [-0.50, 0.20] | 0.796 | 1.000 | not supported |
| H4 | For long-only minimum variance on 49 industries, the sample covariance over 504 days gives a higher net Sharpe ratio than EWMA with decay 0.97 (the EWMA version trades about five times as much). | 0.13 (0.02) | -0.08 [-0.30, 0.14] | 0.759 | 1.000 | not supported |
| H5 | The tangency portfolio of 504-day sample moments with gross leverage capped at 2 has a higher net Sharpe ratio than 1/N on 10 industries (uncapped, the same rule lost all its capital in 1929 in development, so it cannot serve as comparator). | 0.13 (0.05) | -0.40 [-0.84, 0.04] | 0.962 | 1.000 | not supported |
| H6 | Bayes-Stein means raise the net Sharpe ratio of the long-only maximum-Sharpe portfolio on 10 industries (252 days). | 0.00 (0.56) | -0.08 [-0.21, 0.04] | 0.902 | 1.000 | not supported |

## Development and holdout side by side (10 bp unless stated)

| universe | configuration | dev SR | holdout SR [95% CI] | holdout vol % | holdout max DD % | turnover/yr | vs 1/N | vs market |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 10 ind. | 1/N | 0.47 | 0.67 [0.19, 1.26] | 17.1 | 37 | 0.36 |  | -0.06 (0.48) |
| 10 ind. | 1/N (25 bp) | 0.46 | 0.67 | 17.1 | 37 | 0.36 |  | -0.06 (0.45) |
| 10 ind. | market | 0.41 | 0.73 [0.24, 1.28] | 18.3 | 34 |  | 0.06 (0.48) |  |
| 10 ind. | 1/N, L 252, vol target 0.1 (sample, max 1.5) | 0.60 | 0.48 [0.00, 1.09] | 11.9 | 30 | 0.59 | -0.19 (0.04) | -0.25 (0.03) |
| 10 ind. | 1/N, L 252, vol target 0.1 (sample, max 1.5) (25 bp) | 0.59 | 0.47 | 11.9 | 30 | 0.59 | -0.20 (0.04) | -0.26 (0.03) |
| 10 ind. | min-var, LW const-corr, L 252, partial 0.5 | 0.50 | 0.46 [0.04, 1.01] | 14.6 | 33 | 1.20 | -0.21 (0.14) | -0.27 (0.11) |
| 10 ind. | min-var, LW const-corr, L 252, partial 0.5 (25 bp) | 0.49 | 0.44 | 14.6 | 33 | 1.20 | -0.22 (0.12) | -0.28 (0.10) |
| 10 ind. | min-var, LW const-corr, L 252 | 0.49 | 0.47 [0.06, 1.01] | 14.4 | 33 | 1.81 | -0.20 (0.17) | -0.26 (0.13) |
| 10 ind. | min-var, LW const-corr, L 252 (25 bp) | 0.47 | 0.45 | 14.5 | 33 | 1.81 | -0.22 (0.14) | -0.28 (0.11) |
| 10 ind. | tangency cap 2, L 504 | 0.59 | 0.27 [-0.23, 0.80] | 16.9 | 29 | 7.85 | -0.40 (0.08) | -0.46 (0.04) |
| 10 ind. | tangency cap 2, L 504 (25 bp) | 0.53 | 0.20 | 17.0 | 30 | 7.85 | -0.46 (0.04) | -0.53 (0.02) |
| 10 ind. | max-Sharpe BS, L 252 | 0.51 | 0.75 [0.28, 1.27] | 20.5 | 33 | 6.86 | 0.08 (0.69) | 0.02 (0.92) |
| 10 ind. | max-Sharpe BS, L 252 (25 bp) | 0.46 | 0.70 | 20.6 | 33 | 6.87 | 0.03 (0.87) | -0.03 (0.89) |
| 10 ind. | max-Sharpe, L 252 | 0.52 | 0.83 [0.37, 1.37] | 22.7 | 32 | 6.18 | 0.16 (0.45) | 0.11 (0.64) |
| 10 ind. | max-Sharpe, L 252 (25 bp) | 0.46 | 0.79 | 22.7 | 32 | 6.19 | 0.13 (0.56) | 0.06 (0.78) |
| 10 ind. | min-var, LW const-corr, L 504 | 0.50 | 0.38 [-0.03, 0.94] | 14.8 | 35 | 1.05 | -0.29 (0.05) | -0.34 (0.05) |
| 10 ind. | min-var, LW const-corr, L 504 (25 bp) | 0.49 | 0.37 | 14.8 | 35 | 1.05 | -0.29 (0.04) | -0.36 (0.04) |
| 49 ind. | 1/N | 0.49 | 0.64 [0.11, 1.24] | 18.2 | 38 | 0.43 |  | -0.09 (0.38) |
| 49 ind. | 1/N (25 bp) | 0.48 | 0.64 | 18.2 | 38 | 0.43 |  | -0.09 (0.35) |
| 49 ind. | market | 0.43 | 0.73 [0.24, 1.28] | 18.3 | 34 |  | 0.09 (0.38) |  |
| 49 ind. | min-var, LW const-corr, L 504 | 0.63 | 0.49 [0.05, 1.05] | 14.0 | 34 | 1.80 | -0.15 (0.41) | -0.23 (0.24) |
| 49 ind. | min-var, LW const-corr, L 504 (25 bp) | 0.61 | 0.47 | 14.0 | 34 | 1.80 | -0.16 (0.36) | -0.25 (0.20) |
| 49 ind. | min-var, sample, L 504 | 0.62 | 0.51 [0.06, 1.07] | 14.0 | 34 | 1.84 | -0.13 (0.48) | -0.22 (0.28) |
| 49 ind. | min-var, sample, L 504 (25 bp) | 0.61 | 0.49 | 14.0 | 34 | 1.84 | -0.14 (0.43) | -0.24 (0.24) |
| 49 ind. | min-var, EWMA 0.97, L 504 | 0.49 | 0.59 [0.13, 1.17] | 13.5 | 28 | 8.83 | -0.05 (0.80) | -0.13 (0.49) |
| 49 ind. | min-var, EWMA 0.97, L 504 (25 bp) | 0.40 | 0.49 | 13.6 | 28 | 8.83 | -0.14 (0.45) | -0.23 (0.23) |

## Conclusions

Holdout: 2016-01 to 2026-09 (2,701 trading days, vintage 1). Every number above is in `real_data_study.csv`.

1. **No pre-registered hypothesis is supported.** All six Holm-adjusted p-values are 1.00, and all six holdout differences point the wrong way. The development effects did not carry over, and none of the six levers showed a measurable net-of-cost improvement on 2016-2026.
2. **Volatility targeting (H1) hurt.** On 10 industries it trailed 1/N by 0.19 (95% interval -0.38 to 0.00), with lower volatility (11.9% vs 17.1%) and a smaller drawdown (30% vs 37%). In development its gain came almost entirely from 1928-1969. A decade of mostly rising markets with short, sharp sell-offs (2018, 2020) rewarded staying fully invested, and monthly scaling cut exposure after each spike, in time for the rebound.
3. **Minimum variance (H3, and both forward-rule configurations) lagged 1/N.** It lagged by 0.15 on 49 industries and by 0.29 on 10 industries. It again cut volatility by a fifth to a quarter, but low-volatility industries trailed in this decade. The 2000-2015 gains seen in development did not persist.
4. **Estimator choice (H4) and partial rebalancing (H2) made no measurable difference.** The EWMA version did better than the sample covariance at 10 bp (0.59 vs 0.51), which reverses the development sign. At 8.8 times NAV of turnover a year, the gap closes at 25 bp (0.49 vs 0.49).
5. **Capped tangency (H5) was the worst rule: 0.27, 0.40 behind 1/N.** It still traded 7.9 times NAV a year. Keeping it solvent was necessary but not sufficient.
6. **Bayes-Stein means (H6) did not improve on sample means.** The plain long-only max-Sharpe rule happened to beat 1/N in this decade (+0.16, p = 0.45), which is within noise.
7. **Benchmarks.** The market (0.73) beat 1/N on both universes (0.67 and 0.64). Every construction rule tested was within noise of, or below, the cap-weighted index.
8. **What this does and does not show.** The pre-registered holdout had low power (0.04 to 0.28 for the development effect sizes), so "not supported" does not prove these levers are worthless. It does show that their development-period gains were too small and too regime-dependent to rely on. The forward log will keep running both frozen minimum-variance rules as registered.

## Limits

* The series are not investable. Industry portfolios are value-weighted baskets of every CRSP stock in an industry, rebalanced by construction, with no trading cost of their own.
* Costs are assumed, linear and the same in 1930 and 2015. Real costs were far higher before the 1980s, which would penalize the higher-turnover rules more in the early decades.
* One data vintage. The French library rebuilds its history monthly; the numbers here are for vintage 1 (CRSP 202609).
* The development period was not blind: these rules and estimators were published by people who had seen much of this history. The holdout (2016 onward) is clean only with respect to this repository's own choices, which the registry makes countable.
* Standard errors that assume independence understate the uncertainty by about 20% (block bootstrap comparison above).

