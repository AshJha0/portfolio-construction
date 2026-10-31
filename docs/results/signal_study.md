# Return signals on real index data (round 2)

Development 1926-07 to 2015-12 (`period="development"`, study `r2-signal-dev`); seen period 2016-01 to 2026-09 (study `r2-signal-v1`). Every number is in `signal_study.csv`, produced by `python/studies/signal_study.py`. Evidence class: backtest on non-investable index series with assumed linear costs; not investment advice.

## Pre-registered hypotheses (10 bp, one-sided Ledoit-Wolf, Holm over the family)

The 2016-2026 period is a seen period, not a clean holdout: the construction study's results on it were read before these signals were chosen. The clean evidence is the forward log from 2026-11.

| id | hypothesis | dev dSR (p) | seen dSR (p) | Holm p | decision |
|---|---|---|---|---|---|
| S1 | Black-Litterman with 12-1 momentum views (IC 0.05) has a higher net Sharpe ratio than 1/N on 10 industries. | 0.10 (0.01) | 0.01 (0.46) | 0.92 | not supported |
| S2 | The top tercile by 12-1 momentum has a higher net Sharpe ratio than 1/N on 49 industries. | 0.11 (0.03) | 0.07 (0.27) | 0.81 | not supported |
| S3 | Turnover-penalised mean-variance with 12-1 momentum alphas has a higher net Sharpe ratio than long-only minimum variance (LW constant-correlation, 504 days) on 10 industries. | 0.06 (0.17) | 0.12 (0.19) | 0.77 | not supported |
| S4 | The top tercile by low 12-month volatility has a higher net Sharpe ratio than 1/N on 49 industries. | 0.09 (0.03) | -0.16 (0.92) | 0.92 | not supported |

## Registered configurations, development and seen period

| universe | configuration | dev SR | seen SR | seen SE | vol % | max DD % | turnover/yr | vs 1/N | vs min-var |
|---|---|---|---|---|---|---|---|---|---|
| ind10 | BL views mom12_1 | 0.57 | 0.68 | 0.31 | 19.3 | 33 | 5.21 | 0.01 (0.46) | 0.30 (0.06) |
| ind10 | 1/N | 0.47 | 0.67 | 0.31 | 17.1 | 37 | 0.36 |  | 0.29 (0.02) |
| ind10 | BL views mom12_1 (25 bp) | 0.53 | 0.64 | 0.31 | 19.3 | 34 | 5.21 | -0.02 (0.57) | 0.27 (0.09) |
| ind10 | 1/N (25 bp) | 0.46 | 0.67 | 0.31 | 17.1 | 37 | 0.36 |  | 0.29 (0.02) |
| ind49 | top tercile mom12_1 | 0.60 | 0.71 | 0.31 | 19.2 | 36 | 4.78 | 0.07 (0.27) | 0.22 (0.13) |
| ind49 | 1/N | 0.49 | 0.64 | 0.31 | 18.2 | 38 | 0.43 |  | 0.15 (0.20) |
| ind49 | top tercile mom12_1 (25 bp) | 0.55 | 0.67 | 0.31 | 19.3 | 36 | 4.79 | 0.03 (0.38) |  |
| ind49 | 1/N (25 bp) | 0.48 | 0.64 | 0.31 | 18.2 | 38 | 0.43 |  |  |
| ind10 | MV + turnover mom12_1 | 0.56 | 0.51 | 0.31 | 16.9 | 34 | 2.21 | -0.16 (0.88) | 0.12 (0.19) |
| ind10 | min-var LW-CC L504 | 0.50 | 0.38 | 0.31 | 14.8 | 35 | 1.05 | -0.29 (0.98) |  |
| ind10 | MV + turnover mom12_1 (25 bp) | 0.49 | 0.43 | 0.31 | 15.8 | 35 | 0.80 | -0.23 (0.97) | 0.06 (0.29) |
| ind10 | min-var LW-CC L504 (25 bp) | 0.49 | 0.37 | 0.31 | 14.8 | 35 | 1.05 | -0.29 (0.98) |  |
| ind49 | top tercile lowvol | 0.57 | 0.48 | 0.31 | 14.9 | 34 | 1.57 | -0.16 (0.92) | -0.01 (0.55) |
| ind49 | top tercile lowvol (25 bp) | 0.56 | 0.46 | 0.31 | 14.9 | 34 | 1.58 | -0.17 (0.93) |  |
| ind10 | market | 0.41 | 0.73 | 0.31 | 18.3 | 34 | 0.00 | 0.06 (0.24) | 0.34 (0.02) |
| ind49 | market | 0.43 | 0.73 | 0.31 | 18.3 | 34 | 0.00 | 0.09 (0.19) | 0.23 (0.12) |
| ind49 | min-var LW-CC L504 | 0.63 | 0.49 | 0.31 | 14.0 | 34 | 1.80 | -0.15 (0.80) |  |

## Conclusions

1. **No registered hypothesis is supported** (Holm p 0.77 to 0.92). Three of four seen-period differences have the registered sign (S1 +0.01, S2 +0.07, S3 +0.12) but all are well inside one standard error (about 0.1 to 0.2); S4 reversed (-0.16).
2. **Momentum held up best.** Momentum rules kept their development level in the seen period (top tercile on 49 industries 0.71 against 0.64 for 1/N; BL on 10 industries 0.68 against 0.67) but did not beat the market (0.73). At 25 bp the gains shrink to +0.03 and -0.02.
3. **Low-risk signals lagged** in 2016-2026, as minimum variance did in the construction study: low volatility on 49 industries 0.48, 1/N 0.64.
4. **Short-term reversal does not work on industries** in development (about -0.3 Sharpe against 1/N in every sub-period, 15 times NAV turnover a year), so the two combinations that include it lost too. One-month industry returns continue, they do not reverse (Moskowitz and Grinblatt 1999).
5. **Publication split** (development only, to 2015): momentum's gain on 10 industries was larger after 1993 than before, and low volatility's after 2011 rests on five years; none of these splits has the power to show the decay McLean and Pontiff (2016) report.
6. **Trial count.** 66 signal trials plus 206 construction trials. The deflated Sharpe ratio in the development table tests each strategy's Sharpe ratio against the expected maximum of that many null trials, not its difference from 1/N, so a high DSR here says the strategy beat cash, not that the signal added anything.

## Limits

* The seen period was not blind (section 1 of `signal_preregistration.md`); the registration was recorded in `data/real/registry.json` at 2026-10-31T18:58:16Z and the run made at commit fadb496 with the pre-registration not yet committed.
* A month is 21 trading days, not a calendar month; the signal windows drift by a few days against calendar months.
* IC 0.05, risk aversion 5 and the BL parameters are assumptions, not estimates; mean-variance and BL ran on 10 industries only (QP runtime).
* Index series are not investable; costs are linear assumptions, the same in 1930 and 2015.
* Time-series momentum is used here as a long-only cross-sectional score (sign over volatility), not as the long-short futures strategy of Moskowitz, Ooi and Pedersen.

## Development, full period

Net Sharpe (annualized, excess of RF); differences in Sharpe vs 1/N and vs long-only minimum variance (LW constant-correlation, 504 days) at the same cost, one-sided Ledoit-Wolf p in brackets. DSR uses the variance of Sharpe ratios across the signal trials, with the signal trial count and with the construction trials added.

| universe | configuration | SR | SE | vol % | max DD % | turnover/yr | vs 1/N | vs min-var | DSR (sig) | DSR (all) |
|---|---|---|---|---|---|---|---|---|---|---|
| ind10 | market | 0.41 | 0.10 | 17.1 | 84 | 0.00 | -0.06 (1.00) | -0.10 (0.93) | 0.76 | 0.53 |
| ind10 | 1/N | 0.47 | 0.10 | 16.5 | 82 | 0.27 |  | -0.03 (0.72) | 0.90 | 0.74 |
| ind10 | min-var LW-CC L504 | 0.50 | 0.10 | 12.1 | 75 | 0.80 | 0.03 (0.28) |  | 0.95 | 0.84 |
| ind10 | top tercile mom12_1 | 0.51 | 0.10 | 17.6 | 78 | 4.91 | 0.04 (0.15) | 0.01 (0.45) | 0.96 | 0.86 |
| ind10 | top tercile tsmom | 0.38 | 0.10 | 17.1 | 86 | 4.03 | -0.08 (0.98) | -0.12 (0.97) | 0.70 | 0.45 |
| ind10 | top tercile strev | 0.13 | 0.10 | 18.5 | 87 | 15.42 | -0.33 (1.00) | -0.37 (1.00) | 0.03 | 0.01 |
| ind10 | top tercile lowvol | 0.52 | 0.11 | 12.9 | 78 | 1.21 | 0.06 (0.08) | 0.02 (0.29) | 0.97 | 0.88 |
| ind10 | top tercile lowbeta | 0.55 | 0.11 | 12.8 | 75 | 1.15 | 0.08 (0.04) | 0.05 (0.13) | 0.98 | 0.92 |
| ind10 | top tercile combo_eq | 0.37 | 0.10 | 15.5 | 81 | 9.26 | -0.10 (0.99) | -0.13 (0.99) | 0.64 | 0.39 |
| ind10 | top tercile combo_iv | 0.36 | 0.10 | 15.8 | 81 | 9.78 | -0.11 (1.00) | -0.14 (0.99) | 0.61 | 0.36 |
| ind10 | MV + turnover mom12_1 | 0.56 | 0.11 | 15.0 | 76 | 1.92 | 0.09 (0.04) | 0.06 (0.17) | 0.99 | 0.94 |
| ind10 | MV + turnover tsmom | 0.38 | 0.10 | 13.7 | 80 | 2.20 | -0.09 (0.94) | -0.12 (0.99) | 0.68 | 0.43 |
| ind10 | MV + turnover strev | 0.29 | 0.10 | 14.2 | 78 | 7.79 | -0.18 (1.00) | -0.21 (1.00) | 0.35 | 0.15 |
| ind10 | MV + turnover lowvol | 0.50 | 0.10 | 12.2 | 73 | 0.17 | 0.03 (0.32) | -0.00 (0.56) | 0.94 | 0.83 |
| ind10 | MV + turnover lowbeta | 0.50 | 0.10 | 12.4 | 76 | 0.21 | 0.03 (0.30) | -0.00 (0.50) | 0.95 | 0.84 |
| ind10 | MV + turnover combo_eq | 0.44 | 0.11 | 13.2 | 71 | 2.74 | -0.02 (0.65) | -0.06 (0.90) | 0.86 | 0.67 |
| ind10 | MV + turnover combo_iv | 0.46 | 0.11 | 13.3 | 71 | 3.02 | -0.01 (0.56) | -0.04 (0.83) | 0.89 | 0.71 |
| ind10 | BL views mom12_1 | 0.57 | 0.11 | 17.8 | 80 | 4.85 | 0.10 (0.01) | 0.07 (0.19) | 0.99 | 0.94 |
| ind10 | BL views tsmom | 0.39 | 0.10 | 16.5 | 84 | 5.08 | -0.08 (0.99) | -0.11 (0.97) | 0.70 | 0.46 |
| ind10 | BL views strev | 0.11 | 0.10 | 18.1 | 85 | 15.72 | -0.35 (1.00) | -0.39 (1.00) | 0.02 | 0.00 |
| ind10 | BL views lowvol | 0.49 | 0.10 | 13.7 | 78 | 1.58 | 0.03 (0.21) | -0.01 (0.56) | 0.94 | 0.82 |
| ind10 | BL views lowbeta | 0.50 | 0.10 | 13.7 | 76 | 1.48 | 0.03 (0.22) | -0.00 (0.52) | 0.95 | 0.83 |
| ind10 | BL views combo_eq | 0.35 | 0.10 | 15.9 | 80 | 10.83 | -0.12 (1.00) | -0.15 (0.99) | 0.56 | 0.31 |
| ind10 | BL views combo_iv | 0.35 | 0.10 | 16.1 | 80 | 11.10 | -0.12 (1.00) | -0.15 (0.99) | 0.58 | 0.33 |
| ind10 | 1/N (25 bp) | 0.46 | 0.10 | 16.5 | 82 | 0.27 |  | -0.03 (0.68) | 0.90 | 0.74 |
| ind10 | min-var LW-CC L504 (25 bp) | 0.49 | 0.10 | 12.1 | 75 | 0.80 | 0.03 (0.32) |  | 0.94 | 0.81 |
| ind10 | top tercile mom12_1 (25 bp) | 0.47 | 0.10 | 17.6 | 79 | 4.91 | 0.01 (0.44) | -0.02 (0.62) | 0.91 | 0.75 |
| ind10 | top tercile tsmom (25 bp) | 0.35 | 0.10 | 17.1 | 86 | 4.03 | -0.11 (1.00) | -0.14 (0.99) | 0.57 | 0.32 |
| ind10 | top tercile strev (25 bp) | 0.01 | 0.10 | 18.5 | 88 | 15.43 | -0.45 (1.00) | -0.48 (1.00) | 0.00 | 0.00 |
| ind10 | top tercile lowvol (25 bp) | 0.51 | 0.11 | 12.9 | 78 | 1.21 | 0.05 (0.13) | 0.02 (0.33) | 0.96 | 0.86 |
| ind10 | top tercile lowbeta (25 bp) | 0.53 | 0.11 | 12.8 | 75 | 1.15 | 0.07 (0.06) | 0.04 (0.14) | 0.97 | 0.90 |
| ind10 | top tercile combo_eq (25 bp) | 0.28 | 0.10 | 15.6 | 82 | 9.26 | -0.18 (1.00) | -0.21 (1.00) | 0.33 | 0.14 |
| ind10 | top tercile combo_iv (25 bp) | 0.27 | 0.10 | 15.8 | 82 | 9.78 | -0.19 (1.00) | -0.22 (1.00) | 0.28 | 0.11 |
| ind10 | MV + turnover mom12_1 (25 bp) | 0.49 | 0.11 | 13.8 | 77 | 0.62 | 0.03 (0.27) | 0.00 (0.47) | 0.94 | 0.82 |
| ind10 | MV + turnover tsmom (25 bp) | 0.41 | 0.10 | 13.5 | 80 | 0.74 | -0.05 (0.82) | -0.08 (0.98) | 0.78 | 0.55 |
| ind10 | MV + turnover strev (25 bp) | 0.45 | 0.10 | 13.1 | 77 | 1.74 | -0.02 (0.62) | -0.04 (0.91) | 0.87 | 0.68 |
| ind10 | MV + turnover lowvol (25 bp) | 0.52 | 0.10 | 12.7 | 74 | 0.09 | 0.06 (0.13) | 0.03 (0.23) | 0.96 | 0.88 |
| ind10 | MV + turnover lowbeta (25 bp) | 0.49 | 0.10 | 12.8 | 77 | 0.11 | 0.03 (0.31) | 0.00 (0.48) | 0.94 | 0.82 |
| ind10 | MV + turnover combo_eq (25 bp) | 0.47 | 0.10 | 12.9 | 76 | 0.63 | 0.01 (0.45) | -0.02 (0.73) | 0.91 | 0.76 |
| ind10 | MV + turnover combo_iv (25 bp) | 0.48 | 0.10 | 12.9 | 75 | 0.69 | 0.02 (0.38) | -0.01 (0.61) | 0.92 | 0.79 |
| ind10 | BL views mom12_1 (25 bp) | 0.53 | 0.11 | 17.8 | 80 | 4.85 | 0.06 (0.08) | 0.04 (0.31) | 0.97 | 0.89 |
| ind10 | BL views tsmom (25 bp) | 0.34 | 0.10 | 16.6 | 84 | 5.09 | -0.12 (1.00) | -0.15 (0.99) | 0.54 | 0.30 |
| ind10 | BL views strev (25 bp) | -0.01 | 0.10 | 18.2 | 86 | 15.73 | -0.47 (1.00) | -0.50 (1.00) | 0.00 | 0.00 |
| ind10 | BL views lowvol (25 bp) | 0.48 | 0.10 | 13.7 | 78 | 1.58 | 0.01 (0.34) | -0.01 (0.63) | 0.92 | 0.78 |
| ind10 | BL views lowbeta (25 bp) | 0.48 | 0.10 | 13.7 | 77 | 1.48 | 0.02 (0.33) | -0.01 (0.58) | 0.93 | 0.79 |
| ind10 | BL views combo_eq (25 bp) | 0.25 | 0.10 | 15.9 | 81 | 10.84 | -0.22 (1.00) | -0.24 (1.00) | 0.21 | 0.08 |
| ind10 | BL views combo_iv (25 bp) | 0.25 | 0.10 | 16.1 | 81 | 11.11 | -0.21 (1.00) | -0.24 (1.00) | 0.22 | 0.08 |
| ind49 | market | 0.43 | 0.16 | 16.8 | 55 | 0.00 | -0.06 (0.92) | -0.21 (0.97) | 0.73 | 0.57 |
| ind49 | 1/N | 0.49 | 0.16 | 16.4 | 57 | 0.38 |  | -0.15 (0.92) | 0.84 | 0.71 |
| ind49 | min-var LW-CC L504 | 0.63 | 0.16 | 11.2 | 41 | 1.30 | 0.15 (0.08) |  | 0.97 | 0.93 |
| ind49 | top tercile mom12_1 | 0.60 | 0.16 | 17.3 | 53 | 4.72 | 0.11 (0.03) | -0.04 (0.63) | 0.95 | 0.90 |
| ind49 | top tercile tsmom | 0.53 | 0.16 | 16.8 | 65 | 3.85 | 0.04 (0.20) | -0.10 (0.84) | 0.90 | 0.80 |
| ind49 | top tercile strev | 0.22 | 0.15 | 18.3 | 69 | 15.46 | -0.27 (1.00) | -0.41 (1.00) | 0.24 | 0.12 |
| ind49 | top tercile lowvol | 0.57 | 0.16 | 13.5 | 46 | 1.45 | 0.09 (0.03) | -0.06 (0.74) | 0.94 | 0.87 |
| ind49 | top tercile lowbeta | 0.57 | 0.16 | 12.8 | 48 | 1.70 | 0.09 (0.09) | -0.06 (0.76) | 0.94 | 0.87 |
| ind49 | top tercile combo_eq | 0.45 | 0.16 | 15.3 | 59 | 9.08 | -0.04 (0.76) | -0.18 (0.97) | 0.78 | 0.63 |
| ind49 | top tercile combo_iv | 0.46 | 0.16 | 15.7 | 62 | 9.25 | -0.02 (0.68) | -0.17 (0.96) | 0.80 | 0.66 |
| ind49 | 1/N (25 bp) | 0.48 | 0.16 | 16.4 | 57 | 0.38 |  | -0.13 (0.90) | 0.84 | 0.71 |
| ind49 | min-var LW-CC L504 (25 bp) | 0.61 | 0.16 | 11.2 | 41 | 1.30 | 0.13 (0.10) |  | 0.96 | 0.92 |
| ind49 | top tercile mom12_1 (25 bp) | 0.55 | 0.16 | 17.3 | 53 | 4.72 | 0.07 (0.11) | -0.06 (0.70) | 0.92 | 0.84 |
| ind49 | top tercile tsmom (25 bp) | 0.49 | 0.16 | 16.9 | 66 | 3.85 | 0.01 (0.41) | -0.12 (0.88) | 0.85 | 0.73 |
| ind49 | top tercile strev (25 bp) | 0.09 | 0.15 | 18.3 | 70 | 15.47 | -0.39 (1.00) | -0.52 (1.00) | 0.06 | 0.02 |
| ind49 | top tercile lowvol (25 bp) | 0.56 | 0.16 | 13.5 | 47 | 1.45 | 0.07 (0.06) | -0.06 (0.74) | 0.93 | 0.85 |
| ind49 | top tercile lowbeta (25 bp) | 0.55 | 0.16 | 12.8 | 48 | 1.70 | 0.07 (0.14) | -0.06 (0.77) | 0.92 | 0.84 |
| ind49 | top tercile combo_eq (25 bp) | 0.36 | 0.16 | 15.3 | 60 | 9.08 | -0.12 (0.99) | -0.25 (1.00) | 0.58 | 0.41 |
| ind49 | top tercile combo_iv (25 bp) | 0.37 | 0.16 | 15.7 | 63 | 9.25 | -0.11 (0.99) | -0.24 (0.99) | 0.61 | 0.44 |

## Development sub-periods and publication split (10 bp, difference vs 1/N)

| universe | configuration | pre-1963 | 1963-1990 | 1991-2015 | pub. year | before pub. | after pub. (to 2015) |
|---|---|---|---|---|---|---|---|
| ind10 | min-var LW-CC L504 | -0.01 (0.53) | 0.00 (0.49) | 0.12 (0.15) |  |  |  |
| ind10 | top tercile mom12_1 | 0.04 (0.23) | -0.01 (0.56) | 0.10 (0.14) | 1993 | 0.02 (0.32) | 0.11 (0.12) |
| ind10 | top tercile tsmom | -0.09 (0.95) | -0.11 (0.94) | -0.05 (0.72) | 2012 | -0.09 (0.98) | 0.18 (0.19) |
| ind10 | top tercile strev | -0.31 (1.00) | -0.55 (1.00) | -0.22 (1.00) | 1990 | -0.38 (1.00) | -0.23 (1.00) |
| ind10 | top tercile lowvol | 0.05 (0.20) | 0.00 (0.48) | 0.11 (0.09) | 2011 | 0.04 (0.18) | 0.38 (0.01) |
| ind10 | top tercile lowbeta | 0.05 (0.16) | 0.07 (0.22) | 0.13 (0.10) | 1972 | 0.05 (0.19) | 0.12 (0.05) |
| ind10 | top tercile combo_eq | -0.06 (0.86) | -0.22 (1.00) | -0.06 (0.74) |  |  |  |
| ind10 | top tercile combo_iv | -0.07 (0.89) | -0.25 (1.00) | -0.06 (0.74) |  |  |  |
| ind10 | MV + turnover mom12_1 | 0.08 (0.12) | 0.08 (0.24) | 0.12 (0.11) | 1993 | 0.08 (0.09) | 0.13 (0.11) |
| ind10 | MV + turnover tsmom | -0.10 (0.93) | -0.16 (0.91) | -0.00 (0.51) | 2012 | -0.10 (0.95) | 0.12 (0.32) |
| ind10 | MV + turnover strev | -0.13 (0.97) | -0.29 (0.99) | -0.17 (0.96) | 1990 | -0.19 (1.00) | -0.15 (0.94) |
| ind10 | MV + turnover lowvol | 0.03 (0.31) | -0.07 (0.65) | 0.10 (0.22) | 2011 | 0.01 (0.44) | 0.33 (0.05) |
| ind10 | MV + turnover lowbeta | 0.04 (0.28) | -0.06 (0.64) | 0.09 (0.25) | 1972 | -0.03 (0.64) | 0.10 (0.18) |
| ind10 | MV + turnover combo_eq | -0.00 (0.53) | -0.18 (0.92) | 0.09 (0.24) |  |  |  |
| ind10 | MV + turnover combo_iv | -0.00 (0.51) | -0.17 (0.91) | 0.12 (0.17) |  |  |  |
| ind10 | BL views mom12_1 | 0.10 (0.04) | 0.10 (0.17) | 0.10 (0.12) | 1993 | 0.09 (0.03) | 0.12 (0.09) |
| ind10 | BL views tsmom | -0.11 (0.99) | -0.12 (0.93) | 0.01 (0.46) | 2012 | -0.08 (0.99) | 0.14 (0.21) |
| ind10 | BL views strev | -0.28 (1.00) | -0.61 (1.00) | -0.26 (1.00) | 1990 | -0.38 (1.00) | -0.28 (1.00) |
| ind10 | BL views lowvol | -0.01 (0.57) | 0.03 (0.37) | 0.09 (0.11) | 2011 | 0.01 (0.40) | 0.37 (0.00) |
| ind10 | BL views lowbeta | 0.02 (0.31) | 0.00 (0.49) | 0.08 (0.19) | 1972 | -0.02 (0.66) | 0.09 (0.10) |
| ind10 | BL views combo_eq | -0.09 (0.98) | -0.32 (1.00) | 0.00 (0.48) |  |  |  |
| ind10 | BL views combo_iv | -0.08 (0.97) | -0.32 (1.00) | 0.01 (0.43) |  |  |  |
| ind49 | min-var LW-CC L504 |  | 0.13 (0.22) | 0.16 (0.11) |  |  |  |
| ind49 | top tercile mom12_1 |  | 0.11 (0.07) | 0.11 (0.09) | 1993 | 0.11 (0.07) | 0.11 (0.10) |
| ind49 | top tercile tsmom |  | 0.02 (0.35) | 0.05 (0.24) | 2012 | 0.02 (0.35) | 0.49 (0.00) |
| ind49 | top tercile strev |  | -0.40 (1.00) | -0.20 (1.00) | 1990 | -0.41 (1.00) | -0.20 (1.00) |
| ind49 | top tercile lowvol |  | 0.03 (0.34) | 0.13 (0.02) | 2011 | 0.03 (0.28) | 0.53 (0.00) |
| ind49 | top tercile lowbeta |  | 0.09 (0.16) | 0.09 (0.15) | 1972 |  | 0.09 (0.09) |
| ind49 | top tercile combo_eq |  | -0.06 (0.81) | -0.02 (0.62) |  |  |  |
| ind49 | top tercile combo_iv |  | -0.02 (0.65) | -0.02 (0.63) |  |  |  |

