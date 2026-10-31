# Real index data (fetched at run time, never committed)

This directory holds the record of real third-party data used by the
research layer. It does not hold the data itself.

| File | Committed | What it is |
|---|---|---|
| `manifest.json` | yes | Every download: URL, UTC timestamp, SHA-256 of the zip and of the parsed arrays, date range, rows, columns, one entry per vintage |
| `registry.json` | yes | Registered studies and their full configuration lists (the trial count) |
| `raw/<dataset>/<sha256>.zip` | **no** | The downloaded zips, cached by hash. Git-ignored by the root `.gitignore` line `data/real/raw/` and by a self-ignoring `raw/.gitignore` that `fetch_data.py` writes |

Fetch or refresh with `python scripts/fetch_data.py` from the repository
root. Load with `portopt.research.data.load_returns`.

## Source and terms of use

The data comes from the Kenneth R. French Data Library
(<https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html>),
read on 2026-10-31.

What the site says (evidence class: read directly from the library page
and from the CSV files, 2026-10-31):

- The page carries a copyright notice in the names of Eugene F. Fama and
  Kenneth R. French, and every CSV ends with the same notice
  (`Copyright 2026 Eugene F. Fama and Kenneth R. French`).
- The page states no licence, no grant of redistribution rights and no
  other terms of use. We found no wording that permits republishing the
  files.
- The history is rebuilt every month: "We reconstruct the full history of
  returns each month" (library page, notes on the research portfolios).
  CRSP revisions can change past returns.
- Missing values are coded -99.99 or -999.

What we do about it (evidence class: our policy, owner decision
2026-10-31): with no explicit redistribution grant, the repository never
contains the data or anything derived from it at full resolution. Code
downloads the files at run time from the library itself; the repository
commits only hashes, date ranges and counts. Published research output
(monthly forward-log returns, summary statistics) is our own computation,
and every page that shows it credits the library as the source. Anyone
reusing the data should cite Fama and French and read the library page
for themselves.

## Datasets

All three are daily, in percent at the source, converted here to decimal
simple returns (`percent / 100`), missing codes to NaN.

| Name | File | Section read | Columns |
|---|---|---|---|
| `ff3_daily` | `F-F_Research_Data_Factors_daily_CSV.zip` | the single section | Mkt-RF, SMB, HML, RF |
| `ind10_vw_daily` | `10_Industry_Portfolios_daily_CSV.zip` | Average Value Weighted Returns -- Daily | 10 industries |
| `ind49_vw_daily` | `49_Industry_Portfolios_daily_CSV.zip` | Average Value Weighted Returns -- Daily | 49 industries |

Vintage 1 of each (downloaded 2026-10-31, built from the 202609 CRSP
database; figures from `manifest.json`):

| Dataset | First date | Last date | Rows | Missing values | zip SHA-256 (first 16) |
|---|---|---|---:|---:|---|
| `ff3_daily` | 1926-07-01 | 2026-09-30 | 26,338 | 0 | `96c23b88e651cbff` |
| `ind10_vw_daily` | 1926-07-01 | 2026-09-30 | 26,338 | 0 | `4c79bad80b6f4eee` |
| `ind49_vw_daily` | 1926-07-01 | 2026-09-30 | 26,338 | 67,250 | `b9d36d571e968d71` |

The 49-industry file has missing values up to 1971-03-11 (some industries
did not exist yet); every row from 1971-03-12 on is complete.

## Vintages

Because the library rebuilds its history monthly, a download almost
always has a new hash, even for dates already seen.

- A zip whose SHA-256 is not yet in the manifest is appended as a new
  vintage (`vintage_id` 1, 2, ...). Existing vintage records are never
  edited, except that a re-download of an identical file refreshes that
  vintage's `last_seen_utc`.
- The dataset-level fields (`current_vintage`, `zip_sha256`,
  `first_date`, ...) describe the vintage seen most recently.
- `load_returns(..., vintage=...)` takes a vintage id, a SHA-256 or a
  prefix of at least 12 hex digits; the default is the current vintage.
  It re-hashes the cached zip and the parsed arrays and raises
  `VintageMismatchError` if either differs from the manifest.
- The library serves only its current file, so an older vintage can be
  loaded only from a local cache that still holds it. A study that must
  be reproducible should record the vintage it used (the forward ledger
  does this on every row).

## The holdout

`holdout_start` is **2016-01-01**, frozen on 2026-10-31 before any study in
this repository looked at returns after that date. The same value is a
constant in `research/data.py`; loading fails if the manifest and the
code disagree, so the boundary cannot be moved by editing one file.

- Development: 1926-07-01 to 2015-12-31, 23,637 trading days.
- Holdout: 2016-01-04 to 2026-09-30 in vintage 1, 2,701 trading days,
  about 10.7 years, and growing by one month per vintage.

Why this date:

1. Length. Ten years is about the shortest span in which a Sharpe ratio
   says anything. The standard error of an annualized Sharpe estimate is
   roughly `1 / sqrt(years)` under i.i.d. returns (Lo 2002), so 10.7 years
   gives about 0.31: two strategies need to differ by about 0.6 in Sharpe
   before a 2-sigma test notices. A shorter holdout would not separate the
   rules this repository compares; a longer one would take decades of
   development history away from the estimators.
2. Regimes. The holdout contains the 2018 volatility shock, the 2020
   crash and recovery, and the 2022 joint fall of stocks and bonds, so it
   is not one calm market.
3. A round boundary, chosen without looking at results, is easier to
   audit than a date picked to suit a backtest.

What the holdout does not protect against (evidence class: reasoning,
not measured): the construction rules studied here (equal weight, inverse
volatility, risk parity, minimum variance) were all published before 2016,
and the people writing the code have lived through 2016-2026 markets. The
holdout is clean with respect to this repository's own tuning, which the
registry makes countable; it is not clean with respect to the wider
literature.

`load_returns(dataset, "holdout" | "all", study_id=...)` refuses unless
`study_id` names a study in `registry.json`. Register a study with
`portopt.research.walkforward.register_study(id, description, configs)`,
listing every configuration it will try; `n_trials()` then gives the
honest trial count for the deflated Sharpe ratio.
