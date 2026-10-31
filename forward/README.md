# Forward paper log

A pre-registered research log. Rules chosen in a study are frozen, and
from then on the same code applies them, month by month, to data the
study never saw. Every month is published, whatever the result.

**Hypothetical, not investment advice, no real money.** The series are
research index portfolios from the Kenneth R. French Data Library (US
industry portfolios and the Fama/French market factor). They cannot be
bought: they are not ETFs, funds or securities. Nothing here is a
recommendation to buy or sell anything, and no order is ever placed.

## Why it exists

A backtest, however careful, is run on data the researcher could look at.
A forward log cannot be tuned after the fact: the rule is fixed first,
the data arrives later, and the ledger is append-only. It is the cheapest
honest test of whether a rule that looked good in development keeps
looking good.

## Files

| Path | What it is |
|---|---|
| `rules/<id>.json` | One frozen rule per file (written by a study, never edited) |
| `run_forward.py` | The runner: fetch, simulate, append, summarize |
| `ledger.csv` | Append-only record, one row per rule per month, plus restatements |
| `results.md` | Summary regenerated from the ledger on every run |

The workflow `.github/workflows/forward.yml` runs on the 8th of each month
at 06:00 UTC (and on demand). It fetches the data, runs the rules and
commits only `ledger.csv`, `results.md` and `data/real/manifest.json` if
they changed. With no rule files it stops with a notice.

## Rule schema (version 1)

```json
{
  "schema_version": 1,
  "id": "fwd-001-erc-ind10",
  "study_id": "s-2026-11-edge",
  "frozen_at": "2026-11-01",
  "git_tag": "forward/fwd-001",
  "strategy": {"name": "erc", "params": {"shrink": true}},
  "dataset": "ind10_vw_daily",
  "rf_dataset": "ff3_daily",
  "lookback_days": 504,
  "cost_bps": 10,
  "band": 0.05,
  "rebalance": "monthly",
  "benchmark": {"kind": "strategy", "name": "equal_weight", "params": {}},
  "description": "optional free text"
}
```

| Field | Rule |
|---|---|
| `schema_version` | 1 |
| `id` | `^[a-z0-9][a-z0-9._-]{2,63}$`; the file must be named `<id>.json` |
| `study_id` | A study registered in `data/real/registry.json` (the runner reads post-2016 data, which needs one) |
| `frozen_at` | `YYYY-MM-DD`. The first logged month is the first calendar month starting on or after this date |
| `git_tag` | The tag that pins the code the rule was frozen with |
| `strategy` | `name` one of `equal_weight`, `inverse_vol`, `erc`, `min_variance_longonly`, `max_sharpe_longonly`, `tangency_capped` (from `portopt.research.strategies`); `params` are that factory's keyword arguments, checked at load time |
| `dataset` | `ind10_vw_daily` or `ind49_vw_daily` |
| `rf_dataset` | Optional, `ff3_daily` (the only one with an RF column) |
| `lookback_days` | Integer >= 21: the estimation window, trailing trading days before each rebalance |
| `cost_bps` | >= 0, charged per unit of traded risky notional |
| `band` | >= 0, no-trade band on the L1 distance to target (not applied to the first build) |
| `rebalance` | `monthly`, `quarterly` or `annual`, counted from the first logged month |
| `benchmark` | `{"kind": "market"}` (Mkt-RF + RF, no costs) or `{"kind": "strategy", "name": ..., "params": {...}}` run with the rule's own cost, band, lookback and schedule |
| `description` | Optional string |

Unknown keys are an error. The runner stores a hash of the rule file in
every ledger row and refuses to run if the file later changes: a changed
rule is a new rule with a new id.

## How results are computed

For each rule the runner loads the current data vintage, aligns the
industry returns with the daily T-bill rate `RF`, and simulates from the
first trading day of the first logged month with the accounting of
`portopt.research.costs` (see that module's docstring), via
`portopt.research.walkforward.run_schedule_net`:

- The portfolio starts in cash and buys its first target at the start of
  the first month, paying `cost_bps` on that build.
- At each rebalance (first trading day of the month, or of every third or
  twelfth month) the strategy sees only the `lookback_days` trading days
  before that day. Between rebalances weights drift with returns.
- Uninvested cash earns the daily `RF`.
- The monthly return is the compounded daily net return over that
  month's trading days. Turnover is the traded notional at that month's
  rebalance (1.0 for the initial build, 0 in a month without one).

Worked example: a rule frozen on 2026-11-15 logs its first month as
2026-12. The library publishes December data in early February, so the
run on 8 February 2027 appends the 2026-12 row.

## Ledger

`ledger.csv` columns:

| Column | Meaning |
|---|---|
| `month` | `YYYY-MM` |
| `rule_id` | The rule |
| `row_type` | `original` or `restatement` |
| `net_return`, `gross_return` | Monthly simple return after and before trading cost |
| `benchmark_return` | Same month, benchmark |
| `cumulative_net`, `cumulative_benchmark` | Compounded from the first month, as of this row's recording |
| `turnover`, `cost` | Traded notional and total cost in the month (fractions of NAV) |
| `trading_days` | Days in the month |
| `data_vintage` | 16-hex hash of the zip SHA-256s of the datasets used (see `data/real/manifest.json`) |
| `rule_sha256` | 16-hex hash of the rule file |
| `recorded_at_utc` | When the row was appended |

Numbers are written with 10 decimals. The runner never edits or deletes a
row. On each run, for each rule:

- a complete month not yet in the ledger is appended as `original`;
- a month already there with the same numbers (within 5e-9) is left
  alone, so running twice on the same data appends nothing;
- a month whose numbers changed under a **new** data vintage gets a
  `restatement` row, and the original stays;
- a month whose numbers changed under the **same** vintage stops the run
  with an error: something other than the data changed, and history is
  not rewritten.

`results.md` shows, for each month, the latest row, and marks restated
months.

## Data lag and vintages

The French library updates monthly, usually in the first week, with data
through the previous month or the one before, so the log runs one to two
months behind the calendar. A month is logged only once it is complete:
the CRSP database month in the file header (for example `202609`) is the
last complete month.

The library rebuilds its whole history at each update, and CRSP revises
old records, so past months can move slightly. Every download is a new
vintage in `data/real/manifest.json`; each ledger row names the vintage it
came from, and a changed past month shows up as a restatement row rather
than a silent edit. Expect most restatements to be tiny.

## Running it by hand

```sh
python -m pip install -e python
python scripts/fetch_data.py
python forward/run_forward.py --dry-run   # print what would be appended
python forward/run_forward.py             # append and regenerate results.md
```
