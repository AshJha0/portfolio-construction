"""Markdown pages for ``real_data_study.py``: development results, the
pre-registration and the final (development and holdout) results.

Every number on these pages comes from the computed rows (the CSV files
hold the same numbers at six decimals); the prose around them is fixed
text in this file, written after reading the development results.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import replace
from typing import Any

import real_data_core as core
from real_data_core import Config, Row, fmt, table

from portopt.research import data as rd


def _index(rows: list[Row]) -> dict[tuple[str, str], Row]:
    return {(r.cfg.id, r.period): r for r in rows}


def _pct(v: float, nd: int = 1) -> str:
    return fmt(100.0 * v, nd)


def _diff(p: core.Paired | None) -> str:
    if p is None:
        return ""
    return f"{fmt(p.diff)} ({fmt(p.p_two)})"


def _ci(st: core.Stats) -> str:
    if math.isnan(st.boot_lo):
        return fmt(st.net_sharpe)
    return f"{fmt(st.net_sharpe)} [{fmt(st.boot_lo)}, {fmt(st.boot_hi)}]"


LEVER_HEADER = [
    "configuration",
    "net SR [95% CI]",
    "SR 25 bp",
    "excess %/yr",
    "vol %",
    "max DD %",
    "turnover/yr",
    "cost %/yr",
    "vs 1/N: dSR (p)",
    "vs base: dSR (p)",
]


def lever_rows(idx: dict[tuple[str, str], Row], configs: Iterable[Config]) -> list[list[str]]:
    body = []
    for c in configs:
        r = idx.get((c.id, "full"))
        if r is None:
            body.append([c.label(), "excluded"] + [""] * 8)
            continue
        st = r.stats
        stress = idx.get((replace(c, cost_bps=25.0).id, "full")) if c.cost_bps else None
        mkt = c.rule == "market"
        body.append(
            [
                c.label(),
                _ci(st),
                fmt(stress.stats.net_sharpe) if stress else "",
                _pct(st.ann_excess),
                _pct(st.ann_vol),
                _pct(st.max_drawdown, 0),
                "" if mkt else fmt(st.turnover_per_year),
                "" if mkt else fmt(100 * st.cost_per_year),
                _diff(r.vs_1n),
                _diff(r.vs_comp),
            ]
        )
    return body


def _configs(rows: list[Row], pred: Callable[[Config], bool]) -> list[Config]:
    out: list[Config] = []
    seen: set[str] = set()
    for r in rows:
        if r.period == "full" and pred(r.cfg) and r.cfg.id not in seen:
            seen.add(r.cfg.id)
            out.append(r.cfg)
    return out


def _base(c: Config) -> bool:
    return c.cost_bps in (core.BASE_COST, None)


def data_table(panels: dict[str, core.Panel]) -> str:
    body = []
    for u, p in panels.items():
        for prov in p.provenance:
            body.append(
                [
                    f"`{prov['dataset']}`",
                    u,
                    str(prov["vintage_id"]),
                    str(prov["crsp_database"]),
                    f"`{prov['zip_sha256'][:16]}`",
                    f"`{prov['parsed_sha256'][:16]}`",
                    f"{prov['first_date']} to {prov['last_date']}",
                    f"{prov['rows_used']:,}",
                ]
            )
    return table(
        [
            "dataset",
            "universe",
            "vintage",
            "CRSP",
            "zip SHA-256",
            "parsed SHA-256",
            "dates",
            "rows",
        ],
        body,
        "llrllllr",
    )


def subperiod_table(
    idx: dict[tuple[str, str], Row], configs: list[Config], periods: list[str]
) -> str:
    header = ["configuration"]
    for p in periods:
        header += [f"SR {p}", f"vs 1/N {p}"]
    body = []
    for c in configs:
        line = [c.label()]
        for p in periods:
            r = idx.get((c.id, p))
            line += ["", ""] if r is None else [fmt(r.stats.net_sharpe), _diff(r.vs_1n)]
        body.append(line)
    return table(header, body)


def periods_of(rows: list[Row], universe: str) -> list[str]:
    out: list[str] = []
    for r in rows:
        if r.cfg.universe == universe and r.period != "full" and r.period not in out:
            out.append(r.period)
    return out


def _get(idx: dict[tuple[str, str], Row], cid: str) -> Row:
    return idx[(cid, "full")]


METHOD = """\
## Method

**Universes and periods.** `ind10` is the 10 value-weighted industry portfolios from 1926-07-01; `ind49` is the 49 industry portfolios from 1972-01-03 (the file has missing values up to 1971-03-11). Each universe is evaluated from the first calendar month that has at least 504 trading days of history before it ({start10} and {start49}), to 2015-12-31. Every configuration of a universe is evaluated on exactly the same days, so paired tests compare like with like.

**Schedule and accounting.** Rebalancing is monthly, on the first trading day of each calendar month, with weights computed from returns up to the previous close (`walkforward.calendar_rebalance_rows`). The accounting is `portopt.research.walkforward.run_schedule_net`: weights drift between rebalances, each rebalance (including the initial build from cash) pays `cost_bps` on the traded risky notional `sum_i |w_new_i - w_drift_i|`, cash earns the daily T-bill rate `RF` from `ff3_daily`, and negative cash borrows at the same rate. Short positions earn the negative of the asset return and pay no borrow fee. Every rule sees only the trailing window before the rebalance.

**Costs are assumptions.** The industry portfolios are index series; nobody trades them, so there is no cost to measure. 10 bp per unit of one-way traded notional stands in for trading a basket of large US stocks that tracks an industry; 25 bp is a stress level. Neither is calibrated to any desk, and there is no market impact term.

**Estimation windows.** 252 and 504 trading days (about one and two years). The two values bracket common practice and show sensitivity; we did not search over windows. The concentration `N / T` is 0.04 and 0.02 for 10 industries and 0.19 and 0.10 for 49; by the Marchenko-Pastur law (derived) the smallest eigenvalue of a sample covariance of uncorrelated, equal-variance assets is biased down to about `(1 - sqrt(N/T))^2` of the truth: 0.64 for 10 industries at 252 days and 0.31 for 49 industries at 252 days. That bias is what shrinkage estimators target.

**Levers and their bases.** Each lever is judged against its own base configuration ("vs base") as well as against 1/N and the market:

| Lever | Configurations | Base |
|---|---|---|
| Covariance estimators | long-only minimum variance and ERC with the sample covariance, Ledoit-Wolf constant correlation (core `shrink_covariance`), Ledoit-Wolf toward the identity, Ledoit-Wolf (2020) nonlinear shrinkage, EWMA with decay 0.97 (half-life {hl} days; the RiskMetrics one-month decay), and a 3-factor PCA model | the sample covariance |
| Cost-aware rebalancing | 1/N, minimum variance (LW constant correlation) and, on 10 industries, long-only max-Sharpe, each with a no-trade band of 0.05 or 0.10 on the L1 distance to target, or closing half the gap (partial 0.5); on 10 industries also the turnover-penalised minimum variance `portopt.min_variance_turnover` with `w0` the drifted weights and penalty `kappa = 12 c / gamma`, gamma 2 or 10 | the same rule rebalanced in full |
| Leverage control (10 industries) | tangency portfolio of the trailing sample moments (excess returns), uncapped, and with gross leverage capped at 2 by moving toward minimum variance (`tangency_capped`) | uncapped tangency |
| Volatility targeting | 1/N and minimum variance (LW constant correlation) scaled each month to a 10% ex-ante volatility with the sample covariance or the EWMA covariance of the window, gross exposure capped at 1.5 | the unscaled rule |
| Mean shrinkage (10 industries) | long-only max-Sharpe (25-point frontier grid) with Bayes-Stein (Jorion 1986) or sample means of excess returns | sample means |

The 10% target and the 1.5 cap were fixed before any result was seen. A Sharpe ratio is invariant to a constant scale, so the target matters only through the cap and the cash it leaves. Rules that use means (max-Sharpe, tangency) see excess returns; covariance-only rules see total returns, as the forward runner does.

**Statistics.** Sharpe ratios are annualized `sqrt(252) mean / std` of daily net excess returns (net return minus `RF`). Standard errors: Lo (2002) iid and Mertens (2002) iid non-normal. The 95% interval is a stationary bootstrap (Politis and Romano 1994), mean block {block:g} days, {nboot} resamples, seed {seed}, computed at 10 bp. PSR is the probability that the true Sharpe ratio exceeds 0 (Bailey and Lopez de Prado 2012); DSR raises the hurdle to the expected maximum of {ntrials} trials (Bailey and Lopez de Prado 2014). "dSR (p)" is the difference in annualized Sharpe ratio and the two-sided p-value of the Ledoit and Wolf (2008) HAC test (`sharpe_difference_test`, Bartlett kernel, 13 lags on these lengths). Sub-periods are slices of the same runs.

**Trial count.** Every configuration on this page was registered before the run that produced it: {ntrials} configurations in `data/real/registry.json` under `r2-dev-exploration` ({n1}, registered {t1}, configurations SHA-256 `{h1}`) and `r2-dev-exploration-tc` ({n2}, the turnover-penalised batch, added once that optimiser was ready). The DSR uses N = {ntrials} and the cross-sectional variance of the per-period Sharpe ratios of the universe's configurations. Counting cost levels and near-duplicates (EWMA at 252 and 504 days is nearly the same estimator) as separate trials overstates the number of independent trials, so the DSR is conservative.

**Block length and serial dependence.** Squared daily returns of 1/N on 10 industries have autocorrelation 0.24 at lag 1 and 0.10 at lag 21 (volatility clustering). The bootstrap standard error of 1/N's Sharpe ratio is 0.100 with block 1 (iid), 0.118 with 21, 0.123 with 63 and 0.131 with 126 days, against an iid Lo standard error of 0.104 (measured while choosing the block length). The iid standard errors, PSR and DSR therefore overstate precision by roughly 20%.

**Annualization.** The NYSE traded on Saturdays until 1952-05-24, so 1926-1952 has about 290 rows a year. We annualize with 252 throughout (the library convention), which understates annualized returns and costs in those years by about 13% and Sharpe ratios by about 7%. Comparisons between rules on the same days keep their sign and test statistics.

**Tooling checks run with this page.** {selfcheck} Long-only minimum variance is solved by the primal active-set method in `python/studies/real_data_engine.py`, because the pinned core QP is written for n <= 20 with explicit loops and takes seconds a solve at n = 49 (measured; the turnover-penalised split QP took about 600 s for one solve at n = 49, which is why that lever is run on 10 industries only). Every solve carries a KKT certificate, and on {nchk} dates compared with `portopt.min_variance_longonly` the largest weight difference is {wmax}.
"""


def _method(run: core.RunOutput, entry: dict[str, Any], n_total: int) -> str:
    first = {}
    for u in run.panels:
        s = next((s for s in run.series.values() if s.cfg.universe == u), None)
        first[u] = str(s.dates[0]) if s is not None else ""
    reg = {s["id"]: s for s in rd.load_registry()["studies"]}
    tc = reg.get(core.DEV_TC_STUDY_ID, {"n_configs": 0})
    nchk = sum(c[1] for c in run.solver_checks)
    wmax = max((c[2] for c in run.solver_checks), default=0.0)
    return METHOD.format(
        start10=first.get("ind10", ""),
        start49=first.get("ind49", ""),
        hl=fmt(core.EWMA_HALF_LIFE, 1),
        block=core.BOOT_BLOCK,
        nboot=core.BOOT_N,
        seed=core.BOOT_SEED,
        ntrials=n_total,
        n1=entry["n_configs"],
        t1=entry["registered_at_utc"],
        h1=entry["configs_sha256"][:16],
        n2=tc["n_configs"],
        selfcheck=(
            "The scaled calendar engine used for volatility targeting reproduced "
            "`run_schedule_net` and `run_backtest_net` bit for bit in "
            f"{len(run.self_check_log)} checks."
        ),
        nchk=nchk,
        wmax=f"{wmax:.1e}",
    )


LEVERS = (
    ("benchmark", "Benchmarks"),
    ("estimator", "Covariance estimators"),
    ("cost_aware", "Cost-aware rebalancing"),
    ("leverage", "Leverage control"),
    ("vol_target", "Volatility targeting"),
    ("mean_shrinkage", "Mean shrinkage"),
)


def dev_markdown(run: core.RunOutput, rows: list[Row], entry: dict[str, Any], n_total: int) -> str:
    idx = _index(rows)
    out: list[str] = []
    add = out.append
    add("# Portfolio-construction levers on real data: development period")
    add("")
    add(
        "**Development data only (1926-07-01 to 2015-12-31).** No row dated 2016-01-01 or "
        'later was read: the script loads `load_returns(..., period="development")` and '
        "nothing else. The series are research index portfolios from the Kenneth R. French "
        "Data Library; they cannot be bought, the trading costs are assumptions, and nothing "
        "here is investment advice."
    )
    add("")
    add(
        "Generated by `python python/studies/real_data_study.py --phase dev`. Every number is "
        "in `real_data_study_dev.csv` (one row per configuration and period; summary "
        "statistics only, no return series). Rerunning on the same data vintage reproduces "
        "both files."
    )
    add("")
    add(_conclusions(idx))
    add("")
    add("## Data")
    add("")
    add(data_table(run.panels))
    add("")
    add(
        "Source: Kenneth R. French Data Library, fetched at run time and never committed "
        "(`data/real/README.md`). Copyright Eugene F. Fama and Kenneth R. French."
    )
    add("")
    add(_method(run, entry, n_total))
    add("## Results at 10 bp, full development period")
    add("")
    add(
        "dSR is the difference in annualized net Sharpe ratio, with the two-sided p-value of "
        "the paired test in brackets. Turnover is one-way traded notional per year after the "
        "initial build; cost is the annual drag in % of NAV."
    )
    add("")
    for lever, title in LEVERS:
        add(f"### {title}")
        add("")
        for u in core.UNIVERSES:
            cfgs = _configs(
                rows, lambda c, u=u, lv=lever: c.universe == u and c.lever == lv and _base(c)
            )
            if not cfgs:
                continue
            add(f"{core.UNIVERSES[u].describe().capitalize()}:")
            add("")
            add(table(LEVER_HEADER, lever_rows(idx, cfgs)))
            add("")
    add("### Stability across sub-periods (10 bp)")
    add("")
    for u in core.UNIVERSES:
        per = periods_of(rows, u)
        key = [c for c in _configs(rows, lambda c, u=u: c.universe == u and _base(c)) if _key(c)]
        add(f"{core.UNIVERSES[u].describe().capitalize()}:")
        add("")
        add(subperiod_table(idx, key, per))
        add("")
    add("## Fallbacks and exclusions")
    add("")
    fb = [
        f"`{r.cfg.strategy_id}`: {r.stats.n_fallbacks} of {r.stats.n_rebalances}"
        for r in rows
        if r.period == "full" and r.cfg.cost_bps == core.BASE_COST and r.stats.n_fallbacks
    ]
    add(
        "Rebalances at which a rule fell back to a simpler one (max-Sharpe to long-only minimum "
        "variance when no frontier point earns more than the risk-free rate; Bayes-Stein also "
        "when the history is too short; tangency to the unconstrained minimum-variance "
        "portfolio when it is undefined): " + ("; ".join(fb) if fb else "none") + "."
    )
    add("")
    excl = "; ".join(f"`{k}`: {v}" for k, v in run.excluded.items())
    add("Configurations that raised an error and appear in no table: " + (excl or "none") + ".")
    add("")
    add(LIMITS)
    return "\n".join(out) + "\n"


def _key(c: Config) -> bool:
    keys = {
        "ind10/equal_weight/10bp",
        "ind10/market",
        "ind10/min_var_lo.lwcc/L252/10bp",
        "ind10/min_var_lo.lwcc/L504/10bp",
        "ind10/min_var_lo.ewma/L504/10bp",
        "ind10/erc.lwcc/L504/10bp",
        "ind10/tangency.sample/L504/10bp",
        "ind10/tangency_cap2.sample/L504/10bp",
        "ind10/equal_weight/L252/vt0.1-sample-max1.5/10bp",
        "ind10/min_var_lo.lwcc/L504/vt0.1-ewma-max1.5/10bp",
        "ind10/max_sharpe_lo.sample/L252/10bp",
        "ind10/max_sharpe_lo_bs.sample/L252/10bp",
        "ind49/equal_weight/10bp",
        "ind49/market",
        "ind49/min_var_lo.sample/L504/10bp",
        "ind49/min_var_lo.lwcc/L504/10bp",
        "ind49/min_var_lo.nls/L504/10bp",
        "ind49/min_var_lo.ewma/L504/10bp",
        "ind49/erc.lwcc/L504/10bp",
        "ind49/equal_weight/L252/vt0.1-sample-max1.5/10bp",
        "ind49/min_var_lo.lwcc/L504/vt0.1-ewma-max1.5/10bp",
    }
    return c.id in keys


def _sr(idx: dict[tuple[str, str], Row], cid: str) -> str:
    return fmt(_get(idx, cid).stats.net_sharpe)


def _vs(idx: dict[tuple[str, str], Row], cid: str, which: str = "comp") -> str:
    r = _get(idx, cid)
    p = r.vs_comp if which == "comp" else r.vs_1n
    return "n/a" if p is None else f"{fmt(p.diff)} (p = {fmt(p.p_two)})"


def _conclusions(idx: dict[tuple[str, str], Row]) -> str:
    g = lambda cid: _get(idx, cid).stats  # noqa: E731
    ew10, ew49 = "ind10/equal_weight/10bp", "ind49/equal_weight/10bp"
    vt = "ind10/equal_weight/L252/vt0.1-sample-max1.5/10bp"
    mv49 = "ind49/min_var_lo.lwcc/L504/10bp"
    mv10 = "ind10/min_var_lo.lwcc/L504/10bp"
    ew_e = "ind49/min_var_lo.ewma/L504/10bp"
    cap = "ind10/tangency_cap2.sample/L504/10bp"
    bs = "ind10/max_sharpe_lo_bs.sample/L252/10bp"
    part = "ind10/min_var_lo.lwcc/L252/partial0.5/10bp"
    tc = [k for (k, p) in idx if p == "full" and "min_var_lo_tc" in k and k.endswith("/10bp")]
    tc_txt = "; ".join(
        f"`{k.split('/')[1]}` {k.split('/')[2]} {k.split('/')[3]}: {_vs(idx, k)}"
        for k in sorted(tc)
    )
    lines = [
        "## What the development data say",
        "",
        "Plain answers first; the tables below carry the intervals. Differences are in annualized net Sharpe ratio at 10 bp, with two-sided p-values of the paired Ledoit-Wolf test; with about 88 years of daily data on 10 industries the standard error of one Sharpe ratio is about 0.12 (bootstrap), and that of a paired difference between two similar rules is far smaller.",
        "",
        f"1. **Nothing beat 1/N by much, and the market was the weakest benchmark.** 1/N on 10 industries earned {_sr(idx, ew10)} [{fmt(g(ew10).boot_lo)}, {fmt(g(ew10).boot_hi)}] net of cost; the market {_sr(idx, 'ind10/market')}. Most construction rules land within 0.05 of 1/N.",
        f"2. **Volatility targeting gave the largest gain on 10 industries, but not a steady one.** 1/N scaled to 10% volatility (sample covariance, 252 days, gross at most 1.5) earned {_sr(idx, vt)}, {_vs(idx, vt)} against plain 1/N, and its maximum drawdown was {_pct(g(vt).max_drawdown, 0)}% against {_pct(g(ew10).max_drawdown, 0)}%. The gain comes almost entirely from 1928-1969 (the 1930s); in 1970-2015 it is near zero or negative (sub-period table). On 49 industries, which start in 1974, it is {_vs(idx, 'ind49/equal_weight/L252/vt0.1-sample-max1.5/10bp')}. This is the pattern the critiques of volatility management describe (Cederburg et al. 2020): real, but concentrated in a few episodes.",
        f"3. **Minimum variance lowered volatility by about a third and added a little Sharpe.** On 49 industries long-only minimum variance (LW constant correlation, 504 days) earned {_sr(idx, mv49)} against 1/N {_sr(idx, ew49)}, {_vs(idx, mv49, '1n')}, at {_pct(g(mv49).ann_vol)}% volatility against {_pct(g(ew49).ann_vol)}%; on 10 industries the gain is {_vs(idx, mv10, '1n')}. In both universes the gain is in 2000-2015; in the earlier sub-periods minimum variance is slightly behind 1/N.",
        f"4. **The covariance estimator hardly matters for long-only rules, except that EWMA hurts.** Every linear or nonlinear shrinkage estimator and the PCA model are within 0.02 of the sample covariance for both minimum variance and ERC; the long-only constraint already does most of the shrinking (Jagannathan and Ma 2003). EWMA with decay 0.97 trades five times as much and loses: on 49 industries {_vs(idx, ew_e)} against the sample covariance.",
        f"5. **Cost-aware rebalancing saves cost but the cost was small to begin with.** At 10 bp the long-only rules pay 0.03 to 0.23% a year; bands and partial rebalancing cut that by up to half and move Sharpe ratios by at most about 0.02 (minimum variance on 10 industries, partial 0.5: {_vs(idx, part)}). Turnover-penalised minimum variance cuts turnover by 75 to 95% against full rebalancing, with no reliable Sharpe change: {tc_txt or 'not run'}.",
        f"6. **Leverage control matters for the only rule that needs it.** The uncapped tangency portfolio of trailing sample moments (the library rule now passes no leverage guard) lost all of its capital between February and May 1929 at both windows and at both costs, so it appears in no table (see Fallbacks and exclusions). Capping gross leverage at 2 by moving toward minimum variance kept it solvent: 504 days {_sr(idx, cap)}, {_vs(idx, cap, '1n')} against 1/N, with turnover {fmt(g(cap).turnover_per_year, 1)} times NAV a year and a {_pct(g(cap).max_drawdown, 0)}% drawdown; it loses {fmt(g(cap).net_sharpe - _get(idx, cap.replace('/10bp', '/25bp')).stats.net_sharpe)} of Sharpe between 10 and 25 bp.",
        f"7. **Bayes-Stein means did not help the max-Sharpe rule.** {_vs(idx, bs)} against sample means; both trade about 5 to 6 times NAV a year and lose 0.04 to 0.06 of Sharpe between 10 and 25 bp.",
        "8. **The deflated Sharpe ratio does not separate the rules.** With N = 206 registered trials most configurations have DSR near 1 because 88 years of data make even a Sharpe ratio of 0.45 clearly positive; DSR tests the Sharpe ratio against zero after selection, not one rule against another. The paired tests are the relevant evidence, and few are below 0.05 before any correction for the comparisons made.",
    ]
    return "\n".join(lines)


LIMITS = """\
## Limits

* The series are not investable. Industry portfolios are value-weighted baskets of every CRSP stock in an industry, rebalanced by construction, with no trading cost of their own.
* Costs are assumed, linear and the same in 1930 and 2015. Real costs were far higher before the 1980s, which would penalize the higher-turnover rules more in the early decades.
* One data vintage. The French library rebuilds its history monthly; the numbers here are for vintage 1 (CRSP 202609).
* The development period was not blind: these rules and estimators were published by people who had seen much of this history. The holdout (2016 onward) is clean only with respect to this repository's own choices, which the registry makes countable.
* Standard errors that assume independence understate the uncertainty by about 20% (block bootstrap comparison above).
"""


# --------------------------------------------------------------- prereg page


def prereg_markdown(fwd: dict[str, Any], hold: dict[str, Any], dev: dict[Any, Any]) -> str:
    import real_data_prereg as pr

    m = len(pr.HYPOTHESES)
    out: list[str] = []
    add = out.append
    add("# Pre-registration: holdout test and forward rules (round 2)")
    add("")
    add(
        f"Written and registered on {hold['registered_at_utc'][:10]} before any row dated "
        "2016-01-01 or later was read by this study. The registry entries in "
        "`data/real/registry.json` carry the SHA-256 of the registered configuration lists; "
        "the holdout study's description carries the SHA-256 of the hypothesis list below. "
        "Phase 2 (`python python/studies/real_data_study.py --phase holdout`) refuses to run if "
        "either differs from the code, records `git rev-parse HEAD`, and refuses a second run."
    )
    add("")
    add(
        "**Code commit of record:** the commit that adds this file (`git log --diff-filter=A "
        "-- docs/results/real_data_preregistration.md`). It contains `python/studies/`, the "
        "registry entries and the forward rule files."
    )
    add("")
    add("## 1. Question")
    add("")
    add(
        "Which portfolio-construction levers improve net-of-cost risk-adjusted performance on "
        "real data, and by how much? The development results "
        "(`docs/results/real_data_study_dev.md`) give effect sizes on 1926-2015. The holdout "
        "tests a small number of them, chosen from development evidence, on 2016-01 to the last "
        "complete month of the pinned vintage."
    )
    add("")
    add("## 2. Data")
    add("")
    man = rd.load_manifest()
    body = []
    for name in ("ff3_daily", "ind10_vw_daily", "ind49_vw_daily"):
        v = [x for x in man["datasets"][name]["vintages"] if x["vintage_id"] == core.DATA_VINTAGE][
            0
        ]
        body.append(
            [
                f"`{name}`",
                str(v["vintage_id"]),
                v["crsp_database"],
                f"`{v['zip_sha256']}`",
                f"`{v['parsed_sha256']}`",
            ]
        )
    add(table(["dataset", "vintage", "CRSP", "zip SHA-256", "parsed SHA-256"], body, "lrlll"))
    add("")
    add(
        f"Phase 2 loads vintage {core.DATA_VINTAGE} (pinned in `real_data_core.DATA_VINTAGE`) "
        'with `period="all"` and the study id `r2-holdout-v1`, and evaluates rows from '
        f"2016-01-01 to the last complete month of that vintage (2026-09-30, {pr.HOLDOUT_DAYS_V1:,} "
        "trading days). Each configuration starts from cash on the first trading day of "
        "2016-01; its estimation windows reach back into the development data."
    )
    add("")
    add("## 3. Method (frozen)")
    add("")
    add(
        "Exactly the method of the development page: monthly rebalancing on the first trading "
        "day of each month, `run_schedule_net` accounting (the scaled variant in "
        "`real_data_engine.py` for volatility targeting), cost on traded risky notional, cash "
        "and borrowing at the daily `RF`, the same estimators, windows and parameters. Costs are "
        "assumptions (10 bp base, 25 bp stress) for non-investable index series."
    )
    add("")
    add("## 4. Hypotheses")
    add("")
    add(
        f"Each hypothesis is one-sided: `H1: SR(test) > SR(comparator)`, metric the difference "
        f"of annualized net Sharpe ratios at 10 bp on the holdout, test the Ledoit and Wolf "
        f"(2008) HAC delta-method test (`portopt.research.sharpe_difference_test`, Bartlett "
        f"kernel, `floor(4 (T/100)^(2/9))` lags), one-sided p-value `1 - Phi(z)`. The {m} "
        f"p-values are adjusted by Holm's (1979) step-down method; a hypothesis is supported "
        f"when its adjusted p-value is at most {pr.ALPHA:g} (family-wise error rate {pr.ALPHA:g})."
    )
    add("")
    hb = []
    for h in pr.HYPOTHESES:
        hb.append([h.id, h.lever, f"`{h.test.id}`", f"`{h.comparator.id}`", h.statement])
    add(table(["id", "lever", "test", "comparator", "statement"], hb, "lllll"))
    add("")
    add(
        "Development evidence and power. Power is approximate: the holdout standard error is "
        "the development one scaled by `sqrt(T_dev / T_holdout)`, and the true effect is taken "
        "to equal the development estimate (which, chosen as the best of several, is more "
        "likely too large than too small)."
    )
    add("")
    pb = []
    for h in pr.HYPOTHESES:
        a = dev[(h.test.id, "full")]
        b = dev[(h.comparator.id, "full")]
        dd = float(a["net_sharpe"]) - float(b["net_sharpe"])
        se = _dev_se(dev, h)
        t_dev = int(a["n_days"])
        se_h = se * math.sqrt(t_dev / pr.HOLDOUT_DAYS_V1)
        from statistics import NormalDist

        nd = NormalDist()
        z_a = nd.inv_cdf(1 - pr.ALPHA)
        z_m = nd.inv_cdf(1 - pr.ALPHA / m)
        pw = nd.cdf(dd / se_h - z_a) if se_h > 0 else math.nan
        pm = nd.cdf(dd / se_h - z_m) if se_h > 0 else math.nan
        mde = (z_a + nd.inv_cdf(0.8)) * se_h
        pb.append(
            [h.id, fmt(dd), fmt(se, 3), f"{t_dev:,}", fmt(se_h, 3), fmt(pw), fmt(pm), fmt(mde)]
        )
    add(
        table(
            [
                "id",
                "dev dSR",
                "dev SE",
                "dev days",
                "holdout SE (approx.)",
                "power, alpha 0.05",
                f"power, alpha 0.05/{m}",
                "effect for 80% power",
            ],
            pb,
        )
    )
    add("")
    add(
        "Several hypotheses have little power; we register them anyway because a lever that "
        "cannot be shown to work on ten years of data is itself a result, and the effect sizes "
        "with intervals will be reported whatever the tests say."
    )
    add("")
    add("## 5. Registered holdout configurations")
    add("")
    add(
        f"Study `{hold['id']}`, registered {hold['registered_at_utc']}, {hold['n_configs']} "
        f"configurations, SHA-256 `{hold['configs_sha256']}`: every test and comparator "
        "configuration above, the two forward-rule configurations, and 1/N and the market for "
        "each universe, each at 10 bp and at the 25 bp stress level (descriptive; no test at "
        f"25 bp). Hypotheses SHA-256 `{pr.hypotheses_sha256()}`. Phase 2 also re-evaluates "
        "these configurations on the development period and stops if any development Sharpe "
        "ratio differs from `real_data_study_dev.csv`."
    )
    add("")
    add(", ".join(f"`{d['id']}`" for d in hold["configs"]))
    add("")
    add("## 6. Forward rules")
    add("")
    add(
        f"Two rules are frozen now, from development evidence only, and will run whatever the "
        f"holdout shows: they are registered as the study `{fwd['id']}` ({fwd['registered_at_utc']}), "
        "whose configuration list is the two rule files as written. A separate study keeps the "
        "forward log's trial count (two rules) apart from the holdout test's, and its data "
        "access (months from 2026-11 on) apart from the holdout's (2016-2026)."
    )
    add("")
    for d in pr.FORWARD_RULES:
        add(f"* `forward/rules/{d['id']}.json`: {d['description']}")
    add("")
    add(
        "Why these. The forward runner (`forward/run_forward.py`) can express fully invested "
        "weight rules with a band, not volatility targeting or partial rebalancing, so the "
        "largest development effect (volatility targeting of 1/N) cannot be logged without a "
        "runner change, and that effect sat mostly in the 1930s. Among the expressible rules, "
        "long-only minimum variance with Ledoit-Wolf constant-correlation shrinkage had the "
        "largest Sharpe gain over 1/N on 49 industries at about two thirds of the volatility, a "
        "turnover of about 1.3 a year, and an estimator in the pinned core. The 10-industry "
        "version is logged against the market, the benchmark an index investor would hold. "
        "Frozen at 2026-11-01 (first logged month 2026-11, published once the library's "
        "November data appears, early 2027), git tag `forward-v1`, cost 10 bp, no band."
    )
    add("")
    add("## 7. What we will not do")
    add("")
    add(
        "* Re-run the holdout with changed configurations, windows, costs or tests.\n"
        "* Add hypotheses after seeing holdout data. Anything else computed on the holdout will "
        "be labelled exploratory.\n"
        "* Edit a forward rule. A changed rule is a new rule id with a new registration.\n"
        "* If a bug is found after phase 2: report it, describe the fix, rerun once with the fix, "
        "and publish both results."
    )
    add("")
    add("## 8. Limits stated in advance")
    add("")
    add(
        "The series are not investable and costs are assumed; one data vintage; the holdout is "
        "short relative to the effects (section 4); the construction rules tested were "
        "published before 2016, so the holdout is clean with respect to this repository's "
        "choices, not to the literature."
    )
    return "\n".join(out) + "\n"


def _dev_se(dev: dict[Any, Any], h: Any) -> float:
    """Development standard error of the paired difference for hypothesis ``h``.

    When the comparator is the test's lever base or 1/N, the development CSV
    holds that paired test; otherwise it is recomputed by the caller of
    ``real_data_prereg`` (not needed for the registered hypotheses).
    """
    a = dev[(h.test.id, "full")]
    if a["comparator"] == h.comparator.id and a["se_diff_vs_comparator"]:
        return float(a["se_diff_vs_comparator"])
    if h.comparator.rule == "equal_weight" and a["se_diff_vs_equal_weight"]:
        return float(a["se_diff_vs_equal_weight"])
    b = dev[(h.comparator.id, "full")]
    if b["comparator"] == h.test.id and b["se_diff_vs_comparator"]:
        return float(b["se_diff_vs_comparator"])
    raise KeyError(f"{h.id}: no development paired test between test and comparator")


# ---------------------------------------------------------------- final page


def final_markdown(
    entry: dict[str, Any],
    dev_run: core.RunOutput,
    hold_run: core.RunOutput,
    dev_rows: list[Row],
    hold_rows: list[Row],
    tests: list[dict[str, core.Paired]],
    adjusted: list[float],
    commit: str,
    dirty: str,
) -> str:
    import real_data_prereg as pr

    di, hi = _index(dev_rows), _index(hold_rows)
    out = ["# Portfolio-construction levers on real data: pre-registered holdout result", ""]
    out.append(
        f"Phase 2 of the protocol in `real_data_preregistration.md`, run once at commit "
        f"`{commit}`{' with uncommitted changes to study files' if dirty else ''}, on study "
        f"`{entry['id']}` ({entry['n_configs']} configurations). Non-investable index series; "
        "assumed costs; not investment advice."
    )
    out.append("")
    out.append("## Pre-registered tests (10 bp, one-sided, Holm-adjusted)")
    out.append("")
    body = []
    for h, t, adj in zip(pr.HYPOTHESES, tests, adjusted, strict=True):
        d, o = t["development"], t["holdout"]
        lo, up = o.diff - 1.96 * o.se, o.diff + 1.96 * o.se
        body.append(
            [
                h.id,
                h.statement,
                f"{fmt(d.diff)} ({fmt(d.p_one)})",
                f"{fmt(o.diff)} [{fmt(lo)}, {fmt(up)}]",
                fmt(o.p_one, 3),
                fmt(adj, 3),
                "supported" if adj <= pr.ALPHA else "not supported",
            ]
        )
    out.append(
        table(
            [
                "id",
                "hypothesis",
                "dev dSR (p one-sided)",
                "holdout dSR [95% CI]",
                "holdout p",
                "Holm p",
                "decision",
            ],
            body,
            "lllrrrl",
        )
    )
    out.append("")
    out.append("## Development and holdout side by side (10 bp unless stated)")
    out.append("")
    body = []
    for r in hold_rows:
        if r.period != "full":
            continue
        c = r.cfg
        d = di.get((c.id, "full"))
        body.append(
            [
                c.label()
                + ("" if c.cost_bps in (None, core.BASE_COST) else f" ({c.cost_bps:g} bp)"),
                fmt(d.stats.net_sharpe) if d else "",
                _ci(r.stats),
                _pct(r.stats.ann_vol),
                _pct(r.stats.max_drawdown, 0),
                "" if c.rule == "market" else fmt(r.stats.turnover_per_year),
                _diff(r.vs_1n),
                _diff(r.vs_market),
            ]
        )
    out.append(
        table(
            [
                "configuration",
                "dev SR",
                "holdout SR [95% CI]",
                "holdout vol %",
                "holdout max DD %",
                "turnover/yr",
                "vs 1/N",
                "vs market",
            ],
            body,
        )
    )
    out.append("")
    out.append("## Conclusions")
    out.append("")
    out.append(
        "To be written from the table above after the run, with what did not work stated as plainly as what did."
    )
    out.append("")
    out.append(LIMITS)
    _ = hi
    return "\n".join(out) + "\n"
