"""Cost-aware allocation study on the bundled synthetic data.

Run from the repository root::

    python python/studies/cost_aware_allocation.py

Writes ``docs/results/cost_aware_allocation.md`` and
``docs/results/cost_aware_allocation.csv``.  The study is deterministic:
the inputs are the bundled files under ``data/``, the only randomness is
the block bootstrap seeded with ``BOOTSTRAP_SEED``, and every number is
written with a fixed format, so a rerun produces byte-identical files.

This script uses :mod:`portopt.research`, the Python-only research layer
that sits outside the pinned cross-language contract.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

import portopt as po
from portopt import research as rs

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
)
DATA = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "docs", "results")

COSTS_BPS = (0.0, 5.0, 10.0, 25.0)
BANDS = (0.0, 0.05, 0.10)
TANGENCY_CAP = 2.0
BOOTSTRAP_SEED = 42
BOOTSTRAP_N = 2000
BOOTSTRAP_BLOCK = 10.0
INFERENCE_COST_BPS = 10.0
BAND_TABLE_COST_BPS = 25.0
BENCHMARK = "equal_weight"
LOW_TURNOVER = 0.1


class CachedRule:
    """Memoize a weight rule by history length.

    Within one study the history at a rebalance date is always the same
    prefix of the same matrix, so caching by its length is exact and lets
    every cost and band configuration reuse one set of target weights (and
    one fallback log).
    """

    def __init__(self, rule: rs.WeightFunction) -> None:
        self.rule = rule
        self.name = rule.name
        self._cache: dict[int, np.ndarray] = {}

    def __call__(self, history: np.ndarray) -> np.ndarray:
        t = int(history.shape[0])
        if t not in self._cache:
            self._cache[t] = np.asarray(self.rule(history), dtype=float).copy()
        return self._cache[t].copy()


@dataclass
class Row:
    strategy: str
    cost_bps: float
    band: float
    sharpe: float
    gross_sharpe: float
    se_iid: float
    se_nonnormal: float
    psr0: float
    dsr: float
    ann_return: float
    ann_vol: float
    max_drawdown: float
    avg_turnover: float
    annual_turnover: float
    ann_cost: float
    n_trades: int
    n_rebalances: int
    max_target_gross: float
    diff_vs_bench: float
    p_vs_bench: float
    excess: np.ndarray


def load() -> tuple[np.ndarray, list[str], dict[str, float], dict[str, object], np.ndarray]:
    x = np.loadtxt(os.path.join(DATA, "asset_returns.csv"), delimiter=",", skiprows=1)
    with open(os.path.join(DATA, "asset_returns.csv")) as fh:
        names = fh.readline().strip().split(",")
    with open(os.path.join(DATA, "config.json")) as fh:
        cfg = json.load(fh)
    with open(os.path.join(DATA, "views.json")) as fh:
        views = json.load(fh)
    caps = np.loadtxt(os.path.join(DATA, "market_caps.csv"), delimiter=",", skiprows=1, usecols=1)
    return x, names, cfg, views, caps


def build_rules(
    names: list[str], rf: float, views: dict[str, object], caps: np.ndarray
) -> list[CachedRule]:
    view_list = views["views"]
    assert isinstance(view_list, list)
    p, q, conf = po.views_to_matrices(view_list, names)
    tau = float(views["tau"])  # type: ignore[arg-type]
    delta = float(views["delta"])  # type: ignore[arg-type]
    rules: list[rs.WeightFunction] = [
        rs.equal_weight(),
        rs.inverse_vol(),
        rs.erc(),
        rs.erc(shrink=True),
        rs.min_variance_longonly(),
        rs.min_variance_longonly(shrink=True),
        rs.max_sharpe_longonly(rf=rf),
        rs.max_sharpe_longonly(rf=rf, shrink=True),
        rs.tangency_capped(rf=rf),
        rs.tangency_capped(rf=rf, max_gross=TANGENCY_CAP),
        rs.tangency_capped(rf=rf, max_gross=TANGENCY_CAP, shrink=True),
        rs.black_litterman(w_mkt=caps, p=p, q=q, tau=tau, delta=delta, confidences=conf.tolist()),
    ]
    return [CachedRule(r) for r in rules]


def run_all(
    x: np.ndarray,
    rules: list[CachedRule],
    rf: float,
    window: int,
    step: int,
    excluded: list[tuple[str, str]],
) -> tuple[list[Row], dict[str, rs.NetBacktestResult]]:
    rf_d = rf / 252.0
    raw: list[tuple[CachedRule, float, float, rs.NetBacktestResult]] = []
    for rule in rules:
        mine: list[tuple[CachedRule, float, float, rs.NetBacktestResult]] = []
        try:
            for band in BANDS:
                for cost in COSTS_BPS:
                    res = rs.run_backtest_net(
                        x, rule, window=window, step=step, cost_bps=cost, band=band, rf=rf
                    )
                    mine.append((rule, cost, band, res))
        except ValueError as exc:
            # A rule that cannot produce weights (for example a risk-budget
            # solver that reports non-convergence) is excluded from every
            # table and listed in the report, never silently patched.
            if rule.name == BENCHMARK:
                raise
            excluded.append((rule.name, str(exc)))
            continue
        raw.extend(mine)
    n_trials = len(raw)
    srs = np.array([rs.sharpe_per_period(r.net_returns - rf_d) for _, _, _, r in raw])
    sr_var = float(np.var(srs, ddof=1))
    bench = {(c, b): r.net_returns - rf_d for rule, c, b, r in raw if rule.name == BENCHMARK}
    rows: list[Row] = []
    by_key: dict[str, rs.NetBacktestResult] = {}
    for rule, cost, band, res in raw:
        ex = res.net_returns - rf_d
        if rule.name == BENCHMARK:
            diff, pval = 0.0, 1.0
        else:
            test = rs.sharpe_difference_test(ex, bench[(cost, band)])
            diff, pval = rs.annualize_sharpe(test.diff), test.p_value
        rows.append(
            Row(
                strategy=rule.name,
                cost_bps=cost,
                band=band,
                sharpe=res.sharpe,
                gross_sharpe=res.gross_sharpe,
                se_iid=rs.annualize_sharpe(rs.sharpe_se_iid(ex)),
                se_nonnormal=rs.annualize_sharpe(rs.sharpe_se_nonnormal(ex)),
                psr0=rs.probabilistic_sharpe_ratio(ex, 0.0),
                dsr=rs.deflated_sharpe_ratio(ex, n_trials=n_trials, sr_variance=sr_var),
                ann_return=res.ann_return,
                ann_vol=res.ann_vol,
                max_drawdown=res.max_drawdown,
                avg_turnover=res.avg_turnover,
                annual_turnover=res.annual_turnover,
                ann_cost=res.ann_cost,
                n_trades=res.n_trades,
                n_rebalances=len(res.rebalance_dates),
                max_target_gross=max(po.gross_leverage(w) for w in res.target_weights),
                diff_vs_bench=diff,
                p_vs_bench=pval,
                excess=ex,
            )
        )
        by_key[f"{rule.name}|{cost:g}|{band:g}"] = res
    return rows, by_key


def f(v: float, nd: int = 2) -> str:
    out = f"{v:.{nd}f}"
    return "0." + "0" * nd if out == "-0." + "0" * nd else out


def pick(rows: list[Row], name: str, cost: float, band: float) -> Row:
    for r in rows:
        if r.strategy == name and r.cost_bps == cost and r.band == band:
            return r
    raise KeyError((name, cost, band))


def table(header: list[str], body: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def write_csv(rows: list[Row], path: str) -> None:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(
        [
            "strategy",
            "cost_bps",
            "band",
            "net_sharpe",
            "gross_sharpe",
            "se_iid",
            "se_nonnormal",
            "psr_vs_0",
            "dsr",
            "ann_return",
            "ann_vol",
            "max_drawdown",
            "avg_turnover_per_rebalance",
            "annual_turnover",
            "ann_cost",
            "n_trades",
            "n_rebalances",
            "max_target_gross_leverage",
            "sharpe_diff_vs_equal_weight",
            "p_value_vs_equal_weight",
        ]
    )
    for r in rows:
        w.writerow(
            [
                r.strategy,
                f"{r.cost_bps:g}",
                f"{r.band:g}",
                f(r.sharpe, 6),
                f(r.gross_sharpe, 6),
                f(r.se_iid, 6),
                f(r.se_nonnormal, 6),
                f(r.psr0, 6),
                f(r.dsr, 6),
                f(r.ann_return, 6),
                f(r.ann_vol, 6),
                f(r.max_drawdown, 6),
                f(r.avg_turnover, 6),
                f(r.annual_turnover, 6),
                f(r.ann_cost, 6),
                r.n_trades,
                r.n_rebalances,
                f(r.max_target_gross, 6),
                f(r.diff_vs_bench, 6),
                f(r.p_vs_bench, 6),
            ]
        )
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.write(buf.getvalue())


def write_md(
    rows: list[Row],
    rules: list[CachedRule],
    boot: dict[str, rs.BootstrapResult],
    cfg: dict[str, float],
    n_days: int,
    n_reb: int,
    sr_var_ann: float,
    sr0_ann: float,
    n_assets: int,
    excluded: list[tuple[str, str]],
    path: str,
) -> None:
    names = [r.name for r in rules]
    rf = float(cfg["risk_free_rate"])
    out: list[str] = []
    add: Callable[[str], None] = out.append

    add("# Cost-aware allocation on synthetic data")
    add("")
    add(
        "**The data are synthetic.** The 1250 daily returns in `data/asset_returns.csv` "
        "come from `data/generate_data.py` (a three-factor model with Student-t shocks, "
        "seed 42). No result on this page is evidence of an edge in any real market. "
        "What the page does show is how the construction rules in "
        "`portopt.research.strategies` behave once trading costs, weight drift and "
        "sampling error are accounted for, using instruments that would apply unchanged "
        "to real data."
    )
    add("")
    add(
        "Generated by `python python/studies/cost_aware_allocation.py`; "
        "the full grid is in `cost_aware_allocation.csv`. Rerunning the script "
        "reproduces both files byte for byte."
    )
    add("")
    add("## Method")
    add("")
    add(
        f"We use `portopt.research.run_backtest_net` with the pinned schedule: an "
        f"expanding estimation window starting at {int(cfg['estimation_window'])} days "
        f"and a rebalance every {int(cfg['rebalance_step'])} days, which gives "
        f"{n_reb} rebalances and a {n_days}-day holdout. Between rebalances the "
        f"portfolio is buy and hold, so weights drift and the next rebalance pays for "
        f"undoing that drift. Each rebalance, including the initial build from cash, is "
        f"charged `cost_bps` on the traded risky notional `sum_i |w_new_i - w_drift_i|`. "
        f"A no-trade band skips a rebalance when that L1 distance is below `band`. "
        f"Cash earns rf = {rf:g} a year (rf/252 a day). We ran {len(names)} rules at "
        f"cost_bps in {{{', '.join(f'{c:g}' for c in COSTS_BPS)}}} and band in "
        f"{{{', '.join(f'{b:g}' for b in BANDS)}}}, {len(rows)} configurations in all."
    )
    add("")
    add(
        "Sharpe ratios are annualized `(252 mean(r) - rf) / (sqrt(252) std(r))` on net "
        "daily returns. Standard errors are Lo (2002) under iid normal returns and "
        "Mertens (2002) under iid non-normal returns, both scaled by sqrt(252); neither "
        "allows for serial correlation. PSR is the Bailey and Lopez de Prado (2012) "
        "probability that the true Sharpe ratio exceeds zero. DSR (Bailey and Lopez de "
        f"Prado 2014) raises that hurdle to the expected maximum Sharpe ratio of "
        f"N = {len(rows)} trials, using the cross-sectional variance of the "
        f"{len(rows)} daily Sharpe estimates (annualized standard deviation "
        f"{f(math.sqrt(sr_var_ann))}, hurdle SR0 = {f(sr0_ann)} annualized). Counting "
        "every configuration as a trial overstates the number of independent trials, "
        "since the cost and band variants of one rule are highly correlated, so the DSR "
        "here is conservative. The paired test against equal weight is the Ledoit and "
        "Wolf (2008) delta-method test with a Newey-West Bartlett kernel; the bootstrap "
        f"interval is a stationary bootstrap (mean block {BOOTSTRAP_BLOCK:g} days, "
        f"{BOOTSTRAP_N} resamples, seed {BOOTSTRAP_SEED}), percentile 95%."
    )
    add("")
    add(
        "Rules: `max_sharpe_lo` is the best Sharpe point of a 25-point long-only "
        "frontier; `tangency` is the unconstrained tangency portfolio, falling back to "
        f"the unconstrained minimum-variance portfolio when it is undefined; "
        f"`tangency_cap{TANGENCY_CAP:g}` moves it toward minimum variance until gross "
        f"leverage is at most {TANGENCY_CAP:g}; the `_lw` suffix means Ledoit-Wolf "
        "constant-correlation shrinkage of the covariance; `black_litterman` uses "
        "`data/views.json` and the market caps."
    )
    add("")

    add("## Net Sharpe ratio by cost, band 0")
    add("")
    body = []
    for n in names:
        rs0 = [pick(rows, n, c, 0.0) for c in COSTS_BPS]
        body.append(
            [f"`{n}`"]
            + [f(r.sharpe) for r in rs0]
            + [f(rs0[0].se_iid), f(rs0[0].avg_turnover), f(100 * rs0[-1].ann_cost)]
        )
    add(
        table(
            ["rule"]
            + [f"{c:g} bp" for c in COSTS_BPS]
            + ["SE (iid)", "turnover / rebalance", f"cost %/yr at {COSTS_BPS[-1]:g} bp"],
            body,
        )
    )
    add("")
    add(
        "Turnover is the mean traded notional per rebalance after the initial build, as "
        "a multiple of NAV, at 0 bp."
    )
    add("")

    add(f"## Effect of the no-trade band at {BAND_TABLE_COST_BPS:g} bp")
    add("")
    body = []
    for n in names:
        rr = [pick(rows, n, BAND_TABLE_COST_BPS, b) for b in BANDS]
        body.append(
            [f"`{n}`"]
            + [f(r.sharpe) for r in rr]
            + [f(r.annual_turnover) for r in rr]
            + [f"{rr[-1].n_trades}/{rr[-1].n_rebalances}"]
        )
    add(
        table(
            ["rule"]
            + [f"Sharpe, band {b:g}" for b in BANDS]
            + [f"turnover/yr, band {b:g}" for b in BANDS]
            + [f"trades, band {BANDS[-1]:g}"],
            body,
        )
    )
    add("")

    add(f"## Inference at {INFERENCE_COST_BPS:g} bp, band 0")
    add("")
    body = []
    for n in names:
        r = pick(rows, n, INFERENCE_COST_BPS, 0.0)
        b = boot[n]
        bench = "" if n == BENCHMARK else f"{f(r.diff_vs_bench)} (p = {f(r.p_vs_bench)})"
        body.append(
            [
                f"`{n}`",
                f(r.sharpe),
                f(r.se_iid),
                f(r.se_nonnormal),
                f"[{f(rs.annualize_sharpe(b.lower))}, {f(rs.annualize_sharpe(b.upper))}]",
                f(r.psr0),
                f(r.dsr),
                bench,
                f(100 * r.max_drawdown, 1),
            ]
        )
    add(
        table(
            [
                "rule",
                "net Sharpe",
                "SE iid",
                "SE non-normal",
                "bootstrap 95% CI",
                "PSR(0)",
                "DSR",
                "diff vs equal weight",
                "max DD %",
            ],
            body,
        )
    )
    add("")

    if excluded:
        add("## Excluded rules")
        add("")
        add("These rules raised an error at some configuration and appear in no table:")
        add("")
        for name, msg in excluded:
            add(f"* `{name}`: {msg}")
        add("")

    add("## Fallbacks")
    add("")
    body = [
        [f"`{r.name}`", str(len(r.rule.fallback_dates)), str(n_reb)]
        for r in rules
        if r.rule.fallback_dates
    ]
    add(table(["rule", "rebalances that fell back", "rebalances"], body))
    add("")
    add(
        "`tangency` falls back when the risk-free rate is at or above the history's "
        "minimum-variance return; `max_sharpe_lo` falls back to long-only minimum "
        "variance when no frontier point earns more than rf."
    )
    add("")

    # ------------------------------------------------------------ conclusion
    hi = COSTS_BPS[-1]
    long_only = [n for n in names if not n.startswith("tangency")]
    tan0 = pick(rows, "tangency", 0.0, 0.0)
    tan_hi = pick(rows, "tangency", hi, 0.0)
    cap0 = pick(rows, f"tangency_cap{TANGENCY_CAP:g}", 0.0, 0.0)
    lo0 = max((pick(rows, n, 0.0, 0.0) for n in long_only), key=lambda r: r.sharpe)
    lo_hi = pick(rows, lo0.strategy, hi, 0.0)
    lo_loss = max(pick(rows, n, 0.0, 0.0).sharpe - pick(rows, n, hi, 0.0).sharpe for n in long_only)
    low_to = [n for n in names if pick(rows, n, 0.0, 0.0).avg_turnover < LOW_TURNOVER]
    band_moves = [
        abs(pick(rows, n, hi, b).sharpe - pick(rows, n, hi, 0.0).sharpe)
        for n in low_to
        for b in BANDS[1:]
    ]
    band_to = [
        pick(rows, n, hi, BANDS[-1]).annual_turnover / pick(rows, n, hi, 0.0).annual_turnover
        for n in low_to
    ]
    tan_band = pick(rows, "tangency", hi, BANDS[-1])
    lw_gap = max(
        abs(a.sharpe - b.sharpe)
        for a in rows
        for b in rows
        if b.strategy == a.strategy + "_lw"
        and a.cost_bps == b.cost_bps
        and a.band == 0.0
        and b.band == 0.0
    )
    lw_gap_band = max(
        abs(a.sharpe - b.sharpe)
        for a in rows
        for b in rows
        if b.strategy == a.strategy + "_lw" and a.cost_bps == b.cost_bps and a.band == b.band
    )
    ew_inf = pick(rows, BENCHMARK, INFERENCE_COST_BPS, 0.0)
    max_dsr = max(r.dsr for r in rows)
    min_p = min(r.p_vs_bench for r in rows if r.strategy != BENCHMARK)
    max_psr = max(r.psr0 for r in rows)
    n_cmp = sum(1 for r in rows if r.strategy != BENCHMARK)
    tan_fb = len(next(r for r in rules if r.name == "tangency").rule.fallback_dates)
    add("## Conclusion")
    add("")
    add(
        f"1. Costs bite in proportion to turnover. The unconstrained tangency rule trades "
        f"{f(tan0.avg_turnover, 1)} times NAV per rebalance. At {tan_fb} of {n_reb} "
        f"rebalances the risk-free rate is above the history's minimum-variance return, "
        f"tangency is undefined and the rule holds minimum variance; at the others the "
        f"tangency portfolio exists but rf sits close to that return and gross leverage "
        f"reaches {f(tan0.max_target_gross, 0)} times NAV. Its net Sharpe falls from {f(tan0.sharpe)} at "
        f"0 bp to {f(tan_hi.sharpe)} at {hi:g} bp, a cost of "
        f"{f(100 * tan_hi.ann_cost, 1)}% of NAV a year, and its maximum drawdown is "
        f"{f(100 * tan0.max_drawdown, 0)}%. Capping gross leverage at {TANGENCY_CAP:g} "
        f"cuts turnover to {f(cap0.avg_turnover)} per rebalance. The long-only rules "
        f"lose at most {f(lo_loss)} of Sharpe between 0 and {hi:g} bp."
    )
    add(
        f"2. A gross backtest flatters high-turnover rules. At 0 bp the tangency rule "
        f"leads the best long-only rule (`{lo0.strategy}`) by "
        f"{f(tan0.sharpe - lo0.sharpe)} of Sharpe; at {hi:g} bp the lead is "
        f"{f(tan_hi.sharpe - lo_hi.sharpe)}, with roughly "
        f"{f(tan_hi.max_drawdown / lo_hi.max_drawdown, 0)} times the drawdown."
    )
    add(
        f"3. For the {len(low_to)} rules that trade less than {LOW_TURNOVER:g} of NAV per "
        f"rebalance, a band of {BANDS[-1]:g} at {hi:g} bp cuts annual turnover to between {f(100 * min(band_to), 0)}% and {f(100 * max(band_to), 0)}% of "
        f"the band-0 level, and moves their net Sharpe by at most {f(max(band_moves))}, "
        f"in either direction. For the tangency rule the band barely changes turnover "
        f"({f(tan_band.annual_turnover, 1)} against {f(tan_hi.annual_turnover, 1)} a year) "
        f"because its trades come from changes in the target, not from drift."
    )
    add(
        f"4. Without a band, Ledoit-Wolf shrinkage changes no net Sharpe ratio by more "
        f"than {f(lw_gap, 3)}; with a band the gap reaches {f(lw_gap_band)}, because a small "
        f"change in the target can decide whether a rebalance clears the band. With at least {int(cfg['estimation_window'])} observations on "
        f"{n_assets} assets the sample "
        "covariance is already well conditioned, so shrinkage has little to correct."
    )
    add(
        f"5. None of the differences is statistically meaningful. With a {n_days}-day "
        f"holdout the standard error of an annualized Sharpe ratio is about "
        f"{f(ew_inf.se_iid)}, as large as the spread between the rules. The largest "
        f"PSR(0) is {f(max_psr)}, the largest DSR across all {len(rows)} configurations "
        f"is {f(max_dsr)}, and the smallest p-value of a paired test against equal weight "
        f"is {f(min_p)} before any correction for the {n_cmp} comparisons made."
    )
    add(
        "6. What this does not show: anything about real markets, any rule's "
        "out-of-sample edge, or the right cost level for a given desk. The cost model is "
        "linear in traded notional with no market impact, borrowing is at rf, and the "
        "tangency fallback is a choice of ours that drives its turnover. What it does "
        "show is that the instruments (drifted-weight accounting, explicit costs, Sharpe "
        "standard errors, PSR, DSR and paired tests) are in place, and that on a "
        f"{n_days}-day sample they do not support calling any of these rules a winner."
    )
    add("")
    with open(path, "w", newline="\n", encoding="utf-8") as fh:
        fh.write("\n".join(out))


def main() -> int:
    x, names, cfg, views, caps = load()
    rf = float(cfg["risk_free_rate"])
    window = int(cfg["estimation_window"])
    step = int(cfg["rebalance_step"])
    rules = build_rules(names, rf, views, caps)
    excluded: list[tuple[str, str]] = []
    rows, _ = run_all(x, rules, rf, window, step, excluded)
    dropped = {name for name, _ in excluded}
    rules = [r for r in rules if r.name not in dropped]
    for name, msg in excluded:
        print(f"excluded {name}: {msg}")

    srs = np.array([rs.sharpe_per_period(r.excess) for r in rows])
    sr_var = float(np.var(srs, ddof=1))
    sr0 = rs.expected_max_sharpe(len(rows), sr_var)
    boot = {
        r.strategy: rs.sharpe_bootstrap_ci(
            r.excess, seed=BOOTSTRAP_SEED, n_boot=BOOTSTRAP_N, block_length=BOOTSTRAP_BLOCK
        )
        for r in rows
        if r.cost_bps == INFERENCE_COST_BPS and r.band == 0.0
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    write_csv(rows, os.path.join(OUT_DIR, "cost_aware_allocation.csv"))
    write_md(
        rows,
        rules,
        boot,
        cfg,
        x.shape[0] - window,
        len(po.rebalance_indices(x.shape[0], window, step)),
        sr_var * 252.0,
        rs.annualize_sharpe(sr0),
        x.shape[1],
        excluded,
        os.path.join(OUT_DIR, "cost_aware_allocation.md"),
    )
    print(f"wrote {len(rows)} configurations to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
