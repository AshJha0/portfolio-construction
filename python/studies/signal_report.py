"""Rows, CSV and the Markdown page for ``signal_study.py``."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

import real_data_core as rc
import signal_core as sc

COLUMNS = [
    "config_id", "label", "universe", "rule", "signal", "cost_bps", "period", "start", "end",
    "n_days", "sharpe", "se", "ann_excess", "ann_vol", "max_dd", "turnover", "cost",
    "d_vs_ew", "se_vs_ew", "p1_vs_ew", "d_vs_minvar", "se_vs_minvar", "p1_vs_minvar",
    "dsr_signal_trials", "dsr_all_trials", "git_head",
]


def _comparators(cfg: sc.SConfig) -> tuple[str, str]:
    c = 10.0 if cfg.cost_bps is None else cfg.cost_bps
    return (sc.SConfig(cfg.universe, "equal_weight", cost_bps=c).id,
            sc.SConfig(cfg.universe, "minvar", cost_bps=c).id)


def _row(cfg: sc.SConfig, res: dict[str, sc.Result], dates: np.ndarray, period: str,
         mask: np.ndarray) -> dict[str, Any]:
    r = res[cfg.id]
    st = sc.summarize(r, mask)
    d = dates[mask]
    row: dict[str, Any] = {"config_id": cfg.id, "label": cfg.label(), "universe": cfg.universe,
                           "rule": cfg.rule, "signal": cfg.signal or "",
                           "cost_bps": "" if cfg.cost_bps is None else cfg.cost_bps,
                           "period": period, "start": str(d[0]), "end": str(d[-1])} | st
    ew, mv = _comparators(cfg)
    for key, other in (("ew", ew), ("minvar", mv)):
        if other in res and other != cfg.id:
            p = sc.diff(r, res[other], mask)
            row |= {f"d_vs_{key}": p.diff, f"se_vs_{key}": p.se, f"p1_vs_{key}": p.p_one}
    return row


def dev_rows(panels: dict[str, rc.Panel], grid: list[sc.SConfig],
             res: dict[str, sc.Result]) -> list[dict[str, Any]]:
    # Deflated Sharpe ratio: variance of per-period Sharpe ratios across the
    # signal trials, with the signal count and with the 206 construction trials added.
    srs = [sc.per_period(rc.ann_sr(r.excess)) for r in res.values()]
    var = float(np.var(srs, ddof=1))
    n_sig = len(grid)
    n_all = n_sig + sc.construction_trials()
    out: list[dict[str, Any]] = []
    for cfg in grid:
        p = panels[cfg.universe]
        r = res[cfg.id]
        dates = p.dates[r.start:]
        full = np.ones(dates.shape[0], dtype=bool)
        row = _row(cfg, res, dates, "dev", full)
        row["dsr_signal_trials"] = sc.dsr(r, var, n_sig)
        row["dsr_all_trials"] = sc.dsr(r, var, n_all)
        out.append(row)
        for name, lo, hi in sc.SUBPERIODS:
            m = sc.date_mask(dates, lo, hi)
            if m.sum() > 252:
                out.append(_row(cfg, res, dates, name, m))
        if cfg.signal in sc.PUBLICATION:
            y = sc.PUBLICATION[cfg.signal]
            for name, lo, hi in ((f"pre-pub<{y}", None, f"{y}-01-01"),
                                 (f"post-pub>={y}", f"{y}-01-01", None)):
                m = sc.date_mask(dates, lo, hi)
                if m.sum() > 252:
                    out.append(_row(cfg, res, dates, name, m))
    return out


def seen_rows(panels: dict[str, rc.Panel], cfgs: list[sc.SConfig],
              res: dict[str, sc.Result]) -> list[dict[str, Any]]:
    out = []
    for cfg in cfgs:
        r = res[cfg.id]
        dates = panels[cfg.universe].dates[r.start:]
        out.append(_row(cfg, res, dates, "seen 2016-2026", np.ones(dates.shape[0], dtype=bool)))
    return out


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()
                        if k in COLUMNS})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(v: str, nd: int = 2) -> str:
    if v in ("", None):
        return ""
    x = float(v)
    return "" if math.isnan(x) else f"{x:.{nd}f}"


def _pair(r: dict[str, str], key: str) -> str:
    if not r.get(f"d_vs_{key}"):
        return ""
    return f"{f(r[f'd_vs_{key}'])} ({f(r[f'p1_vs_{key}'])})"


def table(header: list[str], body: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(b) + " |" for b in body]
    return "\n".join(lines)


def render(rows: list[dict[str, str]]) -> str:
    try:
        import signal_prereg as pr
    except ImportError:  # before the pre-registration exists
        pr = None
    dev = {r["config_id"]: r for r in rows if r["period"] == "dev"}
    seen = {r["config_id"]: r for r in rows if r["period"].startswith("seen")}
    parts = ["# Return signals on real index data (round 2)", "",
             "Development 1926-07 to 2015-12 (`period=\"development\"`, study `r2-signal-dev`); "
             "seen period 2016-01 to 2026-09 (study `r2-signal-v1`). Every number is in "
             "`signal_study.csv`, produced by `python/studies/signal_study.py`. Evidence class: "
             "backtest on non-investable index series with assumed linear costs; not investment "
             "advice.", ""]
    if pr is not None:
        parts += pr.results_section(dev, seen)
    parts += ["## Development, full period", "",
              "Net Sharpe (annualized, excess of RF); differences in Sharpe vs 1/N and vs "
              "long-only minimum variance (LW constant-correlation, 504 days) at the same cost, "
              "one-sided Ledoit-Wolf p in brackets. DSR uses the variance of Sharpe ratios across "
              "the signal trials, with the signal trial count and with the construction trials "
              "added.", ""]
    body = []
    for r in dev.values():
        body.append([r["universe"], r["label"], f(r["sharpe"]), f(r["se"]),
                     f(str(100 * float(r["ann_vol"])), 1), f(str(100 * float(r["max_dd"])), 0),
                     f(r["turnover"]), _pair(r, "ew"), _pair(r, "minvar"),
                     f(r["dsr_signal_trials"]), f(r["dsr_all_trials"])])
    parts += [table(["universe", "configuration", "SR", "SE", "vol %", "max DD %", "turnover/yr",
                     "vs 1/N", "vs min-var", "DSR (sig)", "DSR (all)"], body), ""]
    parts += ["## Development sub-periods and publication split (10 bp, difference vs 1/N)", ""]
    periods = sorted({r["period"] for r in rows if r["period"] not in ("dev",)
                      and not r["period"].startswith("seen")})
    body = []
    for cid, r in dev.items():
        if r["cost_bps"] not in ("10", "10.0") or r["rule"] in ("equal_weight", "market"):
            continue
        sub = {x["period"]: x for x in rows if x["config_id"] == cid}
        cells = [r["universe"], r["label"]]
        for name, _, _ in sc.SUBPERIODS:
            cells.append(_pair(sub[name], "ew") if name in sub else "")
        pre = next((v for k, v in sub.items() if k.startswith("pre-pub")), None)
        post = next((v for k, v in sub.items() if k.startswith("post-pub")), None)
        cells += [pre["period"].split("<")[1] if pre else "",
                  _pair(pre, "ew") if pre else "", _pair(post, "ew") if post else ""]
        body.append(cells)
    _ = periods
    parts += [table(["universe", "configuration", "pre-1963", "1963-1990", "1991-2015",
                     "pub. year", "before pub.", "after pub. (to 2015)"], body), ""]
    return "\n".join(parts) + "\n"
