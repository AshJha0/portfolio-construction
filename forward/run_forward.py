"""Forward paper log: apply frozen rules to new index data, month by month.

Usage (from the repository root, after ``pip install -e python``)::

    python scripts/fetch_data.py          # record the current data vintage
    python forward/run_forward.py         # append new ledger rows, rewrite results.md
    python forward/run_forward.py --dry-run

What it does.  Every ``forward/rules/*.json`` is a frozen rule (schema in
``forward/README.md`` and :func:`validate_rule`).  For each rule the
runner loads the current vintage of the rule's dataset and of the
risk-free series, simulates the rule from the first calendar month that
starts on or after ``frozen_at`` up to the last month fully covered by the
data, and compares the result with ``forward/ledger.csv``:

* a month not yet in the ledger is appended as an ``original`` row;
* a month already in the ledger whose numbers are unchanged is left alone
  (so a second run on the same data appends nothing);
* a month whose numbers changed because the data vintage changed (the
  French library rebuilds its history monthly) gets a ``restatement`` row;
  the original row is never edited;
* a month whose numbers changed under the same vintage is an error: the
  code or the rule changed, and the runner refuses to rewrite history.

The ledger is append-only.  ``forward/results.md`` is regenerated from the
ledger alone (latest row per rule and month), so it is deterministic.
Nothing here is investable: the series are research index portfolios,
the portfolio is hypothetical and no money is involved.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import inspect
import json
import math
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from portopt.research import data as rd
    from portopt.research import signals as sig
    from portopt.research import strategies as strat
    from portopt.research import strategies_v2 as strat_v2
    from portopt.research import walkforward as wf
except ImportError:  # running from a checkout without `pip install -e python`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))
    from portopt.research import data as rd
    from portopt.research import signals as sig
    from portopt.research import strategies as strat
    from portopt.research import strategies_v2 as strat_v2
    from portopt.research import walkforward as wf

HERE = Path(__file__).resolve().parent
RULE_SCHEMA_VERSION = 1
LEDGER_COLUMNS = (
    "month",
    "rule_id",
    "row_type",
    "net_return",
    "gross_return",
    "benchmark_return",
    "cumulative_net",
    "cumulative_benchmark",
    "turnover",
    "cost",
    "trading_days",
    "data_vintage",
    "rule_sha256",
    "recorded_at_utc",
)
#: Numbers are written with 10 decimals; a change smaller than this is noise.
COMPARE_TOL = 5e-9
FREQUENCIES = {"monthly": 1, "quarterly": 3, "annual": 12}

#: Covariance estimators a rule may name, mapped to the
#: :mod:`portopt.research.strategies_v2` factories.  ``ewma`` and ``ewmacc``
#: take ``half_life`` (trading days); ``pca`` takes ``k`` (factors).
ESTIMATORS: dict[str, Callable[..., strat_v2.CovarianceEstimator]] = {
    "sample": strat_v2.sample_estimator,
    "lwcc": strat_v2.lw_constant_corr_estimator,
    "lwid": strat_v2.lw_identity_estimator,
    "nls": strat_v2.nonlinear_estimator,
    "ewma": strat_v2.ewma_estimator,
    "ewmacc": strat_v2.ewma_cc_estimator,
    "pca": strat_v2.pca_estimator,
}


def _number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number, got {value!r}")
    return float(value)


def estimator_from_json(
    estimator: Any, half_life: Any = None, k: Any = None
) -> strat_v2.CovarianceEstimator:
    """Build a covariance estimator from plain JSON values.

    ``half_life`` is required by, and only accepted for, ``ewma`` and
    ``ewmacc``; ``k`` likewise for ``pca``.  Raises ValueError otherwise,
    so a rule file with a misspelt or incomplete estimator fails
    validation instead of silently running a different rule.
    """
    if not isinstance(estimator, str) or estimator not in ESTIMATORS:
        raise ValueError(f"estimator must be one of {sorted(ESTIMATORS)}, got {estimator!r}")
    factory = ESTIMATORS[estimator]
    if estimator in ("ewma", "ewmacc"):
        if half_life is None or k is not None:
            raise ValueError(f"estimator {estimator!r} takes half_life (and no k)")
        return factory(_number("half_life", half_life))
    if estimator == "pca":
        if k is None or half_life is not None:
            raise ValueError("estimator 'pca' takes k (and no half_life)")
        if isinstance(k, bool) or not isinstance(k, int):
            raise ValueError(f"k must be an integer, got {k!r}")
        return factory(k)
    if half_life is not None or k is not None:
        raise ValueError(f"estimator {estimator!r} takes neither half_life nor k")
    return factory()


def min_variance_longonly_with(
    *, estimator: str, half_life: float | None = None, k: int | None = None
) -> strat.WeightFunction:
    """JSON form of :func:`portopt.research.strategies_v2.min_variance_longonly_with`."""
    return strat_v2.min_variance_longonly_with(estimator_from_json(estimator, half_life, k))


def erc_with(
    *, estimator: str, half_life: float | None = None, k: int | None = None
) -> strat.WeightFunction:
    """JSON form of :func:`portopt.research.strategies_v2.erc_with`."""
    return strat_v2.erc_with(estimator_from_json(estimator, half_life, k))


def max_sharpe_bayes_stein(
    *,
    rf: float,
    estimator: str = "sample",
    half_life: float | None = None,
    k: int | None = None,
    n_points: int = 25,
) -> strat.WeightFunction:
    """JSON form of :func:`portopt.research.strategies_v2.max_sharpe_bayes_stein`.

    ``rf`` is a fixed annual rate: the runner passes total returns, not
    excess returns, to every strategy.
    """
    return strat_v2.max_sharpe_bayes_stein(
        rf=_number("rf", rf),
        estimator=estimator_from_json(estimator, half_life, k),
        n_points=n_points,
    )


STRATEGIES: dict[str, Callable[..., strat.WeightFunction]] = {
    "equal_weight": strat.equal_weight,
    "inverse_vol": strat.inverse_vol,
    "erc": strat.erc,
    "min_variance_longonly": strat.min_variance_longonly,
    "max_sharpe_longonly": strat.max_sharpe_longonly,
    "tangency_capped": strat.tangency_capped,
    # portopt.research.strategies_v2 rules, with the estimator named in JSON.
    "min_variance_longonly_with": min_variance_longonly_with,
    "erc_with": erc_with,
    "max_sharpe_bayes_stein": max_sharpe_bayes_stein,
    # portopt.research.signals rules; params are JSON already: "signals" is a
    # list of signal names, "method" "equal" or "inverse_vol".  Stateless, so
    # the runner's fixed trailing window is all they read (lookback_days must
    # cover portopt.research.signals.combine_rows, and the BL lookback).
    "rank_top_tercile": sig.rank_top_tercile,
    "signal_black_litterman": sig.signal_black_litterman,
}
ASSET_DATASETS = ("ind10_vw_daily", "ind49_vw_daily")
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,63}$")
_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,99}$")
_RULE_KEYS_REQUIRED = {
    "schema_version",
    "id",
    "study_id",
    "frozen_at",
    "git_tag",
    "strategy",
    "dataset",
    "lookback_days",
    "cost_bps",
    "band",
    "rebalance",
    "benchmark",
}
_RULE_KEYS_OPTIONAL = {"rf_dataset", "description"}


class RuleError(ValueError):
    """A rule file does not satisfy the schema."""


class LedgerConflict(RuntimeError):
    """The ledger would have to be rewritten; the runner refuses."""


# --------------------------------------------------------------------- rules


@dataclass(frozen=True)
class Rule:
    id: str
    study_id: str
    frozen_at: dt.date
    git_tag: str
    strategy: str
    params: dict[str, Any]
    dataset: str
    rf_dataset: str
    lookback_days: int
    cost_bps: float
    band: float
    rebalance: str
    benchmark: dict[str, Any]
    description: str
    sha256: str

    @property
    def first_month(self) -> np.datetime64:
        """First calendar month starting on or after ``frozen_at``."""
        m = np.datetime64(self.frozen_at, "M")
        return m if self.frozen_at.day == 1 else m + np.timedelta64(1, "M")


def _num(doc: dict[str, Any], key: str, errors: list[str]) -> float:
    v = doc.get(key)
    if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v) or v < 0:
        errors.append(f"{key} must be a finite number >= 0")
        return 0.0
    return float(v)


def _check_factory(name: Any, params: Any, label: str, errors: list[str]) -> None:
    if name not in STRATEGIES:
        errors.append(f"{label}.name must be one of {sorted(STRATEGIES)}, got {name!r}")
        return
    if not isinstance(params, dict):
        errors.append(f"{label}.params must be an object")
        return
    try:
        inspect.signature(STRATEGIES[name]).bind(**params)
        STRATEGIES[name](**params)
    except (TypeError, ValueError) as exc:
        errors.append(f"{label}.params invalid for {name}: {exc}")


def rule_sha256(doc: dict[str, Any]) -> str:
    text = json.dumps(doc, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def validate_rule(doc: Any, *, source: str = "<rule>") -> Rule:
    """Validate one rule document and return it as a :class:`Rule`.

    Raises :class:`RuleError` listing every problem found.
    """
    if not isinstance(doc, dict):
        raise RuleError(f"{source}: a rule must be a JSON object")
    errors: list[str] = []
    missing = _RULE_KEYS_REQUIRED - doc.keys()
    extra = doc.keys() - _RULE_KEYS_REQUIRED - _RULE_KEYS_OPTIONAL
    if missing:
        errors.append(f"missing keys {sorted(missing)}")
    if extra:
        errors.append(f"unknown keys {sorted(extra)}")
    if doc.get("schema_version") != RULE_SCHEMA_VERSION:
        errors.append(f"schema_version must be {RULE_SCHEMA_VERSION}")
    rid = doc.get("id")
    if not isinstance(rid, str) or not _ID_RE.match(rid):
        errors.append("id must match ^[a-z0-9][a-z0-9._-]{2,63}$")
    sid = doc.get("study_id")
    if not isinstance(sid, str) or not sid:
        errors.append("study_id must be a non-empty string")
    frozen = None
    try:
        frozen = dt.date.fromisoformat(str(doc.get("frozen_at")))
        if len(str(doc.get("frozen_at"))) != 10:
            raise ValueError
    except ValueError:
        errors.append("frozen_at must be a date YYYY-MM-DD")
    tag = doc.get("git_tag")
    if not isinstance(tag, str) or not _TAG_RE.match(tag):
        errors.append("git_tag must be a git tag name")
    st = doc.get("strategy")
    if not isinstance(st, dict) or set(st) - {"name", "params"} or "name" not in st:
        errors.append('strategy must be {"name": ..., "params": {...}}')
    else:
        _check_factory(st["name"], st.get("params", {}), "strategy", errors)
    if doc.get("dataset") not in ASSET_DATASETS:
        errors.append(f"dataset must be one of {ASSET_DATASETS}")
    rf_ds = doc.get("rf_dataset", "ff3_daily")
    if rf_ds != "ff3_daily":
        errors.append("rf_dataset must be 'ff3_daily' (the only dataset with an RF column)")
    lb = doc.get("lookback_days")
    if isinstance(lb, bool) or not isinstance(lb, int) or lb < 21:
        errors.append("lookback_days must be an integer >= 21")
    cost = _num(doc, "cost_bps", errors)
    band = _num(doc, "band", errors)
    if doc.get("rebalance") not in FREQUENCIES:
        errors.append(f"rebalance must be one of {sorted(FREQUENCIES)}")
    bm = doc.get("benchmark")
    if not isinstance(bm, dict) or bm.get("kind") not in ("strategy", "market"):
        errors.append('benchmark must be {"kind": "market"} or {"kind": "strategy", "name": ...}')
    elif bm["kind"] == "market":
        if set(bm) != {"kind"}:
            errors.append('benchmark {"kind": "market"} takes no other keys')
    else:
        if set(bm) - {"kind", "name", "params"} or "name" not in bm:
            errors.append('strategy benchmark must be {"kind", "name", "params"}')
        else:
            _check_factory(bm["name"], bm.get("params", {}), "benchmark", errors)
    desc = doc.get("description", "")
    if not isinstance(desc, str):
        errors.append("description must be a string")
    if errors:
        raise RuleError(f"{source}: " + "; ".join(errors))
    assert frozen is not None and isinstance(st, dict) and isinstance(bm, dict)
    return Rule(
        id=str(rid),
        study_id=str(sid),
        frozen_at=frozen,
        git_tag=str(tag),
        strategy=str(st["name"]),
        params=dict(st.get("params", {})),
        dataset=str(doc["dataset"]),
        rf_dataset=str(rf_ds),
        lookback_days=int(lb),  # type: ignore[arg-type]
        cost_bps=cost,
        band=band,
        rebalance=str(doc["rebalance"]),
        benchmark=dict(bm),
        description=desc,
        sha256=rule_sha256(doc),
    )


def load_rules(rules_dir: Path) -> list[Rule]:
    """Load and validate every ``*.json`` in ``rules_dir`` (sorted by name)."""
    rules: list[Rule] = []
    for path in sorted(rules_dir.glob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuleError(f"{path}: invalid JSON: {exc}") from None
        rule = validate_rule(doc, source=str(path))
        if path.stem != rule.id:
            raise RuleError(f"{path}: file name must be {rule.id}.json")
        rules.append(rule)
    ids = [r.id for r in rules]
    if len(set(ids)) != len(ids):
        raise RuleError("duplicate rule ids")
    return rules


# --------------------------------------------------------------- simulation


@dataclass(frozen=True)
class MonthRow:
    month: str
    net: float
    gross: float
    benchmark: float
    turnover: float
    cost: float
    trading_days: int


def complete_through(panel: rd.ReturnsPanel) -> np.datetime64:
    """Last calendar month fully covered by ``panel``.

    The CRSP database month in the file header (for example ``202609``) is
    the authority; without it, the month of the last row counts only if
    that row is the month's last business day.
    """
    if panel.crsp_database:
        c = panel.crsp_database
        return np.datetime64(f"{c[:4]}-{c[4:]}", "M")
    last = panel.dates[-1]
    m = last.astype("datetime64[M]")
    month_end = (m + np.timedelta64(1, "M")).astype("datetime64[D]") - np.timedelta64(1, "D")
    last_bday = np.busday_offset(month_end, 0, roll="backward")
    return m if last >= last_bday else m - np.timedelta64(1, "M")


def _weight_fn(name: str, params: dict[str, Any], lookback: int) -> wf.WeightFn:
    inner = STRATEGIES[name](**params)

    def fn(history: np.ndarray) -> np.ndarray:
        return inner(history[-lookback:])

    return fn


def _compound(x: np.ndarray) -> float:
    return math.prod((1.0 + x).tolist()) - 1.0


def simulate_rule(rule: Rule, assets: rd.ReturnsPanel, rf: rd.ReturnsPanel) -> list[MonthRow]:
    """Monthly results of ``rule`` from its first month to the last complete one."""
    common, ia, ir = np.intersect1d(assets.dates, rf.dates, return_indices=True)
    dates = common
    x = assets.values[ia]
    rf_d = rf.column("RF")[ir]
    last_month = min(complete_through(assets), complete_through(rf))
    first = rule.first_month
    if first > last_month:
        return []
    months = dates.astype("datetime64[M]")
    end_row = int(np.searchsorted(months, last_month + np.timedelta64(1, "M"), side="left"))
    rows = wf.calendar_rebalance_rows(dates[:end_row], str(first), FREQUENCIES[rule.rebalance])
    if not rows or months[rows[0]] != first:
        raise ValueError(f"{rule.id}: no trading day in the first month {first}")
    start = rows[0]
    if start < rule.lookback_days:
        raise ValueError(f"{rule.id}: only {start} rows of history before {first}")
    # Only the trailing lookback before the first rebalance is ever read, so
    # simulate on that slice (older rows may hold missing-value NaNs).
    offset = start - rule.lookback_days
    xs = x[offset:end_row]
    if not np.isfinite(xs).all():
        raise ValueError(f"{rule.id}: missing values in {rule.dataset} inside the rule's window")
    rows_s = [r - offset for r in rows]
    rf_s = rf_d[offset:end_row]
    sim = wf.run_schedule_net(
        xs,
        _weight_fn(rule.strategy, rule.params, rule.lookback_days),
        rows_s,
        cost_bps=rule.cost_bps,
        band=rule.band,
        rf_daily=rf_s,
    )
    if rule.benchmark["kind"] == "market":
        bench_daily = (rf.column("Mkt-RF")[ir] + rf_d)[start:end_row]
    else:
        bsim = wf.run_schedule_net(
            xs,
            _weight_fn(
                rule.benchmark["name"], rule.benchmark.get("params", {}), rule.lookback_days
            ),
            rows_s,
            cost_bps=rule.cost_bps,
            band=rule.band,
            rf_daily=rf_s,
        )
        bench_daily = bsim.net_returns
    m_sim = months[start:end_row]
    turnover_by_row = dict(zip(rows, sim.turnovers.tolist(), strict=True))
    out: list[MonthRow] = []
    for m in np.unique(m_sim):
        sel = m_sim == m
        idx = np.nonzero(sel)[0]
        out.append(
            MonthRow(
                month=str(m),
                net=_compound(sim.net_returns[sel]),
                gross=_compound(sim.gross_returns[sel]),
                benchmark=_compound(bench_daily[sel]),
                turnover=math.fsum(turnover_by_row.get(int(i) + start, 0.0) for i in idx),
                cost=math.fsum(sim.costs[sel].tolist()),
                trading_days=int(sel.sum()),
            )
        )
    return out


def data_vintage(panels: Sequence[rd.ReturnsPanel]) -> str:
    """Short combined hash of the zip SHA-256s of the panels used."""
    text = "|".join(sorted(f"{p.dataset}:{p.zip_sha256}" for p in panels))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ------------------------------------------------------------------- ledger


def _fmt(v: float) -> str:
    return f"{v:.10f}"


def read_ledger(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if tuple(reader.fieldnames or ()) != LEDGER_COLUMNS:
            raise LedgerConflict(f"{path}: header differs from {LEDGER_COLUMNS}")
        return list(reader)


def plan_rows(
    rule: Rule,
    months: list[MonthRow],
    existing: list[dict[str, str]],
    vintage: str,
    recorded_at: str,
) -> list[dict[str, str]]:
    """Ledger rows to append for one rule (see the module docstring)."""
    mine = [r for r in existing if r["rule_id"] == rule.id]
    for r in mine:
        if r["rule_sha256"] != rule.sha256:
            raise LedgerConflict(
                f"{rule.id}: rule file hash {rule.sha256} differs from {r['rule_sha256']} in the "
                "ledger; a frozen rule may not be edited (add a new rule id instead)"
            )
    latest: dict[str, dict[str, str]] = {}
    for r in mine:
        latest[r["month"]] = r
    computed = {m.month for m in months}
    lost = sorted(set(latest) - computed)
    if lost:
        raise LedgerConflict(f"{rule.id}: ledger months {lost} are no longer computable")
    new: list[dict[str, str]] = []
    cum_n = cum_b = 1.0
    for m in months:
        cum_n *= 1.0 + m.net
        cum_b *= 1.0 + m.benchmark
        row = {
            "month": m.month,
            "rule_id": rule.id,
            "row_type": "original",
            "net_return": _fmt(m.net),
            "gross_return": _fmt(m.gross),
            "benchmark_return": _fmt(m.benchmark),
            "cumulative_net": _fmt(cum_n - 1.0),
            "cumulative_benchmark": _fmt(cum_b - 1.0),
            "turnover": _fmt(m.turnover),
            "cost": _fmt(m.cost),
            "trading_days": str(m.trading_days),
            "data_vintage": vintage,
            "rule_sha256": rule.sha256,
            "recorded_at_utc": recorded_at,
        }
        old = latest.get(m.month)
        if old is None:
            new.append(row)
            continue
        same = (
            all(
                abs(float(old[k]) - float(row[k])) <= COMPARE_TOL
                for k in ("net_return", "gross_return", "benchmark_return", "turnover", "cost")
            )
            and old["trading_days"] == row["trading_days"]
        )
        if same:
            continue
        if old["data_vintage"] == vintage:
            raise LedgerConflict(
                f"{rule.id} {m.month}: numbers differ from the ledger under the same data vintage "
                f"{vintage}; refusing to rewrite a past row"
            )
        row["row_type"] = "restatement"
        new.append(row)
    return new


def append_ledger(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    fresh = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LEDGER_COLUMNS, lineterminator="\n")
        if fresh:
            w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------ results

DISCLAIMER = (
    "Hypothetical research log, not investment advice, no real money.  The series are "
    "Kenneth R. French Data Library research index portfolios, which cannot be bought; "
    "the rules were frozen in advance and every month is published whatever the result."
)


def render_results(rules: list[Rule], ledger: list[dict[str, str]]) -> str:
    """Markdown summary built from the ledger only (deterministic)."""
    lines = [
        "# Forward paper log: results",
        "",
        DISCLAIMER,
        "",
        "Generated by `forward/run_forward.py` from `forward/ledger.csv`.  Each month shows the "
        "latest ledger row for that month; restated months are marked.  Returns are simple "
        "monthly returns, net of the rule's trading cost; see `forward/README.md`.",
        "",
    ]
    if ledger:
        stamp = max(r["recorded_at_utc"] for r in ledger)
        lines += [f"Last ledger update: {stamp}", ""]
    for rule in rules:
        mine = [r for r in ledger if r["rule_id"] == rule.id]
        latest: dict[str, dict[str, str]] = {}
        counts: dict[str, int] = {}
        for r in mine:
            latest[r["month"]] = r
            counts[r["month"]] = counts.get(r["month"], 0) + 1
        bm = rule.benchmark
        bm_label = "market (Mkt-RF + RF)" if bm["kind"] == "market" else bm["name"]
        lines += [
            f"## {rule.id}",
            "",
            f"- Strategy: `{rule.strategy}` {json.dumps(rule.params, sort_keys=True)}, "
            f"dataset `{rule.dataset}`, lookback {rule.lookback_days} days, "
            f"{rule.rebalance} rebalance, cost {rule.cost_bps:g} bps, band {rule.band:g}",
            f"- Benchmark: {bm_label}",
            f"- Frozen at {rule.frozen_at.isoformat()} (tag `{rule.git_tag}`), "
            f"study `{rule.study_id}`, rule hash `{rule.sha256}`",
            "",
        ]
        if not latest:
            lines += ["No complete month yet (the data lags by one to two months).", ""]
            continue
        months = sorted(latest)
        cum_n = cum_b = 1.0
        body = []
        for m in months:
            r = latest[m]
            n, b = float(r["net_return"]), float(r["benchmark_return"])
            cum_n *= 1.0 + n
            cum_b *= 1.0 + b
            flag = "yes" if counts[m] > 1 else ""
            body.append(
                f"| {m} | {100 * n:.2f} | {100 * b:.2f} | {100 * (n - b):.2f} | "
                f"{100 * (cum_n - 1):.2f} | {100 * (cum_b - 1):.2f} | "
                f"{float(r['turnover']):.3f} | {flag} |"
            )
        k = len(months)
        ann_n = cum_n ** (12.0 / k) - 1.0
        ann_b = cum_b ** (12.0 / k) - 1.0
        lines += [
            f"{k} month(s), {months[0]} to {months[-1]}: cumulative net {100 * (cum_n - 1):.2f} % "
            f"vs benchmark {100 * (cum_b - 1):.2f} %; annualized {100 * ann_n:.2f} % vs "
            f"{100 * ann_b:.2f} %.  Fewer than 36 months says little about skill.",
            "",
            "| Month | Net % | Benchmark % | Active % | Cum net % | Cum benchmark % | Turnover | Restated |",
            "|---|---:|---:|---:|---:|---:|---:|:---:|",
            *body,
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------- main


def run(
    *,
    rules_dir: Path,
    ledger_path: Path,
    results_path: Path,
    data_dir: Path | None,
    dry_run: bool = False,
    fetch: bool = False,
    now: dt.datetime | None = None,
    fetcher: Callable[[str], bytes] | None = None,
    log: Callable[[str], None] = print,
) -> list[dict[str, str]]:
    """Run every rule once; return the rows appended (or planned, in dry-run)."""
    rules = load_rules(rules_dir) if rules_dir.is_dir() else []
    if not rules:
        log(f"::notice title=Forward log::No rule files in {rules_dir}; nothing to do.")
        return []
    needed = sorted({r.dataset for r in rules} | {r.rf_dataset for r in rules})
    if fetch:
        for name in needed:
            rd.fetch_dataset(name, data_dir=data_dir, now=now, fetcher=fetcher)
    t = dt.datetime.now(dt.UTC) if now is None else now.astimezone(dt.UTC)
    recorded_at = t.strftime("%Y-%m-%dT%H:%M:%SZ")
    ledger = read_ledger(ledger_path)
    planned: list[dict[str, str]] = []
    for rule in rules:
        assets = rd.load_returns(rule.dataset, "all", study_id=rule.study_id, data_dir=data_dir)
        rf = rd.load_returns(rule.rf_dataset, "all", study_id=rule.study_id, data_dir=data_dir)
        months = simulate_rule(rule, assets, rf)
        rows = plan_rows(rule, months, ledger, data_vintage([assets, rf]), recorded_at)
        for r in rows:
            log(
                f"{'[dry-run] ' if dry_run else ''}{r['row_type']:11s} {r['rule_id']} {r['month']} "
                f"net {r['net_return']} benchmark {r['benchmark_return']}"
            )
        if not rows:
            log(f"{rule.id}: no new or restated months")
        planned.extend(rows)
    if dry_run:
        log(f"[dry-run] {len(planned)} row(s) would be appended; nothing written")
        return planned
    append_ledger(ledger_path, planned)
    results_path.write_text(render_results(rules, read_ledger(ledger_path)), encoding="utf-8")
    log(f"appended {len(planned)} row(s) to {ledger_path}; wrote {results_path}")
    return planned


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Forward paper log runner (see forward/README.md)")
    ap.add_argument("--rules-dir", type=Path, default=HERE / "rules")
    ap.add_argument("--ledger", type=Path, default=HERE / "ledger.csv")
    ap.add_argument("--results", type=Path, default=HERE / "results.md")
    ap.add_argument("--data-dir", type=Path, default=None, help="default: data/real")
    ap.add_argument("--fetch", action="store_true", help="fetch the needed datasets first")
    ap.add_argument("--dry-run", action="store_true", help="compute and print; write nothing")
    args = ap.parse_args(argv)
    try:
        run(
            rules_dir=args.rules_dir,
            ledger_path=args.ledger,
            results_path=args.results,
            data_dir=args.data_dir,
            dry_run=args.dry_run,
            fetch=args.fetch,
        )
    except (RuleError, LedgerConflict, rd.DataError, ValueError) as exc:
        print(f"::error title=Forward log::{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
