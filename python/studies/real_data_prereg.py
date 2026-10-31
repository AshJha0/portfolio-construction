"""Pre-registration and holdout run for ``real_data_study.py``.

``register()`` (``--register-holdout``, end of phase 1) freezes three
things before any holdout row is read, and records each so that a later
change is detectable:

* the forward rules: written to ``forward/rules/<id>.json`` and registered
  as the study ``r2-forward-v1`` whose configuration list is the rule
  files themselves;
* the holdout configurations: registered as ``r2-holdout-v1``; the
  registry stores their SHA-256, and the study description carries the
  SHA-256 of the hypothesis list;
* the pre-registration page ``docs/results/real_data_preregistration.md``.

``run_holdout()`` (``--phase holdout``, phase 2) refuses to run unless the
registered configurations and hypotheses hash to the values in this file,
and refuses a second run once ``docs/results/real_data_study.csv``
exists.  It evaluates the registered configurations on the development
period (for the side-by-side table; the numbers must equal those of the
development CSV) and on the holdout, applies the pre-registered tests with
the Holm correction, and writes ``docs/results/real_data_study.md`` and
``.csv``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import real_data_core as core
from real_data_core import Config

from portopt.research import data as rd
from portopt.research import walkforward as wf

_N01 = NormalDist()

#: Family-wise error rate of the pre-registered tests (Holm 1979).
ALPHA = 0.05
FORWARD_FROZEN_AT = "2026-11-01"
FORWARD_GIT_TAG = "forward-v1"
PREREG_PATH = core.OUT_DIR / "real_data_preregistration.md"
DEV_CSV = core.OUT_DIR / "real_data_study_dev.csv"
FINAL_CSV = core.OUT_DIR / "real_data_study.csv"
FINAL_MD = core.OUT_DIR / "real_data_study.md"
#: Trading days in the holdout of vintage 1 (manifest rows minus development rows).
HOLDOUT_DAYS_V1 = 26338 - 23637


@dataclass(frozen=True)
class Hypothesis:
    """One pre-registered test: ``test`` beats ``comparator`` in net Sharpe.

    Metric: the difference of annualized net Sharpe ratios at the base
    cost (10 bp), on the holdout.  Test: Ledoit and Wolf (2008) HAC
    delta-method test as implemented in
    :func:`portopt.research.sharpe_difference_test` (Bartlett kernel,
    ``floor(4 (T/100)^(2/9))`` lags), one-sided, ``H1: SR_test > SR_comp``.
    """

    id: str
    lever: str
    test: Config
    comparator: Config
    statement: str

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "lever": self.lever,
            "test": self.test.id,
            "comparator": self.comparator.id,
            "metric": "annualized net Sharpe difference at 10 bp, holdout period",
            "test_statistic": "Ledoit-Wolf (2008) HAC delta method, Bartlett kernel, default lags",
            "alternative": "one-sided: SR(test) > SR(comparator)",
            "correction": f"Holm step-down across all hypotheses, family-wise alpha {ALPHA:g}",
            "statement": self.statement,
        }


_EW10 = Config("ind10", "benchmark", "equal_weight")
_EW49 = Config("ind49", "benchmark", "equal_weight")
_MV10 = Config("ind10", "estimator", "min_var_lo", "lwcc", 252)
_MV49 = Config("ind49", "estimator", "min_var_lo", "lwcc", 504)

#: Chosen from the development results (docs/results/real_data_study_dev.md),
#: one per lever plus the construction-versus-1/N question on 49 industries.
HYPOTHESES: tuple[Hypothesis, ...] = (
    Hypothesis(
        "H1",
        "vol_target",
        Config(
            "ind10",
            "vol_target",
            "equal_weight",
            None,
            252,
            vol_target=core.VOL_TARGET,
            vol_estimator="sample",
            max_gross=core.VT_MAX_GROSS,
        ),
        _EW10,
        "Volatility targeting (10% target, sample covariance of the last 252 days, gross "
        "capped at 1.5) raises the net Sharpe ratio of 1/N on 10 industries.",
    ),
    Hypothesis(
        "H2",
        "cost_aware",
        Config("ind10", "cost_aware", "min_var_lo", "lwcc", 252, partial=core.PARTIAL),
        _MV10,
        "Closing half the gap to target at each rebalance raises the net Sharpe ratio of "
        "long-only minimum variance (LW constant-correlation, 252 days) on 10 industries.",
    ),
    Hypothesis(
        "H3",
        "estimator",
        _MV49,
        _EW49,
        "Long-only minimum variance (LW constant-correlation, 504 days) has a higher net "
        "Sharpe ratio than 1/N on 49 industries.",
    ),
    Hypothesis(
        "H4",
        "estimator",
        Config("ind49", "estimator", "min_var_lo", "sample", 504),
        Config("ind49", "estimator", "min_var_lo", "ewma", 504),
        "For long-only minimum variance on 49 industries, the sample covariance over 504 "
        "days gives a higher net Sharpe ratio than EWMA with decay 0.97 (the EWMA version "
        "trades about five times as much).",
    ),
    Hypothesis(
        "H5",
        "leverage",
        Config("ind10", "leverage", "tangency_cap2", "sample", 504),
        _EW10,
        "The tangency portfolio of 504-day sample moments with gross leverage capped at 2 "
        "has a higher net Sharpe ratio than 1/N on 10 industries (uncapped, the same rule "
        "lost all its capital in 1929 in development, so it cannot serve as comparator).",
    ),
    Hypothesis(
        "H6",
        "mean_shrinkage",
        Config("ind10", "mean_shrinkage", "max_sharpe_lo_bs", "sample", 252),
        Config("ind10", "mean_shrinkage", "max_sharpe_lo", "sample", 252),
        "Bayes-Stein means raise the net Sharpe ratio of the long-only maximum-Sharpe "
        "portfolio on 10 industries (252 days).",
    ),
)

_DISCLAIMER = (
    "Hypothetical research log on non-investable Kenneth R. French index series; "
    "no real money, not investment advice."
)
FORWARD_RULES: tuple[dict[str, Any], ...] = (
    {
        "schema_version": 1,
        "id": "r2-minvar-lwcc-ind49",
        "study_id": core.FORWARD_STUDY_ID,
        "frozen_at": FORWARD_FROZEN_AT,
        "git_tag": FORWARD_GIT_TAG,
        "strategy": {"name": "min_variance_longonly_with", "params": {"estimator": "lwcc"}},
        "dataset": "ind49_vw_daily",
        "rf_dataset": "ff3_daily",
        "lookback_days": 504,
        "cost_bps": 10,
        "band": 0.0,
        "rebalance": "monthly",
        "benchmark": {"kind": "strategy", "name": "equal_weight", "params": {}},
        "description": "Long-only minimum variance, Ledoit-Wolf constant-correlation "
        "covariance of the last 504 days, 49 industries, monthly, 10 bp; development "
        "configuration ind49/min_var_lo.lwcc/L504/10bp (hypothesis H3). " + _DISCLAIMER,
    },
    {
        "schema_version": 1,
        "id": "r2-minvar-lwcc-ind10",
        "study_id": core.FORWARD_STUDY_ID,
        "frozen_at": FORWARD_FROZEN_AT,
        "git_tag": FORWARD_GIT_TAG,
        "strategy": {"name": "min_variance_longonly_with", "params": {"estimator": "lwcc"}},
        "dataset": "ind10_vw_daily",
        "rf_dataset": "ff3_daily",
        "lookback_days": 504,
        "cost_bps": 10,
        "band": 0.0,
        "rebalance": "monthly",
        "benchmark": {"kind": "market"},
        "description": "Long-only minimum variance, Ledoit-Wolf constant-correlation "
        "covariance of the last 504 days, 10 industries, monthly, 10 bp, against the "
        "market (Mkt-RF + RF, no cost); development configuration "
        "ind10/min_var_lo.lwcc/L504/10bp. " + _DISCLAIMER,
    },
)
#: Development configurations the forward rules correspond to (also evaluated on the holdout).
FORWARD_CONFIGS = (_MV49, Config("ind10", "estimator", "min_var_lo", "lwcc", 504))


def hypotheses_json() -> list[dict[str, Any]]:
    return [h.to_json() for h in HYPOTHESES]


def hypotheses_sha256() -> str:
    text = json.dumps(hypotheses_json(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def holdout_configs() -> list[Config]:
    """Every configuration the holdout run evaluates, in a fixed order.

    The configurations of every hypothesis (test and comparator) and the
    1/N and market benchmarks of each universe involved, at 10 bp and at
    the 25 bp stress level (the stress results are descriptive; no test
    is run on them).
    """
    seen: dict[str, Config] = {}

    def add(c: Config) -> None:
        if c.id not in seen:
            seen[c.id] = c

    unis = sorted(
        {h.test.universe for h in HYPOTHESES} | {h.comparator.universe for h in HYPOTHESES}
    )
    for u in unis:
        for cost in core.COSTS:
            add(Config(u, "benchmark", "equal_weight", cost_bps=cost))
        add(Config(u, "benchmark", "market", cost_bps=None))
    for c in [x for h in HYPOTHESES for x in (h.test, h.comparator)] + list(FORWARD_CONFIGS):
        if c.rule == "market":
            add(c)
            continue
        for cost in core.COSTS:
            add(_with_cost(c, cost))
    return list(seen.values())


def _with_cost(c: Config, cost: float) -> Config:
    from dataclasses import replace

    return replace(c, cost_bps=cost)


def holm(p_values: list[float]) -> list[float]:
    """Holm (1979) step-down adjusted p-values, in the input order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: (p_values[i], i))
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


# --------------------------------------------------------------- dev numbers


def read_dev_csv() -> dict[tuple[str, str], dict[str, str]]:
    """Development results by (config_id, period)."""
    with DEV_CSV.open(newline="", encoding="utf-8") as fh:
        return {(r["config_id"], r["period"]): r for r in csv.DictReader(fh)}


@dataclass(frozen=True)
class Power:
    dev_diff: float
    dev_se: float
    dev_p_one: float
    holdout_se: float
    power_alpha: float
    power_holm_first: float
    mde80: float


def power_of(h: Hypothesis, dev_test: core.Paired, dev_days: int) -> Power:
    """Approximate power on the holdout if the true effect equals the
    development estimate; the standard error scales as ``1 / sqrt(T)``."""
    se_h = dev_test.se * math.sqrt(dev_days / HOLDOUT_DAYS_V1)
    m = max(1, len(HYPOTHESES))
    z_a = _N01.inv_cdf(1.0 - ALPHA)
    z_first = _N01.inv_cdf(1.0 - ALPHA / m)
    z80 = _N01.inv_cdf(0.8)
    d = dev_test.diff
    return Power(
        dev_diff=d,
        dev_se=dev_test.se,
        dev_p_one=dev_test.p_one,
        holdout_se=se_h,
        power_alpha=_N01.cdf(d / se_h - z_a),
        power_holm_first=_N01.cdf(d / se_h - z_first),
        mde80=(z_a + z80) * se_h,
    )


# ---------------------------------------------------------------- register


def _write_once(path: Path, text: str) -> str:
    """Write ``text`` unless the file exists with different content."""
    if path.exists():
        old = path.read_text(encoding="utf-8")
        if old != text:
            raise RuntimeError(
                f"{path} exists with different content; a frozen file is not rewritten"
            )
        return "unchanged"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return "written"


def rule_text(doc: dict[str, Any]) -> str:
    return json.dumps(doc, indent=2, ensure_ascii=True) + "\n"


def register() -> int:
    if not HYPOTHESES or not FORWARD_RULES:
        raise RuntimeError("HYPOTHESES and FORWARD_RULES must be filled in before registration")
    if len(HYPOTHESES) > 6 or len(FORWARD_RULES) > 2:
        raise RuntimeError("at most 6 hypotheses and 2 forward rules")
    for doc in FORWARD_RULES:
        status = _write_once(core.RULES_DIR / f"{doc['id']}.json", rule_text(doc))
        print(f"forward rule {doc['id']}: {status}")
    fwd = wf.register_study(
        core.FORWARD_STUDY_ID,
        "Round 2 forward paper log: the frozen rules in forward/rules/ (this list is the rule "
        f"files as written), frozen at {FORWARD_FROZEN_AT} from development evidence only, "
        "independent of the holdout outcome.",
        list(FORWARD_RULES),
    )
    print(f"registered {fwd['id']}: {fwd['n_configs']} rules at {fwd['registered_at_utc']}")
    cfgs = holdout_configs()
    hold = wf.register_study(
        core.HOLDOUT_STUDY_ID,
        "Round 2 pre-registered holdout test (2016-01-01 onward), see "
        "docs/results/real_data_preregistration.md; evaluated once by "
        "python/studies/real_data_study.py --phase holdout. "
        f"hypotheses_sha256={hypotheses_sha256()}",
        [c.to_json() for c in cfgs],
    )
    print(
        f"registered {hold['id']}: {hold['n_configs']} configurations, sha256 "
        f"{hold['configs_sha256'][:16]}, at {hold['registered_at_utc']}"
    )
    import real_data_report as report

    text = report.prereg_markdown(fwd, hold, read_dev_csv())
    print(f"{PREREG_PATH.name}: {_write_once(PREREG_PATH, text)}")
    return 0


# ------------------------------------------------------------- holdout run


def _registered(study_id: str) -> dict[str, Any]:
    for s in rd.load_registry()["studies"]:
        if s.get("id") == study_id:
            out: dict[str, Any] = s
            return out
    raise RuntimeError(f"study {study_id} is not registered; run --register-holdout first")


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=core.ROOT, capture_output=True, text=True, check=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip()


def _complete_month(prov: dict[str, Any]) -> np.datetime64:
    c = prov.get("crsp_database")
    if not c:
        raise rd.DataError(f"{prov['dataset']}: no CRSP database month in the file header")
    return np.datetime64(f"{c[:4]}-{c[4:]}", "M")


def holdout_panels(configs: list[Config]) -> tuple[dict[str, core.Panel], dict[str, core.Panel]]:
    """(development, holdout) panels at the pinned data vintage.

    The holdout panels run to the last complete month (the CRSP database
    month in the file headers) and keep the development rows, which the
    estimation windows of the first holdout rebalances need; the
    development panels stop at 2015-12-31.
    """
    dev: dict[str, core.Panel] = {}
    hold: dict[str, core.Panel] = {}
    hs = np.datetime64(rd.HOLDOUT_START, "D")
    for u in sorted({c.universe for c in configs}):
        full = core.load_panel(core.UNIVERSES[u], "all", core.HOLDOUT_STUDY_ID, core.DATA_VINTAGE)
        last = min(_complete_month(p) for p in full.provenance)
        end = int(np.searchsorted(full.dates.astype("datetime64[M]"), last + 1, side="left"))
        n_dev = int(np.searchsorted(full.dates, hs, side="left"))
        hold[u] = _slice(full, end)
        dev[u] = _slice(full, n_dev)
    return dev, hold


def _slice(p: core.Panel, end: int) -> core.Panel:
    return core.Panel(
        p.universe,
        p.dates[:end],
        p.x[:end],
        p.rf[:end],
        p.mkt_excess[:end],
        p.columns,
        p.provenance,
    )


def holdout_rows(panel: core.Panel) -> list[int]:
    hs = np.datetime64(rd.HOLDOUT_START, "D")
    first = next(i for i in core.month_starts(panel.dates) if panel.dates[i] >= hs)
    month = str(panel.dates[first].astype("datetime64[M]"))
    return wf.calendar_rebalance_rows(panel.dates, month)


def run_holdout(workers: int) -> int:
    if FINAL_CSV.exists():
        raise RuntimeError(
            f"{FINAL_CSV} exists: the holdout has been evaluated once and is not re-run"
        )
    entry = _registered(core.HOLDOUT_STUDY_ID)
    cfgs = holdout_configs()
    if wf.configs_sha256([c.to_json() for c in cfgs]) != entry["configs_sha256"]:
        raise RuntimeError("holdout configurations in the code differ from the registered ones")
    if f"hypotheses_sha256={hypotheses_sha256()}" not in entry["description"]:
        raise RuntimeError("hypotheses in the code differ from the registered ones")
    registered = [core.config_from_json(d) for d in entry["configs"]]
    if sorted(c.id for c in registered) != sorted(c.id for c in cfgs):
        raise RuntimeError("registered configuration ids differ from the code")
    commit = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--", "python/studies", "python/src", "forward/rules")
    dev_p, hold_p = holdout_panels(cfgs)
    dev_rows = {u: core.dev_rebalance_rows(p) for u, p in dev_p.items()}
    hold_rows = {u: holdout_rows(p) for u, p in hold_p.items()}
    print("development run of the registered configurations")
    dev_run = core.run_configs(
        cfgs, dev_p, dev_rows, workers=workers, check_solver=False, log=print
    )
    dev_stats = core.build_rows(dev_run, "holdout", core.dev_n_trials(), print)
    print("holdout run")
    hold_run = core.run_configs(
        cfgs, hold_p, hold_rows, workers=workers, check_solver=False, log=print
    )
    hold_stats = core.build_rows(hold_run, "holdout", len(cfgs), print)
    _check_against_dev_csv(dev_stats)
    tests = []
    for h in HYPOTHESES:
        out = {}
        for label, run in (("development", dev_run), ("holdout", hold_run)):
            a = run.series[h.test.id].excess
            b = run.series[h.comparator.id].excess
            out[label] = core.paired(a, b)
        tests.append(out)
    adjusted = holm([t["holdout"].p_one for t in tests])
    import real_data_report as report

    _write_final_csv(dev_stats, hold_stats)
    md = report.final_markdown(
        entry, dev_run, hold_run, dev_stats, hold_stats, tests, adjusted, commit, dirty
    )
    FINAL_MD.write_text(md, encoding="utf-8", newline="\n")
    print(f"wrote {FINAL_MD} and {FINAL_CSV}")
    return 0


def _check_against_dev_csv(rows: list[core.Row]) -> None:
    dev = read_dev_csv()
    for r in rows:
        ref = dev.get((r.cfg.id, "full"))
        if ref is None:
            raise RuntimeError(f"{r.cfg.id} is not in the development CSV")
        got = core.fmt(r.stats.net_sharpe, 6)
        if got != ref["net_sharpe"]:
            raise RuntimeError(
                f"{r.cfg.id}: development Sharpe {got} differs from the development CSV "
                f"{ref['net_sharpe']}; the code or the data changed since phase 1"
            )


def relabel(rows: list[core.Row], period: str) -> list[core.Row]:
    """The full-period rows of one run, with ``period`` as their label."""
    from dataclasses import replace

    return [replace(r, period=period) for r in rows if r.period == "full"]


def _write_final_csv(dev_rows: list[core.Row], hold_rows: list[core.Row]) -> None:
    core.write_csv(relabel(dev_rows, "development") + relabel(hold_rows, "holdout"), FINAL_CSV)
