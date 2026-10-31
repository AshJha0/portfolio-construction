"""Configurations, data, estimation, backtests and statistics for
``real_data_study.py`` (see that script for the protocol and usage).

Kept apart from the command-line script so that the worker processes of
the parallel stages import it by name, and so that the report and
pre-registration helpers share one copy of its types.
"""

from __future__ import annotations

import csv
import io
import math
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

import portopt as po
from portopt import research as rs
from portopt.research import data as rd
from portopt.research import walkforward as wf

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import real_data_engine as eng  # noqa: E402

ROOT = HERE.parents[1]
OUT_DIR = ROOT / "docs" / "results"
RULES_DIR = ROOT / "forward" / "rules"

# ------------------------------------------------------------------ protocol

DEV_STUDY_ID = "r2-dev-exploration"
HOLDOUT_STUDY_ID = "r2-holdout-v1"
FORWARD_STUDY_ID = "r2-forward-v1"
DEV_TC_STUDY_ID = "r2-dev-exploration-tc"
DEV_STUDY_IDS = (DEV_STUDY_ID, DEV_TC_STUDY_ID)
#: Risk aversions for the turnover-penalised minimum variance: the penalty
#: per unit of traded notional is kappa = 12 * cost / gamma in annual-variance
#: units (a monthly holding period; API_SPEC 3.7 maps a cost c to c / gamma).
TC_GAMMAS = (2.0, 10.0)

LOOKBACKS = (252, 504)
ESTIMATORS = ("sample", "lwcc", "lwid", "nls", "ewma", "pca3")
COSTS = (10.0, 25.0)
BASE_COST = 10.0
BANDS = (0.05, 0.10)
PARTIAL = 0.5
VOL_TARGET = 0.10
VT_MAX_GROSS = 1.5
VOL_ESTIMATORS = ("sample", "ewma")
TANGENCY_CAP = 2.0
#: RiskMetrics (1996) decay for one-month-ahead forecasts; half-life ~22.8 days.
EWMA_DECAY = 0.97
EWMA_HALF_LIFE = math.log(0.5) / math.log(EWMA_DECAY)
PCA_K = 3
BOOT_SEED = 42
BOOT_N = 1000
BOOT_BLOCK = 63.0
MAX_LOOKBACK = max(LOOKBACKS)
REBALANCE = "monthly, first trading day of each calendar month"
#: Rebalance dates on which the fast long-only QP is compared with the library's.
SOLVER_CHECK_DATES = {"ind10": 24, "ind49": 6}

MEAN_RULES = ("max_sharpe_lo", "max_sharpe_lo_bs", "tangency", "tangency_cap2")
RULE_LABELS = {
    "equal_weight": "1/N",
    "market": "market",
    "min_var_lo": "min-var",
    "erc": "ERC",
    "max_sharpe_lo": "max-Sharpe",
    "max_sharpe_lo_bs": "max-Sharpe BS",
    "tangency": "tangency",
    "tangency_cap2": "tangency cap 2",
    "min_var_lo_tc": "min-var turnover-penalised",
}
ESTIMATOR_LABELS = {
    "sample": "sample",
    "lwcc": "LW const-corr",
    "lwid": "LW identity",
    "nls": "nonlinear (2020)",
    "ewma": "EWMA 0.97",
    "pca3": "PCA 3 factors",
}


@dataclass(frozen=True)
class Universe:
    name: str
    dataset: str
    data_start: str | None
    split_years: tuple[int, ...]

    def describe(self) -> str:
        return {"ind10": "10 industries", "ind49": "49 industries"}[self.name]


UNIVERSES = {
    "ind10": Universe("ind10", "ind10_vw_daily", None, (1970, 2000)),
    "ind49": Universe("ind49", "ind49_vw_daily", "1972-01-01", (2000,)),
}


@dataclass(frozen=True)
class Config:
    """One evaluated configuration (a trial).

    ``rule`` is the weight rule; ``estimator`` its covariance estimator;
    ``lookback`` the trailing window in trading days.  ``band`` and
    ``partial`` are the rebalancing rules, ``vol_target``,
    ``vol_estimator`` and ``max_gross`` the volatility-targeting overlay,
    ``cost_bps`` the assumed cost (None for the market series).
    """

    universe: str
    lever: str
    rule: str
    estimator: str | None = None
    lookback: int | None = None
    band: float = 0.0
    partial: float = 1.0
    vol_target: float | None = None
    vol_estimator: str | None = None
    max_gross: float | None = None
    cost_bps: float | None = BASE_COST
    tc_gamma: float | None = None

    @property
    def strategy_id(self) -> str:
        parts = [self.universe, self.rule + (f".{self.estimator}" if self.estimator else "")]
        if self.lookback is not None:
            parts.append(f"L{self.lookback}")
        if self.band:
            parts.append(f"band{self.band:g}")
        if self.partial != 1.0:
            parts.append(f"partial{self.partial:g}")
        if self.vol_target is not None:
            parts.append(f"vt{self.vol_target:g}-{self.vol_estimator}-max{self.max_gross:g}")
        if self.tc_gamma is not None:
            parts.append(f"gamma{self.tc_gamma:g}")
        return "/".join(parts)

    @property
    def id(self) -> str:
        if self.cost_bps is None:
            return self.strategy_id
        return f"{self.strategy_id}/{self.cost_bps:g}bp"

    @property
    def weight_key(self) -> tuple[str, str, str | None, int | None]:
        return (self.universe, self.rule, self.estimator, self.lookback)

    @property
    def vol_key(self) -> tuple[str, str, int] | None:
        if self.vol_target is None:
            return None
        assert self.vol_estimator is not None and self.lookback is not None
        return (self.universe, self.vol_estimator, self.lookback)

    def label(self) -> str:
        rule = RULE_LABELS[self.rule]
        if self.estimator and self.rule in ("min_var_lo", "erc"):
            rule += f", {ESTIMATOR_LABELS[self.estimator]}"
        bits = [rule]
        if self.lookback is not None:
            bits.append(f"L {self.lookback}")
        if self.band:
            bits.append(f"band {self.band:g}")
        if self.partial != 1.0:
            bits.append(f"partial {self.partial:g}")
        if self.vol_target is not None:
            bits.append(
                f"vol target {self.vol_target:g} ({self.vol_estimator}, max {self.max_gross:g})"
            )
        return ", ".join(bits)

    def to_json(self) -> dict[str, Any]:
        u = UNIVERSES[self.universe]
        est_params: dict[str, Any] | None = None
        if self.estimator == "ewma" or self.vol_estimator == "ewma":
            est_params = {"ewma_decay": EWMA_DECAY, "ewma_demean": True}
        if self.estimator == "pca3":
            est_params = {"pca_k": PCA_K, "pca_specific_floor": 1e-4}
        return {
            "id": self.id,
            "universe": self.universe,
            "dataset": u.dataset,
            "data_start": u.data_start,
            "evaluation_start": f"first month start with >= {MAX_LOOKBACK} rows of history",
            "lever": self.lever,
            "rule": self.rule,
            "estimator": self.estimator,
            "estimator_params": est_params,
            "lookback_days": self.lookback,
            "band": self.band,
            "partial": self.partial,
            "vol_target": self.vol_target,
            "vol_estimator": self.vol_estimator,
            "max_gross": self.max_gross,
            "tangency_cap": TANGENCY_CAP if self.rule == "tangency_cap2" else None,
            "cost_bps": self.cost_bps,
            "rebalance": REBALANCE if self.rule != "market" else None,
            "rf": "ff3_daily RF (daily T-bill), cash and borrowing",
        } | (
            {}
            if self.tc_gamma is None
            else {
                "tc_gamma": self.tc_gamma,
                "tc_kappa": "12 * cost_bps / 1e4 / tc_gamma; w0 = drifted weights",
            }
        )


def config_from_json(doc: dict[str, Any]) -> Config:
    cfg = Config(
        universe=doc["universe"],
        lever=doc["lever"],
        rule=doc["rule"],
        estimator=doc["estimator"],
        lookback=doc["lookback_days"],
        band=float(doc["band"]),
        partial=float(doc["partial"]),
        vol_target=doc["vol_target"],
        vol_estimator=doc["vol_estimator"],
        max_gross=doc["max_gross"],
        cost_bps=doc["cost_bps"],
        tc_gamma=doc.get("tc_gamma"),
    )
    if cfg.to_json() != doc:
        raise ValueError(f"registered config {doc.get('id')} does not round-trip through Config")
    return cfg


def tc_grid() -> list[Config]:
    """Turnover-penalised minimum variance (ind10 only: the pinned split QP
    takes about 0.1 s a solve at n = 10 and about 600 s at n = 49, measured)."""
    return [
        Config("ind10", "cost_aware", "min_var_lo_tc", "lwcc", lb, cost_bps=cost, tc_gamma=g)
        for cost in COSTS
        for lb in LOOKBACKS
        for g in TC_GAMMAS
    ]


def dev_grid() -> list[Config]:
    """Every configuration evaluated in the development phase."""
    out: list[Config] = []
    for uni in ("ind10", "ind49"):
        per_cost: list[Config] = [Config(uni, "benchmark", "equal_weight")]
        for rule in ("min_var_lo", "erc"):
            for est in ESTIMATORS:
                for lb in LOOKBACKS:
                    per_cost.append(Config(uni, "estimator", rule, est, lb))
        bases = [Config(uni, "cost_aware", "equal_weight")]
        bases += [Config(uni, "cost_aware", "min_var_lo", "lwcc", lb) for lb in LOOKBACKS]
        if uni == "ind10":
            bases += [Config(uni, "cost_aware", "max_sharpe_lo", "sample", lb) for lb in LOOKBACKS]
        for base in bases:
            for band in BANDS:
                per_cost.append(replace(base, band=band))
            per_cost.append(replace(base, partial=PARTIAL))
        if uni == "ind10":
            for rule in ("tangency", "tangency_cap2"):
                for lb in LOOKBACKS:
                    per_cost.append(Config(uni, "leverage", rule, "sample", lb))
            for rule in ("max_sharpe_lo", "max_sharpe_lo_bs"):
                for lb in LOOKBACKS:
                    per_cost.append(Config(uni, "mean_shrinkage", rule, "sample", lb))
        for lb in LOOKBACKS:
            for vest in VOL_ESTIMATORS:
                for rule, est in (("equal_weight", None), ("min_var_lo", "lwcc")):
                    per_cost.append(
                        Config(
                            uni,
                            "vol_target",
                            rule,
                            est,
                            lb,
                            vol_target=VOL_TARGET,
                            vol_estimator=vest,
                            max_gross=VT_MAX_GROSS,
                        )
                    )
        for cost in COSTS:
            out.extend(replace(c, cost_bps=cost) for c in per_cost)
        out.append(Config(uni, "benchmark", "market", cost_bps=None))
    out.extend(tc_grid())
    ids = [c.id for c in out]
    if len(set(ids)) != len(ids):
        raise AssertionError("duplicate configuration ids in the development grid")
    return out


def comparator_of(cfg: Config) -> Config | None:
    """The lever's own base configuration, against which the lever is judged."""
    if cfg.rule == "market" or (cfg.lever == "benchmark"):
        return None
    if cfg.lever == "estimator":
        if cfg.estimator == "sample":
            return None
        return replace(cfg, estimator="sample", lever="estimator")
    if cfg.rule == "min_var_lo_tc":
        return replace(cfg, rule="min_var_lo", tc_gamma=None, lever="estimator")
    if cfg.lever == "cost_aware":
        base_lever = "benchmark" if cfg.rule == "equal_weight" else "estimator"
        if cfg.rule == "max_sharpe_lo":
            base_lever = "mean_shrinkage"
        return replace(cfg, band=0.0, partial=1.0, lever=base_lever)
    if cfg.lever == "leverage":
        return None if cfg.rule == "tangency" else replace(cfg, rule="tangency")
    if cfg.lever == "vol_target":
        lever = "benchmark" if cfg.rule == "equal_weight" else "estimator"
        lb = None if cfg.rule == "equal_weight" else cfg.lookback
        return replace(
            cfg, lever=lever, lookback=lb, vol_target=None, vol_estimator=None, max_gross=None
        )
    if cfg.lever == "mean_shrinkage":
        return None if cfg.rule == "max_sharpe_lo" else replace(cfg, rule="max_sharpe_lo")
    raise AssertionError(f"unknown lever {cfg.lever}")


# ------------------------------------------------------------------- data


@dataclass(frozen=True)
class Panel:
    """One universe aligned with the risk-free and market series."""

    universe: Universe
    dates: np.ndarray
    x: np.ndarray
    rf: np.ndarray
    mkt_excess: np.ndarray
    columns: tuple[str, ...]
    provenance: list[dict[str, Any]]


def _provenance(p: rd.ReturnsPanel, rows: int, first: Any, last: Any) -> dict[str, Any]:
    return {
        "dataset": p.dataset,
        "vintage_id": p.vintage_id,
        "zip_sha256": p.zip_sha256,
        "parsed_sha256": p.parsed_sha256,
        "crsp_database": p.crsp_database,
        "rows_used": rows,
        "first_date": str(first),
        "last_date": str(last),
    }


#: The data vintage the development results were computed on; phase 2 pins it.
DATA_VINTAGE = 1


def load_panel(
    uni: Universe, period: str, study_id: str | None, vintage: int | None = None
) -> Panel:
    """Load one universe and ``ff3_daily`` for ``period``.

    ``period="development"`` never passes a study id, so it cannot read
    holdout rows; any other period needs a registered ``study_id``.
    """
    if period == "development":
        assets = rd.load_returns(uni.dataset, "development", vintage=vintage)
        ff = rd.load_returns("ff3_daily", "development", vintage=vintage)
    else:
        assets = rd.load_returns(uni.dataset, period, study_id=study_id, vintage=vintage)
        ff = rd.load_returns("ff3_daily", period, study_id=study_id, vintage=vintage)
    if not np.array_equal(assets.dates, ff.dates):
        raise rd.DataError(f"{uni.dataset} and ff3_daily dates differ")
    mask = np.ones(assets.dates.shape[0], dtype=bool)
    if uni.data_start is not None:
        mask &= assets.dates >= np.datetime64(uni.data_start, "D")
    x = assets.values[mask]
    if not np.isfinite(x).all():
        raise rd.DataError(f"{uni.dataset}: missing values after {uni.data_start}")
    dates = assets.dates[mask]
    prov = [
        _provenance(assets, int(mask.sum()), dates[0], dates[-1]),
        _provenance(ff, int(mask.sum()), dates[0], dates[-1]),
    ]
    return Panel(
        uni,
        dates,
        x,
        ff.column("RF")[mask].copy(),
        ff.column("Mkt-RF")[mask].copy(),
        assets.columns,
        prov,
    )


def month_starts(dates: np.ndarray) -> list[int]:
    m = dates.astype("datetime64[M]")
    return [i for i in range(m.shape[0]) if i == 0 or m[i] != m[i - 1]]


def dev_rebalance_rows(panel: Panel) -> list[int]:
    """Month starts from the first one with ``MAX_LOOKBACK`` rows of history."""
    first = next(i for i in month_starts(panel.dates) if i >= MAX_LOOKBACK)
    month = str(panel.dates[first].astype("datetime64[M]"))
    rows = wf.calendar_rebalance_rows(panel.dates, month)
    assert rows[0] == first
    return rows


# ------------------------------------------------------------- estimation


def make_estimator(name: str) -> rs.CovarianceEstimator:
    if name == "sample":
        return rs.sample_estimator()
    if name == "lwcc":
        return rs.lw_constant_corr_estimator()
    if name == "lwid":
        return rs.lw_identity_estimator()
    if name == "nls":
        return rs.nonlinear_estimator()
    if name == "ewma":
        return rs.ewma_estimator(EWMA_HALF_LIFE)
    if name == "pca3":
        return rs.pca_estimator(PCA_K)
    raise ValueError(f"unknown estimator {name!r}")


def make_rule(rule: str, estimator: str | None) -> Callable[[np.ndarray], tuple[np.ndarray, bool]]:
    """``window -> (weights, fell_back)`` for one rule.

    Long-only minimum variance uses :func:`real_data_engine.minvar_longonly`
    (see that module); every other rule is the library's weight function.
    """
    if rule == "equal_weight":
        ew = rs.equal_weight()
        return lambda h: (ew(h), False)
    if rule == "min_var_lo":
        assert estimator is not None
        est = make_estimator(estimator)
        return lambda h: (eng.minvar_longonly(est(h)), False)
    lib: rs.WeightFunction
    if rule == "erc":
        assert estimator is not None
        lib = rs.erc_with(make_estimator(estimator))
    elif rule == "max_sharpe_lo":
        lib = rs.max_sharpe_longonly(rf=0.0, shrink=False)
    elif rule == "max_sharpe_lo_bs":
        lib = rs.max_sharpe_bayes_stein(rf=0.0, estimator=rs.sample_estimator())
    elif rule == "tangency":
        lib = rs.tangency_capped(rf=0.0, max_gross=None, shrink=False)
    elif rule == "tangency_cap2":
        lib = rs.tangency_capped(rf=0.0, max_gross=TANGENCY_CAP, shrink=False)
    else:
        raise ValueError(f"unknown rule {rule!r}")

    def fn(h: np.ndarray) -> tuple[np.ndarray, bool]:
        before = len(lib.fallback_dates)
        w = np.asarray(lib(h), dtype=float)
        return w, len(lib.fallback_dates) > before

    return fn


def window(panel_x: np.ndarray, rf: np.ndarray, t: int, lookback: int, excess: bool) -> np.ndarray:
    h = panel_x[t - lookback : t]
    if excess:
        out: np.ndarray = h - rf[t - lookback : t, None]
        return out
    return h


# Worker state: universe name -> (returns, rf), set by the pool initializer.
_WORK: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _init_worker(payload: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    _WORK.clear()
    _WORK.update(payload)


def _task_weights(
    args: tuple[tuple[str, str, str | None, int | None], list[int]],
) -> tuple[Any, list[int], np.ndarray, np.ndarray, str | None]:
    key, rows = args
    uni, rule, est, lb = key
    x, rf = _WORK[uni]
    fn = make_rule(rule, est)
    n = x.shape[1]
    out = np.zeros((len(rows), n))
    fb = np.zeros(len(rows), dtype=bool)
    try:
        for i, t in enumerate(rows):
            h = x[t - lb : t] if lb is not None else x[:t]
            if rule in MEAN_RULES:
                assert lb is not None
                h = window(x, rf, t, lb, True)
            out[i], fb[i] = fn(h)
    except (ValueError, ArithmeticError) as exc:
        return key, rows, out, fb, f"{type(exc).__name__} at row {t}: {exc}"
    return key, rows, out, fb, None


def _task_volcov(args: tuple[tuple[str, str, int], list[int]]) -> tuple[Any, list[int], np.ndarray]:
    key, rows = args
    uni, vest, lb = key
    x, _ = _WORK[uni]
    est = make_estimator(vest)
    n = x.shape[1]
    out = np.zeros((len(rows), n, n))
    for i, t in enumerate(rows):
        out[i] = est(x[t - lb : t])
    return key, rows, out


def _task_solver_check(
    args: tuple[tuple[str, str, int], list[int]],
) -> tuple[Any, int, float, float]:
    key, rows = args
    uni, est_name, lb = key
    x, _ = _WORK[uni]
    est = make_estimator(est_name)
    worst = 0.0
    worst_kkt = 0.0
    for t in rows:
        sig = est(x[t - lb : t])
        a = eng.minvar_longonly(sig)
        b = po.min_variance_longonly(sig)
        worst = max(worst, float(np.max(np.abs(a - b))))
        worst_kkt = max(worst_kkt, eng.minvar_kkt_residual(sig, a))
    return key, len(rows), worst, worst_kkt


def _lookup(rows: Sequence[int], arr: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    pos = {int(t): k for k, t in enumerate(rows)}

    def fn(history: np.ndarray) -> np.ndarray:
        out: np.ndarray = arr[pos[int(history.shape[0])]]
        return out

    return fn


class TurnoverPenalised:
    """Turnover-penalised long-only minimum variance with ``w0`` the drifted
    weights: ``portopt.min_variance_turnover(Sigma, w0, kappa)`` on the
    LW constant-correlation covariance of the trailing window.

    The rule is path dependent, so it runs inside the backtest.  It tracks
    its own last target (with band 0 and partial 1 the backtest executes it
    in full) and drifts it with the asset returns since then, renormalised
    to sum to one; the backtest's own pre-trade weights differ from this
    only by the cash that finances the trading cost (about cost x turnover
    of NAV).  The first call starts from cash, ``w0 = 0``.
    """

    def __init__(self, x: np.ndarray, lookback: int, kappa: float) -> None:
        self.x = x
        self.lookback = lookback
        self.kappa = kappa
        self.est = make_estimator("lwcc")
        self.targets: list[np.ndarray] = []
        self.last_t = -1

    def __call__(self, history: np.ndarray) -> np.ndarray:
        t = int(history.shape[0])
        n = self.x.shape[1]
        if not self.targets:
            w0 = np.zeros(n)
        else:
            growth = np.prod(1.0 + self.x[self.last_t : t], axis=0)
            v = self.targets[-1] * growth
            w0 = v / math.fsum(v.tolist())
        sig = self.est(self.x[t - self.lookback : t])
        w = np.asarray(po.min_variance_turnover(sig, w0, self.kappa), dtype=float)
        self.targets.append(w)
        self.last_t = t
        return w


def _task_backtest(
    args: tuple[Config, list[int], np.ndarray, np.ndarray | None],
) -> tuple[str, dict[str, np.ndarray] | None, str | None]:
    cfg, rows, weights, vol_cov = args
    x, rf = _WORK[cfg.universe]
    cost = float(cfg.cost_bps or 0.0)
    try:
        if cfg.rule == "min_var_lo_tc":
            assert cfg.tc_gamma is not None and cfg.lookback is not None
            fn = TurnoverPenalised(x, cfg.lookback, 12.0 * cost / 1e4 / cfg.tc_gamma)
            res = wf.run_schedule_net(x, fn, rows, cost_bps=cost, rf_daily=rf)
            weights = np.array(fn.targets)
            scales = np.ones(len(rows))
            tgt_gross = np.ones(len(rows))
        elif cfg.vol_target is None:
            res = wf.run_schedule_net(
                x,
                _lookup(rows, weights),
                rows,
                cost_bps=cost,
                band=cfg.band,
                partial=cfg.partial,
                rf_daily=rf,
            )
            scales = np.ones(len(rows))
            tgt_gross = np.array([po.gross_leverage(w) for w in weights])
        else:
            assert vol_cov is not None
            sres = eng.run_schedule_scaled(
                x,
                _lookup(rows, weights),
                rows,
                cost_bps=cost,
                band=cfg.band,
                partial=cfg.partial,
                rf_daily=rf,
                vol_target=cfg.vol_target,
                max_gross=cfg.max_gross,
                vol_estimator=_lookup(rows, vol_cov),
            )
            res = sres.result
            scales = sres.scales
            tgt_gross = np.array([po.gross_leverage(w) for w in sres.targets])
    except ValueError as exc:
        return cfg.id, None, f"backtest: {exc}"
    return (
        cfg.id,
        {
            "net": res.net_returns,
            "gross": res.gross_returns,
            "costs": res.costs,
            "turnovers": res.turnovers,
            "executed": res.executed,
            "scales": scales,
            "target_gross": tgt_gross,
        },
        None,
    )


def _chunks(rows: list[int], size: int) -> list[list[int]]:
    return [rows[i : i + size] for i in range(0, len(rows), size)]


def _chunk_size(key: tuple[str, str, str | None, int | None]) -> int:
    uni, rule, _, _ = key
    if rule in ("max_sharpe_lo", "max_sharpe_lo_bs"):
        return 40
    if uni == "ind49" and rule == "erc":
        return 60
    return 400


# ------------------------------------------------------------- statistics

_N01 = NormalDist()


@dataclass
class Stats:
    n_days: int
    net_sharpe: float
    gross_sharpe: float
    se_iid: float
    se_mertens: float
    psr0: float
    ann_return: float
    ann_excess: float
    ann_vol: float
    max_drawdown: float
    cost_per_year: float
    turnover_per_year: float
    n_rebalances: int
    n_trades: int
    mean_scale: float
    max_target_gross: float
    mean_target_gross: float
    n_fallbacks: int
    boot_lo: float = math.nan
    boot_hi: float = math.nan
    boot_se: float = math.nan
    dsr: float = math.nan


@dataclass
class Series:
    """Daily results of one configuration over an evaluation period."""

    cfg: Config
    dates: np.ndarray
    net: np.ndarray
    gross: np.ndarray
    excess: np.ndarray
    costs: np.ndarray
    reb_dates: np.ndarray
    turnovers: np.ndarray
    executed: np.ndarray
    scales: np.ndarray
    target_gross: np.ndarray
    fallbacks: np.ndarray


def ann_sr(x: np.ndarray) -> float:
    return rs.annualize_sharpe(rs.sharpe_per_period(x))


def compute_stats(s: Series, mask: np.ndarray, reb_mask: np.ndarray) -> Stats:
    ex = s.excess[mask]
    net = s.net[mask]
    gross_ex = (s.gross - (s.net - s.excess))[mask]
    tov = s.turnovers[reb_mask]
    # Turnover per year excludes the initial build (index 0 of the full run).
    later = tov[1:] if reb_mask.size and reb_mask[0] else tov
    n_months = int(reb_mask.sum())
    return Stats(
        n_days=int(mask.sum()),
        net_sharpe=ann_sr(ex),
        gross_sharpe=ann_sr(gross_ex),
        se_iid=rs.annualize_sharpe(rs.sharpe_se_iid(ex)),
        se_mertens=rs.annualize_sharpe(rs.sharpe_se_nonnormal(ex)),
        psr0=rs.probabilistic_sharpe_ratio(ex, 0.0),
        ann_return=po.annualized_return(net),
        ann_excess=po.annualized_return(ex),
        ann_vol=po.annualized_vol(ex),
        max_drawdown=po.max_drawdown(net),
        cost_per_year=252.0 * math.fsum(s.costs[mask].tolist()) / int(mask.sum()),
        turnover_per_year=(12.0 * math.fsum(later.tolist()) / later.size) if later.size else 0.0,
        n_rebalances=n_months,
        n_trades=int(s.executed[reb_mask].sum()),
        mean_scale=float(np.mean(s.scales[reb_mask])) if n_months else math.nan,
        max_target_gross=float(np.max(s.target_gross[reb_mask])) if n_months else math.nan,
        mean_target_gross=float(np.mean(s.target_gross[reb_mask])) if n_months else math.nan,
        n_fallbacks=int(s.fallbacks[reb_mask].sum()),
    )


@dataclass
class Paired:
    diff: float
    se: float
    z: float
    p_two: float
    p_one: float


def paired(a: np.ndarray, b: np.ndarray) -> Paired:
    """Ledoit-Wolf (2008) HAC test of SR(a) - SR(b), annualized.

    ``p_one`` is the one-sided p-value for SR(a) > SR(b).
    """
    t = rs.sharpe_difference_test(a, b)
    p_one = 1.0 - _N01.cdf(t.z)
    return Paired(
        rs.annualize_sharpe(t.diff), rs.annualize_sharpe(t.std_error), t.z, t.p_value, p_one
    )


# ------------------------------------------------------------------ runner


@dataclass
class RunOutput:
    panels: dict[str, Panel]
    series: dict[str, Series]
    excluded: dict[str, str]
    solver_checks: list[tuple[Any, int, float, float]]
    self_check_log: list[str]
    timings: dict[str, float]


def _market_series(panel: Panel, start: int, end: int, cfg: Config) -> Series:
    ex = panel.mkt_excess[start:end]
    rfs = panel.rf[start:end]
    z = np.zeros(end - start)
    return Series(
        cfg,
        panel.dates[start:end],
        ex + rfs,
        ex + rfs,
        ex.copy(),
        z,
        np.array([], dtype="datetime64[D]"),
        np.array([]),
        np.array([], dtype=bool),
        np.array([]),
        np.array([]),
        np.array([], dtype=bool),
    )


def run_configs(
    configs: list[Config],
    panels: dict[str, Panel],
    rows_by_uni: dict[str, list[int]],
    *,
    workers: int,
    check_solver: bool,
    log: Callable[[str], None],
) -> RunOutput:
    timings: dict[str, float] = {}
    payload = {u: (p.x, p.rf) for u, p in panels.items()}
    strategies = [c for c in configs if c.rule != "market"]
    wkeys = sorted({c.weight_key for c in strategies if c.rule != "min_var_lo_tc"}, key=str)
    vkeys = sorted({k for c in strategies if (k := c.vol_key) is not None}, key=str)
    weights: dict[Any, np.ndarray] = {}
    fallbacks: dict[Any, np.ndarray] = {}
    errors: dict[Any, str] = {}
    vol_cov: dict[Any, np.ndarray] = {}
    solver_checks: list[tuple[Any, int, float, float]] = []
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(payload,)
    ) as pool:
        t0 = time.perf_counter()
        w_tasks = [
            (key, chunk)
            for key in wkeys
            for chunk in _chunks(rows_by_uni[key[0]], _chunk_size(key))
        ]
        # Slow tasks first so the pool stays busy to the end.
        w_tasks.sort(key=lambda a: -_task_cost(a[0], len(a[1])))
        parts: dict[Any, list[tuple[list[int], np.ndarray, np.ndarray]]] = {}
        for key, rows, out, fb, err in pool.map(_task_weights, w_tasks):
            parts.setdefault(key, []).append((rows, out, fb))
            if err is not None:
                errors[key] = err
        for key, chunks in parts.items():
            chunks.sort(key=lambda c: c[0][0])
            weights[key] = np.vstack([c[1] for c in chunks])
            fallbacks[key] = np.concatenate([c[2] for c in chunks])
        timings["weights"] = time.perf_counter() - t0
        log(f"  weights: {len(wkeys)} rule paths in {timings['weights']:.0f} s")
        t0 = time.perf_counter()
        v_tasks = [(key, chunk) for key in vkeys for chunk in _chunks(rows_by_uni[key[0]], 400)]
        vparts: dict[Any, list[tuple[list[int], np.ndarray]]] = {}
        for key, rows, out in pool.map(_task_volcov, v_tasks):
            vparts.setdefault(key, []).append((rows, out))
        for key, chunks in vparts.items():
            chunks.sort(key=lambda c: c[0][0])
            vol_cov[key] = np.concatenate([c[1] for c in chunks])
        timings["vol_cov"] = time.perf_counter() - t0
        if check_solver:
            t0 = time.perf_counter()
            c_tasks = []
            for uni, rows in rows_by_uni.items():
                k = SOLVER_CHECK_DATES[uni]
                pick = [rows[int(round(i * (len(rows) - 1) / (k - 1)))] for i in range(k)]
                for est in ESTIMATORS:
                    for lb in LOOKBACKS:
                        if any(c.weight_key == (uni, "min_var_lo", est, lb) for c in strategies):
                            c_tasks.extend(((uni, est, lb), [t]) for t in pick)
            agg: dict[Any, list[float]] = {}
            for key, n_rows, worst, kkt in pool.map(_task_solver_check, c_tasks):
                a = agg.setdefault(key, [0.0, 0.0, 0.0])
                a[0] += n_rows
                a[1] = max(a[1], worst)
                a[2] = max(a[2], kkt)
            solver_checks = [(k, int(v[0]), v[1], v[2]) for k, v in sorted(agg.items(), key=str)]
            timings["solver_check"] = time.perf_counter() - t0
            worst_all = max((c[2] for c in solver_checks), default=0.0)
            log(
                f"  solver check: {sum(c[1] for c in solver_checks)} dates, max |w_fast - w_lib| "
                f"{worst_all:.2e}, {timings['solver_check']:.0f} s"
            )
            if worst_all > 1e-8:
                raise AssertionError(f"fast long-only QP differs from the library by {worst_all}")
        t0 = time.perf_counter()
        b_tasks = []
        excluded: dict[str, str] = {}
        for c in strategies:
            if c.weight_key in errors:
                excluded[c.id] = errors[c.weight_key]
                continue
            vk = c.vol_key
            b_tasks.append(
                (
                    c,
                    rows_by_uni[c.universe],
                    weights.get(c.weight_key, np.zeros((0, 0))),
                    None if vk is None else vol_cov[vk],
                )
            )
        # The path-dependent turnover-penalised rule is the slowest task.
        b_tasks.sort(key=lambda a: a[0].rule != "min_var_lo_tc")
        bt: dict[str, dict[str, np.ndarray]] = {}
        for cid, arrs, err in pool.map(_task_backtest, b_tasks):
            if arrs is None:
                excluded[cid] = str(err)
            else:
                bt[cid] = arrs
        timings["backtests"] = time.perf_counter() - t0
        log(f"  backtests: {len(b_tasks)} configurations in {timings['backtests']:.0f} s")
    series: dict[str, Series] = {}
    for c in configs:
        panel = panels[c.universe]
        rows = rows_by_uni[c.universe]
        start, end = rows[0], panel.x.shape[0]
        if c.rule == "market":
            series[c.id] = _market_series(panel, start, end, c)
            continue
        if c.id not in bt:
            continue
        a = bt[c.id]
        series[c.id] = Series(
            c,
            panel.dates[start:end],
            a["net"],
            a["gross"],
            a["net"] - panel.rf[start:end],
            a["costs"],
            panel.dates[rows],
            a["turnovers"],
            a["executed"],
            a["scales"],
            a["target_gross"],
            fallbacks.get(c.weight_key, np.zeros(len(rows), dtype=bool)),
        )
    t0 = time.perf_counter()
    first = panels[next(iter(panels))]
    lb = LOOKBACKS[0]
    sl = slice(0, min(first.x.shape[0], 4000))
    vol_est = make_estimator("ewma")
    self_log = eng.self_check(first.x[sl], first.rf[sl], lambda h: vol_est(h[-lb:]), lookback=lb)
    timings["self_check"] = time.perf_counter() - t0
    return RunOutput(panels, series, excluded, solver_checks, self_log, timings)


def _task_cost(key: tuple[str, str, str | None, int | None], n_rows: int) -> float:
    uni, rule, _, _ = key
    per = 1.0
    if rule in ("max_sharpe_lo", "max_sharpe_lo_bs"):
        per = 300.0
    elif rule == "erc":
        per = 60.0 if uni == "ind49" else 5.0
    elif rule.startswith("tangency"):
        per = 5.0
    elif rule == "min_var_lo":
        per = 10.0 if uni == "ind49" else 3.0
    return per * n_rows


# ----------------------------------------------------------- period stats


@dataclass
class Row:
    cfg: Config
    period: str
    start: str
    end: str
    stats: Stats
    vs_1n: Paired | None
    vs_market: Paired | None
    comparator: str
    vs_comp: Paired | None


def periods_for(
    panel: Panel, start_row: int, phase: str
) -> list[tuple[str, np.datetime64, np.datetime64]]:
    d0 = panel.dates[start_row]
    d1 = panel.dates[-1]
    out = [("full", d0, d1)]
    if phase != "dev":
        return out
    splits = [np.datetime64(f"{y}-01-01", "D") for y in panel.universe.split_years]
    edges = [d0] + [e for e in splits if d0 < e <= d1]
    for i, lo in enumerate(edges):
        hi = (edges[i + 1] - np.timedelta64(1, "D")) if i + 1 < len(edges) else d1
        lo_y = int(str(lo)[:4])
        hi_y = int(str(hi)[:4])
        out.append((f"{lo_y}-{hi_y}", lo, hi))
    return out


def benchmark_id(cfg: Config, rule: str) -> str:
    """Benchmark series a configuration is paired with: the market, or 1/N at
    the configuration's own cost (the base cost for the market row)."""
    if rule == "market":
        return Config(cfg.universe, "benchmark", "market", cost_bps=None).id
    cost = BASE_COST if cfg.cost_bps is None else cfg.cost_bps
    return Config(cfg.universe, "benchmark", "equal_weight", cost_bps=cost).id


def build_rows(
    run: RunOutput, phase: str, n_trials_total: int, log: Callable[[str], None]
) -> list[Row]:
    rows: list[Row] = []
    by_uni: dict[str, list[Series]] = {}
    for s in run.series.values():
        by_uni.setdefault(s.cfg.universe, []).append(s)
    t0 = time.perf_counter()
    for uni, items in by_uni.items():
        panel = run.panels[uni]
        srs = np.array([rs.sharpe_per_period(s.excess) for s in items])
        sr_var = float(np.var(srs, ddof=1)) if srs.size > 1 else 0.0
        start_row = int(np.searchsorted(panel.dates, items[0].dates[0]))
        for s in items:
            for label, lo, hi in periods_for(panel, start_row, phase):
                mask = (s.dates >= lo) & (s.dates <= hi)
                if s.reb_dates.size:
                    reb_mask = (s.reb_dates >= lo) & (s.reb_dates <= hi)
                else:
                    reb_mask = np.zeros(0, dtype=bool)
                st = compute_stats(s, mask, reb_mask)
                if label == "full":
                    st.dsr = rs.deflated_sharpe_ratio(
                        s.excess, n_trials=n_trials_total, sr_variance=sr_var
                    )
                    if s.cfg.cost_bps in (BASE_COST, None):
                        b = rs.sharpe_bootstrap_ci(
                            s.excess, seed=BOOT_SEED, n_boot=BOOT_N, block_length=BOOT_BLOCK
                        )
                        st.boot_lo = rs.annualize_sharpe(b.lower)
                        st.boot_hi = rs.annualize_sharpe(b.upper)
                        st.boot_se = rs.annualize_sharpe(b.std_error)
                vs = {}
                for rule in ("equal_weight", "market"):
                    bid = benchmark_id(s.cfg, rule)
                    other = run.series.get(bid)
                    if other is None or bid == s.cfg.id:
                        vs[rule] = None
                    else:
                        vs[rule] = paired(s.excess[mask], other.excess[mask])
                comp = comparator_of(s.cfg)
                comp_id = comp.id if comp is not None else ""
                vs_comp = None
                if comp is not None and comp.id in run.series:
                    vs_comp = paired(s.excess[mask], run.series[comp.id].excess[mask])
                rows.append(
                    Row(
                        s.cfg,
                        label,
                        str(lo),
                        str(hi),
                        st,
                        vs["equal_weight"],
                        vs["market"],
                        comp_id,
                        vs_comp,
                    )
                )
    log(f"  statistics: {len(rows)} rows in {time.perf_counter() - t0:.0f} s")
    return rows


# ------------------------------------------------------------------ output


def fmt(v: float | None, nd: int = 2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    out = f"{v:.{nd}f}"
    return "0." + "0" * nd if out == "-0." + "0" * nd else out


CSV_COLUMNS = [
    "config_id",
    "strategy_id",
    "universe",
    "lever",
    "rule",
    "estimator",
    "lookback_days",
    "band",
    "partial",
    "vol_target",
    "vol_estimator",
    "max_gross",
    "cost_bps",
    "period",
    "start",
    "end",
    "n_days",
    "net_sharpe",
    "gross_sharpe",
    "se_lo_iid",
    "se_mertens",
    "boot_ci_lo",
    "boot_ci_hi",
    "boot_se",
    "psr_vs_0",
    "dsr",
    "ann_return",
    "ann_excess_return",
    "ann_vol",
    "max_drawdown",
    "turnover_per_year",
    "cost_per_year",
    "n_rebalances",
    "n_trades",
    "mean_scale",
    "mean_target_gross",
    "max_target_gross",
    "n_fallbacks",
    "diff_vs_equal_weight",
    "se_diff_vs_equal_weight",
    "p_two_vs_equal_weight",
    "p_one_vs_equal_weight",
    "diff_vs_market",
    "p_two_vs_market",
    "comparator",
    "diff_vs_comparator",
    "se_diff_vs_comparator",
    "p_two_vs_comparator",
    "p_one_vs_comparator",
]


def _opt(v: float | None) -> str:
    return "" if v is None else f"{v:g}"


def write_csv(rows: list[Row], path: Path) -> None:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CSV_COLUMNS)
    for r in rows:
        c, st = r.cfg, r.stats
        p1, pm, pc = r.vs_1n, r.vs_market, r.vs_comp
        w.writerow(
            [
                c.id,
                c.strategy_id,
                c.universe,
                c.lever,
                c.rule,
                c.estimator or "",
                "" if c.lookback is None else c.lookback,
                f"{c.band:g}",
                f"{c.partial:g}",
                _opt(c.vol_target),
                c.vol_estimator or "",
                _opt(c.max_gross),
                _opt(c.cost_bps),
                r.period,
                r.start,
                r.end,
                st.n_days,
                fmt(st.net_sharpe, 6),
                fmt(st.gross_sharpe, 6),
                fmt(st.se_iid, 6),
                fmt(st.se_mertens, 6),
                fmt(st.boot_lo, 6),
                fmt(st.boot_hi, 6),
                fmt(st.boot_se, 6),
                fmt(st.psr0, 6),
                fmt(st.dsr, 6),
                fmt(st.ann_return, 6),
                fmt(st.ann_excess, 6),
                fmt(st.ann_vol, 6),
                fmt(st.max_drawdown, 6),
                fmt(st.turnover_per_year, 6) if c.rule != "market" else "",
                fmt(st.cost_per_year, 6) if c.rule != "market" else "",
                st.n_rebalances if c.rule != "market" else "",
                st.n_trades if c.rule != "market" else "",
                fmt(st.mean_scale, 6),
                fmt(st.mean_target_gross, 6),
                fmt(st.max_target_gross, 6),
                st.n_fallbacks if c.rule != "market" else "",
                fmt(p1.diff, 6) if p1 else "",
                fmt(p1.se, 6) if p1 else "",
                fmt(p1.p_two, 6) if p1 else "",
                fmt(p1.p_one, 6) if p1 else "",
                fmt(pm.diff, 6) if pm else "",
                fmt(pm.p_two, 6) if pm else "",
                r.comparator,
                fmt(pc.diff, 6) if pc else "",
                fmt(pc.se, 6) if pc else "",
                fmt(pc.p_two, 6) if pc else "",
                fmt(pc.p_one, 6) if pc else "",
            ]
        )
    path.write_text(buf.getvalue(), encoding="utf-8", newline="\n")


def table(header: list[str], body: list[list[str]], align: str | None = None) -> str:
    al = align or ("l" + "r" * (len(header) - 1))
    sep = ["---:" if a == "r" else "---" for a in al]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(sep) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


# ---------------------------------------------------------------- phase dev


def dev_n_trials() -> int:
    return sum(wf.n_trials(sid) for sid in DEV_STUDY_IDS)


def register_dev(configs: list[Config]) -> dict[str, Any]:
    """Register both development batches; return the first batch's entry."""
    entry = wf.register_study(
        DEV_STUDY_ID,
        "Round 2 development exploration on 1926-07-01..2015-12-31 (period='development' "
        "only): every configuration evaluated by python/studies/real_data_study.py --phase dev, "
        "registered before the first run so that n_trials counts them all.",
        [c.to_json() for c in configs if c.tc_gamma is None],
    )
    wf.register_study(
        DEV_TC_STUDY_ID,
        "Round 2 development exploration, second batch: turnover-penalised long-only minimum "
        "variance (portopt.min_variance_turnover, ready after the first batch was registered), "
        "ind10 only, period='development' only.",
        [c.to_json() for c in configs if c.tc_gamma is not None],
    )
    return entry


def dev_panels() -> tuple[dict[str, Panel], dict[str, list[int]]]:
    """Development panels (``period="development"`` only) and their schedules."""
    panels = {u: load_panel(UNIVERSES[u], "development", None, DATA_VINTAGE) for u in UNIVERSES}
    rows_by_uni = {u: dev_rebalance_rows(p) for u, p in panels.items()}
    return panels, rows_by_uni
