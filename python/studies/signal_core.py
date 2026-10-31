"""Configurations, backtests and statistics for ``signal_study.py``.

Reuses the data loading and the paired test of ``real_data_core`` (read,
not edited).  Every weight is a function of the rows strictly before the
rebalance row: ``run_schedule_net`` passes rows ``[0, t)`` and the rules
below slice nothing after that.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import portopt as po
from portopt import research as rs
from portopt.research import signals as sg
from portopt.research import walkforward as wf

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import real_data_core as rc  # noqa: E402
import real_data_engine as eng  # noqa: E402

ROOT = HERE.parents[1]
OUT_DIR = ROOT / "docs" / "results"

DEV_STUDY_ID = "r2-signal-dev"
SEEN_STUDY_ID = "r2-signal-v1"
FORWARD_STUDY_ID = "r2-forward-signals-v1"
CONSTRUCTION_TRIALS_STUDIES = ("r2-dev-exploration", "r2-dev-exploration-tc")

COSTS = (10.0, 25.0)
LOOKBACK = 504
IC = 0.05
GAMMA = 5.0
BL_DELTA = 2.5
BL_TAU = 0.05
BL_CONF = 1.0
COMBO = ("mom12_1", "tsmom", "strev", "lowbeta")
SIGNAL_KEYS = ("mom12_1", "tsmom", "strev", "lowvol", "lowbeta", "combo_eq", "combo_iv")
#: Publication year of each signal family, for the McLean and Pontiff (2016)
#: before/after split: Jegadeesh and Titman (1993); Moskowitz, Ooi and
#: Pedersen (2012); Jegadeesh (1990); Baker, Bradley and Wurgler (2011);
#: Black, Jensen and Scholes (1972) for the flat beta-return relation.
PUBLICATION = {"mom12_1": 1993, "tsmom": 2012, "strev": 1990, "lowvol": 2011, "lowbeta": 1972}
SEEN_START = "2016-01"
SUBPERIODS = (("pre-1963", None, "1963-01-01"), ("1963-1990", "1963-01-01", "1991-01-01"),
              ("1991-2015", "1991-01-01", "2016-01-01"))


def signal_spec(key: str) -> tuple[tuple[str, ...], str]:
    if key == "combo_eq":
        return COMBO, "equal"
    if key == "combo_iv":
        return COMBO, "inverse_vol"
    return (key,), "equal"


@dataclass(frozen=True)
class SConfig:
    universe: str
    rule: str  # equal_weight | market | minvar | rank | mvtc | bl
    signal: str | None = None
    cost_bps: float | None = 10.0

    @property
    def id(self) -> str:
        r = self.rule + (f".{self.signal}" if self.signal else "")
        if self.rule == "minvar":
            r += ".lwcc/L504"
        return f"{self.universe}/{r}" + ("" if self.cost_bps is None else f"/{self.cost_bps:g}bp")

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id, "universe": self.universe, "rule": self.rule,
                             "signal": self.signal, "cost_bps": self.cost_bps,
                             "rebalance": rc.REBALANCE, "month_rows": sg.MONTH}
        if self.signal:
            names, method = signal_spec(self.signal)
            d |= {"signals": list(names), "combine": method}
        if self.rule in ("mvtc", "bl", "minvar"):
            d |= {"cov": "lwcc", "lookback_days": LOOKBACK}
        if self.rule == "mvtc":
            d |= {"ic": IC, "risk_aversion": GAMMA, "kappa": "12 * cost / gamma; w0 drifted"}
        if self.rule == "bl":
            d |= {"ic": IC, "delta": BL_DELTA, "tau": BL_TAU, "confidence": BL_CONF,
                  "equilibrium": "1/N"}
        if self.rule == "rank":
            d |= {"portfolio": "equal weight on floor(n/3) highest combined z-scores"}
        return d

    def label(self) -> str:
        base = {"equal_weight": "1/N", "market": "market", "minvar": "min-var LW-CC L504",
                "rank": "top tercile", "mvtc": "MV + turnover", "bl": "BL views"}[self.rule]
        s = f"{base} {self.signal}" if self.signal else base
        return s + ("" if self.cost_bps in (None, 10.0) else f" ({self.cost_bps:g} bp)")


def dev_grid() -> list[SConfig]:
    out: list[SConfig] = []
    for u in ("ind10", "ind49"):
        out.append(SConfig(u, "market", cost_bps=None))
        for c in COSTS:
            out += [SConfig(u, "equal_weight", cost_bps=c), SConfig(u, "minvar", cost_bps=c)]
            out += [SConfig(u, "rank", s, c) for s in SIGNAL_KEYS]
            if u == "ind10":  # QP rules on 10 industries only (runtime)
                out += [SConfig(u, "mvtc", s, c) for s in SIGNAL_KEYS]
                out += [SConfig(u, "bl", s, c) for s in SIGNAL_KEYS]
    return out


def config_from_id(cid: str) -> SConfig:
    for c in dev_grid():
        if c.id == cid:
            return c
    raise KeyError(cid)


# ------------------------------------------------------------------ rules


def make_weight_fn(cfg: SConfig, x: np.ndarray, rf: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """``history -> weights``; ``x`` and ``rf`` are used only through the
    length of the history passed in (rows strictly before the rebalance)."""
    if cfg.rule == "equal_weight":
        n = x.shape[1]
        return lambda h: np.full(n, 1.0 / n)
    est = rs.lw_constant_corr_estimator()
    if cfg.rule == "minvar":
        return lambda h: eng.minvar_longonly(est(h[-LOOKBACK:]))
    assert cfg.signal is not None
    names, method = signal_spec(cfg.signal)
    if cfg.rule == "mvtc":
        return sg.SignalMeanVariance(signals=names, method=method, ic=IC, risk_aversion=GAMMA,
                                     cost_bps=float(cfg.cost_bps or 0.0), lookback=LOOKBACK,
                                     rf_daily=rf)

    def fn(h: np.ndarray) -> np.ndarray:
        t = h.shape[0]
        score = sg.combine(names, h, method=method, rf_daily=rf[:t])
        if cfg.rule == "rank":
            return sg.top_fraction_weights(score)
        return sg.bl_signal_weights(est(h[-LOOKBACK:]), score, ic=IC, delta=BL_DELTA,
                                    tau=BL_TAU, confidence=BL_CONF)

    return fn


# ----------------------------------------------------------------- running

_WORK: dict[str, rc.Panel] = {}


def _init(panels: dict[str, rc.Panel]) -> None:
    import os

    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    _WORK.update(panels)


def rebalance_rows(panel: rc.Panel, start_month: str | None) -> list[int]:
    if start_month is None:
        return rc.dev_rebalance_rows(panel)
    return wf.calendar_rebalance_rows(panel.dates, start_month)


@dataclass
class Result:
    cfg_id: str
    start: int
    excess: np.ndarray
    net: np.ndarray
    turnover_per_year: float
    cost_per_year: float


def _run(args: tuple[SConfig, str | None]) -> Result:
    cfg, start_month = args
    p = _WORK[cfg.universe]
    rows = rebalance_rows(p, start_month)
    s = rows[0]
    if cfg.rule == "market":
        ex = p.mkt_excess[s:].copy()
        return Result(cfg.id, s, ex, ex + p.rf[s:], 0.0, 0.0)
    fn = make_weight_fn(cfg, p.x, p.rf)
    res = wf.run_schedule_net(p.x, fn, rows, cost_bps=float(cfg.cost_bps or 0.0),
                              rf_daily=p.rf)
    net = np.asarray(res.net_returns)
    tov = np.asarray(res.turnovers)[1:]
    cost = np.asarray(res.costs)
    return Result(cfg.id, s, net - p.rf[s:], net, 12.0 * float(tov.mean()),
                  252.0 * float(cost.sum()) / net.size)


def run_all(panels: dict[str, rc.Panel], cfgs: list[SConfig], start_month: str | None,
            workers: int = 12) -> dict[str, Result]:
    # Slow rules first so the pool stays busy.
    order = sorted(cfgs, key=lambda c: c.rule not in ("mvtc", "bl"))
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(panels,)) as ex:
        out = list(ex.map(_run, [(c, start_month) for c in order]))
    return {r.cfg_id: r for r in out}


# ------------------------------------------------------------------- stats


def summarize(r: Result, mask: np.ndarray) -> dict[str, float]:
    ex = r.excess[mask]
    net = r.net[mask]
    return {
        "n_days": float(mask.sum()),
        "sharpe": rc.ann_sr(ex),
        "se": rs.annualize_sharpe(rs.sharpe_se_nonnormal(ex)),
        "ann_excess": po.annualized_return(ex),
        "ann_vol": po.annualized_vol(ex),
        "max_dd": po.max_drawdown(net),
        "turnover": r.turnover_per_year,
        "cost": r.cost_per_year,
    }


def diff(a: Result, b: Result, mask: np.ndarray) -> rc.Paired:
    return rc.paired(a.excess[mask], b.excess[mask])


def date_mask(dates: np.ndarray, lo: str | None, hi: str | None) -> np.ndarray:
    m = np.ones(dates.shape[0], dtype=bool)
    if lo:
        m &= dates >= np.datetime64(lo, "D")
    if hi:
        m &= dates < np.datetime64(hi, "D")
    return m


def holm(ps: list[float]) -> list[float]:
    k = len(ps)
    order = sorted(range(k), key=lambda i: ps[i])
    adj = [0.0] * k
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (k - rank) * ps[i]))
        adj[i] = run
    return adj


def dsr(r: Result, sr_var_pp: float, n: int) -> float:
    return rs.deflated_sharpe_ratio(r.excess, n_trials=n, sr_variance=sr_var_pp)


def construction_trials() -> int:
    return sum(wf.n_trials(s) for s in CONSTRUCTION_TRIALS_STUDIES)


def per_period(sr_annual: float) -> float:
    return sr_annual / math.sqrt(252.0)
