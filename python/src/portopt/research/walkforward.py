"""Trial registry, walk-forward evaluation and calendar-scheduled backtests.

Part of the Python-only research layer (not covered by API_SPEC.md).

Trial registry.  :func:`register_study` appends a study (an id, a
description and the full list of configurations it will try) to
``data/real/registry.json`` with a UTC timestamp and the SHA-256 of the
canonical JSON of the configuration list.  :func:`n_trials` then gives the
honest trial count for :func:`portopt.research.deflated_sharpe_ratio`.
A study id is registered once: re-registering the same id with the same
configurations is a no-op, with different ones it is an error.  Only a
registered study may read the holdout (:func:`portopt.research.data.load_returns`).

Walk-forward.  :func:`walk_forward` splits rows ``[train, T)`` into
consecutive, non-overlapping test folds of ``test`` rows (a final shorter
fold is kept if it has at least 2 rows).  Fold ``k`` with test rows
``[s, e)`` is run through :func:`portopt.research.run_backtest_net` on a
slice that ends at ``e``:

* ``mode="rolling"``: the slice starts at ``s - train`` and the weight
  function always sees only the trailing ``train`` rows of its history;
* ``mode="expanding"``: the slice starts at row 0 and the weight function
  sees all rows before each rebalance.

Each fold is an independent paper portfolio built from cash, so every
fold pays its initial build cost.  The pooled result concatenates the
folds' daily returns, which together cover ``[train, T)`` exactly once.

Calendar schedules.  :func:`run_schedule_net` applies the accounting of
:mod:`portopt.research.costs` (drifting weights, cost on traded risky
notional, no-trade band, partial rebalancing) to an arbitrary list of
rebalance rows and a per-row risk-free series, which the fixed-step
engine cannot express (for example the first trading day of each month).
On a regular schedule with a constant risk-free rate it reproduces
``run_backtest_net`` (tested).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from portopt import annualized_return, annualized_vol, max_drawdown, sharpe_ratio, validate_returns

from . import data as _data
from .costs import ANNUALIZATION, NetBacktestResult, run_backtest_net

__all__ = [
    "Fold",
    "ScheduleResult",
    "WalkForwardResult",
    "calendar_rebalance_rows",
    "configs_sha256",
    "n_trials",
    "register_study",
    "run_schedule_net",
    "walk_forward",
]

WeightFn = Callable[[np.ndarray], np.ndarray]
_STUDY_ID_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789-_.")


# ------------------------------------------------------------------ registry


def configs_sha256(configs: Sequence[Any]) -> str:
    """SHA-256 of the canonical JSON (sorted keys, no spaces) of ``configs``."""
    text = json.dumps(list(configs), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def register_study(
    study_id: str,
    description: str,
    configs: Sequence[Any],
    *,
    data_dir: str | os.PathLike[str] | None = None,
    now: _dt.datetime | None = None,
) -> dict[str, Any]:
    """Append a study to ``registry.json`` and return its record.

    ``configs`` is the complete list of configurations the study will
    evaluate (JSON-serializable, at least one).  Its length is the trial
    count used by :func:`n_trials`.  ``study_id`` uses lower-case letters,
    digits and ``-_.`` (3 to 64 characters).
    """
    if not (
        isinstance(study_id, str)
        and 3 <= len(study_id) <= 64
        and set(study_id) <= _STUDY_ID_CHARS
        and study_id[0].isalnum()
    ):
        raise ValueError(f"invalid study id {study_id!r}")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("description must be a non-empty string")
    cfgs = list(configs)
    if not cfgs:
        raise ValueError("configs must list at least one configuration")
    try:
        digest = configs_sha256(cfgs)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"configs must be JSON-serializable without NaN: {exc}") from None
    root = _data.default_data_dir() if data_dir is None else os.fspath(data_dir)
    registry = _data.load_registry(root)
    for entry in registry["studies"]:
        if entry.get("id") == study_id:
            if entry.get("configs_sha256") == digest:
                out: dict[str, Any] = entry
                return out
            raise ValueError(
                f"study {study_id!r} is already registered with different configurations; "
                "register a new id instead of editing a registered study"
            )
    t = _dt.datetime.now(_dt.UTC) if now is None else now.astimezone(_dt.UTC)
    entry = {
        "id": study_id,
        "description": description.strip(),
        "registered_at_utc": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_configs": len(cfgs),
        "configs_sha256": digest,
        "configs": json.loads(json.dumps(cfgs, sort_keys=True)),
    }
    registry["studies"].append(entry)
    text = json.dumps(registry, indent=2, ensure_ascii=True) + "\n"
    path = os.path.join(root, "registry.json")
    os.makedirs(root, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    os.replace(tmp, path)
    return entry


def n_trials(study_id: str | None = None, *, data_dir: str | os.PathLike[str] | None = None) -> int:
    """Configurations registered by one study, or by all studies when None."""
    studies = _data.load_registry(data_dir)["studies"]
    if study_id is None:
        return sum(int(s["n_configs"]) for s in studies)
    for s in studies:
        if s.get("id") == study_id:
            return int(s["n_configs"])
    raise KeyError(f"study {study_id!r} is not registered")


# -------------------------------------------------------------- walk-forward


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold: estimation rows ``[train_start, test_start)``
    (for rolling mode; expanding uses ``[0, test_start)``) and test rows
    ``[test_start, test_end)`` of the input, with its backtest result."""

    index: int
    train_start: int
    test_start: int
    test_end: int
    result: NetBacktestResult


@dataclass(frozen=True)
class WalkForwardResult:
    """Per-fold results and the pooled out-of-sample series.

    Pooled daily arrays cover rows ``[test_start, test_end)`` of the input.
    ``annual_turnover`` excludes each fold's initial build; ``build_cost``
    is the total cost of those builds (already inside ``total_cost``).
    """

    mode: str
    train: int
    test: int
    rebalance_step: int
    folds: list[Fold]
    test_start: int
    test_end: int
    gross_returns: np.ndarray
    net_returns: np.ndarray
    costs: np.ndarray
    total_cost: float
    ann_cost: float
    build_cost: float
    annual_turnover: float
    n_trades: int
    ann_return: float
    ann_vol: float
    sharpe: float
    gross_sharpe: float
    max_drawdown: float

    def fold_sharpes(self) -> np.ndarray:
        return np.array([f.result.sharpe for f in self.folds])


def _trailing(fn: WeightFn, n: int) -> WeightFn:
    def wrapped(history: np.ndarray) -> np.ndarray:
        return fn(history[-n:])

    return wrapped


def walk_forward(
    returns: np.ndarray,
    weight_fn: WeightFn,
    *,
    train: int = 756,
    test: int = 252,
    rebalance_step: int = 21,
    mode: str = "rolling",
    cost_bps: float = 0.0,
    band: float = 0.0,
    partial: float = 1.0,
    rf: float = 0.0,
    vol_target: float | None = None,
    max_gross: float | None = None,
) -> WalkForwardResult:
    """Walk-forward evaluation built on :func:`run_backtest_net`.

    See the module docstring for the fold layout.  ``train`` (>= 2) is the
    estimation window, ``test`` (>= 2) the fold length and
    ``rebalance_step`` (>= 1) the rebalance spacing in rows inside a fold.
    The remaining keywords are passed to ``run_backtest_net`` unchanged.
    """
    x = validate_returns(returns, min_rows=3)
    t = x.shape[0]
    for name, val, lo in (
        ("train", train, 2),
        ("test", test, 2),
        ("rebalance_step", rebalance_step, 1),
    ):
        if isinstance(val, bool) or int(val) != val or int(val) < lo:
            raise ValueError(f"{name} must be an integer >= {lo}, got {val!r}")
    train, test, rebalance_step = int(train), int(test), int(rebalance_step)
    if mode not in ("rolling", "expanding"):
        raise ValueError(f"mode must be 'rolling' or 'expanding', got {mode!r}")
    if t < train + 2:
        raise ValueError(f"need at least train + 2 = {train + 2} rows, got {t}")
    folds: list[Fold] = []
    for k, s in enumerate(range(train, t, test)):
        e = min(s + test, t)
        if e - s < 2:
            break
        lo = s - train if mode == "rolling" else 0
        fn = _trailing(weight_fn, train) if mode == "rolling" else weight_fn
        res = run_backtest_net(
            x[lo:e],
            fn,
            window=s - lo,
            step=rebalance_step,
            cost_bps=cost_bps,
            band=band,
            partial=partial,
            rf=rf,
            vol_target=vol_target,
            max_gross=max_gross,
        )
        folds.append(Fold(k, lo, s, e, res))
    gross = np.concatenate([f.result.gross_returns for f in folds])
    net = np.concatenate([f.result.net_returns for f in folds])
    costs = np.concatenate([f.result.costs for f in folds])
    days = net.shape[0]
    total_cost = math.fsum(costs.tolist())
    build_cost = math.fsum(f.result.costs[0] for f in folds)
    later_turnover = math.fsum(math.fsum(f.result.turnovers[1:].tolist()) for f in folds)
    return WalkForwardResult(
        mode=mode,
        train=train,
        test=test,
        rebalance_step=rebalance_step,
        folds=folds,
        test_start=folds[0].test_start,
        test_end=folds[-1].test_end,
        gross_returns=gross,
        net_returns=net,
        costs=costs,
        total_cost=total_cost,
        ann_cost=ANNUALIZATION * total_cost / days,
        build_cost=build_cost,
        annual_turnover=ANNUALIZATION * later_turnover / days,
        n_trades=sum(f.result.n_trades for f in folds),
        ann_return=annualized_return(net),
        ann_vol=annualized_vol(net),
        sharpe=sharpe_ratio(net, rf),
        gross_sharpe=sharpe_ratio(gross, rf),
        max_drawdown=max_drawdown(net),
    )


# ---------------------------------------------------------- calendar schedule


@dataclass(frozen=True)
class ScheduleResult:
    """Result of :func:`run_schedule_net`.

    Daily arrays cover rows ``[start_row, T)`` where ``start_row`` is the
    first rebalance row; per-rebalance arrays have one entry per row in
    ``rebalance_rows``.
    """

    start_row: int
    rebalance_rows: list[int]
    gross_returns: np.ndarray
    net_returns: np.ndarray
    costs: np.ndarray
    turnovers: np.ndarray
    executed: np.ndarray
    pre_trade_weights: np.ndarray
    post_trade_weights: np.ndarray


def calendar_rebalance_rows(
    dates: np.ndarray, first_month: str, every_months: int = 1
) -> list[int]:
    """First row of every ``every_months``-th calendar month from ``first_month``.

    ``dates`` is an increasing ``datetime64[D]`` array and ``first_month``
    a ``"YYYY-MM"`` string.  Months with no rows are skipped (they cannot
    occur in daily market data, but the function does not assume it).
    """
    if every_months < 1:
        raise ValueError("every_months must be >= 1")
    months = np.asarray(dates, dtype="datetime64[D]").astype("datetime64[M]")
    m0 = np.datetime64(first_month, "M")
    rows: list[int] = []
    for i in range(months.shape[0]):
        m = months[i]
        if m < m0 or (i > 0 and months[i - 1] == m):
            continue
        if int((m - m0).astype(int)) % every_months == 0:
            rows.append(i)
    return rows


def run_schedule_net(
    returns: np.ndarray,
    weight_fn: WeightFn,
    rebalance_rows: Sequence[int],
    *,
    cost_bps: float = 0.0,
    band: float = 0.0,
    partial: float = 1.0,
    rf_daily: float | np.ndarray = 0.0,
) -> ScheduleResult:
    """Cost-aware backtest on an arbitrary, strictly increasing rebalance list.

    The accounting is that of :mod:`portopt.research.costs` (module
    docstring there), with the daily cash rate taken from ``rf_daily``
    (a scalar or one value per row of ``returns``).  At rebalance row
    ``t_k`` the weight function sees rows ``[0, t_k)``.  The first
    rebalance is the full initial build from cash and is charged.
    """
    x = validate_returns(returns, min_rows=3)
    t, n = x.shape
    rows = [int(r) for r in rebalance_rows]
    if not rows:
        raise ValueError("rebalance_rows is empty")
    if rows[0] < 2 or rows[-1] >= t or any(b <= a for a, b in zip(rows, rows[1:], strict=False)):
        raise ValueError("rebalance_rows must be strictly increasing within [2, T)")
    kappa = float(cost_bps) / 1.0e4
    band_v = float(band)
    part = float(partial)
    if not (math.isfinite(kappa) and kappa >= 0.0 and math.isfinite(band_v) and band_v >= 0.0):
        raise ValueError("cost_bps and band must be finite and >= 0")
    if not (math.isfinite(part) and 0.0 < part <= 1.0):
        raise ValueError(f"partial must lie in (0, 1], got {partial!r}")
    rf_arr = np.broadcast_to(np.asarray(rf_daily, dtype=float), (t,)).copy()
    if not np.isfinite(rf_arr).all():
        raise ValueError("rf_daily must be finite")
    start = rows[0]
    h_days = t - start
    gross = np.zeros(h_days)
    net = np.zeros(h_days)
    costs = np.zeros(h_days)
    k_reb = len(rows)
    pre = np.zeros((k_reb, n))
    post = np.zeros((k_reb, n))
    executed = np.zeros(k_reb, dtype=bool)
    turnovers = np.zeros(k_reb)
    w = np.zeros(n)
    cash = 1.0
    for k, t_k in enumerate(rows):
        history = x[:t_k].view()
        history.flags.writeable = False
        u = np.asarray(weight_fn(history), dtype=float).reshape(-1)
        if u.shape[0] != n or not np.isfinite(u).all():
            raise ValueError(f"weight_fn returned an invalid weight vector at row {t_k}")
        if abs(math.fsum(u.tolist()) - 1.0) > 1e-6:
            raise ValueError(f"weight_fn weights must sum to 1 at row {t_k}")
        pre[k] = w
        if k == 0:
            new = u.copy()
        elif math.fsum(np.abs(u - w).tolist()) < band_v:
            new = w.copy()
        else:
            new = w + part * (u - w)
        traded = math.fsum(np.abs(new - w).tolist())
        executed[k] = k == 0 or not np.array_equal(new, w)
        turnovers[k] = traded
        post[k] = new
        cash = cash + math.fsum((w - new).tolist())
        w = new
        cost_today = kappa * traded
        t_end = rows[k + 1] if k + 1 < k_reb else t
        for row in range(t_k, t_end):
            r = x[row]
            rf_d = float(rf_arr[row])
            g_r = math.fsum((w * r).tolist()) + cash * rf_d
            c_r = cost_today if row == t_k else 0.0
            n_r = g_r - c_r
            if not 1.0 + n_r > 0.0:
                raise ValueError(f"net asset value fell to zero or below at row {row}")
            gross[row - start] = g_r
            net[row - start] = n_r
            costs[row - start] = c_r
            w = w * (1.0 + r) / (1.0 + n_r)
            cash = (cash * (1.0 + rf_d) - c_r) / (1.0 + n_r)
    return ScheduleResult(start, rows, gross, net, costs, turnovers, executed, pre, post)
