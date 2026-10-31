"""Tests for portopt.research.walkforward and the forward paper-log runner.

The runner lives in ``forward/run_forward.py`` (outside the package) and is
loaded from its path.  Everything runs on tiny fake French-format files;
no test touches the network.
"""

from __future__ import annotations

import csv
import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from test_research_data import NOW, _no_network, fake_payloads, install  # noqa: F401 (autouse guard)

from portopt.research import data as rd
from portopt.research import equal_weight, erc, run_backtest_net
from portopt.research import walkforward as wf

REPO = Path(__file__).resolve().parents[2]


def _returns(t: int = 400, n: int = 4, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0004, 0.01, size=(t, n))


# ------------------------------------------------------------------ registry


def test_register_study_appends_hash_and_counts(tmp_path: Path) -> None:
    cfgs = [{"rule": "erc", "cost_bps": 5}, {"rule": "erc", "cost_bps": 10}]
    e = wf.register_study("s-001", "first", cfgs, data_dir=tmp_path, now=NOW)
    assert e["registered_at_utc"] == "2026-10-31T12:00:00Z"
    assert e["n_configs"] == 2 and e["configs_sha256"] == wf.configs_sha256(cfgs)
    # key order does not change the hash; re-registering identically is a no-op
    same = [{"cost_bps": 5, "rule": "erc"}, {"cost_bps": 10, "rule": "erc"}]
    assert wf.register_study("s-001", "first", same, data_dir=tmp_path) == e
    with pytest.raises(ValueError, match="different configurations"):
        wf.register_study("s-001", "first", cfgs[:1], data_dir=tmp_path)
    wf.register_study("s-002", "second", [{"rule": "ivol"}] * 3, data_dir=tmp_path, now=NOW)
    reg = json.loads((tmp_path / "registry.json").read_text())
    assert [s["id"] for s in reg["studies"]] == ["s-001", "s-002"]
    assert wf.n_trials(data_dir=tmp_path) == 5 and wf.n_trials("s-002", data_dir=tmp_path) == 3
    assert rd.is_registered("s-002", tmp_path)
    with pytest.raises(KeyError):
        wf.n_trials("s-404", data_dir=tmp_path)


@pytest.mark.parametrize(
    ("sid", "desc", "cfgs"),
    [("A-1", "x", [{}]), ("ok-1", " ", [{}]), ("ok-1", "x", []), ("ok-1", "x", [float("nan")])],
)
def test_register_study_rejects_bad_input(tmp_path: Path, sid: str, desc: str, cfgs: Any) -> None:
    with pytest.raises(ValueError):
        wf.register_study(sid, desc, cfgs, data_dir=tmp_path)


# ---------------------------------------------------------- schedule engine


def test_run_schedule_net_matches_run_backtest_net_on_regular_schedule() -> None:
    x = _returns()
    kw = {"cost_bps": 12.0, "band": 0.01, "partial": 0.5}
    ref = run_backtest_net(x, erc(), window=120, step=21, rf=0.03, **kw)
    got = wf.run_schedule_net(x, erc(), ref.rebalance_dates, rf_daily=0.03 / 252, **kw)
    np.testing.assert_allclose(got.net_returns, ref.net_returns, rtol=0, atol=1e-15)
    np.testing.assert_allclose(got.gross_returns, ref.gross_returns, rtol=0, atol=1e-15)
    np.testing.assert_allclose(got.turnovers, ref.turnovers, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(got.executed, ref.executed)


def test_run_schedule_net_uses_per_row_rf_and_validates() -> None:
    x = np.zeros((10, 2))
    rf = np.linspace(0.0, 0.0009, 10)
    res = wf.run_schedule_net(x, equal_weight(), [3], rf_daily=rf)
    # all-risky portfolio earns nothing on zero returns; cash weight is 0
    np.testing.assert_allclose(res.net_returns, 0.0, atol=1e-18)
    with pytest.raises(ValueError, match="strictly increasing"):
        wf.run_schedule_net(x, equal_weight(), [5, 5])
    with pytest.raises(ValueError, match="strictly increasing"):
        wf.run_schedule_net(x, equal_weight(), [1])


def test_calendar_rebalance_rows() -> None:
    dates = np.array(["2016-12-30", "2017-01-03", "2017-01-04", "2017-02-01", "2017-03-01",
                      "2017-04-03", "2017-05-01"], dtype="datetime64[D]")  # fmt: skip
    assert wf.calendar_rebalance_rows(dates, "2017-01") == [1, 3, 4, 5, 6]
    assert wf.calendar_rebalance_rows(dates, "2017-01", 3) == [1, 5]


# -------------------------------------------------------------- walk-forward


@pytest.mark.parametrize("mode", ["rolling", "expanding"])
def test_walk_forward_folds_tile_the_test_period(mode: str) -> None:
    x = _returns(t=401)
    seen: list[int] = []

    def fn(h: np.ndarray) -> np.ndarray:
        seen.append(h.shape[0])
        return np.full(h.shape[1], 1.0 / h.shape[1])

    res = wf.walk_forward(x, fn, train=100, test=60, rebalance_step=20, mode=mode, cost_bps=10)
    starts = [f.test_start for f in res.folds]
    assert starts == [100, 160, 220, 280, 340]
    # rows [400, 401) form a 1-row stub, which is dropped
    assert res.test_start == 100 and res.test_end == 400
    assert res.net_returns.shape[0] == 300
    for f in res.folds:
        lo = f.test_start - 100 if mode == "rolling" else 0
        assert f.train_start == lo
    if mode == "rolling":
        assert set(seen) == {100}
    else:
        assert seen[0] == 100 and max(seen) > 300
    np.testing.assert_allclose(
        res.net_returns, np.concatenate([f.result.net_returns for f in res.folds])
    )
    assert res.build_cost == pytest.approx(10e-4 * len(res.folds))
    assert res.total_cost >= res.build_cost


def test_walk_forward_fold_equals_direct_backtest() -> None:
    x = _returns(t=330)
    res = wf.walk_forward(x, erc(), train=120, test=100, rebalance_step=21, cost_bps=5, band=0.02)
    f = res.folds[1]
    direct = run_backtest_net(
        x[f.train_start : f.test_end],
        lambda h: erc()(h[-120:]),
        window=120,
        step=21,
        cost_bps=5,
        band=0.02,
    )
    np.testing.assert_allclose(f.result.net_returns, direct.net_returns, rtol=0, atol=1e-15)
    # a trailing stub shorter than 2 rows is dropped
    short = wf.walk_forward(_returns(t=221), equal_weight(), train=120, test=100)
    assert [g.test_end for g in short.folds] == [220]


def test_walk_forward_validates() -> None:
    x = _returns(t=50)
    with pytest.raises(ValueError, match="mode"):
        wf.walk_forward(x, equal_weight(), train=10, test=10, mode="sliding")
    with pytest.raises(ValueError, match="train"):
        wf.walk_forward(x, equal_weight(), train=1, test=10)
    with pytest.raises(ValueError, match="at least"):
        wf.walk_forward(x, equal_weight(), train=49, test=10)


# ------------------------------------------------------------ forward runner


def _load_runner() -> ModuleType:
    path = REPO / "forward" / "run_forward.py"
    spec = importlib.util.spec_from_file_location("run_forward", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_forward"] = mod
    spec.loader.exec_module(mod)
    return mod


runner = _load_runner()


def rule_doc(**over: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "schema_version": 1,
        "id": "fwd-test-erc",
        "study_id": "s-fwd",
        "frozen_at": "2016-01-01",
        "git_tag": "forward/fwd-test-erc",
        "strategy": {"name": "erc", "params": {"shrink": True}},
        "dataset": "ind10_vw_daily",
        "lookback_days": 120,
        "cost_bps": 10,
        "band": 0.02,
        "rebalance": "monthly",
        "benchmark": {"kind": "strategy", "name": "equal_weight", "params": {}},
        "description": "test rule",
    }
    doc.update(over)
    return doc


class Env:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.data = root / "data"
        self.rules = root / "rules"
        self.ledger = root / "ledger.csv"
        self.results = root / "results.md"
        self.rules.mkdir(parents=True)
        self.logs: list[str] = []

    def write_rule(self, doc: dict[str, Any]) -> None:
        (self.rules / f"{doc['id']}.json").write_text(json.dumps(doc, indent=2))

    def run(self, now: dt.datetime = NOW, **kw: Any) -> list[dict[str, str]]:
        out: list[dict[str, str]] = runner.run(
            rules_dir=self.rules,
            ledger_path=self.ledger,
            results_path=self.results,
            data_dir=self.data,
            now=now,
            log=self.logs.append,
            **kw,
        )
        return out

    def ledger_rows(self) -> list[dict[str, str]]:
        with self.ledger.open(newline="") as fh:
            return list(csv.DictReader(fh))


@pytest.fixture()
def env(tmp_path: Path) -> Env:
    e = Env(tmp_path)
    install(e.data, fake_payloads("2016-03-31", "201603"))
    wf.register_study("s-fwd", "forward test", [rule_doc()], data_dir=e.data, now=NOW)
    return e


def test_no_rules_exits_cleanly(tmp_path: Path) -> None:
    e = Env(tmp_path)
    assert e.run() == []
    assert any("No rule files" in s for s in e.logs)
    assert not e.ledger.exists() and not e.results.exists()


def test_forward_appends_months_and_is_idempotent(env: Env) -> None:
    env.write_rule(rule_doc())
    rows = env.run()
    assert [r["month"] for r in rows] == ["2016-01", "2016-02", "2016-03"]
    assert {r["row_type"] for r in rows} == {"original"}
    assert float(rows[0]["turnover"]) == pytest.approx(1.0)  # initial build from cash
    assert float(rows[0]["cost"]) == pytest.approx(10e-4)
    cum = np.prod([1 + float(r["net_return"]) for r in rows]) - 1
    assert float(rows[-1]["cumulative_net"]) == pytest.approx(cum, abs=1e-9)
    assert env.ledger_rows() == rows
    text = env.results.read_text()
    assert "not investment advice" in text and "| 2016-03 |" in text

    before = env.ledger.read_bytes()
    assert env.run(now=NOW + dt.timedelta(days=1)) == []
    assert env.ledger.read_bytes() == before


def test_forward_frozen_mid_month_starts_next_month(env: Env) -> None:
    env.write_rule(rule_doc(frozen_at="2016-01-15"))
    assert [r["month"] for r in env.run()] == ["2016-02", "2016-03"]


def test_forward_market_benchmark(env: Env) -> None:
    env.write_rule(rule_doc(benchmark={"kind": "market"}, rebalance="quarterly"))
    rows = env.run()
    assert len(rows) == 3
    assert float(rows[1]["turnover"]) == 0.0  # quarterly: no rebalance in month 2


def test_forward_new_vintage_appends_and_restates(env: Env) -> None:
    env.write_rule(rule_doc())
    first = env.run()
    # next month's data: same history, one more month -> one new original row
    install(env.data, fake_payloads("2016-04-29", "201604"), now=NOW + dt.timedelta(days=31))
    second = env.run(now=NOW + dt.timedelta(days=31))
    assert [(r["month"], r["row_type"]) for r in second] == [("2016-04", "original")]
    assert second[0]["data_vintage"] != first[0]["data_vintage"]
    # the library restates February: a restatement row, the original untouched
    install(
        env.data,
        fake_payloads("2016-04-29", "201604", tweak=("2016-02-10", 0, 9.99)),
        now=NOW + dt.timedelta(days=40),
    )
    third = env.run(now=NOW + dt.timedelta(days=40))
    types = [(r["month"], r["row_type"]) for r in third]
    assert ("2016-02", "restatement") in types and all(t == "restatement" for _, t in types)
    assert "2016-01" not in {m for m, _ in types}
    all_rows = env.ledger_rows()
    assert all_rows[: len(first)] == first
    assert "yes" in env.results.read_text()


def test_forward_refuses_to_rewrite_under_same_vintage(env: Env) -> None:
    env.write_rule(rule_doc())
    env.run()
    text = env.ledger.read_text().splitlines()
    parts = text[2].split(",")
    parts[3] = "0.1234567890"
    text[2] = ",".join(parts)
    env.ledger.write_text("\n".join(text) + "\n")
    with pytest.raises(runner.LedgerConflict, match="same data vintage"):
        env.run()


def test_forward_refuses_edited_rule(env: Env) -> None:
    env.write_rule(rule_doc())
    env.run()
    env.write_rule(rule_doc(cost_bps=20))
    with pytest.raises(runner.LedgerConflict, match="may not be edited"):
        env.run()


def test_forward_dry_run_writes_nothing(env: Env) -> None:
    env.write_rule(rule_doc())
    rows = env.run(dry_run=True)
    assert len(rows) == 3
    assert not env.ledger.exists() and not env.results.exists()


def test_forward_requires_registered_study(env: Env) -> None:
    env.write_rule(rule_doc(study_id="not-registered"))
    with pytest.raises(rd.HoldoutAccessError):
        env.run()


def test_forward_fetch_flag_uses_fetcher(env: Env) -> None:
    env.write_rule(rule_doc())
    p = fake_payloads("2016-04-29", "201604")
    rows = env.run(fetch=True, fetcher=p.__getitem__)
    assert rows[-1]["month"] == "2016-04"


@pytest.mark.parametrize(
    ("over", "message"),
    [
        ({"surprise": 1}, "unknown keys"),
        ({"frozen_at": "2016-1-1"}, "frozen_at"),
        ({"strategy": {"name": "erc", "params": {"bogus": 1}}}, "params invalid"),
        ({"strategy": {"name": "magic"}}, "strategy.name"),
        ({"dataset": "ff3_daily"}, "dataset"),
        ({"lookback_days": 5}, "lookback_days"),
        ({"cost_bps": -1}, "cost_bps"),
        ({"rebalance": "weekly"}, "rebalance"),
        ({"benchmark": {"kind": "market", "x": 1}}, "no other keys"),
        ({"id": "Bad Id"}, "id must match"),
        ({"schema_version": 2}, "schema_version"),
    ],
)
def test_rule_validation(over: dict[str, Any], message: str) -> None:
    with pytest.raises(runner.RuleError, match=message):
        runner.validate_rule(rule_doc(**over))


def test_rule_file_name_must_match_id(tmp_path: Path) -> None:
    (tmp_path / "other.json").write_text(json.dumps(rule_doc()))
    with pytest.raises(runner.RuleError, match="file name"):
        runner.load_rules(tmp_path)


def test_complete_through_falls_back_to_last_business_day(env: Env) -> None:
    p = rd.load_returns("ind10_vw_daily", data_dir=env.data)
    assert str(runner.complete_through(p)) == "2016-03"
    from dataclasses import replace

    q = replace(p, crsp_database=None, dates=p.dates[:-1], values=p.values[:-1])
    assert str(runner.complete_through(q)) == "2015-11"  # dev ends 2015-12-30, not month end
