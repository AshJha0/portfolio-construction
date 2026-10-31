"""Tests for the signal strategies of ``forward/run_forward.py`` and the
``forward/rules/r2-signal-*.json`` rule files.

No network: the end-to-end test runs on the fake French files of
``test_research_data``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_forward_strategies import REPO, runner
from test_research_data import NOW, _no_network, fake_payloads, install  # noqa: F401 (autouse guard)

from portopt.research import data as rd
from portopt.research import signals as sg
from portopt.research import walkforward as wf


def _history(t: int = 520, n: int = 6, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0003, 0.009, size=(t, 1)) + rng.normal(0.0002, 0.006, size=(t, n))


WRAPPED = [
    ("rank_top_tercile", {"signals": ["mom12_1"]}),
    ("rank_top_tercile", {"signals": ["mom12_1", "tsmom", "strev", "lowbeta"], "method": "inverse_vol"}),
    ("signal_black_litterman", {"signals": ["lowbeta"], "ic": 0.05}),
]  # fmt: skip


@pytest.mark.parametrize(("name", "params"), WRAPPED)
def test_wrapper_is_the_library_factory(name: str, params: dict[str, Any]) -> None:
    built = runner.STRATEGIES[name](**json.loads(json.dumps(params)))
    direct = getattr(sg, name)(**params)
    h = _history()
    assert built.name == direct.name
    np.testing.assert_array_equal(built(h), direct(h))


def _rule(strategy: dict[str, Any], **over: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "schema_version": 1,
        "id": "fwd-signal-test",
        "study_id": "s-fwd-sig",
        "frozen_at": "2016-02-01",
        "git_tag": "forward-test",
        "strategy": strategy,
        "dataset": "ind10_vw_daily",
        "lookback_days": 504,
        "cost_bps": 10,
        "band": 0.0,
        "rebalance": "monthly",
        "benchmark": {"kind": "strategy", "name": "equal_weight", "params": {}},
    }
    doc.update(over)
    return doc


@pytest.mark.parametrize(
    "params",
    [{"signals": []}, {"signals": ["value"]}, {"signals": ["mom12_1"], "method": "fitted"}, {}],
)
def test_bad_signal_params_fail_validation(params: dict[str, Any]) -> None:
    with pytest.raises(runner.RuleError):
        runner.validate_rule(_rule({"name": "rank_top_tercile", "params": params}))


def test_runner_reproduces_direct_schedule(tmp_path: Path) -> None:
    data = tmp_path / "data"
    rules = tmp_path / "rules"
    rules.mkdir()
    install(data, fake_payloads("2016-05-31", "201605"))
    params = {"signals": ["mom12_1", "strev"], "method": "equal"}
    doc = _rule({"name": "rank_top_tercile", "params": params})
    wf.register_study("s-fwd-sig", "signal forward test", [doc], data_dir=data, now=NOW)
    (rules / "fwd-signal-test.json").write_text(json.dumps(doc), encoding="utf-8")
    rows = runner.run(
        rules_dir=rules,
        ledger_path=tmp_path / "ledger.csv",
        results_path=tmp_path / "results.md",
        data_dir=data,
        now=NOW,
        log=lambda _m: None,
    )
    assert [r["month"] for r in rows] == ["2016-02", "2016-03", "2016-04", "2016-05"]
    assets = rd.load_returns("ind10_vw_daily", "all", study_id="s-fwd-sig", data_dir=data)
    rf = rd.load_returns("ff3_daily", "all", study_id="s-fwd-sig", data_dir=data)
    rows_cal = wf.calendar_rebalance_rows(assets.dates, "2016-02")
    fn = sg.rank_top_tercile(**params)
    direct = wf.run_schedule_net(
        assets.values, lambda h: fn(h[-504:]), rows_cal, cost_bps=10, rf_daily=rf.column("RF")
    )
    months = assets.dates[rows_cal[0] :].astype("datetime64[M]")
    for r in rows:
        sel = months == np.datetime64(r["month"], "M")
        net = math.prod((1.0 + direct.net_returns[sel]).tolist()) - 1.0
        assert float(r["net_return"]) == pytest.approx(net, abs=1e-10)


SIGNAL_RULES = sorted((REPO / "forward" / "rules").glob("r2-signal-*.json"))


@pytest.mark.skipif(not SIGNAL_RULES, reason="no signal rule files yet")
def test_signal_rules_validate_and_are_registered() -> None:
    assert len(SIGNAL_RULES) <= 2
    registry = rd.load_registry(REPO / "data" / "real")
    by_id = {s["id"]: s for s in registry["studies"]}
    for path in SIGNAL_RULES:
        doc = json.loads(path.read_text(encoding="utf-8"))
        rule = runner.validate_rule(doc, source=str(path))
        assert path.stem == rule.id
        assert rule.study_id == "r2-forward-signals-v1"
        assert doc in by_id[rule.study_id]["configs"]
        assert rule.frozen_at.isoformat() == "2026-11-01" and rule.git_tag == "forward-v1"
        need = sg.combine_rows(
            doc["strategy"]["params"]["signals"], doc["strategy"]["params"].get("method", "equal")
        )
        assert rule.lookback_days >= need
