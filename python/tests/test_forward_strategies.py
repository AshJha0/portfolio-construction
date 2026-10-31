"""Tests for the strategy wrappers of ``forward/run_forward.py`` and for the
frozen rule files in ``forward/rules/``.

The wrappers let a rule file name a :mod:`portopt.research.strategies_v2`
rule and its covariance estimator with plain JSON values.  The tests check
that each wrapper builds exactly the factory it names, that bad parameters
fail rule validation, that the runner reproduces a direct
``run_schedule_net`` of the same rule month by month, and that every
committed rule validates and names a registered study whose configuration
list is the rule file itself.

No test touches the network or reads the real data: the end-to-end test
runs on the small fake French files of ``test_research_data``.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from test_research_data import NOW, _no_network, fake_payloads, install  # noqa: F401 (autouse guard)

from portopt.research import data as rd
from portopt.research import strategies_v2 as v2
from portopt.research import walkforward as wf

REPO = Path(__file__).resolve().parents[2]


def _load_runner() -> ModuleType:
    if "run_forward" in sys.modules:
        return sys.modules["run_forward"]
    path = REPO / "forward" / "run_forward.py"
    spec = importlib.util.spec_from_file_location("run_forward", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_forward"] = mod
    spec.loader.exec_module(mod)
    return mod


runner = _load_runner()


def _history(t: int = 300, n: int = 5, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    common = rng.normal(0.0003, 0.009, size=(t, 1))
    return common + rng.normal(0.0002, 0.006, size=(t, n))


HL = math.log(0.5) / math.log(0.97)

WRAPPED = [
    ("min_variance_longonly_with", {"estimator": "sample"}, v2.min_variance_longonly_with(v2.sample_estimator())),
    ("min_variance_longonly_with", {"estimator": "lwcc"}, v2.min_variance_longonly_with(v2.lw_constant_corr_estimator())),
    ("min_variance_longonly_with", {"estimator": "lwid"}, v2.min_variance_longonly_with(v2.lw_identity_estimator())),
    ("min_variance_longonly_with", {"estimator": "nls"}, v2.min_variance_longonly_with(v2.nonlinear_estimator())),
    ("min_variance_longonly_with", {"estimator": "ewma", "half_life": HL}, v2.min_variance_longonly_with(v2.ewma_estimator(HL))),
    ("min_variance_longonly_with", {"estimator": "pca", "k": 3}, v2.min_variance_longonly_with(v2.pca_estimator(3))),
    ("erc_with", {"estimator": "nls"}, v2.erc_with(v2.nonlinear_estimator())),
    ("erc_with", {"estimator": "ewmacc", "half_life": 30}, v2.erc_with(v2.ewma_cc_estimator(30))),
    ("max_sharpe_bayes_stein", {"rf": 0.0}, v2.max_sharpe_bayes_stein(rf=0.0)),
    ("max_sharpe_bayes_stein", {"rf": 0.01, "estimator": "lwcc", "n_points": 9}, v2.max_sharpe_bayes_stein(rf=0.01, estimator=v2.lw_constant_corr_estimator(), n_points=9)),
]  # fmt: skip


@pytest.mark.parametrize(("name", "params", "direct"), WRAPPED)
def test_wrapper_builds_the_factory_it_names(
    name: str, params: dict[str, Any], direct: Any
) -> None:
    built = runner.STRATEGIES[name](**params)
    assert built.name == direct.name
    h = _history()
    np.testing.assert_array_equal(built(h), direct(h))
    # the parameters survive a JSON round trip unchanged
    again = runner.STRATEGIES[name](**json.loads(json.dumps(params)))
    np.testing.assert_array_equal(again(h), built(h))


def _rule(strategy: dict[str, Any], **over: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "schema_version": 1,
        "id": "fwd-test-v2",
        "study_id": "s-fwd-v2",
        "frozen_at": "2016-01-01",
        "git_tag": "forward-test",
        "strategy": strategy,
        "dataset": "ind10_vw_daily",
        "lookback_days": 120,
        "cost_bps": 10,
        "band": 0.05,
        "rebalance": "monthly",
        "benchmark": {"kind": "strategy", "name": "equal_weight", "params": {}},
    }
    doc.update(over)
    return doc


@pytest.mark.parametrize(
    "params",
    [
        {"estimator": "ledoit"},
        {"estimator": "ewma"},
        {"estimator": "ewma", "half_life": True},
        {"estimator": "ewma", "half_life": 20, "k": 2},
        {"estimator": "pca"},
        {"estimator": "pca", "k": 2.5},
        {"estimator": "pca", "k": True},
        {"estimator": "nls", "k": 3},
        {"estimator": "sample", "half_life": 10},
        {"estimator": "nls", "shrink": True},
        {},
    ],
)
def test_bad_estimator_params_fail_validation(params: dict[str, Any]) -> None:
    doc = _rule({"name": "min_variance_longonly_with", "params": params})
    with pytest.raises(runner.RuleError, match="params invalid"):
        runner.validate_rule(doc)


def test_v2_rule_validates() -> None:
    doc = _rule({"name": "erc_with", "params": {"estimator": "pca", "k": 2}})
    rule = runner.validate_rule(doc)
    assert rule.strategy == "erc_with" and rule.params == {"estimator": "pca", "k": 2}


def test_runner_reproduces_direct_schedule_for_v2_rule(tmp_path: Path) -> None:
    """The runner's monthly rows equal a direct run_schedule_net of the same
    rule on the trailing window (the computation the study evaluates)."""
    data = tmp_path / "data"
    rules = tmp_path / "rules"
    rules.mkdir()
    install(data, fake_payloads("2016-04-29", "201604"))
    strategy = {"name": "min_variance_longonly_with", "params": {"estimator": "nls"}}
    doc = _rule(strategy)
    wf.register_study("s-fwd-v2", "v2 forward test", [doc], data_dir=data, now=NOW)
    (rules / "fwd-test-v2.json").write_text(json.dumps(doc), encoding="utf-8")
    logs: list[str] = []
    rows = runner.run(
        rules_dir=rules,
        ledger_path=tmp_path / "ledger.csv",
        results_path=tmp_path / "results.md",
        data_dir=data,
        now=NOW,
        log=logs.append,
    )
    assert [r["month"] for r in rows] == ["2016-01", "2016-02", "2016-03", "2016-04"]

    assets = rd.load_returns("ind10_vw_daily", "all", study_id="s-fwd-v2", data_dir=data)
    rf = rd.load_returns("ff3_daily", "all", study_id="s-fwd-v2", data_dir=data)
    x, r_f, dates = assets.values, rf.column("RF"), assets.dates
    rows_cal = wf.calendar_rebalance_rows(dates, "2016-01")
    rule_fn = v2.min_variance_longonly_with(v2.nonlinear_estimator())
    direct = wf.run_schedule_net(
        x, lambda h: rule_fn(h[-120:]), rows_cal, cost_bps=10, band=0.05, rf_daily=r_f
    )
    bench = wf.run_schedule_net(
        x, lambda h: np.full(h.shape[1], 1.0 / h.shape[1]), rows_cal, cost_bps=10, band=0.05, rf_daily=r_f
    )  # fmt: skip
    months = dates[rows_cal[0] :].astype("datetime64[M]")
    for r in rows:
        sel = months == np.datetime64(r["month"], "M")
        net = math.prod((1.0 + direct.net_returns[sel]).tolist()) - 1.0
        bmk = math.prod((1.0 + bench.net_returns[sel]).tolist()) - 1.0
        assert float(r["net_return"]) == pytest.approx(net, abs=1e-10)
        assert float(r["benchmark_return"]) == pytest.approx(bmk, abs=1e-10)


# ---------------------------------------------------------- committed rules

RULE_FILES = sorted((REPO / "forward" / "rules").glob("*.json"))


@pytest.mark.skipif(not RULE_FILES, reason="no frozen rule files yet")
def test_committed_rules_validate_and_are_registered() -> None:
    rules = runner.load_rules(REPO / "forward" / "rules")
    registry = rd.load_registry(REPO / "data" / "real")
    by_id = {s["id"]: s for s in registry["studies"]}
    # Each forward study's pre-registration freezes at most two rules, and
    # every committed rule file is one of its study's registered configs.
    per_study: dict[str, int] = {}
    for rule in rules:
        per_study[rule.study_id] = per_study.get(rule.study_id, 0) + 1
    assert per_study, "no forward rules committed"
    for sid, count in per_study.items():
        assert 1 <= count <= 2, f"study {sid} freezes {count} forward rules (at most two)"
        assert count == len(by_id[sid]["configs"]), f"study {sid}: rule count != registered configs"
    for rule, path in zip(rules, RULE_FILES, strict=True):
        assert rule.study_id in by_id, f"{rule.id}: study {rule.study_id} is not registered"
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert doc in by_id[rule.study_id]["configs"], (
            f"{rule.id}: the registered configuration list of {rule.study_id} "
            "does not contain this rule file as written"
        )
        assert rule.frozen_at.isoformat() == "2026-11-01"
        assert rule.cost_bps == 10.0 and rule.rebalance == "monthly"
