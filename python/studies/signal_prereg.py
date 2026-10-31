"""Pre-registered hypotheses and configurations for ``signal_study.py``.

Fixed from development evidence only (``docs/results/signal_study.csv``,
rows with ``period == "dev"``), before any 2016+ row was read by this
study.  ``register()`` records them in ``data/real/registry.json``;
``check_registered()`` refuses the seen-period run if the code differs
from the registration.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import signal_core as sc
from signal_report import _pair, f, table

from portopt.research import data as rd
from portopt.research import walkforward as wf

#: (id, test config, comparator config, statement)
HYPOTHESES: list[tuple[str, str, str, str]] = [
    (
        "S1",
        "ind10/bl.mom12_1/10bp",
        "ind10/equal_weight/10bp",
        "Black-Litterman with 12-1 momentum views (IC 0.05) has a higher net Sharpe ratio than "
        "1/N on 10 industries.",
    ),
    (
        "S2",
        "ind49/rank.mom12_1/10bp",
        "ind49/equal_weight/10bp",
        "The top tercile by 12-1 momentum has a higher net Sharpe ratio than 1/N on 49 industries.",
    ),
    (
        "S3",
        "ind10/mvtc.mom12_1/10bp",
        "ind10/minvar.lwcc/L504/10bp",
        "Turnover-penalised mean-variance with 12-1 momentum alphas has a higher net Sharpe "
        "ratio than long-only minimum variance (LW constant-correlation, 504 days) on 10 "
        "industries.",
    ),
    (
        "S4",
        "ind49/rank.lowvol/10bp",
        "ind49/equal_weight/10bp",
        "The top tercile by low 12-month volatility has a higher net Sharpe ratio than 1/N on "
        "49 industries.",
    ),
]

FORWARD_RULES: list[str] = ["r2-signal-mom-bl-ind10", "r2-signal-mom-rank-ind49"]


def seen_configs() -> list[str]:
    ids: list[str] = []
    for _, a, b, _ in HYPOTHESES:
        for c in (a, b, a.replace("/10bp", "/25bp"), b.replace("/10bp", "/25bp")):
            if c not in ids:
                ids.append(c)
    for u in ("ind10", "ind49"):
        for c in (f"{u}/market", f"{u}/equal_weight/10bp", f"{u}/minvar.lwcc/L504/10bp"):
            if c not in ids:
                ids.append(c)
    return ids


SEEN_CONFIGS = seen_configs()


def hypotheses_sha256() -> str:
    text = json.dumps(HYPOTHESES, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _forward_docs() -> list[dict[str, Any]]:
    return [
        json.loads((sc.ROOT / "forward" / "rules" / f"{r}.json").read_text(encoding="utf-8"))
        for r in FORWARD_RULES
    ]


def register() -> None:
    if not HYPOTHESES:
        raise SystemExit("no hypotheses written yet")
    if not rd.is_registered(sc.SEEN_STUDY_ID):
        wf.register_study(
            sc.SEEN_STUDY_ID,
            "Round 2 signal study, seen period 2016-01..2026-09 (not a clean holdout: the "
            "construction study's results on it were seen first). Hypotheses SHA-256 "
            f"{hypotheses_sha256()}; see docs/results/signal_preregistration.md.",
            [sc.config_from_id(c).to_json() for c in SEEN_CONFIGS],
        )
    if FORWARD_RULES and not rd.is_registered(sc.FORWARD_STUDY_ID):
        wf.register_study(
            sc.FORWARD_STUDY_ID,
            "Round 2 forward signal rules, frozen 2026-11-01 (git tag forward-v1); the "
            "configuration list is the rule files as written.",
            _forward_docs(),
        )


def check_registered() -> None:
    reg = {s["id"]: s for s in rd.load_registry()["studies"]}
    entry = reg.get(sc.SEEN_STUDY_ID)
    if entry is None:
        raise SystemExit(f"{sc.SEEN_STUDY_ID} is not registered")
    if hypotheses_sha256() not in entry["description"]:
        raise SystemExit("hypotheses differ from the registration")
    want = wf.configs_sha256([sc.config_from_id(c).to_json() for c in SEEN_CONFIGS])
    if want != entry["configs_sha256"]:
        raise SystemExit("configurations differ from the registration")


def results_section(dev: dict[str, dict[str, str]], seen: dict[str, dict[str, str]]) -> list[str]:
    if not HYPOTHESES:
        return []
    out = [
        "## Pre-registered hypotheses (10 bp, one-sided Ledoit-Wolf, Holm over the family)",
        "",
        "The 2016-2026 period is a seen period, not a clean holdout: the construction "
        "study's results on it were read before these signals were chosen. The clean "
        "evidence is the forward log from 2026-11.",
        "",
    ]
    import signal_core as core

    body = []
    ps: list[float] = []
    for hid, a, b, text in HYPOTHESES:
        key = "ew" if "equal_weight" in b else "minvar"
        d = dev.get(a, {})
        s = seen.get(a, {})
        p = float(s[f"p1_vs_{key}"]) if s.get(f"p1_vs_{key}") else float("nan")
        ps.append(p)
        body.append([hid, text, _pair(d, key), _pair(s, key)])
    if seen and all(p == p for p in ps):
        adj = core.holm(ps)
        for row, q in zip(body, adj, strict=True):
            row += [f(str(q)), "supported" if q <= 0.05 else "not supported"]
    else:
        for row in body:
            row += ["", "not yet run"]
    out += [
        table(["id", "hypothesis", "dev dSR (p)", "seen dSR (p)", "Holm p", "decision"], body),
        "",
    ]
    if seen:
        rows = []
        for cid in SEEN_CONFIGS:
            s = seen.get(cid)
            d = dev.get(cid, {})
            if s is None:
                continue
            rows.append(
                [
                    s["universe"],
                    s["label"],
                    f(d.get("sharpe", "")),
                    f(s["sharpe"]),
                    f(s["se"]),
                    f(str(100 * float(s["ann_vol"])), 1),
                    f(str(100 * float(s["max_dd"])), 0),
                    f(s["turnover"]),
                    _pair(s, "ew"),
                    _pair(s, "minvar"),
                ]
            )
        out += [
            "## Registered configurations, development and seen period",
            "",
            table(
                [
                    "universe",
                    "configuration",
                    "dev SR",
                    "seen SR",
                    "seen SE",
                    "vol %",
                    "max DD %",
                    "turnover/yr",
                    "vs 1/N",
                    "vs min-var",
                ],
                rows,
            ),
            "",
        ]
        out += CONCLUSIONS
    return out


#: Written after the single seen-period run; every number is in signal_study.csv.
CONCLUSIONS = [
    "## Conclusions",
    "",
    "1. **No registered hypothesis is supported** (Holm p 0.77 to 0.92). Three of four seen-"
    "period differences have the registered sign (S1 +0.01, S2 +0.07, S3 +0.12) but all are "
    "well inside one standard error (about 0.1 to 0.2); S4 reversed (-0.16).",
    "2. **Momentum held up best.** Momentum rules kept their development level in the seen "
    "period (top tercile on 49 industries 0.71 against 0.64 for 1/N; BL on 10 industries 0.68 "
    "against 0.67) but did not beat the market (0.73). At 25 bp the gains shrink to +0.03 and "
    "-0.02.",
    "3. **Low-risk signals lagged** in 2016-2026, as minimum variance did in the construction "
    "study: low volatility on 49 industries 0.48, 1/N 0.64.",
    "4. **Short-term reversal does not work on industries** in development (about -0.3 Sharpe "
    "against 1/N in every sub-period, 15 times NAV turnover a year), so the two combinations "
    "that include it lost too. One-month industry returns continue, they do not reverse "
    "(Moskowitz and Grinblatt 1999).",
    "5. **Publication split** (development only, to 2015): momentum's gain on 10 industries was "
    "larger after 1993 than before, and low volatility's after 2011 rests on five years; none "
    "of these splits has the power to show the decay McLean and Pontiff (2016) report.",
    "6. **Trial count.** 66 signal trials plus 206 construction trials. The deflated Sharpe "
    "ratio in the development table tests each strategy's Sharpe ratio against the expected "
    "maximum of that many null trials, not its difference from 1/N, so a high DSR here says "
    "the strategy beat cash, not that the signal added anything.",
    "",
    "## Limits",
    "",
    "* The seen period was not blind (section 1 of `signal_preregistration.md`); the "
    "registration was recorded in `data/real/registry.json` at 2026-10-31T18:58:16Z and the "
    "run made at commit fadb496 with the pre-registration not yet committed.",
    "* A month is 21 trading days, not a calendar month; the signal windows drift by a few "
    "days against calendar months.",
    "* IC 0.05, risk aversion 5 and the BL parameters are assumptions, not estimates; "
    "mean-variance and BL ran on 10 industries only (QP runtime).",
    "* Index series are not investable; costs are linear assumptions, the same in 1930 and 2015.",
    "* Time-series momentum is used here as a long-only cross-sectional score (sign over "
    "volatility), not as the long-short futures strategy of Moskowitz, Ooi and Pedersen.",
    "",
]
