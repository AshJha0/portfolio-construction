"""Pre-registered study of portfolio-construction levers on real index data.

Run from the repository root::

    python python/studies/real_data_study.py --phase dev          # phase 1
    python python/studies/real_data_study.py --register-holdout   # end of phase 1
    python python/studies/real_data_study.py --phase holdout      # phase 2, once

Phase ``dev`` reads only the development period (1926-07-01 to
2015-12-31), through ``portopt.research.data.load_returns(...,
period="development")``.  Before computing anything it registers its
whole configuration grid as the study ``r2-dev-exploration``, so the
trial count used by the deflated Sharpe ratio is the number of
configurations actually evaluated.  It writes
``docs/results/real_data_study_dev.md`` and ``.csv``.

``--register-holdout`` writes the frozen forward rules to
``forward/rules/``, registers them as the study ``r2-forward-v1``,
registers the pre-registered holdout configurations as ``r2-holdout-v1``
and writes ``docs/results/real_data_preregistration.md``.  It reads no
holdout data.

Phase ``holdout`` (phase 2, run once after the pre-registration is public)
evaluates exactly the configurations registered as ``r2-holdout-v1`` on
2016-01-01 onward and writes ``docs/results/real_data_study.md`` and
``.csv``.

The series are research index portfolios from the Kenneth R. French Data
Library; none is investable, and the trading costs are assumptions.  The
raw data is fetched by ``scripts/fetch_data.py`` and never committed; the
outputs hold summary statistics only.

The run is deterministic: BLAS is pinned to one thread before numpy is
imported, every weight is a pure function of the history it sees, and the
only randomness is the bootstrap, seeded with ``BOOT_SEED``.  Modules:
``real_data_core`` (configurations, data, estimation, backtests,
statistics), ``real_data_engine`` (a scaled calendar-schedule backtest and
a fast long-only QP, both checked against the library on every run),
``real_data_report`` (the Markdown pages) and ``real_data_prereg``
(hypotheses, holdout configurations, forward rules, registration and the
holdout run).
"""

from __future__ import annotations

import os

# One BLAS thread in this process and in every spawned worker, so that
# matrix products are summed in the same order on every run.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import real_data_core as core  # noqa: E402
import real_data_report as report  # noqa: E402


def run_dev(workers: int, check_solver: bool, state: Path | None) -> int:
    t_all = time.perf_counter()
    configs = core.dev_grid()
    entry = core.register_dev(configs)
    n_total = core.dev_n_trials()
    print(
        f"registered {core.DEV_STUDY_ID}: {entry['n_configs']} configurations, sha256 "
        f"{entry['configs_sha256'][:16]}, at {entry['registered_at_utc']}; development "
        f"trials counted for the DSR: {n_total}"
    )
    if state is not None and state.exists():
        with state.open("rb") as fh:
            run, rows = pickle.load(fh)
        print(f"loaded computed results from {state}")
    else:
        panels, rows_by_uni = core.dev_panels()
        for u, p in panels.items():
            r = rows_by_uni[u]
            print(
                f"{u}: {p.x.shape[0]} rows x {p.x.shape[1]} assets, {len(r)} monthly "
                f"rebalances from {p.dates[r[0]]}"
            )
        run = core.run_configs(
            configs, panels, rows_by_uni, workers=workers, check_solver=check_solver, log=print
        )
        rows = core.build_rows(run, "dev", n_total, print)
        if state is not None:
            with state.open("wb") as fh:
                pickle.dump((run, rows), fh)
    core.OUT_DIR.mkdir(parents=True, exist_ok=True)
    core.write_csv(rows, core.OUT_DIR / "real_data_study_dev.csv")
    md = report.dev_markdown(run, rows, entry, n_total)
    (core.OUT_DIR / "real_data_study_dev.md").write_text(md, encoding="utf-8", newline="\n")
    print(
        f"wrote docs/results/real_data_study_dev.md and .csv ({len(rows)} rows) in "
        f"{time.perf_counter() - t_all:.0f} s; stages: "
        + ", ".join(f"{k} {v:.0f} s" for k, v in run.timings.items())
    )
    for cid, err in run.excluded.items():
        print(f"excluded {cid}: {err}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", choices=("dev", "holdout"))
    ap.add_argument(
        "--register-holdout",
        action="store_true",
        help="register the holdout and forward studies and write the pre-registration",
    )
    ap.add_argument("--workers", type=int, default=4, help="worker processes (default 4)")
    ap.add_argument(
        "--no-solver-check",
        action="store_true",
        help="skip comparing the fast long-only QP with the library's",
    )
    ap.add_argument(
        "--state",
        type=Path,
        default=None,
        help="development phase: reuse computed results from this pickle if it exists, "
        "else save them there (to rebuild the pages without recomputing)",
    )
    args = ap.parse_args(argv)
    if args.phase == "dev":
        return run_dev(max(1, args.workers), not args.no_solver_check, args.state)
    if args.register_holdout:
        import real_data_prereg as prereg

        return prereg.register()
    if args.phase == "holdout":
        import real_data_prereg as prereg

        return prereg.run_holdout(max(1, args.workers))
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
