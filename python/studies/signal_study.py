"""Return-signal study on real index data (round 2).

Run from the repository root::

    python python/studies/signal_study.py --phase dev       # 1926-2015 only
    python python/studies/signal_study.py --register        # pre-registration
    python python/studies/signal_study.py --phase seen      # 2016-01..2026-09, once

``--phase dev`` registers the whole grid as ``r2-signal-dev`` before it
computes anything, reads only ``period="development"`` and writes
the development rows of ``docs/results/signal_study.csv`` and
``docs/results/signal_study.md``.

``--register`` registers the configurations and hypotheses of
``signal_prereg.py`` as ``r2-signal-v1`` and the forward rule files as
``r2-forward-signals-v1``.  It reads no 2016+ row.

``--phase seen`` evaluates only the ``r2-signal-v1`` configurations on
2016-01 to 2026-09.  That period is not a clean holdout: the analyst saw
the construction study's results on it before choosing these signals.

Series are non-investable Kenneth R. French index portfolios; costs are
assumptions.  BLAS is pinned to one thread for determinism.
"""

from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import real_data_core as rc  # noqa: E402
import signal_core as sc  # noqa: E402
import signal_report as srep  # noqa: E402

from portopt.research import data as rd  # noqa: E402
from portopt.research import walkforward as wf  # noqa: E402

CSV = sc.OUT_DIR / "signal_study.csv"


def _panels(period: str, study_id: str | None) -> dict[str, rc.Panel]:
    return {u: rc.load_panel(rc.UNIVERSES[u], period, study_id, rc.DATA_VINTAGE)
            for u in rc.UNIVERSES}


def phase_dev(workers: int) -> None:
    grid = sc.dev_grid()
    if not rd.is_registered(sc.DEV_STUDY_ID):
        wf.register_study(
            sc.DEV_STUDY_ID,
            "Round 2 signal exploration on 1926-07-01..2015-12-31 (period='development' only): "
            "every configuration evaluated by python/studies/signal_study.py --phase dev, "
            "registered before the first run.",
            [c.to_json() for c in grid],
        )
    t0 = time.time()
    panels = _panels("development", None)
    res = sc.run_all(panels, grid, None, workers)
    rows = srep.dev_rows(panels, grid, res)
    old = srep.read_csv(CSV) if CSV.exists() else []
    srep.write_csv(rows + [r for r in old if r["period"].startswith("seen")], CSV)
    print(f"dev: {len(grid)} configs in {time.time() - t0:.0f} s")


def phase_seen(workers: int) -> None:
    import signal_prereg as pr

    old = srep.read_csv(CSV)
    if any(r["period"].startswith("seen") for r in old):
        raise SystemExit(f"{CSV} already holds seen-period rows: the run is done once")
    pr.check_registered()
    t0 = time.time()
    panels = _panels("all", sc.SEEN_STUDY_ID)
    cfgs = [sc.config_from_id(c) for c in pr.SEEN_CONFIGS]
    res = sc.run_all(panels, cfgs, sc.SEEN_START, workers)
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          cwd=rc.ROOT).stdout.strip()
    rows = srep.seen_rows(panels, cfgs, res)
    for r in rows:
        r["git_head"] = head
    srep.write_csv(old + rows, CSV)
    print(f"seen: {len(cfgs)} configs in {time.time() - t0:.0f} s at {head}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("dev", "seen"))
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    if a.phase == "dev":
        phase_dev(a.workers)
    if a.register:
        import signal_prereg as pr

        pr.register()
    if a.phase == "seen":
        phase_seen(a.workers)
    if a.report or a.phase:
        (sc.OUT_DIR / "signal_study.md").write_text(srep.render(srep.read_csv(CSV)), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
