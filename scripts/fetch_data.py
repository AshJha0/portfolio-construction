"""Fetch Kenneth French Data Library daily files and record their vintages.

Usage (from the repository root)::

    python scripts/fetch_data.py                      # all known datasets
    python scripts/fetch_data.py ind10_vw_daily ff3_daily
    python scripts/fetch_data.py --list

Each zip is downloaded over HTTPS with a timeout, retries and a user
agent, cached under ``data/real/raw/<dataset>/<sha256>.zip`` (git-ignored)
and recorded in ``data/real/manifest.json``.  A zip whose hash is new is
added as a new vintage; earlier vintages are never overwritten.  The raw
data is never committed.  See ``data/real/README.md``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from portopt.research import data as rd
except ImportError:  # running from a checkout without `pip install -e python`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))
    from portopt.research import data as rd


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("datasets", nargs="*", help="dataset names (default: all)")
    ap.add_argument("--data-dir", type=Path, default=None, help="default: data/real")
    ap.add_argument("--list", action="store_true", help="list known datasets and exit")
    args = ap.parse_args(argv)
    if args.list:
        for spec in rd.DATASETS.values():
            print(f"{spec.name:16s} {spec.url}\n{'':16s} {spec.description}")
        return 0
    names = args.datasets or list(rd.DATASETS)
    unknown = [n for n in names if n not in rd.DATASETS]
    if unknown:
        print(f"unknown dataset(s): {unknown}; known: {sorted(rd.DATASETS)}", file=sys.stderr)
        return 2
    failed = 0
    for name in names:
        try:
            out = rd.fetch_dataset(name, data_dir=args.data_dir)
        except rd.DataError as exc:
            print(f"{name}: FAILED: {exc}", file=sys.stderr)
            failed += 1
            continue
        manifest = rd.load_manifest(args.data_dir)
        ds = manifest["datasets"][name]
        status = "new vintage" if out.is_new_vintage else "unchanged"
        print(
            f"{name}: vintage {out.vintage_id} ({status}) sha256 {out.zip_sha256[:16]}... "
            f"{ds['first_date']} to {ds['last_date']}, {ds['rows']} rows x {len(ds['columns'])} cols"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
