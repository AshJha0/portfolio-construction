"""Real third-party return data: fetch, parse, pin by vintage, guard the holdout.

Part of the Python-only research layer (not covered by API_SPEC.md).

Source.  The Kenneth R. French Data Library publishes daily returns of
industry portfolios and of the Fama/French factors as zipped CSV files.
These are research index series, not investable securities.  The data is
fetched at run time and never committed; the repository commits only
``data/real/manifest.json`` (what was downloaded, when, and its SHA-256)
and ``data/real/registry.json`` (the registered studies).  Raw zips are
cached under ``data/real/raw/<dataset>/<sha256>.zip``, which is
git-ignored.

Vintages.  The library rebuilds the full history every month, so past
returns can change between downloads.  Each distinct zip (by SHA-256) is
recorded in the manifest as a new vintage; an earlier vintage is never
overwritten.  :func:`load_returns` loads one chosen vintage (the most
recently seen by default), re-hashes the cached zip and the parsed arrays
and fails loudly on any mismatch.

Parsing.  A French daily CSV holds a few free-text lines, then one or more
sections (for the industry files: value-weighted, then equal-weighted).
Each section is a header line starting with a comma followed by rows
``YYYYMMDD, v1, v2, ...`` in percent.  Missing values are coded -99.99 or
-999 and become NaN.  Returns are converted to decimal simple returns
(``percent / 100``).  Only 8-digit (daily) dates are accepted.

Holdout.  Rows dated on or after ``holdout_start`` (2016-01-01, frozen
in the manifest on 2026-10-31) are the holdout.  ``load_returns`` serves
the development period freely; ``period="holdout"`` or ``"all"`` needs the
id of a study registered in ``data/real/registry.json`` (see
:func:`portopt.research.walkforward.register_study`), so that every look
at the holdout is attached to a recorded set of trials.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "DATASETS",
    "FRENCH_LIBRARY_URL",
    "HOLDOUT_START",
    "DataError",
    "DatasetSpec",
    "FetchOutcome",
    "HoldoutAccessError",
    "ParsedTable",
    "ReturnsPanel",
    "VintageMismatchError",
    "default_data_dir",
    "download",
    "fetch_dataset",
    "is_registered",
    "load_manifest",
    "load_registry",
    "load_returns",
    "parse_french_csv",
    "parsed_sha256",
    "read_zip_csv",
    "record_vintage",
    "save_manifest",
    "sha256_bytes",
]

FRENCH_LIBRARY_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html"
FRENCH_FTP_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

#: First day of the frozen holdout.  The manifest carries the same value and
#: loading fails if the two disagree, so the boundary cannot drift silently.
HOLDOUT_START = "2016-01-01"

MANIFEST_SCHEMA_VERSION = 1
USER_AGENT = "portopt-research/1.0 (+https://github.com/AshJha0/portfolio-construction)"
MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024
MISSING_CODES = (-99.99, -999.0)
PERIODS = ("development", "holdout", "all")
_PARSED_HASH_TAG = b"portopt-french-parsed-v1\n"
_ROW_RE = re.compile(r"^\s*(\d+)\s*,")
_CRSP_RE = re.compile(r"(\d{6}) CRSP database")


class DataError(RuntimeError):
    """Raised for any problem fetching, caching, parsing or loading real data."""


class VintageMismatchError(DataError):
    """A cached file or its parsed arrays do not match the manifest's record."""


class HoldoutAccessError(DataError):
    """Holdout rows were requested without a registered study id."""


@dataclass(frozen=True)
class DatasetSpec:
    """One downloadable dataset: a zip file and the CSV section to read.

    ``section`` is the exact title line of the section (stripped), or None
    for files with a single unnamed section (the factor files).
    """

    name: str
    filename: str
    section: str | None
    description: str

    @property
    def url(self) -> str:
        return FRENCH_FTP_URL + self.filename


DATASETS: dict[str, DatasetSpec] = {
    spec.name: spec
    for spec in (
        DatasetSpec(
            "ff3_daily",
            "F-F_Research_Data_Factors_daily_CSV.zip",
            None,
            "Fama/French 3 research factors, daily: Mkt-RF, SMB, HML and RF (one-month T-bill)",
        ),
        DatasetSpec(
            "ind10_vw_daily",
            "10_Industry_Portfolios_daily_CSV.zip",
            "Average Value Weighted Returns -- Daily",
            "10 industry portfolios, value-weighted, daily",
        ),
        DatasetSpec(
            "ind49_vw_daily",
            "49_Industry_Portfolios_daily_CSV.zip",
            "Average Value Weighted Returns -- Daily",
            "49 industry portfolios, value-weighted, daily",
        ),
    )
}


@dataclass(frozen=True)
class ParsedTable:
    """One parsed CSV section.

    ``values`` are decimal simple returns (T x n) with NaN where the source
    used a missing code; ``dates`` is ``datetime64[D]``, strictly increasing.
    """

    dates: np.ndarray
    columns: tuple[str, ...]
    values: np.ndarray
    crsp_database: str | None
    n_missing: int


@dataclass(frozen=True)
class ReturnsPanel:
    """Returns of one dataset, one vintage and one period, as loaded."""

    dataset: str
    period: str
    dates: np.ndarray
    columns: tuple[str, ...]
    values: np.ndarray
    vintage_id: int
    zip_sha256: str
    parsed_sha256: str
    crsp_database: str | None
    holdout_start: str
    study_id: str | None

    def column(self, name: str) -> np.ndarray:
        """One column by name (raises KeyError if absent)."""
        try:
            j = self.columns.index(name)
        except ValueError:
            raise KeyError(f"{self.dataset} has no column {name!r}") from None
        out: np.ndarray = self.values[:, j]
        return out


@dataclass(frozen=True)
class FetchOutcome:
    """What :func:`fetch_dataset` did."""

    dataset: str
    vintage_id: int
    zip_sha256: str
    is_new_vintage: bool
    path: Path


# --------------------------------------------------------------------- paths


def default_data_dir() -> Path:
    """``$PORTOPT_REAL_DATA_DIR`` if set, else ``<repo>/data/real``."""
    env = os.environ.get("PORTOPT_REAL_DATA_DIR")
    if env:
        return Path(env)
    # python/src/portopt/research/data.py -> repository root is parents[4].
    return Path(__file__).resolve().parents[4] / "data" / "real"


def _dir(data_dir: str | os.PathLike[str] | None) -> Path:
    return default_data_dir() if data_dir is None else Path(data_dir)


def _raw_path(data_dir: Path, dataset: str, zip_sha: str) -> Path:
    return data_dir / "raw" / dataset / f"{zip_sha}.zip"


def _ensure_raw_ignored(data_dir: Path) -> None:
    """Make ``raw/`` self-ignoring so a stray ``git add -A`` cannot commit data."""
    raw = data_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    marker = raw / ".gitignore"
    if not marker.exists():
        marker.write_text(
            "# Third-party data cached at run time; never committed.\n*\n", encoding="utf-8"
        )


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


# ------------------------------------------------------------------- hashing


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def parsed_sha256(dates: np.ndarray, columns: tuple[str, ...], values: np.ndarray) -> str:
    """SHA-256 of the parsed arrays in a fixed byte layout.

    Layout: a version tag, the JSON list of column names, the dates as
    little-endian int64 days since 1970-01-01, then the values as
    little-endian float64 in row-major order with every NaN replaced by the
    canonical quiet NaN.
    """
    h = hashlib.sha256(_PARSED_HASH_TAG)
    h.update(json.dumps(list(columns)).encode("utf-8"))
    h.update(np.asarray(dates, dtype="datetime64[D]").astype("<i8").tobytes())
    v = np.array(values, dtype="<f8", order="C", copy=True)
    v[np.isnan(v)] = np.float64("nan")
    h.update(v.tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------- parsing


def _is_missing(v: float) -> bool:
    return any(abs(v - code) < 1e-9 for code in MISSING_CODES)


def parse_french_csv(text: str, section: str | None) -> ParsedTable:
    """Parse one daily section of a French Data Library CSV.

    Raises :class:`DataError` on a missing or duplicated section title, a
    missing header, a row with the wrong number of fields, a non-numeric
    value, a date that is not a valid 8-digit ``YYYYMMDD`` (monthly and
    annual sections use 6 and 4 digits and are rejected) or dates that are
    not strictly increasing.
    """
    lines = text.splitlines()
    m = _CRSP_RE.search("\n".join(lines[:5]))
    crsp = m.group(1) if m else None
    start = 0
    if section is not None:
        hits = [i for i, line in enumerate(lines) if line.strip() == section]
        if len(hits) != 1:
            raise DataError(f"expected exactly one section titled {section!r}, found {len(hits)}")
        start = hits[0] + 1
    header_idx = None
    for i in range(start, len(lines)):
        s = lines[i].strip()
        if not s:
            continue
        if s.startswith(","):
            header_idx = i
            break
        if section is not None or _ROW_RE.match(s):
            break
    if header_idx is None:
        raise DataError("no header line (starting with ',') found for the section")
    columns = tuple(c.strip() for c in lines[header_idx].strip().split(",")[1:])
    if not columns or any(not c for c in columns) or len(set(columns)) != len(columns):
        raise DataError(f"bad column header: {lines[header_idx]!r}")
    n = len(columns)
    day_numbers: list[int] = []
    rows: list[list[float]] = []
    epoch = _dt.date(1970, 1, 1)
    for lineno in range(header_idx + 1, len(lines)):
        line = lines[lineno]
        if not _ROW_RE.match(line):
            break
        fields = [f.strip() for f in line.split(",")]
        token = fields[0]
        if len(token) != 8:
            raise DataError(
                f"line {lineno + 1}: date {token!r} is not YYYYMMDD; only daily sections are read"
            )
        try:
            day = _dt.date(int(token[:4]), int(token[4:6]), int(token[6:]))
        except ValueError:
            raise DataError(f"line {lineno + 1}: invalid date {token!r}") from None
        if len(fields) - 1 != n:
            raise DataError(f"line {lineno + 1}: expected {n} values, got {len(fields) - 1}")
        try:
            vals = [float(f) for f in fields[1:]]
        except ValueError:
            raise DataError(f"line {lineno + 1}: non-numeric value in {line!r}") from None
        dn = (day - epoch).days
        if day_numbers and dn <= day_numbers[-1]:
            raise DataError(f"line {lineno + 1}: dates are not strictly increasing at {token}")
        day_numbers.append(dn)
        rows.append([float("nan") if _is_missing(v) else v / 100.0 for v in vals])
    if not rows:
        raise DataError("the section has a header but no data rows")
    values = np.array(rows, dtype=float).reshape(len(rows), n)
    dates = np.array(day_numbers, dtype="int64").astype("datetime64[D]")
    return ParsedTable(dates, columns, values, crsp, int(np.isnan(values).sum()))


def read_zip_csv(payload: bytes) -> tuple[str, str]:
    """Return ``(member_name, text)`` of the single CSV inside a zip."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as exc:
        raise DataError(f"not a valid zip file: {exc}") from None
    with zf:
        members = [i for i in zf.infolist() if i.filename.lower().endswith(".csv")]
        if len(members) != 1:
            raise DataError(f"expected one CSV member in the zip, found {len(members)}")
        info = members[0]
        if info.file_size > MAX_DOWNLOAD_BYTES * 4:
            raise DataError(f"zip member {info.filename} is implausibly large")
        return info.filename, zf.read(info).decode("latin-1")


def _parse_payload(spec: DatasetSpec, payload: bytes) -> tuple[str, ParsedTable]:
    member, text = read_zip_csv(payload)
    return member, parse_french_csv(text, spec.section)


# ------------------------------------------------------------------ download

Opener = Callable[..., Any]


def download(
    url: str,
    *,
    timeout: float = 60.0,
    retries: int = 3,
    backoff_s: float = 2.0,
    urlopen: Opener | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> bytes:
    """GET ``url`` with a user agent, a timeout and exponential-backoff retries.

    Only ``https`` URLs under the French library's ``ftp/`` directory are
    accepted.  A response that is not a zip file counts as a failure (the
    server sometimes answers with an HTML error page).  Raises
    :class:`DataError` after ``retries + 1`` failed attempts.
    """
    if not url.startswith(FRENCH_FTP_URL):
        raise DataError(f"refusing to download from outside the French library: {url}")
    opener = urllib.request.urlopen if urlopen is None else urlopen
    last: Exception | None = None
    for attempt in range(retries + 1):
        if attempt:
            sleep(backoff_s * 2 ** (attempt - 1))
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with opener(req, timeout=timeout) as resp:
                payload = resp.read(MAX_DOWNLOAD_BYTES + 1)
        except urllib.error.HTTPError as exc:
            last = exc
            if 400 <= exc.code < 500 and exc.code not in (408, 429):
                break
            continue
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last = exc
            continue
        if not isinstance(payload, bytes) or len(payload) > MAX_DOWNLOAD_BYTES:
            last = DataError("response missing or larger than the size cap")
            continue
        if not zipfile.is_zipfile(io.BytesIO(payload)):
            last = DataError("response is not a zip file")
            continue
        return payload
    raise DataError(f"download failed for {url}: {last!r}")


# ------------------------------------------------------------------ manifest


def _empty_manifest() -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source": "Kenneth R. French Data Library",
        "source_url": FRENCH_LIBRARY_URL,
        "policy": "Raw data is fetched at run time and never committed; this file records what was fetched.",
        "units": "decimal simple returns (source percent / 100); -99.99 and -999 become NaN",
        "holdout_start": HOLDOUT_START,
        "holdout_frozen_at": "2026-10-31",
        "datasets": {},
    }


def load_manifest(data_dir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Read ``manifest.json``; a missing file yields a fresh, empty manifest."""
    path = _dir(data_dir) / "manifest.json"
    if not path.exists():
        return _empty_manifest()
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or doc.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise DataError(f"{path}: unsupported manifest schema")
    if doc.get("holdout_start") != HOLDOUT_START:
        raise DataError(
            f"{path}: holdout_start is {doc.get('holdout_start')!r} but the frozen value is "
            f"{HOLDOUT_START!r}; the holdout boundary may not be moved"
        )
    return doc


def save_manifest(manifest: dict[str, Any], data_dir: str | os.PathLike[str] | None = None) -> Path:
    path = _dir(data_dir) / "manifest.json"
    text = json.dumps(manifest, indent=2, ensure_ascii=True) + "\n"
    _atomic_write(path, text.encode("utf-8"))
    return path


def _utc_stamp(now: _dt.datetime | None) -> str:
    t = _dt.datetime.now(_dt.UTC) if now is None else now.astimezone(_dt.UTC)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def record_vintage(
    manifest: dict[str, Any],
    spec: DatasetSpec,
    payload: bytes,
    *,
    now: _dt.datetime | None = None,
) -> tuple[dict[str, Any], bool]:
    """Record ``payload`` as a vintage of ``spec`` in ``manifest`` (in place).

    A zip whose SHA-256 is already listed only refreshes that vintage's
    ``last_seen_utc``; a new hash is appended as a new vintage.  Existing
    vintage records are never altered otherwise.  The dataset-level fields
    describe the vintage seen most recently (``current_vintage``).
    Returns ``(vintage_record, is_new)``.
    """
    member, table = _parse_payload(spec, payload)
    zip_sha = sha256_bytes(payload)
    stamp = _utc_stamp(now)
    ds = manifest["datasets"].setdefault(
        spec.name,
        {
            "url": spec.url,
            "description": spec.description,
            "section": spec.section,
            "vintages": [],
        },
    )
    if ds.get("url") != spec.url:
        raise DataError(f"manifest URL for {spec.name} differs from the code: {ds.get('url')}")
    existing = [v for v in ds["vintages"] if v["zip_sha256"] == zip_sha]
    if existing:
        vin = existing[0]
        vin["last_seen_utc"] = stamp
        is_new = False
    else:
        vin = {
            "vintage_id": len(ds["vintages"]) + 1,
            "zip_sha256": zip_sha,
            "parsed_sha256": parsed_sha256(table.dates, table.columns, table.values),
            "downloaded_at_utc": stamp,
            "last_seen_utc": stamp,
            "zip_bytes": len(payload),
            "zip_member": member,
            "crsp_database": table.crsp_database,
            "first_date": str(table.dates[0]),
            "last_date": str(table.dates[-1]),
            "rows": int(table.values.shape[0]),
            "columns": list(table.columns),
            "n_missing": table.n_missing,
        }
        ds["vintages"].append(vin)
        is_new = True
    ds.update(
        {
            "current_vintage": vin["vintage_id"],
            "downloaded_at_utc": vin["downloaded_at_utc"],
            "zip_sha256": vin["zip_sha256"],
            "first_date": vin["first_date"],
            "last_date": vin["last_date"],
            "rows": vin["rows"],
            "columns": vin["columns"],
        }
    )
    return vin, is_new


def fetch_dataset(
    name: str,
    *,
    data_dir: str | os.PathLike[str] | None = None,
    now: _dt.datetime | None = None,
    fetcher: Callable[[str], bytes] | None = None,
) -> FetchOutcome:
    """Download ``name``, cache the zip by hash and record its vintage.

    ``fetcher(url) -> bytes`` replaces :func:`download` (tests use it to
    stay off the network).  The manifest is written back on success.
    """
    spec = _spec(name)
    root = _dir(data_dir)
    payload = (download if fetcher is None else fetcher)(spec.url)
    manifest = load_manifest(root)
    vin, is_new = record_vintage(manifest, spec, payload, now=now)
    _ensure_raw_ignored(root)
    path = _raw_path(root, spec.name, vin["zip_sha256"])
    if not path.exists():
        _atomic_write(path, payload)
    save_manifest(manifest, root)
    return FetchOutcome(spec.name, int(vin["vintage_id"]), vin["zip_sha256"], is_new, path)


# ------------------------------------------------------------------ registry


def load_registry(data_dir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Read ``registry.json``; a missing file yields an empty registry."""
    path = _dir(data_dir) / "registry.json"
    if not path.exists():
        return {"schema_version": 1, "studies": []}
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or not isinstance(doc.get("studies"), list):
        raise DataError(f"{path}: malformed registry")
    return doc


def is_registered(study_id: str, data_dir: str | os.PathLike[str] | None = None) -> bool:
    return any(s.get("id") == study_id for s in load_registry(data_dir)["studies"])


# ------------------------------------------------------------------- loading


def _spec(name: str) -> DatasetSpec:
    try:
        return DATASETS[name]
    except KeyError:
        raise DataError(f"unknown dataset {name!r}; known: {sorted(DATASETS)}") from None


def _select_vintage(ds: dict[str, Any], vintage: int | str | None) -> dict[str, Any]:
    vins: list[dict[str, Any]] = ds["vintages"]
    if vintage is None:
        vintage = int(ds["current_vintage"])
    if isinstance(vintage, bool):
        raise DataError("vintage must be an id or a SHA-256, not a bool")
    if isinstance(vintage, int):
        hits = [v for v in vins if v["vintage_id"] == vintage]
    else:
        if len(vintage) < 12:
            raise DataError("a vintage hash prefix must have at least 12 hex digits")
        hits = [v for v in vins if v["zip_sha256"].startswith(vintage.lower())]
    if len(hits) != 1:
        raise DataError(f"vintage {vintage!r} matches {len(hits)} recorded vintages")
    return hits[0]


def load_returns(
    dataset: str,
    period: str = "development",
    *,
    study_id: str | None = None,
    vintage: int | str | None = None,
    data_dir: str | os.PathLike[str] | None = None,
) -> ReturnsPanel:
    """Load one dataset at one recorded vintage, cut to ``period``.

    ``period`` is ``"development"`` (dates before ``holdout_start``),
    ``"holdout"`` (on or after it) or ``"all"``.  The last two require
    ``study_id`` to name a study in ``registry.json``.  ``vintage`` is a
    vintage id, a SHA-256 (or a prefix of at least 12 hex digits), or None
    for the vintage seen most recently.

    Raises :class:`HoldoutAccessError` when the holdout is requested
    without a registered study, :class:`VintageMismatchError` when the
    cached zip or its parsed arrays do not hash to the manifest's values,
    and :class:`DataError` for anything else (unknown dataset or vintage,
    zip not in the local cache).
    """
    if period not in PERIODS:
        raise DataError(f"period must be one of {PERIODS}, got {period!r}")
    spec = _spec(dataset)
    root = _dir(data_dir)
    if period != "development":
        if not isinstance(study_id, str) or not study_id:
            raise HoldoutAccessError(
                f"period={period!r} reads the holdout; pass study_id= of a registered study"
            )
        if not is_registered(study_id, root):
            raise HoldoutAccessError(f"study {study_id!r} is not in {root / 'registry.json'}")
    manifest = load_manifest(root)
    ds = manifest["datasets"].get(spec.name)
    if not ds or not ds.get("vintages"):
        raise DataError(f"{spec.name} has no recorded vintage; run scripts/fetch_data.py")
    vin = _select_vintage(ds, vintage)
    path = _raw_path(root, spec.name, vin["zip_sha256"])
    if not path.exists():
        raise DataError(
            f"{path} is not in the local cache.  The French library serves only its current "
            "vintage, so an older vintage can be loaded only from a cache that already holds it; "
            "run scripts/fetch_data.py for the current one."
        )
    payload = path.read_bytes()
    got = sha256_bytes(payload)
    if got != vin["zip_sha256"]:
        raise VintageMismatchError(
            f"{path}: SHA-256 {got} does not match vintage {vin['vintage_id']} "
            f"({vin['zip_sha256']}); the cache is corrupt or was replaced"
        )
    _, table = _parse_payload(spec, payload)
    psha = parsed_sha256(table.dates, table.columns, table.values)
    if psha != vin["parsed_sha256"]:
        raise VintageMismatchError(
            f"{spec.name} vintage {vin['vintage_id']}: parsed arrays hash to {psha}, manifest has "
            f"{vin['parsed_sha256']}; the parser changed since the vintage was recorded"
        )
    hs = np.datetime64(manifest["holdout_start"], "D")
    if period == "development":
        mask = table.dates < hs
    elif period == "holdout":
        mask = table.dates >= hs
    else:
        mask = np.ones(table.dates.shape[0], dtype=bool)
    if not mask.any():
        raise DataError(f"{spec.name} vintage {vin['vintage_id']} has no rows in {period!r}")
    return ReturnsPanel(
        dataset=spec.name,
        period=period,
        dates=table.dates[mask],
        columns=table.columns,
        values=table.values[mask],
        vintage_id=int(vin["vintage_id"]),
        zip_sha256=vin["zip_sha256"],
        parsed_sha256=psha,
        crsp_database=table.crsp_database,
        holdout_start=str(manifest["holdout_start"]),
        study_id=study_id if period != "development" else None,
    )
