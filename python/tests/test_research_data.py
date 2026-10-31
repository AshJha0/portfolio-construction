"""Tests for portopt.research.data (French library fetch, parse, vintages, holdout guard).

No test touches the network: downloads go through injected fakes and an
autouse fixture makes any real ``urlopen`` call fail.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from portopt.research import data as rd
from portopt.research.walkforward import register_study

NOW = dt.datetime(2026, 10, 31, 12, 0, tzinfo=dt.UTC)
IND_COLS = ("NoDur", "HiTec", "Utils")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("tests must not touch the network")

    monkeypatch.setattr(urllib.request, "urlopen", boom)


# ------------------------------------------------------------ fake French files


def business_days(start: str, end: str) -> np.ndarray:
    d0 = np.busday_offset(np.datetime64(start, "D"), 0, roll="forward")
    n = int(np.busday_count(d0, np.datetime64(end, "D") + np.timedelta64(1, "D")))
    return np.busday_offset(d0, np.arange(n), roll="forward")


def fake_percent(dates: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Percent returns rounded to 2 decimals, a pure function of (date, column)."""
    days = dates.astype("int64")
    out = np.empty((dates.shape[0], n))
    for j in range(n):
        rng_vals = np.sin(days * (0.37 + 0.11 * j) + seed) * (0.8 + 0.2 * j)
        out[:, j] = np.round(rng_vals + 0.03, 2)
    return out


def _rows(dates: np.ndarray, pct: np.ndarray) -> list[str]:
    lines = []
    for d, row in zip(dates, pct, strict=True):
        ymd = str(d).replace("-", "")
        lines.append(ymd + "," + ",".join(f"{v:8.2f}" for v in row))
    return lines


def industry_text(dates: np.ndarray, vw_pct: np.ndarray, crsp: str, cols=IND_COLS) -> str:
    header = "," + ",".join(cols)
    ew = np.round(vw_pct * 0.5, 2)
    parts = [
        f"This file was created using the {crsp} CRSP database.",
        f"It contains value- and equal-weighted returns for {len(cols)} industry portfolios.",
        "",
        "Missing data are indicated by -99.99 or -999.",
        "",
        "",
        "  Average Value Weighted Returns -- Daily",
        header,
        *_rows(dates, vw_pct),
        "",
        "  Average Equal Weighted Returns -- Daily",
        header,
        *_rows(dates, ew),
        "",
        "Copyright 2026 Eugene F. Fama and Kenneth R. French",
    ]
    return "\r\n".join(parts) + "\r\n"


def factors_text(dates: np.ndarray, pct: np.ndarray, crsp: str) -> str:
    parts = [
        f"This file was created by using the {crsp} CRSP database.",
        "The Tbill return is the simple daily rate.",
        "",
        ",Mkt-RF,SMB,HML,RF",
        *_rows(dates, pct),
        "",
        "Copyright 2026 Eugene F. Fama and Kenneth R. French",
    ]
    return "\r\n".join(parts) + "\r\n"


def make_zip(member: str, text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        info = zipfile.ZipInfo(member, date_time=(2026, 10, 1, 0, 0, 0))
        zf.writestr(info, text.encode("latin-1"))
    return buf.getvalue()


def fake_payloads(
    end: str, crsp: str, *, tweak: tuple[str, int, float] | None = None
) -> dict[str, bytes]:
    """Zips for ind10_vw_daily and ff3_daily covering 2014-01-02 to ``end``.

    ``tweak = (date, column, new_percent)`` restates one industry value.
    """
    dates = business_days("2014-01-02", end)
    ind = fake_percent(dates, len(IND_COLS), seed=1)
    if tweak is not None:
        i = int(np.nonzero(dates == np.datetime64(tweak[0], "D"))[0][0])
        ind[i, tweak[1]] = tweak[2]
    ff = fake_percent(dates, 4, seed=7)
    ff[:, 3] = 0.01  # RF, percent per day
    return {
        rd.DATASETS["ind10_vw_daily"].url: make_zip(
            "10_Industry_Portfolios_Daily.csv", industry_text(dates, ind, crsp)
        ),
        rd.DATASETS["ff3_daily"].url: make_zip(
            "F-F_Research_Data_Factors_daily.csv", factors_text(dates, ff, crsp)
        ),
    }


def install(data_dir: Path, payloads: dict[str, bytes], now: dt.datetime = NOW) -> None:
    for name in ("ind10_vw_daily", "ff3_daily"):
        rd.fetch_dataset(name, data_dir=data_dir, now=now, fetcher=payloads.__getitem__)


# ------------------------------------------------------------------- parsing


def test_parse_reads_value_weighted_daily_section_in_decimals() -> None:
    dates = business_days("2015-12-28", "2016-01-06")
    pct = fake_percent(dates, 3, seed=1)
    pct[1, 2] = -99.99
    pct[2, 0] = -999.0
    t = rd.parse_french_csv(
        industry_text(dates, pct, "202609"), rd.DATASETS["ind10_vw_daily"].section
    )
    assert t.columns == IND_COLS
    assert t.crsp_database == "202609"
    np.testing.assert_array_equal(t.dates, dates)
    assert t.n_missing == 2
    assert np.isnan(t.values[1, 2]) and np.isnan(t.values[2, 0])
    ok = ~np.isnan(t.values)
    np.testing.assert_allclose(t.values[ok], (pct / 100.0)[ok], rtol=0, atol=1e-15)


def test_parse_factor_file_without_section_title() -> None:
    dates = business_days("2020-01-02", "2020-01-10")
    pct = fake_percent(dates, 4, seed=3)
    t = rd.parse_french_csv(factors_text(dates, pct, "202001"), None)
    assert t.columns == ("Mkt-RF", "SMB", "HML", "RF")
    assert t.values.shape == (dates.shape[0], 4)


@pytest.mark.parametrize(
    ("bad_line", "message"),
    [
        ("201601,1.0,2.0,3.0", "YYYYMMDD"),
        ("20160231,1.0,2.0,3.0", "invalid date"),
        ("20160111,1.0,2.0", "expected 3 values"),
        ("20160111,1.0,x,3.0", "non-numeric"),
        ("20151201,1.0,2.0,3.0", "strictly increasing"),
    ],
)
def test_parse_rejects_malformed_rows(bad_line: str, message: str) -> None:
    dates = business_days("2016-01-04", "2016-01-08")
    text = industry_text(dates, fake_percent(dates, 3, seed=1), "202601")
    lines = text.split("\r\n")
    i = lines.index("  Average Equal Weighted Returns -- Daily") - 1  # the blank after VW rows
    lines.insert(i, bad_line)
    with pytest.raises(rd.DataError, match=message):
        rd.parse_french_csv("\r\n".join(lines), "Average Value Weighted Returns -- Daily")


def test_parse_rejects_missing_section() -> None:
    dates = business_days("2016-01-04", "2016-01-08")
    text = industry_text(dates, fake_percent(dates, 3, seed=1), "202601")
    with pytest.raises(rd.DataError, match="exactly one section"):
        rd.parse_french_csv(text, "Average Value Weighted Returns -- Monthly")


def test_parsed_hash_is_stable_and_sensitive() -> None:
    dates = business_days("2016-01-04", "2016-01-08")
    v = fake_percent(dates, 3, seed=1) / 100
    v[0, 0] = np.nan
    h = rd.parsed_sha256(dates, IND_COLS, v)
    assert h == rd.parsed_sha256(dates.copy(), IND_COLS, v.copy())
    w = v.copy()
    w[1, 1] += 1e-12
    assert rd.parsed_sha256(dates, IND_COLS, w) != h
    assert rd.parsed_sha256(dates, ("a", "b", "c"), v) != h


def test_read_zip_csv_requires_one_csv_member() -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("a.csv", "x")
        zf.writestr("b.csv", "y")
    with pytest.raises(rd.DataError, match="one CSV member"):
        rd.read_zip_csv(buf.getvalue())
    with pytest.raises(rd.DataError, match="not a valid zip"):
        rd.read_zip_csv(b"<html>error</html>")


# ------------------------------------------------------------------ download


class _Resp:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self, n: int = -1) -> bytes:
        return self.payload


def test_download_retries_then_succeeds_with_user_agent() -> None:
    good = make_zip("x.csv", "hello")
    calls: list[urllib.request.Request] = []
    sleeps: list[float] = []

    def opener(req: urllib.request.Request, timeout: float) -> _Resp:
        calls.append(req)
        assert timeout == 5.0
        if len(calls) == 1:
            raise urllib.error.URLError("temporary")
        if len(calls) == 2:
            return _Resp(b"<html>busy</html>")
        return _Resp(good)

    url = rd.DATASETS["ff3_daily"].url
    out = rd.download(
        url, timeout=5.0, retries=3, backoff_s=1.0, urlopen=opener, sleep=sleeps.append
    )
    assert out == good
    assert len(calls) == 3 and sleeps == [1.0, 2.0]
    assert calls[0].get_header("User-agent") == rd.USER_AGENT


def test_download_gives_up_and_does_not_retry_404() -> None:
    n = {"calls": 0}

    def opener(req: urllib.request.Request, timeout: float) -> _Resp:
        n["calls"] += 1
        raise urllib.error.HTTPError(req.full_url, 404, "nf", None, None)  # type: ignore[arg-type]

    with pytest.raises(rd.DataError, match="download failed"):
        rd.download(rd.DATASETS["ff3_daily"].url, urlopen=opener, sleep=lambda s: None)
    assert n["calls"] == 1


def test_download_refuses_foreign_urls() -> None:
    with pytest.raises(rd.DataError, match="outside the French library"):
        rd.download("https://example.com/x.zip", urlopen=lambda *a, **k: None)


# --------------------------------------------------------- manifest/vintages


def test_fetch_records_vintages_and_never_overwrites(tmp_path: Path) -> None:
    p1 = fake_payloads("2016-03-31", "201603")
    install(tmp_path, p1)
    m = rd.load_manifest(tmp_path)
    assert m["holdout_start"] == "2016-01-01"
    ds = m["datasets"]["ind10_vw_daily"]
    assert ds["current_vintage"] == 1 and len(ds["vintages"]) == 1
    v1 = dict(ds["vintages"][0])
    assert v1["zip_sha256"] == rd.sha256_bytes(p1[rd.DATASETS["ind10_vw_daily"].url])
    assert v1["first_date"] == "2014-01-02" and v1["last_date"] == "2016-03-31"
    assert v1["columns"] == list(IND_COLS) and v1["crsp_database"] == "201603"
    assert (tmp_path / "raw" / ".gitignore").read_text().strip().endswith("*")

    later = NOW + dt.timedelta(days=30)
    install(tmp_path, p1, now=later)  # same bytes: same vintage, only last_seen moves
    ds = rd.load_manifest(tmp_path)["datasets"]["ind10_vw_daily"]
    assert len(ds["vintages"]) == 1
    assert ds["vintages"][0]["last_seen_utc"] == "2026-11-30T12:00:00Z"
    assert ds["vintages"][0]["downloaded_at_utc"] == v1["downloaded_at_utc"]

    p2 = fake_payloads("2016-04-29", "201604")
    out = rd.fetch_dataset("ind10_vw_daily", data_dir=tmp_path, now=later, fetcher=p2.__getitem__)
    assert out.is_new_vintage and out.vintage_id == 2
    ds = rd.load_manifest(tmp_path)["datasets"]["ind10_vw_daily"]
    assert [v["vintage_id"] for v in ds["vintages"]] == [1, 2]
    first = ds["vintages"][0]
    assert {k: first[k] for k in v1 if k != "last_seen_utc"} == {
        k: v1[k] for k in v1 if k != "last_seen_utc"
    }
    assert ds["current_vintage"] == 2 and ds["last_date"] == "2016-04-29"


# ------------------------------------------------------------------- loading


@pytest.fixture()
def installed(tmp_path: Path) -> Path:
    install(tmp_path, fake_payloads("2016-03-31", "201603"))
    return tmp_path


def test_development_period_stops_before_holdout(installed: Path) -> None:
    p = rd.load_returns("ind10_vw_daily", data_dir=installed)
    assert p.period == "development" and p.study_id is None
    assert p.dates[-1] < np.datetime64("2016-01-01") and p.dates[0] == np.datetime64("2014-01-02")
    assert p.columns == IND_COLS and p.vintage_id == 1


def test_holdout_requires_registered_study(installed: Path) -> None:
    for period in ("holdout", "all"):
        with pytest.raises(rd.HoldoutAccessError, match="registered study"):
            rd.load_returns("ind10_vw_daily", period, data_dir=installed)
        with pytest.raises(rd.HoldoutAccessError, match="not in"):
            rd.load_returns("ind10_vw_daily", period, study_id="nope", data_dir=installed)
    register_study("s-001", "test study", [{"rule": "erc"}], data_dir=installed, now=NOW)
    h = rd.load_returns("ind10_vw_daily", "holdout", study_id="s-001", data_dir=installed)
    assert h.dates[0] >= np.datetime64("2016-01-01") and h.study_id == "s-001"
    a = rd.load_returns("ind10_vw_daily", "all", study_id="s-001", data_dir=installed)
    d = rd.load_returns("ind10_vw_daily", data_dir=installed)
    assert a.values.shape[0] == d.values.shape[0] + h.values.shape[0]


def test_vintage_selection_and_cache_checks(installed: Path) -> None:
    p2 = fake_payloads("2016-04-29", "201604")
    rd.fetch_dataset("ind10_vw_daily", data_dir=installed, now=NOW, fetcher=p2.__getitem__)
    latest = rd.load_returns("ind10_vw_daily", data_dir=installed)
    assert latest.vintage_id == 2
    ds = rd.load_manifest(installed)["datasets"]["ind10_vw_daily"]
    sha1 = ds["vintages"][0]["zip_sha256"]
    assert rd.load_returns("ind10_vw_daily", vintage=1, data_dir=installed).zip_sha256 == sha1
    assert rd.load_returns("ind10_vw_daily", vintage=sha1[:12], data_dir=installed).vintage_id == 1
    with pytest.raises(rd.DataError, match="matches 0"):
        rd.load_returns("ind10_vw_daily", vintage=9, data_dir=installed)

    raw = installed / "raw" / "ind10_vw_daily" / f"{sha1}.zip"
    raw.write_bytes(raw.read_bytes() + b"x")
    with pytest.raises(rd.VintageMismatchError, match="does not match vintage 1"):
        rd.load_returns("ind10_vw_daily", vintage=1, data_dir=installed)
    raw.unlink()
    with pytest.raises(rd.DataError, match="not in the local cache"):
        rd.load_returns("ind10_vw_daily", vintage=1, data_dir=installed)


def test_parsed_hash_mismatch_fails_loudly(installed: Path) -> None:
    path = installed / "manifest.json"
    m = json.loads(path.read_text())
    m["datasets"]["ind10_vw_daily"]["vintages"][0]["parsed_sha256"] = "0" * 64
    path.write_text(json.dumps(m))
    with pytest.raises(rd.VintageMismatchError, match="parser changed"):
        rd.load_returns("ind10_vw_daily", data_dir=installed)


def test_holdout_boundary_cannot_move(installed: Path) -> None:
    path = installed / "manifest.json"
    m = json.loads(path.read_text())
    m["holdout_start"] = "2020-01-01"
    path.write_text(json.dumps(m))
    with pytest.raises(rd.DataError, match="may not be moved"):
        rd.load_returns("ind10_vw_daily", data_dir=installed)


def test_unknown_dataset_and_period(installed: Path) -> None:
    with pytest.raises(rd.DataError, match="unknown dataset"):
        rd.load_returns("nope", data_dir=installed)
    with pytest.raises(rd.DataError, match="period must be"):
        rd.load_returns("ind10_vw_daily", "train", data_dir=installed)
