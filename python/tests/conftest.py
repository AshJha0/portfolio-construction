"""Shared fixtures: bundled data, moments, and golden cases."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

import portopt as po

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "data")


@pytest.fixture(scope="session")
def returns() -> np.ndarray:
    return np.loadtxt(os.path.join(DATA_DIR, "asset_returns.csv"), delimiter=",", skiprows=1)


@pytest.fixture(scope="session")
def asset_names() -> list[str]:
    with open(os.path.join(DATA_DIR, "asset_returns.csv")) as fh:
        return fh.readline().strip().split(",")


@pytest.fixture(scope="session")
def config() -> dict:
    with open(os.path.join(DATA_DIR, "config.json")) as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def views_doc() -> dict:
    with open(os.path.join(DATA_DIR, "views.json")) as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def market_caps() -> np.ndarray:
    caps = []
    with open(os.path.join(DATA_DIR, "market_caps.csv")) as fh:
        next(fh)
        for line in fh:
            caps.append(float(line.strip().split(",")[1]))
    return np.array(caps)


@pytest.fixture(scope="session")
def golden() -> dict:
    with open(os.path.join(DATA_DIR, "golden", "golden.json")) as fh:
        doc = json.load(fh)
    return {c["name"]: c for c in doc["cases"]}


@pytest.fixture(scope="session")
def moments(returns) -> dict:
    """Annualized sample moments used by most portfolio-level tests."""
    mu = po.annualize_mean(po.sample_mean(returns))
    sig = po.annualize_cov(po.sample_cov(returns))
    return {"mu": mu, "sigma": sig}
