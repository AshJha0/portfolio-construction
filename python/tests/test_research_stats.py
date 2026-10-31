"""Tests for portopt.research.stats against hand-computed paper formulas."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats as sps

from portopt.research import (
    annualize_sharpe,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    lo_annualization_factor,
    moments,
    probabilistic_sharpe_ratio,
    psr_from_moments,
    sharpe_bootstrap_ci,
    sharpe_difference_test,
    sharpe_per_period,
    sharpe_se_iid,
    sharpe_se_nonnormal,
)


def phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


@pytest.fixture(scope="module")
def series() -> np.ndarray:
    rng = np.random.default_rng(123)
    return 0.0004 + 0.01 * rng.standard_t(5, size=1000) / math.sqrt(5 / 3)


def test_moments_match_scipy(series):
    mu, sd, g3, g4 = moments(series)
    assert mu == pytest.approx(float(np.mean(series)), rel=1e-12)
    assert sd == pytest.approx(float(np.std(series, ddof=1)), rel=1e-12)
    assert g3 == pytest.approx(float(sps.skew(series, bias=True)), rel=1e-10)
    assert g4 == pytest.approx(float(sps.kurtosis(series, fisher=False, bias=True)), rel=1e-10)
    assert sharpe_per_period(series) == pytest.approx(mu / sd, rel=1e-15)


def test_lo_iid_se_formula(series):
    sr = sharpe_per_period(series)
    assert sharpe_se_iid(series) == pytest.approx(math.sqrt((1 + sr * sr / 2) / 1000), rel=1e-14)
    assert annualize_sharpe(0.1, 252) == pytest.approx(0.1 * math.sqrt(252))


def test_nonnormal_se_formula(series):
    _, _, g3, g4 = moments(series)
    sr = sharpe_per_period(series)
    hand = math.sqrt((1 + sr**2 / 2 - g3 * sr + (g4 - 3) / 4 * sr**2) / 1000)
    assert sharpe_se_nonnormal(series) == pytest.approx(hand, rel=1e-14)
    assert g4 > 3.0  # Student-t(5) is fat tailed


def test_nonnormal_se_reduces_to_iid_for_symmetric_two_point():
    # A symmetric two-point series has g3 = 0 and g4 = 1, so the non-normal
    # variance is 1 + SR^2/2 - SR^2/2 = 1 exactly.
    x = np.tile([1.0, -1.0], 50) + 0.5
    assert sharpe_se_nonnormal(x) == pytest.approx(math.sqrt(1 / 100), rel=1e-12)


def test_psr_hand_computed():
    # Bailey and Lopez de Prado (2012): Phi((SR - SR*) sqrt(T-1) / sqrt(1 - g3 SR + (g4-1)/4 SR^2))
    z = (0.1 - 0.0) * math.sqrt(99) / math.sqrt(1 - (-0.5) * 0.1 + (4.0 - 1) / 4 * 0.01)
    assert psr_from_moments(0.1, 0.0, 100, -0.5, 4.0) == pytest.approx(phi(z), abs=1e-14)
    assert psr_from_moments(0.07, 0.07, 50, 0.3, 6.0) == pytest.approx(0.5, abs=1e-15)
    # Negative skew and fat tails lower the PSR of a positive Sharpe ratio.
    assert psr_from_moments(0.1, 0.0, 100, -1.0, 8.0) < psr_from_moments(0.1, 0.0, 100, 0.0, 3.0)


def test_psr_on_series(series):
    mu, sd, g3, g4 = moments(series)
    assert probabilistic_sharpe_ratio(series) == pytest.approx(
        psr_from_moments(mu / sd, 0.0, 1000, g3, g4), abs=0
    )


def test_expected_max_sharpe_formula_and_monte_carlo():
    g = 0.5772156649015329
    hand = (1 - g) * sps.norm.ppf(1 - 1 / 100) + g * sps.norm.ppf(1 - 1 / (100 * math.e))
    assert expected_max_sharpe(100, 1.0) == pytest.approx(hand, rel=1e-12)
    assert expected_max_sharpe(100, 0.04) == pytest.approx(0.2 * hand, rel=1e-12)
    # The approximation is close to the exact expected maximum of 100 N(0,1): 2.5076.
    assert abs(expected_max_sharpe(100, 1.0) - 2.50759) < 0.03
    assert expected_max_sharpe(1, 1.0) == 0.0


def test_dsr_is_psr_at_expected_max(series):
    sr0 = expected_max_sharpe(50, 1e-4)
    assert deflated_sharpe_ratio(series, n_trials=50, sr_variance=1e-4) == pytest.approx(
        probabilistic_sharpe_ratio(series, sr0), abs=0
    )
    assert deflated_sharpe_ratio(series, n_trials=1, sr_variance=1e-4) == pytest.approx(
        probabilistic_sharpe_ratio(series, 0.0), abs=0
    )
    more = deflated_sharpe_ratio(series, n_trials=500, sr_variance=1e-4)
    assert more < deflated_sharpe_ratio(series, n_trials=50, sr_variance=1e-4)


def test_lo_annualization_hand_computed():
    # x alternates 1, 0: rho_1 = (5 * -0.25) / 1.5 = -5/6, eta(2) = 2 / sqrt(2 + 2 rho_1).
    x = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    assert lo_annualization_factor(x, 2) == pytest.approx(2 / math.sqrt(2 - 10 / 6), rel=1e-14)


def test_lo_annualization_near_sqrt_q_for_white_noise():
    x = np.random.default_rng(5).standard_normal(20000)
    assert lo_annualization_factor(x, 12) == pytest.approx(math.sqrt(12), rel=0.05)


def test_bootstrap_deterministic_with_seed(series):
    a = sharpe_bootstrap_ci(series, seed=11, n_boot=300)
    b = sharpe_bootstrap_ci(series, seed=11, n_boot=300)
    c = sharpe_bootstrap_ci(series, seed=12, n_boot=300)
    assert a == b
    assert (a.lower, a.upper) != (c.lower, c.upper)
    assert a.lower < a.point < a.upper
    circ = sharpe_bootstrap_ci(series, seed=11, n_boot=300, method="circular", block_length=20)
    assert circ.method == "circular" and circ.lower < circ.point < circ.upper
    # The bootstrap standard error is in the neighbourhood of the analytic one.
    assert a.std_error == pytest.approx(sharpe_se_nonnormal(series), rel=0.25)


def test_bootstrap_requires_seed(series):
    with pytest.raises(TypeError):
        sharpe_bootstrap_ci(series)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="seed"):
        sharpe_bootstrap_ci(series, seed=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="block_length"):
        sharpe_bootstrap_ci(series, seed=1, block_length=0.5)
    with pytest.raises(ValueError, match="method"):
        sharpe_bootstrap_ci(series, seed=1, method="iid")


def test_paired_test_identical_series_is_zero(series):
    res = sharpe_difference_test(series, series.copy())
    assert res.diff == 0.0 and res.z == 0.0 and res.p_value == 1.0


def test_paired_test_antisymmetric_and_calibrated():
    rng = np.random.default_rng(9)
    a = 0.0005 + 0.01 * rng.standard_normal(2000)
    b = 0.0001 + 0.01 * rng.standard_normal(2000)
    ab = sharpe_difference_test(a, b, lags=0)
    ba = sharpe_difference_test(b, a, lags=0)
    assert ab.z == pytest.approx(-ba.z, rel=1e-12)
    assert ab.p_value == pytest.approx(ba.p_value, rel=1e-12)
    # Independent iid normal series: Var(SR1 - SR2) ~ (2 + SR1^2/2 + SR2^2/2) / T.
    hand = math.sqrt((2 + ab.sr1**2 / 2 + ab.sr2**2 / 2) / 2000)
    assert ab.std_error == pytest.approx(hand, rel=0.1)
    assert ab.p_value == pytest.approx(2 * (1 - phi(abs(ab.z))), abs=1e-14)
    auto = sharpe_difference_test(a, b)
    assert auto.lags == math.floor(4 * (2000 / 100) ** (2 / 9))


def test_paired_test_detects_large_difference():
    rng = np.random.default_rng(10)
    noise = 0.01 * rng.standard_normal(5000)
    res = sharpe_difference_test(noise + 0.002, noise + 0.01 * rng.standard_normal(5000) * 0.1)
    assert res.p_value < 1e-3


def test_validation():
    with pytest.raises(ValueError, match="1-D"):
        sharpe_per_period(np.zeros((3, 2)))
    with pytest.raises(ValueError, match="constant"):
        sharpe_per_period(np.ones(10))
    with pytest.raises(ValueError, match="NaN"):
        sharpe_se_iid(np.array([0.1, np.nan, 0.2]))
    with pytest.raises(ValueError, match="same length"):
        sharpe_difference_test(np.arange(5.0), np.arange(6.0))
    with pytest.raises(ValueError, match="lags"):
        sharpe_difference_test(np.arange(5.0), np.arange(5.0) ** 2, lags=-1)
    with pytest.raises(ValueError, match="n_trials"):
        expected_max_sharpe(0, 1.0)
    with pytest.raises(ValueError, match="sr_variance"):
        expected_max_sharpe(10, -1.0)
    with pytest.raises(ValueError, match="q must"):
        lo_annualization_factor(np.arange(5.0), 5)
    with pytest.raises(ValueError, match="T must"):
        psr_from_moments(0.1, 0.0, 1, 0.0, 3.0)
