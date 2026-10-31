"""Tests for portopt.research.estimators."""

from __future__ import annotations

import math

import numpy as np
import pytest

import portopt as po
from portopt.research import estimators as est
from portopt.research.estimators import (
    _epanechnikov_hilbert,
    _kernel_density_and_hilbert,
    _null_eigenvalue,
    _oracle_eigenvalues,
    _singular_oracle,
)


def gaussian(t: int, n: int, seed: int, scale: float = 0.01) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return scale * rng.standard_normal((t, n))


def correlated(t: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n))
    sig = a @ a.T / n + 0.2 * np.eye(n)
    low = np.linalg.cholesky(sig)
    return 0.01 * rng.standard_normal((t, n)) @ low.T + 0.0005 * rng.standard_normal(n)


@pytest.fixture(scope="module")
def x6() -> np.ndarray:
    return correlated(300, 6, seed=11)


# --------------------------------------------------------------------------
# EWMA
# --------------------------------------------------------------------------


def test_ewma_decay_values():
    assert est.ewma_decay(1.0) == 0.5
    assert est.ewma_decay(math.inf) == 1.0
    # RiskMetrics daily lambda = 0.94 is a half-life of ln(0.5)/ln(0.94) days.
    assert est.ewma_decay(math.log(0.5) / math.log(0.94)) == pytest.approx(0.94, abs=1e-15)


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, -math.inf])
def test_ewma_decay_rejects(bad):
    with pytest.raises(ValueError, match="half_life"):
        est.ewma_decay(bad)


def test_ewma_weights_shape():
    w = est.ewma_weights(5, 2.0)
    assert math.fsum(w.tolist()) == pytest.approx(1.0, abs=1e-15)
    lam = est.ewma_decay(2.0)
    np.testing.assert_allclose(w[:-1] / w[1:], lam, rtol=1e-14)
    assert w[-1] == w.max()
    np.testing.assert_allclose(est.ewma_weights(4, math.inf), 0.25, rtol=0, atol=1e-16)
    for bad in (0, 1.5, True):
        with pytest.raises(ValueError, match="t must be"):
            est.ewma_weights(bad, 2.0)


def test_ewma_zero_mean_hand_case():
    # T = 3, N = 1, half-life 1 -> lambda = 0.5, raw weights (0.25, 0.5, 1) / 1.75.
    r = np.array([[0.01], [-0.02], [0.03]])
    got = est.ewma_covariance(r, 1.0, demean=False)
    want = (0.25 * 0.0001 + 0.5 * 0.0004 + 1.0 * 0.0009) / 1.75
    assert got.shape == (1, 1)
    assert got[0, 0] == pytest.approx(want, rel=1e-14)


def test_ewma_demeaned_matches_numpy_reliability_weights(x6):
    w = est.ewma_weights(x6.shape[0], 20.0)
    got = est.ewma_covariance(x6, 20.0)
    want = np.cov(x6.T, aweights=w, ddof=1)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-18)


def test_ewma_zero_mean_equals_renormalised_riskmetrics_recursion(x6):
    lam = est.ewma_decay(15.0)
    t, n = x6.shape
    sig = np.zeros((n, n))
    for row in x6:
        sig = lam * sig + (1.0 - lam) * np.outer(row, row)
    # Recursion started at zero carries total weight 1 - lam^T; renormalising removes it.
    want = sig / (1.0 - lam**t)
    np.testing.assert_allclose(est.ewma_covariance(x6, 15.0, demean=False), want, rtol=1e-11)


def test_ewma_half_life_to_infinity_is_sample_cov(x6):
    s = po.sample_cov(x6)
    np.testing.assert_allclose(est.ewma_covariance(x6, math.inf), s, rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(est.ewma_covariance(x6, 1e9), s, rtol=1e-6)
    far = np.max(np.abs(est.ewma_covariance(x6, 10.0) - s))
    near = np.max(np.abs(est.ewma_covariance(x6, 1e4) - s))
    assert near < far


def test_ewma_invariances_and_psd(x6):
    base = est.ewma_covariance(x6, 30.0)
    np.testing.assert_allclose(est.ewma_covariance(3.0 * x6, 30.0), 9.0 * base, rtol=1e-12)
    perm = np.array([3, 0, 5, 1, 4, 2])
    np.testing.assert_allclose(
        est.ewma_covariance(x6[:, perm], 30.0), base[np.ix_(perm, perm)], rtol=1e-12
    )
    assert np.all(np.linalg.eigvalsh(base) > 0.0)
    np.testing.assert_array_equal(base, base.T)


def test_ewma_validation():
    with pytest.raises(ValueError, match="rows"):
        est.ewma_covariance(np.zeros((1, 2)), 5.0)
    with pytest.raises(ValueError, match="NaN"):
        est.ewma_covariance(np.array([[0.0, math.nan], [0.0, 0.0]]), 5.0)
    with pytest.raises(ValueError, match="half_life"):
        est.ewma_covariance(np.zeros((5, 2)), -2.0)


def test_ewma_constant_correlation(x6):
    got = est.ewma_constant_correlation(x6, 25.0)
    ew = est.ewma_covariance(x6, 25.0)
    np.testing.assert_allclose(np.diag(got), np.diag(ew), rtol=1e-14)
    s = po.sample_cov(x6)
    sd = np.sqrt(np.diag(s))
    corr = s / np.outer(sd, sd)
    rho = corr[np.triu_indices(6, 1)].mean()
    vol = np.sqrt(np.diag(ew))
    implied = got / np.outer(vol, vol)
    off = implied[~np.eye(6, dtype=bool)]
    np.testing.assert_allclose(off, rho, rtol=1e-12)
    assert np.all(np.linalg.eigvalsh(got) > 0.0)
    one = x6[:, :1]
    np.testing.assert_allclose(
        est.ewma_constant_correlation(one, 25.0), est.ewma_covariance(one, 25.0)
    )
    flat = x6.copy()
    flat[:, 2] = 0.001
    with pytest.raises(ValueError, match="positive sample variances"):
        est.ewma_constant_correlation(flat, 25.0)


# --------------------------------------------------------------------------
# Mean shrinkage
# --------------------------------------------------------------------------


def bayes_stein_reference(x: np.ndarray) -> tuple[np.ndarray, float]:
    t, n = x.shape
    mu = x.mean(axis=0)
    sig = np.cov(x.T, ddof=1) * (t - 1) / (t - n - 2)
    inv = np.linalg.inv(sig)
    one = np.ones(n)
    mu0 = one @ inv @ mu / (one @ inv @ one)
    d = mu - mu0
    phi = (n + 2) / ((n + 2) + t * d @ inv @ d)
    return (1 - phi) * mu + phi * mu0, phi


def test_bayes_stein_matches_reference(x6):
    res = est.bayes_stein_mean(x6)
    mean, phi = bayes_stein_reference(x6)
    assert res.weight == pytest.approx(phi, rel=1e-10)
    np.testing.assert_allclose(res.mean, mean, rtol=1e-10)
    np.testing.assert_allclose(res.sample, x6.mean(axis=0), rtol=1e-14)
    assert 0.0 < res.weight <= 1.0
    # The grand mean is the mean return of the sample minimum-variance portfolio.
    w = po.min_variance_weights(po.sample_cov(x6))
    assert res.grand_mean == pytest.approx(float(w @ res.sample), rel=1e-9)


def test_bayes_stein_invariances(x6):
    base = est.bayes_stein_mean(x6)
    scaled = est.bayes_stein_mean(2.5 * x6)
    assert scaled.weight == pytest.approx(base.weight, rel=1e-10)
    np.testing.assert_allclose(scaled.mean, 2.5 * base.mean, rtol=1e-10)
    perm = np.array([5, 4, 3, 2, 1, 0])
    p = est.bayes_stein_mean(x6[:, perm])
    assert p.weight == pytest.approx(base.weight, rel=1e-10)
    np.testing.assert_allclose(p.mean, base.mean[perm], rtol=1e-10)


def test_bayes_stein_equal_means_shrinks_fully(x6):
    flat = x6 - x6.mean(axis=0) + 0.0004
    res = est.bayes_stein_mean(flat)
    assert res.weight == pytest.approx(1.0, abs=1e-9)
    np.testing.assert_allclose(res.mean, 0.0004, rtol=1e-8)


def test_bayes_stein_weight_falls_with_sample_size():
    rng = np.random.default_rng(5)
    mus = np.array([0.0, 0.01, 0.02, -0.01])
    weights = []
    for t in (50, 500, 5000):
        x = 0.02 * rng.standard_normal((t, 4)) + mus
        weights.append(est.bayes_stein_mean(x).weight)
    assert weights[0] > weights[1] > weights[2]
    assert weights[2] < 0.05


def test_bayes_stein_validation():
    with pytest.raises(ValueError, match="T > N \\+ 2"):
        est.bayes_stein_mean(gaussian(5, 3, 0))
    with pytest.raises(ValueError, match="at least 2 assets"):
        est.bayes_stein_mean(gaussian(50, 1, 0))
    x = gaussian(50, 3, 0)
    x[:, 2] = x[:, 1]
    with pytest.raises(ValueError, match="positive definite"):
        est.bayes_stein_mean(x)


def test_james_stein_hand_case():
    # Means 0, 1, 2, 3 (+ noise-free deviations of +-1): hand-computed weight.
    base = np.array([[1.0, -1.0, 1.0, -1.0], [-1.0, 1.0, -1.0, 1.0]]) * 0.01
    x = np.vstack([base, base]) + np.array([0.0, 0.001, 0.002, 0.003])
    res = est.james_stein_mean(x)
    t, n = x.shape
    s2 = np.trace(np.cov(x.T, ddof=1)) / n
    mu = x.mean(axis=0)
    dss = np.sum((mu - mu.mean()) ** 2)
    want = min(1.0, (n - 3) * s2 / (t * dss))
    assert res.weight == pytest.approx(want, rel=1e-12)
    assert res.grand_mean == pytest.approx(0.0015, rel=1e-12)
    np.testing.assert_allclose(res.mean, (1 - want) * mu + want * 0.0015, rtol=1e-12)


def test_james_stein_properties(x6):
    res = est.james_stein_mean(x6)
    assert 0.0 <= res.weight <= 1.0
    sc = est.james_stein_mean(4.0 * x6)
    assert sc.weight == pytest.approx(res.weight, rel=1e-12)
    perm = np.array([1, 0, 3, 2, 5, 4])
    assert est.james_stein_mean(x6[:, perm]).weight == pytest.approx(res.weight, rel=1e-12)
    flat = x6 - x6.mean(axis=0)
    assert est.james_stein_mean(flat).weight == 1.0
    with pytest.raises(ValueError, match="at least 4 assets"):
        est.james_stein_mean(x6[:, :3])


# --------------------------------------------------------------------------
# Ledoit-Wolf toward the identity
# --------------------------------------------------------------------------


def lw_identity_loop(x: np.ndarray) -> float:
    t, n = x.shape
    xc = x - x.mean(axis=0)
    s = xc.T @ xc / t
    m = np.trace(s) / n
    d2 = np.sum((s - m * np.eye(n)) ** 2) / n
    bbar2 = sum(np.sum((np.outer(r, r) - s) ** 2) / n for r in xc) / t**2
    return float(min(bbar2, d2) / d2)


def test_lw_identity_matches_loop(x6):
    res = est.ledoit_wolf_identity(x6)
    assert res.intensity == pytest.approx(lw_identity_loop(x6), rel=1e-10)
    s = po.sample_cov(x6)
    np.testing.assert_allclose(res.sample, s, rtol=1e-12)
    assert res.scale == pytest.approx(np.trace(s) / 6, rel=1e-12)
    want = res.intensity * res.scale * np.eye(6) + (1 - res.intensity) * s
    np.testing.assert_allclose(res.sigma, want, rtol=1e-12)


def test_lw_identity_matches_sklearn(x6):
    cov = pytest.importorskip("sklearn.covariance")
    sk_sigma, sk_delta = cov.ledoit_wolf(x6)
    res = est.ledoit_wolf_identity(x6)
    t = x6.shape[0]
    assert res.intensity == pytest.approx(sk_delta, rel=1e-9)
    np.testing.assert_allclose(res.sigma * (t - 1) / t, sk_sigma, rtol=1e-9)


def test_lw_identity_bounds_and_limits():
    small = est.ledoit_wolf_identity(gaussian(12, 10, 3))
    big = est.ledoit_wolf_identity(correlated(20000, 4, 3))
    for r in (small, big):
        assert 0.0 <= r.intensity <= 1.0
        assert np.all(np.linalg.eigvalsh(r.sigma) > 0.0)
    assert big.intensity < 0.01
    # Columns orthogonal with equal norms: the sample is already m I, so intensity = 1.
    h = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]]) * 0.01
    r = est.ledoit_wolf_identity(h)
    assert r.intensity == 1.0
    np.testing.assert_allclose(r.sigma, r.scale * np.eye(2), atol=1e-18)


def test_lw_identity_invariances(x6):
    base = est.ledoit_wolf_identity(x6)
    sc = est.ledoit_wolf_identity(0.5 * x6)
    assert sc.intensity == pytest.approx(base.intensity, rel=1e-10)
    np.testing.assert_allclose(sc.sigma, 0.25 * base.sigma, rtol=1e-10)
    perm = np.array([2, 4, 0, 1, 5, 3])
    p = est.ledoit_wolf_identity(x6[:, perm])
    np.testing.assert_allclose(p.sigma, base.sigma[np.ix_(perm, perm)], rtol=1e-10)


def test_lw_identity_validation():
    with pytest.raises(ValueError, match="positive average sample variance"):
        est.ledoit_wolf_identity(np.full((5, 3), 0.01))
    with pytest.raises(ValueError, match="rows"):
        est.ledoit_wolf_identity(np.zeros((1, 3)))


# --------------------------------------------------------------------------
# Ledoit-Wolf (2020) analytical nonlinear shrinkage
# --------------------------------------------------------------------------


def _k(x: float) -> float:
    return 3.0 / (4.0 * math.sqrt(5.0)) * max(1.0 - x * x / 5.0, 0.0)


@pytest.mark.parametrize("x", [-3.7, -2.0, -0.6, 0.0, 0.3, 1.9, 2.2, 4.0, 11.0])
def test_kernel_hilbert_matches_numerical_principal_value(x):
    integrate = pytest.importorskip("scipy.integrate")
    r5 = math.sqrt(5.0)
    if -r5 < x < r5:
        pv, _ = integrate.quad(_k, -r5, r5, weight="cauchy", wvar=x, epsabs=1e-13)
    else:
        pv, _ = integrate.quad(lambda t: _k(t) / (t - x), -r5, r5, epsabs=1e-13)
    got = float(_epanechnikov_hilbert(np.array([x]))[0])
    assert got == pytest.approx(pv / math.pi, rel=1e-9, abs=1e-12)


def test_kernel_hilbert_edges_and_tail():
    r5 = math.sqrt(5.0)
    edge = _epanechnikov_hilbert(np.array([-r5, r5]))
    assert np.isfinite(edge).all()
    near = _epanechnikov_hilbert(np.array([-r5 + 1e-9, r5 - 1e-9]))
    np.testing.assert_allclose(edge, near, atol=1e-7)
    big = 1e3
    tail = float(_epanechnikov_hilbert(np.array([big]))[0])
    assert tail == pytest.approx(-1.0 / (math.pi * big), rel=1e-5)


def mp_density(c: float):
    a, b = (1 - math.sqrt(c)) ** 2, (1 + math.sqrt(c)) ** 2

    def f(t: float) -> float:
        return math.sqrt(max((b - t) * (t - a), 0.0)) / (2 * math.pi * c * t)

    return f, a, b


@pytest.mark.parametrize("c", [0.1, 0.5, 0.9])
def test_oracle_formula_returns_identity_on_marchenko_pastur(c):
    # Population covariance I: the limiting sample spectrum is Marchenko-Pastur
    # with ratio c, and the oracle maps every eigenvalue back to 1.  The Hilbert
    # transform is computed by numerical principal-value integration.
    integrate = pytest.importorskip("scipy.integrate")
    f, a, b = mp_density(c)
    lam = np.linspace(a, b, 9)[1:-1]
    fv = np.array([f(x) for x in lam])
    hv = (
        np.array(
            [
                integrate.quad(f, a, b, weight="cauchy", wvar=x, epsabs=1e-12, limit=200)[0]
                for x in lam
            ]
        )
        / math.pi
    )
    np.testing.assert_allclose(_oracle_eigenvalues(lam, fv, hv, c), 1.0, rtol=1e-6)


@pytest.mark.parametrize("c", [1.5, 3.0])
def test_singular_oracle_returns_identity_on_marchenko_pastur(c):
    # For c > 1 the n non-zero eigenvalues of S = X'X/n are c times
    # Marchenko-Pastur(1/c); the formula should again map them, and the null
    # eigenvalues, to 1.
    integrate = pytest.importorskip("scipy.integrate")
    g0, a0, b0 = mp_density(1.0 / c)

    def g(t: float) -> float:
        return g0(t / c) / c

    a, b = c * a0, c * b0
    lam = np.linspace(a, b, 9)[1:-1]
    gv = np.array([g(x) for x in lam])
    hv = (
        np.array(
            [
                integrate.quad(g, a, b, weight="cauchy", wvar=x, epsabs=1e-12, limit=200)[0]
                for x in lam
            ]
        )
        / math.pi
    )
    np.testing.assert_allclose(_singular_oracle(lam, gv, hv), 1.0, rtol=1e-6)
    hf0 = integrate.quad(lambda t: g(t) / t, a, b, epsabs=1e-13, limit=200)[0] / math.pi
    assert _null_eigenvalue(c, hf0) == pytest.approx(1.0, rel=1e-7)


def direct_nonlinear(x: np.ndarray) -> np.ndarray:
    """Shrunk eigenvalues from scratch: kernel density by summation, its
    Hilbert transform by numerical principal-value integration of each
    kernel (no closed form used), then the 2020 formula (case N <= T - 1)."""
    integrate = pytest.importorskip("scipy.integrate")
    t, p = x.shape
    n = t - 1
    xc = x - x.mean(axis=0)
    lam = np.sort(np.linalg.eigvalsh(xc.T @ xc / n))
    h = n ** (-1.0 / 3.0)
    c = p / n
    r5 = math.sqrt(5.0)
    out = []
    for li in lam:
        f = 0.0
        hf = 0.0
        for lj in lam:
            bw = h * lj
            f += _k((li - lj) / bw) / bw
            # PV int k((s - lj)/bw)/bw / (s - li) ds, substituted s = lj + bw z.
            z0 = (li - lj) / bw
            if -r5 < z0 < r5:
                v = integrate.quad(_k, -r5, r5, weight="cauchy", wvar=z0, epsabs=1e-12)[0]
            else:
                v = integrate.quad(lambda z, z0=z0: _k(z) / (z - z0), -r5, r5, epsabs=1e-12)[0]
            v /= bw
            hf += v / math.pi
        f /= p
        hf /= p
        out.append(li / ((math.pi * c * li * f) ** 2 + (1 - c - math.pi * c * li * hf) ** 2))
    return np.array(out)


def test_nonlinear_matches_direct_numerical_formula():
    x = correlated(40, 5, seed=21)
    res = est.nonlinear_shrinkage(x)
    np.testing.assert_allclose(res.shrunk, direct_nonlinear(x), rtol=1e-8)


def test_kernel_density_matches_loops():
    lam = np.array([0.5, 0.9, 1.0, 1.7, 3.0])
    h = 0.4
    f, hf = _kernel_density_and_hilbert(lam, h)
    for i, li in enumerate(lam):
        fi = sum(_k((li - lj) / (h * lj)) / (h * lj) for lj in lam) / lam.size
        hi = (
            sum(
                float(_epanechnikov_hilbert(np.array([(li - lj) / (h * lj)]))[0]) / (h * lj)
                for lj in lam
            )
            / lam.size
        )
        assert f[i] == pytest.approx(fi, rel=1e-14)
        assert hf[i] == pytest.approx(hi, rel=1e-12, abs=1e-14)


@pytest.mark.parametrize(("p", "t"), [(200, 1001), (400, 201)])
def test_nonlinear_recovers_identity(p, t):
    x = gaussian(t, p, seed=p)
    res = est.nonlinear_shrinkage(x)
    ev = np.linalg.eigvalsh(res.sigma) / 1e-4
    sample = res.sample_eigenvalues / 1e-4
    assert sample.max() > 2.0
    assert 0.75 < ev.min() and ev.max() < 1.6
    assert abs(ev.mean() - 1.0) < 0.1
    assert np.std(ev) < 0.3 * np.std(sample)


def test_nonlinear_properties(x6):
    x = correlated(60, 20, seed=8)
    for data in (x, x6, gaussian(10, 30, 1)):
        res = est.nonlinear_shrinkage(data)
        # Positive definite, symmetric.
        assert np.all(np.linalg.eigvalsh(res.sigma) > 0.0)
        np.testing.assert_array_equal(res.sigma, res.sigma.T)
        # Sample eigenvectors are the estimator's eigenvectors.
        u = res.eigenvectors
        np.testing.assert_allclose(
            res.sigma @ u, u * res.shrunk, rtol=1e-9, atol=1e-12 * res.shrunk.max()
        )
    res = est.nonlinear_shrinkage(x)
    lam = res.sample_eigenvalues
    # Shrinks dispersion: narrower range, closer to the grand average.
    assert res.shrunk.max() < lam.max()
    assert res.shrunk.min() > lam.min()
    assert np.std(res.shrunk) < np.std(lam)
    # The largest and smallest stay at the ends of the shrunk spectrum.
    assert np.argmax(res.shrunk) == lam.size - 1
    assert res.concentration == pytest.approx(20 / 59)
    assert res.bandwidth == pytest.approx(59 ** (-1 / 3))


def test_nonlinear_near_sample_when_t_much_larger_than_n():
    x = correlated(40001, 5, seed=4)
    res = est.nonlinear_shrinkage(x)
    np.testing.assert_allclose(res.shrunk, res.sample_eigenvalues, rtol=2e-2)
    s = po.sample_cov(x)
    assert np.max(np.abs(res.sigma - s)) < 2e-2 * np.max(np.abs(s))


def test_nonlinear_invariances_and_determinism():
    x = correlated(80, 12, seed=13)
    base = est.nonlinear_shrinkage(x)
    sc = est.nonlinear_shrinkage(7.0 * x)
    np.testing.assert_allclose(sc.sigma, 49.0 * base.sigma, rtol=1e-9, atol=1e-15)
    perm = np.random.default_rng(0).permutation(12)
    p = est.nonlinear_shrinkage(x[:, perm])
    np.testing.assert_allclose(p.sigma, base.sigma[np.ix_(perm, perm)], rtol=1e-8, atol=1e-14)
    again = est.nonlinear_shrinkage(x)
    np.testing.assert_array_equal(again.sigma, base.sigma)
    # Singular case is also invariant to the basis chosen for the null space.
    y = gaussian(15, 25, seed=2)
    q = np.random.default_rng(1).permutation(25)
    a = est.nonlinear_shrinkage(y)
    b = est.nonlinear_shrinkage(y[:, q])
    np.testing.assert_allclose(b.sigma, a.sigma[np.ix_(q, q)], rtol=1e-8, atol=1e-12)


def test_nonlinear_validation():
    x = gaussian(50, 4, seed=0)
    x[:, 3] = x[:, 0]
    with pytest.raises(ValueError, match="zero eigenvalue"):
        est.nonlinear_shrinkage(x)
    with pytest.raises(ValueError, match="rows"):
        est.nonlinear_shrinkage(np.zeros((1, 4)))
    with pytest.raises(ValueError, match="<= -1"):
        est.nonlinear_shrinkage(np.full((5, 2), -1.0))


# --------------------------------------------------------------------------
# PCA factor model
# --------------------------------------------------------------------------


def test_pca_structure(x6):
    res = est.pca_factor_covariance(x6, 2)
    s = po.sample_cov(x6)
    b = res.loadings
    np.testing.assert_allclose(b.T @ b, np.eye(2), atol=1e-12)
    top = np.sort(np.linalg.eigvalsh(s))[::-1][:2]
    np.testing.assert_allclose(res.factor_variances, top, rtol=1e-12)
    common = b @ np.diag(res.factor_variances) @ b.T
    np.testing.assert_allclose(res.sigma, common + np.diag(res.specific_variances), rtol=1e-12)
    # Residual diagonal above the floor: the model reproduces the sample variances.
    np.testing.assert_allclose(np.diag(res.sigma), np.diag(s), rtol=1e-12)
    assert res.explained == pytest.approx(top.sum() / np.trace(s), rel=1e-12)
    assert 0.0 < res.explained < 1.0
    assert np.all(np.linalg.eigvalsh(res.sigma) > 0.0)
    for j in range(2):
        assert b[np.argmax(np.abs(b[:, j])), j] > 0.0


def test_pca_full_rank_uses_floor(x6):
    s = po.sample_cov(x6)
    res = est.pca_factor_covariance(x6, 6, specific_floor=1e-3)
    np.testing.assert_allclose(res.specific_variances, 1e-3 * np.diag(s), rtol=1e-12)
    np.testing.assert_allclose(res.sigma, s + np.diag(1e-3 * np.diag(s)), rtol=1e-10, atol=1e-18)
    assert res.explained == pytest.approx(1.0, rel=1e-12)


def test_pca_invariances(x6):
    base = est.pca_factor_covariance(x6, 3)
    np.testing.assert_allclose(
        est.pca_factor_covariance(2.0 * x6, 3).sigma, 4.0 * base.sigma, rtol=1e-10
    )
    perm = np.array([4, 2, 0, 5, 3, 1])
    p = est.pca_factor_covariance(x6[:, perm], 3)
    np.testing.assert_allclose(p.sigma, base.sigma[np.ix_(perm, perm)], rtol=1e-10)
    np.testing.assert_array_equal(est.pca_factor_covariance(x6, 3).sigma, base.sigma)


@pytest.mark.parametrize("k", [0, 7, 1.5, True, -1])
def test_pca_rejects_k(x6, k):
    with pytest.raises(ValueError, match="k must be"):
        est.pca_factor_covariance(x6, k)


@pytest.mark.parametrize("fl", [0.0, 1.0, -0.1, math.nan])
def test_pca_rejects_floor(x6, fl):
    with pytest.raises(ValueError, match="specific_floor"):
        est.pca_factor_covariance(x6, 2, specific_floor=fl)


def test_pca_rejects_zero_variance(x6):
    flat = x6.copy()
    flat[:, 0] = 0.003
    with pytest.raises(ValueError, match="positive sample variances"):
        est.pca_factor_covariance(flat, 2)
