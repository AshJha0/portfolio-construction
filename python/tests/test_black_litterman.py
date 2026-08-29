"""Tests for Black-Litterman reverse optimization and posterior."""
import numpy as np
import pytest

import portopt as po


@pytest.fixture(scope="module")
def bl_setup(moments, market_caps, views_doc, asset_names):
    mu, sig = moments["mu"], moments["sigma"]
    w_mkt = market_caps / market_caps.sum()
    delta = views_doc["delta"]
    tau = views_doc["tau"]
    pi = po.implied_equilibrium_returns(sig, w_mkt, delta)
    p, q, conf = po.views_to_matrices(views_doc["views"], asset_names)
    return {
        "sigma": sig,
        "w_mkt": w_mkt,
        "pi": pi,
        "p": p,
        "q": q,
        "conf": conf,
        "tau": tau,
        "delta": delta,
    }


def test_implied_pi_reverse_roundtrip(bl_setup):
    """Tangency at rf=0 on (pi, Sigma) must recover the market weights."""
    s = bl_setup
    w = po.tangency_weights(s["pi"], s["sigma"], 0.0)
    assert np.allclose(w, s["w_mkt"], atol=1e-10)


def test_views_to_matrices(bl_setup, asset_names):
    p, q = bl_setup["p"], bl_setup["q"]
    assert p.shape == (2, len(asset_names))
    assert np.isclose(p[0].sum(), 1.0) and p[0, asset_names.index("EM_EQ")] == 1.0
    assert np.isclose(p[1].sum(), 0.0)
    assert p[1, asset_names.index("US_EQ")] == 1.0
    assert p[1, asset_names.index("EU_EQ")] == -1.0
    assert q[0] == pytest.approx(0.10) and q[1] == pytest.approx(0.02)


def test_default_omega_diagonal_and_confidence_scaling(bl_setup):
    s = bl_setup
    om1 = po.default_omega(s["sigma"], s["p"], s["tau"], None)
    om2 = po.default_omega(s["sigma"], s["p"], s["tau"], [2.0, 4.0])
    core = s["p"] @ (s["tau"] * s["sigma"]) @ s["p"].T
    assert np.allclose(np.diag(om1), np.diag(core), atol=1e-15)
    assert np.count_nonzero(om1 - np.diag(np.diag(om1))) == 0
    assert np.allclose(np.diag(om2), np.diag(core) / np.array([2.0, 4.0]), atol=1e-15)


def test_posterior_moves_toward_views(bl_setup, asset_names):
    s = bl_setup
    res = po.bl_posterior(s["sigma"], s["pi"], s["p"], s["q"], s["tau"], confidences=s["conf"])
    i_em = asset_names.index("EM_EQ")
    # The absolute view q=10% exceeds equilibrium pi -> posterior EM return rises.
    assert s["q"][0] > s["pi"][i_em]
    assert res.mu_post[i_em] > s["pi"][i_em]
    view_gap_prior = s["p"][0] @ s["pi"] - s["q"][0]
    view_gap_post = s["p"][0] @ res.mu_post - s["q"][0]
    assert abs(view_gap_post) < abs(view_gap_prior)


def test_posterior_m_is_psd_and_shrinks_uncertainty(bl_setup):
    s = bl_setup
    res = po.bl_posterior(s["sigma"], s["pi"], s["p"], s["q"], s["tau"], confidences=s["conf"])
    eig = np.linalg.eigvalsh(res.m)
    assert np.all(eig > -1e-14)  # M PSD
    # M <= tau Sigma (views reduce estimation variance)
    eig2 = np.linalg.eigvalsh(s["tau"] * s["sigma"] - res.m)
    assert np.all(eig2 > -1e-14)


def test_tau_to_zero_limit_fixed_omega(bl_setup):
    s = bl_setup
    om = po.default_omega(s["sigma"], s["p"], s["tau"], s["conf"])
    res = po.bl_posterior(s["sigma"], s["pi"], s["p"], s["q"], 1e-10, omega=om)
    assert np.max(np.abs(res.mu_post - s["pi"])) < 1e-8
    assert np.max(np.abs(res.m)) < 1e-8


def test_omega_to_infinity_returns_prior(bl_setup):
    """Zero-confidence views: posterior collapses to the equilibrium prior."""
    s = bl_setup
    big = np.diag(np.full(2, 1e14))
    res = po.bl_posterior(s["sigma"], s["pi"], s["p"], s["q"], s["tau"], omega=big)
    assert np.max(np.abs(res.mu_post - s["pi"])) < 1e-10
    assert np.max(np.abs(res.m - s["tau"] * s["sigma"])) < 1e-10
    # Tiny confidences through the default Omega behave the same way.
    res2 = po.bl_posterior(
        s["sigma"], s["pi"], s["p"], s["q"], s["tau"], confidences=[1e-10, 1e-10]
    )
    assert np.max(np.abs(res2.mu_post - s["pi"])) < 1e-8


def test_conflicting_extreme_views_stay_finite(bl_setup):
    s = bl_setup
    n = s["sigma"].shape[0]
    p = np.zeros((2, n))
    p[0, 0] = 1.0
    p[1, 0] = 1.0  # same asset, contradictory targets
    q = np.array([0.50, -0.50])
    res = po.bl_posterior(s["sigma"], s["pi"], p, q, s["tau"])
    assert np.isfinite(res.mu_post).all()
    assert np.isfinite(res.sigma_post).all()
    w, _ = po.bl_weights(s["sigma"], s["pi"], p, q, s["tau"])
    assert np.isfinite(w).all()
    assert w.sum() == pytest.approx(1.0, abs=1e-10)


def test_bl_weights_no_views_equals_market(bl_setup):
    """A single vacuous (huge-Omega) view keeps weights at the market."""
    s = bl_setup
    p = s["p"]
    big = np.diag(np.full(2, 1e14))
    w, _ = po.bl_weights(s["sigma"], s["pi"], p, s["q"], s["tau"], omega=big)
    assert np.allclose(w, s["w_mkt"], atol=1e-6)


def test_bl_error_paths(bl_setup):
    s = bl_setup
    n = s["sigma"].shape[0]
    zero_row = np.zeros((1, n))
    with pytest.raises(ValueError, match="identically zero"):
        po.bl_posterior(s["sigma"], s["pi"], zero_row, np.array([0.05]), s["tau"])
    for bad_tau in (0.0, -0.1, 1.5, np.nan):
        with pytest.raises(ValueError, match="tau"):
            po.bl_posterior(s["sigma"], s["pi"], s["p"], s["q"], bad_tau)
    with pytest.raises(ValueError, match="confidence"):
        po.default_omega(s["sigma"], s["p"], s["tau"], [1.0, -1.0])
    with pytest.raises(ValueError, match="non-negative"):
        po.implied_equilibrium_returns(s["sigma"], -np.ones(n), 2.5)
    with pytest.raises(ValueError, match="delta"):
        po.implied_equilibrium_returns(s["sigma"], np.ones(n), -2.5)
    with pytest.raises(ValueError, match="length"):
        po.bl_posterior(s["sigma"], s["pi"], s["p"], np.array([0.05]), s["tau"])
    with pytest.raises(ValueError, match="non-empty"):
        po.views_to_matrices([], ["A"])
