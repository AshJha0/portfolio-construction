"""portopt end-to-end demo.

Run:  cd python && PYTHONPATH=src python3 demo.py

Prints: estimator table, unconstrained + long-only frontier sweep,
Black-Litterman prior vs posterior weights, ERC vs inverse-vol risk
contributions, and a quarterly-rebalance comparison across six methods.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import portopt as po

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data")


def rule(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 66 - len(title)))


def wrow(label: str, w: np.ndarray) -> str:
    return f"{label:<14}" + "".join(f"{x:>9.4f}" for x in w)


def main() -> int:
    x = np.loadtxt(os.path.join(DATA, "asset_returns.csv"), delimiter=",", skiprows=1)
    with open(os.path.join(DATA, "asset_returns.csv")) as fh:
        names = fh.readline().strip().split(",")
    with open(os.path.join(DATA, "config.json")) as fh:
        cfg = json.load(fh)
    with open(os.path.join(DATA, "views.json")) as fh:
        views_doc = json.load(fh)
    caps = np.loadtxt(
        os.path.join(DATA, "market_caps.csv"), delimiter=",", skiprows=1, usecols=1
    )
    rf = cfg["risk_free_rate"]

    print("portopt demo — 8-asset universe, 1250 daily returns")
    print(f"assets: {', '.join(names)}")

    # ------------------------------------------------------------- estimators
    rule("Estimators (annualized, Ledoit-Wolf constant-correlation shrinkage)")
    mu = po.annualize_mean(po.sample_mean(x))
    shrink = po.shrink_covariance(x)
    sig = po.annualize_cov(po.sample_cov(x))
    sig_sh = po.annualize_cov(shrink.sigma)
    print(f"shrinkage intensity delta = {shrink.intensity:.6f}   rbar = {shrink.rbar:.4f}")
    print(f"{'asset':<10}{'mean':>9}{'vol(sample)':>13}{'vol(shrunk)':>13}")
    for i, name in enumerate(names):
        print(
            f"{name:<10}{mu[i]:>9.4f}{np.sqrt(sig[i, i]):>13.4f}"
            f"{np.sqrt(sig_sh[i, i]):>13.4f}"
        )

    # --------------------------------------------------------------- frontier
    rule("Efficient frontier sweeps (25 points each, showing every 4th)")
    targets, _, var_u = po.efficient_frontier(mu, sig, cfg["frontier_points"])
    targets_c, w_c, var_c = po.constrained_frontier(mu, sig, cfg["frontier_points"])
    print("unconstrained grid: A/C .. max(mu);  long-only grid: constrained GMV .. max(mu)")
    print(f"{'target':>8}{'vol uncon':>11}{'target':>10}{'vol long-only':>15}")
    for k in range(0, len(targets), 4):
        print(
            f"{targets[k]:>8.4f}{np.sqrt(var_u[k]):>11.4f}"
            f"{targets_c[k]:>10.4f}{np.sqrt(var_c[k]):>15.4f}"
        )
    print(f"long-only targets met exactly: max |mu'w - target| = {np.max(np.abs(w_c @ mu - targets_c)):.1e}")
    w_tan = po.tangency_weights(mu, sig, rf)
    print(
        f"tangency (rf={rf:.2%}): sharpe = "
        f"{(mu @ w_tan - rf) / np.sqrt(w_tan @ sig @ w_tan):.4f}, "
        f"gross leverage = {po.gross_leverage(w_tan):.2f}"
    )

    # -------------------------------------------------------- black-litterman
    rule("Black-Litterman: prior (equilibrium) vs posterior weights")
    w_mkt = caps / caps.sum()
    pi = po.implied_equilibrium_returns(sig, w_mkt, views_doc["delta"])
    p, q, conf = po.views_to_matrices(views_doc["views"], names)
    for v in views_doc["views"]:
        if v["type"] == "absolute":
            print(f"view: {v['asset']} returns {v['q']:.1%} (confidence {v['confidence']})")
        else:
            print(
                f"view: {v['long']} beats {v['short']} by {v['q']:.1%} "
                f"(confidence {v['confidence']})"
            )
    w_bl, res = po.bl_weights(sig, pi, p, q, views_doc["tau"], confidences=conf)
    print(f"{'':14}" + "".join(f"{n[:8]:>9}" for n in names))
    print(wrow("pi (equil.)", pi))
    print(wrow("mu posterior", res.mu_post))
    print(wrow("w market", w_mkt))
    print(wrow("w posterior", w_bl))
    print(f"posterior gross leverage = {po.gross_leverage(w_bl):.4f}")

    # ------------------------------------------------------------- risk parity
    rule("Risk contributions: ERC vs inverse-vol")
    w_erc = po.erc_weights(sig)
    w_iv = po.inverse_vol_weights(sig)
    rc_erc = po.risk_contributions(w_erc, sig)
    rc_iv = po.risk_contributions(w_iv, sig)
    print(f"{'':14}" + "".join(f"{n[:8]:>9}" for n in names))
    print(wrow("w ERC", w_erc))
    print(wrow("RC ERC", rc_erc))
    print(wrow("w inv-vol", w_iv))
    print(wrow("RC inv-vol", rc_iv))
    print(
        f"RC spread: ERC {np.max(rc_erc) - np.min(rc_erc):.2e}, "
        f"inv-vol {np.max(rc_iv) - np.min(rc_iv):.2e}"
    )

    # ---------------------------------------------------------------- backtest
    rule("Quarterly-rebalance backtest, expanding window (holdout = last 750d)")

    fallbacks = {"n": 0}

    def tangency_fn(h: np.ndarray) -> np.ndarray:
        """Pinned backtest method: tangency, falling back to unconstrained
        min-variance for a rebalance where tangency is undefined or
        ill-conditioned — reported, not crashed (API_SPEC 6.3).  Any other
        error (e.g. a singular covariance) propagates."""
        s = po.annualize_cov(po.sample_cov(h))
        try:
            return po.tangency_weights(po.annualize_mean(po.sample_mean(h)), s, rf)
        except po.TangencyUndefinedError:
            fallbacks["n"] += 1
            return po.min_variance_weights(s)

    def bl_fn(h: np.ndarray) -> np.ndarray:
        s = po.annualize_cov(po.sample_cov(h))
        pi_h = po.implied_equilibrium_returns(s, w_mkt, views_doc["delta"])
        w, _ = po.bl_weights(s, pi_h, p, q, views_doc["tau"], confidences=conf)
        return w

    methods = {
        "equal_weight": lambda h: np.full(h.shape[1], 1.0 / h.shape[1]),
        "min_var_lo": lambda h: po.min_variance_longonly(
            po.annualize_cov(po.sample_cov(h))
        ),
        "tangency": tangency_fn,
        "black_litt": bl_fn,
        "erc": lambda h: po.erc_weights(po.sample_cov(h)),
        "inverse_vol": lambda h: po.inverse_vol_weights(po.sample_cov(h)),
    }
    print(
        f"{'method':<14}{'ann ret':>9}{'ann vol':>9}{'sharpe':>9}"
        f"{'max DD':>9}{'avg turno':>11}"
    )
    for name, fn in methods.items():
        r = po.run_backtest(
            x, fn, rf=rf, window=cfg["estimation_window"], step=cfg["rebalance_step"]
        )
        print(
            f"{name:<14}{r.ann_return:>9.4f}{r.ann_vol:>9.4f}{r.sharpe:>9.4f}"
            f"{r.max_drawdown:>9.4f}{r.avg_turnover:>11.4f}"
        )

    if fallbacks["n"]:
        print(
            f"note: tangency undefined/ill-conditioned at {fallbacks['n']} rebalance "
            "date(s) (rf at or above the min-var return); fell back to min-variance there."
        )

    print("\ndemo complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
