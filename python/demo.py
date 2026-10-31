"""portopt end-to-end demo.

Run:  cd python && PYTHONPATH=src python3 demo.py

Prints: estimator table, unconstrained + long-only frontier sweep,
Black-Litterman prior vs posterior weights, ERC vs inverse-vol risk
contributions, cost-aware (turnover-penalised) long-only min variance,
and a 63-day re-estimation backtest across seven methods.
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
    caps = np.loadtxt(os.path.join(DATA, "market_caps.csv"), delimiter=",", skiprows=1, usecols=1)
    rf = cfg["risk_free_rate"]

    print("portopt demo: 8-asset universe, 1250 daily returns")
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
        print(f"{name:<10}{mu[i]:>9.4f}{np.sqrt(sig[i, i]):>13.4f}{np.sqrt(sig_sh[i, i]):>13.4f}")

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
    print(
        f"long-only targets met exactly: max |mu'w - target| = {np.max(np.abs(w_c @ mu - targets_c)):.1e}"
    )
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

    # ------------------------------------------------------------- cost-aware
    rule("Cost-aware rebalancing: long-only min variance from market weights")
    print("objective 1/2 w'Sw + kappa*sum|w - w0|, w0 = market weights, kappa in variance units")
    print(f"{'kappa':>10}{'turnover':>10}{'vol':>9}{'certified':>11}")
    for kappa in (0.0, 3e-5, 1e-4, 3e-4, 1e-3):
        r_tc = po.min_variance_turnover_solve(sig, w_mkt, kappa)
        vol_tc = np.sqrt(r_tc.weights @ sig @ r_tc.weights)
        ok = "yes" if r_tc.converged and r_tc.kkt_residual < 1e-12 else "NO"
        print(f"{kappa:>10.6f}{r_tc.turnover:>10.4f}{vol_tc:>9.4f}{ok:>11}")
    r_cap = po.min_variance_turnover_solve(sig, w_mkt, 0.0, max_turnover=0.25)
    print(
        f"kappa 0 with turnover cap 0.25: turnover {r_cap.turnover:.4f}, "
        f"vol {np.sqrt(r_cap.weights @ sig @ r_cap.weights):.4f}"
    )
    print(f"{'':14}" + "".join(f"{n[:8]:>9}" for n in names))
    print(wrow("w market", w_mkt))
    print(wrow("w kappa 1e-4", po.min_variance_turnover(sig, w_mkt, 1e-4)))
    print(wrow("w cap 0.25", r_cap.weights))

    # ---------------------------------------------------------------- backtest
    rule("Backtest: re-estimated every 63 days, daily constant mix (holdout = last 750d)")

    fallbacks = {"undefined": 0, "leverage": 0}

    def tangency_fn(h: np.ndarray) -> np.ndarray:
        """Pinned backtest method: tangency, falling back to unconstrained
        min-variance for a rebalance where tangency is undefined (rf at or
        above A/C) or above the gross-leverage cap, reported, not crashed
        (API_SPEC 6.3).  Any other error (e.g. a singular covariance)
        propagates."""
        s = po.annualize_cov(po.sample_cov(h))
        m = po.annualize_mean(po.sample_mean(h))
        try:
            return po.tangency_weights(m, s, rf)
        except po.TangencyUndefinedError:
            sc = po.two_fund_scalars(m, s)
            fallbacks["undefined" if sc.a - rf * sc.c <= 0.0 else "leverage"] += 1
            return po.min_variance_weights(s)

    prev_tc: dict[str, np.ndarray] = {}

    def min_var_tc_fn(h: np.ndarray) -> np.ndarray:
        """Long-only min variance with turnover penalty kappa = 3e-5 against
        the previous target (equal weight before the first rebalance)."""
        w0 = prev_tc.get("w", np.full(h.shape[1], 1.0 / h.shape[1]))
        w = po.min_variance_turnover(po.annualize_cov(po.sample_cov(h)), w0, 3e-5)
        prev_tc["w"] = w
        return w

    def bl_fn(h: np.ndarray) -> np.ndarray:
        s = po.annualize_cov(po.sample_cov(h))
        pi_h = po.implied_equilibrium_returns(s, w_mkt, views_doc["delta"])
        w, _ = po.bl_weights(s, pi_h, p, q, views_doc["tau"], confidences=conf)
        return w

    methods = {
        "equal_weight": lambda h: np.full(h.shape[1], 1.0 / h.shape[1]),
        "min_var_lo": lambda h: po.min_variance_longonly(po.annualize_cov(po.sample_cov(h))),
        "min_var_tc": min_var_tc_fn,
        "tangency": tangency_fn,
        "black_litt": bl_fn,
        "erc": lambda h: po.erc_weights(po.sample_cov(h)),
        "inverse_vol": lambda h: po.inverse_vol_weights(po.sample_cov(h)),
    }
    print(f"{'method':<14}{'ann ret':>9}{'ann vol':>9}{'sharpe':>9}{'max DD':>9}{'avg turno':>11}")
    for name, fn in methods.items():
        r = po.run_backtest(
            x, fn, rf=rf, window=cfg["estimation_window"], step=cfg["rebalance_step"]
        )
        print(
            f"{name:<14}{r.ann_return:>9.4f}{r.ann_vol:>9.4f}{r.sharpe:>9.4f}"
            f"{r.max_drawdown:>9.4f}{r.avg_turnover:>11.4f}"
        )

    n_fb = fallbacks["undefined"] + fallbacks["leverage"]
    if n_fb:
        print(
            f"note: tangency fell back to min-variance at {n_fb} rebalance date(s): "
            f"{fallbacks['undefined']} with rf at or above the min-var return A/C, "
            f"{fallbacks['leverage']} above the {po.TANGENCY_MAX_LEVERAGE:g}x gross-leverage cap."
        )
    print("min_var_tc = min_var_lo with turnover penalty kappa = 3e-5 against the previous target.")

    print("\ndemo complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
