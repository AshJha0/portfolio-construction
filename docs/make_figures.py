"""Draw the three README figures from the bundled synthetic data.

Run from the repository root:

    python -m pip install -e "python[docs]"   # numpy + matplotlib
    python docs/make_figures.py

Writes docs/img/frontier.png, docs/img/bl_weights.png and
docs/img/risk_contributions.png. Every number comes from the public
portopt API applied to data/asset_returns.csv, which data/generate_data.py
simulates (three-factor model plus Student-t noise, seed 42). The figures
therefore illustrate the algorithms; they say nothing about real markets.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "python", "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import portopt as po  # noqa: E402

DATA = os.path.join(ROOT, "data")
OUT = os.path.join(HERE, "img")

# Reference palette: categorical slots 1 and 2, neutral inks, light surface.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#ffffff"
CAPTION = (
    "Synthetic data from data/generate_data.py (3-factor model + Student-t noise, seed 42, "
    "8 assets, 1250 days). Not market data."
)


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": INK2,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "text.color": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "legend.frameon": False,
        }
    )


def caption(fig: plt.Figure) -> None:
    fig.text(0.01, 0.01, CAPTION, fontsize=8, color=INK2, ha="left", va="bottom")


def load():
    path = os.path.join(DATA, "asset_returns.csv")
    x = np.loadtxt(path, delimiter=",", skiprows=1)
    with open(path) as fh:
        names = fh.readline().strip().split(",")
    with open(os.path.join(DATA, "config.json")) as fh:
        cfg = json.load(fh)
    with open(os.path.join(DATA, "views.json")) as fh:
        views = json.load(fh)
    caps = np.loadtxt(os.path.join(DATA, "market_caps.csv"), delimiter=",", skiprows=1, usecols=1)
    return x, names, cfg, views, caps


def frontier(x, names, cfg) -> str:
    mu = po.annualize_mean(po.sample_mean(x))
    sig = po.annualize_cov(po.sample_cov(x))
    rf = cfg["risk_free_rate"]
    n_pts = cfg["frontier_points"]

    t_u, _, v_u = po.efficient_frontier(mu, sig, n_pts)
    t_c, _, v_c = po.constrained_frontier(mu, sig, n_pts)
    w_gmv = po.min_variance_weights(sig)
    w_tan = po.tangency_weights(mu, sig, rf)

    def pt(w):
        return 100 * np.sqrt(w @ sig @ w), 100 * (mu @ w)

    gmv_v, gmv_r = pt(w_gmv)
    tan_v, tan_r = pt(w_tan)

    # The pinned grid stops at max(mu); extend the unconstrained curve with
    # frontier_weights so that it passes through the tangency portfolio.
    ext = np.linspace(t_u[-1], 1.15 * float(mu @ w_tan), 30)
    v_ext = np.array([w @ sig @ w for w in (po.frontier_weights(mu, sig, m) for m in ext)])

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(
        100 * np.sqrt(v_u),
        100 * t_u,
        color=BLUE,
        lw=2,
        label="Unconstrained frontier (pinned 25-point grid)",
    )
    ax.plot(
        100 * np.sqrt(v_ext),
        100 * ext,
        color=BLUE,
        lw=1.5,
        ls=":",
        label="Unconstrained, extended beyond max(mu)",
    )
    ax.plot(100 * np.sqrt(v_c), 100 * t_c, color=ORANGE, lw=2, label="Long-only frontier (QP)")
    vols = 100 * np.sqrt(np.diag(sig))
    ax.scatter(vols, 100 * mu, s=28, color=INK2, zorder=3, label="Single assets")
    nudge = {"US_BOND": (6, -10), "UK_EQ": (-8, -12), "EU_EQ": (4, 5)}
    for v, r, nm in zip(vols, 100 * mu, names, strict=True):
        ax.annotate(
            nm,
            (v, r),
            xytext=nudge.get(nm, (4, 3)),
            textcoords="offset points",
            fontsize=8,
            color=INK2,
            ha="right" if nm == "UK_EQ" else "left",
        )
    # Capital market line from rf through tangency, drawn up to the plotted range.
    xmax = max(vols.max(), 100 * np.sqrt(v_c).max(), tan_v) * 1.08
    slope = (tan_r - 100 * rf) / tan_v
    ax.plot(
        [0, xmax],
        [100 * rf, 100 * rf + slope * xmax],
        color=INK2,
        lw=1,
        ls="--",
        label=f"Line from rf = {rf:.0%} through tangency",
    )
    ax.scatter([gmv_v], [gmv_r], s=70, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=4)
    ax.annotate(
        f"GMV ({gmv_v:.1f}%, {gmv_r:.1f}%)",
        (gmv_v, gmv_r),
        xytext=(-6, 10),
        textcoords="offset points",
        fontsize=9,
        ha="right",
    )
    ax.scatter([tan_v], [tan_r], s=70, color=INK, zorder=4)
    ax.annotate(
        f"Tangency ({tan_v:.1f}%, {tan_r:.1f}%)\ngross leverage {po.gross_leverage(w_tan):.2f}",
        (tan_v, tan_r),
        xytext=(10, -26),
        textcoords="offset points",
        fontsize=9,
    )
    ax.set_xlim(0, xmax)
    ax.set_xlabel("Annualised volatility (%)")
    ax.set_ylabel("Annualised mean return (%)")
    ax.set_title("Efficient frontiers on the sample moments")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    caption(fig)
    path = os.path.join(OUT, "frontier.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def bars(ax, names, a, b, la, lb):
    idx = np.arange(len(names))
    wd = 0.38
    ax.bar(idx - wd / 2 - 0.01, a, wd, color=BLUE, label=la)
    ax.bar(idx + wd / 2 + 0.01, b, wd, color=ORANGE, label=lb)
    ax.set_xticks(idx)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.grid(axis="x", visible=False)


def bl_weights(x, names, views, caps) -> str:
    sig = po.annualize_cov(po.sample_cov(x))
    w_mkt = caps / caps.sum()
    pi = po.implied_equilibrium_returns(sig, w_mkt, views["delta"])
    p, q, conf = po.views_to_matrices(views["views"], names)
    w_bl, _ = po.bl_weights(sig, pi, p, q, views["tau"], confidences=conf)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars(ax, names, 100 * w_mkt, 100 * w_bl, "Market weights (prior)", "Black-Litterman posterior")
    ax.set_ylabel("Portfolio weight (% of NAV)")
    ax.set_title("Black-Litterman: views EM_EQ 10%, US_EQ beats EU_EQ by 2%")
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    caption(fig)
    path = os.path.join(OUT, "bl_weights.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def risk_contributions(x, names) -> str:
    sig = po.annualize_cov(po.sample_cov(x))
    w_erc = po.erc_weights(sig)
    w_iv = po.inverse_vol_weights(sig)
    rc_erc = po.risk_contributions(w_erc, sig)
    rc_iv = po.risk_contributions(w_iv, sig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars(
        ax, names, 100 * rc_erc, 100 * rc_iv, "Equal risk contribution (ERC)", "Inverse volatility"
    )
    ax.axhline(100 / len(names), color=INK2, lw=1, ls="--")
    ax.annotate(
        f"1/n = {100 / len(names):.1f}%",
        (len(names) - 0.5, 100 / len(names)),
        xytext=(0, 4),
        textcoords="offset points",
        ha="right",
        fontsize=8.5,
        color=INK2,
    )
    ax.set_ylabel("Share of portfolio variance (%)")
    ax.set_title("Risk contributions: ERC equalises them, inverse-vol does not")
    ax.legend(loc="upper center", fontsize=8.5, ncol=2)
    ax.set_ylim(0, max(rc_iv.max(), rc_erc.max()) * 100 * 1.25)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    caption(fig)
    path = os.path.join(OUT, "risk_contributions.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    style()
    x, names, cfg, views, caps = load()
    paths = (
        frontier(x, names, cfg),
        bl_weights(x, names, views, caps),
        risk_contributions(x, names),
    )
    for p in paths:
        print("wrote", os.path.relpath(p, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
