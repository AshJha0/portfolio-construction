"""Regenerate bundled datasets and golden values for portfolio-construction.

Run from anywhere:  python3 data/generate_data.py

Deterministic: numpy default_rng(seed=42).  Writes
    asset_returns.csv   1250 days x 8 assets, factor structure + Student-t noise
    market_caps.csv     market caps for the reverse-optimization weights
    views.json          example absolute + relative Black-Litterman views
    config.json         shared configuration constants
    golden/golden.json  24 named cross-language golden cases

Golden values are computed by the portopt reference implementation from
the CSV *as written* (values are re-read from disk so that every language
consuming the file reproduces them exactly).  The script also validates:
closed-form solutions against numpy.linalg.lstsq, the Ledoit-Wolf
intensity against a direct transcription of the authors' covCor.m, the
active-set QP solutions against scipy SLSQP and their KKT certificates,
ERC risk contributions equality, and the BL limits tau -> 0 and
Omega -> inf.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, os.pardir, "python", "src"))

import portopt as po  # noqa: E402

SEED = 42
T_DAYS = 1250
ASSETS = ["US_EQ", "EU_EQ", "JP_EQ", "UK_EQ", "EM_EQ", "US_BOND", "IG_CORP", "FX_CARRY"]
N = len(ASSETS)

# Annual drifts (converted to daily below).
MU_ANNUAL = np.array([0.07, 0.06, 0.05, 0.055, 0.09, 0.03, 0.035, 0.045])
# Factor loadings on [global equity, rates, carry] factors.
LOADINGS = np.array(
    [
        [1.00, -0.10, 0.10],
        [0.95, -0.05, 0.15],
        [0.85, 0.00, 0.10],
        [0.90, -0.05, 0.20],
        [1.15, 0.05, 0.45],
        [-0.05, 1.00, -0.05],
        [0.15, 0.85, 0.05],
        [0.25, 0.05, 1.00],
    ]
)
FACTOR_VOLS = np.array([0.009, 0.0028, 0.004])  # daily factor vols
IDIO_VOLS = np.array([0.004, 0.0045, 0.005, 0.0045, 0.007, 0.0012, 0.0015, 0.003])
T_DOF = 6.0

MARKET_CAPS = [30.0, 20.0, 8.0, 6.0, 7.0, 25.0, 15.0, 10.0]  # arbitrary units

RF = 0.02
DELTA = 2.5
TAU = 0.05
WINDOW = 500
STEP = 63
FRONTIER_TARGET = 0.08
BOX_LB, BOX_UB = 0.02, 0.40
BOX_TARGET = 0.06
RB_BUDGETS = [0.20, 0.20, 0.15, 0.15, 0.10, 0.10, 0.05, 0.05]

VIEWS = [
    {"type": "absolute", "asset": "EM_EQ", "q": 0.10, "confidence": 1.0},
    {"type": "relative", "long": "US_EQ", "short": "EU_EQ", "q": 0.02, "confidence": 0.5},
]


def generate_returns() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    factors = rng.standard_normal((T_DAYS, 3)) * FACTOR_VOLS
    idio = rng.standard_t(T_DOF, size=(T_DAYS, N)) * (
        IDIO_VOLS / np.sqrt(T_DOF / (T_DOF - 2.0))
    )
    return MU_ANNUAL / 252.0 + factors @ LOADINGS.T + idio


def write_returns(path: str, x: np.ndarray) -> None:
    with open(path, "w", newline="\n") as fh:
        fh.write(",".join(ASSETS) + "\n")
        for row in x:
            fh.write(",".join(f"{v:.17g}" for v in row) + "\n")


def read_returns(path: str) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1)


def case(name: str, inputs: dict, expect: dict, tol: float) -> dict:
    return {
        "name": name,
        "inputs": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in inputs.items()},
        "expect": {k: float(v) for k, v in expect.items()},
        "tol": tol,
    }


def wdict(prefix: str, v: np.ndarray) -> dict:
    return {f"{prefix}_{i}": float(x) for i, x in enumerate(np.asarray(v).reshape(-1))}


def main() -> None:
    x = generate_returns()
    returns_path = os.path.join(_HERE, "asset_returns.csv")
    write_returns(returns_path, x)
    x = read_returns(returns_path)  # golden values defined on the file as written
    assert x.shape == (T_DAYS, N)

    with open(os.path.join(_HERE, "market_caps.csv"), "w", newline="\n") as fh:
        fh.write("asset,market_cap\n")
        for name, cap in zip(ASSETS, MARKET_CAPS):
            fh.write(f"{name},{cap:.17g}\n")

    with open(os.path.join(_HERE, "views.json"), "w", newline="\n") as fh:
        json.dump({"tau": TAU, "delta": DELTA, "views": VIEWS}, fh, indent=2)
        fh.write("\n")

    with open(os.path.join(_HERE, "config.json"), "w", newline="\n") as fh:
        json.dump(
            {
                "assets": ASSETS,
                "annualization_factor": 252,
                "risk_free_rate": RF,
                "risk_aversion_delta": DELTA,
                "tau": TAU,
                "estimation_window": WINDOW,
                "rebalance_step": STEP,
                "frontier_points": 25,
                "seed": SEED,
            },
            fh,
            indent=2,
        )
        fh.write("\n")

    # ------------------------------------------------------------------ moments
    mu_d = po.sample_mean(x)
    cov_d = po.sample_cov(x)
    mu = po.annualize_mean(mu_d)
    sig = po.annualize_cov(cov_d)
    shrink = po.shrink_covariance(x)

    # ------------------------------------------------------- markowitz closed forms
    sc = po.two_fund_scalars(mu, sig)
    w_mv = po.min_variance_weights(sig)
    w_tan = po.tangency_weights(mu, sig, RF)
    tan_leverage = po.gross_leverage(w_tan)
    w_frontier = po.frontier_weights(mu, sig, FRONTIER_TARGET)
    qp_lo = po.min_variance_longonly_solve(sig)
    qp_box = po.min_variance_longonly_solve(sig, lb=BOX_LB, ub=BOX_UB)
    qp_cf = po.constrained_frontier_solve(mu, sig, FRONTIER_TARGET)
    qp_cf_box = po.constrained_frontier_solve(mu, sig, BOX_TARGET, lb=BOX_LB, ub=BOX_UB)
    for name, r in (("minvar_longonly", qp_lo), ("minvar_box", qp_box),
                    ("frontier_constrained", qp_cf), ("frontier_constrained_box", qp_cf_box)):
        assert r.converged and r.kkt_residual < 1e-12, f"{name}: QP not certified ({r.kkt_residual:.2e})"
    w_mv_lo, w_box, w_cf, w_cf_box = qp_lo.weights, qp_box.weights, qp_cf.weights, qp_cf_box.weights
    low = po.cholesky(sig)

    # ------------------------------------------------------- validation vs lstsq
    ones = np.ones(N)
    y_ls = np.linalg.lstsq(sig, ones, rcond=None)[0]
    assert np.allclose(w_mv, y_ls / y_ls.sum(), atol=1e-11), "min-var vs lstsq"
    z_ls = np.linalg.lstsq(sig, mu - RF, rcond=None)[0]
    assert np.allclose(w_tan, z_ls / z_ls.sum(), atol=1e-11), "tangency vs lstsq"
    g_ls = np.linalg.lstsq(sig, mu, rcond=None)[0]
    a_chk, b_chk, c_chk = ones @ g_ls, mu @ g_ls, ones @ y_ls
    assert abs(sc.a - a_chk) < 1e-10 and abs(sc.b - b_chk) < 1e-10 and abs(sc.c - c_chk) < 1e-10
    print("validated: closed forms vs numpy.linalg.lstsq")

    # ------------------------------------------------------- validation vs scipy
    from scipy.optimize import minimize  # noqa: E402  (generator-only dependency)

    def slsqp(target, lb, ub):
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        if target is not None:
            cons.append({"type": "eq", "fun": lambda w: mu @ w - target})
        w0 = np.clip(np.full(N, 1.0 / N), lb, ub)
        r = minimize(lambda w: 0.5 * w @ sig @ w, w0, jac=lambda w: sig @ w, method="SLSQP",
                     bounds=[(lb, None if ub == np.inf else ub)] * N, constraints=cons,
                     options={"ftol": 1e-16, "maxiter": 2000})
        assert r.success, r.message
        return r.x

    for name, w_qp, tgt, lb, ub in (("minvar_longonly", w_mv_lo, None, 0.0, np.inf),
                                    ("minvar_box", w_box, None, BOX_LB, BOX_UB),
                                    ("frontier_constrained", w_cf, FRONTIER_TARGET, 0.0, np.inf),
                                    ("frontier_constrained_box", w_cf_box, BOX_TARGET, BOX_LB, BOX_UB)):
        w_ref = slsqp(tgt, lb, ub)
        assert w_qp @ sig @ w_qp <= w_ref @ sig @ w_ref + 1e-12, f"{name}: QP worse than SLSQP"
        assert np.max(np.abs(w_qp - w_ref)) < 1e-6, f"{name}: QP differs from SLSQP"
        if tgt is not None:
            assert abs(mu @ w_qp - tgt) < 1e-12, f"{name}: target not met"
    print("validated: active-set QP vs scipy SLSQP (4 problems, KKT residual < 1e-12)")

    # ---------------------------------------------------- validation vs covCor.m
    def covcor_intensity(z):
        t, n = z.shape
        zc = z - z.mean(axis=0)
        sample = (zc.T @ zc) / t
        var = np.diag(sample)
        sqrtvar = np.sqrt(var)
        rb = (np.sum(sample / np.outer(sqrtvar, sqrtvar)) - n) / (n * (n - 1))
        prior = rb * np.outer(sqrtvar, sqrtvar)
        np.fill_diagonal(prior, var)
        y = zc ** 2
        phi = ((y.T @ y) / t - 2 * (zc.T @ zc) * sample / t + sample ** 2).sum()
        term1 = ((zc ** 3).T @ zc) / t
        help_ = (zc.T @ zc) / t
        theta = term1 - np.diag(help_)[:, None] * sample - help_ * var[:, None] + var[:, None] * sample
        np.fill_diagonal(theta, 0.0)
        rho = np.sum(np.diag((y.T @ y) / t - 2 * (zc.T @ zc) * sample / t + sample ** 2)) \
            + rb * np.sum(np.outer(1 / sqrtvar, sqrtvar) * theta)
        gamma = np.linalg.norm(sample - prior, "fro") ** 2
        return max(0.0, min(1.0, (phi - rho) / gamma / t))

    assert abs(shrink.intensity - covcor_intensity(x)) < 1e-14, "LW intensity vs covCor.m"
    print("validated: Ledoit-Wolf intensity vs covCor.m transcription")

    # ------------------------------------------------------------- black-litterman
    caps = np.array(MARKET_CAPS)
    w_mkt = caps / caps.sum()
    pi = po.implied_equilibrium_returns(sig, w_mkt, DELTA)
    p_mat, q_vec, conf = po.views_to_matrices(VIEWS, ASSETS)
    omega = po.default_omega(sig, p_mat, TAU, conf)
    bl = po.bl_posterior(sig, pi, p_mat, q_vec, TAU, confidences=conf)
    # pi/mu_post are excess returns, so BL tangency is pinned at rf = 0.
    w_bl = po.tangency_weights(bl.mu_post, bl.sigma_post, 0.0)

    # BL limit validation. Note: with the default Omega (proportional to tau)
    # the posterior is tau-invariant, so the tau->0 limit is taken at FIXED Omega.
    bl_small_tau = po.bl_posterior(sig, pi, p_mat, q_vec, 1e-9, omega=omega)
    assert np.max(np.abs(bl_small_tau.mu_post - pi)) < 1e-8, "BL tau->0 limit"
    bl_no_conf = po.bl_posterior(
        sig, pi, p_mat, q_vec, TAU, omega=np.diag(np.full(2, 1e12))
    )
    assert np.max(np.abs(bl_no_conf.mu_post - pi)) < 1e-10, "BL Omega->inf limit"
    print("validated: BL limits (tau->0, Omega->inf) return the prior")

    # ------------------------------------------------------------------ risk parity
    w_erc, sweeps, conv = po.risk_budget_solve(sig)
    assert conv, "ERC CCD did not converge"
    rc = po.risk_contributions(w_erc, sig)
    assert np.max(rc) - np.min(rc) < 1e-9, "ERC risk contributions not equal"
    print(f"validated: ERC RCs equal (spread {np.max(rc)-np.min(rc):.2e}, {sweeps} sweeps)")
    w_iv = po.inverse_vol_weights(sig)
    w_rb = po.risk_budget_weights(sig, RB_BUDGETS)
    rc_rb = po.risk_contributions(w_rb, sig)
    b_norm = np.array(RB_BUDGETS) / np.sum(RB_BUDGETS)
    assert np.max(np.abs(rc_rb - b_norm)) < 1e-9, "risk budgets not met"

    # -------------------------------------------------------------------- backtest
    def equal_fn(h: np.ndarray) -> np.ndarray:
        return np.full(h.shape[1], 1.0 / h.shape[1])

    def erc_fn(h: np.ndarray) -> np.ndarray:
        return po.erc_weights(po.sample_cov(h))

    def mv_lo_fn(h: np.ndarray) -> np.ndarray:
        return po.min_variance_longonly(po.annualize_cov(po.sample_cov(h)))

    bt_eq = po.run_backtest(x, equal_fn, rf=RF, window=WINDOW, step=STEP)
    bt_erc = po.run_backtest(x, erc_fn, rf=RF, window=WINDOW, step=STEP)
    bt_mv = po.run_backtest(x, mv_lo_fn, rf=RF, window=WINDOW, step=STEP)

    # ---------------------------------------------------------------- golden cases
    cases = []
    cases.append(
        case(
            "sample_moments_daily",
            {},
            {
                "mean_0": mu_d[0],
                "mean_4": mu_d[4],
                "mean_7": mu_d[7],
                "cov_0_0": cov_d[0, 0],
                "cov_0_1": cov_d[0, 1],
                "cov_4_4": cov_d[4, 4],
                "cov_6_7": cov_d[6, 7],
            },
            1e-12,
        )
    )
    cases.append(
        case(
            "shrinkage_const_corr_daily",
            {},
            {
                "intensity": shrink.intensity,
                "rbar": shrink.rbar,
                "shrunk_0_0": shrink.sigma[0, 0],
                "shrunk_0_1": shrink.sigma[0, 1],
                "shrunk_2_5": shrink.sigma[2, 5],
                "shrunk_7_7": shrink.sigma[7, 7],
            },
            1e-10,
        )
    )
    cases.append(
        case(
            "two_fund_scalars_annualized",
            {},
            {"a": sc.a, "b": sc.b, "c": sc.c, "d": sc.d},
            1e-10,
        )
    )
    cases.append(
        case(
            "minvar_unconstrained",
            {},
            {
                **wdict("w", w_mv),
                "ret": float(mu @ w_mv),
                "vol": float(np.sqrt(w_mv @ sig @ w_mv)),
            },
            1e-10,
        )
    )
    cases.append(
        case(
            "tangency_unconstrained",
            {"rf": RF},
            {
                **wdict("w", w_tan),
                "sharpe": float((mu @ w_tan - RF) / np.sqrt(w_tan @ sig @ w_tan)),
                "gross_leverage": tan_leverage,
            },
            1e-10,
        )
    )
    cases.append(
        case(
            "frontier_point_t080",
            {"target": FRONTIER_TARGET},
            {**wdict("w", w_frontier), "variance": float(w_frontier @ sig @ w_frontier)},
            1e-10,
        )
    )
    cases.append(
        case(
            "minvar_longonly_qp",
            {},
            {**wdict("w", w_mv_lo), "variance": float(w_mv_lo @ sig @ w_mv_lo)},
            1e-10,
        )
    )
    cases.append(
        case(
            "minvar_box_qp",
            {"lb": BOX_LB, "ub": BOX_UB},
            {**wdict("w", w_box), "variance": float(w_box @ sig @ w_box)},
            1e-10,
        )
    )
    cases.append(
        case(
            "frontier_constrained_t080",
            {"target": FRONTIER_TARGET},
            {**wdict("w", w_cf), "variance": float(w_cf @ sig @ w_cf)},
            1e-10,
        )
    )
    cases.append(
        case(
            "frontier_constrained_box_t060",
            {"target": BOX_TARGET, "lb": BOX_LB, "ub": BOX_UB},
            {**wdict("w", w_cf_box), "variance": float(w_cf_box @ sig @ w_cf_box)},
            1e-10,
        )
    )
    cases.append(
        case(
            "cholesky_annualized_cov",
            {},
            {"l_0_0": low[0, 0], "l_7_0": low[7, 0], "l_7_7": low[7, 7]},
            1e-12,
        )
    )
    v_hand = np.array([0.9, 0.5, -0.2])
    w_proj = po.project_simplex_box(v_hand)
    cases.append(
        case(
            "simplex_projection_hand",
            {"v_0": 0.9, "v_1": 0.5, "v_2": -0.2, "total": 1.0},
            wdict("w", w_proj),
            1e-12,
        )
    )
    v_box = np.array([0.8, 0.6, 0.4, -0.3])
    w_box = po.project_simplex_box(v_box, 1.0, lb=0.05, ub=0.5)
    cases.append(
        case(
            "simplex_projection_box",
            {
                "v_0": 0.8,
                "v_1": 0.6,
                "v_2": 0.4,
                "v_3": -0.3,
                "lb": 0.05,
                "ub": 0.5,
                "total": 1.0,
            },
            wdict("w", w_box),
            1e-12,
        )
    )
    cases.append(case("implied_pi", {"delta": DELTA}, wdict("pi", pi), 1e-10))
    cases.append(
        case(
            "bl_omega_default",
            {"tau": TAU, "conf_0": 1.0, "conf_1": 0.5},
            {"omega_0_0": omega[0, 0], "omega_1_1": omega[1, 1]},
            1e-12,
        )
    )
    cases.append(
        case("bl_posterior_returns", {"tau": TAU}, wdict("mu", bl.mu_post), 1e-10)
    )
    cases.append(
        case("bl_posterior_weights", {"tau": TAU, "rf": 0.0}, wdict("w", w_bl), 1e-10)
    )
    cases.append(case("inverse_vol", {}, wdict("w", w_iv), 1e-12))
    cases.append(
        case(
            "erc_weights",
            {},
            {**wdict("w", w_erc), "rc_spread": float(np.max(rc) - np.min(rc))},
            1e-8,
        )
    )
    cases.append(
        case(
            "risk_budget_pinned",
            {f"b_{i}": b for i, b in enumerate(RB_BUDGETS)},
            wdict("w", w_rb),
            1e-8,
        )
    )
    cases.append(
        case(
            "backtest_equal_weight",
            {"rf": RF, "window": WINDOW, "step": STEP},
            {
                "ann_return": bt_eq.ann_return,
                "ann_vol": bt_eq.ann_vol,
                "sharpe": bt_eq.sharpe,
                "max_drawdown": bt_eq.max_drawdown,
            },
            1e-8,
        )
    )
    cases.append(
        case(
            "backtest_erc",
            {"rf": RF, "window": WINDOW, "step": STEP},
            {"sharpe": bt_erc.sharpe, "max_drawdown": bt_erc.max_drawdown},
            1e-8,
        )
    )
    cases.append(
        case(
            "turnover_minvar_longonly",
            {"window": WINDOW, "step": STEP},
            {
                "avg_turnover": bt_mv.avg_turnover,
                "n_rebalances": float(len(bt_mv.rebalance_dates)),
            },
            1e-9,
        )
    )

    golden_path = os.path.join(_HERE, "golden", "golden.json")
    os.makedirs(os.path.dirname(golden_path), exist_ok=True)
    with open(golden_path, "w", newline="\n") as fh:
        json.dump({"cases": cases}, fh, indent=2)
        fh.write("\n")

    print(f"wrote {returns_path} ({T_DAYS} x {N})")
    print(f"wrote market_caps.csv, views.json, config.json")
    print(f"wrote {golden_path} with {len(cases)} cases")


if __name__ == "__main__":
    main()
