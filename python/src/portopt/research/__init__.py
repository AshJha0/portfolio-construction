"""portopt.research: cost-aware backtesting and performance statistics.

This subpackage is a Python-only research layer.  It sits outside the
pinned cross-language contract of API_SPEC.md: nothing here has a C++,
Rust or Java port, there are no golden values for it, and its numerics
may change between releases without a spec change.  It uses only the
public functions of :mod:`portopt` and adds three things the core
backtest deliberately leaves out:

* :mod:`portopt.research.costs`: a backtest that lets weights drift
  between rebalances, charges transaction costs on the traded notional,
  and supports a no-trade band, partial rebalancing, ex-ante volatility
  targeting and a gross-leverage cap.
* :mod:`portopt.research.stats`: the standard error of the Sharpe ratio
  (Lo 2002, Mertens 2002), the Probabilistic and Deflated Sharpe ratios
  (Bailey and Lopez de Prado 2012, 2014), a seeded block bootstrap and
  the Ledoit-Wolf (2008) paired test of a Sharpe difference.
* :mod:`portopt.research.strategies`: weight functions for the
  construction rules compared in ``python/studies/cost_aware_allocation.py``.
* :mod:`portopt.research.estimators`: EWMA covariance, Bayes-Stein and
  James-Stein mean shrinkage, Ledoit-Wolf shrinkage toward the identity,
  the Ledoit-Wolf (2020) analytical nonlinear shrinkage, and a PCA
  statistical factor model.
* :mod:`portopt.research.strategies_v2`: covariance-estimator objects and
  weight rules built on them, including a volatility-targeting wrapper.
* :mod:`portopt.research.data`: loaders for real daily returns from the
  Kenneth French Data Library, fetched at run time and checked against
  the SHA-256 vintages in ``data/real/manifest.json``; the holdout period
  is refused unless a registered study asks for it.
* :mod:`portopt.research.walkforward`: a trial registry
  (``data/real/registry.json``) and walk-forward evaluation on top of
  :func:`run_backtest_net`.

Nothing in this layer can show that a strategy has an edge on the
bundled data, because that data is synthetic (``data/generate_data.py``).
On the real data it measures, net of costs and with error bars, what a
construction rule did; it does not by itself show that the rule will
keep doing it.
"""

from __future__ import annotations

from . import data, walkforward
from .costs import NetBacktestResult, run_backtest_net
from .estimators import (
    FactorModelResult,
    LinearShrinkageResult,
    MeanShrinkageResult,
    NonlinearShrinkageResult,
    bayes_stein_mean,
    ewma_constant_correlation,
    ewma_covariance,
    ewma_decay,
    ewma_weights,
    james_stein_mean,
    ledoit_wolf_identity,
    nonlinear_shrinkage,
    pca_factor_covariance,
)
from .stats import (
    BootstrapResult,
    PairedSharpeTest,
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
from .strategies import (
    WeightFunction,
    black_litterman,
    equal_weight,
    erc,
    inverse_vol,
    max_sharpe_longonly,
    min_variance_longonly,
    tangency_capped,
)
from .strategies_v2 import (
    CovarianceEstimator,
    VolTargetedWeights,
    erc_with,
    ewma_cc_estimator,
    ewma_estimator,
    lw_constant_corr_estimator,
    lw_identity_estimator,
    max_sharpe_bayes_stein,
    min_variance_longonly_with,
    nonlinear_estimator,
    pca_estimator,
    sample_estimator,
    vol_target,
)
from .walkforward import (
    WalkForwardResult,
    n_trials,
    register_study,
    run_schedule_net,
    walk_forward,
)

__all__ = [
    "NetBacktestResult",
    "run_backtest_net",
    "BootstrapResult",
    "PairedSharpeTest",
    "annualize_sharpe",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "lo_annualization_factor",
    "moments",
    "probabilistic_sharpe_ratio",
    "psr_from_moments",
    "sharpe_bootstrap_ci",
    "sharpe_difference_test",
    "sharpe_per_period",
    "sharpe_se_iid",
    "sharpe_se_nonnormal",
    "WeightFunction",
    "black_litterman",
    "equal_weight",
    "erc",
    "inverse_vol",
    "max_sharpe_longonly",
    "min_variance_longonly",
    "tangency_capped",
    "FactorModelResult",
    "LinearShrinkageResult",
    "MeanShrinkageResult",
    "NonlinearShrinkageResult",
    "bayes_stein_mean",
    "ewma_constant_correlation",
    "ewma_covariance",
    "ewma_decay",
    "ewma_weights",
    "james_stein_mean",
    "ledoit_wolf_identity",
    "nonlinear_shrinkage",
    "pca_factor_covariance",
    "CovarianceEstimator",
    "VolTargetedWeights",
    "erc_with",
    "ewma_cc_estimator",
    "ewma_estimator",
    "lw_constant_corr_estimator",
    "lw_identity_estimator",
    "max_sharpe_bayes_stein",
    "min_variance_longonly_with",
    "nonlinear_estimator",
    "pca_estimator",
    "sample_estimator",
    "vol_target",
    "data",
    "walkforward",
    "WalkForwardResult",
    "n_trials",
    "register_study",
    "run_schedule_net",
    "walk_forward",
]
