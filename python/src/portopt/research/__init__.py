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

Nothing in this layer can show that a strategy has an edge on the
bundled data, because that data is synthetic (``data/generate_data.py``).
What it can do is measure, net of costs and with honest error bars,
whatever data it is given.
"""

from __future__ import annotations

from .costs import NetBacktestResult, run_backtest_net
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
]
