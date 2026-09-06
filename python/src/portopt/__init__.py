"""portopt — portfolio construction reference implementation.

Markowitz mean-variance (closed forms + constrained active-set QP),
Ledoit-Wolf constant-correlation shrinkage, Black-Litterman, risk parity /
risk budgeting via CCD, and a quarterly expanding-window backtest.  Every
numerical algorithm is pinned in the project's API_SPEC.md so that
scipy-free ports in other languages reproduce the golden values.
"""
from .linalg import cholesky, chol_solve, spd_solve, spd_inverse
from .estimators import (
    ShrinkageResult,
    annualize_cov,
    annualize_mean,
    sample_cov,
    sample_mean,
    shrink_covariance,
    validate_returns,
)
from .markowitz import (
    QP_MAX_ITER,
    TANGENCY_COND_RTOL,
    QpResult,
    TangencyUndefinedError,
    TwoFundScalars,
    attainable_return_range,
    constrained_frontier,
    constrained_frontier_solve,
    constrained_frontier_weights,
    efficient_frontier,
    frontier_weights,
    gross_leverage,
    min_variance_longonly,
    min_variance_longonly_solve,
    min_variance_weights,
    project_simplex_box,
    tangency_weights,
    two_fund_scalars,
)
from .black_litterman import (
    BLResult,
    bl_posterior,
    bl_weights,
    default_omega,
    implied_equilibrium_returns,
    views_to_matrices,
)
from .risk_parity import (
    erc_weights,
    inverse_vol_weights,
    risk_budget_solve,
    risk_budget_weights,
    risk_contributions,
)
from .backtest import (
    BacktestResult,
    annualized_return,
    annualized_vol,
    max_drawdown,
    rebalance_indices,
    run_backtest,
    sharpe_ratio,
)

__version__ = "2.0.0"

__all__ = [
    "cholesky",
    "chol_solve",
    "spd_solve",
    "spd_inverse",
    "ShrinkageResult",
    "annualize_cov",
    "annualize_mean",
    "sample_cov",
    "sample_mean",
    "shrink_covariance",
    "validate_returns",
    "QP_MAX_ITER",
    "TANGENCY_COND_RTOL",
    "QpResult",
    "TangencyUndefinedError",
    "TwoFundScalars",
    "attainable_return_range",
    "constrained_frontier",
    "constrained_frontier_solve",
    "constrained_frontier_weights",
    "efficient_frontier",
    "frontier_weights",
    "gross_leverage",
    "min_variance_longonly",
    "min_variance_longonly_solve",
    "min_variance_weights",
    "project_simplex_box",
    "tangency_weights",
    "two_fund_scalars",
    "BLResult",
    "bl_posterior",
    "bl_weights",
    "default_omega",
    "implied_equilibrium_returns",
    "views_to_matrices",
    "erc_weights",
    "inverse_vol_weights",
    "risk_budget_solve",
    "risk_budget_weights",
    "risk_contributions",
    "BacktestResult",
    "annualized_return",
    "annualized_vol",
    "max_drawdown",
    "rebalance_indices",
    "run_backtest",
    "sharpe_ratio",
    "__version__",
]
