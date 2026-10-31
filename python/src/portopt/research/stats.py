"""Statistics for judging a Sharpe ratio: standard errors, PSR, DSR,
block bootstrap and a paired test against a benchmark.

Part of the Python-only research layer (not covered by API_SPEC.md).

Conventions.  Every function takes a 1-D series of per-period *excess*
returns (strategy return minus the risk-free rate for the same period)
and works in per-period units.  Annualize a Sharpe ratio or its standard
error with :func:`annualize_sharpe` (multiply by ``sqrt(periods)``), which
is exact only when returns are iid; :func:`lo_annualization_factor` gives
Lo's (2002) correction for serial correlation.

* ``SR = mean(x) / std(x, ddof=1)`` (the core library's convention).
* Skewness ``g3 = m3 / m2^1.5`` and kurtosis ``g4 = m4 / m2^2`` use the
  biased central moments ``m_k = mean((x - mean(x))^k)``; ``g4`` is raw
  kurtosis (3 for a normal distribution), as in Bailey and Lopez de Prado.

References.

* A. W. Lo (2002), "The Statistics of Sharpe Ratios", Financial Analysts
  Journal 58(4), 36-52.
* E. Mertens (2002), "Comments on variance of the IID estimator in Lo
  (2002)", working note; the iid non-normal variance also derived by
  J. D. Opdyke (2007), "Comparing Sharpe ratios: so where are the
  p-values?", Journal of Asset Management 8(5), 308-336.
* D. H. Bailey and M. Lopez de Prado (2012), "The Sharpe Ratio Efficient
  Frontier", Journal of Risk 15(2), 3-44.
* D. H. Bailey and M. Lopez de Prado (2014), "The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting and Non-Normality",
  Journal of Portfolio Management 40(5), 94-107.
* D. N. Politis and J. P. Romano (1994), "The Stationary Bootstrap",
  Journal of the American Statistical Association 89(428), 1303-1313.
* O. Ledoit and M. Wolf (2008), "Robust Performance Hypothesis Testing
  with the Sharpe Ratio", Journal of Empirical Finance 15(5), 850-859.
* W. K. Newey and K. D. West (1987), "A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix",
  Econometrica 55(3), 703-708; lag rule from Newey and West (1994),
  Review of Economic Studies 61(4), 631-653.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

__all__ = [
    "BootstrapResult",
    "PairedSharpeTest",
    "EULER_GAMMA",
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
]

#: Euler-Mascheroni constant used in the expected maximum of N Gaussians.
EULER_GAMMA = 0.5772156649015329

_N01 = NormalDist()


def _series(x: np.ndarray, min_len: int = 3, name: str = "returns") -> np.ndarray:
    v = np.asarray(x, dtype=float)
    if v.ndim != 1:
        raise ValueError(f"{name} must be a 1-D series, got ndim={v.ndim}")
    if v.shape[0] < min_len:
        raise ValueError(f"{name} must have at least {min_len} observations, got {v.shape[0]}")
    if not np.isfinite(v).all():
        raise ValueError(f"{name} contains NaN or infinite entries")
    return v


def moments(x: np.ndarray) -> tuple[float, float, float, float]:
    """Return ``(mean, std_ddof1, skew, kurtosis)`` of a series.

    Skew and (raw, non-excess) kurtosis use biased central moments; see the
    module docstring.  Raises ValueError if the series is constant.
    """
    v = _series(x)
    t = v.shape[0]
    mu = math.fsum(v.tolist()) / t
    d = v - mu
    m2 = math.fsum((d * d).tolist()) / t
    if not m2 > 0.0:
        raise ValueError("series is constant: Sharpe ratio and moments are undefined")
    m3 = math.fsum((d**3).tolist()) / t
    m4 = math.fsum((d**4).tolist()) / t
    sd = math.sqrt(m2 * t / (t - 1))
    return mu, sd, m3 / m2**1.5, m4 / (m2 * m2)


def sharpe_per_period(x: np.ndarray) -> float:
    """Per-period Sharpe ratio ``mean(x) / std(x, ddof=1)`` of excess returns."""
    mu, sd, _, _ = moments(x)
    return mu / sd


def annualize_sharpe(sr: float, periods: float = 252.0) -> float:
    """Multiply a per-period Sharpe ratio (or its standard error) by
    ``sqrt(periods)``; exact under iid returns (Lo 2002)."""
    p = float(periods)
    if not (math.isfinite(p) and p > 0.0):
        raise ValueError("periods must be positive and finite")
    return float(sr) * math.sqrt(p)


def sharpe_se_iid(x: np.ndarray) -> float:
    """Asymptotic standard error of the per-period Sharpe ratio under iid
    normal returns, Lo (2002): ``sqrt((1 + SR^2 / 2) / T)``.

    This is the iid version; it ignores serial correlation and fat tails.
    """
    v = _series(x)
    sr = sharpe_per_period(v)
    return math.sqrt((1.0 + 0.5 * sr * sr) / v.shape[0])


def sharpe_se_nonnormal(x: np.ndarray) -> float:
    """Standard error of the per-period Sharpe ratio for iid but non-normal
    returns (Mertens 2002): ``sqrt((1 + SR^2/2 - g3 SR + (g4 - 3)/4 SR^2) / T)``.

    Still assumes no serial correlation.  Reduces to :func:`sharpe_se_iid`
    when ``g3 = 0`` and ``g4 = 3``.
    """
    v = _series(x)
    mu, sd, g3, g4 = moments(v)
    sr = mu / sd
    var = 1.0 + 0.5 * sr * sr - g3 * sr + 0.25 * (g4 - 3.0) * sr * sr
    if not var > 0.0:
        raise ValueError("non-normal Sharpe variance is not positive")
    return math.sqrt(var / v.shape[0])


def lo_annualization_factor(x: np.ndarray, q: int) -> float:
    """Lo (2002) aggregation factor for serially correlated
    returns: ``eta(q) = q / sqrt(q + 2 sum_{k=1}^{q-1} (q - k) rho_k)`` with
    sample autocorrelations ``rho_k``.  The q-period Sharpe ratio is
    ``eta(q) * SR``; under iid returns ``eta(q) = sqrt(q)``.

    Uses all ``q - 1`` lags literally, so it is noisy when q is a sizeable
    fraction of the sample.  Requires ``2 <= q < len(x)``.
    """
    v = _series(x)
    if isinstance(q, bool) or int(q) != q or not 2 <= int(q) < v.shape[0]:
        raise ValueError(f"q must be an integer with 2 <= q < {v.shape[0]}, got {q!r}")
    qq = int(q)
    d = v - math.fsum(v.tolist()) / v.shape[0]
    denom = math.fsum((d * d).tolist())
    if not denom > 0.0:
        raise ValueError("series is constant: autocorrelations are undefined")
    acc = [(qq - k) * math.fsum((d[k:] * d[:-k]).tolist()) / denom for k in range(1, qq)]
    inner = qq + 2.0 * math.fsum(acc)
    if not inner > 0.0:
        raise ValueError("Lo annualization: variance of the aggregated return is not positive")
    return qq / math.sqrt(inner)


def psr_from_moments(sr: float, sr_benchmark: float, t: int, skew: float, kurt: float) -> float:
    """Probabilistic Sharpe Ratio from summary statistics (per-period units).

    Bailey and Lopez de Prado (2012)::

        PSR(SR*) = Phi( (SR - SR*) sqrt(T - 1)
                        / sqrt(1 - g3 SR + (g4 - 1) / 4 SR^2) )

    with ``g4`` the raw kurtosis.
    """
    if int(t) < 2:
        raise ValueError("T must be >= 2")
    vals = (float(sr), float(sr_benchmark), float(skew), float(kurt))
    if not all(math.isfinite(v) for v in vals):
        raise ValueError("PSR inputs must be finite")
    s, s_star, g3, g4 = vals
    denom = 1.0 - g3 * s + 0.25 * (g4 - 1.0) * s * s
    if not denom > 0.0:
        raise ValueError("PSR denominator is not positive (inconsistent skew/kurtosis)")
    z = (s - s_star) * math.sqrt(int(t) - 1) / math.sqrt(denom)
    return _N01.cdf(z)


def probabilistic_sharpe_ratio(x: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """Probability that the true per-period Sharpe ratio exceeds
    ``sr_benchmark`` (per-period units), given the sample's length, skew
    and kurtosis.  See :func:`psr_from_moments`."""
    v = _series(x)
    mu, sd, g3, g4 = moments(v)
    return psr_from_moments(mu / sd, sr_benchmark, v.shape[0], g3, g4)


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum of ``n_trials`` per-period Sharpe estimates whose
    true values are all zero, as used by Bailey and Lopez de Prado (2014)::

        SR0 = sqrt(V) ((1 - g) Phi^-1(1 - 1/N) + g Phi^-1(1 - 1/(N e)))

    with g the Euler-Mascheroni constant and V the variance of the Sharpe
    estimates across trials.  For ``N = 1`` there is no selection and
    ``SR0 = 0``.
    """
    if isinstance(n_trials, bool) or int(n_trials) != n_trials or int(n_trials) < 1:
        raise ValueError(f"n_trials must be an integer >= 1, got {n_trials!r}")
    var = float(sr_variance)
    if not (math.isfinite(var) and var >= 0.0):
        raise ValueError("sr_variance must be finite and >= 0")
    n = int(n_trials)
    if n == 1:
        return 0.0
    z1 = _N01.inv_cdf(1.0 - 1.0 / n)
    z2 = _N01.inv_cdf(1.0 - 1.0 / (n * math.e))
    return math.sqrt(var) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def deflated_sharpe_ratio(x: np.ndarray, *, n_trials: int, sr_variance: float) -> float:
    """Deflated Sharpe Ratio (Bailey and Lopez de Prado 2014):
    ``PSR(SR0)`` with ``SR0 = expected_max_sharpe(n_trials, sr_variance)``.

    ``sr_variance`` is the variance of the per-period Sharpe estimates of
    all ``n_trials`` configurations tried.  When the trials are correlated
    the paper uses an effective number of independent trials; passing the
    raw count is conservative (it raises SR0).
    """
    sr0 = expected_max_sharpe(n_trials, sr_variance)
    return probabilistic_sharpe_ratio(x, sr0)


@dataclass(frozen=True)
class BootstrapResult:
    """Percentile bootstrap interval for the per-period Sharpe ratio."""

    point: float
    lower: float
    upper: float
    std_error: float
    n_boot: int
    block_length: float
    method: str
    alpha: float


def _bootstrap_indices(
    rng: np.random.Generator, t: int, n_boot: int, block: float, method: str
) -> np.ndarray:
    idx = np.empty((n_boot, t), dtype=np.int64)
    if method == "stationary":
        p_new = 1.0 / block
        starts = rng.integers(0, t, size=(n_boot, t))
        new = rng.random(size=(n_boot, t)) < p_new
        idx[:, 0] = starts[:, 0]
        for j in range(1, t):
            idx[:, j] = np.where(new[:, j], starts[:, j], (idx[:, j - 1] + 1) % t)
        return idx
    b = int(block)
    n_blocks = -(-t // b)
    starts = rng.integers(0, t, size=(n_boot, n_blocks))
    offs = np.arange(b)
    full = (starts[:, :, None] + offs[None, None, :]) % t
    idx[:] = full.reshape(n_boot, n_blocks * b)[:, :t]
    return idx


def sharpe_bootstrap_ci(
    x: np.ndarray,
    *,
    seed: int,
    n_boot: int = 2000,
    block_length: float = 10.0,
    alpha: float = 0.05,
    method: str = "stationary",
) -> BootstrapResult:
    """Block-bootstrap percentile interval for the per-period Sharpe ratio.

    ``method="stationary"``: Politis and Romano (1994) stationary bootstrap
    with geometric block lengths of mean ``block_length`` and circular
    wrapping.  ``method="circular"``: Politis and Romano (1992) circular
    block bootstrap with fixed blocks of ``int(block_length)`` rows.

    ``seed`` is required and seeds a fresh ``numpy.random.Generator``
    (PCG64), so the result is a pure function of the inputs.  The interval
    is the ``(alpha/2, 1 - alpha/2)`` empirical quantiles (numpy's default
    linear interpolation); ``std_error`` is the ddof=1 standard deviation
    of the bootstrap replicates.
    """
    v = _series(x)
    if isinstance(seed, bool) or not isinstance(seed, int | np.integer):
        raise ValueError("seed must be an integer")
    if isinstance(n_boot, bool) or int(n_boot) != n_boot or int(n_boot) < 2:
        raise ValueError("n_boot must be an integer >= 2")
    bl = float(block_length)
    if not (math.isfinite(bl) and 1.0 <= bl <= v.shape[0]):
        raise ValueError(f"block_length must lie in [1, {v.shape[0]}], got {block_length!r}")
    a = float(alpha)
    if not (math.isfinite(a) and 0.0 < a < 1.0):
        raise ValueError("alpha must lie in (0, 1)")
    if method not in ("stationary", "circular"):
        raise ValueError(f"method must be 'stationary' or 'circular', got {method!r}")
    rng = np.random.default_rng(int(seed))
    idx = _bootstrap_indices(rng, v.shape[0], int(n_boot), bl, method)
    samples = v[idx]
    sd = samples.std(axis=1, ddof=1)
    if not np.all(sd > 0.0):
        raise ValueError("a bootstrap resample is constant; Sharpe ratio undefined")
    srs = samples.mean(axis=1) / sd
    lo, hi = np.quantile(srs, [a / 2.0, 1.0 - a / 2.0])
    return BootstrapResult(
        point=sharpe_per_period(v),
        lower=float(lo),
        upper=float(hi),
        std_error=float(srs.std(ddof=1)),
        n_boot=int(n_boot),
        block_length=bl,
        method=method,
        alpha=a,
    )


@dataclass(frozen=True)
class PairedSharpeTest:
    """Ledoit-Wolf (2008) HAC test of ``H0: SR_1 = SR_2`` (per-period units,
    Sharpe ratios with the ``ddof=0`` standard deviation as in the paper)."""

    sr1: float
    sr2: float
    diff: float
    std_error: float
    z: float
    p_value: float
    lags: int


def sharpe_difference_test(
    x1: np.ndarray, x2: np.ndarray, *, lags: int | None = None
) -> PairedSharpeTest:
    """Paired test of equal Sharpe ratios for two excess-return series
    observed on the same dates: the HAC delta-method test of Ledoit and
    Wolf (2008).

    With ``v = (mu1, mu2, g1, g2)`` the first and second uncentred moments,
    ``SR_i = mu_i / sqrt(g_i - mu_i^2)`` and ``Delta = f(v) = SR1 - SR2``.
    The delta method gives ``se(Delta) = sqrt(grad' Psi grad / T)`` where
    ``Psi`` is a HAC estimate of the long-run covariance of
    ``y_t = (x1_t - mu1, x2_t - mu2, x1_t^2 - g1, x2_t^2 - g2)`` and
    ``grad = (g1/(g1-mu1^2)^1.5, -g2/(g2-mu2^2)^1.5,
    -mu1/(2(g1-mu1^2)^1.5), mu2/(2(g2-mu2^2)^1.5))``.

    Kernel: Newey-West Bartlett weights ``1 - j/(L+1)``.  Ledoit and Wolf
    recommend a prewhitened quadratic-spectral kernel (Andrews and Monahan
    1992) or their studentized bootstrap for small samples; the Bartlett
    kernel is a standard consistent alternative, not their exact choice.
    ``lags=None`` uses ``floor(4 (T/100)^(2/9))``; ``lags=0`` is the iid
    version.  The p-value is two-sided under the normal approximation.
    """
    a = _series(x1, name="x1")
    b = _series(x2, name="x2")
    if a.shape != b.shape:
        raise ValueError(f"x1 and x2 must have the same length, got {a.shape[0]} and {b.shape[0]}")
    t = a.shape[0]
    if lags is None:
        n_lags = int(math.floor(4.0 * (t / 100.0) ** (2.0 / 9.0)))
    else:
        if isinstance(lags, bool) or int(lags) != lags or not 0 <= int(lags) < t:
            raise ValueError(f"lags must be an integer in [0, {t}), got {lags!r}")
        n_lags = int(lags)
    mu1 = math.fsum(a.tolist()) / t
    mu2 = math.fsum(b.tolist()) / t
    g1 = math.fsum((a * a).tolist()) / t
    g2 = math.fsum((b * b).tolist()) / t
    v1 = g1 - mu1 * mu1
    v2 = g2 - mu2 * mu2
    if not (v1 > 0.0 and v2 > 0.0):
        raise ValueError("a series is constant: Sharpe ratio undefined")
    sr1 = mu1 / math.sqrt(v1)
    sr2 = mu2 / math.sqrt(v2)
    diff = sr1 - sr2
    grad = np.array([g1 / v1**1.5, -g2 / v2**1.5, -mu1 / (2.0 * v1**1.5), mu2 / (2.0 * v2**1.5)])
    y = np.column_stack([a - mu1, b - mu2, a * a - g1, b * b - g2])
    psi = (y.T @ y) / t
    for j in range(1, n_lags + 1):
        gam = (y[j:].T @ y[:-j]) / t
        psi = psi + (1.0 - j / (n_lags + 1.0)) * (gam + gam.T)
    var = float(grad @ psi @ grad) / t
    if diff == 0.0:
        return PairedSharpeTest(sr1, sr2, 0.0, math.sqrt(max(var, 0.0)), 0.0, 1.0, n_lags)
    if not var > 0.0:
        raise ValueError("Sharpe difference variance is not positive; series may be collinear")
    se = math.sqrt(var)
    z = diff / se
    p = 2.0 * (1.0 - _N01.cdf(abs(z)))
    return PairedSharpeTest(sr1, sr2, diff, se, z, p, n_lags)
