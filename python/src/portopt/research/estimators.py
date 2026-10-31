"""Covariance and mean estimators for the research layer.

Part of the Python-only research layer (not covered by API_SPEC.md):
nothing here has a port in another language, there are no golden values,
and the numerics may change between releases.  Every function takes a
T x N matrix of simple per-period returns, validated with
:func:`portopt.validate_returns`, and returns per-period moments (no
annualization; multiply by 252 with :func:`portopt.annualize_cov`).

Conventions shared by the module.

* "Demeaned" means each column minus its equal-weighted sample mean.
* The sample covariance is ``S = Xc' Xc / (T - 1)`` (ddof=1, the
  project-wide convention).  It is computed with numpy matrix products,
  so it agrees with :func:`portopt.sample_cov` to rounding, not bit for
  bit.
* Where a paper states its estimator with divisor T, the scale-free
  quantity (a shrinkage intensity) is computed exactly as in the paper
  and then applied to the ddof=1 matrices, as the core
  :func:`portopt.shrink_covariance` does.  The returned matrix is then the
  paper's estimator times ``T / (T - 1)``.

Estimators.

* :func:`ewma_covariance`: exponentially weighted covariance in the
  RiskMetrics style, parameterised by a half-life;
  :func:`ewma_constant_correlation`: EWMA volatilities with one constant
  correlation.
* :func:`bayes_stein_mean` (Jorion 1986) and :func:`james_stein_mean`
  (Efron and Morris 1973): shrinkage of the mean vector toward a grand
  mean.
* :func:`ledoit_wolf_identity`: linear shrinkage toward a multiple of the
  identity (Ledoit and Wolf 2004, Journal of Multivariate Analysis).
* :func:`nonlinear_shrinkage`: the analytical nonlinear shrinkage of
  Ledoit and Wolf (2020, Annals of Statistics).
* :func:`pca_factor_covariance`: a statistical (principal-component)
  factor model with a floored diagonal of specific variances.

References.

* J.P. Morgan / Reuters (1996), "RiskMetrics: Technical Document", 4th
  edition, chapter 5 (exponentially weighted moving averages).
* P. Jorion (1986), "Bayes-Stein Estimation for Portfolio Analysis",
  Journal of Financial and Quantitative Analysis 21(3), 279-292.
* B. Efron and C. Morris (1973), "Stein's Estimation Rule and Its
  Competitors: An Empirical Bayes Approach", Journal of the American
  Statistical Association 68(341), 117-130.
* O. Ledoit and M. Wolf (2004), "A well-conditioned estimator for
  large-dimensional covariance matrices", Journal of Multivariate
  Analysis 88(2), 365-411.
* O. Ledoit and M. Wolf (2020), "Analytical nonlinear shrinkage of
  large-dimensional covariance matrices", Annals of Statistics 48(5),
  3043-3065.
* E. J. Elton and M. J. Gruber (1973), "Estimating the Dependence
  Structure of Share Prices: Implications for Portfolio Selection",
  Journal of Finance 28(5), 1203-1232 (constant correlation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from portopt import spd_solve, validate_returns

__all__ = [
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
]

_SQRT5 = math.sqrt(5.0)
# Relative threshold below which a retained sample eigenvalue counts as zero.
_EIG_RTOL = 1e-12
# Relative threshold below which a sample variance counts as zero (a column
# that is constant up to rounding has variance ~1e-30 of its neighbours').
_VAR_RTOL = 1e-20


def _returns(returns: np.ndarray, min_rows: int = 2) -> np.ndarray:
    return validate_returns(returns, min_rows=min_rows)


def _demean(x: np.ndarray) -> np.ndarray:
    out: np.ndarray = x - x.mean(axis=0)
    return out


def _sym(a: np.ndarray) -> np.ndarray:
    out: np.ndarray = 0.5 * (a + a.T)
    return out


def _sample_cov(x: np.ndarray) -> np.ndarray:
    xc = _demean(x)
    return _sym(xc.T @ xc / (x.shape[0] - 1))


# --------------------------------------------------------------------------
# EWMA
# --------------------------------------------------------------------------


def ewma_decay(half_life: float) -> float:
    """Decay factor ``lambda = 0.5 ** (1 / half_life)``.

    An observation ``half_life`` periods old carries half the weight of
    the newest one.  ``half_life = inf`` gives ``lambda = 1`` (equal
    weights).  RiskMetrics' daily ``lambda = 0.94`` is a half-life of
    about 11.2 days.

    Raises ValueError unless ``half_life`` is a positive number (``inf``
    allowed).
    """
    h = float(half_life)
    if math.isnan(h) or not h > 0.0:
        raise ValueError(f"half_life must be > 0 (inf allowed), got {half_life!r}")
    if math.isinf(h):
        return 1.0
    return float(0.5 ** (1.0 / h))


def ewma_weights(t: int, half_life: float) -> np.ndarray:
    """Normalised EWMA weights for rows ``0 .. t-1`` (newest last).

    ``w_s = lambda ** (t - 1 - s) / sum_u lambda ** (t - 1 - u)``, so the
    weights sum to one exactly up to rounding.
    """
    if isinstance(t, bool) or int(t) != t or int(t) < 1:
        raise ValueError(f"t must be an integer >= 1, got {t!r}")
    lam = ewma_decay(half_life)
    age = np.arange(int(t) - 1, -1, -1, dtype=float)
    raw = lam**age
    out: np.ndarray = raw / math.fsum(raw.tolist())
    return out


def ewma_covariance(returns: np.ndarray, half_life: float, *, demean: bool = True) -> np.ndarray:
    """Exponentially weighted covariance (RiskMetrics 1996 style).

    RiskMetrics (Technical Document, 1996, chapter 5) forecasts the daily
    covariance with the recursion
    ``Sigma_t = lambda Sigma_{t-1} + (1 - lambda) r_{t-1} r_{t-1}'``,
    taking the mean daily return to be zero.  Unrolled over a finite
    sample that recursion weights the observation of age ``a`` by
    ``(1 - lambda) lambda ** a`` and leaves a remainder ``lambda ** T``
    on whatever seed ``Sigma_0`` was used.

    Initialisation choice.  We use no seed.  The weights
    ``w_s`` proportional to ``lambda ** (T - 1 - s)`` are renormalised to sum to one
    (:func:`ewma_weights`), which is the recursion with the seed's share
    redistributed over the observed rows.  The result does not depend on
    an arbitrary starting matrix, and for ``T >> half_life`` it equals the
    RiskMetrics recursion to within ``lambda ** T``.

    Mean and bias choice.

    * ``demean=False``: the RiskMetrics zero-mean form,
      ``Sigma = sum_s w_s r_s r_s'``.
    * ``demean=True`` (default): the weighted mean
      ``m = sum_s w_s r_s`` is removed and the sum is divided by the
      reliability-weights correction ``1 - sum_s w_s**2``:
      ``Sigma = sum_s w_s (r_s - m)(r_s - m)' / (1 - sum_s w_s**2)``.
      With iid observations this is unbiased for the covariance, and as
      ``half_life -> inf`` (equal weights ``1/T``) the divisor becomes
      ``(T - 1) / T``, so the estimate tends to the ddof=1 sample
      covariance exactly.

    The decay is ``lambda = 0.5 ** (1 / half_life)`` (:func:`ewma_decay`).
    Requires T >= 2.  The estimate is positive semi-definite; it is
    singular when the effective number of observations is below N.
    """
    x = _returns(returns, min_rows=2)
    w = ewma_weights(x.shape[0], half_life)
    if demean:
        m = w @ x
        xc = x - m
        denom = 1.0 - math.fsum((w * w).tolist())
        if not denom > 0.0:
            raise ValueError("ewma_covariance: weights are degenerate (1 - sum w^2 <= 0)")
        cov = (xc * w[:, None]).T @ xc / denom
    else:
        cov = (x * w[:, None]).T @ x
    return _sym(cov)


def ewma_constant_correlation(
    returns: np.ndarray, half_life: float, *, demean: bool = True
) -> np.ndarray:
    """EWMA volatilities combined with one constant correlation.

    ``Sigma_ij = rho sigma_i sigma_j`` for ``i != j`` and
    ``Sigma_ii = sigma_i ** 2``, where ``sigma_i ** 2`` is the diagonal of
    :func:`ewma_covariance` (same ``half_life`` and ``demean``) and
    ``rho`` is the average off-diagonal entry of the equal-weighted
    sample correlation matrix (the constant-correlation model of Elton and
    Gruber 1973).

    Assumption: volatilities move fast and are tracked by the EWMA, while
    correlations are stable and are better estimated from the whole
    window and pooled into one number.  Because ``rho`` is the average of
    a valid correlation matrix, ``rho >= -1 / (N - 1)`` and the result is
    positive semi-definite; it is positive definite when
    ``-1 / (N - 1) < rho < 1`` and every EWMA variance is positive.

    For N = 1 this is the EWMA variance.  Raises ValueError if any sample
    variance is zero to working precision (below ``1e-20`` times the
    largest), since the correlation is then undefined.
    """
    x = _returns(returns, min_rows=2)
    ew = ewma_covariance(x, half_life, demean=demean)
    n = x.shape[1]
    if n == 1:
        return ew
    s = _sample_cov(x)
    var = np.diag(s)
    if not np.all(var > _VAR_RTOL * float(np.max(var))):
        raise ValueError(
            "ewma_constant_correlation requires strictly positive sample variances "
            "(a column is constant to working precision)"
        )
    sd = np.sqrt(var)
    corr = s / np.outer(sd, sd)
    iu = np.triu_indices(n, k=1)
    rho = math.fsum(corr[iu].tolist()) / (n * (n - 1) / 2)
    vol = np.sqrt(np.diag(ew))
    out = rho * np.outer(vol, vol)
    np.fill_diagonal(out, np.diag(ew))
    return _sym(out)


# --------------------------------------------------------------------------
# Mean shrinkage
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MeanShrinkageResult:
    """Shrunk mean ``(1 - weight) * sample + weight * grand_mean * 1``.

    Attributes
    ----------
    mean:
        Shrunk per-period mean vector.
    weight:
        Shrinkage weight in [0, 1] placed on the grand mean.
    grand_mean:
        The scalar the means are shrunk toward.
    sample:
        Sample mean vector (divisor T).
    """

    mean: np.ndarray
    weight: float
    grand_mean: float
    sample: np.ndarray


def bayes_stein_mean(returns: np.ndarray) -> MeanShrinkageResult:
    """Bayes-Stein mean shrinkage toward the minimum-variance grand mean.

    Jorion (1986).  With sample mean ``mu`` (divisor T), ddof=1 sample
    covariance ``S`` and

    * ``Sigma = S (T - 1) / (T - N - 2)`` (Jorion's scaling, which makes
      ``Sigma^{-1}`` unbiased for the inverse covariance under normality),
    * grand mean ``mu0 = 1' Sigma^{-1} mu / 1' Sigma^{-1} 1``, the mean
      return of the sample minimum-variance portfolio,
    * ``q = (mu - mu0 1)' Sigma^{-1} (mu - mu0 1)``,

    the weight is ``phi = (N + 2) / ((N + 2) + T q)`` and the estimate is
    ``(1 - phi) mu + phi mu0 1``.  ``phi`` lies in (0, 1]; it equals 1
    when all sample means are already equal to ``mu0``.

    Assumptions: iid multivariate normal returns and Jorion's informative
    prior centred on a common mean.  The weight is invariant to rescaling
    all returns and to permuting assets.  Requires N >= 2, T > N + 2 and a
    positive definite sample covariance (ValueError otherwise).
    """
    x = _returns(returns, min_rows=2)
    t, n = x.shape
    if n < 2:
        raise ValueError("bayes_stein_mean requires at least 2 assets")
    if t <= n + 2:
        raise ValueError(f"bayes_stein_mean requires T > N + 2, got T={t}, N={n}")
    mu = x.mean(axis=0)
    sig = _sample_cov(x) * ((t - 1) / (t - n - 2))
    ones = np.ones(n)
    try:
        sinv_1 = spd_solve(sig, ones)
        sinv_mu = spd_solve(sig, mu)
    except ValueError as exc:
        raise ValueError(
            f"bayes_stein_mean: sample covariance is not positive definite ({exc})"
        ) from exc
    mu0 = float(ones @ sinv_mu) / float(ones @ sinv_1)
    d = mu - mu0
    q = float(d @ spd_solve(sig, d))
    q = max(q, 0.0)
    phi = (n + 2.0) / ((n + 2.0) + t * q)
    est = (1.0 - phi) * mu + phi * mu0
    return MeanShrinkageResult(mean=est, weight=phi, grand_mean=mu0, sample=mu)


def james_stein_mean(returns: np.ndarray) -> MeanShrinkageResult:
    """Positive-part James-Stein shrinkage toward the cross-sectional mean.

    The Efron and Morris (1973) form of Stein's rule that shrinks toward
    the estimated common mean (Lindley's modification), which costs one
    degree of freedom: with sample means ``mu_i``, their average
    ``mu_bar``, the average sample variance ``s2 = tr(S) / N`` (ddof=1)
    and ``D = sum_i (mu_i - mu_bar) ** 2``,

    ``weight = min(1, (N - 3) s2 / (T D))``,
    ``mean = (1 - weight) mu + weight mu_bar 1``.

    Assumptions: the sample means are independent with a common sampling
    variance ``s2 / T``.  Correlated or heteroscedastic assets violate
    this; :func:`bayes_stein_mean` uses the full covariance instead.  The
    weight is invariant to rescaling returns and to permuting assets.
    Requires N >= 4 (Stein's rule toward an estimated mean dominates the
    sample mean only for N >= 4) and T >= 2.  ``weight = 1`` when
    ``D = 0``.
    """
    x = _returns(returns, min_rows=2)
    t, n = x.shape
    if n < 4:
        raise ValueError("james_stein_mean requires at least 4 assets")
    mu = x.mean(axis=0)
    mbar = math.fsum(mu.tolist()) / n
    s2 = float(np.trace(_sample_cov(x))) / n
    dev = mu - mbar
    dss = math.fsum((dev * dev).tolist())
    weight = 1.0 if dss <= 0.0 else min(1.0, (n - 3.0) * s2 / (t * dss))
    weight = max(weight, 0.0)
    est = (1.0 - weight) * mu + weight * mbar
    return MeanShrinkageResult(mean=est, weight=weight, grand_mean=mbar, sample=mu)


# --------------------------------------------------------------------------
# Ledoit-Wolf linear shrinkage toward the identity
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LinearShrinkageResult:
    """Output of :func:`ledoit_wolf_identity`.

    Attributes
    ----------
    sigma:
        ``intensity * scale * I + (1 - intensity) * sample``.
    intensity:
        Shrinkage intensity ``b^2 / d^2`` in [0, 1].
    scale:
        ``tr(sample) / N``, the multiple of the identity in the target.
    sample:
        ddof=1 sample covariance.
    """

    sigma: np.ndarray
    intensity: float
    scale: float
    sample: np.ndarray


def ledoit_wolf_identity(returns: np.ndarray) -> LinearShrinkageResult:
    """Ledoit and Wolf (2004, JMVA) linear shrinkage toward ``m I``.

    Using the paper's notation with the scaled Frobenius norm
    ``||A||^2 = tr(A A') / N``, demeaned rows ``x_t`` and the paper's
    sample covariance ``S_T = Xc' Xc / T`` (divisor T):

    * ``m = tr(S_T) / N``,
    * ``d^2 = ||S_T - m I||^2``,
    * ``bbar^2 = (1 / T^2) sum_t ||x_t x_t' - S_T||^2``,
    * ``b^2 = min(bbar^2, d^2)``,

    and the estimator is ``(b^2 / d^2) m I + (1 - b^2 / d^2) S_T``
    (section 3 of the paper).  The paper assumes zero-mean data; we
    demean first, as is standard in applications.

    The intensity ``b^2 / d^2`` is scale-free and is computed exactly as
    above.  It is then applied to the ddof=1 sample ``S`` and target
    ``tr(S) / N I``, so ``sigma`` is the paper's estimator times
    ``T / (T - 1)`` (the convention of :func:`portopt.shrink_covariance`).
    When ``d^2 = 0`` (the sample is already a multiple of the identity)
    the intensity is 1.  The result is positive definite whenever the
    intensity is positive and ``tr(S) > 0``.

    Requires T >= 2 and at least one non-zero sample variance.
    """
    x = _returns(returns, min_rows=2)
    t, n = x.shape
    xc = _demean(x)
    s_t = _sym(xc.T @ xc / t)
    m = float(np.trace(s_t)) / n
    if not m > 0.0:
        raise ValueError("ledoit_wolf_identity requires a positive average sample variance")
    dev = s_t - m * np.eye(n)
    d2 = float(np.sum(dev * dev)) / n
    # sum_t ||x_t x_t' - S_T||_F^2 = sum_t (x_t' x_t)^2 - T ||S_T||_F^2
    row_sq = np.sum(xc * xc, axis=1)
    total = math.fsum((row_sq * row_sq).tolist()) - t * float(np.sum(s_t * s_t))
    bbar2 = max(total, 0.0) / (n * t * t)
    if d2 <= 0.0:
        delta = 1.0
    else:
        delta = min(bbar2, d2) / d2
    s = s_t * (t / (t - 1))
    scale = float(np.trace(s)) / n
    sigma = delta * scale * np.eye(n) + (1.0 - delta) * s
    return LinearShrinkageResult(sigma=_sym(sigma), intensity=delta, scale=scale, sample=s)


# --------------------------------------------------------------------------
# Ledoit-Wolf (2020) analytical nonlinear shrinkage
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class NonlinearShrinkageResult:
    """Output of :func:`nonlinear_shrinkage`.

    Attributes
    ----------
    sigma:
        ``U diag(shrunk) U'``.
    shrunk:
        Shrunk eigenvalues, aligned with ``sample_eigenvalues``.
    sample_eigenvalues:
        Eigenvalues of the ddof=1 sample covariance, ascending.
    eigenvectors:
        Sample eigenvectors (columns), aligned with the eigenvalues.
    bandwidth:
        The global bandwidth ``h = n ** (-1/3)``.
    concentration:
        ``c = N / n`` with effective sample size ``n = T - 1``.
    """

    sigma: np.ndarray
    shrunk: np.ndarray
    sample_eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    bandwidth: float
    concentration: float


def _epanechnikov_hilbert(x: np.ndarray) -> np.ndarray:
    """Hilbert transform of the Epanechnikov kernel used by Ledoit-Wolf (2020).

    Kernel ``k(x) = 3 / (4 sqrt 5) * max(1 - x^2 / 5, 0)`` (unit variance,
    support ``[-sqrt 5, sqrt 5]``).  With the convention
    ``H_k(x) = (1 / pi) PV int k(t) / (t - x) dt``,

    ``H_k(x) = -3 x / (10 pi)
    + 3 / (4 sqrt 5 pi) (1 - x^2 / 5) log|(sqrt 5 - x) / (sqrt 5 + x)|``,

    with the log term taken as 0 at ``|x| = sqrt 5`` (its limit).
    """
    lin = (-3.0 / (10.0 * math.pi)) * x
    with np.errstate(divide="ignore", invalid="ignore"):
        logt = np.log(np.abs((_SQRT5 - x) / (_SQRT5 + x)))
        term = (3.0 / (4.0 * _SQRT5 * math.pi)) * (1.0 - x * x / 5.0) * logt
    term = np.where(np.abs(x) == _SQRT5, 0.0, term)
    out: np.ndarray = lin + term
    return out


def _kernel_density_and_hilbert(lam: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    """Kernel estimates of the sample spectral density and its Hilbert transform.

    Variable-bandwidth Epanechnikov estimator of Ledoit and Wolf (2020,
    section 4): eigenvalue ``lam_j`` gets bandwidth ``h lam_j``, so at
    ``lam_i``

    ``f(lam_i) = (1/m) sum_j k((lam_i - lam_j) / (h lam_j)) / (h lam_j)``,
    ``Hf(lam_i) = (1/m) sum_j H_k((lam_i - lam_j) / (h lam_j)) / (h lam_j)``.
    """
    bw = h * lam[None, :]
    xx = (lam[:, None] - lam[None, :]) / bw
    kern = (3.0 / (4.0 * _SQRT5)) * np.maximum(1.0 - xx * xx / 5.0, 0.0)
    f = np.mean(kern / bw, axis=1)
    hf = np.mean(_epanechnikov_hilbert(xx) / bw, axis=1)
    return f, hf


def _oracle_eigenvalues(lam: np.ndarray, f: np.ndarray, hf: np.ndarray, c: float) -> np.ndarray:
    """Ledoit-Wolf (2020) nonlinear shrinkage formula for ``c = p / n <= 1``.

    ``d_i = lam_i / ((pi c lam_i f_i)^2 + (1 - c - pi c lam_i Hf_i)^2)``,
    the plug-in version of the oracle
    ``lam / |1 - c - c lam m(lam)|^2`` with ``m(lam) = pi Hf + i pi f``.
    """
    a = math.pi * c * lam * f
    b = 1.0 - c - math.pi * c * lam * hf
    out: np.ndarray = lam / (a * a + b * b)
    return out


def _singular_oracle(lam: np.ndarray, f: np.ndarray, hf: np.ndarray) -> np.ndarray:
    """Formula for the n non-zero sample eigenvalues when ``c = p / n > 1``.

    ``d_i = 1 / (pi^2 lam_i (f_i^2 + Hf_i^2))``, where ``f`` and ``Hf`` are
    estimated from the non-zero eigenvalues only.
    """
    out: np.ndarray = 1.0 / (math.pi**2 * lam * (f * f + hf * hf))
    return out


def _null_eigenvalue(c: float, hf0: float) -> float:
    """Shrunk value of the ``p - n`` null sample eigenvalues when ``c > 1``.

    ``d0 = 1 / (pi (c - 1) Hf(0))``, with ``Hf(0)`` the Hilbert transform
    of the non-zero spectral density at zero.
    """
    return 1.0 / (math.pi * (c - 1.0) * hf0)


def nonlinear_shrinkage(returns: np.ndarray) -> NonlinearShrinkageResult:
    """Analytical nonlinear shrinkage of Ledoit and Wolf (2020).

    Rotation-equivariant estimator ``U diag(d) U'`` that keeps the sample
    eigenvectors ``U`` and replaces each sample eigenvalue by an estimate
    of the oracle ``u_i' Sigma u_i``.  Steps, for N assets, effective
    sample size ``n`` and ``c = N / n``:

    1. Demean, ``S = Xc' Xc / n`` with ``n = T - 1`` (the ddof=1
       convention; the 2020 paper writes the zero-mean case with divisor
       T; the authors' updated reference code makes the same ``T - 1``
       adjustment after demeaning).  Eigen-decompose ``S``; keep the
       ``m = min(N, n)`` largest eigenvalues ``lam``.
    2. Global bandwidth ``h = n ** (-1/3)``; local bandwidth ``h lam_j``.
    3. Epanechnikov kernel estimates ``f`` of the limiting sample spectral
       density and ``Hf`` of its Hilbert transform at each ``lam_i``
       (:func:`_kernel_density_and_hilbert`).
    4. If ``N <= n``:
       ``d_i = lam_i / ((pi c lam_i f_i)^2 + (1 - c - pi c lam_i Hf_i)^2)``.
       If ``N > n`` (singular sample), the ``N - n`` null eigenvalues get
       ``d0 = 1 / (pi (c - 1) Hf(0))`` with
       ``Hf(0) = (1/m) sum_j H_k(-1 / h) / (h lam_j)`` (the paper's
       appendix gives the same closed form for ``h < 1 / sqrt 5``), and
       the others ``d_i = 1 / (pi^2 lam_i (f_i^2 + Hf_i^2))``.  The null
       space enters as ``d0 (I - U1 U1')`` so the result does not depend
       on how the eigen solver picks a basis for it.

    Assumptions: the large-dimensional asymptotics of the paper (N and n
    grow at a fixed ratio, iid rows with finite fourth moments).  For
    small N the formula is still defined, and by construction tends to
    the sample eigenvalues when ``c -> 0``, but its optimality claims are
    asymptotic.  The 2020 estimator does not force the shrunk eigenvalues
    to keep the order of the sample ones; with few assets neighbouring
    values can swap slightly.

    The output is positive definite, equivariant to permutations and to
    rescaling all returns (``a X -> a^2 Sigma``), and shares the sample's
    eigenvectors.  Requires T >= 2 and that the ``min(N, T - 1)`` largest
    sample eigenvalues are positive (no exact collinearity among the
    assets); ValueError otherwise.
    """
    x = _returns(returns, min_rows=2)
    t, p = x.shape
    n = t - 1
    s = _sample_cov(x)
    lam_all, u = np.linalg.eigh(s)
    m = min(p, n)
    lam = lam_all[p - m :]
    top = float(lam[-1])
    if not (top > 0.0 and float(lam[0]) > _EIG_RTOL * top):
        raise ValueError(
            "nonlinear_shrinkage: the sample covariance has a zero eigenvalue among its "
            f"min(N, T-1) = {m} largest (collinear assets or a constant column)"
        )
    c = p / n
    h = float(n) ** (-1.0 / 3.0)
    f, hf = _kernel_density_and_hilbert(lam, h)
    shrunk = np.empty(p)
    if p <= n:
        d = _oracle_eigenvalues(lam, f, hf, c)
        shrunk[:] = d
        sigma = (u * d) @ u.T
    else:
        hk0 = float(_epanechnikov_hilbert(np.array([-1.0 / h]))[0])
        hf0 = float(np.mean(hk0 / (h * lam)))
        d0 = _null_eigenvalue(c, hf0)
        d1 = _singular_oracle(lam, f, hf)
        shrunk[: p - m] = d0
        shrunk[p - m :] = d1
        u1 = u[:, p - m :]
        sigma = (u1 * d1) @ u1.T + d0 * (np.eye(p) - u1 @ u1.T)
    if not (np.isfinite(shrunk).all() and np.all(shrunk > 0.0)):
        raise ValueError("nonlinear_shrinkage: non-positive or non-finite shrunk eigenvalue")
    return NonlinearShrinkageResult(
        sigma=_sym(sigma),
        shrunk=shrunk,
        sample_eigenvalues=lam_all,
        eigenvectors=u,
        bandwidth=h,
        concentration=c,
    )


# --------------------------------------------------------------------------
# Statistical factor model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorModelResult:
    """Output of :func:`pca_factor_covariance`: ``sigma = B F B' + diag(D)``.

    Attributes
    ----------
    sigma:
        Model covariance.
    loadings:
        ``B``, N x k, orthonormal columns (the top-k sample eigenvectors,
        sign fixed so that each column's largest-magnitude entry is
        positive).
    factor_variances:
        Diagonal of ``F`` (the top-k sample eigenvalues, descending).
    specific_variances:
        ``D``, floored as described in :func:`pca_factor_covariance`.
    explained:
        Fraction of the sample's total variance carried by the k factors.
    """

    sigma: np.ndarray
    loadings: np.ndarray
    factor_variances: np.ndarray
    specific_variances: np.ndarray
    explained: float


def pca_factor_covariance(
    returns: np.ndarray, k: int, *, specific_floor: float = 1e-4
) -> FactorModelResult:
    """Statistical factor model from the principal components of ``S``.

    With the ddof=1 sample covariance ``S = V Lambda V'`` (eigenvalues
    descending), take the k leading components: ``B = V_k``,
    ``F = diag(Lambda_k)``.  The specific variances are the residual
    diagonal, floored relative to each asset's own variance:

    ``D_i = max(S_ii - (B F B')_ii, specific_floor * S_ii)``.

    The residual diagonal is mathematically non-negative (it is the
    diagonal of the discarded components), but it can be zero or round to
    a tiny negative value, for instance with ``k = N`` or an asset that
    loads on the factors alone.  The floor keeps ``sigma`` positive
    definite whenever every sample variance is positive.

    Assumptions: returns are driven by a few common factors identified by
    variance alone, with uncorrelated specific risk.  The model is
    equivariant to permuting assets and to rescaling all returns together,
    not to rescaling one asset (PCA on the covariance favours high-variance
    assets).

    Requires 1 <= k <= N, ``0 < specific_floor < 1`` and positive sample
    variances; ValueError otherwise.
    """
    x = _returns(returns, min_rows=2)
    n = x.shape[1]
    if isinstance(k, bool) or int(k) != k or not 1 <= int(k) <= n:
        raise ValueError(f"k must be an integer in [1, N={n}], got {k!r}")
    kk = int(k)
    fl = float(specific_floor)
    if not (math.isfinite(fl) and 0.0 < fl < 1.0):
        raise ValueError(f"specific_floor must lie in (0, 1), got {specific_floor!r}")
    s = _sample_cov(x)
    diag = np.diag(s).copy()
    if not np.all(diag > _VAR_RTOL * float(np.max(diag))):
        raise ValueError(
            "pca_factor_covariance requires strictly positive sample variances "
            "(a column is constant to working precision)"
        )
    lam, v = np.linalg.eigh(s)
    order = np.argsort(-lam, kind="stable")
    lam = np.maximum(lam[order][:kk], 0.0)
    b = v[:, order][:, :kk].copy()
    for j in range(kk):
        col = b[:, j]
        if col[int(np.argmax(np.abs(col)))] < 0.0:
            b[:, j] = -col
    common = (b * lam) @ b.T
    resid = diag - np.diag(common)
    spec = np.maximum(resid, fl * diag)
    sigma = common + np.diag(spec)
    explained = math.fsum(lam.tolist()) / math.fsum(diag.tolist())
    return FactorModelResult(
        sigma=_sym(sigma),
        loadings=b,
        factor_variances=lam,
        specific_variances=spec,
        explained=explained,
    )
