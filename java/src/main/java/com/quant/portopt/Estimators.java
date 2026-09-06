package com.quant.portopt;

/**
 * Return-moment estimators: sample moments and Ledoit-Wolf shrinkage.
 *
 * <p>Pinned conventions (API_SPEC.md section 2):
 * <ul>
 *   <li>sample mean uses divisor T; sample covariance uses divisor T-1 (ddof=1);</li>
 *   <li>Ledoit-Wolf (2004) constant-correlation shrinkage: the intensity is
 *       computed exactly as in the paper's Appendix B / covCor.m (divisor T
 *       throughout, both theta terms per ordered pair) and applied in the
 *       ddof=1 convention — see {@link #shrinkCovariance};</li>
 *   <li>annualization multiplies means and covariances by 252 (configurable).</li>
 * </ul>
 *
 * <p>Every T-length reduction uses the compensated {@link Linalg#kahanSum},
 * matching the reference's correctly rounded sums to a couple of ulps; no
 * golden value depends on a BLAS or a summation order.
 */
public final class Estimators {

    private Estimators() {
    }

    /**
     * Ledoit-Wolf shrinkage output.
     *
     * @param sigma     shrunk covariance {@code delta * F + (1 - delta) * S}
     * @param intensity shrinkage intensity delta in [0, 1]
     * @param rbar      average sample correlation of the constant-correlation target
     * @param sample    sample covariance S (ddof=1)
     * @param target    constant-correlation target F
     */
    public record ShrinkageResult(
            double[][] sigma, double intensity, double rbar, double[][] sample, double[][] target) {
    }

    /**
     * Validates a T x N simple-returns matrix: finite entries strictly greater
     * than -1 (a simple return at or below -100% is not a valid price move),
     * T &ge; minRows, N &ge; 1, rectangular.
     *
     * @param returns candidate matrix
     * @param minRows minimum number of rows required
     * @return the validated matrix (same reference; never mutated)
     * @throws IllegalArgumentException on any violation
     */
    public static double[][] validateReturns(double[][] returns, int minRows) {
        if (returns == null || returns.length < minRows) {
            throw new IllegalArgumentException("returns must contain at least " + minRows
                    + " rows, got " + (returns == null ? 0 : returns.length));
        }
        if (returns.length == 0 || returns[0] == null || returns[0].length < 1) {
            throw new IllegalArgumentException("returns must contain at least one asset column");
        }
        int n = returns[0].length;
        for (double[] row : returns) {
            if (row == null || row.length != n) {
                throw new IllegalArgumentException("returns matrix must be rectangular");
            }
            for (double v : row) {
                if (!Double.isFinite(v)) {
                    throw new IllegalArgumentException("returns contain NaN or infinite entries");
                }
                if (v <= -1.0) {
                    throw new IllegalArgumentException(
                            "returns contain a simple return <= -1 (-100% or worse)");
                }
            }
        }
        return returns;
    }

    private static void validatePeriods(int periods) {
        if (periods <= 0) {
            throw new IllegalArgumentException("periods must be positive and finite");
        }
    }

    /**
     * Per-asset sample mean {@code mu[i] = (1/T) * sum_t X[t][i]} (compensated sum).
     *
     * @param returns T x N returns matrix (T &ge; 1)
     * @return mean vector of length N
     */
    public static double[] sampleMean(double[][] returns) {
        double[][] x = validateReturns(returns, 1);
        int t = x.length;
        int n = x[0].length;
        double[] mu = new double[n];
        double[] col = new double[t];
        for (int i = 0; i < n; i++) {
            for (int r = 0; r < t; r++) {
                col[r] = x[r][i];
            }
            mu[i] = Linalg.kahanSum(col) / t;
        }
        return mu;
    }

    private static double[][] demean(double[][] x) {
        double[] mu = sampleMean(x);
        int t = x.length;
        int n = x[0].length;
        double[][] xc = new double[t][n];
        for (int r = 0; r < t; r++) {
            for (int i = 0; i < n; i++) {
                xc[r][i] = x[r][i] - mu[i];
            }
        }
        return xc;
    }

    /** {@code S[i][j] = (1/(T-1)) sum_t xc[t][i] xc[t][j]} with a compensated sum. */
    private static double[][] covFromDemeaned(double[][] xc) {
        int t = xc.length;
        int n = xc[0].length;
        double[][] s = new double[n][n];
        double[] prod = new double[t];
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                for (int r = 0; r < t; r++) {
                    prod[r] = xc[r][i] * xc[r][j];
                }
                double v = Linalg.kahanSum(prod) / (t - 1);
                s[i][j] = v;
                s[j][i] = v;
            }
        }
        return s;
    }

    /**
     * Sample covariance with ddof=1: demean columns, then
     * {@code S[i][j] = (1/(T-1)) * sum_t xc[t][i] * xc[t][j]}.
     *
     * @param returns T x N returns matrix (T &ge; 2)
     * @return N x N covariance matrix
     */
    public static double[][] sampleCov(double[][] returns) {
        double[][] x = validateReturns(returns, 2);
        return covFromDemeaned(demean(x));
    }

    /**
     * Ledoit-Wolf (2004) shrinkage towards the constant-correlation target,
     * pinned term by term to the paper's Appendix B / covCor.m (API_SPEC
     * section 2.2). With demeaned returns {@code x[t][i]}, {@code S_T = Xc'Xc / T}
     * (divisor T) and {@code s_ii = S_T[i][i]}:
     * <ul>
     *   <li>{@code rbar = (2 / (N(N-1))) * sum_(i<j) S_T_ij / sqrt(s_ii s_jj)}</li>
     *   <li>{@code pi_ij = (1/T) sum_t (x_ti x_tj - S_T_ij)^2}; {@code pi_hat = sum_ij pi_ij}</li>
     *   <li>{@code theta_ii,ij = (1/T) sum_t (x_ti^2 - s_ii)(x_ti x_tj - S_T_ij)}</li>
     *   <li>{@code rho_hat = sum_i pi_ii + sum_(i != j) rbar * sqrt(s_jj / s_ii) * theta_ii,ij}</li>
     *   <li>{@code gamma_hat = sum_ij (F_T_ij - S_T_ij)^2}, {@code F_T_ii = s_ii},
     *       {@code F_T_ij = rbar sqrt(s_ii s_jj)}</li>
     *   <li>{@code kappa = (pi_hat - rho_hat) / gamma_hat};
     *       {@code delta = max(0, min(1, kappa / T))} (delta = 0 if gamma_hat &le; 0)</li>
     * </ul>
     * The intensity is exactly the paper's; the returned matrices use the
     * project-wide ddof=1 convention: {@code sample = S = T/(T-1) S_T},
     * {@code target = F = T/(T-1) F_T}, {@code sigma = delta F + (1 - delta) S}.
     *
     * @param returns T x N returns matrix (T &ge; 2, N &ge; 2, positive variances)
     * @return the shrinkage result (intensity, rbar, S, F and the shrunk matrix)
     * @throws IllegalArgumentException if N &lt; 2 or any sample variance is not
     *         strictly positive
     */
    public static ShrinkageResult shrinkCovariance(double[][] returns) {
        double[][] x = validateReturns(returns, 2);
        int t = x.length;
        int n = x[0].length;
        if (n < 2) {
            throw new IllegalArgumentException("shrinkage requires at least 2 assets");
        }
        double[][] xc = demean(x);
        double[][] s = covFromDemeaned(xc);
        for (int i = 0; i < n; i++) {
            if (s[i][i] <= 0.0) {
                throw new IllegalArgumentException(
                        "shrinkage requires strictly positive sample variances");
            }
        }
        double tf = t;
        // Paper convention (divisor T) for every dispersion quantity.
        double[][] sT = new double[n][n];
        double[] varT = new double[n];
        double[] sdT = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                sT[i][j] = s[i][j] * ((tf - 1.0) / tf);
            }
        }
        for (int i = 0; i < n; i++) {
            varT[i] = sT[i][i];
            sdT[i] = Math.sqrt(varT[i]);
        }

        double corrSum = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                corrSum += sT[i][j] / (sdT[i] * sdT[j]);
            }
        }
        double rbar = 2.0 * corrSum / ((double) n * (n - 1));

        double[][] fT = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                fT[i][j] = (i == j) ? varT[i] : rbar * sdT[i] * sdT[j];
            }
        }

        double piHat = 0.0;
        double rhoHat = 0.0;
        double gammaHat = 0.0;
        double[] terms = new double[t];
        double[] xi2Dev = new double[t];
        for (int i = 0; i < n; i++) {
            for (int r = 0; r < t; r++) {
                xi2Dev[r] = xc[r][i] * xc[r][i] - varT[i];
            }
            for (int j = 0; j < n; j++) {
                double sij = sT[i][j];
                for (int r = 0; r < t; r++) {
                    double dev = xc[r][i] * xc[r][j] - sij;
                    terms[r] = dev * dev;
                }
                double pij = Linalg.kahanSum(terms) / tf;
                piHat += pij;
                double gd = fT[i][j] - sij;
                gammaHat += gd * gd;
                if (i == j) {
                    rhoHat += pij;
                } else {
                    for (int r = 0; r < t; r++) {
                        terms[r] = xi2Dev[r] * (xc[r][i] * xc[r][j] - sij);
                    }
                    double theta = Linalg.kahanSum(terms) / tf;
                    rhoHat += rbar * Math.sqrt(varT[j] / varT[i]) * theta;
                }
            }
        }

        double delta;
        if (gammaHat <= 0.0) {
            delta = 0.0;
        } else {
            double kappa = (piHat - rhoHat) / gammaHat;
            delta = Math.max(0.0, Math.min(1.0, kappa / tf));
        }

        // Output in the project-wide ddof=1 convention.
        double[] sd = new double[n];
        for (int i = 0; i < n; i++) {
            sd[i] = Math.sqrt(s[i][i]);
        }
        double[][] f = new double[n][n];
        double[][] sigma = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                f[i][j] = (i == j) ? s[i][i] : rbar * sd[i] * sd[j];
                sigma[i][j] = delta * f[i][j] + (1.0 - delta) * s[i][j];
            }
        }
        return new ShrinkageResult(sigma, delta, rbar, s, f);
    }

    /**
     * Annualizes a per-period mean vector: {@code mu_ann = periods * mu}.
     *
     * @param mu      per-period mean vector (finite)
     * @param periods periods per year (must be positive)
     * @return annualized copy
     */
    public static double[] annualizeMean(double[] mu, int periods) {
        validatePeriods(periods);
        double[] out = new double[mu.length];
        for (int i = 0; i < mu.length; i++) {
            if (!Double.isFinite(mu[i])) {
                throw new IllegalArgumentException("mu contains NaN or infinite entries");
            }
            out[i] = periods * mu[i];
        }
        return out;
    }

    /**
     * Annualizes a per-period covariance: {@code Sigma_ann = periods * Sigma}.
     *
     * @param cov     per-period covariance (finite)
     * @param periods periods per year (must be positive)
     * @return annualized copy
     */
    public static double[][] annualizeCov(double[][] cov, int periods) {
        validatePeriods(periods);
        double[][] out = new double[cov.length][];
        for (int i = 0; i < cov.length; i++) {
            out[i] = new double[cov[i].length];
            for (int j = 0; j < cov[i].length; j++) {
                if (!Double.isFinite(cov[i][j])) {
                    throw new IllegalArgumentException("cov contains NaN or infinite entries");
                }
                out[i][j] = periods * cov[i][j];
            }
        }
        return out;
    }
}
