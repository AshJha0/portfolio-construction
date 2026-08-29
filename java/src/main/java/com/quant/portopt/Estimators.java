package com.quant.portopt;

/**
 * Return-moment estimators: sample moments and Ledoit-Wolf shrinkage.
 *
 * <p>Pinned conventions (API_SPEC.md section 2):
 * <ul>
 *   <li>sample mean uses divisor T; sample covariance uses divisor T-1 (ddof=1);</li>
 *   <li>Ledoit-Wolf (2004) constant-correlation shrinkage with the classic
 *       closed-form intensity; the dispersion sums (pi/rho/gamma) use divisor T
 *       while S keeps divisor T-1 — this deliberate simplification is pinned so
 *       every language port computes the identical number;</li>
 *   <li>annualization multiplies means and covariances by 252 (configurable).</li>
 * </ul>
 *
 * <p>The mean and covariance accumulations replicate the reference's numpy
 * arithmetic operation for operation (see the method notes) — the covariance
 * feeds the projected-gradient golden cases, which pin an iterate that is
 * sensitive to single-ulp input changes. The Ledoit-Wolf dispersion sums,
 * whose tolerances are far looser, use Kahan compensated summation.
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
     * Validates a T x N returns matrix (finite entries, T &ge; minRows, N &ge; 1,
     * rectangular).
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
            }
        }
        return returns;
    }

    /** Kahan-compensated sum of {@code f(t)} over t = 0..len-1. */
    private interface TermFn {
        double term(int t);
    }

    private static double kahanSum(int len, TermFn f) {
        double s = 0.0;
        double c = 0.0;
        for (int t = 0; t < len; t++) {
            double y = f.term(t) - c;
            double u = s + y;
            c = (u - s) - y;
            s = u;
        }
        return s;
    }

    /**
     * Per-asset sample mean {@code mu[i] = (1/T) * sum_t X[t][i]}.
     *
     * <p>The column sums are plain sequential additions over t — exactly what
     * the reference's {@code x.sum(axis=0)} executes for a row-major matrix —
     * so the demeaned columns feeding the covariance match bit for bit.
     *
     * @param returns T x N returns matrix (T &ge; 1)
     * @return mean vector of length N
     */
    public static double[] sampleMean(double[][] returns) {
        double[][] x = validateReturns(returns, 1);
        int t = x.length;
        int n = x[0].length;
        double[] mu = new double[n];
        for (int r = 0; r < t; r++) {
            for (int i = 0; i < n; i++) {
                mu[i] += x[r][i];
            }
        }
        for (int i = 0; i < n; i++) {
            mu[i] /= t;
        }
        return mu;
    }

    /**
     * Sample covariance with ddof=1: demean columns, then
     * {@code S[i][j] = (1/(T-1)) * sum_t xc[t][i] * xc[t][j]}.
     *
     * <p>The Gram accumulation replicates the reference's {@code xc.T @ xc}
     * (a BLAS syrk on an AVX-512 host) operation for operation: T is split
     * into chunks — repeatedly take 384 while at least 768 rows remain, then
     * split the sub-768 remainder in ceil-halves — each chunk is reduced by a
     * sequential fused multiply-add, and the chunk sums are added in order.
     * This bit-exactness is required downstream: the projected-gradient golden
     * values pin an iterate whose stopping point shifts by ~1e-8 per ulp of
     * covariance perturbation (see Linalg's kernel notes).
     *
     * @param returns T x N returns matrix (T &ge; 2)
     * @return N x N covariance matrix
     */
    public static double[][] sampleCov(double[][] returns) {
        double[][] x = validateReturns(returns, 2);
        int t = x.length;
        int n = x[0].length;
        double[][] xc = demean(x);
        int[] splits = gramChunks(t);
        double[][] s = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                double acc = 0.0;
                int k0 = 0;
                for (int len : splits) {
                    double chunk = 0.0;
                    for (int k = k0; k < k0 + len; k++) {
                        chunk = Math.fma(xc[k][i], xc[k][j], chunk);
                    }
                    acc = acc + chunk;
                    k0 += len;
                }
                double v = acc / (t - 1);
                s[i][j] = v;
                s[j][i] = v;
            }
        }
        return s;
    }

    /**
     * The pinned K-chunking of the Gram accumulation (OpenBLAS level-3 rule
     * with blocking parameter 384): while at least 2*384 rows remain take 384,
     * a remainder in (384, 768) is split into ceil- and floor-halves, and a
     * remainder of at most 384 is one chunk.
     */
    private static int[] gramChunks(int t) {
        final int q = 384;
        int count = 0;
        int rem = t;
        while (rem > 0) {
            rem -= (rem >= 2 * q) ? q : (rem > q ? (rem + 1) / 2 : rem);
            count++;
        }
        int[] out = new int[count];
        rem = t;
        for (int c = 0; c < count; c++) {
            int len = (rem >= 2 * q) ? q : (rem > q ? (rem + 1) / 2 : rem);
            out[c] = len;
            rem -= len;
        }
        return out;
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

    /**
     * Ledoit-Wolf (2004) shrinkage towards the constant-correlation target,
     * pinned term by term (API_SPEC section 2.2). With demeaned returns
     * {@code x[t][i]} and S the ddof=1 sample covariance:
     * <ul>
     *   <li>{@code rbar = (2 / (N(N-1))) * sum_(i<j) S_ij / sqrt(S_ii S_jj)}</li>
     *   <li>target {@code F_ii = S_ii}, {@code F_ij = rbar * sqrt(S_ii S_jj)}</li>
     *   <li>{@code pi_ij = (1/T) sum_t (x_ti x_tj - S_ij)^2}; {@code pi_hat = sum_ij pi_ij}</li>
     *   <li>{@code theta_ii,ij = (1/T) sum_t (x_ti^2 - S_ii)(x_ti x_tj - S_ij)}</li>
     *   <li>{@code rho_hat = sum_i pi_ii + sum_(i != j) (rbar/2) * sqrt(S_jj/S_ii) * theta_ii,ij}
     *       (the ordered double sum supplies both symmetric terms of the classic
     *       formula exactly once each)</li>
     *   <li>{@code gamma_hat = sum_ij (F_ij - S_ij)^2}</li>
     *   <li>{@code kappa = (pi_hat - rho_hat) / gamma_hat};
     *       {@code delta = max(0, min(1, kappa / T))} (delta = 0 if gamma_hat &le; 0)</li>
     * </ul>
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
        double[][] s = sampleCov(x);
        double[] sd = new double[n];
        for (int i = 0; i < n; i++) {
            if (s[i][i] <= 0.0) {
                throw new IllegalArgumentException(
                        "shrinkage requires strictly positive sample variances");
            }
            sd[i] = Math.sqrt(s[i][i]);
        }

        double corrSum = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                corrSum += s[i][j] / (sd[i] * sd[j]);
            }
        }
        double rbar = 2.0 * corrSum / ((double) n * (n - 1));

        double[][] f = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                f[i][j] = (i == j) ? s[i][i] : rbar * sd[i] * sd[j];
            }
        }

        double piHat = 0.0;
        double rhoHat = 0.0;
        double gammaHat = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                final int ci = i;
                final int cj = j;
                final double sij = s[i][j];
                double pij = kahanSum(t, r -> {
                    double dev = xc[r][ci] * xc[r][cj] - sij;
                    return dev * dev;
                }) / t;
                piHat += pij;
                if (i == j) {
                    rhoHat += pij;
                }
                double diff = f[i][j] - s[i][j];
                gammaHat += diff * diff;
            }
        }
        for (int i = 0; i < n; i++) {
            final int ci = i;
            final double sii = s[i][i];
            for (int j = 0; j < n; j++) {
                if (i == j) {
                    continue;
                }
                final int cj = j;
                final double sij = s[i][j];
                double theta = kahanSum(t, r -> {
                    double xi2Dev = xc[r][ci] * xc[r][ci] - sii;
                    return xi2Dev * (xc[r][ci] * xc[r][cj] - sij);
                }) / t;
                // The symmetric partner term of the classic formula is produced
                // by the (j, i) pass of this same loop, so each unordered pair
                // contributes both terms exactly once.
                rhoHat += (rbar / 2.0) * Math.sqrt(s[j][j] / s[i][i]) * theta;
            }
        }

        double delta;
        if (gammaHat <= 0.0) {
            delta = 0.0;
        } else {
            double kappa = (piHat - rhoHat) / gammaHat;
            delta = Math.max(0.0, Math.min(1.0, kappa / t));
        }

        double[][] sigma = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                sigma[i][j] = delta * f[i][j] + (1.0 - delta) * s[i][j];
            }
        }
        return new ShrinkageResult(sigma, delta, rbar, s, f);
    }

    /**
     * Annualizes a per-period mean vector: {@code mu_ann = periods * mu}.
     *
     * @param mu      per-period mean vector
     * @param periods periods per year (must be positive)
     * @return annualized copy
     */
    public static double[] annualizeMean(double[] mu, int periods) {
        if (periods <= 0) {
            throw new IllegalArgumentException("periods must be positive");
        }
        double[] out = new double[mu.length];
        for (int i = 0; i < mu.length; i++) {
            out[i] = periods * mu[i];
        }
        return out;
    }

    /**
     * Annualizes a per-period covariance: {@code Sigma_ann = periods * Sigma}.
     *
     * @param cov     per-period covariance
     * @param periods periods per year (must be positive)
     * @return annualized copy
     */
    public static double[][] annualizeCov(double[][] cov, int periods) {
        if (periods <= 0) {
            throw new IllegalArgumentException("periods must be positive");
        }
        double[][] out = new double[cov.length][];
        for (int i = 0; i < cov.length; i++) {
            out[i] = new double[cov[i].length];
            for (int j = 0; j < cov[i].length; j++) {
                out[i][j] = periods * cov[i][j];
            }
        }
        return out;
    }
}
