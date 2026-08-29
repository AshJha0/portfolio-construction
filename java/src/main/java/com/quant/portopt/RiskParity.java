package com.quant.portopt;

import java.util.Arrays;

/**
 * Risk parity / risk budgeting via cyclical coordinate descent.
 *
 * <p>CCD update derivation (Griveau-Billion, Richard &amp; Roncalli 2013 style,
 * log-barrier formulation; pinned in API_SPEC section 5). Solve
 *
 * <pre>min_(y &gt; 0)  F(y) = 1/2 y' Sigma y - sum_i b_i ln y_i</pre>
 *
 * for risk budgets b_i &gt; 0 (normalized to sum 1). Coordinate-wise, with
 * {@code m_i = (Sigma y)_i - Sigma_ii y_i} (the cross term, held fixed):
 *
 * <pre>
 *   dF/dy_i = Sigma_ii y_i + m_i - b_i / y_i = 0
 *   =&gt;  y_i = ( -m_i + sqrt(m_i^2 + 4 Sigma_ii b_i) ) / (2 Sigma_ii)
 * </pre>
 *
 * taking the positive root (the discriminant is &gt; 0 because Sigma_ii &gt; 0
 * and b_i &gt; 0, so y_i stays strictly positive). At the fixed point
 * {@code y_i (Sigma y)_i = b_i} exactly, so the normalized weights
 * {@code w = y / sum(y)} have risk contributions equal to the budgets. F is
 * strictly convex on the positive orthant, so the solution is unique and CCD
 * converges monotonically.
 *
 * <p>Pinned iteration: initialize {@code y_i = 1/n}; cache {@code sy = Sigma y}
 * and update it incrementally after each coordinate change; sweep coordinates
 * in ascending order. Converged when {@code max_i |y_i - y_prev_sweep_i| <
 * 1e-10} after a full sweep; at most 10000 sweeps (non-convergence is flagged,
 * never thrown).
 */
public final class RiskParity {

    /** Pinned sweep-level convergence tolerance. */
    public static final double CCD_TOL = 1.0e-10;
    /** Pinned sweep cap (non-convergence is flagged, never thrown). */
    public static final int CCD_MAX_SWEEPS = 10_000;

    private RiskParity() {
    }

    /**
     * Result of the CCD risk-budget solver.
     *
     * @param weights   long-only weights on the simplex
     * @param sweeps    full sweeps executed
     * @param converged whether the pinned stopping rule fired before the cap
     */
    public record Solution(double[] weights, int sweeps, boolean converged) {
    }

    /**
     * Normalized risk contributions
     * {@code RC_i = w_i (Sigma w)_i / (w' Sigma w)}; the contributions sum to
     * exactly 1 (portfolio variance normalized).
     *
     * @param w     portfolio weights
     * @param sigma symmetric covariance
     * @return risk-contribution vector
     * @throws IllegalArgumentException if lengths mismatch, weights are not
     *         finite, or the portfolio variance is not strictly positive
     */
    public static double[] riskContributions(double[] w, double[][] sigma) {
        double[][] s = Linalg.checkSquareSymmetric(sigma);
        if (w == null || w.length != s.length) {
            throw new IllegalArgumentException("weights length "
                    + (w == null ? 0 : w.length) + " does not match sigma dimension " + s.length);
        }
        for (double v : w) {
            if (!Double.isFinite(v)) {
                throw new IllegalArgumentException("weights contain NaN or infinite entries");
            }
        }
        double[] sw = Linalg.matVec(s, w);
        double var = Linalg.dot(w, sw);
        if (var <= 0.0) {
            throw new IllegalArgumentException("portfolio variance must be strictly positive");
        }
        double[] rc = new double[w.length];
        for (int i = 0; i < w.length; i++) {
            rc[i] = w[i] * sw[i] / var;
        }
        return rc;
    }

    /**
     * Naive inverse-volatility weights
     * {@code w_i = (1/sqrt(S_ii)) / sum_j (1/sqrt(S_jj))}.
     *
     * @param sigma symmetric covariance with strictly positive diagonal
     * @return inverse-vol weights summing to 1
     */
    public static double[] inverseVolWeights(double[][] sigma) {
        double[][] s = Linalg.checkSquareSymmetric(sigma);
        int n = s.length;
        double[] inv = new double[n];
        double tot = 0.0;
        for (int i = 0; i < n; i++) {
            if (s[i][i] <= 0) {
                throw new IllegalArgumentException(
                        "all variances must be strictly positive for inverse-vol");
            }
            inv[i] = 1.0 / Math.sqrt(s[i][i]);
            tot += inv[i];
        }
        double[] w = new double[n];
        for (int i = 0; i < n; i++) {
            w[i] = inv[i] / tot;
        }
        return w;
    }

    private static double[] validateBudgets(double[] budgets, int n) {
        double[] b = new double[n];
        if (budgets == null) {
            Arrays.fill(b, 1.0 / n);
            return b;
        }
        if (budgets.length != n) {
            throw new IllegalArgumentException("budgets must have length " + n);
        }
        double tot = 0.0;
        for (double v : budgets) {
            if (!Double.isFinite(v) || v <= 0) {
                throw new IllegalArgumentException("risk budgets must be positive and finite");
            }
            tot += v;
        }
        for (int i = 0; i < n; i++) {
            b[i] = budgets[i] / tot;
        }
        return b;
    }

    /**
     * General risk budgeting via the pinned CCD (class javadoc).
     *
     * <p>An almost-zero-vol asset (tiny {@code Sigma_ii}) yields a large
     * {@code y_i} and hence a weight near 1 after normalization — the correct
     * ERC limit; convergence is preserved because the coordinate quadratic
     * keeps a positive discriminant.
     *
     * @param sigma     SPD covariance (validated via Cholesky; singular or
     *                  non-PSD input errors)
     * @param budgets   target risk budgets b_i &gt; 0 (normalized internally),
     *                  or null for equal budgets (ERC)
     * @param tol       sweep-level convergence tolerance (pinned default 1e-10)
     * @param maxSweeps sweep cap (pinned default 10000)
     * @return weights, sweep count and convergence flag
     */
    public static Solution riskBudgetSolve(double[][] sigma, double[] budgets, double tol,
            int maxSweeps) {
        double[][] s = Linalg.checkSquareSymmetric(sigma);
        Linalg.cholesky(s); // SPD gate: rejects singular / non-PSD inputs
        int n = s.length;
        double[] b = validateBudgets(budgets, n);
        if (!Double.isFinite(tol) || tol <= 0) {
            throw new IllegalArgumentException("tol must be positive");
        }
        if (maxSweeps < 1) {
            throw new IllegalArgumentException("max_sweeps must be >= 1");
        }

        double[] y = new double[n];
        Arrays.fill(y, 1.0 / n);
        double[] sy = Linalg.matVec(s, y); // cached Sigma y, updated incrementally
        double[] yPrev = new double[n];
        for (int sweep = 1; sweep <= maxSweeps; sweep++) {
            System.arraycopy(y, 0, yPrev, 0, n);
            for (int i = 0; i < n; i++) {
                double mI = sy[i] - s[i][i] * y[i];
                double yNew = (-mI + Math.sqrt(mI * mI + 4.0 * s[i][i] * b[i]))
                        / (2.0 * s[i][i]);
                double dy = yNew - y[i];
                if (dy != 0.0) {
                    for (int r = 0; r < n; r++) {
                        sy[r] += s[r][i] * dy;
                    }
                    y[i] = yNew;
                }
            }
            double maxStep = 0.0;
            for (int i = 0; i < n; i++) {
                maxStep = Math.max(maxStep, Math.abs(y[i] - yPrev[i]));
            }
            if (maxStep < tol) {
                return new Solution(normalize(y), sweep, true);
            }
        }
        return new Solution(normalize(y), maxSweeps, false);
    }

    private static double[] normalize(double[] y) {
        double tot = Linalg.sum(y);
        double[] w = new double[y.length];
        for (int i = 0; i < y.length; i++) {
            w[i] = y[i] / tot;
        }
        return w;
    }

    /**
     * Risk-budget weights with the pinned tolerances (weights only).
     *
     * @param sigma   SPD covariance
     * @param budgets target risk budgets, or null for equal budgets (ERC)
     * @return long-only weights on the simplex
     */
    public static double[] riskBudgetWeights(double[][] sigma, double[] budgets) {
        return riskBudgetSolve(sigma, budgets, CCD_TOL, CCD_MAX_SWEEPS).weights();
    }

    /**
     * Equal-risk-contribution weights (equal budgets b_i = 1/n).
     *
     * @param sigma SPD covariance
     * @return ERC weights
     */
    public static double[] ercWeights(double[][] sigma) {
        return riskBudgetWeights(sigma, null);
    }
}
