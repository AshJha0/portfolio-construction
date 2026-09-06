package com.quant.portopt;

/**
 * Pinned dense linear algebra kernels for small SPD systems (n &le; 20).
 *
 * <p>Every routine is the reference algorithm of API_SPEC.md section 3 that all
 * language ports reproduce up to IEEE-754 rounding:
 * <ul>
 *   <li>Cholesky: Cholesky-Banachiewicz, lower triangular, computed row by row,
 *       with a pinned <em>relative</em> pivot threshold (a numerically singular
 *       matrix such as a duplicated-asset covariance yields a pivot that is a
 *       tiny rounding residual rather than an exact zero, so the guard must be
 *       relative);</li>
 *   <li>SPD solve: forward substitution with L, then back substitution with L';</li>
 *   <li>SPD inverse: column-by-column Cholesky solves against identity columns
 *       (no post-symmetrization).</li>
 * </ul>
 * All loops are explicit so the operation order is pinned. Long (T-length)
 * reductions use {@link #kahanSum}; every other sum is plain sequential.
 */
public final class Linalg {

    /** Symmetry tolerance: max|A - A'| must not exceed {@code SYM_TOL * (1 + max|A|)}. */
    static final double SYM_TOL = 1e-10;

    /**
     * Pinned relative pivot threshold: the Cholesky pivot s at row i must exceed
     * {@code PIVOT_RTOL * a[i][i]} (with {@code a[i][i] > 0} required). Rejects
     * matrices with an effective condition number beyond ~1e13 — far outside
     * anything a valid n &le; 20 covariance produces, but safely catching
     * duplicated assets.
     */
    public static final double PIVOT_RTOL = 1e-13;

    private Linalg() {
    }

    /**
     * Validates that {@code a} is a finite, square, symmetric matrix.
     *
     * @param a candidate matrix
     * @return the validated matrix (same reference; never mutated by this class)
     * @throws IllegalArgumentException if {@code a} is empty, ragged, non-square,
     *         contains NaN/inf, or the maximum absolute asymmetry exceeds
     *         {@code 1e-10 * (1 + max|a|)}
     */
    public static double[][] checkSquareSymmetric(double[][] a) {
        if (a == null || a.length == 0) {
            throw new IllegalArgumentException("matrix must be non-empty");
        }
        int n = a.length;
        for (double[] row : a) {
            if (row == null || row.length != n) {
                throw new IllegalArgumentException(
                        "matrix must be square 2-D, got a row of length "
                                + (row == null ? "null" : row.length) + " for n=" + n);
            }
        }
        double maxAbs = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double v = a[i][j];
                if (!Double.isFinite(v)) {
                    throw new IllegalArgumentException("matrix contains NaN or infinite entries");
                }
                maxAbs = Math.max(maxAbs, Math.abs(v));
            }
        }
        double scale = 1.0 + maxAbs;
        double asym = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                asym = Math.max(asym, Math.abs(a[i][j] - a[j][i]));
            }
        }
        if (asym > SYM_TOL * scale) {
            throw new IllegalArgumentException(String.format(
                    "matrix is not symmetric (max asymmetry %.3e exceeds tolerance)", asym));
        }
        return a;
    }

    /**
     * Lower-triangular Cholesky factor L with {@code a = L L'}.
     *
     * <p>Pinned algorithm (Cholesky-Banachiewicz): for rows i = 0..n-1 and
     * columns j = 0..i accumulate {@code s = a[i][j] - sum_(k<j) L[i][k] L[j][k]};
     * the diagonal pivot is {@code L[i][i] = sqrt(s)} and off-diagonals are
     * {@code L[i][j] = s / L[j][j]}.
     *
     * @param a symmetric positive-definite matrix
     * @return lower-triangular factor L
     * @throws IllegalArgumentException if the matrix fails symmetry validation,
     *         any diagonal entry is &le; 0, or any pivot satisfies
     *         {@code s <= PIVOT_RTOL * a[i][i]} (not positive definite to working
     *         precision — the pinned detector for singular or indefinite inputs,
     *         e.g. a duplicated asset)
     */
    public static double[][] cholesky(double[][] a) {
        double[][] m = checkSquareSymmetric(a);
        int n = m.length;
        double[][] low = new double[n][n];
        for (int i = 0; i < n; i++) {
            if (m[i][i] <= 0.0) {
                throw new IllegalArgumentException(String.format(
                        "matrix is not positive definite (diagonal entry %.3e <= 0 at row %d)",
                        m[i][i], i));
            }
            for (int j = 0; j <= i; j++) {
                double s = m[i][j];
                for (int k = 0; k < j; k++) {
                    s -= low[i][k] * low[j][k];
                }
                if (i == j) {
                    if (s <= PIVOT_RTOL * m[i][i]) {
                        throw new IllegalArgumentException(String.format(
                                "matrix is not positive definite (Cholesky pivot %.3e <= %g * diag "
                                        + "at row %d); covariance may be singular (e.g. duplicated asset)",
                                s, PIVOT_RTOL, i));
                    }
                    low[i][j] = Math.sqrt(s);
                } else {
                    low[i][j] = s / low[j][j];
                }
            }
        }
        return low;
    }

    /**
     * Solves {@code (L L') x = b} given the lower Cholesky factor.
     *
     * <p>Pinned: forward substitution {@code L y = b} with rows ascending, then
     * back substitution {@code L' x = y} with rows descending.
     *
     * @param low lower-triangular Cholesky factor
     * @param b   right-hand side of length n
     * @return the solution vector x
     * @throws IllegalArgumentException if {@code b} does not have length n
     */
    public static double[] cholSolve(double[][] low, double[] b) {
        int n = low.length;
        if (b == null || b.length != n) {
            throw new IllegalArgumentException(
                    "rhs must have length " + n + ", got " + (b == null ? "null" : b.length));
        }
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            double s = b[i];
            for (int k = 0; k < i; k++) {
                s -= low[i][k] * y[k];
            }
            y[i] = s / low[i][i];
        }
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            double s = y[i];
            for (int k = i + 1; k < n; k++) {
                s -= low[k][i] * x[k];
            }
            x[i] = s / low[i][i];
        }
        return x;
    }

    /**
     * Solves {@code a x = b} for symmetric positive-definite {@code a} via Cholesky.
     *
     * @param a SPD matrix
     * @param b right-hand side
     * @return solution vector
     */
    public static double[] spdSolve(double[][] a, double[] b) {
        return cholSolve(cholesky(a), b);
    }

    /**
     * Inverse of an SPD matrix via column-by-column Cholesky solves.
     *
     * <p>Pinned: factor once, then solve {@code A x_j = e_j} for unit vectors
     * {@code e_j}, j = 0..n-1, stacking solutions as columns. No
     * post-symmetrization is applied.
     *
     * @param a SPD matrix
     * @return the inverse of {@code a}
     */
    public static double[][] spdInverse(double[][] a) {
        double[][] low = cholesky(a);
        int n = low.length;
        double[][] inv = new double[n][n];
        for (int j = 0; j < n; j++) {
            double[] e = new double[n];
            e[j] = 1.0;
            double[] col = cholSolve(low, e);
            for (int i = 0; i < n; i++) {
                inv[i][j] = col[i];
            }
        }
        return inv;
    }

    // ------------------------------------------------------------------
    // Small dense helpers shared by the package (row-major, explicit loops).
    // ------------------------------------------------------------------

    /** Plain left-to-right dot product of two equal-length vectors. */
    static double dot(double[] a, double[] b) {
        double s = 0.0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    /** Matrix-vector product {@code a v} with plain sequential row sums. */
    static double[] matVec(double[][] a, double[] v) {
        double[] out = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            out[i] = dot(a[i], v);
        }
        return out;
    }

    /** Quadratic form {@code w' a w}. */
    static double quadForm(double[] w, double[][] a) {
        return dot(w, matVec(a, w));
    }

    /**
     * Sum of a vector with Neumaier compensation (a Kahan variant that also
     * handles terms larger than the running sum). Used for every T-length
     * reduction (means, covariances, shrinkage dispersion sums, backtest
     * metrics): the Python reference sums with correctly rounded
     * {@code math.fsum}, and the compensated sum agrees with it to a couple
     * of ulps, so no golden value depends on a BLAS or a summation order.
     *
     * @param a terms (an empty array sums to 0)
     * @return the compensated sum
     */
    public static double kahanSum(double[] a) {
        double sum = 0.0;
        double c = 0.0;
        for (double x : a) {
            double t = sum + x;
            if (Math.abs(sum) >= Math.abs(x)) {
                c += (sum - t) + x;
            } else {
                c += (x - t) + sum;
            }
            sum = t;
        }
        return sum + c;
    }

    /** Plain sequential sum of a short vector. */
    static double sum(double[] a) {
        double s = 0.0;
        for (double x : a) {
            s += x;
        }
        return s;
    }
}
