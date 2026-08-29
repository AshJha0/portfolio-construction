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
 * All loops are explicit so the operation order is pinned.
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
    //
    // These reductions replicate, operation for operation, what the Python
    // reference actually executes (numpy 2.x backed by OpenBLAS on an AVX-512
    // host). That matters because the PGD golden values (API_SPEC 3.6) pin the
    // exact iterate of a fixed-point iteration whose stopping point is
    // sensitive to sub-ulp rounding: reproducing them to 1e-8 requires the
    // same floating-point reduction trees, not merely "accurate" ones.
    // ------------------------------------------------------------------

    /**
     * Dot product of two equal-length vectors: sequential fused multiply-add
     * ({@code s = fma(a_i, b_i, s)}), matching BLAS ddot for small n as used
     * by numpy's {@code x @ y}.
     */
    static double dot(double[] a, double[] b) {
        double s = 0.0;
        for (int i = 0; i < a.length; i++) {
            s = Math.fma(a[i], b[i], s);
        }
        return s;
    }

    /**
     * One row-times-vector reduction of the matrix-vector product
     * {@code a @ v}: four FMA accumulator lanes assigned round-robin
     * ({@code lane = j % 4}), combined as {@code (l0+l2) + (l1+l3)} —
     * the dgemv_t microkernel numpy dispatches to for {@code A @ x}.
     */
    static double rowDot(double[] row, double[] v) {
        double l0 = 0.0;
        double l1 = 0.0;
        double l2 = 0.0;
        double l3 = 0.0;
        int n = row.length;
        for (int j = 0; j < n; j += 4) {
            l0 = Math.fma(row[j], v[j], l0);
            if (j + 1 < n) {
                l1 = Math.fma(row[j + 1], v[j + 1], l1);
            }
            if (j + 2 < n) {
                l2 = Math.fma(row[j + 2], v[j + 2], l2);
            }
            if (j + 3 < n) {
                l3 = Math.fma(row[j + 3], v[j + 3], l3);
            }
        }
        return (l0 + l2) + (l1 + l3);
    }

    /** Matrix-vector product {@code a @ v} via {@link #rowDot} per row. */
    static double[] matVec(double[][] a, double[] v) {
        int rows = a.length;
        double[] out = new double[rows];
        for (int i = 0; i < rows; i++) {
            out[i] = rowDot(a[i], v);
        }
        return out;
    }

    /**
     * Vector-matrix product {@code v @ a} (numpy's dgemv_n path). Per output
     * column, rows are consumed in blocks of four with the microkernel's
     * quirky order — the second row's product seeds the block, then rows
     * 0, 2, 3 join by FMA — and block sums are added sequentially:
     *
     * <pre>S_b = fma(a3,v3, fma(a2,v2, fma(a0,v0, a1*v1))); out += S_b</pre>
     */
    static double[] vecMat(double[] v, double[][] a) {
        int n = v.length;
        int cols = a[0].length;
        double[] out = new double[cols];
        int full = n - (n % 4);
        for (int j = 0; j < cols; j++) {
            double acc = 0.0;
            for (int b = 0; b < full; b += 4) {
                double s = v[b + 1] * a[b + 1][j];
                s = Math.fma(v[b], a[b][j], s);
                s = Math.fma(v[b + 2], a[b + 2][j], s);
                s = Math.fma(v[b + 3], a[b + 3][j], s);
                acc = acc + s;
            }
            int rem = n - full;
            if (rem >= 2) {
                acc = acc + Math.fma(v[full + 1], a[full + 1][j], v[full] * a[full][j]);
            }
            if (rem % 2 == 1) {
                acc = acc + v[n - 1] * a[n - 1][j];
            }
            out[j] = acc;
        }
        return out;
    }

    /**
     * Sum of vector entries with numpy's pairwise algorithm: sequential below
     * 8 elements; eight interleaved accumulators combined pairwise up to 128;
     * recursive halving (aligned to multiples of 8) above.
     */
    static double sum(double[] a) {
        return pairwiseSum(a, 0, a.length);
    }

    private static double pairwiseSum(double[] a, int off, int n) {
        if (n < 8) {
            double s = 0.0;
            for (int i = 0; i < n; i++) {
                s += a[off + i];
            }
            return s;
        }
        if (n <= 128) {
            double r0 = a[off];
            double r1 = a[off + 1];
            double r2 = a[off + 2];
            double r3 = a[off + 3];
            double r4 = a[off + 4];
            double r5 = a[off + 5];
            double r6 = a[off + 6];
            double r7 = a[off + 7];
            int i = 8;
            for (; i + 8 <= n; i += 8) {
                r0 += a[off + i];
                r1 += a[off + i + 1];
                r2 += a[off + i + 2];
                r3 += a[off + i + 3];
                r4 += a[off + i + 4];
                r5 += a[off + i + 5];
                r6 += a[off + i + 6];
                r7 += a[off + i + 7];
            }
            double s = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
            for (; i < n; i++) {
                s += a[off + i];
            }
            return s;
        }
        int n2 = n / 2;
        n2 -= n2 % 8;
        return pairwiseSum(a, off, n2) + pairwiseSum(a, off + n2, n - n2);
    }

    /**
     * Quadratic form {@code w' a w}, evaluated exactly as the Python reference
     * writes it: {@code (w @ a) @ w} — vector-matrix product first, then a dot.
     */
    static double quadForm(double[] w, double[][] a) {
        return dot(vecMat(w, a), w);
    }
}
