package com.quant.portopt;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Markowitz mean-variance: closed forms, simplex projection, and the
 * constrained active-set QP.
 *
 * <p>Closed-form results use the classical two-fund theorem. With
 * {@code x = Sigma^-1 mu} and {@code y = Sigma^-1 1} (both computed by the
 * pinned Cholesky solves of {@link Linalg}) define the scalars:
 *
 * <pre>
 *   A = 1' Sigma^-1 mu = sum(x)
 *   B = mu' Sigma^-1 mu = mu . x
 *   C = 1' Sigma^-1 1  = sum(y)
 *   D = B*C - A*A
 * </pre>
 *
 * <p>Minimizing {@code w' Sigma w} subject to {@code 1'w = 1} (and optionally
 * {@code mu'w = m}) with Lagrange multipliers gives every frontier portfolio
 * as {@code w(m) = g + h*m} with {@code g = (B*y - A*x)/D} and
 * {@code h = (C*x - A*y)/D}. Special points: minimum variance
 * {@code w_mv = y/C} (return A/C, variance 1/C); tangency
 * {@code w_tan = Sigma^-1 (mu - rf 1) / (A - rf C)}, defined only when
 * {@code A - rf C > 0} and rejected as ill-conditioned when that difference
 * is within a relative 1e-6 of cancelling.
 *
 * <p>Constrained problems (sum-to-one, optional target return, box bounds)
 * are strictly convex QPs solved by the finite dual active-set method of
 * Goldfarb &amp; Idnani (1983) whose every linear solve is the pinned Cholesky
 * (API_SPEC.md section 3.6). The result is the KKT point of the QP up to
 * rounding, so all ports agree to ~1e-13 and {@code converged} is a statement
 * about the KKT conditions, not about an iterate step size.
 */
public final class Markowitz {

    /** Pinned iteration cap of the dual active-set QP (finite method; the cap is
     *  only hit on numerically inconsistent inputs, reported via converged=false). */
    public static final int QP_MAX_ITER = 1000;
    /** Pinned feasibility tolerance: a bound is violated when n'w - b &lt; -tol*(1+|b|). */
    public static final double QP_FEAS_TOL = 1.0e-13;
    /** Pinned tolerance below which a rounding-level violation of a constraint
     *  dependent on the active set (primal-degenerate vertex) is accepted. */
    public static final double QP_DEGENERATE_TOL = 1.0e-9;
    /** Pinned KKT certificate tolerance (relative) for {@code converged}. */
    public static final double QP_CONV_TOL = 1.0e-9;
    /** Pinned relative conditioning threshold for the tangency denominator. */
    public static final double TANGENCY_COND_RTOL = 1.0e-6;

    private Markowitz() {
    }

    /**
     * Thrown when the tangency portfolio is undefined ({@code A - rf*C <= 0})
     * or ill-conditioned ({@code A - rf*C} within a relative
     * {@link #TANGENCY_COND_RTOL} of cancelling). Extends
     * {@link IllegalArgumentException} so generic handlers still work; the
     * demo's pinned fallback catches exactly this type.
     */
    public static final class TangencyUndefinedException extends IllegalArgumentException {
        private static final long serialVersionUID = 1L;

        /**
         * Creates the exception.
         *
         * @param message descriptive message
         */
        public TangencyUndefinedException(String message) {
            super(message);
        }
    }

    /**
     * Two-fund theorem scalars (see class javadoc).
     *
     * @param a {@code 1' Sigma^-1 mu}
     * @param b {@code mu' Sigma^-1 mu}
     * @param c {@code 1' Sigma^-1 1}
     * @param d {@code b*c - a*a}
     */
    public record TwoFundScalars(double a, double b, double c, double d) {
    }

    /**
     * Result of the active-set QP solvers.
     *
     * @param weights     feasible weights (bounds exact, equalities to rounding)
     * @param iterations  dual active-set iterations (constraint adds/drops)
     * @param converged   KKT certificate holds at {@code weights} (see QP_CONV_TOL)
     * @param kktResidual largest KKT violation: equality residuals, bound
     *                    violations and the stationarity residual |S w - N u|
     */
    public record QpResult(double[] weights, int iterations, boolean converged, double kktResidual) {
    }

    /**
     * A frontier sweep: {@code points} targets with their weights and variances.
     *
     * @param targets   target returns (pinned grid, last point exactly hi)
     * @param weights   one weight vector per target (points x n)
     * @param variances portfolio variance {@code w' Sigma w} per target
     */
    public record Frontier(double[] targets, double[][] weights, double[] variances) {
    }

    private static void validateMuSigma(double[] mu, double[][] sigma) {
        if (sigma == null || sigma.length == 0 || sigma[0] == null
                || sigma.length != sigma[0].length) {
            throw new IllegalArgumentException("sigma must be a non-empty square matrix");
        }
        for (double[] row : sigma) {
            if (row == null || row.length != sigma.length) {
                throw new IllegalArgumentException("sigma must be square");
            }
        }
        if (mu == null || mu.length != sigma.length) {
            throw new IllegalArgumentException("mu length "
                    + (mu == null ? 0 : mu.length) + " does not match sigma dimension "
                    + sigma.length);
        }
        for (double v : mu) {
            if (!Double.isFinite(v)) {
                throw new IllegalArgumentException("mu contains NaN or infinite entries");
            }
        }
    }

    private static void validate1x1(double[][] sigma) {
        if (sigma[0].length != 1 || sigma[0][0] <= 0 || !Double.isFinite(sigma[0][0])) {
            throw new IllegalArgumentException("1x1 covariance must be positive and finite");
        }
    }

    /**
     * Computes the two-fund scalars A, B, C, D on {@code (mu, sigma)}.
     *
     * @param mu    expected-return vector
     * @param sigma SPD covariance
     * @return the scalars
     * @throws IllegalArgumentException if sigma is not symmetric positive definite
     */
    public static TwoFundScalars twoFundScalars(double[] mu, double[][] sigma) {
        validateMuSigma(mu, sigma);
        double[][] low = Linalg.cholesky(sigma);
        double[] x = Linalg.cholSolve(low, mu);
        double[] ones = new double[mu.length];
        Arrays.fill(ones, 1.0);
        double[] y = Linalg.cholSolve(low, ones);
        double a = Linalg.sum(x);
        double b = Linalg.dot(mu, x);
        double c = Linalg.sum(y);
        return new TwoFundScalars(a, b, c, b * c - a * a);
    }

    /**
     * Unconstrained (sum-to-one) minimum-variance weights {@code Sigma^-1 1 / C}.
     * For n = 1 returns the trivial {1.0}.
     *
     * @param sigma SPD covariance
     * @return weight vector summing to 1
     */
    public static double[] minVarianceWeights(double[][] sigma) {
        if (sigma == null || sigma.length == 0) {
            throw new IllegalArgumentException("sigma must be a non-empty square matrix");
        }
        if (sigma.length == 1) {
            validate1x1(sigma);
            return new double[] {1.0};
        }
        double[] ones = new double[sigma.length];
        Arrays.fill(ones, 1.0);
        double[] y = Linalg.cholSolve(Linalg.cholesky(sigma), ones);
        double c = Linalg.sum(y);
        double[] w = new double[y.length];
        for (int i = 0; i < y.length; i++) {
            w[i] = y[i] / c;
        }
        return w;
    }

    /**
     * Tangency (max-Sharpe) weights {@code Sigma^-1 (mu - rf 1) / (A - rf C)}.
     *
     * @param mu    expected returns
     * @param sigma SPD covariance
     * @param rf    risk-free rate (finite)
     * @return tangency weights summing to 1
     * @throws TangencyUndefinedException if
     *         {@code A - rf*C <= TANGENCY_COND_RTOL * max(|A|, |rf|*C)}: the
     *         tangency portfolio is undefined when the risk-free rate is at or
     *         above the minimum-variance return A/C (in particular when rf
     *         exceeds every expected return) and numerically meaningless when
     *         rf sits within a relative 1e-6 of A/C (weights scale like
     *         1/(A - rf C) and reach gross leverage of 1e6+)
     * @throws IllegalArgumentException on malformed inputs (non-finite rf, ...)
     */
    public static double[] tangencyWeights(double[] mu, double[][] sigma, double rf) {
        validateMuSigma(mu, sigma);
        if (!Double.isFinite(rf)) {
            throw new IllegalArgumentException("rf must be finite");
        }
        if (sigma.length == 1) {
            validate1x1(sigma);
            if (mu[0] <= rf) {
                throw new TangencyUndefinedException(
                        "tangency undefined: risk-free rate >= expected return of the single asset");
            }
            return new double[] {1.0};
        }
        double[][] low = Linalg.cholesky(sigma);
        double[] excess = new double[mu.length];
        for (int i = 0; i < mu.length; i++) {
            excess[i] = mu[i] - rf;
        }
        double[] z = Linalg.cholSolve(low, excess);
        double denom = Linalg.sum(z); // = A - rf*C
        if (denom <= 0.0) {
            throw new TangencyUndefinedException(
                    "tangency undefined: A - rf*C <= 0 (risk-free rate is at or above "
                            + "the minimum-variance portfolio return)");
        }
        // Conditioning guard: A - rf*C is a difference of two O(A) numbers.
        double[] ones = new double[mu.length];
        Arrays.fill(ones, 1.0);
        double a = Linalg.sum(Linalg.cholSolve(low, mu));
        double c = Linalg.sum(Linalg.cholSolve(low, ones));
        double scale = Math.max(Math.abs(a), Math.abs(rf) * c);
        if (denom <= TANGENCY_COND_RTOL * scale) {
            throw new TangencyUndefinedException(String.format(
                    "tangency ill-conditioned: A - rf*C = %.3e is within a relative %g of "
                            + "cancelling (rf within %g of the minimum-variance return %.6g); "
                            + "weights would carry gross leverage of order 1e6 or more",
                    denom, TANGENCY_COND_RTOL, TANGENCY_COND_RTOL, a / c));
        }
        double[] w = new double[z.length];
        for (int i = 0; i < z.length; i++) {
            w[i] = z[i] / denom;
        }
        return w;
    }

    /**
     * Gross exposure {@code sum_i |w_i|} (1.0 for any long-only portfolio); a
     * desk should cap it before unconstrained weights reach an OMS.
     *
     * @param w weights (non-empty, finite)
     * @return the gross leverage
     */
    public static double grossLeverage(double[] w) {
        if (w == null || w.length == 0) {
            throw new IllegalArgumentException("weights must be a non-empty finite vector");
        }
        double[] a = new double[w.length];
        for (int i = 0; i < w.length; i++) {
            if (!Double.isFinite(w[i])) {
                throw new IllegalArgumentException("weights must be a non-empty finite vector");
            }
            a[i] = Math.abs(w[i]);
        }
        return Linalg.kahanSum(a);
    }

    /** Frontier generators g, h and the min-variance return A/C (n &ge; 2). */
    private record TwoFundGh(double[] g, double[] h, double mvRet) {
    }

    private static TwoFundGh twoFundGh(double[] mu, double[][] sigma) {
        int n = sigma.length;
        double[][] low = Linalg.cholesky(sigma);
        double[] x = Linalg.cholSolve(low, mu);
        double[] ones = new double[n];
        Arrays.fill(ones, 1.0);
        double[] y = Linalg.cholSolve(low, ones);
        double a = Linalg.sum(x);
        double b = Linalg.dot(mu, x);
        double c = Linalg.sum(y);
        double d = b * c - a * a;
        // Pinned relative degeneracy test: by Cauchy-Schwarz D >= 0 with equality
        // iff mu is proportional to ones; rounding turns the exact zero into a
        // tiny residual, so compare against B*C (> 0).
        if (d <= 1e-12 * b * c) {
            throw new IllegalArgumentException(String.format(
                    "degenerate frontier: D = B*C - A^2 = %.3e <= 1e-12*B*C "
                            + "(expected returns proportional to ones?)", d));
        }
        double[] g = new double[n];
        double[] h = new double[n];
        for (int i = 0; i < n; i++) {
            g[i] = (b * y[i] - a * x[i]) / d;
            h[i] = (c * x[i] - a * y[i]) / d;
        }
        return new TwoFundGh(g, h, a / c);
    }

    /**
     * Efficient-frontier weights for a target return via the two-fund theorem:
     * {@code w(m) = g + h*m}.
     *
     * @param mu     expected returns
     * @param sigma  SPD covariance
     * @param target target portfolio return m
     * @return frontier weights at the target
     * @throws IllegalArgumentException if n = 1 and the target differs from the
     *         single asset's mean; if {@code D <= 1e-12*B*C} (degenerate
     *         frontier, mu proportional to ones); or if the target lies below
     *         the minimum-variance return A/C (includes negative targets)
     */
    public static double[] frontierWeights(double[] mu, double[][] sigma, double target) {
        validateMuSigma(mu, sigma);
        if (!Double.isFinite(target)) {
            throw new IllegalArgumentException("target return must be finite");
        }
        int n = sigma.length;
        if (n == 1) {
            validate1x1(sigma);
            if (Math.abs(target - mu[0]) > 1e-12 * (1.0 + Math.abs(mu[0]))) {
                throw new IllegalArgumentException(
                        "n=1: target return must equal the single asset's expected return");
            }
            return new double[] {1.0};
        }
        TwoFundGh gh = twoFundGh(mu, sigma);
        if (target < gh.mvRet() - 1e-12 * (1.0 + Math.abs(gh.mvRet()))) {
            throw new IllegalArgumentException(String.format(
                    "target return %.6g lies below the minimum-variance return %.6g; "
                            + "the efficient frontier is defined for targets >= A/C",
                    target, gh.mvRet()));
        }
        double[] w = new double[n];
        for (int i = 0; i < n; i++) {
            w[i] = gh.g()[i] + gh.h()[i] * target;
        }
        return w;
    }

    /**
     * Closed-form frontier variance at target m: {@code (C m^2 - 2 A m + B) / D}.
     *
     * @param sc     two-fund scalars
     * @param target target return
     * @return frontier variance
     */
    public static double frontierVariance(TwoFundScalars sc, double target) {
        return (sc.c() * target * target - 2.0 * sc.a() * target + sc.b()) / sc.d();
    }

    private static double[] frontierGrid(double lo, double hi, int nPoints, String what) {
        if (hi <= lo + 1e-12 * (1.0 + Math.abs(lo))) {
            throw new IllegalArgumentException(String.format(
                    "frontier grid empty: %s (%.6g) does not exceed the minimum-variance "
                            + "return (%.6g); supply targets explicitly via frontierWeights / "
                            + "constrainedFrontierWeights", what, hi, lo));
        }
        double[] targets = new double[nPoints];
        for (int k = 0; k < nPoints; k++) {
            targets[k] = lo + k * (hi - lo) / (nPoints - 1);
        }
        targets[nPoints - 1] = hi; // pinned: the last grid point is exactly hi
        return targets;
    }

    /**
     * Unconstrained frontier sweep on the pinned grid: {@code nPoints} equally
     * spaced targets from the minimum-variance return A/C to {@code max(mu)}
     * inclusive; the two-fund generators are computed once and reused.
     *
     * @param mu      expected returns
     * @param sigma   SPD covariance
     * @param nPoints number of grid points (&ge; 2)
     * @return targets, weights and variances
     * @throws IllegalArgumentException if {@code max(mu) <= A/C} (frontier grid empty)
     */
    public static Frontier efficientFrontier(double[] mu, double[][] sigma, int nPoints) {
        validateMuSigma(mu, sigma);
        if (nPoints < 2) {
            throw new IllegalArgumentException("n_points must be >= 2");
        }
        if (sigma.length == 1) {
            throw new IllegalArgumentException("frontier sweep requires at least 2 assets");
        }
        TwoFundGh gh = twoFundGh(mu, sigma);
        double hi = mu[0];
        for (double v : mu) {
            hi = Math.max(hi, v);
        }
        double[] targets = frontierGrid(gh.mvRet(), hi, nPoints, "max(mu)");
        double[][] weights = new double[nPoints][];
        double[] variances = new double[nPoints];
        int n = mu.length;
        for (int k = 0; k < nPoints; k++) {
            double[] w = new double[n];
            for (int i = 0; i < n; i++) {
                w[i] = gh.g()[i] + gh.h()[i] * targets[k];
            }
            weights[k] = w;
            variances[k] = Linalg.quadForm(w, sigma);
        }
        return new Frontier(targets, weights, variances);
    }

    // ------------------------------------------------------------------ bounds

    /** Broadcast/validate bounds (API_SPEC 3.5 feasibility gate): returns {lo, hi}. */
    private static double[][] broadcastBounds(int n, double[] lb, double[] ub, double total) {
        double[] lo = new double[n];
        double[] hi = new double[n];
        for (int i = 0; i < n; i++) {
            lo[i] = (lb == null) ? 0.0 : lb[boundIndex(lb, i, n)];
            hi[i] = (ub == null) ? Double.POSITIVE_INFINITY : ub[boundIndex(ub, i, n)];
            if (Double.isNaN(lo[i]) || Double.isNaN(hi[i])) {
                throw new IllegalArgumentException("bounds contain NaN");
            }
            if (!Double.isFinite(lo[i])) {
                throw new IllegalArgumentException("lower bounds must be finite");
            }
            if (hi[i] == Double.NEGATIVE_INFINITY) {
                throw new IllegalArgumentException("upper bounds must be > -inf");
            }
            if (lo[i] > hi[i]) {
                throw new IllegalArgumentException("infeasible bounds: some lb_i > ub_i");
            }
        }
        double loSum = Linalg.kahanSum(lo);
        double hiSum = Linalg.kahanSum(hi);
        double feasTol = 1e-12 * (1.0 + Math.abs(total));
        if (loSum > total + feasTol) {
            throw new IllegalArgumentException(String.format(
                    "infeasible bounds: sum(lb) = %.6g exceeds total %.6g", loSum, total));
        }
        if (hiSum < total - feasTol) {
            throw new IllegalArgumentException(String.format(
                    "infeasible bounds: sum(ub) = %.6g is below total %.6g", hiSum, total));
        }
        return new double[][] {lo, hi};
    }

    private static int boundIndex(double[] bound, int i, int n) {
        if (bound.length == n) {
            return i;
        }
        if (bound.length == 1) {
            return 0; // scalar bound broadcast to every coordinate
        }
        throw new IllegalArgumentException(
                "bounds must have length 1 or " + n + ", got " + bound.length);
    }

    // -------------------------------------------------------------- projection

    /**
     * Exact Euclidean projection onto {@code {w : sum w = total, lb <= w <= ub}}
     * with default bounds lb = 0, ub = +inf (long-only simplex).
     *
     * @param v     point to project
     * @param total required coordinate sum
     * @return the projection
     */
    public static double[] projectSimplexBox(double[] v, double total) {
        return projectSimplexBox(v, total, null, null);
    }

    /**
     * Exact Euclidean projection onto {@code {w : sum w = total, lb <= w <= ub}}.
     *
     * <p>Pinned sort-based algorithm (API_SPEC section 3.5). The KKT solution is
     * {@code w_i = clip(v_i - t, lb_i, ub_i)} for the unique shift t making the
     * sum equal {@code total}; {@code g(t) = sum_i clip(v_i - t, lb_i, ub_i)} is
     * piecewise linear and non-increasing, with kinks at {@code v_i - lb_i} and,
     * for finite bounds, {@code v_i - ub_i}. The algorithm sorts the breakpoints
     * ascending, scans them to bracket {@code g(t) = total}, and finishes with
     * one exact linear interpolation on the bracketing segment — a finite,
     * deterministic procedure with no iteration tolerance. With lb = 0,
     * ub = +inf it reproduces the classic Duchi et al. sorted simplex projection.
     *
     * @param v     point to project (finite entries)
     * @param total required coordinate sum (finite)
     * @param lb    per-coordinate lower bounds (finite), or null for all 0; length 1 broadcasts
     * @param ub    per-coordinate upper bounds, or null for all +inf; length 1 broadcasts
     * @return the projection
     * @throws IllegalArgumentException if bounds are infeasible
     *         ({@code sum(lb) > total}, {@code sum(ub) < total}, any
     *         {@code lb_i > ub_i}) or inputs are invalid (empty v, NaN/inf in v,
     *         NaN in bounds, infinite lb)
     */
    public static double[] projectSimplexBox(double[] v, double total, double[] lb, double[] ub) {
        if (v == null || v.length == 0) {
            throw new IllegalArgumentException("cannot project an empty vector");
        }
        int n = v.length;
        for (double x : v) {
            if (!Double.isFinite(x)) {
                throw new IllegalArgumentException("projection input contains NaN or infinite entries");
            }
        }
        if (!Double.isFinite(total)) {
            throw new IllegalArgumentException("total must be finite");
        }
        double[][] bounds = broadcastBounds(n, lb, ub, total);
        double[] lo = bounds[0];
        double[] hi = bounds[1];

        // Breakpoints: all v_i - lb_i, plus v_i - ub_i for finite ub_i.
        int nFiniteUb = 0;
        for (int i = 0; i < n; i++) {
            if (Double.isFinite(hi[i])) {
                nFiniteUb++;
            }
        }
        double[] bps = new double[n + nFiniteUb];
        int idx = 0;
        for (int i = 0; i < n; i++) {
            bps[idx++] = v[i] - lo[i];
        }
        for (int i = 0; i < n; i++) {
            if (Double.isFinite(hi[i])) {
                bps[idx++] = v[i] - hi[i];
            }
        }
        Arrays.sort(bps);

        double tFirst = bps[0];
        double gFirst = gOfT(v, lo, hi, tFirst);
        if (gFirst < total) {
            // t* lies left of every breakpoint; there the unclipped (infinite-ub)
            // coordinates give slope -k with k = number of infinite upper bounds.
            int k = n - nFiniteUb;
            if (k == 0) {
                // sum(ub) >= total guarantees gFirst >= total up to rounding; snap.
                return clipShift(v, lo, hi, tFirst);
            }
            return clipShift(v, lo, hi, tFirst - (total - gFirst) / k);
        }

        double tPrev = tFirst;
        double gPrev = gFirst;
        for (int c = 1; c < bps.length; c++) {
            double tCur = bps[c];
            double gCur = gOfT(v, lo, hi, tCur);
            if (gCur <= total) {
                double tStar;
                if (gPrev == gCur) {
                    tStar = tCur;
                } else {
                    tStar = tPrev + (gPrev - total) * (tCur - tPrev) / (gPrev - gCur);
                }
                return clipShift(v, lo, hi, tStar);
            }
            tPrev = tCur;
            gPrev = gCur;
        }
        // Right of every breakpoint g(t) = sum(lb) <= total; feasibility means equality.
        return clipShift(v, lo, hi, bps[bps.length - 1]);
    }

    private static double gOfT(double[] v, double[] lo, double[] hi, double t) {
        return Linalg.kahanSum(clipShift(v, lo, hi, t));
    }

    private static double[] clipShift(double[] v, double[] lo, double[] hi, double t) {
        double[] w = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            w[i] = Math.min(Math.max(v[i] - t, lo[i]), hi[i]);
        }
        return w;
    }

    // ------------------------------------------------------- attainable range

    /**
     * Vertex of {sum w = total, lb &le; w &le; ub} maximizing (descending) or
     * minimizing (ascending) mu'w: fill from lb in mu order (ties by index).
     */
    private static double[] greedyExtreme(double[] mu, double[] lo, double[] hi, double total,
            boolean descending) {
        int n = mu.length;
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> {
            int cmp = descending ? Double.compare(mu[b], mu[a]) : Double.compare(mu[a], mu[b]);
            return cmp != 0 ? cmp : Integer.compare(a, b);
        });
        double[] w = lo.clone();
        double remaining = total - Linalg.kahanSum(lo);
        if (remaining < 0.0) {
            remaining = 0.0;
        }
        for (int i : order) {
            if (remaining <= 0.0) {
                break;
            }
            double room = hi[i] - lo[i];
            double add = room > remaining ? remaining : room;
            w[i] = lo[i] + add;
            remaining -= add;
        }
        return w;
    }

    /**
     * Range {@code [r_min, r_max]} of {@code mu'w} over
     * {@code {sum w = 1, lb <= w <= ub}}; both extremes are vertices found by
     * the pinned greedy fill (assets in ascending / descending mu order, ties
     * by index).
     *
     * @param mu expected returns (non-empty, finite)
     * @param lb lower bounds (null = 0; length 1 broadcasts)
     * @param ub upper bounds (null = +inf; length 1 broadcasts)
     * @return {r_min, r_max}
     */
    public static double[] attainableReturnRange(double[] mu, double[] lb, double[] ub) {
        if (mu == null || mu.length == 0) {
            throw new IllegalArgumentException("mu must be a non-empty finite vector");
        }
        for (double v : mu) {
            if (!Double.isFinite(v)) {
                throw new IllegalArgumentException("mu must be a non-empty finite vector");
            }
        }
        double[][] bounds = broadcastBounds(mu.length, lb, ub, 1.0);
        double[] wMin = greedyExtreme(mu, bounds[0], bounds[1], 1.0, false);
        double[] wMax = greedyExtreme(mu, bounds[0], bounds[1], 1.0, true);
        return new double[] {Linalg.dot(mu, wMin), Linalg.dot(mu, wMax)};
    }

    // --------------------------------------------------------------------- QP

    private enum Kind { EQUALITY, LOWER, UPPER, FIXED }

    private record Constraint(Kind kind, int idx, double rhs, double[] normal) {
    }

    private static double[] unit(int n, int i, double sign) {
        double[] e = new double[n];
        e[i] = sign;
        return e;
    }

    /**
     * Goldfarb-Idnani dual active-set method for min 1/2 w'Sw s.t. aEq w = bEq,
     * lo &le; w &le; hi (API_SPEC 3.6). Constraints are n_c'w &ge; b_c: the
     * equalities (rows of aEq, then every fixed coordinate lo_i == hi_i; added
     * first, multipliers unrestricted), then the lower bounds and the finite
     * upper bounds of the non-fixed coordinates.
     */
    private static QpResult dualActiveSetQp(double[][] s, double[][] aEq, double[] bEq,
            double[] lo, double[] hi, int maxIter) {
        int n = s.length;
        double[][] low = Linalg.cholesky(s);
        List<Constraint> cons = new ArrayList<>();
        for (int j = 0; j < aEq.length; j++) {
            cons.add(new Constraint(Kind.EQUALITY, j, bEq[j], aEq[j].clone()));
        }
        for (int i = 0; i < n; i++) {
            if (lo[i] == hi[i]) {
                cons.add(new Constraint(Kind.FIXED, i, lo[i], unit(n, i, 1.0)));
            }
        }
        int meq = cons.size();
        for (int i = 0; i < n; i++) {
            if (lo[i] != hi[i]) {
                cons.add(new Constraint(Kind.LOWER, i, lo[i], unit(n, i, 1.0)));
            }
        }
        for (int i = 0; i < n; i++) {
            if (lo[i] != hi[i] && Double.isFinite(hi[i])) {
                cons.add(new Constraint(Kind.UPPER, i, -hi[i], unit(n, i, -1.0)));
            }
        }
        int nCons = cons.size();

        double[] w = new double[n];
        List<Integer> active = new ArrayList<>();
        List<Double> u = new ArrayList<>();
        boolean[] skipped = new boolean[nCons];
        int iterations = 0;
        int p = -1;
        double uP = 0.0;
        while (iterations < maxIter) {
            iterations++;
            if (p < 0) {
                // (1) pick the next constraint: equalities first, then the most
                // violated inequality (ties -> lowest index).
                if (active.size() < meq) {
                    p = active.size();
                } else {
                    double worst = 0.0;
                    for (int c = meq; c < nCons; c++) {
                        if (skipped[c] || active.contains(c)) {
                            continue;
                        }
                        Constraint con = cons.get(c);
                        double viol = Linalg.dot(con.normal(), w) - con.rhs();
                        if (viol < -QP_FEAS_TOL * (1.0 + Math.abs(con.rhs())) && viol < worst) {
                            worst = viol;
                            p = c;
                        }
                    }
                    if (p < 0) {
                        break; // no violated constraint: dual method finished
                    }
                }
                uP = 0.0;
            }
            Constraint cp = cons.get(p);
            // (2) primal / dual step directions for constraint p.
            int q = active.size();
            double[][] ys = new double[q][];
            for (int i = 0; i < q; i++) {
                ys[i] = Linalg.cholSolve(low, cons.get(active.get(i)).normal());
            }
            double[] v = Linalg.cholSolve(low, cp.normal());
            double[] r = new double[q];
            if (q > 0) {
                double[][] gram = new double[q][q];
                for (int i = 0; i < q; i++) {
                    for (int j = i; j < q; j++) {
                        gram[i][j] = Linalg.dot(cons.get(active.get(i)).normal(), ys[j]);
                        gram[j][i] = gram[i][j];
                    }
                }
                double[] nv = new double[q];
                for (int i = 0; i < q; i++) {
                    nv[i] = Linalg.dot(cons.get(active.get(i)).normal(), v);
                }
                r = Linalg.cholSolve(Linalg.cholesky(gram), nv);
            }
            double[] z = v.clone();
            for (int i = 0; i < q; i++) {
                for (int k = 0; k < n; k++) {
                    z[k] -= r[i] * ys[i][k];
                }
            }
            double zn = Linalg.dot(z, cp.normal());
            double vn = Linalg.dot(v, cp.normal());
            double sP = Linalg.dot(cp.normal(), w) - cp.rhs();
            // (3) step lengths: t1 keeps inequality multipliers >= 0, t2 satisfies p.
            double t1 = Double.POSITIVE_INFINITY;
            int jBlock = -1;
            for (int i = 0; i < q; i++) {
                if (active.get(i) >= meq && r[i] > 0.0) {
                    double cand = u.get(i) / r[i];
                    if (cand < t1) {
                        t1 = cand;
                        jBlock = i;
                    }
                }
            }
            double t2 = (zn > 1e-10 * vn) ? -sP / zn : Double.POSITIVE_INFINITY;
            double t = Math.min(t1, t2);
            if (!Double.isFinite(t)) {
                // n_p is a combination of the active normals and no multiplier
                // blocks: accept a rounding-level violation at a primal-degenerate
                // vertex as satisfied, else give up (reported via the certificate).
                if (Math.abs(sP) <= QP_DEGENERATE_TOL * (1.0 + Math.abs(cp.rhs()))) {
                    skipped[p] = true;
                    p = -1;
                    continue;
                }
                break;
            }
            if (!Double.isFinite(t2)) {
                // Dependent normal: dual step only, drop the blocking constraint.
                for (int i = 0; i < q; i++) {
                    u.set(i, u.get(i) - t * r[i]);
                }
                uP += t;
                active.remove(jBlock);
                u.remove(jBlock);
                continue;
            }
            for (int k = 0; k < n; k++) {
                w[k] += t * z[k];
            }
            for (int i = 0; i < q; i++) {
                u.set(i, u.get(i) - t * r[i]);
            }
            uP += t;
            if (t == t2) {
                active.add(p);
                u.add(uP);
                p = -1;
            } else {
                active.remove(jBlock);
                u.remove(jBlock);
            }
        }
        // Polish: minimizer on the final active set in closed form,
        // w = S^-1 N u with (N' S^-1 N) u = b_active, plus one step of iterative
        // refinement on the equality residual.
        int q = active.size();
        double[] uArr = new double[q];
        if (q > 0) {
            double[][] ys = new double[q][];
            for (int i = 0; i < q; i++) {
                ys[i] = Linalg.cholSolve(low, cons.get(active.get(i)).normal());
            }
            double[][] gram = new double[q][q];
            for (int i = 0; i < q; i++) {
                for (int j = i; j < q; j++) {
                    gram[i][j] = Linalg.dot(cons.get(active.get(i)).normal(), ys[j]);
                    gram[j][i] = gram[i][j];
                }
            }
            double[][] lowG = Linalg.cholesky(gram);
            double[] bAct = new double[q];
            for (int i = 0; i < q; i++) {
                bAct[i] = cons.get(active.get(i)).rhs();
            }
            double[] u0 = Linalg.cholSolve(lowG, bAct);
            w = new double[n];
            for (int i = 0; i < q; i++) {
                for (int k = 0; k < n; k++) {
                    w[k] += u0[i] * ys[i][k];
                }
            }
            double[] resid = new double[q];
            for (int i = 0; i < q; i++) {
                resid[i] = bAct[i] - Linalg.dot(cons.get(active.get(i)).normal(), w);
            }
            double[] du = Linalg.cholSolve(lowG, resid);
            for (int i = 0; i < q; i++) {
                for (int k = 0; k < n; k++) {
                    w[k] += du[i] * ys[i][k];
                }
                uArr[i] = u0[i] + du[i];
            }
        }
        for (int c : active) {
            Constraint con = cons.get(c);
            if (con.kind() == Kind.LOWER || con.kind() == Kind.FIXED) {
                w[con.idx()] = lo[con.idx()];
            } else if (con.kind() == Kind.UPPER) {
                w[con.idx()] = hi[con.idx()];
            }
        }
        // KKT certificate on the polished point.
        double feas = 0.0;
        double rhsScale = 0.0;
        for (int c = 0; c < nCons; c++) {
            Constraint con = cons.get(c);
            double viol = Linalg.dot(con.normal(), w) - con.rhs();
            rhsScale = Math.max(rhsScale, Math.abs(con.rhs()));
            feas = Math.max(feas, c < meq ? Math.abs(viol) : -viol);
        }
        double[] stat = Linalg.matVec(s, w);
        double statScale = 0.0;
        for (double v : stat) {
            statScale = Math.max(statScale, Math.abs(v));
        }
        for (int i = 0; i < q; i++) {
            Constraint con = cons.get(active.get(i));
            double nmax = 0.0;
            for (int k = 0; k < n; k++) {
                stat[k] -= uArr[i] * con.normal()[k];
                nmax = Math.max(nmax, Math.abs(con.normal()[k]));
            }
            statScale += Math.abs(uArr[i]) * nmax;
        }
        double statRes = 0.0;
        for (double v : stat) {
            statRes = Math.max(statRes, Math.abs(v));
        }
        boolean converged = feas <= QP_CONV_TOL * (1.0 + rhsScale)
                && statRes <= QP_CONV_TOL * (1.0 + statScale);
        for (int k = 0; k < n; k++) {
            w[k] = Math.min(Math.max(w[k], lo[k]), hi[k]);
        }
        return new QpResult(w, iterations, converged, Math.max(feas, statRes));
    }

    /** Shared QP setup: square + SPD gate, bound validation, iteration cap. */
    private static double[][] qpSetup(double[][] sigma, double[] lb, double[] ub, int maxIter) {
        if (sigma == null || sigma.length == 0) {
            throw new IllegalArgumentException("sigma must be a non-empty square matrix");
        }
        Linalg.cholesky(sigma); // SPD/validity gate
        if (maxIter < 1) {
            throw new IllegalArgumentException("max_iter must be >= 1");
        }
        return broadcastBounds(sigma.length, lb, ub, 1.0);
    }

    private static double[] requireConverged(QpResult res, String what) {
        if (!res.converged()) {
            throw new IllegalArgumentException(String.format(
                    "%s: active-set QP did not converge within %d iterations (KKT residual %.3e); "
                            + "the input is degenerate — use the *Solve variant to inspect the iterate",
                    what, res.iterations(), res.kktResidual()));
        }
        return res.weights();
    }

    private static double[] ones(int n) {
        double[] o = new double[n];
        Arrays.fill(o, 1.0);
        return o;
    }

    /**
     * Constrained minimum variance {@code min 1/2 w' Sigma w} over
     * {@code {sum w = 1, lb <= w <= ub}} (defaults: long-only), solved by the
     * pinned dual active-set QP. Strictly convex: the minimizer is unique.
     *
     * @param sigma   SPD covariance
     * @param lb      lower bounds (null = 0; length 1 broadcasts)
     * @param ub      upper bounds (null = +inf; length 1 broadcasts)
     * @param maxIter iteration cap (pinned default {@link #QP_MAX_ITER})
     * @return weights, iterations, convergence certificate and KKT residual
     */
    public static QpResult minVarianceLongOnlySolve(double[][] sigma, double[] lb, double[] ub,
            int maxIter) {
        double[][] bounds = qpSetup(sigma, lb, ub, maxIter);
        return dualActiveSetQp(sigma, new double[][] {ones(sigma.length)}, new double[] {1.0},
                bounds[0], bounds[1], maxIter);
    }

    /**
     * Constrained minimum variance over the long-only simplex (lb = 0, ub = +inf).
     *
     * @param sigma SPD covariance (validated via Cholesky)
     * @return the unique constrained minimum-variance weights
     * @throws IllegalArgumentException if the solver did not certify convergence
     */
    public static double[] minVarianceLongOnly(double[][] sigma) {
        return minVarianceLongOnly(sigma, null, null);
    }

    /**
     * Weights of {@link #minVarianceLongOnlySolve} with the pinned iteration cap.
     *
     * @param sigma SPD covariance
     * @param lb    lower bounds (null = 0; length 1 broadcasts)
     * @param ub    upper bounds (null = +inf; length 1 broadcasts)
     * @return constrained minimum-variance weights
     * @throws IllegalArgumentException if the solver did not certify convergence
     */
    public static double[] minVarianceLongOnly(double[][] sigma, double[] lb, double[] ub) {
        return requireConverged(minVarianceLongOnlySolve(sigma, lb, ub, QP_MAX_ITER),
                "minVarianceLongOnly");
    }

    private static QpResult frontierPoint(double[] mu, double[][] sigma, double tgt, double[] lo,
            double[] hi, double[] wGmv, double rGmv, double rMin, double rMax, int maxIter) {
        double tol = 1e-12 * (1.0 + Math.abs(tgt));
        if (tgt < rGmv - tol) {
            throw new IllegalArgumentException(String.format(
                    "target return %.6g lies below the constrained minimum-variance return %.6g; "
                            + "the constrained efficient frontier is defined for targets in [%.6g, %.6g]",
                    tgt, rGmv, rGmv, rMax));
        }
        if (tgt > rMax + tol) {
            throw new IllegalArgumentException(String.format(
                    "target return %.6g is not attainable under the bounds (attainable range "
                            + "[%.6g, %.6g])", tgt, rMin, rMax));
        }
        if (Math.abs(tgt - rGmv) <= tol || rMax - rMin <= tol) {
            return new QpResult(wGmv.clone(), 0, true, 0.0);
        }
        int n = mu.length;
        if (Math.abs(tgt - rMax) <= tol) {
            // Maximum-return endpoint: reduce the degenerate face analytically
            // (assets above the greedy cut level at their caps, below it at their
            // floors, tied assets free) and solve the sum-only min-variance QP.
            double[] wMax = greedyExtreme(mu, lo, hi, 1.0, true);
            boolean any = false;
            double cut = 0.0;
            for (int i = 0; i < n; i++) {
                if (wMax[i] > lo[i]) {
                    cut = any ? Math.min(cut, mu[i]) : mu[i];
                    any = true;
                }
            }
            if (!any) {
                return new QpResult(wGmv.clone(), 0, true, 0.0);
            }
            double[] lo2 = lo.clone();
            double[] hi2 = hi.clone();
            for (int i = 0; i < n; i++) {
                if (mu[i] > cut) {
                    lo2[i] = hi[i];
                } else if (mu[i] < cut) {
                    hi2[i] = lo[i];
                }
            }
            QpResult res = dualActiveSetQp(sigma, new double[][] {ones(n)}, new double[] {1.0},
                    lo2, hi2, maxIter);
            double gap = Math.abs(Linalg.dot(mu, res.weights()) - tgt);
            return new QpResult(res.weights(), res.iterations(), res.converged(),
                    Math.max(res.kktResidual(), gap));
        }
        return dualActiveSetQp(sigma, new double[][] {ones(n), mu.clone()},
                new double[] {1.0, tgt}, lo, hi, maxIter);
    }

    /**
     * Constrained frontier point: {@code min 1/2 w' Sigma w} subject to
     * {@code sum w = 1}, {@code mu'w = target} and {@code lb <= w <= ub}. The
     * target is met exactly (to rounding) — there is no penalty term.
     *
     * @param mu      expected returns
     * @param sigma   SPD covariance
     * @param target  target portfolio return
     * @param lb      lower bounds (null = 0; length 1 broadcasts)
     * @param ub      upper bounds (null = +inf; length 1 broadcasts)
     * @param maxIter iteration cap (pinned default {@link #QP_MAX_ITER})
     * @return weights, iterations, convergence certificate and KKT residual
     * @throws IllegalArgumentException if the target lies below the constrained
     *         minimum-variance return or above the maximum attainable return
     */
    public static QpResult constrainedFrontierSolve(double[] mu, double[][] sigma, double target,
            double[] lb, double[] ub, int maxIter) {
        validateMuSigma(mu, sigma);
        double[][] bounds = qpSetup(sigma, lb, ub, maxIter);
        if (!Double.isFinite(target)) {
            throw new IllegalArgumentException("target return must be finite");
        }
        double[] wGmv = requireConverged(
                minVarianceLongOnlySolve(sigma, bounds[0], bounds[1], maxIter),
                "constrainedFrontierSolve (min-variance point)");
        double[] range = attainableReturnRange(mu, bounds[0], bounds[1]);
        return frontierPoint(mu, sigma, target, bounds[0], bounds[1], wGmv,
                Linalg.dot(mu, wGmv), range[0], range[1], maxIter);
    }

    /**
     * Weights of {@link #constrainedFrontierSolve} with the pinned iteration cap.
     *
     * @param mu     expected returns
     * @param sigma  SPD covariance
     * @param target target portfolio return
     * @param lb     lower bounds (null = 0)
     * @param ub     upper bounds (null = +inf)
     * @return the frontier weights
     * @throws IllegalArgumentException if the solver did not certify convergence
     */
    public static double[] constrainedFrontierWeights(double[] mu, double[][] sigma, double target,
            double[] lb, double[] ub) {
        return requireConverged(constrainedFrontierSolve(mu, sigma, target, lb, ub, QP_MAX_ITER),
                "constrainedFrontierWeights");
    }

    /**
     * Constrained frontier point with long-only bounds.
     *
     * @param mu     expected returns
     * @param sigma  SPD covariance
     * @param target target portfolio return
     * @return the frontier weights on the long-only simplex
     */
    public static double[] constrainedFrontierWeights(double[] mu, double[][] sigma, double target) {
        return constrainedFrontierWeights(mu, sigma, target, null, null);
    }

    /**
     * Constrained frontier sweep on the pinned grid from the constrained
     * minimum-variance return to the maximum attainable return under the
     * bounds ({@code nPoints} equally spaced targets, both ends inclusive).
     * Every point is solved to KKT optimality; a non-converged point throws.
     *
     * @param mu      expected returns
     * @param sigma   SPD covariance
     * @param nPoints number of grid points (&ge; 2)
     * @param lb      lower bounds (null = 0)
     * @param ub      upper bounds (null = +inf)
     * @return targets, weights and variances
     */
    public static Frontier constrainedFrontier(double[] mu, double[][] sigma, int nPoints,
            double[] lb, double[] ub) {
        validateMuSigma(mu, sigma);
        if (nPoints < 2) {
            throw new IllegalArgumentException("n_points must be >= 2");
        }
        double[][] bounds = qpSetup(sigma, lb, ub, QP_MAX_ITER);
        double[] wGmv = requireConverged(
                minVarianceLongOnlySolve(sigma, bounds[0], bounds[1], QP_MAX_ITER),
                "constrainedFrontier (min-variance point)");
        double rGmv = Linalg.dot(mu, wGmv);
        double[] range = attainableReturnRange(mu, bounds[0], bounds[1]);
        double[] targets = frontierGrid(rGmv, range[1], nPoints, "maximum attainable return");
        double[][] weights = new double[nPoints][];
        double[] variances = new double[nPoints];
        for (int k = 0; k < nPoints; k++) {
            QpResult res = frontierPoint(mu, sigma, targets[k], bounds[0], bounds[1], wGmv, rGmv,
                    range[0], range[1], QP_MAX_ITER);
            weights[k] = requireConverged(res, String.format(
                    "constrainedFrontier point %d (target %.6g)", k, targets[k]));
            variances[k] = Linalg.quadForm(weights[k], sigma);
        }
        return new Frontier(targets, weights, variances);
    }

    /**
     * Constrained frontier sweep with long-only bounds.
     *
     * @param mu      expected returns
     * @param sigma   SPD covariance
     * @param nPoints number of grid points (&ge; 2)
     * @return targets, weights and variances of the long-only points
     */
    public static Frontier constrainedFrontier(double[] mu, double[][] sigma, int nPoints) {
        return constrainedFrontier(mu, sigma, nPoints, null, null);
    }
}
