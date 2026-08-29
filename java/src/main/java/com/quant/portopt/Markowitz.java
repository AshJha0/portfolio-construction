package com.quant.portopt;

import java.util.Arrays;
import java.util.function.ToDoubleFunction;
import java.util.function.UnaryOperator;

/**
 * Markowitz mean-variance: closed forms, simplex projection, and PGD.
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
 * {@code A - rf C > 0}.
 *
 * <p>Constrained problems (long-only, sum-to-one, optional box bounds) are
 * solved by projected gradient descent with monotone backtracking and an exact
 * sort-based projection onto the box-constrained simplex; all steps are pinned
 * in API_SPEC.md sections 3.5-3.6 so ports match to below the golden 1e-8.
 */
public final class Markowitz {

    /** Pinned quadratic-penalty coefficient for the constrained frontier. */
    public static final double FRONTIER_PENALTY = 1.0e4;
    /** Pinned PGD convergence tolerance on the max-abs step. */
    public static final double PGD_TOL = 1.0e-12;
    /** Pinned PGD iteration cap (non-convergence is flagged, never thrown). */
    public static final int PGD_MAX_ITER = 10_000;

    private Markowitz() {
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
     * Result of a projected-gradient-descent run.
     *
     * @param weights    the final iterate on the feasible set
     * @param iterations iterations executed
     * @param converged  whether the pinned stopping rule fired before the cap
     */
    public record PgdResult(double[] weights, int iterations, boolean converged) {
    }

    /**
     * A frontier sweep: {@code points} targets with their weights and variances.
     *
     * @param targets   target returns (pinned grid)
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
            if (sigma[0].length != 1 || sigma[0][0] <= 0 || !Double.isFinite(sigma[0][0])) {
                throw new IllegalArgumentException("1x1 covariance must be positive and finite");
            }
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
     * @throws IllegalArgumentException if {@code A - rf*C <= 0} — the tangency
     *         portfolio is undefined when the risk-free rate is at or above the
     *         minimum-variance return (in particular when rf exceeds every
     *         expected return)
     */
    public static double[] tangencyWeights(double[] mu, double[][] sigma, double rf) {
        validateMuSigma(mu, sigma);
        if (!Double.isFinite(rf)) {
            throw new IllegalArgumentException("rf must be finite");
        }
        if (sigma.length == 1) {
            if (mu[0] <= rf) {
                throw new IllegalArgumentException(
                        "tangency undefined: risk-free rate >= expected return of the single asset");
            }
            if (sigma[0][0] <= 0 || !Double.isFinite(sigma[0][0])) {
                throw new IllegalArgumentException("1x1 covariance must be positive and finite");
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
            throw new IllegalArgumentException(
                    "tangency undefined: A - rf*C <= 0 (risk-free rate is at or above "
                            + "the minimum-variance portfolio return)");
        }
        double[] w = new double[z.length];
        for (int i = 0; i < z.length; i++) {
            w[i] = z[i] / denom;
        }
        return w;
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
            if (Math.abs(target - mu[0]) > 1e-12 * (1.0 + Math.abs(mu[0]))) {
                throw new IllegalArgumentException(
                        "n=1: target return must equal the single asset's expected return");
            }
            return new double[] {1.0};
        }
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
        double mvRet = a / c;
        if (target < mvRet - 1e-12 * (1.0 + Math.abs(mvRet))) {
            throw new IllegalArgumentException(String.format(
                    "target return %.6g lies below the minimum-variance return %.6g; "
                            + "the efficient frontier is defined for targets >= A/C",
                    target, mvRet));
        }
        double[] w = new double[n];
        for (int i = 0; i < n; i++) {
            double g = (b * y[i] - a * x[i]) / d;
            double h = (c * x[i] - a * y[i]) / d;
            w[i] = g + h * target;
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

    /**
     * Unconstrained frontier sweep on the pinned grid: {@code nPoints} equally
     * spaced targets from the minimum-variance return A/C to {@code max(mu)}
     * inclusive ({@code target_k = lo + k*(hi - lo)/(nPoints - 1)}).
     *
     * @param mu      expected returns
     * @param sigma   SPD covariance
     * @param nPoints number of grid points (&ge; 2)
     * @return targets, weights and variances
     */
    public static Frontier efficientFrontier(double[] mu, double[][] sigma, int nPoints) {
        validateMuSigma(mu, sigma);
        if (nPoints < 2) {
            throw new IllegalArgumentException("n_points must be >= 2");
        }
        TwoFundScalars sc = twoFundScalars(mu, sigma);
        if (sc.d() <= 1e-12 * sc.b() * sc.c()) {
            throw new IllegalArgumentException("degenerate frontier: D <= 1e-12*B*C");
        }
        double[] targets = frontierGrid(mu, sc, nPoints);
        double[][] weights = new double[nPoints][];
        double[] variances = new double[nPoints];
        for (int k = 0; k < nPoints; k++) {
            weights[k] = frontierWeights(mu, sigma, targets[k]);
            variances[k] = Linalg.quadForm(weights[k], sigma);
        }
        return new Frontier(targets, weights, variances);
    }

    private static double[] frontierGrid(double[] mu, TwoFundScalars sc, int nPoints) {
        double lo = sc.a() / sc.c();
        double hi = mu[0];
        for (double v : mu) {
            hi = Math.max(hi, v);
        }
        double[] targets = new double[nPoints];
        for (int k = 0; k < nPoints; k++) {
            targets[k] = lo + k * (hi - lo) / (nPoints - 1);
        }
        return targets;
    }

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
     * ub = +inf it reproduces the classic Duchi et al. sorted simplex
     * projection.
     *
     * @param v     point to project (finite entries)
     * @param total required coordinate sum (finite)
     * @param lb    per-coordinate lower bounds, or null for all 0
     * @param ub    per-coordinate upper bounds, or null for all +inf
     * @return the projection
     * @throws IllegalArgumentException if bounds are infeasible
     *         ({@code sum(lb) > total}, {@code sum(ub) < total}, any
     *         {@code lb_i > ub_i}) or inputs are invalid (empty v, NaN/inf in v,
     *         NaN in bounds)
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
        double[] lo = new double[n];
        double[] hi = new double[n];
        for (int i = 0; i < n; i++) {
            lo[i] = (lb == null) ? 0.0 : lb[boundIndex(lb, i, n)];
            hi[i] = (ub == null) ? Double.POSITIVE_INFINITY : ub[boundIndex(ub, i, n)];
            if (Double.isNaN(lo[i]) || Double.isNaN(hi[i])) {
                throw new IllegalArgumentException("bounds contain NaN");
            }
            if (lo[i] > hi[i]) {
                throw new IllegalArgumentException("infeasible bounds: some lb_i > ub_i");
            }
        }
        double loSum = Linalg.sum(lo);
        double hiSum = Linalg.sum(hi);
        double feasTol = 1e-12 * (1.0 + Math.abs(total));
        if (loSum > total + feasTol) {
            throw new IllegalArgumentException(String.format(
                    "infeasible bounds: sum(lb) = %.6g exceeds total %.6g", loSum, total));
        }
        if (hiSum < total - feasTol) {
            throw new IllegalArgumentException(String.format(
                    "infeasible bounds: sum(ub) = %.6g is below total %.6g", hiSum, total));
        }

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

    private static double gOfT(double[] v, double[] lo, double[] hi, double t) {
        // Clip into a temporary and reduce with the pinned pairwise sum —
        // mirrors the reference's sum(clip(v - t, lb, ub)) bit for bit.
        double[] clipped = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            clipped[i] = Math.min(Math.max(v[i] - t, lo[i]), hi[i]);
        }
        return Linalg.sum(clipped);
    }

    private static double[] clipShift(double[] v, double[] lo, double[] hi, double t) {
        double[] w = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            w[i] = Math.min(Math.max(v[i] - t, lo[i]), hi[i]);
        }
        return w;
    }

    /**
     * Projected gradient descent with monotone backtracking (pinned,
     * API_SPEC section 3.6):
     * <ol>
     *   <li>{@code w = project(w0)}; step {@code t = 1};</li>
     *   <li>each iteration: {@code t = 2t}; {@code g = grad(w)};
     *       {@code w+ = project(w - t g)}; while the sufficient-decrease test
     *       {@code f(w+) <= f(w) + g.(w+-w) + |w+-w|^2/(2t)} fails, halve t and
     *       recompute w+ (at most 100 halvings, then accept);</li>
     *   <li>stop when {@code max_i |w+_i - w_i| < tol} (converged) or after
     *       {@code maxIter} iterations (flagged, not thrown).</li>
     * </ol>
     * Deterministic — no randomness anywhere.
     *
     * @param fun     objective
     * @param grad    gradient of the objective
     * @param project projection onto the feasible set
     * @param w0      starting point (projected before use)
     * @param tol     stopping tolerance on the max-abs step
     * @param maxIter iteration cap
     * @return final iterate, iteration count and convergence flag
     */
    public static PgdResult pgdMinimize(
            ToDoubleFunction<double[]> fun,
            UnaryOperator<double[]> grad,
            UnaryOperator<double[]> project,
            double[] w0,
            double tol,
            int maxIter) {
        double[] w = project.apply(w0.clone());
        double t = 1.0;
        for (int it = 1; it <= maxIter; it++) {
            t *= 2.0;
            double[] g = grad.apply(w);
            double fw = fun.applyAsDouble(w);
            double[] wNew = null;
            double[] dw = null;
            boolean accepted = false;
            for (int h = 0; h < 100; h++) {
                wNew = project.apply(step(w, g, t));
                dw = diff(wNew, w);
                if (fun.applyAsDouble(wNew)
                        <= fw + Linalg.dot(g, dw) + Linalg.dot(dw, dw) / (2.0 * t)) {
                    accepted = true;
                    break;
                }
                t *= 0.5;
            }
            if (!accepted) {
                wNew = project.apply(step(w, g, t));
                dw = diff(wNew, w);
            }
            double maxStep = 0.0;
            for (double d : dw) {
                maxStep = Math.max(maxStep, Math.abs(d));
            }
            w = wNew;
            if (maxStep < tol) {
                return new PgdResult(w, it, true);
            }
        }
        return new PgdResult(w, maxIter, false);
    }

    private static double[] step(double[] w, double[] g, double t) {
        double[] out = new double[w.length];
        for (int i = 0; i < w.length; i++) {
            out[i] = w[i] - t * g[i];
        }
        return out;
    }

    private static double[] diff(double[] a, double[] b) {
        double[] out = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            out[i] = a[i] - b[i];
        }
        return out;
    }

    /**
     * Constrained minimum variance {@code min 1/2 w' Sigma w} over the long-only
     * simplex (lb = 0, ub = +inf).
     *
     * @param sigma SPD covariance (validated via Cholesky)
     * @return the unique constrained minimum-variance weights
     */
    public static double[] minVarianceLongOnly(double[][] sigma) {
        return minVarianceLongOnly(sigma, null, null);
    }

    /**
     * Constrained minimum variance over the box-constrained simplex, solved by
     * the pinned PGD started from projected equal weights. The objective is
     * strictly convex, so the minimizer is unique and ports agree to well below
     * the golden tolerance 1e-8.
     *
     * @param sigma SPD covariance
     * @param lb    lower bounds (null = 0; length 1 broadcasts)
     * @param ub    upper bounds (null = +inf; length 1 broadcasts)
     * @return constrained minimum-variance weights
     */
    public static double[] minVarianceLongOnly(double[][] sigma, double[] lb, double[] ub) {
        final double[][] s = sigma;
        Linalg.cholesky(s); // PSD/validity gate
        int n = s.length;
        UnaryOperator<double[]> project = v -> projectSimplexBox(v, 1.0, lb, ub);
        double[] w0 = new double[n];
        Arrays.fill(w0, 1.0 / n);
        ToDoubleFunction<double[]> fun = w -> 0.5 * Linalg.quadForm(w, s);
        UnaryOperator<double[]> grad = w -> Linalg.matVec(s, w);
        return pgdMinimize(fun, grad, project, w0, PGD_TOL, PGD_MAX_ITER).weights();
    }

    /**
     * Constrained frontier point via the pinned quadratic-penalty PGD
     * (API_SPEC section 3.6):
     *
     * <pre>f(w) = 1/2 w' Sigma w + penalty/2 * (mu'w - target)^2</pre>
     *
     * minimized over {@code {sum w = 1, lb <= w <= ub}}. The penalty formulation
     * itself is the contract: golden values are the exact minimizer of THIS
     * objective (the achieved return deviates from the target by O(1/penalty)
     * by design).
     *
     * @param mu      expected returns
     * @param sigma   SPD covariance
     * @param target  target portfolio return
     * @param penalty positive penalty coefficient (pinned default 1e4)
     * @param lb      lower bounds (null = 0)
     * @param ub      upper bounds (null = +inf)
     * @return minimizer of the penalized objective on the feasible set
     */
    public static double[] constrainedFrontierWeights(
            double[] mu, double[][] sigma, double target, double penalty,
            double[] lb, double[] ub) {
        validateMuSigma(mu, sigma);
        if (penalty <= 0) {
            throw new IllegalArgumentException("penalty must be positive");
        }
        final double[][] s = sigma;
        final double[] m = mu;
        final double tgt = target;
        final double pen = penalty;
        Linalg.cholesky(s); // PSD/validity gate
        int n = s.length;
        UnaryOperator<double[]> project = v -> projectSimplexBox(v, 1.0, lb, ub);
        double[] w0 = new double[n];
        Arrays.fill(w0, 1.0 / n);
        ToDoubleFunction<double[]> fun = w -> {
            double gap = Linalg.dot(m, w) - tgt;
            return 0.5 * Linalg.quadForm(w, s) + 0.5 * pen * gap * gap;
        };
        UnaryOperator<double[]> grad = w -> {
            double gap = Linalg.dot(m, w) - tgt;
            double[] g = Linalg.matVec(s, w);
            for (int i = 0; i < g.length; i++) {
                g[i] += pen * gap * m[i];
            }
            return g;
        };
        return pgdMinimize(fun, grad, project, w0, PGD_TOL, PGD_MAX_ITER).weights();
    }

    /**
     * Constrained frontier point with the pinned penalty and long-only bounds.
     *
     * @param mu     expected returns
     * @param sigma  SPD covariance
     * @param target target portfolio return
     * @return minimizer of the penalized objective on the long-only simplex
     */
    public static double[] constrainedFrontierWeights(double[] mu, double[][] sigma, double target) {
        return constrainedFrontierWeights(mu, sigma, target, FRONTIER_PENALTY, null, null);
    }

    /**
     * Constrained frontier sweep over the same pinned target grid as
     * {@link #efficientFrontier} (min-variance return A/C to max(mu)).
     *
     * @param mu      expected returns
     * @param sigma   SPD covariance
     * @param nPoints number of grid points (&ge; 2)
     * @return targets, weights and variances of the penalized long-only points
     */
    public static Frontier constrainedFrontier(double[] mu, double[][] sigma, int nPoints) {
        validateMuSigma(mu, sigma);
        if (nPoints < 2) {
            throw new IllegalArgumentException("n_points must be >= 2");
        }
        TwoFundScalars sc = twoFundScalars(mu, sigma);
        double[] targets = frontierGrid(mu, sc, nPoints);
        double[][] weights = new double[nPoints][];
        double[] variances = new double[nPoints];
        for (int k = 0; k < nPoints; k++) {
            weights[k] = constrainedFrontierWeights(mu, sigma, targets[k]);
            variances[k] = Linalg.quadForm(weights[k], sigma);
        }
        return new Frontier(targets, weights, variances);
    }
}
