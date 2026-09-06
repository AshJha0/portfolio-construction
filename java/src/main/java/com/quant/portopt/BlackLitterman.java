package com.quant.portopt;

import java.util.List;

/**
 * Black-Litterman: reverse optimization and the posterior master formula.
 *
 * <p>Master formula (pinned form, API_SPEC section 4). With prior (equilibrium)
 * returns pi, uncertainty scaling tau in (0, 1], pick matrix P (k x n), view
 * returns Q (k), and view covariance Omega (k x k):
 *
 * <pre>
 *   A_k     = P (tau Sigma) P' + Omega                       (k x k, SPD)
 *   mu_post = pi + (tau Sigma) P' A_k^-1 (Q - P pi)
 *   M       = tau Sigma - (tau Sigma) P' A_k^-1 P (tau Sigma)
 *   S_post  = Sigma + M
 * </pre>
 *
 * This is algebraically identical to the textbook
 * {@code [(tau S)^-1 + P' O^-1 P]^-1 [(tau S)^-1 pi + P' O^-1 Q]} form but
 * needs only one k x k Cholesky factorization, which is the pinned
 * implementation.
 *
 * <p>Limits (tested): tau to 0 at fixed Omega gives mu_post to pi and M to 0;
 * Omega to infinity (zero confidence) gives mu_post to pi and M to tau Sigma.
 *
 * <p>Omega default (He &amp; Litterman 1999, scaled by a confidence divisor):
 * {@code Omega = diag(P (tau Sigma) P')} with each entry divided by a per-view
 * confidence c_k &gt; 0: {@code Omega_kk = [P tau S P']_kk / c_k} (c_k to
 * infinity: dogmatic view; c_k to 0: view ignored). This is <em>not</em>
 * Idzorek's (2005) calibration, which solves for Omega such that the
 * posterior tilt is a stated fraction of the full-confidence tilt.
 */
public final class BlackLitterman {

    private BlackLitterman() {
    }

    /**
     * One investor view (schema of data/views.json).
     *
     * <p>Type "absolute": P row is the unit vector of {@code asset}. Type
     * "relative": P row is +1 on {@code longAsset} and -1 on {@code shortAsset}.
     * {@code confidence} must be positive (1 = the plain He-Litterman Omega).
     *
     * @param type       "absolute" or "relative"
     * @param asset      asset name for an absolute view (null otherwise)
     * @param longAsset  long leg of a relative view (null otherwise)
     * @param shortAsset short leg of a relative view (null otherwise)
     * @param q          the view return
     * @param confidence per-view confidence (&gt; 0)
     */
    public record View(String type, String asset, String longAsset, String shortAsset,
                       double q, double confidence) {

        /**
         * Creates an absolute view "asset returns q".
         *
         * @param asset      asset name
         * @param q          view return
         * @param confidence per-view confidence
         * @return the view
         */
        public static View absolute(String asset, double q, double confidence) {
            return new View("absolute", asset, null, null, q, confidence);
        }

        /**
         * Creates a relative view "longAsset beats shortAsset by q".
         *
         * @param longAsset  long leg
         * @param shortAsset short leg
         * @param q          view return spread
         * @param confidence per-view confidence
         * @return the view
         */
        public static View relative(String longAsset, String shortAsset, double q,
                double confidence) {
            return new View("relative", null, longAsset, shortAsset, q, confidence);
        }
    }

    /**
     * The (P, Q, confidences) triple built from a view list.
     *
     * @param p           pick matrix (k x n)
     * @param q           view returns (k)
     * @param confidences per-view confidences (k)
     */
    public record ViewMatrices(double[][] p, double[] q, double[] confidences) {
    }

    /**
     * Black-Litterman posterior.
     *
     * @param muPost    posterior expected returns
     * @param sigmaPost posterior return covariance {@code Sigma + M}
     * @param m         posterior covariance of the mean estimate,
     *                  {@code tau Sigma - tau Sigma P' A_k^-1 P tau Sigma}
     * @param omega     the view covariance actually used
     */
    public record Result(double[] muPost, double[][] sigmaPost, double[][] m, double[][] omega) {
    }

    /**
     * Reverse optimization: {@code pi = delta * Sigma * w_mkt} with the
     * market-cap weights renormalized to sum 1. The resulting pi are
     * equilibrium <em>excess</em> returns: the tangency portfolio at rf = 0 on
     * (pi, Sigma) recovers exactly w_mkt.
     *
     * @param sigma SPD covariance
     * @param wMkt  non-negative market-cap weights with positive sum
     * @param delta risk aversion (&gt; 0, finite)
     * @return implied equilibrium excess returns
     */
    public static double[] impliedEquilibriumReturns(double[][] sigma, double[] wMkt, double delta) {
        double[][] s = Linalg.checkSquareSymmetric(sigma);
        if (wMkt == null || wMkt.length != s.length) {
            throw new IllegalArgumentException("w_mkt length "
                    + (wMkt == null ? 0 : wMkt.length) + " does not match sigma dimension "
                    + s.length);
        }
        double tot = 0.0;
        for (double w : wMkt) {
            if (!Double.isFinite(w)) {
                throw new IllegalArgumentException("w_mkt contains NaN or infinite entries");
            }
            if (w < 0) {
                throw new IllegalArgumentException("market-cap weights must be non-negative");
            }
            tot += w;
        }
        if (tot <= 0) {
            throw new IllegalArgumentException("market-cap weights must have positive sum");
        }
        if (!Double.isFinite(delta) || delta <= 0) {
            throw new IllegalArgumentException("risk aversion delta must be positive and finite");
        }
        double[] wNorm = new double[wMkt.length];
        for (int i = 0; i < wMkt.length; i++) {
            wNorm[i] = wMkt[i] / tot;
        }
        double[] pi = Linalg.matVec(s, wNorm);
        for (int i = 0; i < pi.length; i++) {
            pi[i] = delta * pi[i];
        }
        return pi;
    }

    /**
     * Builds (P, Q, confidences) from a list of views.
     *
     * @param views      non-empty view list
     * @param assetNames universe asset names, in column order (non-empty, unique)
     * @return the pick matrix, view returns and confidences
     * @throws IllegalArgumentException on an empty view or asset list, duplicate
     *         asset names, unknown view type, unknown asset name, a relative
     *         view whose legs coincide, or non-finite q / non-positive confidence
     */
    public static ViewMatrices viewsToMatrices(List<View> views, List<String> assetNames) {
        if (assetNames == null || assetNames.isEmpty()) {
            throw new IllegalArgumentException("asset_names must be non-empty");
        }
        if (new java.util.HashSet<>(assetNames).size() != assetNames.size()) {
            throw new IllegalArgumentException("asset_names contains duplicates");
        }
        if (views == null || views.isEmpty()) {
            throw new IllegalArgumentException("views list must be non-empty");
        }
        int k = views.size();
        int n = assetNames.size();
        double[][] p = new double[k][n];
        double[] q = new double[k];
        double[] conf = new double[k];
        for (int i = 0; i < k; i++) {
            View view = views.get(i);
            String type = view.type();
            if ("absolute".equals(type)) {
                p[i][nameIndex(assetNames, view.asset())] = 1.0;
            } else if ("relative".equals(type)) {
                int iLong = nameIndex(assetNames, view.longAsset());
                int iShort = nameIndex(assetNames, view.shortAsset());
                if (iLong == iShort) {
                    throw new IllegalArgumentException(
                            "view " + i + ": relative view: long and short legs must differ");
                }
                p[i][iLong] = 1.0;
                p[i][iShort] = -1.0;
            } else {
                throw new IllegalArgumentException("unknown view type: " + type);
            }
            if (!Double.isFinite(view.q())) {
                throw new IllegalArgumentException("view " + i + ": q must be finite");
            }
            q[i] = view.q();
            double c = view.confidence();
            if (!Double.isFinite(c) || c <= 0) {
                throw new IllegalArgumentException(
                        "view " + i + ": confidence must be positive and finite");
            }
            conf[i] = c;
        }
        return new ViewMatrices(p, q, conf);
    }

    private static int nameIndex(List<String> names, String name) {
        int idx = names.indexOf(name);
        if (idx < 0) {
            throw new IllegalArgumentException("unknown asset name in view: " + name);
        }
        return idx;
    }

    private static double[][] validateViews(double[][] p, int n) {
        if (p == null || p.length < 1) {
            throw new IllegalArgumentException("P must contain at least one view row");
        }
        for (double[] row : p) {
            if (row == null || row.length != n) {
                throw new IllegalArgumentException("P must be k x " + n);
            }
            double maxAbs = 0.0;
            for (double v : row) {
                if (!Double.isFinite(v)) {
                    throw new IllegalArgumentException("P contains NaN or infinite entries");
                }
                maxAbs = Math.max(maxAbs, Math.abs(v));
            }
            if (maxAbs == 0.0) {
                throw new IllegalArgumentException("a view row of P is identically zero");
            }
        }
        return p;
    }

    private static double validateTau(double tau) {
        if (!Double.isFinite(tau) || tau <= 0.0 || tau > 1.0) {
            throw new IllegalArgumentException("tau must lie in (0, 1], got " + tau);
        }
        return tau;
    }

    /**
     * He-Litterman default view covariance scaled by a confidence divisor:
     * {@code Omega = diag(P (tau Sigma) P')} with each diagonal entry divided by
     * the corresponding per-view confidence ({@code c_k > 0}; null means all 1).
     *
     * @param sigma       SPD covariance
     * @param p           pick matrix (k x n)
     * @param tau         uncertainty scaling in (0, 1]
     * @param confidences per-view confidences, or null for all 1
     * @return diagonal k x k Omega
     */
    public static double[][] defaultOmega(double[][] sigma, double[][] p, double tau,
            double[] confidences) {
        double[][] s = Linalg.checkSquareSymmetric(sigma);
        double[][] pm = validateViews(p, s.length);
        double t = validateTau(tau);
        int k = pm.length;
        double[] conf = new double[k];
        if (confidences == null) {
            java.util.Arrays.fill(conf, 1.0);
        } else {
            if (confidences.length != k) {
                throw new IllegalArgumentException("confidences must have length " + k);
            }
            for (int i = 0; i < k; i++) {
                if (!Double.isFinite(confidences[i]) || confidences[i] <= 0) {
                    throw new IllegalArgumentException("confidences must be positive and finite");
                }
                conf[i] = confidences[i];
            }
        }
        double[][] ts = scale(s, t);
        double[][] b = matMul(pm, ts);          // k x n
        double[][] omega = new double[k][k];
        for (int i = 0; i < k; i++) {
            omega[i][i] = Linalg.dot(b[i], pm[i]) / conf[i];
        }
        return omega;
    }

    /**
     * Black-Litterman posterior via the pinned master formula with the default
     * (He-Litterman) Omega built from per-view confidences.
     *
     * @param sigma       SPD covariance
     * @param pi          prior (equilibrium) returns
     * @param p           pick matrix (k x n)
     * @param q           view returns (k)
     * @param tau         uncertainty scaling in (0, 1]
     * @param confidences per-view confidences, or null for all 1
     * @return the posterior
     */
    public static Result blPosterior(double[][] sigma, double[] pi, double[][] p, double[] q,
            double tau, double[] confidences) {
        return blPosterior(sigma, pi, p, q, tau, null, confidences);
    }

    /**
     * Black-Litterman posterior via the pinned master formula.
     *
     * <p>The k x k system {@code A_k = P tau Sigma P' + Omega} is factored once
     * by the pinned Cholesky and solved column-wise (error if not SPD).
     *
     * @param sigma       SPD covariance (n x n)
     * @param pi          prior (equilibrium) returns (n)
     * @param p           pick matrix (k x n)
     * @param q           view returns (k)
     * @param tau         uncertainty scaling in (0, 1]
     * @param omega       view covariance, or null for the default of
     *                    {@link #defaultOmega}
     * @param confidences per-view confidences for the default Omega (null means
     *                    all 1); passing both {@code omega} and confidences is
     *                    an error, since the confidences only parametrize the
     *                    default Omega
     * @return the posterior (mu_post, Sigma_post, M, Omega)
     */
    public static Result blPosterior(double[][] sigma, double[] pi, double[][] p, double[] q,
            double tau, double[][] omega, double[] confidences) {
        double[][] s = Linalg.checkSquareSymmetric(sigma);
        int n = s.length;
        if (pi == null || pi.length != n) {
            throw new IllegalArgumentException("pi must be a finite vector of length " + n);
        }
        for (double v : pi) {
            if (!Double.isFinite(v)) {
                throw new IllegalArgumentException("pi must be a finite vector of length " + n);
            }
        }
        double[][] pm = validateViews(p, n);
        int k = pm.length;
        if (q == null || q.length != k) {
            throw new IllegalArgumentException("Q must be a finite vector of length " + k);
        }
        for (double v : q) {
            if (!Double.isFinite(v)) {
                throw new IllegalArgumentException("Q must be a finite vector of length " + k);
            }
        }
        double t = validateTau(tau);
        double[][] om;
        if (omega == null) {
            om = defaultOmega(s, pm, t, confidences);
        } else {
            if (confidences != null) {
                throw new IllegalArgumentException(
                        "confidences are only used with the default Omega; pass one or the other");
            }
            om = Linalg.checkSquareSymmetric(omega);
            if (om.length != k) {
                throw new IllegalArgumentException("Omega must be " + k + " x " + k);
            }
        }

        double[][] ts = scale(s, t);
        double[][] b = matMul(pm, ts);                    // k x n = P (tau Sigma)
        double[][] aK = new double[k][k];                 // B P' + Omega
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                aK[i][j] = Linalg.dot(b[i], pm[j]) + om[i][j];
            }
        }
        double[][] low = Linalg.cholesky(aK); // throws if Omega makes A_k non-SPD

        double[] resid = new double[k];
        for (int i = 0; i < k; i++) {
            resid[i] = q[i] - Linalg.dot(pm[i], pi);
        }
        double[] x = Linalg.cholSolve(low, resid);
        double[] muPost = new double[n];
        for (int j = 0; j < n; j++) {
            double acc = 0.0;
            for (int i = 0; i < k; i++) {
                acc += b[i][j] * x[i];
            }
            muPost[j] = pi[j] + acc;
        }

        // M = tau S - B' A_k^-1 B: A_k^-1 B via n solves, one per column of B.
        double[][] ainvB = new double[k][n];
        double[] col = new double[k];
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < k; i++) {
                col[i] = b[i][j];
            }
            double[] sol = Linalg.cholSolve(low, col);
            for (int i = 0; i < k; i++) {
                ainvB[i][j] = sol[i];
            }
        }
        double[][] mMat = new double[n][n];
        double[][] sigmaPost = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double acc = 0.0;
                for (int r = 0; r < k; r++) {
                    acc += b[r][i] * ainvB[r][j];
                }
                mMat[i][j] = ts[i][j] - acc;
                sigmaPost[i][j] = s[i][j] + mMat[i][j];
            }
        }
        return new Result(muPost, sigmaPost, mMat, om);
    }

    /**
     * Posterior-based optimal weights.
     *
     * <p>Pinned: unconstrained tangency portfolio on
     * {@code (mu_post, Sigma_post)} at {@code rf = 0} — the posterior returns
     * are excess returns (pi = delta Sigma w_mkt), so a non-zero rf would
     * double-count the risk-free rate. With no active views the tangency on
     * (pi, Sigma) recovers exactly w_mkt.
     *
     * @param sigma       SPD covariance
     * @param pi          prior (equilibrium) returns
     * @param p           pick matrix
     * @param q           view returns
     * @param tau         uncertainty scaling in (0, 1]
     * @param confidences per-view confidences, or null for all 1
     * @return the BL optimal weights (tangency at rf = 0 on the posterior)
     */
    public static double[] blWeights(double[][] sigma, double[] pi, double[][] p, double[] q,
            double tau, double[] confidences) {
        return blWeights(sigma, pi, p, q, tau, 0.0, null, confidences);
    }

    /**
     * Posterior-based optimal weights with an explicit risk-free rate and an
     * optional explicit Omega (same surface as the Python reference).
     *
     * @param sigma       SPD covariance
     * @param pi          prior (equilibrium) returns
     * @param p           pick matrix
     * @param q           view returns
     * @param tau         uncertainty scaling in (0, 1]
     * @param rf          risk-free rate for the tangency (finite; pinned default 0)
     * @param omega       view covariance, or null for the default Omega
     * @param confidences per-view confidences for the default Omega, or null;
     *                    must be null when {@code omega} is given
     * @return the BL optimal weights (tangency at rf on the posterior)
     */
    public static double[] blWeights(double[][] sigma, double[] pi, double[][] p, double[] q,
            double tau, double rf, double[][] omega, double[] confidences) {
        if (!Double.isFinite(rf)) {
            throw new IllegalArgumentException("rf must be finite");
        }
        Result res = blPosterior(sigma, pi, p, q, tau, omega, confidences);
        return Markowitz.tangencyWeights(res.muPost(), res.sigmaPost(), rf);
    }

    private static double[][] scale(double[][] a, double factor) {
        double[][] out = new double[a.length][a.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a.length; j++) {
                out[i][j] = factor * a[i][j];
            }
        }
        return out;
    }

    private static double[][] matMul(double[][] a, double[][] b) {
        int rows = a.length;
        int inner = b.length;
        int cols = b[0].length;
        double[][] out = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double s = 0.0;
                for (int r = 0; r < inner; r++) {
                    s += a[i][r] * b[r][j];
                }
                out[i][j] = s;
            }
        }
        return out;
    }
}
