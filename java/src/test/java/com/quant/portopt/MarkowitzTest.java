package com.quant.portopt;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/** Unit and property tests for the Markowitz closed forms, projection and the dual active-set QP. */
public class MarkowitzTest {

    private static double portReturn(double[] w) {
        return Linalg.dot(TestData.muAnnualized(), w);
    }

    private static double portVariance(double[] w) {
        return Linalg.quadForm(w, TestData.sigmaAnnualized());
    }

    @Test
    public void twoFundIdentities() {
        Markowitz.TwoFundScalars sc =
                Markowitz.twoFundScalars(TestData.muAnnualized(), TestData.sigmaAnnualized());
        assertTrue("B > 0", sc.b() > 0);
        assertTrue("C > 0", sc.c() > 0);
        assertTrue("D > 0 (Cauchy-Schwarz)", sc.d() > 0);
        double[] wMv = Markowitz.minVarianceWeights(TestData.sigmaAnnualized());
        assertEquals("min-var weights sum to 1", 1.0, Linalg.sum(wMv), 1e-12);
        assertEquals("min-var return = A/C", sc.a() / sc.c(), portReturn(wMv), 1e-12);
        assertEquals("min-var variance = 1/C", 1.0 / sc.c(), portVariance(wMv), 1e-14);
    }

    @Test
    public void frontierWeightsHitTargetReturn() {
        Markowitz.TwoFundScalars sc =
                Markowitz.twoFundScalars(TestData.muAnnualized(), TestData.sigmaAnnualized());
        double lo = sc.a() / sc.c();
        for (int k = 0; k <= 10; k++) {
            double target = lo + k * 0.01;
            double[] w = Markowitz.frontierWeights(
                    TestData.muAnnualized(), TestData.sigmaAnnualized(), target);
            assertEquals("sum(w) at target " + target, 1.0, Linalg.sum(w), 1e-10);
            assertEquals("achieved return at target " + target, target, portReturn(w), 1e-10);
            // Closed-form variance identity var(m) = (C m^2 - 2 A m + B) / D.
            assertEquals("variance identity at target " + target,
                    Markowitz.frontierVariance(sc, target), portVariance(w), 1e-12);
        }
    }

    @Test
    public void frontierVarianceConvexInTarget() {
        Markowitz.Frontier front = Markowitz.efficientFrontier(
                TestData.muAnnualized(), TestData.sigmaAnnualized(), 25);
        double muMax = TestData.muAnnualized()[0];
        for (double m : TestData.muAnnualized()) {
            muMax = Math.max(muMax, m);
        }
        assertEquals("last grid point is exactly max(mu)", muMax, front.targets()[24], 0.0);
        for (int k : new int[] {0, 7, 24}) {
            assertArrayEquals("sweep equals the point function", Markowitz.frontierWeights(
                    TestData.muAnnualized(), TestData.sigmaAnnualized(), front.targets()[k]),
                    front.weights()[k], 1e-15);
        }
        double[] v = front.variances();
        for (int k = 1; k + 1 < v.length; k++) {
            double secondDiff = v[k + 1] - 2.0 * v[k] + v[k - 1];
            assertTrue("second difference at k=" + k, secondDiff >= -1e-12);
        }
        // Variance is increasing along the efficient branch above min-var.
        for (int k = 1; k < v.length; k++) {
            assertTrue("variance non-decreasing at k=" + k, v[k] >= v[k - 1] - 1e-14);
        }
    }

    @Test
    public void tangencyHasMaxSharpeOnFrontierGrid() {
        double rf = 0.02;
        double[] wTan = Markowitz.tangencyWeights(
                TestData.muAnnualized(), TestData.sigmaAnnualized(), rf);
        double tanSharpe = (portReturn(wTan) - rf) / Math.sqrt(portVariance(wTan));
        Markowitz.Frontier front = Markowitz.efficientFrontier(
                TestData.muAnnualized(), TestData.sigmaAnnualized(), 25);
        for (int k = 0; k < front.targets().length; k++) {
            double sharpe = (portReturn(front.weights()[k]) - rf)
                    / Math.sqrt(front.variances()[k]);
            assertTrue("tangency dominates grid point " + k, tanSharpe >= sharpe - 1e-12);
        }
    }

    @Test
    public void frontierTargetBelowMinvarRejected() {
        Markowitz.TwoFundScalars sc =
                Markowitz.twoFundScalars(TestData.muAnnualized(), TestData.sigmaAnnualized());
        double below = sc.a() / sc.c() - 0.01;
        assertThrows(IllegalArgumentException.class, () -> Markowitz.frontierWeights(
                TestData.muAnnualized(), TestData.sigmaAnnualized(), below));
        assertThrows(IllegalArgumentException.class, () -> Markowitz.frontierWeights(
                TestData.muAnnualized(), TestData.sigmaAnnualized(), -0.5));
    }

    @Test
    public void degenerateFrontierRejected() {
        double[][] sigma = {{0.04, 0.01}, {0.01, 0.09}};
        double[] muProp = {0.05, 0.05}; // proportional to ones -> D ~ 0
        assertThrows(IllegalArgumentException.class,
                () -> Markowitz.frontierWeights(muProp, sigma, 0.06));
    }

    @Test
    public void tangencyRfTooHighRejected() {
        double maxMu = 0.0;
        for (double v : TestData.muAnnualized()) {
            maxMu = Math.max(maxMu, v);
        }
        final double rf = maxMu + 0.01; // above every expected return
        assertThrows(Markowitz.TangencyUndefinedException.class, () -> Markowitz.tangencyWeights(
                TestData.muAnnualized(), TestData.sigmaAnnualized(), rf));
    }

    @Test
    public void singleAssetTrivialWeights() {
        double[][] sigma1 = {{0.04}};
        double[] mu1 = {0.07};
        assertArrayEquals(new double[] {1.0}, Markowitz.minVarianceWeights(sigma1), 0.0);
        assertArrayEquals(new double[] {1.0},
                Markowitz.tangencyWeights(mu1, sigma1, 0.02), 0.0);
        assertThrows(IllegalArgumentException.class,
                () -> Markowitz.tangencyWeights(mu1, sigma1, 0.08));
        assertArrayEquals(new double[] {1.0},
                Markowitz.frontierWeights(mu1, sigma1, 0.07), 0.0);
        assertArrayEquals(new double[] {1.0}, Markowitz.minVarianceLongOnly(sigma1), 0.0);
        assertThrows(IllegalArgumentException.class,
                () -> Markowitz.frontierWeights(mu1, sigma1, 0.05));
        // MIN-11: the n = 1 paths validate the covariance too.
        assertThrows(IllegalArgumentException.class,
                () -> Markowitz.frontierWeights(mu1, new double[][] {{-1.0}}, 0.07));
        assertThrows(IllegalArgumentException.class,
                () -> Markowitz.tangencyWeights(mu1, new double[][] {{0.0}}, 0.02));
        assertThrows(IllegalArgumentException.class,
                () -> Markowitz.efficientFrontier(mu1, sigma1, 5));
    }

    @Test
    public void simplexProjectionHandCases() {
        assertArrayEquals(new double[] {0.7, 0.3, 0.0},
                Markowitz.projectSimplexBox(new double[] {0.9, 0.5, -0.2}, 1.0), 1e-12);
        // Already on the simplex: fixed point.
        assertArrayEquals(new double[] {0.25, 0.25, 0.25, 0.25},
                Markowitz.projectSimplexBox(new double[] {0.25, 0.25, 0.25, 0.25}, 1.0), 1e-12);
        // A dominant coordinate collapses to a vertex.
        assertArrayEquals(new double[] {1.0, 0.0},
                Markowitz.projectSimplexBox(new double[] {5.0, 0.0}, 1.0), 1e-12);
        // Box case: everything clipped to the bounds.
        assertArrayEquals(new double[] {0.5, 0.325, 0.125, 0.05},
                Markowitz.projectSimplexBox(new double[] {0.8, 0.6, 0.4, -0.3}, 1.0,
                        new double[] {0.05}, new double[] {0.5}), 1e-12);
    }

    @Test
    public void simplexProjectionFeasibleOverGrid() {
        // Property loop: for a deterministic grid of inputs the projection sums
        // to the total and respects the bounds.
        double[] lb = {0.0, 0.05, 0.0, 0.1};
        double[] ub = {0.6, 0.5, 0.4, 0.9};
        for (int a = -3; a <= 3; a++) {
            for (int b = -3; b <= 3; b++) {
                double[] v = {0.3 * a, 0.2 * b, 0.1 * a * b, -0.15 * a};
                double[] w = Markowitz.projectSimplexBox(v, 1.0, lb, ub);
                assertEquals("sum", 1.0, Linalg.sum(w), 1e-12);
                for (int i = 0; i < 4; i++) {
                    assertTrue("lb respected", w[i] >= lb[i] - 1e-12);
                    assertTrue("ub respected", w[i] <= ub[i] + 1e-12);
                }
            }
        }
    }

    @Test
    public void simplexProjectionIsEuclideanOptimal() {
        // The projection must beat any other feasible point in distance.
        double[] v = {0.9, 0.5, -0.2};
        double[] w = Markowitz.projectSimplexBox(v, 1.0);
        double dw = sqDist(v, w);
        double[][] feasible = {
                {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0},
                {0.5, 0.5, 0.0}, {0.4, 0.3, 0.3}, {0.7, 0.2, 0.1},
        };
        for (double[] f : feasible) {
            assertTrue("projection dominates " + java.util.Arrays.toString(f),
                    dw <= sqDist(v, f) + 1e-12);
        }
    }

    private static double sqDist(double[] a, double[] b) {
        double s = 0.0;
        for (int i = 0; i < a.length; i++) {
            s += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return s;
    }

    @Test
    public void simplexProjectionInfeasibleBoundsRejected() {
        double[] v = {0.5, 0.5};
        assertThrows(IllegalArgumentException.class, () -> Markowitz.projectSimplexBox(
                v, 1.0, new double[] {0.6, 0.6}, null)); // sum(lb) > 1
        assertThrows(IllegalArgumentException.class, () -> Markowitz.projectSimplexBox(
                v, 1.0, null, new double[] {0.3, 0.3})); // sum(ub) < 1
        assertThrows(IllegalArgumentException.class, () -> Markowitz.projectSimplexBox(
                v, 1.0, new double[] {0.5, 0.5}, new double[] {0.4, 0.6})); // lb > ub
        assertThrows(IllegalArgumentException.class, () -> Markowitz.projectSimplexBox(
                new double[] {Double.NaN, 0.0}, 1.0));
        assertThrows(IllegalArgumentException.class,
                () -> Markowitz.projectSimplexBox(new double[0], 1.0));
    }

    @Test
    public void qpMatchesClosedFormWhenConstraintsInactive() {
        // Wide box bounds leave only sum(w) = 1 active, so the QP must agree with
        // the closed-form unconstrained minimum-variance portfolio to rounding.
        int n = TestData.sigmaAnnualized().length;
        double[] lb = new double[n];
        double[] ub = new double[n];
        java.util.Arrays.fill(lb, -10.0);
        java.util.Arrays.fill(ub, 10.0);
        Markowitz.QpResult res = Markowitz.minVarianceLongOnlySolve(
                TestData.sigmaAnnualized(), lb, ub, Markowitz.QP_MAX_ITER);
        assertTrue(res.converged());
        assertTrue(res.iterations() <= 2);
        assertArrayEquals(Markowitz.minVarianceWeights(TestData.sigmaAnnualized()),
                res.weights(), 1e-14);
    }

    @Test
    public void qpLongOnlyCertifiedAndKkt() {
        double[][] sig = TestData.sigmaAnnualized();
        Markowitz.QpResult res = Markowitz.minVarianceLongOnlySolve(
                sig, null, null, Markowitz.QP_MAX_ITER);
        assertTrue(res.converged());
        assertTrue(res.kktResidual() < 1e-12);
        double[] w = res.weights();
        assertEquals(1.0, Linalg.sum(w), 1e-12);
        // KKT for {sum w = 1, w >= 0}: constant gradient on the support, >= off it.
        double[] g = Linalg.matVec(sig, w);
        double lam = Double.POSITIVE_INFINITY;
        for (int i = 0; i < w.length; i++) {
            assertTrue("long-only", w[i] >= 0.0);
            if (w[i] > 1e-14) {
                lam = Math.min(lam, g[i]);
            }
        }
        for (int i = 0; i < w.length; i++) {
            if (w[i] > 1e-14) {
                assertEquals(lam, g[i], 1e-13);
            } else {
                assertTrue(g[i] >= lam - 1e-13);
            }
        }
        double[] eq = new double[w.length];
        java.util.Arrays.fill(eq, 1.0 / w.length);
        assertTrue("QP at least as good as equal weight",
                portVariance(w) <= portVariance(eq) + 1e-15);
        assertTrue("cannot beat the unconstrained optimum",
                portVariance(w) >= portVariance(Markowitz.minVarianceWeights(sig)) - 1e-14);
    }

    @Test
    public void qpBoxBoundsRespectedAndOptimal() {
        // PT-14: lb = 0.02, ub = 0.40 hold exactly; beats projected random points.
        double[][] sig = TestData.sigmaAnnualized();
        Markowitz.QpResult res = Markowitz.minVarianceLongOnlySolve(
                sig, new double[] {0.02}, new double[] {0.40}, Markowitz.QP_MAX_ITER);
        assertTrue(res.converged());
        assertTrue(res.kktResidual() < 1e-12);
        for (double v : res.weights()) {
            assertTrue(v >= 0.02 && v <= 0.40);
        }
        assertEquals(1.0, Linalg.sum(res.weights()), 1e-12);
        double var = portVariance(res.weights());
        java.util.Random rng = new java.util.Random(3);
        for (int trial = 0; trial < 20; trial++) {
            double[] v = new double[8];
            for (int i = 0; i < 8; i++) {
                v[i] = rng.nextGaussian();
            }
            double[] z = Markowitz.projectSimplexBox(v, 1.0, new double[] {0.02}, new double[] {0.40});
            assertTrue(var <= portVariance(z) + 1e-15);
        }
    }

    @Test
    public void constrainedFrontierReportsConvergenceAndHitsTarget() {
        // PT-3: every point certified, target met exactly, variance monotone.
        double[] mu = TestData.muAnnualized();
        double[][] sig = TestData.sigmaAnnualized();
        Markowitz.QpResult res = Markowitz.constrainedFrontierSolve(
                mu, sig, 0.08, null, null, Markowitz.QP_MAX_ITER);
        assertTrue(res.converged());
        assertTrue(res.kktResidual() < 1e-9);
        assertEquals(0.08, portReturn(res.weights()), 1e-12);
        assertEquals(1.0, Linalg.sum(res.weights()), 1e-12);
        Markowitz.Frontier fc = Markowitz.constrainedFrontier(mu, sig, 25);
        double[] wLo = Markowitz.minVarianceLongOnly(sig);
        double muMax = mu[0];
        for (double v : mu) {
            muMax = Math.max(muMax, v);
        }
        assertEquals(portReturn(wLo), fc.targets()[0], 1e-15);
        assertEquals(muMax, fc.targets()[24], 0.0);
        for (int k = 0; k < 25; k++) {
            assertEquals("target " + k, fc.targets()[k], portReturn(fc.weights()[k]), 1e-12);
            for (double v : fc.weights()[k]) {
                assertTrue(v >= 0.0);
            }
            assertEquals(portVariance(fc.weights()[k]), fc.variances()[k], 1e-15);
            if (k > 0) {
                assertTrue("monotone variance " + k, fc.variances()[k] >= fc.variances()[k - 1] - 1e-14);
            }
        }
        assertArrayEquals(wLo, fc.weights()[0], 1e-15);
        for (int i = 0; i < 8; i++) {
            assertEquals(mu[i] == muMax ? 1.0 : 0.0, fc.weights()[24][i], 1e-15);
        }
    }

    @Test
    public void constrainedFrontierRejectsUnattainableTargets() {
        double[] mu = TestData.muAnnualized();
        double[][] sig = TestData.sigmaAnnualized();
        double rLo = portReturn(Markowitz.minVarianceLongOnly(sig));
        double muMax = mu[0];
        for (double v : mu) {
            muMax = Math.max(muMax, v);
        }
        final double above = muMax + 1e-3;
        assertThrows(IllegalArgumentException.class,
                () -> Markowitz.constrainedFrontierWeights(mu, sig, rLo - 1e-3));
        assertThrows(IllegalArgumentException.class,
                () -> Markowitz.constrainedFrontierWeights(mu, sig, above));
        assertThrows(IllegalArgumentException.class, () -> Markowitz.constrainedFrontierWeights(
                mu, sig, 0.09, null, new double[] {0.4}));
        assertThrows(IllegalArgumentException.class, () -> Markowitz.constrainedFrontierWeights(
                mu, sig, Double.POSITIVE_INFINITY));
        assertThrows(IllegalArgumentException.class, () -> Markowitz.constrainedFrontierSolve(
                mu, sig, 0.06, null, null, 0));
        assertThrows(IllegalArgumentException.class, () -> Markowitz.constrainedFrontier(mu, sig, 1));
    }

    @Test
    public void attainableReturnRangeAndDegenerateInputs() {
        double[] mu = TestData.muAnnualized();
        double[][] sig = TestData.sigmaAnnualized();
        double[] r = Markowitz.attainableReturnRange(mu, null, null);
        double[] sorted = mu.clone();
        java.util.Arrays.sort(sorted);
        assertEquals(sorted[0], r[0], 0.0);
        assertEquals(sorted[7], r[1], 0.0);
        double[] rc = Markowitz.attainableReturnRange(mu, new double[] {0.0}, new double[] {0.4});
        assertEquals(0.4 * sorted[7] + 0.4 * sorted[6] + 0.2 * sorted[5], rc[1], 1e-15);
        assertEquals(0.4 * sorted[0] + 0.4 * sorted[1] + 0.2 * sorted[2], rc[0], 1e-15);
        assertThrows(IllegalArgumentException.class,
                () -> Markowitz.attainableReturnRange(mu, new double[] {0.2}, null));
        // Fixed asset (lb == ub), tied top returns with caps, all-equal returns.
        double[] lb = new double[8];
        double[] ub = new double[8];
        java.util.Arrays.fill(ub, Double.POSITIVE_INFINITY);
        lb[5] = 0.30;
        ub[5] = 0.30;
        Markowitz.QpResult fixed = Markowitz.minVarianceLongOnlySolve(sig, lb, ub, Markowitz.QP_MAX_ITER);
        assertTrue(fixed.converged());
        assertEquals(0.30, fixed.weights()[5], 0.0);
        double[] mu2 = mu.clone();
        mu2[1] = mu2[0];
        double mu2Max = mu2[0];
        for (double v : mu2) {
            mu2Max = Math.max(mu2Max, v);
        }
        Markowitz.QpResult tie = Markowitz.constrainedFrontierSolve(
                mu2, sig, mu2Max, new double[] {0.0}, new double[] {0.7}, Markowitz.QP_MAX_ITER);
        assertTrue(tie.converged());
        assertTrue(tie.kktResidual() < 1e-12);
        assertEquals(1.0, tie.weights()[0] + tie.weights()[1], 1e-12);
        double[] flatMu = new double[8];
        java.util.Arrays.fill(flatMu, 0.05);
        Markowitz.QpResult flat = Markowitz.constrainedFrontierSolve(
                flatMu, sig, 0.05, null, null, Markowitz.QP_MAX_ITER);
        assertTrue(flat.converged());
        assertArrayEquals(Markowitz.minVarianceLongOnly(sig), flat.weights(), 0.0);
    }

    @Test
    public void frontierGridWhenGmvReturnExceedsMaxMu() {
        // MAJ-3: GMV shorts the low-return asset, A/C = 0.1185 > max(mu) = 0.10.
        double[] mu = {0.10, 0.05};
        double[][] sig = {{0.01, 0.027}, {0.027, 0.09}};
        Markowitz.TwoFundScalars sc = Markowitz.twoFundScalars(mu, sig);
        assertTrue(sc.a() / sc.c() > 0.10);
        assertThrows(IllegalArgumentException.class, () -> Markowitz.efficientFrontier(mu, sig, 25));
        assertThrows(IllegalArgumentException.class, () -> Markowitz.constrainedFrontier(mu, sig, 25));
        double[] w = Markowitz.frontierWeights(mu, sig, 0.13);
        assertEquals(0.13, Linalg.dot(mu, w), 1e-12);
    }

    @Test
    public void tangencyNearGmvRejected() {
        // MAJ-1: rf within 1e-6 (relative) of A/C cancels A - rf*C catastrophically.
        double[] mu = TestData.muAnnualized();
        double[][] sig = TestData.sigmaAnnualized();
        Markowitz.TwoFundScalars sc = Markowitz.twoFundScalars(mu, sig);
        double rGmv = sc.a() / sc.c();
        assertThrows(Markowitz.TangencyUndefinedException.class,
                () -> Markowitz.tangencyWeights(mu, sig, rGmv - 1e-13));
        assertThrows(Markowitz.TangencyUndefinedException.class,
                () -> Markowitz.tangencyWeights(mu, sig, rGmv - 1e-9));
        double[] w = Markowitz.tangencyWeights(mu, sig, rGmv - 1e-3);
        assertEquals(1.0, Linalg.sum(w), 1e-8);
        double l1 = 0.0;
        for (double v : w) {
            l1 += Math.abs(v);
        }
        assertEquals(l1, Markowitz.grossLeverage(w), 1e-12);
        assertEquals(1.0, Markowitz.grossLeverage(new double[] {0.2, 0.3, 0.5}), 1e-15);
        assertEquals(2.0, Markowitz.grossLeverage(new double[] {1.5, -0.5}), 1e-15);
        assertThrows(IllegalArgumentException.class, () -> Markowitz.grossLeverage(new double[0]));
        assertThrows(IllegalArgumentException.class,
                () -> Markowitz.grossLeverage(new double[] {Double.NaN}));
    }

    @Test
    public void qpIsDeterministic() {
        double[] w1 = Markowitz.minVarianceLongOnly(TestData.sigmaAnnualized());
        double[] w2 = Markowitz.minVarianceLongOnly(TestData.sigmaAnnualized());
        assertArrayEquals("bit-identical repeated runs", w1, w2, 0.0);
        Markowitz.Frontier a = Markowitz.constrainedFrontier(TestData.muAnnualized(), TestData.sigmaAnnualized(), 7);
        Markowitz.Frontier b = Markowitz.constrainedFrontier(TestData.muAnnualized(), TestData.sigmaAnnualized(), 7);
        for (int k = 0; k < 7; k++) {
            assertArrayEquals(a.weights()[k], b.weights()[k], 0.0);
        }
    }

    @Test
    public void invalidInputsRejected() {
        assertThrows(IllegalArgumentException.class, () -> Markowitz.twoFundScalars(
                new double[] {0.1}, TestData.sigmaAnnualized())); // length mismatch
        assertThrows(IllegalArgumentException.class, () -> Markowitz.tangencyWeights(
                TestData.muAnnualized(), TestData.sigmaAnnualized(), Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> Markowitz.frontierWeights(
                TestData.muAnnualized(), TestData.sigmaAnnualized(), Double.POSITIVE_INFINITY));
        assertThrows(IllegalArgumentException.class, () -> Markowitz.efficientFrontier(
                TestData.muAnnualized(), TestData.sigmaAnnualized(), 1)); // < 2 points
        assertThrows(IllegalArgumentException.class, () -> Markowitz.projectSimplexBox(
                new double[] {0.5, 0.5}, 1.0, new double[] {Double.NEGATIVE_INFINITY}, null));
        // Singular covariance rejected by the Cholesky gate in the QP too.
        assertThrows(IllegalArgumentException.class,
                () -> Markowitz.minVarianceLongOnly(new double[][] {{0.04, 0.04}, {0.04, 0.04}}));
    }

    @Test
    public void kktCertificateChecksDualFeasibility() {
        // min 1/2 |w|^2 s.t. sum w = 1, w >= 0 (S = I, n = 2). The true active
        // set {sum} gives w = (1/2, 1/2), u = 1/2 and is certified. Forcing the
        // wrong active set {sum, w_0 >= 0} gives w = (0, 1): primal feasible and
        // stationary (S w = 1*(1,1) - 1*e_0), but the bound multiplier is -1.
        double[][] s = {{1.0, 0.0}, {0.0, 1.0}};
        double[][] normals = {{1.0, 1.0}, {1.0, 0.0}, {0.0, 1.0}};
        double[] rhs = {1.0, 0.0, 0.0};
        Markowitz.KktCertificate ok = Markowitz.kktCertificate(
                s, normals, rhs, 1, new int[] {0}, new double[] {0.5}, new double[] {0.5, 0.5});
        assertTrue(ok.certified());
        assertEquals(0.0, ok.residual(), 0.0);
        Markowitz.KktCertificate bad = Markowitz.kktCertificate(s, normals, rhs, 1,
                new int[] {0, 1}, new double[] {1.0, -1.0}, new double[] {0.0, 1.0});
        assertTrue(!bad.certified());
        assertEquals(1.0, bad.residual(), 0.0);
        // A rounding-level negative bound multiplier is tolerated (scaled tol).
        Markowitz.KktCertificate tiny = Markowitz.kktCertificate(s, normals, rhs, 1,
                new int[] {0, 1}, new double[] {0.5, -1e-12}, new double[] {0.5, 0.5});
        assertTrue(tiny.certified());
        assertEquals(1e-12, tiny.residual(), 1e-15);
        // An equality multiplier may be negative.
        Markowitz.KktCertificate negEq = Markowitz.kktCertificate(s, new double[][] {{1.0, 1.0}},
                new double[] {-1.0}, 1, new int[] {0}, new double[] {-0.5},
                new double[] {-0.5, -0.5});
        assertTrue(negEq.certified());
    }

    @Test
    public void qpWithActiveBoundsStillCertified() {
        // Dual feasibility holds at the true optimum: problems with several
        // active bounds stay certified with the same residual bound as before.
        double[][] sig = TestData.sigmaAnnualized();
        double[] mu = TestData.muAnnualized();
        double[][] bounds = {{0.0, Double.POSITIVE_INFINITY}, {0.02, 0.40}, {0.0, 0.25}};
        for (double[] b : bounds) {
            Markowitz.QpResult res = Markowitz.minVarianceLongOnlySolve(
                    sig, new double[] {b[0]}, new double[] {b[1]}, Markowitz.QP_MAX_ITER);
            assertTrue(res.converged());
            assertTrue(res.kktResidual() < 1e-12);
            boolean anyActive = false;
            for (double v : res.weights()) {
                anyActive = anyActive || v == b[0] || v == b[1];
            }
            assertTrue(anyActive);
        }
        Markowitz.QpResult fr = Markowitz.constrainedFrontierSolve(mu, sig, 0.07,
                new double[] {0.0}, new double[] {0.5}, Markowitz.QP_MAX_ITER);
        assertTrue(fr.converged());
        assertTrue(fr.kktResidual() < 1e-9);
    }
}
