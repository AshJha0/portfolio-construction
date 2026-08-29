package com.quant.portopt;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/** Unit and property tests for the Markowitz closed forms, projection and PGD. */
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
        assertThrows(IllegalArgumentException.class, () -> Markowitz.tangencyWeights(
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
        assertThrows(IllegalArgumentException.class,
                () -> Markowitz.frontierWeights(mu1, sigma1, 0.05));
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
    public void pgdMatchesClosedFormWhenConstraintsInactive() {
        // Wide box bounds leave only sum(w) = 1 active, so PGD must agree with
        // the closed-form unconstrained minimum-variance portfolio.
        int n = TestData.sigmaAnnualized().length;
        double[] lb = new double[n];
        double[] ub = new double[n];
        java.util.Arrays.fill(lb, -10.0);
        java.util.Arrays.fill(ub, 10.0);
        double[] wPgd = Markowitz.minVarianceLongOnly(TestData.sigmaAnnualized(), lb, ub);
        double[] wClosed = Markowitz.minVarianceWeights(TestData.sigmaAnnualized());
        assertArrayEquals(wClosed, wPgd, 1e-6);
    }

    @Test
    public void pgdLongOnlyIsFeasibleAndNoWorseThanEqualWeight() {
        double[] w = Markowitz.minVarianceLongOnly(TestData.sigmaAnnualized());
        assertEquals(1.0, Linalg.sum(w), 1e-10);
        double[] eq = new double[w.length];
        java.util.Arrays.fill(eq, 1.0 / w.length);
        for (double v : w) {
            assertTrue("long-only", v >= -1e-12);
        }
        assertTrue("PGD at least as good as equal weight",
                portVariance(w) <= portVariance(eq) + 1e-15);
    }

    @Test
    public void constrainedFrontierPenaltyKeepsReturnNearTarget() {
        double target = 0.06;
        double[] w = Markowitz.constrainedFrontierWeights(
                TestData.muAnnualized(), TestData.sigmaAnnualized(), target);
        // The achieved return deviates from the target by O(1/penalty).
        assertEquals(target, portReturn(w), 1e-3);
        assertEquals(1.0, Linalg.sum(w), 1e-10);
    }

    @Test
    public void pgdIsDeterministic() {
        double[] w1 = Markowitz.minVarianceLongOnly(TestData.sigmaAnnualized());
        double[] w2 = Markowitz.minVarianceLongOnly(TestData.sigmaAnnualized());
        assertArrayEquals("bit-identical repeated runs", w1, w2, 0.0);
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
        assertThrows(IllegalArgumentException.class, () -> Markowitz.constrainedFrontierWeights(
                TestData.muAnnualized(), TestData.sigmaAnnualized(), 0.06, -1.0, null, null));
    }
}
