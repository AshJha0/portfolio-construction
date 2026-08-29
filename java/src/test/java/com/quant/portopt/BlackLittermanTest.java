package com.quant.portopt;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.List;
import org.junit.Test;

/** Unit and limit tests for Black-Litterman. */
public class BlackLittermanTest {

    private static double[] pi() {
        return BlackLitterman.impliedEquilibriumReturns(
                TestData.sigmaAnnualized(), TestData.wMkt(),
                (Double) TestData.viewsDoc().get("delta"));
    }

    private static BlackLitterman.ViewMatrices vm() {
        return BlackLitterman.viewsToMatrices(TestData.views(), TestData.assetNames());
    }

    @Test
    public void reverseOptimizationRoundTrip() {
        // pi are excess equilibrium returns: tangency at rf = 0 on (pi, Sigma)
        // recovers exactly the market-cap weights.
        double[] w = Markowitz.tangencyWeights(pi(), TestData.sigmaAnnualized(), 0.0);
        assertArrayEquals(TestData.wMkt(), w, 1e-10);
    }

    @Test
    public void viewsToMatricesStructure() {
        BlackLitterman.ViewMatrices vm = vm();
        List<String> names = TestData.assetNames();
        assertEquals(2, vm.p().length);
        // Absolute view on EM_EQ: unit row.
        assertEquals(1.0, vm.p()[0][names.indexOf("EM_EQ")], 0.0);
        assertEquals(1.0, Linalg.sum(vm.p()[0]), 0.0);
        // Relative view US_EQ - EU_EQ: +1/-1 row summing to 0.
        assertEquals(1.0, vm.p()[1][names.indexOf("US_EQ")], 0.0);
        assertEquals(-1.0, vm.p()[1][names.indexOf("EU_EQ")], 0.0);
        assertEquals(0.0, Linalg.sum(vm.p()[1]), 0.0);
        assertArrayEquals(new double[] {0.1, 0.02}, vm.q(), 0.0);
        assertArrayEquals(new double[] {1.0, 0.5}, vm.confidences(), 0.0);
    }

    @Test
    public void defaultOmegaDiagonalAndConfidenceScaling() {
        BlackLitterman.ViewMatrices vm = vm();
        double tau = (Double) TestData.viewsDoc().get("tau");
        double[][] om1 = BlackLitterman.defaultOmega(
                TestData.sigmaAnnualized(), vm.p(), tau, null);
        assertEquals(0.0, om1[0][1], 0.0);
        assertEquals(0.0, om1[1][0], 0.0);
        // Doubling the confidence halves the variance of the view.
        double[][] om2 = BlackLitterman.defaultOmega(
                TestData.sigmaAnnualized(), vm.p(), tau, new double[] {2.0, 2.0});
        assertEquals(om1[0][0] / 2.0, om2[0][0], 1e-18);
        assertEquals(om1[1][1] / 2.0, om2[1][1], 1e-18);
    }

    @Test
    public void posteriorMovesTowardViews() {
        BlackLitterman.ViewMatrices vm = vm();
        double tau = (Double) TestData.viewsDoc().get("tau");
        double[] p = pi();
        BlackLitterman.Result res = BlackLitterman.blPosterior(
                TestData.sigmaAnnualized(), p, vm.p(), vm.q(), tau, vm.confidences());
        int em = TestData.assetNames().indexOf("EM_EQ");
        // The bullish EM_EQ view (q = 10% > pi) must raise the EM_EQ posterior.
        assertTrue(res.muPost()[em] > p[em]);
        // The relative view must widen the US-EU spread toward q = 2%.
        int us = TestData.assetNames().indexOf("US_EQ");
        int eu = TestData.assetNames().indexOf("EU_EQ");
        double priorSpread = p[us] - p[eu];
        double postSpread = res.muPost()[us] - res.muPost()[eu];
        assertTrue(postSpread > priorSpread);
    }

    @Test
    public void posteriorCovarianceIsValid() {
        BlackLitterman.ViewMatrices vm = vm();
        double tau = (Double) TestData.viewsDoc().get("tau");
        BlackLitterman.Result res = BlackLitterman.blPosterior(
                TestData.sigmaAnnualized(), pi(), vm.p(), vm.q(), tau, vm.confidences());
        // Sigma_post = Sigma + M must remain SPD, and M PSD means the posterior
        // uncertainty shrinks below tau*Sigma on the diagonal.
        Linalg.cholesky(res.sigmaPost());
        for (int i = 0; i < res.m().length; i++) {
            assertTrue("M diagonal nonnegative", res.m()[i][i] >= -1e-15);
            assertTrue("M below tau*Sigma on diagonal",
                    res.m()[i][i] <= tau * TestData.sigmaAnnualized()[i][i] + 1e-15);
        }
    }

    @Test
    public void tauToZeroLimitWithFixedOmega() {
        BlackLitterman.ViewMatrices vm = vm();
        double[][] omega = BlackLitterman.defaultOmega(
                TestData.sigmaAnnualized(), vm.p(), 0.05, vm.confidences());
        double[] p = pi();
        BlackLitterman.Result res = BlackLitterman.blPosterior(
                TestData.sigmaAnnualized(), p, vm.p(), vm.q(), 1e-9, omega, null);
        assertArrayEquals("mu_post -> pi as tau -> 0", p, res.muPost(), 1e-6);
        for (int i = 0; i < res.m().length; i++) {
            for (int j = 0; j < res.m().length; j++) {
                assertEquals("M -> 0", 0.0, res.m()[i][j], 1e-9);
            }
        }
    }

    @Test
    public void omegaToInfinityReturnsPrior() {
        BlackLitterman.ViewMatrices vm = vm();
        double tau = 0.05;
        double[][] omega = BlackLitterman.defaultOmega(
                TestData.sigmaAnnualized(), vm.p(), tau, null);
        for (int i = 0; i < omega.length; i++) {
            omega[i][i] *= 1e12; // zero-confidence views
        }
        double[] p = pi();
        BlackLitterman.Result res = BlackLitterman.blPosterior(
                TestData.sigmaAnnualized(), p, vm.p(), vm.q(), tau, omega, null);
        assertArrayEquals("mu_post -> pi as Omega -> inf", p, res.muPost(), 1e-9);
        // M -> tau * Sigma in the same limit.
        for (int i = 0; i < res.m().length; i++) {
            for (int j = 0; j < res.m().length; j++) {
                assertEquals("M -> tau*Sigma", tau * TestData.sigmaAnnualized()[i][j],
                        res.m()[i][j], 1e-9);
            }
        }
    }

    @Test
    public void zeroConfidenceViaTinyConfidenceReturnsPrior() {
        BlackLitterman.ViewMatrices vm = vm();
        double[] p = pi();
        BlackLitterman.Result res = BlackLitterman.blPosterior(
                TestData.sigmaAnnualized(), p, vm.p(), vm.q(), 0.05,
                new double[] {1e-12, 1e-12});
        assertArrayEquals(p, res.muPost(), 1e-9);
    }

    @Test
    public void conflictingExtremeViewsStayFinite() {
        List<String> names = TestData.assetNames();
        List<BlackLitterman.View> views = List.of(
                BlackLitterman.View.absolute("EM_EQ", 5.0, 100.0),
                BlackLitterman.View.absolute("EM_EQ", -5.0, 100.0));
        BlackLitterman.ViewMatrices vm = BlackLitterman.viewsToMatrices(views, names);
        BlackLitterman.Result res = BlackLitterman.blPosterior(
                TestData.sigmaAnnualized(), pi(), vm.p(), vm.q(), 0.05, vm.confidences());
        for (double v : res.muPost()) {
            assertTrue("posterior stays finite", Double.isFinite(v));
        }
        Linalg.cholesky(res.sigmaPost()); // still SPD
    }

    @Test
    public void blWeightsMatchTangencyOnPosterior() {
        BlackLitterman.ViewMatrices vm = vm();
        double tau = (Double) TestData.viewsDoc().get("tau");
        double[] w = BlackLitterman.blWeights(TestData.sigmaAnnualized(), pi(),
                vm.p(), vm.q(), tau, vm.confidences());
        BlackLitterman.Result res = BlackLitterman.blPosterior(TestData.sigmaAnnualized(),
                pi(), vm.p(), vm.q(), tau, vm.confidences());
        assertArrayEquals(Markowitz.tangencyWeights(res.muPost(), res.sigmaPost(), 0.0), w, 0.0);
        assertEquals(1.0, Linalg.sum(w), 1e-12);
    }

    @Test
    public void errorPaths() {
        List<String> names = TestData.assetNames();
        double[][] sig = TestData.sigmaAnnualized();
        double[] p = pi();
        BlackLitterman.ViewMatrices vm = vm();
        // Empty view list.
        assertThrows(IllegalArgumentException.class,
                () -> BlackLitterman.viewsToMatrices(List.of(), names));
        // Unknown view type.
        assertThrows(IllegalArgumentException.class, () -> BlackLitterman.viewsToMatrices(
                List.of(new BlackLitterman.View("momentum", "US_EQ", null, null, 0.1, 1.0)),
                names));
        // Unknown asset name.
        assertThrows(IllegalArgumentException.class, () -> BlackLitterman.viewsToMatrices(
                List.of(BlackLitterman.View.absolute("NOPE", 0.1, 1.0)), names));
        // Non-positive confidence.
        assertThrows(IllegalArgumentException.class, () -> BlackLitterman.viewsToMatrices(
                List.of(BlackLitterman.View.absolute("US_EQ", 0.1, 0.0)), names));
        // Zero P row.
        assertThrows(IllegalArgumentException.class, () -> BlackLitterman.blPosterior(
                sig, p, new double[][] {new double[8]}, new double[] {0.1}, 0.05, null));
        // Q length mismatch.
        assertThrows(IllegalArgumentException.class, () -> BlackLitterman.blPosterior(
                sig, p, vm.p(), new double[] {0.1}, 0.05, vm.confidences()));
        // tau outside (0, 1].
        assertThrows(IllegalArgumentException.class, () -> BlackLitterman.blPosterior(
                sig, p, vm.p(), vm.q(), 0.0, vm.confidences()));
        assertThrows(IllegalArgumentException.class, () -> BlackLitterman.blPosterior(
                sig, p, vm.p(), vm.q(), 1.5, vm.confidences()));
        // delta <= 0 and negative caps.
        assertThrows(IllegalArgumentException.class, () -> BlackLitterman
                .impliedEquilibriumReturns(sig, TestData.wMkt(), 0.0));
        double[] negCaps = TestData.wMkt().clone();
        negCaps[0] = -0.1;
        assertThrows(IllegalArgumentException.class, () -> BlackLitterman
                .impliedEquilibriumReturns(sig, negCaps, 2.5));
        // Zero cap sum.
        assertThrows(IllegalArgumentException.class, () -> BlackLitterman
                .impliedEquilibriumReturns(sig, new double[8], 2.5));
    }
}
