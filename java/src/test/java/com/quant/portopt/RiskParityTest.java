package com.quant.portopt;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/** Unit and property tests for risk contributions and the CCD solver. */
public class RiskParityTest {

    @Test
    public void riskContributionsSumToOne() {
        // Identity: normalized RCs sum to exactly 1 for any portfolio.
        double[][] sig = TestData.sigmaAnnualized();
        double[][] portfolios = {
                {0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125},
                {0.5, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05},
                {-0.2, 0.3, 0.2, 0.1, 0.1, 0.2, 0.2, 0.1},
        };
        for (double[] w : portfolios) {
            double[] rc = RiskParity.riskContributions(w, sig);
            assertEquals("sum of RCs", 1.0, Linalg.sum(rc), 1e-12);
        }
    }

    @Test
    public void ercContributionsEqual() {
        double[][] sig = TestData.sigmaAnnualized();
        double[] w = RiskParity.ercWeights(sig);
        double[] rc = RiskParity.riskContributions(w, sig);
        for (double v : rc) {
            assertEquals("equal contribution", 1.0 / 8.0, v, 1e-10);
        }
    }

    @Test
    public void riskBudgetTargetsMet() {
        double[][] sig = TestData.sigmaAnnualized();
        double[] budgets = {0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05};
        double[] w = RiskParity.riskBudgetWeights(sig, budgets);
        double[] rc = RiskParity.riskContributions(w, sig);
        for (int i = 0; i < 8; i++) {
            assertEquals("budget " + i, budgets[i], rc[i], 1e-8);
        }
    }

    @Test
    public void budgetsAreNormalizedInternally() {
        double[][] sig = TestData.sigmaAnnualized();
        double[] b1 = {0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05};
        double[] b2 = new double[8];
        for (int i = 0; i < 8; i++) {
            b2[i] = 7.0 * b1[i]; // same proportions, different scale
        }
        assertArrayEquals(RiskParity.riskBudgetWeights(sig, b1),
                RiskParity.riskBudgetWeights(sig, b2), 1e-12);
    }

    @Test
    public void inverseVolHandCase() {
        double[][] sig = {{0.04, 0.0}, {0.0, 0.01}}; // vols 0.2 and 0.1
        double[] w = RiskParity.inverseVolWeights(sig);
        assertEquals(1.0 / 3.0, w[0], 1e-15);
        assertEquals(2.0 / 3.0, w[1], 1e-15);
    }

    @Test
    public void ercOnDiagonalCovEqualsInverseVol() {
        double[][] sig = {
                {0.04, 0.0, 0.0},
                {0.0, 0.09, 0.0},
                {0.0, 0.0, 0.0225},
        };
        assertArrayEquals(RiskParity.inverseVolWeights(sig),
                RiskParity.ercWeights(sig), 1e-10);
    }

    @Test
    public void ercTwoAssetsIsInverseVolRegardlessOfCorrelation() {
        // Analytic 2-asset ERC: w_i proportional to 1/sigma_i for any correlation.
        double[][] sig = {{0.04, 0.012}, {0.012, 0.09}}; // corr = 0.2
        double[] w = RiskParity.ercWeights(sig);
        double[] iv = RiskParity.inverseVolWeights(sig);
        assertArrayEquals(iv, w, 1e-10);
    }

    @Test
    public void ercWithAlmostZeroVolAssetConverges() {
        double[][] sig = {
                {0.04, 0.001, 0.0},
                {0.001, 0.09, 0.0},
                {0.0, 0.0, 1e-8},
        };
        RiskParity.Solution sol = RiskParity.riskBudgetSolve(
                sig, null, RiskParity.CCD_TOL, RiskParity.CCD_MAX_SWEEPS);
        assertTrue("converged", sol.converged());
        // The near-zero-vol asset absorbs almost all weight (correct ERC limit).
        assertTrue("tiny-vol asset dominates", sol.weights()[2] > 0.9);
        double[] rc = RiskParity.riskContributions(sol.weights(), sig);
        for (double v : rc) {
            assertEquals(1.0 / 3.0, v, 1e-6);
        }
    }

    @Test
    public void ccdReportsSweepsAndConverges() {
        RiskParity.Solution sol = RiskParity.riskBudgetSolve(
                TestData.sigmaAnnualized(), null, RiskParity.CCD_TOL, RiskParity.CCD_MAX_SWEEPS);
        assertTrue(sol.converged());
        assertTrue("sweeps within cap", sol.sweeps() >= 1 && sol.sweeps() <= 10_000);
        // A tiny sweep cap flags non-convergence instead of throwing.
        RiskParity.Solution capped = RiskParity.riskBudgetSolve(
                TestData.sigmaAnnualized(), null, 1e-16, 1);
        assertTrue(!capped.converged());
        assertEquals(1, capped.sweeps());
    }

    @Test
    public void errorPaths() {
        double[][] sig = TestData.sigmaAnnualized();
        assertThrows(IllegalArgumentException.class,
                () -> RiskParity.riskBudgetWeights(sig, new double[] {1.0, 1.0})); // bad length
        double[] negBudget = {0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.05, -0.05};
        assertThrows(IllegalArgumentException.class,
                () -> RiskParity.riskBudgetWeights(sig, negBudget));
        // Singular covariance rejected by the Cholesky gate.
        double[][] singular = {{0.04, 0.04}, {0.04, 0.04}};
        assertThrows(IllegalArgumentException.class, () -> RiskParity.ercWeights(singular));
        // Risk contributions: length mismatch and zero-variance portfolio.
        assertThrows(IllegalArgumentException.class,
                () -> RiskParity.riskContributions(new double[] {1.0}, sig));
        assertThrows(IllegalArgumentException.class,
                () -> RiskParity.riskContributions(new double[8], sig));
        // Inverse-vol needs positive variances.
        double[][] zeroVar = {{0.0, 0.0}, {0.0, 0.01}};
        assertThrows(IllegalArgumentException.class,
                () -> RiskParity.inverseVolWeights(zeroVar));
        // tol / sweep validation.
        assertThrows(IllegalArgumentException.class,
                () -> RiskParity.riskBudgetSolve(sig, null, -1.0, 10));
        assertThrows(IllegalArgumentException.class,
                () -> RiskParity.riskBudgetSolve(sig, null, 1e-10, 0));
    }

    @Test
    public void ccdIsDeterministic() {
        double[] w1 = RiskParity.ercWeights(TestData.sigmaAnnualized());
        double[] w2 = RiskParity.ercWeights(TestData.sigmaAnnualized());
        assertArrayEquals("bit-identical repeated runs", w1, w2, 0.0);
    }
}
