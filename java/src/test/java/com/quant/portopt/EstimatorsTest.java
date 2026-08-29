package com.quant.portopt;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/** Unit tests for sample moments and Ledoit-Wolf shrinkage. */
public class EstimatorsTest {

    private static final double[][] SMALL = {
            {0.01, -0.02},
            {0.03, 0.01},
            {-0.01, 0.00},
            {0.02, 0.02},
    };

    @Test
    public void sampleMomentsMatchHandComputation() {
        double[] mu = Estimators.sampleMean(SMALL);
        assertEquals(0.0125, mu[0], 1e-15);
        assertEquals(0.0025, mu[1], 1e-15);
        double[][] cov = Estimators.sampleCov(SMALL);
        // ddof=1 by hand: sum of demeaned cross products / 3.
        double s00 = (Math.pow(-0.0025, 2) + Math.pow(0.0175, 2)
                + Math.pow(-0.0225, 2) + Math.pow(0.0075, 2)) / 3.0;
        assertEquals(s00, cov[0][0], 1e-18);
        double s01 = (-0.0025 * -0.0225 + 0.0175 * 0.0075
                + -0.0225 * -0.0025 + 0.0075 * 0.0175) / 3.0;
        assertEquals(s01, cov[0][1], 1e-18);
        assertEquals(cov[0][1], cov[1][0], 0.0);
    }

    @Test
    public void shrinkageIntensityInUnitInterval() {
        Estimators.ShrinkageResult res = Estimators.shrinkCovariance(TestData.returns());
        assertTrue("intensity >= 0", res.intensity() >= 0.0);
        assertTrue("intensity <= 1", res.intensity() <= 1.0);
        assertTrue("rbar within (-1, 1)", Math.abs(res.rbar()) < 1.0);
    }

    @Test
    public void shrunkMatrixIsPsd() {
        Estimators.ShrinkageResult res = Estimators.shrinkCovariance(TestData.returns());
        Linalg.cholesky(res.sigma()); // must not throw
    }

    @Test
    public void shrinkageIsConvexCombinationOfSampleAndTarget() {
        Estimators.ShrinkageResult res = Estimators.shrinkCovariance(TestData.returns());
        double d = res.intensity();
        int n = res.sigma().length;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double expect = d * res.target()[i][j] + (1.0 - d) * res.sample()[i][j];
                assertEquals(expect, res.sigma()[i][j], 1e-18);
                // Diagonal is preserved exactly by the constant-correlation target.
                if (i == j) {
                    assertEquals(res.sample()[i][i], res.sigma()[i][i], 1e-18);
                }
            }
        }
    }

    @Test
    public void shrinkagePullsCorrelationsTowardRbar() {
        Estimators.ShrinkageResult res = Estimators.shrinkCovariance(TestData.returns());
        int n = res.sigma().length;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double sampleCorr = res.sample()[i][j]
                        / Math.sqrt(res.sample()[i][i] * res.sample()[j][j]);
                double shrunkCorr = res.sigma()[i][j]
                        / Math.sqrt(res.sigma()[i][i] * res.sigma()[j][j]);
                double distBefore = Math.abs(sampleCorr - res.rbar());
                double distAfter = Math.abs(shrunkCorr - res.rbar());
                assertTrue("pair " + i + "," + j + " moved toward rbar",
                        distAfter <= distBefore + 1e-15);
            }
        }
    }

    @Test
    public void annualizationScales() {
        double[] mu = {0.001, -0.002};
        double[] muAnn = Estimators.annualizeMean(mu, 252);
        assertEquals(0.252, muAnn[0], 1e-15);
        assertEquals(-0.504, muAnn[1], 1e-15);
        double[][] cov = {{1e-4, 2e-5}, {2e-5, 3e-4}};
        double[][] covAnn = Estimators.annualizeCov(cov, 252);
        assertEquals(252e-4, covAnn[0][0], 1e-15);
        assertEquals(252 * 2e-5, covAnn[1][0], 1e-15);
    }

    @Test
    public void inputValidationErrors() {
        assertThrows(IllegalArgumentException.class,
                () -> Estimators.sampleMean(new double[0][0]));
        assertThrows(IllegalArgumentException.class,
                () -> Estimators.sampleCov(new double[][] {{0.01, 0.02}})); // T < 2
        assertThrows(IllegalArgumentException.class,
                () -> Estimators.sampleMean(new double[][] {{0.01, Double.NaN}}));
        assertThrows(IllegalArgumentException.class,
                () -> Estimators.annualizeMean(new double[] {0.01}, 0));
        assertThrows(IllegalArgumentException.class,
                () -> Estimators.annualizeCov(new double[][] {{1.0}}, -1));
        // Shrinkage needs at least 2 assets ...
        assertThrows(IllegalArgumentException.class,
                () -> Estimators.shrinkCovariance(new double[][] {{0.01}, {0.02}, {0.03}}));
        // ... and strictly positive variances (constant column has zero variance).
        assertThrows(IllegalArgumentException.class,
                () -> Estimators.shrinkCovariance(new double[][] {
                        {0.01, 5.0}, {0.02, 5.0}, {-0.01, 5.0}}));
        // Ragged input is rejected.
        assertThrows(IllegalArgumentException.class,
                () -> Estimators.sampleMean(new double[][] {{0.01, 0.02}, {0.01}}));
    }
}
