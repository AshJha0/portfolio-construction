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
        // A simple return <= -1 is not a valid price move.
        assertThrows(IllegalArgumentException.class,
                () -> Estimators.sampleMean(new double[][] {{0.01, -1.0}, {0.02, 0.01}}));
        // Non-finite moments cannot be annualized.
        assertThrows(IllegalArgumentException.class,
                () -> Estimators.annualizeMean(new double[] {Double.NaN}, 252));
    }

    /**
     * Direct transcription of Ledoit and Wolf's covCor.m (JPM 2004, App. B):
     * divisor T throughout, both theta terms per ordered pair. Returns {delta, rbar}.
     */
    private static double[] covcor(double[][] x) {
        int t = x.length;
        int n = x[0].length;
        double tf = t;
        double[][] xc = new double[n][t];
        for (int j = 0; j < n; j++) {
            double m = 0.0;
            for (int r = 0; r < t; r++) {
                m += x[r][j];
            }
            m /= tf;
            for (int r = 0; r < t; r++) {
                xc[j][r] = x[r][j] - m;
            }
        }
        double[][] sample = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double acc = 0.0;
                for (int r = 0; r < t; r++) {
                    acc += xc[i][r] * xc[j][r];
                }
                sample[i][j] = acc / tf;
            }
        }
        double[] sqrtvar = new double[n];
        for (int i = 0; i < n; i++) {
            sqrtvar[i] = Math.sqrt(sample[i][i]);
        }
        double corr = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                corr += sample[i][j] / (sqrtvar[i] * sqrtvar[j]);
            }
        }
        double rbar = (corr - n) / ((double) n * (n - 1));
        double phi = 0.0;
        double rho = 0.0;
        double gamma = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double phiIj = 0.0;
                double thetaIj = 0.0;
                for (int r = 0; r < t; r++) {
                    double d = xc[i][r] * xc[j][r] - sample[i][j];
                    phiIj += d * d;
                    thetaIj += (xc[i][r] * xc[i][r] - sample[i][i]) * d;
                }
                phi += phiIj / tf;
                double prior = (i == j) ? sample[i][i] : rbar * sqrtvar[i] * sqrtvar[j];
                gamma += (sample[i][j] - prior) * (sample[i][j] - prior);
                if (i == j) {
                    rho += phiIj / tf;
                } else {
                    rho += rbar * (sqrtvar[j] / sqrtvar[i]) * thetaIj / tf;
                }
            }
        }
        double kappa = (phi - rho) / gamma;
        return new double[] {Math.max(0.0, Math.min(1.0, kappa / tf)), rbar};
    }

    private static final double[][] HAND12 = {
            {0.010, 0.020, -0.010}, {-0.020, 0.000, 0.010}, {0.030, -0.010, 0.020},
            {0.005, 0.040, -0.030}, {-0.015, -0.030, 0.000}, {0.025, 0.010, 0.015},
            {-0.010, 0.050, -0.020}, {0.000, -0.020, 0.025}, {0.020, 0.030, 0.005},
            {-0.030, -0.040, -0.010}, {0.015, 0.000, 0.030}, {0.010, 0.060, -0.005},
    };

    @Test
    public void lwRhoMatchesPaperFormula() {
        // PT-1 (CRIT-1): 12x3 hand panel with unequal vols so that
        // theta_ii,ij != theta_jj,ij and the old halved rho_hat would be wrong.
        double[] scale = {1.0, 3.0, 0.5};
        double[][] x = new double[12][3];
        for (int r = 0; r < 12; r++) {
            for (int c = 0; c < 3; c++) {
                x[r][c] = HAND12[r][c] * scale[c];
            }
        }
        double[] paper = covcor(x);
        Estimators.ShrinkageResult res = Estimators.shrinkCovariance(x);
        assertEquals(paper[0], res.intensity(), 1e-12);
        assertEquals(paper[1], res.rbar(), 1e-12);
        // n = 2: target == sample (gamma_hat = 0) -> delta = 0, sigma == sample.
        double[][] x2 = new double[12][2];
        for (int r = 0; r < 12; r++) {
            x2[r][0] = HAND12[r][0];
            x2[r][1] = HAND12[r][1];
        }
        Estimators.ShrinkageResult res2 = Estimators.shrinkCovariance(x2);
        assertEquals(0.0, res2.intensity(), 0.0);
        assertEquals(res2.sample()[0][1], res2.sigma()[0][1], 0.0);
    }

    @Test
    public void lwBundledDataMatchesPaper() {
        double[] paper = covcor(TestData.returns());
        Estimators.ShrinkageResult res = Estimators.shrinkCovariance(TestData.returns());
        assertEquals(paper[0], res.intensity(), 1e-13);
        assertEquals(0.00721176522548, res.intensity(), 1e-12); // paper value, not 0.009430
        assertEquals(paper[1], res.rbar(), 1e-13);
    }

    @Test
    public void lwIntensitySmallTRepairsSingularSample() {
        // PT-2: T = 5 <= N = 8 — sample singular, shrunk SPD, 0 < delta <= 1.
        double[][] head = new double[5][];
        for (int r = 0; r < 5; r++) {
            head[r] = TestData.returns()[r].clone();
        }
        Estimators.ShrinkageResult res = Estimators.shrinkCovariance(head);
        assertTrue(res.intensity() > 0.0 && res.intensity() <= 1.0);
        assertThrows(IllegalArgumentException.class, () -> Linalg.cholesky(res.sample()));
        Linalg.cholesky(res.sigma()); // must not throw
        assertEquals(covcor(head)[0], res.intensity(), 1e-12);
    }

    @Test
    public void shrinkageZeroVarianceRejected() {
        // PT-9: a constant column has zero sample variance.
        double[][] x = new double[10][2];
        for (int r = 0; r < 10; r++) {
            x[r][0] = -0.01 + 0.02 * r / 9.0;
            x[r][1] = 0.004;
        }
        assertThrows(IllegalArgumentException.class, () -> Estimators.shrinkCovariance(x));
    }
}
