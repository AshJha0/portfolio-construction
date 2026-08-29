package com.quant.portopt;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

import org.junit.Test;

/** Unit tests for the pinned Cholesky / solve / inverse kernels. */
public class LinalgTest {

    private static final double[][] SPD = {
            {4.0, 2.0, 0.6},
            {2.0, 5.0, 1.5},
            {0.6, 1.5, 3.0},
    };

    @Test
    public void choleskyReconstructsMatrix() {
        double[][] low = Linalg.cholesky(SPD);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                double s = 0.0;
                for (int k = 0; k < 3; k++) {
                    s += low[i][k] * low[j][k];
                }
                assertEquals("LL' entry " + i + "," + j, SPD[i][j], s, 1e-14);
            }
        }
        // Property-style check on the bundled annualized covariance too.
        double[][] sig = TestData.sigmaAnnualized();
        double[][] l2 = Linalg.cholesky(sig);
        for (int i = 0; i < sig.length; i++) {
            for (int j = 0; j < sig.length; j++) {
                double s = 0.0;
                for (int k = 0; k < sig.length; k++) {
                    s += l2[i][k] * l2[j][k];
                }
                assertEquals(sig[i][j], s, 1e-14);
            }
        }
    }

    @Test
    public void cholSolveSolvesSystem() {
        double[] b = {1.0, -2.0, 0.5};
        double[] x = Linalg.spdSolve(SPD, b);
        double[] ax = Linalg.matVec(SPD, x);
        for (int i = 0; i < 3; i++) {
            assertEquals("residual row " + i, b[i], ax[i], 1e-13);
        }
    }

    @Test
    public void spdInverseTimesMatrixIsIdentity() {
        double[][] inv = Linalg.spdInverse(SPD);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                double s = 0.0;
                for (int k = 0; k < 3; k++) {
                    s += SPD[i][k] * inv[k][j];
                }
                assertEquals("identity entry " + i + "," + j, i == j ? 1.0 : 0.0, s, 1e-13);
            }
        }
    }

    @Test
    public void choleskyRejectsIndefinite() {
        double[][] indefinite = {{1.0, 2.0}, {2.0, 1.0}};
        assertThrows(IllegalArgumentException.class, () -> Linalg.cholesky(indefinite));
        double[][] negDiag = {{-1.0, 0.0}, {0.0, 1.0}};
        assertThrows(IllegalArgumentException.class, () -> Linalg.cholesky(negDiag));
    }

    @Test
    public void choleskyRejectsSingularDuplicatedAsset() {
        // Covariance of a returns matrix with a duplicated column is exactly
        // singular; the pivot becomes a rounding residual caught by the
        // relative threshold.
        double[][] x = TestData.returns();
        double[][] dup = new double[x.length][9];
        for (int t = 0; t < x.length; t++) {
            System.arraycopy(x[t], 0, dup[t], 0, 8);
            dup[t][8] = x[t][0];
        }
        double[][] cov = Estimators.sampleCov(dup);
        assertThrows(IllegalArgumentException.class, () -> Linalg.cholesky(cov));
    }

    @Test
    public void choleskyRejectsAsymmetricAndNonSquare() {
        double[][] asym = {{1.0, 0.5}, {0.2, 1.0}};
        assertThrows(IllegalArgumentException.class, () -> Linalg.cholesky(asym));
        double[][] rect = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
        assertThrows(IllegalArgumentException.class, () -> Linalg.cholesky(rect));
        assertThrows(IllegalArgumentException.class, () -> Linalg.cholesky(new double[0][0]));
    }

    @Test
    public void choleskyRejectsNanAndInf() {
        double[][] nan = {{1.0, Double.NaN}, {Double.NaN, 1.0}};
        assertThrows(IllegalArgumentException.class, () -> Linalg.cholesky(nan));
        double[][] inf = {{1.0, 0.0}, {0.0, Double.POSITIVE_INFINITY}};
        assertThrows(IllegalArgumentException.class, () -> Linalg.cholesky(inf));
    }

    @Test
    public void cholSolveRejectsBadRhsLength() {
        double[][] low = Linalg.cholesky(SPD);
        assertThrows(IllegalArgumentException.class,
                () -> Linalg.cholSolve(low, new double[] {1.0, 2.0}));
    }
}
