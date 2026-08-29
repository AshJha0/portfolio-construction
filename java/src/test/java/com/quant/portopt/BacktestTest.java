package com.quant.portopt;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.List;
import org.junit.Test;

/** Unit tests for the pinned backtest schedule, metrics and accounting. */
public class BacktestTest {

    private static Backtest.WeightFunction equalWeight() {
        return h -> {
            double[] w = new double[h[0].length];
            java.util.Arrays.fill(w, 1.0 / h[0].length);
            return w;
        };
    }

    @Test
    public void rebalanceSchedulePinned() {
        List<Integer> dates = Backtest.rebalanceIndices(1250, 500, 63);
        assertEquals(12, dates.size());
        assertEquals(500, (int) dates.get(0));
        assertEquals(563, (int) dates.get(1));
        assertEquals(1193, (int) dates.get(11));
        // Minimal case: exactly one rebalance.
        assertEquals(List.of(500), Backtest.rebalanceIndices(501, 500, 63));
        assertThrows(IllegalArgumentException.class,
                () -> Backtest.rebalanceIndices(500, 500, 63)); // T <= window
        assertThrows(IllegalArgumentException.class,
                () -> Backtest.rebalanceIndices(1250, 1, 63)); // window < 2
        assertThrows(IllegalArgumentException.class,
                () -> Backtest.rebalanceIndices(1250, 500, 0)); // step < 1
    }

    @Test
    public void metricHandValues() {
        double[] r = {0.01, -0.02, 0.03};
        double mean = (0.01 - 0.02 + 0.03) / 3.0;
        assertEquals(252 * mean, Backtest.annualizedReturn(r, 252), 1e-15);
        double var = (Math.pow(0.01 - mean, 2) + Math.pow(-0.02 - mean, 2)
                + Math.pow(0.03 - mean, 2)) / 2.0;
        assertEquals(Math.sqrt(252) * Math.sqrt(var), Backtest.annualizedVol(r, 252), 1e-15);
        double sharpe = (252 * mean - 0.02) / (Math.sqrt(252) * Math.sqrt(var));
        assertEquals(sharpe, Backtest.sharpeRatio(r, 0.02, 252), 1e-14);
    }

    @Test
    public void maxDrawdownHandCase() {
        // Wealth: 1.1, 0.55, 0.605 -> worst drawdown 50% from the 1.1 peak.
        double[] r = {0.1, -0.5, 0.1};
        assertEquals(0.5, Backtest.maxDrawdown(r), 1e-15);
        // Monotone growth has zero drawdown.
        assertEquals(0.0, Backtest.maxDrawdown(new double[] {0.01, 0.02, 0.03}), 0.0);
    }

    @Test
    public void sharpeZeroVolRejected() {
        double[] flat = {0.01, 0.01, 0.01, 0.01};
        assertThrows(IllegalArgumentException.class,
                () -> Backtest.sharpeRatio(flat, 0.0, 252));
    }

    @Test
    public void metricInputValidation() {
        assertThrows(IllegalArgumentException.class,
                () -> Backtest.annualizedReturn(new double[0], 252));
        assertThrows(IllegalArgumentException.class,
                () -> Backtest.annualizedVol(new double[] {0.01}, 252)); // needs 2
        assertThrows(IllegalArgumentException.class,
                () -> Backtest.maxDrawdown(new double[] {0.01, Double.NaN}));
    }

    @Test
    public void accountingIdentity() {
        // The daily portfolio return of a block must be exactly w . r_t with the
        // weight vector active for that block.
        double[][] x = TestData.returns();
        Backtest.Result res = Backtest.runBacktest(x, equalWeight(), 0.02, 500, 63);
        assertEquals(x.length - 500, res.portfolioReturns().length);
        List<Integer> dates = res.rebalanceDates();
        for (int k = 0; k < dates.size(); k++) {
            int tK = dates.get(k);
            int tEnd = (k + 1 < dates.size()) ? dates.get(k + 1) : x.length;
            for (int row : new int[] {tK, tEnd - 1}) {
                // w . r_t, up to the reduction order of the pinned matvec kernel
                double expect = Linalg.dot(res.weights()[k], x[row]);
                assertEquals("row " + row, expect, res.portfolioReturns()[row - 500], 1e-15);
            }
        }
        // Metrics are recomputable from the returned series (identity check).
        assertEquals(Backtest.annualizedReturn(res.portfolioReturns(), 252),
                res.annReturn(), 0.0);
        assertEquals(Backtest.sharpeRatio(res.portfolioReturns(), 0.02, 252),
                res.sharpe(), 0.0);
        assertEquals(Backtest.maxDrawdown(res.portfolioReturns()), res.maxDrawdown(), 0.0);
    }

    @Test
    public void equalWeightHasZeroTurnover() {
        Backtest.Result res = Backtest.runBacktest(TestData.returns(), equalWeight(),
                0.02, 500, 63);
        assertEquals(11, res.turnovers().length);
        for (double to : res.turnovers()) {
            assertEquals(0.0, to, 0.0);
        }
        assertEquals(0.0, res.avgTurnover(), 0.0);
    }

    @Test
    public void expandingWindowIsPassedToWeightFn() {
        int[] calls = {0};
        int[][] seenRows = new int[12][1];
        Backtest.Result res = Backtest.runBacktest(TestData.returns(), h -> {
            seenRows[calls[0]][0] = h.length;
            calls[0]++;
            double[] w = new double[h[0].length];
            java.util.Arrays.fill(w, 1.0 / h[0].length);
            return w;
        }, 0.02, 500, 63);
        assertEquals(12, calls[0]);
        for (int k = 0; k < 12; k++) {
            assertEquals("history length at rebalance " + k,
                    (int) res.rebalanceDates().get(k), seenRows[k][0]);
        }
    }

    @Test
    public void badWeightFunctionsRejected() {
        double[][] x = TestData.returns();
        assertThrows(IllegalArgumentException.class, () -> Backtest.runBacktest(
                x, h -> new double[] {1.0}, 0.0, 500, 63)); // wrong length
        assertThrows(IllegalArgumentException.class, () -> Backtest.runBacktest(
                x, h -> new double[8], 0.0, 500, 63)); // sums to 0
        assertThrows(IllegalArgumentException.class, () -> Backtest.runBacktest(
                x, h -> {
                    double[] w = new double[8];
                    w[0] = Double.NaN;
                    return w;
                }, 0.0, 500, 63)); // NaN
    }

    @Test
    public void backtestIsDeterministic() {
        Backtest.WeightFunction erc = h -> RiskParity.ercWeights(Estimators.sampleCov(h));
        Backtest.Result r1 = Backtest.runBacktest(TestData.returns(), erc, 0.02, 500, 63);
        Backtest.Result r2 = Backtest.runBacktest(TestData.returns(), erc, 0.02, 500, 63);
        assertArrayEquals(r1.portfolioReturns(), r2.portfolioReturns(), 0.0);
        assertEquals(r1.sharpe(), r2.sharpe(), 0.0);
        assertEquals(r1.avgTurnover(), r2.avgTurnover(), 0.0);
        assertTrue(r1.maxDrawdown() == r2.maxDrawdown());
    }
}
