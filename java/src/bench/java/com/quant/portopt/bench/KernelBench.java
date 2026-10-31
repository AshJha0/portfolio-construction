package com.quant.portopt.bench;

import com.quant.portopt.Backtest;
import com.quant.portopt.Estimators;
import com.quant.portopt.Linalg;
import com.quant.portopt.Markowitz;
import com.quant.portopt.RiskParity;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

/**
 * JMH micro-benchmarks of the portopt kernels.
 *
 * <p>The problems are the ones every port builds (docs/results/benchmarks.md):
 * SplitMix64 with seed 20261031, a T = 1250 one-factor return panel,
 * sigma = 252 * sampleCov(returns), b = ones(n), and an equal-weight backtest
 * with window 500 and step 63, at n = 8, 20, 50.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 5, time = 2)
@Measurement(iterations = 5, time = 1)
@Fork(1)
public class KernelBench {

    static final long SEED = 20261031L;
    static final int ROWS = 1250;
    static final int WINDOW = 500;
    static final int STEP = 63;

    /** Number of assets. */
    @Param({"8", "20", "50"})
    public int n;

    double[][] returns;
    double[][] sigma;
    double[] ones;
    double[] equal;

    /** SplitMix64 (Steele, Lea and Flood 2014); identical in every port. */
    static final class SplitMix64 {
        private long state;

        SplitMix64(long seed) {
            state = seed;
        }

        long next() {
            state += 0x9E3779B97F4A7C15L;
            long z = state;
            z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
            return z ^ (z >>> 31);
        }

        /** Uniform on [0, 1) with 53 random bits. */
        double uniform() {
            return (next() >>> 11) * 0x1.0p-53;
        }
    }

    /**
     * x[t][i] = 0.02 (1 + 0.25 (i mod 5)) (0.6 f_t + e_ti), f_t and e_ti
     * uniform on [-0.5, 0.5), drawn row by row (f_t first).
     */
    static double[][] syntheticReturns(int n) {
        SplitMix64 rng = new SplitMix64(SEED);
        double[][] x = new double[ROWS][n];
        for (int t = 0; t < ROWS; t++) {
            double f = rng.uniform() - 0.5;
            for (int i = 0; i < n; i++) {
                double e = rng.uniform() - 0.5;
                x[t][i] = 0.02 * (1.0 + 0.25 * (i % 5)) * (0.6 * f + e);
            }
        }
        return x;
    }

    /** Builds the problem for the current n (outside the timed region). */
    @Setup(Level.Trial)
    public void setUp() {
        returns = syntheticReturns(n);
        sigma = Estimators.annualizeCov(Estimators.sampleCov(returns), 252);
        ones = new double[n];
        Arrays.fill(ones, 1.0);
        equal = new double[n];
        Arrays.fill(equal, 1.0 / n);
    }

    /** Cholesky factor plus the two triangular solves. */
    @Benchmark
    public double[] choleskySolve() {
        return Linalg.cholSolve(Linalg.cholesky(sigma), ones);
    }

    /** Ledoit-Wolf constant-correlation shrinkage on the T x n panel. */
    @Benchmark
    public Estimators.ShrinkageResult shrinkCovariance() {
        return Estimators.shrinkCovariance(returns);
    }

    /** Long-only minimum-variance active-set QP. */
    @Benchmark
    public double[] minVarianceLongOnly() {
        return Markowitz.minVarianceLongOnly(sigma);
    }

    /** Equal risk contribution weights by cyclical coordinate descent. */
    @Benchmark
    public double[] ercWeights() {
        return RiskParity.ercWeights(sigma);
    }

    /** Equal-weight walk-forward backtest (history copy plus accounting). */
    @Benchmark
    public Backtest.Result runBacktest() {
        return Backtest.runBacktest(returns, h -> equal, 0.0, WINDOW, STEP);
    }
}
