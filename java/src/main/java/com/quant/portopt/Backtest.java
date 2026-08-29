package com.quant.portopt;

import java.util.ArrayList;
import java.util.List;

/**
 * Backtest-lite: quarterly expanding-window rebalance loop and metrics.
 *
 * <p>Pinned schedule (API_SPEC section 6): with T rows of daily returns,
 * estimation window W (default 500) and rebalance step of 63 trading days,
 * rebalance dates are the row indices {@code t_k = W + 63 k} for k = 0, 1, ...
 * while {@code t_k < T}. At each date the weight function sees the expanding
 * history rows [0, t_k) and the resulting weights apply to rows
 * [t_k, min(t_(k+1), T)). Within a holding block the portfolio is rebalanced
 * back to target daily, so the daily portfolio return is exactly {@code w'r_t}.
 *
 * <p>Turnover at rebalance k &ge; 1 (pinned):
 * {@code sum_i |w_k[i] - w_(k-1)[i]|} (two-sided L1 between consecutive target
 * vectors; nothing is charged for the initial position).
 *
 * <p>Metric formulas (pinned): {@code ann_return = 252 * mean(r)} (arithmetic);
 * {@code ann_vol = sqrt(252) * std(r, ddof=1)};
 * {@code sharpe = (ann_return - rf) / ann_vol} (error if vol ~ 0); max
 * drawdown from wealth {@code V_0 = 1, V_t = V_(t-1)(1 + r_t)} with the peak
 * running max including V_0, reported as a positive fraction.
 */
public final class Backtest {

    /** Pinned default estimation window (rows). */
    public static final int DEFAULT_WINDOW = 500;
    /** Pinned default rebalance step (trading days). */
    public static final int DEFAULT_STEP = 63;
    /** Pinned annualization factor. */
    public static final int ANNUALIZATION = 252;

    private Backtest() {
    }

    /** Maps the expanding daily-return history rows [0, t_k) to target weights. */
    @FunctionalInterface
    public interface WeightFunction {
        /**
         * Computes target weights from history.
         *
         * @param history rows [0, t_k) of the returns matrix (do not mutate)
         * @return weight vector of length n summing to 1 within 1e-6
         */
        double[] apply(double[][] history);
    }

    /**
     * Result of {@link #runBacktest}.
     *
     * @param portfolioReturns daily portfolio returns over rows [window, T)
     * @param rebalanceDates   row indices at which weights were recomputed
     * @param weights          target weights per rebalance (K x n)
     * @param turnovers        L1 turnover at each rebalance after the first (K-1)
     * @param annReturn        annualized arithmetic mean return
     * @param annVol           annualized volatility (ddof=1)
     * @param sharpe           annualized Sharpe ratio vs the supplied rf
     * @param maxDrawdown      maximum drawdown (positive fraction)
     * @param avgTurnover      mean of {@code turnovers} (0 if fewer than 2 rebalances)
     */
    public record Result(
            double[] portfolioReturns,
            List<Integer> rebalanceDates,
            double[][] weights,
            double[] turnovers,
            double annReturn,
            double annVol,
            double sharpe,
            double maxDrawdown,
            double avgTurnover) {
    }

    private static double[] validateSeries(double[] r, int minLen) {
        if (r == null || r.length < minLen) {
            throw new IllegalArgumentException(
                    "return series must have at least " + minLen + " observations");
        }
        for (double v : r) {
            if (!Double.isFinite(v)) {
                throw new IllegalArgumentException("return series contains NaN or infinite entries");
            }
        }
        return r;
    }

    /**
     * Arithmetic annualized return {@code periods * mean(r)}.
     *
     * <p>The series sum uses the pinned pairwise reduction of
     * {@link Linalg#sum} — numpy's {@code np.sum} tree — so the metric agrees
     * with the reference bit for bit.
     *
     * @param r       daily return series (&ge; 1 observation)
     * @param periods periods per year
     * @return annualized return
     */
    public static double annualizedReturn(double[] r, int periods) {
        double[] x = validateSeries(r, 1);
        return periods * Linalg.sum(x) / x.length;
    }

    /**
     * Annualized volatility {@code sqrt(periods) * std(r, ddof=1)}.
     *
     * @param r       daily return series (&ge; 2 observations)
     * @param periods periods per year
     * @return annualized volatility
     */
    public static double annualizedVol(double[] r, int periods) {
        double[] x = validateSeries(r, 2);
        double mu = Linalg.sum(x) / x.length;
        double[] sq = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            double d = x[i] - mu;
            sq[i] = d * d;
        }
        double var = Linalg.sum(sq) / (x.length - 1);
        return Math.sqrt(periods) * Math.sqrt(var);
    }

    /**
     * Annualized Sharpe ratio {@code (ann_return - rf) / ann_vol}.
     *
     * @param r       daily return series
     * @param rf      annual risk-free rate
     * @param periods periods per year
     * @return the Sharpe ratio
     * @throws IllegalArgumentException if the annualized volatility is below 1e-15
     */
    public static double sharpeRatio(double[] r, double rf, int periods) {
        double vol = annualizedVol(r, periods);
        if (vol < 1e-15) {
            throw new IllegalArgumentException("Sharpe undefined: portfolio volatility is zero");
        }
        return (annualizedReturn(r, periods) - rf) / vol;
    }

    /**
     * Maximum drawdown of the compounded wealth path (positive fraction).
     *
     * @param r daily return series (&ge; 1 observation)
     * @return the maximum drawdown
     */
    public static double maxDrawdown(double[] r) {
        double[] x = validateSeries(r, 1);
        double wealth = 1.0;
        double peak = 1.0;
        double mdd = 0.0;
        for (double ret : x) {
            wealth *= 1.0 + ret;
            if (wealth > peak) {
                peak = wealth;
            }
            double dd = 1.0 - wealth / peak;
            if (dd > mdd) {
                mdd = dd;
            }
        }
        return mdd;
    }

    /**
     * Pinned rebalance dates {@code window, window+step, ...} strictly below t.
     *
     * @param t      sample length (must exceed the window)
     * @param window estimation window (&ge; 2)
     * @param step   rebalance step (&ge; 1)
     * @return the rebalance row indices
     */
    public static List<Integer> rebalanceIndices(int t, int window, int step) {
        if (window < 2) {
            throw new IllegalArgumentException("estimation window must be >= 2");
        }
        if (step < 1) {
            throw new IllegalArgumentException("rebalance step must be >= 1");
        }
        if (t <= window) {
            throw new IllegalArgumentException(
                    "sample length " + t + " must exceed the estimation window " + window);
        }
        List<Integer> dates = new ArrayList<>();
        for (int idx = window; idx < t; idx += step) {
            dates.add(idx);
        }
        return dates;
    }

    /**
     * Quarterly expanding-window backtest with the pinned schedule.
     *
     * @param returns  T x n daily returns matrix
     * @param weightFn maps the expanding history rows [0, t_k) to target
     *                 weights of length n summing to 1 within 1e-6 (validated)
     * @param rf       annual risk-free rate used for the Sharpe ratio
     * @param window   estimation window
     * @param step     rebalance step
     * @return portfolio series, per-rebalance weights, turnover and metrics
     * @throws IllegalArgumentException on invalid input or an invalid weight
     *         vector from {@code weightFn}
     */
    public static Result runBacktest(double[][] returns, WeightFunction weightFn, double rf,
            int window, int step) {
        double[][] x = Estimators.validateReturns(returns, 3);
        int t = x.length;
        int n = x[0].length;
        List<Integer> dates = rebalanceIndices(t, window, step);
        int k = dates.size();
        double[][] weights = new double[k][];
        double[] port = new double[t - window];
        for (int idx = 0; idx < k; idx++) {
            int tK = dates.get(idx);
            double[][] history = new double[tK][];
            System.arraycopy(x, 0, history, 0, tK);
            double[] w = weightFn.apply(history);
            if (w == null || w.length != n) {
                throw new IllegalArgumentException(
                        "weight_fn returned an invalid weight vector at t=" + tK);
            }
            for (double v : w) {
                if (!Double.isFinite(v)) {
                    throw new IllegalArgumentException(
                            "weight_fn returned an invalid weight vector at t=" + tK);
                }
            }
            double wSum = Linalg.sum(w);
            if (Math.abs(wSum - 1.0) > 1e-6) {
                throw new IllegalArgumentException(String.format(
                        "weight_fn weights must sum to 1 (got %.8f) at t=%d", wSum, tK));
            }
            weights[idx] = w.clone();
            int tEnd = (idx + 1 < k) ? dates.get(idx + 1) : t;
            for (int row = tK; row < tEnd; row++) {
                // block @ w row by row — the same matvec reduction as the reference
                port[row - window] = Linalg.rowDot(x[row], w);
            }
        }
        double[] turnovers = new double[Math.max(0, k - 1)];
        double[] absDiff = new double[n];
        for (int i = 1; i < k; i++) {
            for (int j = 0; j < n; j++) {
                absDiff[j] = Math.abs(weights[i][j] - weights[i - 1][j]);
            }
            turnovers[i - 1] = Linalg.sum(absDiff);
        }
        double avgTo = turnovers.length > 0 ? Linalg.sum(turnovers) / turnovers.length : 0.0;
        return new Result(
                port,
                dates,
                weights,
                turnovers,
                annualizedReturn(port, ANNUALIZATION),
                annualizedVol(port, ANNUALIZATION),
                sharpeRatio(port, rf, ANNUALIZATION),
                maxDrawdown(port),
                avgTo);
    }
}
