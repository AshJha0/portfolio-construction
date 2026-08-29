package com.quant.portopt;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * portopt end-to-end demo.
 *
 * <p>Run from the {@code java/} directory: {@code ./demo.sh} (data files are
 * resolved at {@code ../data} relative to the working directory; pass a
 * different data directory as the first argument if needed).
 *
 * <p>Prints: estimator table, unconstrained + long-only frontier sweep,
 * Black-Litterman prior vs posterior weights, ERC vs inverse-vol risk
 * contributions, and a quarterly-rebalance comparison across six methods.
 */
public final class Demo {

    private Demo() {
    }

    private static void rule(String title) {
        StringBuilder sb = new StringBuilder("\n=== ").append(title).append(' ');
        while (sb.length() < 72) {
            sb.append('=');
        }
        System.out.println(sb);
    }

    private static String wrow(String label, double[] w) {
        StringBuilder sb = new StringBuilder(String.format("%-14s", label));
        for (double x : w) {
            sb.append(String.format("%9.4f", x));
        }
        return sb.toString();
    }

    /**
     * Entry point.
     *
     * @param args optional: data directory (default {@code ../data})
     */
    public static void main(String[] args) {
        Path data = Paths.get(args.length > 0 ? args[0] : "../data");
        double[][] x = DataFiles.loadReturns(data.resolve("asset_returns.csv"));
        List<String> names = DataFiles.assetNames(data.resolve("asset_returns.csv"));
        double[] caps = DataFiles.loadMarketCaps(data.resolve("market_caps.csv"));
        Map<String, Object> cfg = MiniJson.parseObjectFile(data.resolve("config.json"));
        Map<String, Object> viewsDoc = MiniJson.parseObjectFile(data.resolve("views.json"));
        double rf = (Double) cfg.get("risk_free_rate");
        int annualization = (int) (double) (Double) cfg.get("annualization_factor");
        int frontierPoints = (int) (double) (Double) cfg.get("frontier_points");
        int window = (int) (double) (Double) cfg.get("estimation_window");
        int step = (int) (double) (Double) cfg.get("rebalance_step");
        double tau = (Double) viewsDoc.get("tau");
        double delta = (Double) viewsDoc.get("delta");

        System.out.printf("portopt demo - %d-asset universe, %d daily returns%n",
                names.size(), x.length);
        System.out.println("assets: " + String.join(", ", names));

        // ----------------------------------------------------------- estimators
        rule("Estimators (annualized, Ledoit-Wolf constant-correlation shrinkage)");
        double[] mu = Estimators.annualizeMean(Estimators.sampleMean(x), annualization);
        Estimators.ShrinkageResult shrink = Estimators.shrinkCovariance(x);
        double[][] sig = Estimators.annualizeCov(Estimators.sampleCov(x), annualization);
        double[][] sigSh = Estimators.annualizeCov(shrink.sigma(), annualization);
        System.out.printf("shrinkage intensity delta = %.6f   rbar = %.4f%n",
                shrink.intensity(), shrink.rbar());
        System.out.printf("%-10s%9s%13s%13s%n", "asset", "mean", "vol(sample)", "vol(shrunk)");
        for (int i = 0; i < names.size(); i++) {
            System.out.printf("%-10s%9.4f%13.4f%13.4f%n", names.get(i), mu[i],
                    Math.sqrt(sig[i][i]), Math.sqrt(sigSh[i][i]));
        }

        // ------------------------------------------------------------- frontier
        rule("Efficient frontier sweep (25 points, showing every 4th)");
        Markowitz.Frontier front = Markowitz.efficientFrontier(mu, sig, frontierPoints);
        Markowitz.Frontier frontC = Markowitz.constrainedFrontier(mu, sig, frontierPoints);
        System.out.printf("%8s%11s%15s%n", "target", "vol uncon", "vol long-only");
        for (int k = 0; k < front.targets().length; k += 4) {
            System.out.printf("%8.4f%11.4f%15.4f%n", front.targets()[k],
                    Math.sqrt(front.variances()[k]), Math.sqrt(frontC.variances()[k]));
        }
        double[] wTan = Markowitz.tangencyWeights(mu, sig, rf);
        double tanSharpe = (Linalg.dot(mu, wTan) - rf) / Math.sqrt(Linalg.quadForm(wTan, sig));
        System.out.printf("tangency (rf=%.2f%%): sharpe = %.4f%n", 100 * rf, tanSharpe);

        // ------------------------------------------------------ black-litterman
        rule("Black-Litterman: prior (equilibrium) vs posterior weights");
        double capSum = Linalg.sum(caps);
        double[] wMkt = new double[caps.length];
        for (int i = 0; i < caps.length; i++) {
            wMkt[i] = caps[i] / capSum;
        }
        double[] pi = BlackLitterman.impliedEquilibriumReturns(sig, wMkt, delta);
        List<BlackLitterman.View> views = parseViews(viewsDoc);
        for (BlackLitterman.View v : views) {
            if ("absolute".equals(v.type())) {
                System.out.printf("view: %s returns %.1f%% (confidence %s)%n",
                        v.asset(), 100 * v.q(), fmtConf(v.confidence()));
            } else {
                System.out.printf("view: %s beats %s by %.1f%% (confidence %s)%n",
                        v.longAsset(), v.shortAsset(), 100 * v.q(), fmtConf(v.confidence()));
            }
        }
        BlackLitterman.ViewMatrices vm = BlackLitterman.viewsToMatrices(views, names);
        BlackLitterman.Result res =
                BlackLitterman.blPosterior(sig, pi, vm.p(), vm.q(), tau, vm.confidences());
        double[] wBl = Markowitz.tangencyWeights(res.muPost(), res.sigmaPost(), 0.0);
        StringBuilder hdr = new StringBuilder(String.format("%-14s", ""));
        for (String n : names) {
            hdr.append(String.format("%9s", n.substring(0, Math.min(8, n.length()))));
        }
        System.out.println(hdr);
        System.out.println(wrow("pi (equil.)", pi));
        System.out.println(wrow("mu posterior", res.muPost()));
        System.out.println(wrow("w market", wMkt));
        System.out.println(wrow("w posterior", wBl));

        // ---------------------------------------------------------- risk parity
        rule("Risk contributions: ERC vs inverse-vol");
        double[] wErc = RiskParity.ercWeights(sig);
        double[] wIv = RiskParity.inverseVolWeights(sig);
        double[] rcErc = RiskParity.riskContributions(wErc, sig);
        double[] rcIv = RiskParity.riskContributions(wIv, sig);
        System.out.println(hdr);
        System.out.println(wrow("w ERC", wErc));
        System.out.println(wrow("RC ERC", rcErc));
        System.out.println(wrow("w inv-vol", wIv));
        System.out.println(wrow("RC inv-vol", rcIv));
        System.out.printf("RC spread: ERC %.2e, inv-vol %.2e%n", spread(rcErc), spread(rcIv));

        // ------------------------------------------------------------- backtest
        rule("Quarterly-rebalance backtest, expanding window (holdout = last 750d)");
        int[] fallbacks = {0};
        Map<String, Backtest.WeightFunction> methods = new LinkedHashMap<>();
        methods.put("equal_weight", h -> {
            double[] w = new double[h[0].length];
            java.util.Arrays.fill(w, 1.0 / h[0].length);
            return w;
        });
        methods.put("min_var_lo", h -> Markowitz.minVarianceLongOnly(
                Estimators.annualizeCov(Estimators.sampleCov(h), annualization)));
        methods.put("tangency", h -> {
            // Pinned backtest method: tangency, falling back to unconstrained
            // min-variance where tangency is undefined (A - rf*C <= 0) —
            // reported, not crashed (API_SPEC 6.3).
            double[][] s = Estimators.annualizeCov(Estimators.sampleCov(h), annualization);
            try {
                return Markowitz.tangencyWeights(
                        Estimators.annualizeMean(Estimators.sampleMean(h), annualization), s, rf);
            } catch (IllegalArgumentException e) {
                fallbacks[0]++;
                return Markowitz.minVarianceWeights(s);
            }
        });
        methods.put("black_litt", h -> {
            double[][] s = Estimators.annualizeCov(Estimators.sampleCov(h), annualization);
            double[] piH = BlackLitterman.impliedEquilibriumReturns(s, wMkt, delta);
            return BlackLitterman.blWeights(s, piH, vm.p(), vm.q(), tau, vm.confidences());
        });
        methods.put("erc", h -> RiskParity.ercWeights(Estimators.sampleCov(h)));
        methods.put("inverse_vol", h -> RiskParity.inverseVolWeights(Estimators.sampleCov(h)));

        System.out.printf("%-14s%9s%9s%9s%9s%11s%n",
                "method", "ann ret", "ann vol", "sharpe", "max DD", "avg turno");
        for (Map.Entry<String, Backtest.WeightFunction> e : methods.entrySet()) {
            Backtest.Result r = Backtest.runBacktest(x, e.getValue(), rf, window, step);
            System.out.printf("%-14s%9.4f%9.4f%9.4f%9.4f%11.4f%n", e.getKey(),
                    r.annReturn(), r.annVol(), r.sharpe(), r.maxDrawdown(), r.avgTurnover());
        }
        if (fallbacks[0] > 0) {
            System.out.printf("note: tangency undefined at %d rebalance date(s) "
                    + "(rf above min-var return); fell back to min-variance there.%n",
                    fallbacks[0]);
        }

        System.out.println("\ndemo complete.");
    }

    private static String fmtConf(double c) {
        return (c == Math.floor(c)) ? String.format("%.1f", c) : String.valueOf(c);
    }

    private static double spread(double[] v) {
        double lo = v[0];
        double hi = v[0];
        for (double x : v) {
            lo = Math.min(lo, x);
            hi = Math.max(hi, x);
        }
        return hi - lo;
    }

    @SuppressWarnings("unchecked")
    private static List<BlackLitterman.View> parseViews(Map<String, Object> viewsDoc) {
        List<BlackLitterman.View> out = new ArrayList<>();
        for (Object o : (List<Object>) viewsDoc.get("views")) {
            Map<String, Object> v = (Map<String, Object>) o;
            String type = (String) v.get("type");
            double q = (Double) v.get("q");
            double conf = v.containsKey("confidence") ? (Double) v.get("confidence") : 1.0;
            if ("absolute".equals(type)) {
                out.add(BlackLitterman.View.absolute((String) v.get("asset"), q, conf));
            } else {
                out.add(BlackLitterman.View.relative(
                        (String) v.get("long"), (String) v.get("short"), q, conf));
            }
        }
        return out;
    }
}
