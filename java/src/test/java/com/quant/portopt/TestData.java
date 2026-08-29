package com.quant.portopt;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Shared, lazily-loaded test fixtures: the bundled data files and the golden
 * cases. Paths are resolved relative to the working directory, which the
 * build scripts pin to the java/ directory ({@code ../data/...} per the
 * project contract).
 */
final class TestData {

    static final Path DATA_DIR = Paths.get("..", "data");

    private static double[][] returns;
    private static List<String> names;
    private static double[] caps;
    private static Map<String, Map<String, Object>> golden;
    private static Map<String, Object> config;
    private static Map<String, Object> viewsDoc;
    private static double[] muAnn;
    private static double[][] sigAnn;

    private TestData() {
    }

    static synchronized double[][] returns() {
        if (returns == null) {
            returns = DataFiles.loadReturns(DATA_DIR.resolve("asset_returns.csv"));
        }
        return returns;
    }

    static synchronized List<String> assetNames() {
        if (names == null) {
            names = DataFiles.assetNames(DATA_DIR.resolve("asset_returns.csv"));
        }
        return names;
    }

    static synchronized double[] marketCaps() {
        if (caps == null) {
            caps = DataFiles.loadMarketCaps(DATA_DIR.resolve("market_caps.csv"));
        }
        return caps;
    }

    static synchronized Map<String, Object> config() {
        if (config == null) {
            config = MiniJson.parseObjectFile(DATA_DIR.resolve("config.json"));
        }
        return config;
    }

    static synchronized Map<String, Object> viewsDoc() {
        if (viewsDoc == null) {
            viewsDoc = MiniJson.parseObjectFile(DATA_DIR.resolve("views.json"));
        }
        return viewsDoc;
    }

    /** Golden cases keyed by name; each value holds "inputs", "expect", "tol". */
    @SuppressWarnings("unchecked")
    static synchronized Map<String, Map<String, Object>> golden() {
        if (golden == null) {
            Map<String, Object> doc =
                    MiniJson.parseObjectFile(DATA_DIR.resolve("golden").resolve("golden.json"));
            golden = new LinkedHashMap<>();
            for (Object o : (List<Object>) doc.get("cases")) {
                Map<String, Object> c = (Map<String, Object>) o;
                golden.put((String) c.get("name"), c);
            }
        }
        return golden;
    }

    /** Annualized sample mean of the bundled returns (x252). */
    static synchronized double[] muAnnualized() {
        if (muAnn == null) {
            muAnn = Estimators.annualizeMean(Estimators.sampleMean(returns()), 252);
        }
        return muAnn;
    }

    /** Annualized sample covariance of the bundled returns (x252). */
    static synchronized double[][] sigmaAnnualized() {
        if (sigAnn == null) {
            sigAnn = Estimators.annualizeCov(Estimators.sampleCov(returns()), 252);
        }
        return sigAnn;
    }

    @SuppressWarnings("unchecked")
    static Map<String, Object> expect(String caseName) {
        return (Map<String, Object>) golden().get(caseName).get("expect");
    }

    @SuppressWarnings("unchecked")
    static Map<String, Object> inputs(String caseName) {
        return (Map<String, Object>) golden().get(caseName).get("inputs");
    }

    static double tol(String caseName) {
        return (Double) golden().get(caseName).get("tol");
    }

    static double expectScalar(String caseName, String key) {
        return (Double) expect(caseName).get(key);
    }

    static double inputScalar(String caseName, String key) {
        return (Double) inputs(caseName).get(key);
    }

    /** Extracts a flattened vector prefix_0..prefix_(k-1) from a case's expect map. */
    static double[] expectVector(String caseName, String prefix) {
        Map<String, Object> e = expect(caseName);
        List<Double> vals = new ArrayList<>();
        for (int i = 0; e.containsKey(prefix + "_" + i); i++) {
            vals.add((Double) e.get(prefix + "_" + i));
        }
        double[] out = new double[vals.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = vals.get(i);
        }
        return out;
    }

    /** The bundled views as typed View objects. */
    @SuppressWarnings("unchecked")
    static List<BlackLitterman.View> views() {
        List<BlackLitterman.View> out = new ArrayList<>();
        for (Object o : (List<Object>) viewsDoc().get("views")) {
            Map<String, Object> v = (Map<String, Object>) o;
            double q = (Double) v.get("q");
            double conf = v.containsKey("confidence") ? (Double) v.get("confidence") : 1.0;
            if ("absolute".equals(v.get("type"))) {
                out.add(BlackLitterman.View.absolute((String) v.get("asset"), q, conf));
            } else {
                out.add(BlackLitterman.View.relative(
                        (String) v.get("long"), (String) v.get("short"), q, conf));
            }
        }
        return out;
    }

    /** Market-cap weights renormalized to sum 1. */
    static double[] wMkt() {
        double[] c = marketCaps();
        double tot = 0.0;
        for (double v : c) {
            tot += v;
        }
        double[] w = new double[c.length];
        for (int i = 0; i < c.length; i++) {
            w[i] = c[i] / tot;
        }
        return w;
    }
}
