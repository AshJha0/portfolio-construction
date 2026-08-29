package com.quant.portopt;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Loaders for the bundled CSV data files. Package-private: used by the demo
 * and the test suite; the library API itself takes plain arrays.
 */
final class DataFiles {

    private DataFiles() {
    }

    /** Reads all lines of a text file, wrapping I/O failure unchecked. */
    private static List<String> lines(Path path) {
        try {
            return Files.readAllLines(path);
        } catch (IOException e) {
            throw new IllegalStateException("cannot read data file " + path, e);
        }
    }

    /** Asset names from the header row of asset_returns.csv. */
    static List<String> assetNames(Path returnsCsv) {
        return List.of(lines(returnsCsv).get(0).trim().split(","));
    }

    /** The T x N returns matrix of asset_returns.csv (header skipped). */
    static double[][] loadReturns(Path returnsCsv) {
        List<String> all = lines(returnsCsv);
        int t = all.size() - 1;
        double[][] x = new double[t][];
        for (int r = 0; r < t; r++) {
            String[] parts = all.get(r + 1).trim().split(",");
            double[] row = new double[parts.length];
            for (int i = 0; i < parts.length; i++) {
                row[i] = Double.parseDouble(parts[i]);
            }
            x[r] = row;
        }
        return x;
    }

    /** Market caps (second column) of market_caps.csv, in file order. */
    static double[] loadMarketCaps(Path capsCsv) {
        List<String> all = lines(capsCsv);
        List<Double> caps = new ArrayList<>();
        for (int r = 1; r < all.size(); r++) {
            String line = all.get(r).trim();
            if (line.isEmpty()) {
                continue;
            }
            caps.add(Double.parseDouble(line.split(",")[1]));
        }
        double[] out = new double[caps.size()];
        for (int i = 0; i < caps.size(); i++) {
            out[i] = caps.get(i);
        }
        return out;
    }
}
