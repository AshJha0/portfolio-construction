package com.quant.portopt;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Minimal recursive-descent JSON parser for the bundled data files
 * (config.json, views.json, golden.json). Package-private: it exists only so
 * the demo and the test suite can read the project's flat JSON documents
 * without an external dependency (Maven Central is unavailable in this build).
 *
 * <p>Values map to Java as: object -&gt; {@code Map<String, Object>}
 * (insertion-ordered), array -&gt; {@code List<Object>}, string -&gt;
 * {@code String}, number -&gt; {@code Double}, true/false -&gt;
 * {@code Boolean}, null -&gt; {@code null}.
 */
final class MiniJson {

    private final String src;
    private int pos;

    private MiniJson(String src) {
        this.src = src;
        this.pos = 0;
    }

    /** Parses a JSON document from a string. */
    static Object parse(String text) {
        MiniJson p = new MiniJson(text);
        p.skipWs();
        Object v = p.parseValue();
        p.skipWs();
        if (p.pos != text.length()) {
            throw new IllegalArgumentException("trailing characters after JSON value at " + p.pos);
        }
        return v;
    }

    /** Parses a JSON document from a file. */
    static Object parseFile(Path path) {
        try {
            return parse(Files.readString(path));
        } catch (IOException e) {
            throw new IllegalStateException("cannot read JSON file " + path, e);
        }
    }

    /** Convenience: parses a file whose root is an object. */
    @SuppressWarnings("unchecked")
    static Map<String, Object> parseObjectFile(Path path) {
        return (Map<String, Object>) parseFile(path);
    }

    private Object parseValue() {
        char c = peek();
        switch (c) {
            case '{':
                return parseObject();
            case '[':
                return parseArray();
            case '"':
                return parseString();
            case 't':
                expect("true");
                return Boolean.TRUE;
            case 'f':
                expect("false");
                return Boolean.FALSE;
            case 'n':
                expect("null");
                return null;
            default:
                return parseNumber();
        }
    }

    private Map<String, Object> parseObject() {
        consume('{');
        Map<String, Object> obj = new LinkedHashMap<>();
        skipWs();
        if (peek() == '}') {
            pos++;
            return obj;
        }
        while (true) {
            skipWs();
            String key = parseString();
            skipWs();
            consume(':');
            skipWs();
            obj.put(key, parseValue());
            skipWs();
            char c = next();
            if (c == '}') {
                return obj;
            }
            if (c != ',') {
                throw new IllegalArgumentException("expected ',' or '}' at " + (pos - 1));
            }
        }
    }

    private List<Object> parseArray() {
        consume('[');
        List<Object> arr = new ArrayList<>();
        skipWs();
        if (peek() == ']') {
            pos++;
            return arr;
        }
        while (true) {
            skipWs();
            arr.add(parseValue());
            skipWs();
            char c = next();
            if (c == ']') {
                return arr;
            }
            if (c != ',') {
                throw new IllegalArgumentException("expected ',' or ']' at " + (pos - 1));
            }
        }
    }

    private String parseString() {
        consume('"');
        StringBuilder sb = new StringBuilder();
        while (true) {
            char c = next();
            if (c == '"') {
                return sb.toString();
            }
            if (c == '\\') {
                char e = next();
                switch (e) {
                    case '"' -> sb.append('"');
                    case '\\' -> sb.append('\\');
                    case '/' -> sb.append('/');
                    case 'b' -> sb.append('\b');
                    case 'f' -> sb.append('\f');
                    case 'n' -> sb.append('\n');
                    case 'r' -> sb.append('\r');
                    case 't' -> sb.append('\t');
                    case 'u' -> {
                        int cp = Integer.parseInt(src.substring(pos, pos + 4), 16);
                        pos += 4;
                        sb.append((char) cp);
                    }
                    default -> throw new IllegalArgumentException(
                            "bad escape '\\" + e + "' at " + (pos - 1));
                }
            } else {
                sb.append(c);
            }
        }
    }

    private Double parseNumber() {
        int start = pos;
        while (pos < src.length()) {
            char c = src.charAt(pos);
            if (c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E'
                    || (c >= '0' && c <= '9')) {
                pos++;
            } else {
                break;
            }
        }
        if (start == pos) {
            throw new IllegalArgumentException("expected a JSON value at " + start);
        }
        return Double.parseDouble(src.substring(start, pos));
    }

    private void expect(String literal) {
        if (!src.startsWith(literal, pos)) {
            throw new IllegalArgumentException("expected '" + literal + "' at " + pos);
        }
        pos += literal.length();
    }

    private void skipWs() {
        while (pos < src.length() && Character.isWhitespace(src.charAt(pos))) {
            pos++;
        }
    }

    private char peek() {
        if (pos >= src.length()) {
            throw new IllegalArgumentException("unexpected end of JSON input");
        }
        return src.charAt(pos);
    }

    private char next() {
        char c = peek();
        pos++;
        return c;
    }

    private void consume(char expected) {
        char c = next();
        if (c != expected) {
            throw new IllegalArgumentException(
                    "expected '" + expected + "' but found '" + c + "' at " + (pos - 1));
        }
    }
}
