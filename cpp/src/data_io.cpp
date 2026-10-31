#include "portopt/data_io.hpp"

#include <cctype>
#include <charconv>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace portopt {

namespace {

std::string read_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open file: " + path);
  std::ostringstream buf;
  buf << in.rdbuf();
  return buf.str();
}

/// Trim ASCII whitespace (including the '\r' of CRLF files) from both ends.
std::string trim(const std::string& s) {
  std::size_t b = 0;
  std::size_t e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
  return s.substr(b, e - b);
}

std::vector<std::string> split_csv_line(const std::string& line) {
  std::vector<std::string> out;
  std::string cell;
  std::istringstream ss(line);
  while (std::getline(ss, cell, ',')) out.push_back(trim(cell));
  if (!line.empty() && line.back() == ',') out.push_back("");
  return out;
}

/// Locale-independent, correctly rounded decimal parsing (std::from_chars);
/// the whole token must be consumed.
double parse_double(const std::string& tok, const std::string& path) {
  double v = 0.0;
  const char* first = tok.data();
  const char* last = tok.data() + tok.size();
  const auto res = std::from_chars(first, last, v);
  if (res.ec != std::errc() || res.ptr != last) {
    throw std::runtime_error("malformed number '" + tok + "' in " + path);
  }
  return v;
}

/// Minimal JSON scanner for the bundled config/views documents: objects,
/// arrays, strings (no escapes needed by the schema), and numbers.
class JsonScanner {
 public:
  struct Node {
    enum class Type { kNumber, kString, kArray, kObject } type = Type::kNumber;
    double number = 0.0;
    std::string str;
    std::vector<Node> arr;
    // Object members as parallel key/value vectors: std::pair<std::string, Node>
    // would need Node complete here, and Node is still being defined.
    std::vector<std::string> keys;
    std::vector<Node> vals;

    const Node& at(const std::string& key) const {
      for (std::size_t i = 0; i < keys.size(); ++i) {
        if (keys[i] == key) return vals[i];
      }
      throw std::runtime_error("json: missing key '" + key + "'");
    }
    bool has(const std::string& key) const {
      for (const auto& k : keys) {
        if (k == key) return true;
      }
      return false;
    }
  };

  explicit JsonScanner(std::string text) : text_(std::move(text)) {}

  Node parse() {
    Node n = value();
    ws();
    if (pos_ != text_.size()) fail("trailing content");
    return n;
  }

 private:
  std::string text_;
  std::size_t pos_ = 0;

  [[noreturn]] void fail(const std::string& why) const {
    throw std::runtime_error("json: " + why + " at offset " + std::to_string(pos_));
  }
  void ws() {
    while (pos_ < text_.size() && (text_[pos_] == ' ' || text_[pos_] == '\t' ||
                                   text_[pos_] == '\n' || text_[pos_] == '\r')) {
      ++pos_;
    }
  }
  char peek() {
    if (pos_ >= text_.size()) fail("unexpected end of input");
    return text_[pos_];
  }
  void expect(char c) {
    if (peek() != c) fail(std::string("expected '") + c + "'");
    ++pos_;
  }
  std::string string_lit() {
    expect('"');
    std::string out;
    while (true) {
      if (pos_ >= text_.size()) fail("unterminated string");
      const char c = text_[pos_++];
      if (c == '"') break;
      if (c == '\\') fail("string escapes are not supported");
      out += c;
    }
    return out;
  }
  Node value() {
    ws();
    const char c = peek();
    if (c == '{') return object();
    if (c == '[') return array();
    if (c == '"') {
      Node n;
      n.type = Node::Type::kString;
      n.str = string_lit();
      return n;
    }
    return number();
  }
  Node number() {
    const std::size_t start = pos_;
    while (pos_ < text_.size() &&
           (std::isdigit(static_cast<unsigned char>(text_[pos_])) || text_[pos_] == '-' ||
            text_[pos_] == '+' || text_[pos_] == '.' || text_[pos_] == 'e' ||
            text_[pos_] == 'E')) {
      ++pos_;
    }
    if (pos_ == start) fail("invalid value");
    const std::string tok = text_.substr(start, pos_ - start);
    double v = 0.0;
    const auto res = std::from_chars(tok.data(), tok.data() + tok.size(), v);
    if (res.ec != std::errc() || res.ptr != tok.data() + tok.size()) {
      fail("invalid number '" + tok + "'");
    }
    Node n;
    n.type = Node::Type::kNumber;
    n.number = v;
    return n;
  }
  Node array() {
    expect('[');
    Node n;
    n.type = Node::Type::kArray;
    ws();
    if (peek() == ']') {
      ++pos_;
      return n;
    }
    while (true) {
      n.arr.push_back(value());
      ws();
      const char c = peek();
      ++pos_;
      if (c == ']') break;
      if (c != ',') fail("expected ',' or ']'");
    }
    return n;
  }
  Node object() {
    expect('{');
    Node n;
    n.type = Node::Type::kObject;
    ws();
    if (peek() == '}') {
      ++pos_;
      return n;
    }
    while (true) {
      ws();
      std::string key = string_lit();
      ws();
      expect(':');
      n.keys.push_back(std::move(key));
      n.vals.push_back(value());
      ws();
      const char c = peek();
      ++pos_;
      if (c == '}') break;
      if (c != ',') fail("expected ',' or '}'");
    }
    return n;
  }
};

}  // namespace

ReturnsData load_returns_csv(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open file: " + path);
  std::string line;
  if (!std::getline(in, line)) throw std::runtime_error("empty returns file: " + path);
  ReturnsData out;
  out.names = split_csv_line(trim(line));
  const std::size_t n = out.names.size();
  std::vector<double> values;
  std::size_t rows = 0;
  while (std::getline(in, line)) {
    line = trim(line);
    if (line.empty()) continue;
    const std::vector<std::string> cells = split_csv_line(line);
    if (cells.size() != n) {
      throw std::runtime_error("inconsistent column count in " + path);
    }
    for (const std::string& cell : cells) values.push_back(parse_double(cell, path));
    ++rows;
  }
  out.returns = Matrix(rows, n);
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t c = 0; c < n; ++c) out.returns(r, c) = values[r * n + c];
  }
  return out;
}

MarketCaps load_market_caps(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open file: " + path);
  std::string line;
  if (!std::getline(in, line)) throw std::runtime_error("empty caps file: " + path);
  MarketCaps out;
  while (std::getline(in, line)) {
    line = trim(line);
    if (line.empty()) continue;
    const std::vector<std::string> cells = split_csv_line(line);
    if (cells.size() != 2) throw std::runtime_error("malformed row in " + path);
    out.names.push_back(cells[0]);
    out.caps.push_back(parse_double(cells[1], path));
  }
  return out;
}

ViewsDocument load_views(const std::string& path) {
  const JsonScanner::Node doc = JsonScanner(read_file(path)).parse();
  ViewsDocument out;
  out.tau = doc.at("tau").number;
  out.delta = doc.at("delta").number;
  for (const auto& node : doc.at("views").arr) {
    View v;
    v.type = node.at("type").str;
    if (node.has("asset")) v.asset = node.at("asset").str;
    if (node.has("long")) v.long_name = node.at("long").str;
    if (node.has("short")) v.short_name = node.at("short").str;
    v.q = node.at("q").number;
    if (node.has("confidence")) v.confidence = node.at("confidence").number;
    out.views.push_back(v);
  }
  return out;
}

Config load_config(const std::string& path) {
  const JsonScanner::Node doc = JsonScanner(read_file(path)).parse();
  Config cfg;
  cfg.annualization_factor = static_cast<int>(doc.at("annualization_factor").number);
  cfg.risk_free_rate = doc.at("risk_free_rate").number;
  cfg.risk_aversion_delta = doc.at("risk_aversion_delta").number;
  cfg.tau = doc.at("tau").number;
  cfg.estimation_window = static_cast<int>(doc.at("estimation_window").number);
  cfg.rebalance_step = static_cast<int>(doc.at("rebalance_step").number);
  cfg.frontier_points = static_cast<int>(doc.at("frontier_points").number);
  return cfg;
}

}  // namespace portopt
