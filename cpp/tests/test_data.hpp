/// \file test_data.hpp
/// \brief Shared lazy-loaded fixtures for the portopt test suite: the
///        bundled data files and the golden case table. Tests run with
///        the working directory set to cpp/, so data lives at ../data.
#pragma once

#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_json.hpp"
#include "portopt/data_io.hpp"
#include "portopt/estimators.hpp"
#include "portopt/linalg.hpp"

namespace testdata {

inline const std::string kDataDir = "../data";

/// One golden case: flat scalar inputs/expectations plus tolerance.
struct GoldenCase {
  std::map<std::string, double> inputs;
  std::map<std::string, double> expect;
  double tol = 0.0;
};

/// All golden cases keyed by name (loaded once).
inline const std::map<std::string, GoldenCase>& golden() {
  static const std::map<std::string, GoldenCase> cases = [] {
    const mini_json::Value doc = mini_json::parse_file(kDataDir + "/golden/golden.json");
    std::map<std::string, GoldenCase> out;
    for (const mini_json::Value& c : doc.at("cases").arr) {
      GoldenCase gc;
      const mini_json::Value& in = c.at("inputs");
      for (std::size_t i = 0; i < in.keys.size(); ++i) gc.inputs[in.keys[i]] = in.vals[i].number;
      const mini_json::Value& ex = c.at("expect");
      for (std::size_t i = 0; i < ex.keys.size(); ++i) gc.expect[ex.keys[i]] = ex.vals[i].number;
      gc.tol = c.at("tol").number;
      out[c.at("name").str] = gc;
    }
    return out;
  }();
  return cases;
}

/// Golden case lookup; throws when the case is missing. Takes a C string
/// so no temporary is bound to a reference parameter (keeps
/// -Wdangling-reference quiet at call sites).
inline const GoldenCase& golden_case(const char* name) {
  const auto it = golden().find(name);
  if (it == golden().end()) {
    throw std::runtime_error(std::string("missing golden case: ") + name);
  }
  return it->second;
}

/// Expected vector prefix_0..prefix_{k-1} from a golden expect/input map.
inline portopt::Vector golden_vec(const std::map<std::string, double>& m,
                                  const std::string& prefix) {
  portopt::Vector out;
  for (std::size_t i = 0;; ++i) {
    const auto it = m.find(prefix + "_" + std::to_string(i));
    if (it == m.end()) break;
    out.push_back(it->second);
  }
  return out;
}

/// Bundled returns file (parsed once).
inline const portopt::ReturnsData& returns_data() {
  static const portopt::ReturnsData data =
      portopt::load_returns_csv(kDataDir + "/asset_returns.csv");
  return data;
}

/// Market caps (parsed once).
inline const portopt::MarketCaps& market_caps() {
  static const portopt::MarketCaps caps =
      portopt::load_market_caps(kDataDir + "/market_caps.csv");
  return caps;
}

/// Bundled views document (parsed once).
inline const portopt::ViewsDocument& views_doc() {
  static const portopt::ViewsDocument doc =
      portopt::load_views(kDataDir + "/views.json");
  return doc;
}

/// Bundled config (parsed once).
inline const portopt::Config& config() {
  static const portopt::Config cfg = portopt::load_config(kDataDir + "/config.json");
  return cfg;
}

/// Annualized sample mean of the bundled returns.
inline const portopt::Vector& mu_ann() {
  static const portopt::Vector mu =
      portopt::annualize_mean(portopt::sample_mean(returns_data().returns));
  return mu;
}

/// Annualized sample covariance of the bundled returns.
inline const portopt::Matrix& sigma_ann() {
  static const portopt::Matrix sig =
      portopt::annualize_cov(portopt::sample_cov(returns_data().returns));
  return sig;
}

/// Market weights w_mkt = caps / sum(caps).
inline const portopt::Vector& w_mkt() {
  static const portopt::Vector w = [] {
    portopt::Vector caps = market_caps().caps;
    double tot = 0.0;
    for (double c : caps) tot += c;
    for (double& c : caps) c /= tot;
    return caps;
  }();
  return w;
}

/// Small SPD test matrix (3 x 3) used across unit tests.
inline portopt::Matrix spd3() {
  portopt::Matrix a(3, 3, 0.0);
  a(0, 0) = 0.04;  a(0, 1) = 0.012; a(0, 2) = 0.006;
  a(1, 0) = 0.012; a(1, 1) = 0.09;  a(1, 2) = 0.018;
  a(2, 0) = 0.006; a(2, 1) = 0.018; a(2, 2) = 0.0225;
  return a;
}

}  // namespace testdata
