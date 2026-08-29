/// \file data_io.hpp
/// \brief Loaders for the bundled data files (CSV returns/caps, JSON
///        views/config) used by the demo and tests.
#pragma once

#include <string>
#include <vector>

#include "portopt/black_litterman.hpp"
#include "portopt/linalg.hpp"

namespace portopt {

/// Parsed returns file: header names + T x N matrix.
struct ReturnsData {
  std::vector<std::string> names;
  Matrix returns;
};

/// Read data/asset_returns.csv: a header row of asset names followed by
/// rows of comma-separated daily simple returns.
/// \throws std::runtime_error if the file cannot be read or is malformed.
ReturnsData load_returns_csv(const std::string& path);

/// Parsed market-caps file (header asset,market_cap).
struct MarketCaps {
  std::vector<std::string> names;
  Vector caps;
};

/// Read data/market_caps.csv.
MarketCaps load_market_caps(const std::string& path);

/// Parsed views document (data/views.json).
struct ViewsDocument {
  double tau = 0.05;
  double delta = 2.5;
  std::vector<View> views;
};

/// Read data/views.json (tau, delta, and the view list).
ViewsDocument load_views(const std::string& path);

/// Pinned project constants (data/config.json).
struct Config {
  int annualization_factor = 252;
  double risk_free_rate = 0.02;
  double risk_aversion_delta = 2.5;
  double tau = 0.05;
  int estimation_window = 500;
  int rebalance_step = 63;
  int frontier_points = 25;
  double frontier_penalty = 1.0e4;
};

/// Read data/config.json.
Config load_config(const std::string& path);

}  // namespace portopt
