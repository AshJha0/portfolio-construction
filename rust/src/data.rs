//! Loaders for the bundled project data files (CSV + JSON).
//!
//! Golden values are defined on `data/asset_returns.csv` *as parsed*:
//! the file stores doubles with `%.17g`, which round-trips exactly
//! through Rust's correctly rounded `str::parse::<f64>`.

use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::black_litterman::View;
use crate::error::{PortOptError, Result};
use crate::linalg::Matrix;

/// Contents of `data/views.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct ViewsDoc {
    /// Black-Litterman uncertainty scaling.
    pub tau: f64,
    /// Risk-aversion coefficient for reverse optimization.
    pub delta: f64,
    /// Bundled example views (one absolute, one relative).
    pub views: Vec<View>,
}

/// Contents of `data/config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    /// Asset names in column order.
    pub assets: Vec<String>,
    /// Trading days per year (252).
    pub annualization_factor: f64,
    /// Annual risk-free rate.
    pub risk_free_rate: f64,
    /// Reverse-optimization risk aversion.
    pub risk_aversion_delta: f64,
    /// Black-Litterman tau.
    pub tau: f64,
    /// Backtest expanding estimation window (rows).
    pub estimation_window: usize,
    /// Backtest rebalance step (rows).
    pub rebalance_step: usize,
    /// Number of frontier sweep points.
    pub frontier_points: usize,
    /// Quadratic penalty for the constrained frontier.
    pub frontier_penalty: f64,
}

fn read_to_string(path: &Path) -> Result<String> {
    fs::read_to_string(path)
        .map_err(|e| PortOptError::Data(format!("cannot read {}: {e}", path.display())))
}

/// Load `asset_returns.csv`: header of asset names, then `T` rows of `N`
/// daily simple returns. Returns `(names, T x N matrix)`.
pub fn load_returns_csv(path: &Path) -> Result<(Vec<String>, Matrix)> {
    let text = read_to_string(path)?;
    let mut lines = text.lines();
    let header = lines
        .next()
        .ok_or_else(|| PortOptError::Data(format!("{}: empty file", path.display())))?;
    let names: Vec<String> = header.trim().split(',').map(str::to_string).collect();
    let n = names.len();
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for (lineno, line) in lines.enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let row: Vec<f64> = line
            .split(',')
            .map(|tok| {
                tok.trim().parse::<f64>().map_err(|e| {
                    PortOptError::Data(format!(
                        "{} line {}: bad float {tok:?}: {e}",
                        path.display(),
                        lineno + 2
                    ))
                })
            })
            .collect::<Result<_>>()?;
        if row.len() != n {
            return Err(PortOptError::Data(format!(
                "{} line {}: expected {n} columns, got {}",
                path.display(),
                lineno + 2,
                row.len()
            )));
        }
        rows.push(row);
    }
    Ok((names, Matrix::from_rows(&rows)?))
}

/// Load `market_caps.csv` (`asset,market_cap` rows). Returns
/// `(names, caps)` in file order.
pub fn load_market_caps(path: &Path) -> Result<(Vec<String>, Vec<f64>)> {
    let text = read_to_string(path)?;
    let mut names = Vec::new();
    let mut caps = Vec::new();
    for (lineno, line) in text.lines().enumerate().skip(1) {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let (name, cap) = line.split_once(',').ok_or_else(|| {
            PortOptError::Data(format!(
                "{} line {}: expected 'asset,market_cap'",
                path.display(),
                lineno + 1
            ))
        })?;
        names.push(name.trim().to_string());
        caps.push(cap.trim().parse::<f64>().map_err(|e| {
            PortOptError::Data(format!(
                "{} line {}: bad market cap: {e}",
                path.display(),
                lineno + 1
            ))
        })?);
    }
    Ok((names, caps))
}

/// Load and parse `views.json`.
pub fn load_views(path: &Path) -> Result<ViewsDoc> {
    serde_json::from_str(&read_to_string(path)?)
        .map_err(|e| PortOptError::Data(format!("cannot parse {}: {e}", path.display())))
}

/// Load and parse `config.json`.
pub fn load_config(path: &Path) -> Result<Config> {
    serde_json::from_str(&read_to_string(path)?)
        .map_err(|e| PortOptError::Data(format!("cannot parse {}: {e}", path.display())))
}
