//! Error type shared by every fallible `portopt` routine.

use std::error::Error;
use std::fmt;

/// Error returned by all fallible `portopt` operations.
///
/// Mirrors the Python reference's `ValueError` taxonomy: every variant
/// carries a descriptive message so callers can report the exact failure
/// (singular covariance, undefined tangency, infeasible bounds, ...).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PortOptError {
    /// Malformed input: wrong shape/length, NaN/inf entries, bad parameter.
    InvalidInput(String),
    /// Matrix failed the Cholesky positive-definiteness gate (singular or
    /// indefinite covariance, e.g. a duplicated asset).
    NotPositiveDefinite(String),
    /// A quantity is mathematically undefined for the given inputs
    /// (tangency with `A - rf*C <= 0`, degenerate frontier, target below
    /// the minimum-variance return, zero-vol Sharpe, ...).
    Undefined(String),
    /// Infeasible constraint set for the box-simplex projection.
    Infeasible(String),
    /// Failure reading or parsing a bundled data file.
    Data(String),
}

impl fmt::Display for PortOptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PortOptError::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            PortOptError::NotPositiveDefinite(msg) => write!(f, "{msg}"),
            PortOptError::Undefined(msg) => write!(f, "{msg}"),
            PortOptError::Infeasible(msg) => write!(f, "infeasible bounds: {msg}"),
            PortOptError::Data(msg) => write!(f, "data error: {msg}"),
        }
    }
}

impl Error for PortOptError {}

/// Convenience alias used across the crate.
pub type Result<T> = std::result::Result<T, PortOptError>;
