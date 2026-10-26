// SPDX-License-Identifier: MIT OR Apache-2.0
//! [`Figure`], the form in which the API returns every predicted or measured number (RFC-0001
//! §5.16, RFC-0051 §3.1), so the label belongs to the data and not to each view: the dashboard,
//! the CLI, `doctor` and API clients all render the same kind and basis.

use std::fmt;

use serde::{Deserialize, Serialize};

/// What a number is (RFC-0051 §4 rule 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FigureKind {
    /// Read from this machine or this run: a probe, a counter, a version, a limit the OS reports.
    Measured,
    /// Derived from a model, such as the cost model; its basis says how.
    Estimate,
    /// A commitment or a chosen budget, compared against measured values.
    Target,
}

impl FigureKind {
    /// The label as text, as the CLI prints it: `measured`, `estimate` or `target`.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Measured => "measured",
            Self::Estimate => "estimate",
            Self::Target => "target",
        }
    }
}

/// A number, or any value shown as one, with its kind and where it comes from (RFC-0001 §5.16).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Figure<T> {
    /// The value.
    pub value: T,
    /// Measured, estimate or target.
    pub kind: FigureKind,
    /// The measurement, formula or budget it comes from.
    pub basis: Option<String>,
}

impl<T> Figure<T> {
    /// A measured value, with the measurement it comes from.
    pub fn measured(value: T, basis: impl Into<String>) -> Self {
        Self {
            value,
            kind: FigureKind::Measured,
            basis: Some(basis.into()),
        }
    }

    /// An estimate, with its derivation.
    pub fn estimate(value: T, basis: impl Into<String>) -> Self {
        Self {
            value,
            kind: FigureKind::Estimate,
            basis: Some(basis.into()),
        }
    }

    /// A target, with the budget or commitment it states.
    pub fn target(value: T, basis: impl Into<String>) -> Self {
        Self {
            value,
            kind: FigureKind::Target,
            basis: Some(basis.into()),
        }
    }
}

/// `value [kind]`, or `value [kind: basis]` when the basis is known.
impl<T: fmt::Display> fmt::Display for Figure<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.basis {
            Some(basis) => write!(f, "{} [{}: {basis}]", self.value, self.kind.label()),
            None => write!(f, "{} [{}]", self.value, self.kind.label()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_figure_prints_its_kind_and_basis() {
        let f = Figure::measured("580.65".to_owned(), "nvidia-smi");
        assert_eq!(f.to_string(), "580.65 [measured: nvidia-smi]");
        let f = Figure {
            value: 7433,
            kind: FigureKind::Target,
            basis: None,
        };
        assert_eq!(f.to_string(), "7433 [target]");
    }

    #[test]
    fn a_figure_round_trips_through_json_with_its_kind_in_snake_case() {
        let f = Figure::estimate(1.5_f64, "cost model, decode term");
        let json = serde_json::to_string(&f).unwrap();
        assert_eq!(
            json,
            r#"{"value":1.5,"kind":"estimate","basis":"cost model, decode term"}"#
        );
        assert_eq!(serde_json::from_str::<Figure<f64>>(&json).unwrap(), f);
    }
}
