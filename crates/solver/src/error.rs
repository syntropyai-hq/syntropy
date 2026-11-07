// SPDX-License-Identifier: MIT OR Apache-2.0

//! Why a solver entry point returned no result.

use std::fmt;

/// One of the solver's entry points (RFC-0011 §2).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Entry {
    /// [`solve`](crate::solve), the slow-phase solve.
    Solve,
    /// [`solve_delta`](crate::solve_delta), the delta solve (RFC-0011 §12.3).
    SolveDelta,
    /// [`route`](crate::route), per-request routing (RFC-0011 §13).
    Route,
    /// [`evaluate`](crate::evaluate), the evaluator (RFC-0011 §11.3).
    Evaluate,
}

impl Entry {
    /// The function's name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Entry::Solve => "solve",
            Entry::SolveDelta => "solve_delta",
            Entry::Route => "route",
            Entry::Evaluate => "evaluate",
        }
    }
}

/// Why a solver entry point returned no result. Later tasks add the failures of RFC-0011 §17,
/// such as `Capacity` with its `InfeasibilityReport` (PLC-E4-T02, PLC-E6-T08), and RFC-0050
/// §4.2's `context_length_exceeded` from routing (PLC-E6-T19).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SolveError {
    /// The entry point's body is not built yet. `task` is the plan task that builds it.
    NotBuilt {
        /// The entry point.
        entry: Entry,
        /// The task that builds its body, such as `"PLC-E4-T02"`.
        task: &'static str,
    },
}

impl fmt::Display for SolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolveError::NotBuilt { entry, task } => write!(
                f,
                "the solver's `{}` is not built yet; {task} builds it",
                entry.name()
            ),
        }
    }
}

impl std::error::Error for SolveError {}
