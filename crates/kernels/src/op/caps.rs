// SPDX-License-Identifier: MIT OR Apache-2.0

//! What `query` answers for an (op, shape), and the op `prepare` returns (RFC-0040 §3.2,
//! §5.3).

use syntropy_core::{Blake3, Bytes, ErrorClass, Nanos, Numerics};

use super::key::LayoutVersion;
use crate::error::Error;

/// A provider's answer to `query` for one (`OpKey`, `ShapeSig`) on one device (RFC-0040
/// §3.2).
///
/// It has no serde derives: it is returned in process and never stored, and `core`'s
/// [`Numerics`] has none.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpCaps {
    /// Scratch the prepared op needs (RFC-0040 §5.3). The executor sums the maximum over the
    /// ops of each concurrently live region and asks the tier manager for it; providers never
    /// allocate workspaces themselves.
    pub workspace: Bytes,
    /// The provider's own time estimate: an ordering hint only (RFC-0040 §6.3).
    pub est: Option<Nanos>,
    /// The weight layout the op reads (RFC-0040 §5.4).
    pub layout: LayoutVersion,
    /// Safe to record in a captured graph (RFC-0040 §4.4 rule 5).
    pub capture_safe: bool,
    /// Gives identical output for identical input on every run (RFC-0040 §3.3).
    pub deterministic: bool,
    /// Accumulation, activation quantisation and output dtype (RFC-0042 §4).
    pub numerics: Numerics,
    /// The hash its `OpCert` is keyed on (RFC-0040 §10.2; RFC-0060 §6.4).
    pub kernel_hash: Blake3,
    /// The ADR-0623 `SourceKey` of its translation unit (RFC-0040 §6.2).
    pub source_key: Blake3,
}

/// An op made ready to launch on one device at one shape (RFC-0001 §5.7).
///
/// RFC-0001 names the type without fields. It carries the workspace the prepared op asks
/// for, which [`PreparedOp::check_workspace`] holds to what `query` declared, and the
/// provider's handle for what it prepared, as other provider objects do ([`memory`]).
///
/// [`memory`]: crate::memory
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparedOp {
    /// Scratch the prepared op asks for.
    pub workspace: Bytes,
    /// The provider's handle for the prepared op, meaningful only to the provider that issued
    /// it: the CPU provider's kernel entry, for example. 0 when the provider needs none.
    pub handle: u64,
}

impl PreparedOp {
    /// A prepared op that asks for `workspace` bytes of scratch, with no provider handle.
    #[must_use]
    pub fn new(workspace: Bytes) -> Self {
        Self {
            workspace,
            handle: 0,
        }
    }

    /// A prepared op that asks for `workspace` bytes of scratch, with the provider's `handle`.
    #[must_use]
    pub fn with_handle(workspace: Bytes, handle: u64) -> Self {
        Self { workspace, handle }
    }

    /// Checks that the prepared op asks for no more workspace than `caps` declared for the
    /// same (op, shape) (RFC-0040 §5.3).
    ///
    /// # Errors
    ///
    /// `WorkspaceExceeded` (RFC-0040 §13 code 14, class `Internal`) when it asks for more:
    /// a provider bug, on which the runner fails with a diagnostic bundle.
    pub fn check_workspace(&self, caps: &OpCaps) -> Result<(), Error> {
        if self.workspace <= caps.workspace {
            return Ok(());
        }
        Err(Error::new(
            ErrorClass::Internal,
            crate::error::codes::WORKSPACE_EXCEEDED,
            format!(
                "prepared op asks for {} bytes of workspace; query declared {}",
                self.workspace.0, caps.workspace.0
            ),
        ))
    }
}
