// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hub range reads, the BLAKE3 chunk store, P2P distribution, the pack builder, GC and persisted
//! session blobs.
//!
//! Layer 2 in RFC-0001 §4.1, designed in RFC-0041. Built so far: store roots and the v1
//! directory layout (RFC-0041 §4.1), and the index of local models (§4.3).
//!
//! - [`Layout`] names every path under a root's `v1/` directory, with no I/O.
//! - [`WritableRoot`] opens a root of `store.dirs`, creates its layout and holds its writer
//!   lock, so one process at a time writes to a root.
//! - [`ReadOnlyRoot`] opens a root of `store.readonly_dirs` without writing to it.
//! - [`StoreRoots`] opens both lists.
//! - [`LocalIndex`] indexes local model files in place: one SHA-256 pass, a (device, inode,
//!   size, mtime) record per file in `local/<model-id>.sources` that later starts reuse, and
//!   `LocalSourceChanged` when a file changes ([`local`]).

#![forbid(unsafe_code)]

mod error;
pub mod layout;
pub mod local;
pub mod root;

pub use error::StoreError;
pub use layout::{Layout, Name, NameError};
pub use local::{LocalError, LocalFile, LocalIndex};
pub use root::{ReadOnlyRoot, StoreRoots, WritableRoot};
