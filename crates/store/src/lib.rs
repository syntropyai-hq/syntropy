// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hub range reads, the BLAKE3 chunk store, P2P distribution, the pack builder, GC and persisted
//! session blobs.
//!
//! Layer 2 in RFC-0001 §4.1, designed in RFC-0041. Built so far: store roots and the v1
//! directory layout (RFC-0041 §4.1), the index of local models (§4.3), and sparse source blobs
//! with their inventories (§4.2).
//!
//! - [`Layout`] names every path under a root's `v1/` directory, with no I/O.
//! - [`WritableRoot`] opens a root of `store.dirs`, creates its layout and holds its writer
//!   lock, so one process at a time writes to a root.
//! - [`ReadOnlyRoot`] opens a root of `store.readonly_dirs` without writing to it.
//! - [`StoreRoots`] opens both lists.
//! - [`LocalIndex`] indexes local model files in place: one SHA-256 pass, a (device, inode,
//!   size, mtime) record per file in `local/<model-id>.sources` that later starts reuse, and
//!   `LocalSourceChanged` when a file changes ([`local`]).
//! - [`Blobs`] probes a writable root's volume for sparse files at start, and holds its source
//!   blobs: sparse files, or one file per chunk on a volume without sparse files, each with a
//!   synced inventory of the chunks it holds, hole punching on eviction, and disk use counted in
//!   allocated blocks ([`blob`]).

#![forbid(unsafe_code)]

pub mod blob;
mod error;
pub mod layout;
pub mod local;
pub mod root;

pub use blob::{
    Blob, BlobError, BlobMode, Blobs, DiskUsage, NoSparse, StorageMode, read_storage_mode,
};
pub use error::StoreError;
pub use layout::{Layout, Name, NameError};
pub use local::{LocalError, LocalFile, LocalIndex};
pub use root::{ReadOnlyRoot, StoreRoots, WritableRoot};
