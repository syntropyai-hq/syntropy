// SPDX-License-Identifier: MIT OR Apache-2.0

//! The control plane's wire encoding (RFC-0031 §9). Owner: RFC-0031.
//!
//! [`envelope`] is the deterministic CBOR codec and the stream and datagram framing that every
//! control and join message travels in (DIST-E8-T12). The `CtlMsg` enum it carries is
//! DIST-E8-T17, which waits on the tasks that define its payload types.

pub mod envelope;
