// SPDX-License-Identifier: MIT OR Apache-2.0

//! The daemon-runner protocol, the shared-memory layout and SPSC rings.
//!
//! Layer 1 in RFC-0001 §4.1, designed in RFC-0001 §3.5. It holds the shared-memory region
//! ([`shm`]): its header, its rings and the segment each OS provides; and the command channel
//! ([`cmd`]): its frames, its messages, its endpoint and peer check, and the handshake.
//!
//! `unsafe` is allowed in this crate (RFC-0001 §4.2 rule 8). Every `unsafe` block and impl carries
//! a `// SAFETY:` comment that states its invariant (ADR-0001 item 4).

#![deny(clippy::undocumented_unsafe_blocks)]
#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(any(target_os = "linux", target_vendor = "apple", windows))]
pub mod cmd;
#[cfg(any(target_os = "linux", target_vendor = "apple", windows))]
pub mod shm;

/// The test-only features (RFC-0001 §4.2 rule 10) this build of the crate has on. The binary's
/// build metadata lists them (`cli`'s `build_id::features`, RFC-0062 §8.1), since cargo's
/// feature unification can turn one on without the binary's own crate seeing it.
pub const TEST_FEATURES: &[&str] = if cfg!(feature = "fault-injection") {
    &["fault-injection"]
} else {
    &[]
};

#[cfg(test)]
mod feature_tests {
    /// `TEST_FEATURES` names `fault-injection` exactly when it is on.
    #[test]
    fn test_features_name_what_is_on() {
        assert_eq!(
            super::TEST_FEATURES.contains(&"fault-injection"),
            cfg!(feature = "fault-injection")
        );
    }
}
