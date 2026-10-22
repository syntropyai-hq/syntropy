// SPDX-License-Identifier: MIT OR Apache-2.0

//! A value alone on its cache line, so the producer's and the consumer's writes never share one
//! (false sharing).

use std::ops::Deref;

/// `T`, aligned and padded to 128 bytes on x86-64 and AArch64, and to 64 bytes elsewhere.
///
/// x86-64 cores prefetch cache lines in adjacent pairs, so two 64-byte lines still interfere, and
/// some AArch64 cores have 128-byte lines.
#[cfg_attr(any(target_arch = "x86_64", target_arch = "aarch64"), repr(align(128)))]
#[cfg_attr(
    not(any(target_arch = "x86_64", target_arch = "aarch64")),
    repr(align(64))
)]
#[derive(Debug, Default)]
pub(crate) struct CachePadded<T>(pub(crate) T);

impl<T> Deref for CachePadded<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}
