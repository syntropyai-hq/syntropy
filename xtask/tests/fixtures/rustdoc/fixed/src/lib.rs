// SPDX-License-Identifier: MIT OR Apache-2.0

//! The two failing fixtures' links, fixed: [`present`] exists, and [`thing()`] and
//! [`mod@thing`] each name one item.

/// Does nothing.
pub fn present() {}

/// A function named `thing`.
pub fn thing() {}

/// A module named `thing`.
pub mod thing {}
