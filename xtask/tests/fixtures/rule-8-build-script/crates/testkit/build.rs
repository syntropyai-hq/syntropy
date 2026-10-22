// SPDX-License-Identifier: MIT OR Apache-2.0

//! Generates code with its own `allow`, for `lib.rs` to include.

fn main() {
    let out = std::env::var("OUT_DIR").unwrap();
    let code = "#[allow(unsafe_code)]\npub fn read(p: *const u8) -> u8 { unsafe { *p } }\n";
    std::fs::write(format!("{out}/gen.rs"), code).unwrap();
}
