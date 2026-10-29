// SPDX-License-Identifier: MIT OR Apache-2.0

//! Links the test provider library (`examples/test_provider.rs`) the way a provider library is
//! linked on Linux: with `-Bsymbolic`, so the library's internal references never bind to symbols
//! of the core or of another module (RFC-0040 §10.4; ADR-0400 decision 4).
//!
//! The single export needs no flag. For every `cdylib`, rustc passes the linker an export list of
//! the `#[no_mangle]` symbols (a version script on Linux, `-exported_symbols_list` on macOS, an
//! export file on Windows) and makes everything else local, so `export!`'s entry symbol is the
//! only one the library exports. `tests/exports.rs` checks that on each OS. A named ELF version
//! node (`SYNTROPY_PROVIDER_1`) cannot be added with a second script: rustc's script is
//! anonymous, and GNU ld refuses to combine an anonymous version tag with a named one.
//!
//! `cargo::rustc-link-arg-examples` does not reach an example built as a `cdylib`, so this uses
//! `cargo::rustc-link-arg`, which applies to every binary-like target of the package: the test
//! provider and the integration-test executables, which `-Bsymbolic` leaves working. A provider
//! crate (`kernels-cuda` and the rest) passes it with `cargo::rustc-cdylib-link-arg`.

use std::env;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if matches!(os.as_str(), "linux" | "android" | "freebsd") {
        println!("cargo::rustc-link-arg=-Wl,-Bsymbolic");
    }
}
