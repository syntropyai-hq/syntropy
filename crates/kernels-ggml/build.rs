// SPDX-License-Identifier: MIT OR Apache-2.0

//! Builds the vendored ggml tensor library and its CPU backend as one static library, linked
//! into this crate (RFC-0040 §8.1; ADR-0403 item 6; COMP-E7-T01).
//!
//! The sources are `third_party/ggml/` at the commit `third_party/ggml.lock` records. The build
//! names every file and flag itself, with the `cc` crate, so it needs no CMake and does exactly
//! what ADR-0403 item 6 asks:
//!
//! - **The baseline ISA.** x86-64 builds for x86-64-v2 (`-march=x86-64-v2`, or `/arch:SSE4.2`
//!   under MSVC), and AArch64 for ARMv8.0 (`-march=armv8-a`). Wider kernels come from the CPU
//!   provider, never from ggml (ADR-0403 item 6). ggml's optional CPU sub-backends are not built.
//! - **No floating-point contraction.** `-ffp-contract=off` under GCC and Clang. MSVC does not
//!   contract under its default `/fp:precise`, which this build keeps.
//! - **Every symbol hidden.** `-fvisibility=hidden`, so no ggml symbol is exported from a shared
//!   object that links this crate, and the core binary's dynamic symbol table names none.
//!
//! `src/bindings.rs` is generated once, at vendoring time, by a pinned `bindgen`, and checked in
//! (its header names the version and the command), so a build needs no libclang.

use std::env;
use std::path::{Path, PathBuf};

/// ggml's version and commit at the lock, for `ggml-version.h` (`ggml_version()`, `ggml_commit()`).
const GGML_VERSION: &str = "0.27.0";
const GGML_COMMIT: &str = "a05c9922613c6a2230dbb1d0de679519ecafb4f0";

/// The tensor library itself (ggml-base and the backend registry).
const BASE_C: &[&str] = &["src/ggml.c", "src/ggml-alloc.c", "src/ggml-quants.c"];
const BASE_CXX: &[&str] = &[
    "src/ggml.cpp",
    "src/ggml-backend.cpp",
    "src/ggml-backend-dl.cpp",
    "src/ggml-backend-meta.cpp",
    "src/ggml-backend-reg.cpp",
    "src/ggml-threading.cpp",
];
/// The CPU backend, without its optional sub-backends.
const CPU_C: &[&str] = &["src/ggml-cpu/ggml-cpu.c", "src/ggml-cpu/quants.c"];
const CPU_CXX: &[&str] = &[
    "src/ggml-cpu/ggml-cpu.cpp",
    "src/ggml-cpu/repack.cpp",
    "src/ggml-cpu/hbm.cpp",
    "src/ggml-cpu/traits.cpp",
    "src/ggml-cpu/amx/amx.cpp",
    "src/ggml-cpu/amx/mmq.cpp",
    "src/ggml-cpu/tiled/tiled.cpp",
    "src/ggml-cpu/tiled/tiled-kernel.cpp",
    "src/ggml-cpu/binary-ops.cpp",
    "src/ggml-cpu/unary-ops.cpp",
    "src/ggml-cpu/vec.cpp",
    "src/ggml-cpu/ops.cpp",
];

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("set by cargo"));
    let root = manifest.join("../../third_party/ggml");
    let out = PathBuf::from(env::var("OUT_DIR").expect("set by cargo"));
    println!("cargo:rerun-if-changed={}", root.display());
    println!("cargo:rerun-if-changed=build.rs");

    let template = std::fs::read_to_string(root.join("src/ggml-version.h.in"))
        .expect("third_party/ggml/src/ggml-version.h.in is vendored");
    std::fs::write(
        out.join("ggml-version.h"),
        template
            .replace("@GGML_VERSION@", GGML_VERSION)
            .replace("@GGML_BUILD_COMMIT@", GGML_COMMIT),
    )
    .expect("OUT_DIR is writable");

    let arch = env::var("CARGO_CFG_TARGET_ARCH").expect("set by cargo");
    let (arch_c, arch_cxx): (&[&str], &[&str]) = match arch.as_str() {
        "x86_64" => (
            &["src/ggml-cpu/arch/x86/quants.c"],
            &["src/ggml-cpu/arch/x86/repack.cpp"],
        ),
        "aarch64" => (
            &["src/ggml-cpu/arch/arm/quants.c"],
            &["src/ggml-cpu/arch/arm/repack.cpp"],
        ),
        other => panic!("syntropy-kernels-ggml: no ggml CPU build for target_arch {other}"),
    };

    let c_files: Vec<&str> = BASE_C.iter().chain(CPU_C).chain(arch_c).copied().collect();
    let cxx_files: Vec<&str> = BASE_CXX
        .iter()
        .chain(CPU_CXX)
        .chain(arch_cxx)
        .copied()
        .collect();
    let mut c = configured(&root, &out, &arch, false);
    let msvc = c.get_compiler().is_like_msvc();
    println!(
        "cargo:rustc-env=SYNTROPY_GGML_FLAGS={}",
        flags(&arch, msvc, false).join(" ")
    );
    c.files(c_files.iter().map(|f| root.join(f)));
    c.compile("syntropy_ggml_c");
    let mut cxx = configured(&root, &out, &arch, true);
    let target_os = env::var("CARGO_CFG_TARGET_OS").expect("set by cargo");
    let static_stdlib = target_os == "linux" && !msvc;
    if static_stdlib {
        // The core binary needs no C++ runtime library on Linux (rule 7 refuses `libstdc++`):
        // the GNU C++ runtime is linked statically, as its GCC Runtime Library Exception
        // allows. `cc` would otherwise ask for the shared `-lstdc++`.
        cxx.cpp_link_stdlib(None);
    }
    cxx.files(cxx_files.iter().map(|f| root.join(f)));
    cxx.compile("syntropy_ggml_cxx");
    if static_stdlib {
        link_static_stdlib(&cxx);
    }
    if target_os == "windows" {
        // The CPU backend reads the processor's name from the registry.
        println!("cargo:rustc-link-lib=advapi32");
    }
}

/// Links `libstdc++.a`, from the directory the C++ compiler itself names for it.
fn link_static_stdlib(cxx: &cc::Build) {
    let compiler = cxx.get_compiler();
    let output = compiler
        .to_command()
        .arg("-print-file-name=libstdc++.a")
        .output()
        .expect("the C++ compiler runs");
    let path = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
    let dir = path
        .parent()
        .filter(|_| path.is_absolute() && path.is_file())
        .unwrap_or_else(|| {
            panic!(
                "syntropy-kernels-ggml: the C++ compiler names no libstdc++.a ({}); install the \
                 static C++ runtime (libstdc++-dev)",
                path.display()
            )
        });
    println!("cargo:rustc-link-search=native={}", dir.display());
    println!("cargo:rustc-link-lib=static=stdc++");
}

/// A `cc` build with ggml's include paths, definitions and Syntropy's flags.
fn configured(root: &Path, out: &Path, arch: &str, cxx: bool) -> cc::Build {
    let target_os = env::var("CARGO_CFG_TARGET_OS").expect("set by cargo");
    let mut b = cc::Build::new();
    b.cpp(cxx)
        .warnings(false)
        .include(root.join("include"))
        .include(root.join("src"))
        .include(root.join("src/ggml-cpu"))
        .include(out)
        .define("GGML_SCHED_MAX_COPIES", "4")
        .define("GGML_USE_CPU", None);
    let msvc = b.get_compiler().is_like_msvc();
    for flag in flags(arch, msvc, cxx) {
        b.flag(flag);
    }
    if msvc {
        b.define("_CRT_SECURE_NO_WARNINGS", None);
    } else {
        b.define("_XOPEN_SOURCE", "600");
    }
    match target_os.as_str() {
        "linux" | "android" => {
            b.define("_GNU_SOURCE", None);
        }
        "macos" | "ios" => {
            b.define("_DARWIN_C_SOURCE", None);
        }
        _ => {}
    }
    if arch == "x86_64" {
        b.define("GGML_SSE42", None);
    }
    b
}

/// The compiler flags beyond include paths and definitions: the language standard, the baseline
/// ISA, no floating-point contraction and hidden symbols (ADR-0403 item 6).
fn flags(arch: &str, msvc: bool, cxx: bool) -> Vec<&'static str> {
    let mut flags = Vec::new();
    if msvc {
        flags.push(if cxx { "/std:c++17" } else { "/std:c11" });
        // MSVC does not contract a * b + c under /fp:precise, its default.
        flags.push("/fp:precise");
        if arch == "x86_64" {
            flags.push("/arch:SSE4.2");
        }
    } else {
        flags.push(if cxx { "-std=c++17" } else { "-std=c11" });
        flags.push("-ffp-contract=off");
        flags.push("-fvisibility=hidden");
        if cxx {
            flags.push("-fvisibility-inlines-hidden");
        }
        match arch {
            "x86_64" => flags.push("-march=x86-64-v2"),
            "aarch64" => flags.push("-march=armv8-a"),
            _ => {}
        }
    }
    flags
}
