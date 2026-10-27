// SPDX-License-Identifier: MIT OR Apache-2.0

//! The crates and features that the rules of RFC-0001 §4.2 name, as the check applies them.
//!
//! Workspace crates are named as in RFC-0001: `exec` is the crate `syntropy-exec`. The layer table
//! itself is data in `ci/layers.toml` (RFC-0062 §12.1); everything else a rule names is here, next
//! to the rule it comes from. A unit test checks that every workspace crate named here is in the
//! real layer table, so renaming a crate without updating this file fails the tests.

/// Every workspace member's package name starts with this (RFC-0001 §4.1).
pub const PREFIX: &str = "syntropy-";

/// Rule 1: `core` depends on no workspace crate.
pub const CORE: &str = "core";

/// Rule 1: the only external crates `core` may depend on, as a normal or build dependency.
pub const CORE_EXTERNALS: &[&str] = &["serde", "thiserror", "smallvec", "bitflags"];

/// Rule 2: every crate except `core` may depend on these, whatever its layer.
pub const SHARED: &[&str] = &["obs", "config"];

/// Rule 2: a crate and the proc-macro crate that holds its macros, as `(crate, macro crate)`. Rust
/// requires a proc-macro to be its own crate. The crate may depend on its macro crate inside their
/// shared layer, and no other crate may depend on the macro crate at all: they use the macros
/// through the crate that owns them (RFC-0001 §4.2 rule 2, note of 2026-10-21).
pub const MACRO_CRATES: &[(&str, &str)] = &[("config", "config-derive")];

/// Rule 2: a crate that named crates of its own layer may depend on, as `(crate, users)`.
/// `spsc` holds the lock-free SPSC ring of RFC-0001 §8.3, which `io`, `net` and `ipc` all need
/// although all three sit in layer 1 with it (RFC-0001 §4.2 rule 2, note of 2026-10-22). Such a
/// crate depends on no workspace crate itself, `core` included; a dev edge to a dev-only crate
/// stays rule 9's, so its tests can use `testkit`.
pub const ON_SAME_LAYER: &[(&str, &[&str])] = &[("spsc", &["io", "net", "ipc"])];

/// Rule 2: the crates of its own layer that may depend on `name`, if `name` is such a crate.
pub fn same_layer_users(name: &str) -> Option<&'static [&'static str]> {
    ON_SAME_LAYER
        .iter()
        .find(|&&(shared, _)| shared == name)
        .map(|&(_, users)| users)
}

/// Rule 2: the crate that owns `name`'s macros, if `name` is a macro crate.
pub fn macro_owner(name: &str) -> Option<&'static str> {
    MACRO_CRATES
        .iter()
        .find(|&&(_, macros)| macros == name)
        .map(|&(owner, _)| owner)
}

/// Rule 3 and rule 8 apply to every crate whose name starts with this, `kernels-loader` included.
pub const PROVIDER_PREFIX: &str = "kernels-";

/// Rule 3: the compute crates, besides every `kernels-*` crate. `spsc` is here because compute
/// crates use its ring and it has no tokio (RFC-0001 §4.1, note of 2026-10-22).
pub const COMPUTE: &[&str] = &["graph", "kernels", "io", "tier", "exec", "sched", "spsc"];

/// Rule 3: workspace crates no compute crate may reach through normal dependencies.
pub const COMPUTE_MUST_NOT_REACH: &[&str] = &["net", "cluster", "store", "api", "daemon"];

/// Rule 3 and ADR-0012 §6: the async runtime no compute crate may reach.
pub const TOKIO: &str = "tokio";

/// ADR-0012 §6 names these crates (and every `kernels-*`) for the no-tokio invariant. Rule 3
/// already covers them; a finding on one of them cites both.
pub const ADR_0012_NO_TOKIO: &[&str] = &["exec", "io"];

/// Rule 4: the crate that never links a `kernels-*` crate.
pub const DAEMON: &str = "daemon";

/// Rule 5: the crate that must stay pure.
pub const SOLVER: &str = "solver";

/// Rule 5: workspace crates that do I/O, which `solver` must not reach.
pub const SOLVER_MUST_NOT_REACH: &[&str] = &["io", "net", "ipc", "store"];

/// Rule 5: external crates `solver` must not reach: async runtimes and sockets, filesystem crates
/// and thread pools (RFC-0062 §12.1 names tokio and filesystem crates). Clock and `std::net` calls
/// are source-level, so this metadata check cannot see them.
pub const SOLVER_MUST_NOT_REACH_EXTERNAL: &[&str] = &[
    "tokio",
    "async-std",
    "smol",
    "mio",
    "socket2",
    "walkdir",
    "tempfile",
    "notify",
    "fs-err",
    "rayon",
    "rayon-core",
    "threadpool",
];

/// Rule 6: an external system and the one workspace crate that may use it.
pub struct System {
    /// The system's name, for messages.
    pub name: &'static str,
    /// External packages that bind it, matched by package name.
    pub packages: &'static [&'static str],
    /// Native libraries that bind it, matched by a package's `links` key.
    pub links: &'static [&'static str],
    /// The one workspace crate allowed to reach it.
    pub only_in: &'static str,
    /// Whether only a member's own normal or build dependency on a binding counts. Set for a
    /// system whose binding most crates reach through other packages, such as `getrandom` under
    /// the TLS stack and test crates: the rule then names the one crate that depends on it
    /// directly, and a binding reached through another package is not a finding.
    pub direct_only: bool,
}

/// Rule 6: one crate per external system. A system with no binding crate listed is not checked
/// until the task that adopts one lists it here. IoRing and IOCP are calls through the Windows API
/// crates that the async ecosystem also uses, so a package name cannot single them out, and the
/// Thunderbolt RDMA API has no binding crate yet. `getrandom` is checked on direct dependencies
/// only (RFC-0001 §4.2 rule 6's 2026-10-22 note).
pub const SYSTEMS: &[System] = &[
    System {
        name: "iroh",
        // iroh 1.x's own crates (its QUIC is `noq`, with `noq-udp` its socket layer), plus the pre-1.0 names `iroh-net` and
        // `iroh-quinn`, so an older binding is caught too (DIST-E7-T23 pins iroh 1.4.1).
        packages: &[
            "iroh",
            "iroh-base",
            "iroh-relay",
            "iroh-dns",
            "noq",
            "noq-proto",
            "noq-udp",
            "iroh-net",
            "iroh-quinn",
        ],
        links: &[],
        only_in: "net",
        direct_only: false,
    },
    System {
        name: "ibverbs",
        packages: &["ibverbs", "ibverbs-sys", "rdma-sys"],
        links: &[],
        only_in: "net",
        direct_only: false,
    },
    System {
        name: "io_uring",
        packages: &["io-uring", "tokio-uring", "glommio", "monoio"],
        links: &[],
        only_in: "io",
        direct_only: false,
    },
    System {
        name: "HiGHS",
        packages: &["highs", "highs-sys"],
        links: &[],
        only_in: "solver",
        direct_only: false,
    },
    System {
        name: "ggml",
        packages: &["ggml"],
        links: &["ggml"],
        only_in: "kernels-ggml",
        direct_only: false,
    },
    System {
        name: "the CUDA runtime",
        packages: &["cudarc", "cust"],
        links: &["cuda"],
        only_in: "kernels-cuda",
        direct_only: false,
    },
    System {
        name: "Vulkan",
        packages: &["ash", "vulkano"],
        links: &[],
        only_in: "kernels-vulkan",
        direct_only: false,
    },
    System {
        name: "Metal",
        packages: &["metal", "objc2-metal"],
        links: &[],
        only_in: "kernels-metal",
        direct_only: false,
    },
    // ADR-0022 decision 3: `OsEntropy` in `io` is the one crate that names the OS CSPRNG.
    System {
        name: "the OS random source",
        packages: &["getrandom"],
        links: &[],
        only_in: "io",
        direct_only: true,
    },
];

/// Rule 7: the core binary is this bin target of this crate.
pub const CORE_BINARY_CRATE: &str = "cli";
pub const CORE_BINARY: &str = "syntropy";

/// Rule 7: the shared libraries the core binary may need on Linux, by stem: the C library and its
/// parts, and the unwinder. `libc` matches `libc.so` and `libc.so.6`. Rule 7 rejects every other
/// `DT_NEEDED` entry except the dynamic loader, so a vendor runtime that [`VENDOR_RUNTIMES`] does
/// not know still fails. A library added here must be a system library the core binary needs on
/// every Linux host, never a vendor GPU runtime (ADR-0018 decision 5), and the change needs a
/// maintainer's review.
pub const SYSTEM_LIBRARIES: &[&str] = &["libc", "libm", "libpthread", "libdl", "librt", "libgcc_s"];

/// Rule 7: the dynamic loader, named per architecture: `ld-linux-x86-64.so.2`,
/// `ld-linux-aarch64.so.1`.
pub const LOADER_PREFIX: &str = "ld-linux";

/// Rule 7: the DLLs the core binary may import or delay-load on Windows, compared without case.
/// The list is what the core binary, built with no default features for `x86_64-pc-windows-msvc`,
/// imports on the hosted `windows-2025` runner (QUAL-E1-T28): the Windows kernel (`kernel32`,
/// `ntdll`, and the API set of the wait-on-address calls the standard library's locks use) and the
/// Visual C++ runtime. Rule 7 rejects every other DLL but the Universal CRT's API sets on
/// [`WINDOWS_CRT_API_SETS`]. A DLL added here must ship with every supported Windows, or with
/// the Visual C++ runtime, never a vendor GPU runtime (ADR-0018 decision 5), and the change needs a
/// maintainer's review.
///
/// Added 2026-10-27, when the core binary gained the CLI's client and the API (PROD-E6-T01), from
/// what rule 7 reported on the hosted `windows-2025` runner:
/// - `ws2_32.dll`, Winsock: TCP sockets for the client and the API's listeners;
/// - `advapi32.dll`: the security calls behind `syntropy-ipc`'s owner-only pipe and peer check
///   (`ConvertStringSecurityDescriptorToSecurityDescriptorW`, `OpenProcessToken`,
///   `GetTokenInformation`, `EqualSid`, `ConvertSidToStringSidW`);
/// - `bcryptprimitives.dll`: `ProcessPrng`, the system random source the standard library and
///   the OS entropy crates call.
pub const WINDOWS_SYSTEM_LIBRARIES: &[&str] = &[
    "kernel32.dll",
    "ntdll.dll",
    "api-ms-win-core-synch-l1-2-0.dll",
    "vcruntime140.dll",
    "ws2_32.dll",
    "advapi32.dll",
    "bcryptprimitives.dll",
];

/// Rule 7: the Universal CRT's API sets the core binary may import, compared without case. The
/// list is the six the core binary imports on the hosted `windows-2025` runner (QUAL-E1-T28,
/// `llvm-readobj --coff-imports`). Every Universal CRT set ships with Windows 10 and later, but
/// only these are allowed, so a dependency that pulls in another one, such as `time`, fails rule 7
/// until a maintainer adds it here, as for [`WINDOWS_SYSTEM_LIBRARIES`].
///
/// `api-ms-win-crt-utility-l1-1-0.dll` was added 2026-10-27 (PROD-E6-T01), when the core binary
/// gained the API: it holds the C runtime's `qsort`, `rand` and similar, which the bundled C code
/// of a dependency calls (which dependency is unconfirmed).
pub const WINDOWS_CRT_API_SETS: &[&str] = &[
    "api-ms-win-crt-heap-l1-1-0.dll",
    "api-ms-win-crt-locale-l1-1-0.dll",
    "api-ms-win-crt-math-l1-1-0.dll",
    "api-ms-win-crt-runtime-l1-1-0.dll",
    "api-ms-win-crt-stdio-l1-1-0.dll",
    "api-ms-win-crt-string-l1-1-0.dll",
    "api-ms-win-crt-utility-l1-1-0.dll",
];

/// Rule 7: the libraries the core binary may load on macOS, by install name. `libSystem` is what
/// the core binary, built with no default features, loads on the hosted `macos-26` runner
/// (QUAL-E1-T28). `libobjc` is the Objective-C runtime, which the Metal provider's planned
/// Objective-C binding (`objc2-metal`, RFC-0040 §7.2) calls; that provider is linked into the core
/// (RFC-0001 §4.2 rule 7). No Metal provider is built yet, so `libobjc` and [`MACOS_FRAMEWORKS`]
/// are a forecast from that binding, not a measurement: re-derive both from the real binary when
/// COMP-E6-T01 lands it, and drop any entry it does not load. Rule 7 rejects every other library but [`MACOS_FRAMEWORKS`]. A library added here must
/// ship with every supported macOS, never a vendor GPU runtime (ADR-0018 decision 5), and the
/// change needs a maintainer's review.
///
/// `libiconv` was added 2026-10-27 (PROD-E6-T01), when the core binary gained the CLI's client
/// and the API, from what rule 7 reported on the hosted `macos-26` runner. It ships with every
/// macOS. Which dependency links it is unconfirmed.
pub const MACOS_SYSTEM_LIBRARIES: &[&str] = &[
    "/usr/lib/libSystem.B.dylib",
    "/usr/lib/libobjc.A.dylib",
    "/usr/lib/libiconv.2.dylib",
];

/// Rule 7: the system frameworks the core binary may load on macOS, by name: `Metal` matches
/// `/System/Library/Frameworks/Metal.framework/Versions/A/Metal`. The Metal provider is linked into
/// the core, and its binding calls `MTLCreateSystemDefaultDevice` from `Metal` (RFC-0040 §7.2).
/// `Foundation` and `CoreFoundation` are the frameworks an Objective-C binding's object types
/// come from, and `CoreGraphics` is the one `MTLCreateSystemDefaultDevice` needs linked to find a
/// device on macOS. RFC-0040 names only the binding and that call, so the three are a forecast, as
/// [`MACOS_SYSTEM_LIBRARIES`] says. The same review rule applies.
///
/// `SystemConfiguration` and `Security` were added 2026-10-27 (PROD-E6-T01), from what rule 7
/// reported on the hosted `macos-26` runner once the core binary gained the API. Both ship with
/// macOS: the network configuration and change notices, and the keychain and trust store, which
/// the networking stack under `syntropy-net` reads (which crate links each is unconfirmed).
pub const MACOS_FRAMEWORKS: &[&str] = &[
    "Metal",
    "Foundation",
    "CoreFoundation",
    "CoreGraphics",
    "SystemConfiguration",
    "Security",
];

/// Where the system frameworks live.
const MACOS_FRAMEWORK_DIR: &str = "/System/Library/Frameworks/";

/// Rule 7: one vendor GPU runtime, as it shows in a binary's dynamic linkage. The library names
/// only name the vendor in a finding; the allow-lists above decide which libraries fail. The
/// symbols are a second check, for a runtime linked in a way that names no library of its own.
pub struct VendorRuntime {
    pub name: &'static str,
    /// Shared-library stems: `libcuda` matches `libcuda.so` and `libcuda.so.1` on Linux, and
    /// `libcuda.dylib` and `@rpath/libcuda.1.dylib` on macOS.
    pub libraries: &'static [&'static str],
    /// DLL stems, compared without case: `cudart` matches `cudart64_12.dll`, `vulkan` matches
    /// `vulkan-1.dll`.
    pub dlls: &'static [&'static str],
    /// Prefixes of the runtime's own entry points, too short to match alone: one matches when
    /// the next character is an ASCII capital or `_`, so `cu` matches `cuInit` and not `cupcake`.
    pub entry_prefixes: &'static [&'static str],
    /// Prefixes of the vendor's library families, such as `cufft`: any symbol that starts with one
    /// matches.
    pub families: &'static [&'static str],
}

/// Rule 7: vendor GPU runtimes and libraries, which reach a host only inside backend modules.
pub const VENDOR_RUNTIMES: &[VendorRuntime] = &[
    VendorRuntime {
        name: "CUDA",
        libraries: &[
            "libcuda",
            "libcudart",
            "libcublas",
            "libcublasLt",
            "libcufft",
            "libcurand",
            "libcusparse",
            "libcusolver",
            "libcudnn",
            "libnccl",
            "libnvrtc",
            "libnvJitLink",
            "libnvidia-ml",
        ],
        dlls: &[
            "nvcuda",
            "cudart",
            "cublas",
            "cublasLt",
            "cufft",
            "curand",
            "cusparse",
            "cusolver",
            "cudnn",
            "nvrtc",
            "nvJitLink",
            "nvml",
        ],
        entry_prefixes: &["cu"],
        families: &[
            "cuda",
            "__cuda",
            "cublas",
            "cufft",
            "curand",
            "cusparse",
            "cusolver",
            "cudnn",
            "nccl",
            "nvrtc",
            "nvJitLink",
            "nvml",
        ],
    },
    VendorRuntime {
        name: "HIP",
        libraries: &[
            "libamdhip64",
            "libhiprtc",
            "librocblas",
            "libhipblas",
            "libhipblaslt",
            "librocfft",
            "libhipfft",
            "librocrand",
            "libhiprand",
            "librocsparse",
            "libhipsparse",
            "librccl",
            "libMIOpen",
            "librocm_smi64",
        ],
        dlls: &[
            "amdhip64",
            "hiprtc",
            "rocblas",
            "hipblas",
            "hipblaslt",
            "rocfft",
            "hipfft",
            "rocrand",
            "hiprand",
            "rocsparse",
            "hipsparse",
            "MIOpen",
        ],
        entry_prefixes: &["hip"],
        families: &[
            "hipblas",
            "hipfft",
            "hiprand",
            "hipsparse",
            "hiprtc",
            "rocblas",
            "rocfft",
            "rocrand",
            "rocsparse",
            "rccl",
            "miopen",
            "rsmi",
        ],
    },
    VendorRuntime {
        name: "Vulkan",
        libraries: &["libvulkan", "libMoltenVK"],
        dlls: &["vulkan"],
        entry_prefixes: &["vk"],
        families: &[],
    },
    VendorRuntime {
        name: "oneAPI",
        libraries: &["libze_loader", "libsycl"],
        dlls: &["ze_loader", "sycl"],
        entry_prefixes: &["ze"],
        families: &[],
    },
    VendorRuntime {
        name: "OpenCL",
        libraries: &["libOpenCL"],
        dlls: &["OpenCL"],
        entry_prefixes: &["cl"],
        families: &[],
    },
];

/// Whether the `DT_NEEDED` entry `needed` is the shared library with stem `stem`.
pub fn is_library(needed: &str, stem: &str) -> bool {
    needed
        .strip_prefix(stem)
        .is_some_and(|rest| rest == ".so" || rest.starts_with(".so."))
}

/// Rule 7: whether the core binary may need `needed`: a system library or the dynamic loader.
pub fn is_system_library(needed: &str) -> bool {
    SYSTEM_LIBRARIES.iter().any(|stem| is_library(needed, stem))
        || needed
            .strip_prefix(LOADER_PREFIX)
            .is_some_and(|rest| rest.starts_with(['-', '.']))
}

/// Whether the DLL `dll` has stem `stem`, without case: the stem, then nothing or a version that
/// starts with a digit, `-` or `_`, then `.dll`.
pub fn is_dll(dll: &str, stem: &str) -> bool {
    let dll = dll.to_ascii_lowercase();
    let stem = stem.to_ascii_lowercase();
    dll.strip_suffix(".dll")
        .and_then(|base| base.strip_prefix(stem.as_str()))
        .is_some_and(|rest| {
            rest.is_empty()
                || rest.starts_with(|c: char| c.is_ascii_digit() || c == '-' || c == '_')
        })
}

/// Rule 7: whether the core binary may import `dll` on Windows.
pub fn is_windows_system_library(dll: &str) -> bool {
    let lower = dll.to_ascii_lowercase();
    WINDOWS_SYSTEM_LIBRARIES.contains(&lower.as_str())
        || WINDOWS_CRT_API_SETS.contains(&lower.as_str())
}

/// Whether the install name `name` is the dynamic library with stem `stem`: its last path
/// component is the stem, then an optional version, then `.dylib`.
pub fn is_dylib(name: &str, stem: &str) -> bool {
    let file = name.rsplit('/').next().unwrap_or(name);
    file.strip_prefix(stem)
        .and_then(|rest| rest.strip_suffix(".dylib"))
        .is_some_and(|version| version.is_empty() || version.starts_with('.'))
}

/// Rule 7: whether the core binary may load the install name `name` on macOS: a system library
/// or a system framework on the lists above.
pub fn is_macos_system_library(name: &str) -> bool {
    if MACOS_SYSTEM_LIBRARIES.contains(&name) {
        return true;
    }
    let Some(rest) = name.strip_prefix(MACOS_FRAMEWORK_DIR) else {
        return false;
    };
    MACOS_FRAMEWORKS.iter().any(|framework| {
        rest.strip_prefix(framework)
            .and_then(|rest| rest.strip_prefix(".framework/"))
            .is_some_and(|binary| {
                binary == *framework
                    || binary
                        .strip_prefix("Versions/")
                        .and_then(|rest| rest.split_once('/'))
                        .is_some_and(|(version, file)| {
                            !version.is_empty() && !version.contains('.') && file == *framework
                        })
            })
    })
}

/// Rule 8: crates allowed `unsafe`, besides every `kernels-*` crate. `spsc` was added on
/// 2026-10-22 (RFC-0001 §4.2 rule 8).
pub const UNSAFE_ALLOWED: &[&str] = &["io", "tier", "net", "ipc", "spsc"];

/// Rule 8's module-scoped exception (RFC-0001 §4.2 rule 8, note 2026-10-22): a crate that sets
/// `#![deny(unsafe_code)]` at its root instead of `forbid`, and whose only `unsafe` is one
/// `unsafe impl` of one trait, in one file, under the crate's one `#[allow(unsafe_code)]`.
pub struct UnsafeModule {
    /// The crate's short name, such as `testkit`.
    pub krate: &'static str,
    /// The file that holds the impl, relative to the crate's directory.
    pub file: &'static str,
    /// The trait the impl implements.
    pub implements: &'static str,
}

/// The crates that hold a module-scoped `unsafe` exception. `testkit::guards` holds the counting
/// allocator, a `GlobalAlloc` that delegates to `std::alloc::System` (RFC-0060 §3.5).
pub const UNSAFE_MODULES: &[UnsafeModule] = &[UnsafeModule {
    krate: "testkit",
    file: "src/guards.rs",
    implements: "GlobalAlloc",
}];

/// Rule 8: the module-scoped exception `name` holds, if any.
pub fn unsafe_module(name: &str) -> Option<&'static UnsafeModule> {
    UNSAFE_MODULES.iter().find(|module| module.krate == name)
}

/// Rule 10: the crates release artefacts are built from: the core binary, and the provider
/// libraries of the backend modules (rule 7). The check resolves each one's features on its own,
/// as its artefact is built.
pub const RELEASE_CRATES: &[&str] = &["cli", "kernels-cuda", "kernels-hip", "kernels-vulkan"];

/// Rule 10: test-only features, which never reach a release.
pub const TEST_FEATURES: &[&str] = &["fault-injection", "debug-taps", "guards"];

/// Whether `name` is a `kernels-*` crate.
pub fn is_provider(name: &str) -> bool {
    name.starts_with(PROVIDER_PREFIX)
}

/// Rule 3: whether `name` is a compute crate.
pub fn is_compute(name: &str) -> bool {
    is_provider(name) || COMPUTE.contains(&name)
}

/// Rule 8: whether `name` may use `unsafe`.
pub fn allows_unsafe(name: &str) -> bool {
    is_provider(name) || UNSAFE_ALLOWED.contains(&name)
}

/// Every workspace crate this module names, for the test that ties it to `ci/layers.toml`.
#[cfg(test)]
pub fn named_crates() -> Vec<&'static str> {
    let mut names = vec![CORE, DAEMON, SOLVER, CORE_BINARY_CRATE];
    names.extend(SHARED);
    names.extend(
        MACRO_CRATES
            .iter()
            .flat_map(|&(owner, macros)| [owner, macros]),
    );
    names.extend(
        ON_SAME_LAYER
            .iter()
            .flat_map(|&(shared, users)| std::iter::once(shared).chain(users.iter().copied())),
    );
    names.extend(COMPUTE);
    names.extend(COMPUTE_MUST_NOT_REACH);
    names.extend(ADR_0012_NO_TOKIO);
    names.extend(SOLVER_MUST_NOT_REACH);
    names.extend(SYSTEMS.iter().map(|system| system.only_in));
    names.extend(UNSAFE_ALLOWED);
    names.extend(UNSAFE_MODULES.iter().map(|module| module.krate));
    names.extend(RELEASE_CRATES);
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernels_itself_is_compute_but_not_a_provider_and_has_no_unsafe() {
        assert!(is_compute("kernels"));
        assert!(!is_provider("kernels"));
        assert!(!allows_unsafe("kernels"));
        for provider in ["kernels-cuda", "kernels-loader", "kernels-ggml"] {
            assert!(is_provider(provider) && is_compute(provider) && allows_unsafe(provider));
        }
    }

    #[test]
    fn spsc_is_shared_inside_layer_1_allows_unsafe_and_is_compute() {
        assert_eq!(same_layer_users("spsc"), Some(&["io", "net", "ipc"][..]));
        assert_eq!(same_layer_users("ipc"), None);
        assert!(allows_unsafe("spsc"));
        assert!(is_compute("spsc"));
    }

    #[test]
    fn only_testkit_holds_a_module_scoped_unsafe_exception() {
        let module = unsafe_module("testkit").expect("testkit has the exception");
        assert_eq!(
            (module.file, module.implements),
            ("src/guards.rs", "GlobalAlloc")
        );
        // No other crate has one, and the exception does not put testkit on the allow-list.
        for name in ["solver", "bench", "core", "cli", "kernels", "runner"] {
            assert!(unsafe_module(name).is_none(), "{name}");
        }
        assert!(!allows_unsafe("testkit"));
    }

    #[test]
    fn control_crates_are_not_compute() {
        for name in [
            "net", "cluster", "store", "api", "daemon", "solver", "runner", "cli",
        ] {
            assert!(!is_compute(name), "{name}");
        }
    }

    #[test]
    fn the_system_allow_list_matches_whole_stems_and_the_loader() {
        for needed in [
            "libc.so.6",
            "libm.so.6",
            "libgcc_s.so.1",
            "libpthread.so.0",
            "libdl.so",
            "ld-linux-x86-64.so.2",
            "ld-linux-aarch64.so.1",
            "ld-linux.so.2",
        ] {
            assert!(is_system_library(needed), "{needed}");
        }
        for needed in [
            "libcuda.so.1",
            "libcufft.so",
            "libnccl.so.2",
            "libcrypt.so.1",
            "libmagic.so.1",
            "libstdc++.so.6",
            "ld-linuxish.so",
            "libc",
        ] {
            assert!(!is_system_library(needed), "{needed}");
        }
    }

    #[test]
    fn the_windows_allow_list_matches_whole_names_without_case() {
        for dll in [
            "KERNEL32.dll",
            "ntdll.dll",
            "api-ms-win-core-synch-l1-2-0.dll",
            "VCRUNTIME140.dll",
            "api-ms-win-crt-runtime-l1-1-0.dll",
            "API-MS-WIN-CRT-HEAP-L1-1-0.DLL",
            "WS2_32.dll",
            "advapi32.dll",
            "bcryptprimitives.dll",
            "api-ms-win-crt-utility-l1-1-0.dll",
        ] {
            assert!(is_windows_system_library(dll), "{dll}");
        }
        for dll in [
            "nvcuda.dll",
            "vulkan-1.dll",
            "kernel32.dll.evil.dll",
            "kernel32",
            "api-ms-win-crt-.dll",
            "api-ms-win-crt-evil-l1-1-0.dll",
            "api-ms-win-crt-time-l1-1-0.dll",
            "api-ms-win-core-file-l1-1-0.dll",
            "bcrypt.dll",
            "ws2_32.dll.evil.dll",
            "libc.so.6",
        ] {
            assert!(!is_windows_system_library(dll), "{dll}");
        }
    }

    #[test]
    fn dll_stems_match_a_version_but_not_a_longer_name() {
        assert!(is_dll("nvcuda.dll", "nvcuda"));
        assert!(is_dll("NVCUDA.DLL", "nvcuda"));
        assert!(is_dll("cudart64_12.dll", "cudart"));
        assert!(is_dll("cublasLt64_12.dll", "cublasLt"));
        assert!(is_dll("vulkan-1.dll", "vulkan"));
        assert!(!is_dll("cublasLt64_12.dll", "cublas"));
        assert!(!is_dll("nvcudax.dll", "nvcuda"));
        assert!(!is_dll("nvcuda.so", "nvcuda"));
    }

    #[test]
    fn the_macos_allow_list_matches_install_names_and_framework_layouts() {
        for name in [
            "/usr/lib/libSystem.B.dylib",
            "/usr/lib/libobjc.A.dylib",
            "/System/Library/Frameworks/Metal.framework/Versions/A/Metal",
            "/System/Library/Frameworks/Metal.framework/Metal",
            "/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation",
            "/System/Library/Frameworks/Foundation.framework/Versions/C/Foundation",
            "/System/Library/Frameworks/CoreGraphics.framework/Versions/A/CoreGraphics",
            "/usr/lib/libiconv.2.dylib",
            "/System/Library/Frameworks/SystemConfiguration.framework/Versions/A/SystemConfiguration",
            "/System/Library/Frameworks/Security.framework/Versions/A/Security",
        ] {
            assert!(is_macos_system_library(name), "{name}");
        }
        for name in [
            "@rpath/libvulkan.1.dylib",
            "/usr/local/lib/libMoltenVK.dylib",
            "/usr/lib/libSystem.dylib",
            "/usr/lib/libc++.1.dylib",
            "/System/Library/Frameworks/OpenCL.framework/Versions/A/OpenCL",
            "/System/Library/Frameworks/Metal.framework/Versions/A/Other",
            "/System/Library/Frameworks/Metal.framework/Versions/../Metal",
            "/System/Library/Frameworks/Metal.framework/../../../../tmp/Metal",
            "/Library/Frameworks/Metal.framework/Versions/A/Metal",
            "@rpath/Metal.framework/Metal",
            "libSystem.B.dylib",
        ] {
            assert!(!is_macos_system_library(name), "{name}");
        }
    }

    #[test]
    fn dylib_stems_match_the_last_component_and_a_version() {
        assert!(is_dylib("@rpath/libvulkan.1.dylib", "libvulkan"));
        assert!(is_dylib("/usr/local/lib/libvulkan.dylib", "libvulkan"));
        assert!(is_dylib("libMoltenVK.dylib", "libMoltenVK"));
        assert!(!is_dylib("@rpath/libvulkanx.dylib", "libvulkan"));
        assert!(!is_dylib("/opt/libvulkan/libother.dylib", "libvulkan"));
        assert!(!is_dylib("libvulkan.so.1", "libvulkan"));
    }

    #[test]
    fn each_system_names_at_least_one_binding() {
        for system in SYSTEMS {
            assert!(
                !system.packages.is_empty() || !system.links.is_empty(),
                "{}",
                system.name
            );
        }
    }
}
