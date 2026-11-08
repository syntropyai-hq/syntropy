// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared by the boundary tests: where the test provider library is, a copy of it per case, and
//! the core's services, steered per thread.
//!
//! The module calls `config` on the thread that calls into it, so a thread-local table gives
//! every test its own `fixture.*` settings while the services stay process-wide.

#![allow(dead_code)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Once;
use std::sync::atomic::{AtomicUsize, Ordering};

use syntropy_kernels_loader::HostServices;

thread_local! {
    static CONFIG: RefCell<HashMap<String, String>> = RefCell::new(HashMap::new());
    /// The metrics the module sent on this thread, by ID, so a test reads only its own calls.
    static METRICS: RefCell<HashMap<u32, u64>> = RefCell::new(HashMap::new());
}

/// The test provider's metric IDs for its release work (`examples/test_provider.rs`).
pub const FREED: u32 = 101;
pub const CONTEXT_CLOSED: u32 = 102;
pub const REGISTRAR_CLOSED: u32 = 103;

/// How much the module has sent of metric `id` on the calling thread.
pub fn metric(id: u32) -> u64 {
    METRICS.with(|m| m.borrow().get(&id).copied().unwrap_or(0))
}

/// Log lines and metrics the module sent, process-wide.
pub static LOG_LINES: AtomicUsize = AtomicUsize::new(0);
pub static METRIC_TOTAL: AtomicUsize = AtomicUsize::new(0);

/// The build ID the test provider states in its header.
pub const MODULE_BUILD_ID: [u8; 32] = [7; 32];

/// Sets `fixture.<key>` for the calling thread.
pub fn set(key: &str, value: &str) {
    CONFIG.with(|c| {
        c.borrow_mut()
            .insert(format!("fixture.{key}"), value.to_owned())
    });
}

/// Clears `fixture.<key>` for the calling thread.
pub fn clear(key: &str) {
    CONFIG.with(|c| c.borrow_mut().remove(&format!("fixture.{key}")));
}

/// Installs the services once. The core's build ID differs from the module's, to show the entry
/// call does not depend on it.
pub fn install_services() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        HostServices::new(
            [9; 32],
            |_level, _target, _msg| {
                LOG_LINES.fetch_add(1, Ordering::Relaxed);
            },
            |id, value| {
                METRIC_TOTAL.fetch_add(value as usize, Ordering::Relaxed);
                METRICS.with(|m| *m.borrow_mut().entry(id).or_insert(0) += value);
            },
            |key| CONFIG.with(|c| c.borrow().get(key).cloned()),
        )
        .install();
    });
}

/// The file name of a `cdylib` named `name` on this platform.
fn dylib_name(name: &str) -> String {
    format!(
        "{}{name}{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    )
}

/// The built test provider library. `cargo test -p syntropy-kernels-loader` builds the example
/// before it runs the tests.
pub fn provider_library() -> PathBuf {
    let exe = std::env::current_exe().expect("the test binary has a path");
    // The test binary sits in target/<profile>/deps, or deeper under target/<profile>/build with
    // cargo's newer layout; the examples directory is beside one of its ancestors.
    let name = dylib_name("test_provider");
    exe.ancestors()
        .map(|dir| dir.join("examples").join(&name))
        .find(|lib| lib.is_file())
        .unwrap_or_else(|| {
            panic!(
                "no examples/{name} above {}: build the example with `cargo test -p \
                 syntropy-kernels-loader` (which builds it) or `cargo build -p \
                 syntropy-kernels-loader --example test_provider`",
                exe.display()
            )
        })
}

/// A fresh directory for one test, removed when dropped.
pub struct TempDir(pub PathBuf);

impl TempDir {
    pub fn new(tag: &str) -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "syntropy-loader-{}-{}-{tag}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).expect("a temp directory");
        Self(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A copy of the test provider library under a new name. The platform loader treats it as a
/// separate library, so its provider has its own state: a provider made `Sticky` in one test
/// does not affect another.
pub fn private_copy(dir: &TempDir) -> PathBuf {
    let copy = dir.0.join(dylib_name("test_provider_copy"));
    std::fs::copy(provider_library(), &copy).expect("copy the test provider library");
    copy
}
