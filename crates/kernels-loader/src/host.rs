// SPDX-License-Identifier: MIT OR Apache-2.0

//! What the core offers a module: [`SynHostApi`]'s `log`, `metric` and `config` (RFC-0040
//! §10.4). The C callbacks carry no user pointer, so the services live in one process-wide slot.

use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex, PoisonError, RwLock};

use syntropy_kernels::abi::{SynHostApi, types::SynHeader};

/// The log sink: `(level, target, message)`, level 0 error … 4 trace.
type LogFn = dyn Fn(u8, &str, &str) + Send + Sync;
/// The metric sink: `(id, value)`.
type MetricFn = dyn Fn(u32, u64) + Send + Sync;
/// The config lookup: a key such as `features.disable`, and its value if set.
type ConfigFn = dyn Fn(&str) -> Option<String> + Send + Sync;

/// The core's services for modules. Install them with [`HostServices::install`] before loading
/// a module.
#[derive(Clone)]
pub struct HostServices {
    /// The core's build ID (ADR-0507 rule 3), stated in the header of the host table.
    pub build_id: [u8; 32],
    log: Arc<LogFn>,
    metric: Arc<MetricFn>,
    config: Arc<ConfigFn>,
}

impl HostServices {
    /// Services with the given sinks.
    #[must_use]
    pub fn new(
        build_id: [u8; 32],
        log: impl Fn(u8, &str, &str) + Send + Sync + 'static,
        metric: impl Fn(u32, u64) + Send + Sync + 'static,
        config: impl Fn(&str) -> Option<String> + Send + Sync + 'static,
    ) -> Self {
        Self {
            build_id,
            log: Arc::new(log),
            metric: Arc::new(metric),
            config: Arc::new(config),
        }
    }

    /// Services that drop every log line and metric and set no config key.
    #[must_use]
    pub fn silent(build_id: [u8; 32]) -> Self {
        Self::new(build_id, |_, _, _| {}, |_, _| {}, |_| None)
    }

    /// Makes these the services every module sees, replacing earlier ones.
    pub fn install(self) {
        *SERVICES.write().unwrap_or_else(PoisonError::into_inner) = Some(self);
    }

    pub(crate) fn current() -> Option<Self> {
        SERVICES
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// The host table to pass to a module's entry symbol.
    pub(crate) fn table(&self) -> SynHostApi {
        SynHostApi {
            h: SynHeader::new(self.build_id, size_of::<SynHostApi>()),
            log,
            metric,
            config,
        }
    }
}

static SERVICES: RwLock<Option<HostServices>> = RwLock::new(None);

/// The `CString`s `config` has returned. A module reads the pointer during the call, but the
/// core keeps each value alive for the process, which is the contract of `SynHostApi::config`.
static CONFIG_VALUES: Mutex<Option<ConfigValues>> = Mutex::new(None);

/// The live value of each key, and the values it replaced: a module may still hold a pointer to
/// the old one.
#[derive(Default)]
struct ConfigValues {
    live: HashMap<String, CString>,
    retired: Vec<CString>,
}

fn services() -> Option<HostServices> {
    HostServices::current()
}

fn text(p: *const c_char) -> Option<String> {
    if p.is_null() {
        return None;
    }
    // SAFETY: the ABI contract makes a module pass NUL-terminated strings.
    Some(unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
}

extern "C" fn log(level: u8, target: *const c_char, msg: *const c_char, len: usize) {
    // A panic must not unwind into the module's frames.
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let Some(services) = services() else { return };
        let target = text(target).unwrap_or_default();
        let message = if msg.is_null() {
            String::new()
        } else {
            // SAFETY: the ABI contract makes the module pass `len` readable bytes at `msg`.
            String::from_utf8_lossy(unsafe { std::slice::from_raw_parts(msg.cast::<u8>(), len) })
                .into_owned()
        };
        (services.log)(level, &target, &message);
    }));
}

extern "C" fn metric(id: u32, value: u64) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if let Some(services) = services() {
            (services.metric)(id, value);
        }
    }));
}

extern "C" fn config(key: *const c_char) -> *const c_char {
    catch_unwind(AssertUnwindSafe(|| {
        let key = text(key)?;
        let value = (services()?.config)(&key)?;
        let value = CString::new(value).ok()?;
        let mut values = CONFIG_VALUES.lock().unwrap_or_else(PoisonError::into_inner);
        let values = values.get_or_insert_with(ConfigValues::default);
        if values.live.get(&key) != Some(&value)
            && let Some(old) = values.live.insert(key.clone(), value)
        {
            values.retired.push(old);
        }
        values.live.get(&key).map(|v| v.as_ptr())
    }))
    .ok()
    .flatten()
    .unwrap_or(std::ptr::null())
}
