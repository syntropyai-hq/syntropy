// SPDX-License-Identifier: MIT OR Apache-2.0

//! Apple GPUs through IOKit's `IOAccelerator` (RFC-0010 §2.2's "Apple GPU" row): the model and
//! `gpu-core-count`, which fills `DeviceInfoExt::gpu_cores` (ADR-0405 decision 4). One chip name
//! ships with several core counts, so the count is read, never derived from the name.
//!
//! The Metal facts (`recommendedMaxWorkingSetSize`, `supportsFamily`, …) are E5's: reading them
//! creates an `MTLDevice`, which opens the GPU (RFC-0010 §2.2).
//!
//! [`live_accelerators`] reads the registry on macOS, and [`render_fixture`] writes what it read
//! as fixture text, which [`parse_fixture`] reads back, so a recorded host runs the same mapping
//! ([`apple_gpus`]). The format is one accelerator per line; `#` starts a comment:
//!
//! ```text
//! accelerator class=<IOClass>;model=<text>;gpu-core-count=<n>
//! ```
//!
//! A field left out is a property the registry entry does not have.

use syntropy_core::resource::{DeviceInfoExt, DeviceKind};

use super::GpuRecord;

/// One `IOAccelerator` registry entry's properties.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Accelerator {
    /// The entry's class, such as `AGXAcceleratorG16X`.
    pub class: String,
    /// `model`, such as `Apple M4 Pro`.
    pub model: Option<String>,
    /// `gpu-core-count`.
    pub gpu_core_count: Option<i64>,
}

/// The GPU records of `accelerators`: one integrated, unified-memory GPU each, named by its
/// model and carrying its core count. A count outside `u16` is left unknown.
#[must_use]
pub fn apple_gpus(accelerators: &[Accelerator]) -> Vec<GpuRecord> {
    accelerators
        .iter()
        .map(|a| GpuRecord {
            kind: DeviceKind::IntegratedGpu,
            name: a.model.clone().unwrap_or_else(|| a.class.clone()),
            unified: true,
            numa: None,
            vram_total: None,
            vram_budget: None,
            bar1_total: None,
            temperature_c: None,
            nvlinks: 0,
            peers: Vec::new(),
            ext: DeviceInfoExt {
                gpu_cores: a.gpu_core_count.and_then(|n| u16::try_from(n).ok()),
                ..DeviceInfoExt::default()
            },
        })
        .collect()
}

/// `accelerators` as fixture text (module docs).
#[must_use]
pub fn render_fixture(accelerators: &[Accelerator]) -> String {
    let mut out = String::new();
    for a in accelerators {
        let mut fields = vec![format!("class={}", a.class)];
        if let Some(model) = &a.model {
            fields.push(format!("model={model}"));
        }
        if let Some(n) = a.gpu_core_count {
            fields.push(format!("gpu-core-count={n}"));
        }
        out.push_str("accelerator ");
        out.push_str(&fields.join(";"));
        out.push('\n');
    }
    out
}

/// The accelerators a fixture records (module docs).
///
/// # Errors
///
/// The line and why, for a line that is not an `accelerator` record, a field without `=`, an
/// unknown field, a record without `class`, or a count that is not an integer.
pub fn parse_fixture(text: &str) -> Result<Vec<Accelerator>, String> {
    let mut out = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let at = |why: &str| format!("line {}: {why}", index + 1);
        let fields = line
            .strip_prefix("accelerator ")
            .ok_or_else(|| at("not an accelerator record"))?;
        let mut a = Accelerator::default();
        let mut class = None;
        for field in fields.split(';') {
            let (key, value) = field
                .split_once('=')
                .ok_or_else(|| at(&format!("{field}: no '='")))?;
            match key {
                "class" => class = Some(value.to_owned()),
                "model" => a.model = Some(value.to_owned()),
                "gpu-core-count" => {
                    a.gpu_core_count = Some(
                        value
                            .parse()
                            .map_err(|_| at(&format!("gpu-core-count {value}")))?,
                    );
                }
                other => return Err(at(&format!("unknown field {other}"))),
            }
        }
        a.class = class.ok_or_else(|| at("no class"))?;
        out.push(a);
    }
    Ok(out)
}

#[cfg(target_os = "macos")]
pub use live::live_accelerators;

#[cfg(target_os = "macos")]
mod live {
    use std::ffi::{CStr, c_char, c_void};
    use std::ptr;

    use super::Accelerator;

    type CFTypeRef = *const c_void;
    type CFTypeID = usize;
    type CFIndex = isize;
    type IoObject = u32;

    const K_CF_STRING_ENCODING_UTF8: u32 = 0x0800_0100;
    const K_CF_NUMBER_SINT64_TYPE: CFIndex = 4;
    /// `kIOMainPortDefault`.
    const MAIN_PORT_DEFAULT: u32 = 0;

    // SAFETY: the declarations of `<CoreFoundation/CFBase.h>`, `CFString.h` and `CFNumber.h`,
    // with the C types they use.
    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        fn CFRelease(cf: CFTypeRef);
        fn CFGetTypeID(cf: CFTypeRef) -> CFTypeID;
        safe fn CFStringGetTypeID() -> CFTypeID;
        safe fn CFNumberGetTypeID() -> CFTypeID;
        fn CFStringCreateWithCString(
            alloc: CFTypeRef,
            s: *const c_char,
            encoding: u32,
        ) -> CFTypeRef;
        fn CFStringGetCString(s: CFTypeRef, buf: *mut c_char, size: CFIndex, encoding: u32) -> u8;
        fn CFNumberGetValue(n: CFTypeRef, ty: CFIndex, out: *mut c_void) -> u8;
    }

    // SAFETY: the declarations of `<IOKit/IOKitLib.h>`, with the C types they use.
    #[link(name = "IOKit", kind = "framework")]
    unsafe extern "C" {
        fn IOServiceMatching(name: *const c_char) -> CFTypeRef;
        fn IOServiceGetMatchingServices(main: u32, matching: CFTypeRef, it: *mut IoObject) -> i32;
        fn IOIteratorNext(it: IoObject) -> IoObject;
        fn IOObjectRelease(object: IoObject) -> i32;
        fn IOObjectGetClass(object: IoObject, name: *mut c_char) -> i32;
        fn IORegistryEntryCreateCFProperty(
            entry: IoObject,
            key: CFTypeRef,
            alloc: CFTypeRef,
            options: u32,
        ) -> CFTypeRef;
    }

    /// A CoreFoundation object this code owns (Create rule), released on drop.
    struct Owned(CFTypeRef);

    impl Drop for Owned {
        fn drop(&mut self) {
            if !self.0.is_null() {
                // SAFETY: `self.0` came from a Create function, so this code holds one reference,
                // released exactly once here.
                unsafe { CFRelease(self.0) };
            }
        }
    }

    /// An IOKit object this code owns, released on drop.
    struct Object(IoObject);

    impl Drop for Object {
        fn drop(&mut self) {
            if self.0 != 0 {
                // SAFETY: `self.0` came from IOKit with one reference this code holds.
                unsafe { IOObjectRelease(self.0) };
            }
        }
    }

    /// The property `key` of `entry`, owned.
    fn property(entry: &Object, key: &CStr) -> Owned {
        // SAFETY: `key` is NUL-terminated; a null allocator is the default. Owned (Create rule).
        let k = Owned(unsafe {
            CFStringCreateWithCString(ptr::null(), key.as_ptr(), K_CF_STRING_ENCODING_UTF8)
        });
        if k.0.is_null() {
            return Owned(ptr::null());
        }
        // SAFETY: `entry` is a live registry entry and `k` a live CFString; the result is owned
        // (Create rule).
        Owned(unsafe { IORegistryEntryCreateCFProperty(entry.0, k.0, ptr::null(), 0) })
    }

    fn string(value: &Owned) -> Option<String> {
        // SAFETY: `value.0` is null or a live object this code owns.
        if value.0.is_null() || unsafe { CFGetTypeID(value.0) } != CFStringGetTypeID() {
            return None;
        }
        let mut buf = [0 as c_char; 256];
        // SAFETY: `value.0` is a CFString, checked above; `buf` is writable for the length
        // passed, and the call writes a NUL-terminated string within it.
        let ok = unsafe {
            CFStringGetCString(
                value.0,
                buf.as_mut_ptr(),
                buf.len() as CFIndex,
                K_CF_STRING_ENCODING_UTF8,
            )
        };
        if ok == 0 {
            return None;
        }
        // SAFETY: on success `buf` holds a NUL-terminated string.
        let text = unsafe { CStr::from_ptr(buf.as_ptr()) };
        Some(text.to_string_lossy().into_owned())
    }

    fn number(value: &Owned) -> Option<i64> {
        // SAFETY: `value.0` is null or a live object this code owns.
        if value.0.is_null() || unsafe { CFGetTypeID(value.0) } != CFNumberGetTypeID() {
            return None;
        }
        let mut n: i64 = 0;
        // SAFETY: `value.0` is a CFNumber, checked above, and `n` an `i64`, the size of
        // `kCFNumberSInt64Type`.
        let ok = unsafe { CFNumberGetValue(value.0, K_CF_NUMBER_SINT64_TYPE, (&raw mut n).cast()) };
        (ok != 0).then_some(n)
    }

    /// Every `IOAccelerator` registry entry's class, model and core count. Opens no GPU.
    ///
    /// # Errors
    ///
    /// The IOKit return code, when the registry cannot be searched.
    pub fn live_accelerators() -> Result<Vec<Accelerator>, String> {
        // SAFETY: the name is NUL-terminated. The dictionary is consumed by the next call.
        let matching = unsafe { IOServiceMatching(c"IOAccelerator".as_ptr()) };
        if matching.is_null() {
            return Err("IOServiceMatching returned no dictionary".to_owned());
        }
        let mut it: IoObject = 0;
        // SAFETY: `matching` is a live dictionary, which the call consumes; `it` is writable.
        let code =
            unsafe { IOServiceGetMatchingServices(MAIN_PORT_DEFAULT, matching, &raw mut it) };
        if code != 0 {
            return Err(format!("IOServiceGetMatchingServices returned {code}"));
        }
        let it = Object(it);
        let mut out = Vec::new();
        loop {
            // SAFETY: `it` is a live iterator; the entry is owned.
            let entry = Object(unsafe { IOIteratorNext(it.0) });
            if entry.0 == 0 {
                break;
            }
            // `io_name_t` is 128 bytes.
            let mut class = [0 as c_char; 128];
            // SAFETY: `entry` is live and `class` is an `io_name_t`.
            let class = if unsafe { IOObjectGetClass(entry.0, class.as_mut_ptr()) } == 0 {
                // SAFETY: on success `class` holds a NUL-terminated name.
                unsafe { CStr::from_ptr(class.as_ptr()) }
                    .to_string_lossy()
                    .into_owned()
            } else {
                String::new()
            };
            out.push(Accelerator {
                class,
                model: string(&property(&entry, c"model")),
                gpu_core_count: number(&property(&entry, c"gpu-core-count")),
            });
        }
        Ok(out)
    }
}
