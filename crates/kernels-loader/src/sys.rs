// SPDX-License-Identifier: MIT OR Apache-2.0

//! The system's dynamic loader, declared by hand: `dlopen`, `dlsym`, `dlclose` and `dlerror` on
//! Unix, `LoadLibraryExW`, `GetProcAddress` and `GetLastError` on Windows.
//!
//! This is the only file that talks to the platform loader. The declarations link through the C
//! library and kernel32, which std already links, so the crate needs no binding crate; swapping
//! in `libc` and `windows-sys` later changes this file alone.

use std::ffi::c_void;
use std::path::Path;

/// A handle the platform loader gave for an opened library.
pub struct RawLibrary(*mut c_void);

// SAFETY: a loader handle is an opaque token that the platform loader accepts on any thread.
unsafe impl Send for RawLibrary {}
// SAFETY: as above; `symbol` only reads through it, and the loaders serialise their own state.
unsafe impl Sync for RawLibrary {}

#[cfg(unix)]
mod imp {
    use std::ffi::{CStr, CString, c_char, c_int, c_void};
    use std::os::unix::ffi::OsStrExt;
    use std::path::Path;

    use super::RawLibrary;

    // SAFETY (the block): these are the POSIX declarations; every call below passes valid
    // NUL-terminated strings.
    unsafe extern "C" {
        fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        fn dlerror() -> *const c_char;
    }

    const RTLD_NOW: c_int = 0x2;
    // `RTLD_LOCAL` is 0 on Linux and FreeBSD and 0x4 on macOS; `RTLD_LOCAL` keeps the module's
    // symbols out of the core's lookup scope.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    const RTLD_LOCAL: c_int = 0x4;
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    const RTLD_LOCAL: c_int = 0;

    fn last_error() -> String {
        // SAFETY: `dlerror` returns null or a NUL-terminated string valid until the next loader
        // call on this thread; it is copied at once.
        let message = unsafe { dlerror() };
        if message.is_null() {
            return "the loader gave no reason".to_owned();
        }
        // SAFETY: non-null, NUL-terminated, as above.
        unsafe { CStr::from_ptr(message) }
            .to_string_lossy()
            .into_owned()
    }

    pub fn open(path: &Path) -> Result<RawLibrary, String> {
        let c_path = CString::new(path.as_os_str().as_bytes())
            .map_err(|_| "the path holds a NUL byte".to_owned())?;
        // SAFETY: `c_path` is NUL-terminated and outlives the call.
        let handle = unsafe { dlopen(c_path.as_ptr(), RTLD_NOW | RTLD_LOCAL) };
        if handle.is_null() {
            return Err(last_error());
        }
        Ok(RawLibrary(handle))
    }

    pub fn symbol(lib: &RawLibrary, name: &str) -> Result<*mut c_void, String> {
        let c_name =
            CString::new(name).map_err(|_| "the symbol name holds a NUL byte".to_owned())?;
        // Clears any stale error, so a null result can be told from an error.
        // SAFETY: `dlerror` takes no arguments.
        unsafe { dlerror() };
        // SAFETY: `lib.0` came from a successful `dlopen` and is never closed; `c_name` is
        // NUL-terminated.
        let address = unsafe { dlsym(lib.0, c_name.as_ptr()) };
        if address.is_null() {
            return Err(last_error());
        }
        Ok(address)
    }
}

#[cfg(windows)]
mod imp {
    use std::ffi::{CString, c_char, c_void};
    use std::os::windows::ffi::OsStrExt;
    use std::path::Path;

    use super::RawLibrary;

    #[link(name = "kernel32")]
    // SAFETY (the block): these are the Win32 declarations; every call below passes valid
    // NUL-terminated strings.
    unsafe extern "system" {
        fn LoadLibraryExW(name: *const u16, file: *mut c_void, flags: u32) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
        fn GetLastError() -> u32;
    }

    // Resolve the library's own dependencies from its directory and the default system
    // directories, never from the current directory or `PATH` (RFC-0052 §4.3).
    const LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR: u32 = 0x0000_0100;
    const LOAD_LIBRARY_SEARCH_DEFAULT_DIRS: u32 = 0x0000_1000;

    pub fn open(path: &Path) -> Result<RawLibrary, String> {
        let wide: Vec<u16> = path.as_os_str().encode_wide().chain([0]).collect();
        // SAFETY: `wide` is NUL-terminated and outlives the call; the file handle must be null.
        let handle = unsafe {
            LoadLibraryExW(
                wide.as_ptr(),
                std::ptr::null_mut(),
                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
            )
        };
        if handle.is_null() {
            // SAFETY: `GetLastError` takes no arguments.
            return Err(format!("Windows error {}", unsafe { GetLastError() }));
        }
        Ok(RawLibrary(handle))
    }

    pub fn symbol(lib: &RawLibrary, name: &str) -> Result<*mut c_void, String> {
        let c_name =
            CString::new(name).map_err(|_| "the symbol name holds a NUL byte".to_owned())?;
        // SAFETY: `lib.0` came from a successful `LoadLibraryExW` and is never freed; `c_name`
        // is NUL-terminated.
        let address = unsafe { GetProcAddress(lib.0, c_name.as_ptr()) };
        if address.is_null() {
            // SAFETY: `GetLastError` takes no arguments.
            return Err(format!("Windows error {}", unsafe { GetLastError() }));
        }
        Ok(address)
    }
}

impl RawLibrary {
    /// Opens the library at the absolute `path`. Its symbols stay out of the global lookup scope.
    ///
    /// The library is never unloaded: a provider library may hold thread-local destructors and
    /// device threads, which must not outlive their code.
    ///
    /// # Errors
    ///
    /// The loader's reason, as text.
    pub fn open(path: &Path) -> Result<Self, String> {
        if !path.is_absolute() {
            return Err("a provider library is opened by absolute path".to_owned());
        }
        imp::open(path)
    }

    /// The address of the symbol `name`.
    ///
    /// # Errors
    ///
    /// The loader's reason, as text, when the library does not export `name`.
    pub fn symbol(&self, name: &str) -> Result<*mut c_void, String> {
        imp::symbol(self, name)
    }
}
