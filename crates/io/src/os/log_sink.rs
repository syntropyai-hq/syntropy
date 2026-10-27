// SPDX-License-Identifier: MIT OR Apache-2.0

//! The OS log facilities the daemon's log sinks write to (RFC-0053 §4.2; PROD-E8-T04), behind
//! safe functions, because each takes `unsafe` FFI (RFC-0001 §4.2 rule 8; ADR-0022):
//!
//! - **macOS: unified logging.** `OsLog` is one `os_log_t`, made by `os_log_create` for a
//!   subsystem (the app bundle identifier) and a category (the crate). `OsLog::write` logs one
//!   preformatted line as `%{public}s`, through `_os_log_impl`, the function the `os_log` macros
//!   expand to, with the argument buffer those macros build for one public C string. Both live in
//!   `libsystem_trace`, part of `libSystem`, which every macOS binary loads (rule 7).
//! - **Windows: the event log.** `EventLog` is one event source, registered with
//!   `RegisterEventSourceW` (`Syntropy` in the Application log), and `EventLog::report` writes
//!   one line with `ReportEventW`. Both are in `advapi32.dll`, which rule 7 already allows.
//!
//! journald needs no FFI: the daemon writes its native protocol to a Unix datagram socket with
//! the standard library.
//!
//! Every `unsafe` block carries a `// SAFETY:` comment with its invariant (ADR-0001 item 4).

/// How severe a line is, as the OS facilities take it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Severity {
    /// An error.
    Error,
    /// A warning.
    Warning,
    /// Everything else.
    Info,
    /// Detail for debugging.
    Debug,
}

#[cfg(target_os = "macos")]
pub use macos::{OsLog, OsLogError};

#[cfg(windows)]
pub use windows::{EventLog, EventLogError};

#[cfg(target_os = "macos")]
mod macos {
    use std::ffi::{CString, c_char, c_void};
    use std::fmt;

    use super::Severity;

    /// `os_log_t`: an opaque, reference-counted log object.
    type OsLogT = *mut c_void;

    // `<os/log.h>`'s types (`os_log_type_t`).
    const OS_LOG_TYPE_DEFAULT: u8 = 0x00;
    const OS_LOG_TYPE_INFO: u8 = 0x01;
    const OS_LOG_TYPE_DEBUG: u8 = 0x02;
    const OS_LOG_TYPE_ERROR: u8 = 0x10;

    /// The format every line is logged with: one public C string. It sits in this image's
    /// `__TEXT,__oslogstring` section, where the `os_log` macros put their formats, because the
    /// logging system reads a message's format back from that section of the image `dso` names
    /// when it decodes the message.
    #[unsafe(link_section = "__TEXT,__oslogstring,cstring_literals")]
    static FORMAT: [u8; 11] = *b"%{public}s\0";

    unsafe extern "C" {
        fn os_log_create(subsystem: *const c_char, category: *const c_char) -> OsLogT;
        fn os_release(object: *mut c_void);
        fn _os_log_impl(
            dso: *mut c_void,
            log: OsLogT,
            kind: u8,
            format: *const c_char,
            buf: *mut u8,
            size: u32,
        );
    }

    /// Why a log object could not be made.
    #[derive(Debug)]
    pub enum OsLogError {
        /// The subsystem or category holds a NUL byte.
        Nul,
        /// `os_log_create` returned nothing, or this image's base could not be found.
        Unavailable,
    }

    impl fmt::Display for OsLogError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                OsLogError::Nul => f.write_str("a subsystem or category holds a NUL byte"),
                OsLogError::Unavailable => f.write_str("unified logging is unavailable"),
            }
        }
    }

    impl std::error::Error for OsLogError {}

    /// One unified-logging log object: a subsystem and a category.
    #[derive(Debug)]
    pub struct OsLog {
        log: OsLogT,
        /// The base of the image `FORMAT` is in: what the macros pass as `__dso_handle`.
        dso: *mut c_void,
    }

    // SAFETY: an `os_log_t` may be used from any thread (`<os/log.h>`: log objects are
    // thread-safe), and `dso` is the address of this image's header, which never changes.
    unsafe impl Send for OsLog {}
    // SAFETY: as for `Send`; `write` takes `&self` and the OS serialises its own state.
    unsafe impl Sync for OsLog {}

    impl OsLog {
        /// A log object for `subsystem` (the app bundle identifier) and `category` (the crate).
        ///
        /// # Errors
        ///
        /// [`OsLogError::Nul`] for a name holding a NUL byte, and [`OsLogError::Unavailable`]
        /// when the OS makes no object.
        pub fn new(subsystem: &str, category: &str) -> Result<OsLog, OsLogError> {
            let subsystem = CString::new(subsystem).map_err(|_| OsLogError::Nul)?;
            let category = CString::new(category).map_err(|_| OsLogError::Nul)?;
            // SAFETY: both arguments are NUL-terminated strings that live through the call; the
            // function copies what it keeps.
            let log = unsafe { os_log_create(subsystem.as_ptr(), category.as_ptr()) };
            if log.is_null() {
                return Err(OsLogError::Unavailable);
            }
            let mut info = std::mem::MaybeUninit::<libc::Dl_info>::zeroed();
            // SAFETY: `FORMAT` is a static of this image and `info` is writable for one
            // `Dl_info`; `dladdr` fills it and returns nonzero when it found the image.
            let found = unsafe { libc::dladdr(FORMAT.as_ptr().cast(), info.as_mut_ptr()) };
            // SAFETY: `dladdr` zeroed or filled `info`, which started zeroed, so it is valid.
            let base = unsafe { info.assume_init() }.dli_fbase;
            if found == 0 || base.is_null() {
                // SAFETY: `log` came from `os_log_create` and is released once, here.
                unsafe { os_release(log) };
                return Err(OsLogError::Unavailable);
            }
            Ok(OsLog { log, dso: base })
        }

        /// Logs `line` at `severity`. A line holding a NUL byte is cut at it.
        pub fn write(&self, severity: Severity, line: &str) {
            let text = match CString::new(line) {
                Ok(text) => text,
                Err(error) => {
                    let cut = error.nul_position();
                    CString::new(&line.as_bytes()[..cut]).unwrap_or_default()
                }
            };
            let kind = match severity {
                Severity::Error => OS_LOG_TYPE_ERROR,
                Severity::Warning => OS_LOG_TYPE_DEFAULT,
                Severity::Info => OS_LOG_TYPE_INFO,
                Severity::Debug => OS_LOG_TYPE_DEBUG,
            };
            // The argument buffer the `os_log` macros build for `"%{public}s"`: the summary
            // (0x02, the arguments hold a non-scalar), one argument, its descriptor (0x22: a
            // string, public), its size (8) and the pointer.
            let mut buf = [0u8; 12];
            buf[0] = 0x02;
            buf[1] = 0x01;
            buf[2] = 0x22;
            buf[3] = 0x08;
            buf[4..12].copy_from_slice(&(text.as_ptr() as usize as u64).to_ne_bytes());
            // SAFETY: `self.log` is a live log object; `dso` is the base of the image that holds
            // `FORMAT`, a NUL-terminated format with one `%s`; `buf` describes exactly one
            // argument, a pointer to `text`, which lives through the call; `size` is the
            // buffer's length.
            unsafe {
                _os_log_impl(
                    self.dso,
                    self.log,
                    kind,
                    FORMAT.as_ptr().cast(),
                    buf.as_mut_ptr(),
                    buf.len() as u32,
                );
            }
        }
    }

    impl Drop for OsLog {
        fn drop(&mut self) {
            // SAFETY: `self.log` came from `os_log_create` and is released once, here.
            unsafe { os_release(self.log) };
        }
    }
}

#[cfg(windows)]
mod windows {
    use std::fmt;

    use windows_sys::Win32::Foundation::HANDLE;
    use windows_sys::Win32::System::EventLog::{
        DeregisterEventSource, EVENTLOG_ERROR_TYPE, EVENTLOG_INFORMATION_TYPE,
        EVENTLOG_WARNING_TYPE, RegisterEventSourceW, ReportEventW,
    };

    use super::Severity;

    /// Why the event log could not be written.
    #[derive(Debug)]
    pub enum EventLogError {
        /// The OS refused the call.
        Os(std::io::Error),
    }

    impl fmt::Display for EventLogError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                EventLogError::Os(error) => write!(f, "the event log: {error}"),
            }
        }
    }

    impl std::error::Error for EventLogError {}

    /// One registered event source in the Application log.
    #[derive(Debug)]
    pub struct EventLog {
        handle: HANDLE,
    }

    // SAFETY: an event-source handle may be used from any thread; `ReportEventW` is thread-safe.
    unsafe impl Send for EventLog {}
    // SAFETY: as for `Send`.
    unsafe impl Sync for EventLog {}

    fn wide(text: &str) -> Vec<u16> {
        text.encode_utf16()
            .filter(|&unit| unit != 0)
            .chain([0])
            .collect()
    }

    impl EventLog {
        /// Registers `source` on this machine.
        ///
        /// # Errors
        ///
        /// [`EventLogError::Os`] when the OS refuses.
        pub fn register(source: &str) -> Result<EventLog, EventLogError> {
            let name = wide(source);
            // SAFETY: a null server name is the local machine; `name` is a NUL-terminated wide
            // string that lives through the call.
            let handle = unsafe { RegisterEventSourceW(std::ptr::null(), name.as_ptr()) };
            if handle.is_null() {
                return Err(EventLogError::Os(std::io::Error::last_os_error()));
            }
            Ok(EventLog { handle })
        }

        /// Writes `line` at `severity`.
        ///
        /// # Errors
        ///
        /// [`EventLogError::Os`] when the OS refuses.
        pub fn report(&self, severity: Severity, line: &str) -> Result<(), EventLogError> {
            let kind = match severity {
                Severity::Error => EVENTLOG_ERROR_TYPE,
                Severity::Warning => EVENTLOG_WARNING_TYPE,
                Severity::Info | Severity::Debug => EVENTLOG_INFORMATION_TYPE,
            };
            let text = wide(line);
            let strings = [text.as_ptr()];
            // SAFETY: `self.handle` is a registered source; `strings` holds one NUL-terminated
            // wide string that lives through the call; no user SID and no raw data.
            let ok = unsafe {
                ReportEventW(
                    self.handle,
                    kind,
                    0,
                    0,
                    std::ptr::null_mut(),
                    1,
                    0,
                    strings.as_ptr(),
                    std::ptr::null(),
                )
            };
            if ok == 0 {
                return Err(EventLogError::Os(std::io::Error::last_os_error()));
            }
            Ok(())
        }
    }

    impl Drop for EventLog {
        fn drop(&mut self) {
            // SAFETY: `self.handle` came from `RegisterEventSourceW` and is closed once, here.
            unsafe { DeregisterEventSource(self.handle) };
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(any(target_os = "macos", windows))]
    use super::*;

    /// On macOS a log object is made and a line of every severity is written without fault. What
    /// reaches the unified log is read back only by the ignored test below, which needs `log`.
    #[cfg(target_os = "macos")]
    #[test]
    fn unified_logging_takes_lines() {
        let log = OsLog::new("org.syntropy.test", "syntropy_io").unwrap();
        for severity in [
            Severity::Error,
            Severity::Warning,
            Severity::Info,
            Severity::Debug,
        ] {
            log.write(severity, "syntropy io log-sink test line");
        }
        log.write(Severity::Info, "cut\0here");
        assert!(matches!(OsLog::new("a\0b", "c"), Err(OsLogError::Nul)));
    }

    /// Writes a unique line and reads it back with `log show`.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "reads the unified log with `log show`, which a runner may not allow"]
    fn unified_logging_is_readable() {
        let marker = format!("syntropy-oslog-{}", std::process::id());
        OsLog::new("org.syntropy.test", "syntropy_io")
            .unwrap()
            .write(Severity::Warning, &marker);
        let out = std::process::Command::new("log")
            .args(["show", "--last", "1m", "--style", "compact", "--predicate"])
            .arg("subsystem == \"org.syntropy.test\"")
            .output()
            .unwrap();
        assert!(String::from_utf8_lossy(&out.stdout).contains(&marker));
    }

    /// On Windows the source registers and a warning and an error are reported.
    #[cfg(windows)]
    #[test]
    fn the_event_log_takes_lines() {
        let log = EventLog::register("Syntropy").unwrap();
        log.report(Severity::Warning, "syntropy io log-sink test warning")
            .unwrap();
        log.report(Severity::Error, "syntropy io log-sink test error")
            .unwrap();
    }

    /// Reports a unique line and reads it back with `wevtutil`.
    #[cfg(windows)]
    #[test]
    #[ignore = "reads the Application log with `wevtutil`"]
    fn the_event_log_is_readable() {
        let marker = format!("syntropy-eventlog-{}", std::process::id());
        EventLog::register("Syntropy")
            .unwrap()
            .report(Severity::Warning, &marker)
            .unwrap();
        let out = std::process::Command::new("wevtutil")
            .args(["qe", "Application", "/c:20", "/rd:true", "/f:text"])
            .arg("/q:*[System[Provider[@Name='Syntropy']]]")
            .output()
            .unwrap();
        assert!(String::from_utf8_lossy(&out.stdout).contains(&marker));
    }
}
