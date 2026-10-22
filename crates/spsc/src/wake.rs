// SPDX-License-Identifier: MIT OR Apache-2.0

//! The wake handle: one OS object per ring that an async runtime can wait on, so a ring into the
//! async domain wakes tokio without polling (ADR-0012 item 4). `net` or the runner registers it:
//! with tokio's `AsyncFd` on Unix, and as an event handle on Windows.
//!
//! | OS | Object | Signal | Reset |
//! |---|---|---|---|
//! | Linux, Android | an eventfd, non-blocking | write 1 | read the counter |
//! | Apple platforms | a kqueue with one `EVFILT_USER` event, `EV_CLEAR` | `NOTE_TRIGGER` | retrieve the event |
//! | Windows | a manual-reset event | `SetEvent` | `ResetEvent` |
//!
//! On each OS the object stays signalled until it is reset, so a signal that arrives before the
//! consumer waits is not lost.

use std::fmt;
use std::io;

use crate::ring::{Consumer, Producer, Waker, with_waker};

/// A ring of `capacity` slots whose consumer waits on a [`WakeHandle`], as a consumer on an async
/// runtime does. See [`Consumer::arm`] for the consumer's loop.
///
/// # Errors
///
/// [`io::ErrorKind::InvalidInput`] for a capacity [`ring`](crate::ring) refuses, and the OS error
/// if the wake handle cannot be made.
pub fn ring_with_wake<T>(capacity: usize) -> io::Result<(Producer<T>, Consumer<T>)> {
    let handle = WakeHandle::new()?;
    with_waker(capacity, Waker::with_handle(handle))
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))
}

impl<T> Consumer<T> {
    /// The handle to wait on, for a ring made with [`ring_with_wake`].
    #[must_use]
    pub fn wake_handle(&self) -> Option<&WakeHandle> {
        self.shared.waker.handle.as_ref()
    }
}

/// The OS object a ring's consumer waits on: signalled by the producer while the consumer is
/// armed, and reset by the consumer before it pops.
pub struct WakeHandle {
    os: os::Handle,
}

impl WakeHandle {
    fn new() -> io::Result<WakeHandle> {
        Ok(WakeHandle {
            os: os::Handle::new()?,
        })
    }

    /// Signals the handle. It stays signalled until [`WakeHandle::reset`].
    ///
    /// # Errors
    ///
    /// The OS error, if the OS refuses the signal.
    pub fn notify(&self) -> io::Result<()> {
        self.os.notify()
    }

    /// Clears the signal, and returns whether the handle was signalled.
    ///
    /// # Errors
    ///
    /// The OS error, if the OS refuses the read or the reset.
    pub fn reset(&self) -> io::Result<bool> {
        self.os.reset()
    }
}

impl fmt::Debug for WakeHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WakeHandle").finish_non_exhaustive()
    }
}

#[cfg(unix)]
impl std::os::fd::AsFd for WakeHandle {
    fn as_fd(&self) -> std::os::fd::BorrowedFd<'_> {
        self.os.as_fd()
    }
}

#[cfg(unix)]
impl std::os::fd::AsRawFd for WakeHandle {
    fn as_raw_fd(&self) -> std::os::fd::RawFd {
        std::os::fd::AsRawFd::as_raw_fd(&self.os.as_fd())
    }
}

#[cfg(windows)]
impl std::os::windows::io::AsHandle for WakeHandle {
    fn as_handle(&self) -> std::os::windows::io::BorrowedHandle<'_> {
        self.os.as_handle()
    }
}

#[cfg(windows)]
impl std::os::windows::io::AsRawHandle for WakeHandle {
    fn as_raw_handle(&self) -> std::os::windows::io::RawHandle {
        std::os::windows::io::AsRawHandle::as_raw_handle(&self.os.as_handle())
    }
}

/// An eventfd: a 64-bit counter the kernel keeps, readable while it is above zero.
#[cfg(any(target_os = "linux", target_os = "android"))]
mod os {
    use std::fs::File;
    use std::io::{self, Read, Write};
    use std::os::fd::{AsFd, BorrowedFd, FromRawFd, OwnedFd};

    pub(super) struct Handle {
        file: File,
    }

    impl Handle {
        pub(super) fn new() -> io::Result<Handle> {
            // SAFETY: `eventfd` takes two integers and no pointers, and returns a new descriptor
            // or -1.
            let fd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
            if fd < 0 {
                return Err(io::Error::last_os_error());
            }
            // SAFETY: `fd` is the descriptor `eventfd` just returned, and nothing else owns it.
            let fd = unsafe { OwnedFd::from_raw_fd(fd) };
            Ok(Handle {
                file: File::from(fd),
            })
        }

        pub(super) fn notify(&self) -> io::Result<()> {
            match (&self.file).write(&1u64.to_ne_bytes()) {
                // `WouldBlock` means the counter is at its maximum, so it is already signalled.
                Ok(_) => Ok(()),
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => Ok(()),
                Err(error) => Err(error),
            }
        }

        pub(super) fn reset(&self) -> io::Result<bool> {
            let mut counter = [0u8; 8];
            match (&self.file).read(&mut counter) {
                Ok(_) => Ok(true),
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => Ok(false),
                Err(error) => Err(error),
            }
        }

        pub(super) fn as_fd(&self) -> BorrowedFd<'_> {
            self.file.as_fd()
        }
    }
}

/// A kqueue holding one user event, which a kqueue reports as readable while it is triggered.
#[cfg(target_vendor = "apple")]
mod os {
    use std::io;
    use std::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, OwnedFd};

    pub(super) struct Handle {
        kqueue: OwnedFd,
    }

    /// The change record for the one user event, identifier 0.
    fn user_event(flags: u16, fflags: u32) -> libc::kevent {
        libc::kevent {
            ident: 0,
            filter: libc::EVFILT_USER,
            flags,
            fflags,
            data: 0,
            udata: std::ptr::null_mut(),
        }
    }

    impl Handle {
        pub(super) fn new() -> io::Result<Handle> {
            // SAFETY: `kqueue` takes no arguments, and returns a new descriptor or -1.
            let fd = unsafe { libc::kqueue() };
            if fd < 0 {
                return Err(io::Error::last_os_error());
            }
            // SAFETY: `fd` is the descriptor `kqueue` just returned, and nothing else owns it.
            let kqueue = unsafe { OwnedFd::from_raw_fd(fd) };
            // SAFETY: `fcntl` with `F_SETFD` takes the descriptor and an integer flag, no pointers.
            if unsafe { libc::fcntl(kqueue.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC) } < 0 {
                return Err(io::Error::last_os_error());
            }
            let handle = Handle { kqueue };
            handle.change(user_event(libc::EV_ADD | libc::EV_CLEAR, 0))?;
            Ok(handle)
        }

        /// Applies one change record, reading no events back.
        fn change(&self, change: libc::kevent) -> io::Result<()> {
            // SAFETY: `change` is one live record, the output list is empty (null with length 0),
            // and a null timeout is allowed when no events are read.
            let applied = unsafe {
                libc::kevent(
                    self.kqueue.as_raw_fd(),
                    &change,
                    1,
                    std::ptr::null_mut(),
                    0,
                    std::ptr::null(),
                )
            };
            if applied < 0 {
                Err(io::Error::last_os_error())
            } else {
                Ok(())
            }
        }

        pub(super) fn notify(&self) -> io::Result<()> {
            self.change(user_event(0, libc::NOTE_TRIGGER))
        }

        pub(super) fn reset(&self) -> io::Result<bool> {
            let mut event = user_event(0, 0);
            let now = libc::timespec {
                tv_sec: 0,
                tv_nsec: 0,
            };
            // SAFETY: no change records (null with length 0), one live output record, and a live
            // zero timeout, so the call only collects a pending event without blocking. With
            // `EV_CLEAR`, collecting the event clears it.
            let collected = unsafe {
                libc::kevent(
                    self.kqueue.as_raw_fd(),
                    std::ptr::null(),
                    0,
                    &mut event,
                    1,
                    &now,
                )
            };
            if collected < 0 {
                Err(io::Error::last_os_error())
            } else {
                Ok(collected > 0)
            }
        }

        pub(super) fn as_fd(&self) -> BorrowedFd<'_> {
            self.kqueue.as_fd()
        }
    }
}

/// A manual-reset event object.
#[cfg(windows)]
mod os {
    use std::io;
    use std::os::windows::io::{AsHandle, AsRawHandle, BorrowedHandle, FromRawHandle, OwnedHandle};

    use windows_sys::Win32::Foundation::{HANDLE, WAIT_OBJECT_0};
    use windows_sys::Win32::System::Threading::{
        CreateEventW, ResetEvent, SetEvent, WaitForSingleObject,
    };

    pub(super) struct Handle {
        event: OwnedHandle,
    }

    impl Handle {
        pub(super) fn new() -> io::Result<Handle> {
            // SAFETY: null security attributes and a null name are both allowed. The event is
            // manual-reset (1) and starts unsignalled (0). It returns a new handle or null.
            let event = unsafe { CreateEventW(std::ptr::null(), 1, 0, std::ptr::null()) };
            if event.is_null() {
                return Err(io::Error::last_os_error());
            }
            // SAFETY: `event` is the handle `CreateEventW` just returned, and nothing else owns
            // it.
            let event = unsafe { OwnedHandle::from_raw_handle(event.cast()) };
            Ok(Handle { event })
        }

        fn raw(&self) -> HANDLE {
            self.event.as_raw_handle().cast()
        }

        pub(super) fn notify(&self) -> io::Result<()> {
            // SAFETY: `raw` is this handle's live event object, which `SetEvent` accepts.
            if unsafe { SetEvent(self.raw()) } == 0 {
                Err(io::Error::last_os_error())
            } else {
                Ok(())
            }
        }

        pub(super) fn reset(&self) -> io::Result<bool> {
            // SAFETY: a zero timeout only reads the live event's state, without waiting.
            let signalled = unsafe { WaitForSingleObject(self.raw(), 0) } == WAIT_OBJECT_0;
            // SAFETY: `raw` is this handle's live event object, which `ResetEvent` accepts.
            if unsafe { ResetEvent(self.raw()) } == 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(signalled)
        }

        pub(super) fn as_handle(&self) -> BorrowedHandle<'_> {
            self.event.as_handle()
        }
    }
}
