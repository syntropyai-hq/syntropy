// SPDX-License-Identifier: MIT OR Apache-2.0

//! The shared-memory segment each OS provides (ADR-0015 item 4): created owner-only, sized once,
//! mapped read-write and shared.

#[cfg(any(target_vendor = "apple", windows))]
use syntropy_core::Entropy;

/// How many names to try when one is already taken by another segment.
#[cfg(any(target_vendor = "apple", windows))]
const NAME_ATTEMPTS: u32 = 64;

/// A fresh segment name: `prefix`, then 128 random bits from `entropy` as 26 lowercase base32
/// characters, so another process can neither guess the name nor take it first. With Apple's
/// prefix, `/syn`, the name is 30 characters, inside the 31-character limit on POSIX
/// shared-memory names.
#[cfg(any(target_vendor = "apple", windows))]
fn random_name(prefix: &str, entropy: &dyn Entropy) -> String {
    const DIGITS: &[u8; 32] = b"abcdefghijklmnopqrstuvwxyz234567";
    let mut bits = [0u8; 16];
    entropy.fill(&mut bits);
    let value = u128::from_le_bytes(bits);
    let mut name = String::with_capacity(prefix.len() + 26);
    name.push_str(prefix);
    for digit in 0..26 {
        name.push(char::from(DIGITS[(value >> (digit * 5)) as usize & 31]));
    }
    name
}

pub(super) use imp::Segment;

/// The mapping both Unix builds share: `mmap` of a descriptor, and the owner-only check.
#[cfg(any(target_os = "linux", target_vendor = "apple"))]
mod unix {
    use std::io;
    use std::os::fd::{AsRawFd, OwnedFd};
    use std::ptr::NonNull;

    /// A descriptor mapped read-write and shared.
    pub(super) struct Mapped {
        pub(super) fd: OwnedFd,
        pub(super) base: NonNull<u8>,
        pub(super) len: usize,
    }

    // SAFETY: the mapping is plain shared memory, valid until `drop` unmaps it; nothing in it is
    // tied to the thread that made it.
    unsafe impl Send for Mapped {}
    // SAFETY: shared access goes through the region's atomics and rings, which synchronise it.
    unsafe impl Sync for Mapped {}

    /// Checks a libc return value of -1.
    pub(super) fn check(result: libc::c_int) -> io::Result<libc::c_int> {
        if result == -1 {
            Err(io::Error::last_os_error())
        } else {
            Ok(result)
        }
    }

    /// The size of the object behind `fd`.
    pub(super) fn size(fd: &OwnedFd) -> io::Result<u64> {
        Ok(stat(fd)?.st_size as u64)
    }

    fn stat(fd: &OwnedFd) -> io::Result<libc::stat> {
        let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
        // SAFETY: `fstat` writes one `stat` to the live pointer, and the descriptor is open.
        check(unsafe { libc::fstat(fd.as_raw_fd(), stat.as_mut_ptr()) })?;
        // SAFETY: `fstat` succeeded, so it filled the whole struct.
        Ok(unsafe { stat.assume_init() })
    }

    /// The mode and owner, for diagnostics.
    pub(super) fn permissions(fd: &OwnedFd) -> io::Result<String> {
        let stat = stat(fd)?;
        // SAFETY: `geteuid` takes no arguments and cannot fail.
        let me = unsafe { libc::geteuid() };
        Ok(format!(
            "mode {:o}, owner uid {}, this process's euid {me}",
            stat.st_mode & 0o7777,
            stat.st_uid
        ))
    }

    /// Mode 0600 and owned by this process's effective user.
    pub(super) fn is_owner_only(fd: &OwnedFd) -> io::Result<bool> {
        let stat = stat(fd)?;
        // SAFETY: `geteuid` takes no arguments and cannot fail.
        let me = unsafe { libc::geteuid() };
        Ok(stat.st_mode & 0o777 == 0o600 && stat.st_uid == me)
    }

    impl Mapped {
        /// Maps the first `len` bytes of `fd`, which is at least that long.
        pub(super) fn new(fd: OwnedFd, len: usize) -> io::Result<Mapped> {
            // SAFETY: a null hint lets the kernel choose the address; `len` bytes of an open
            // descriptor that is at least `len` long are mapped shared and read-write, at offset 0.
            let base = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    fd.as_raw_fd(),
                    0,
                )
            };
            if base == libc::MAP_FAILED {
                return Err(io::Error::last_os_error());
            }
            let base = NonNull::new(base.cast::<u8>())
                .ok_or_else(|| io::Error::other("mmap returned a null mapping"))?;
            Ok(Mapped { fd, base, len })
        }
    }

    impl Drop for Mapped {
        fn drop(&mut self) {
            // SAFETY: `base` and `len` are this mapping's, mapped by `new` and not unmapped
            // since. Every reference into it borrows the region, which is being dropped.
            unsafe { libc::munmap(self.base.as_ptr().cast(), self.len) };
        }
    }
}

/// Linux: a sealed memfd, which has no name and lives while a descriptor or mapping of it does.
#[cfg(target_os = "linux")]
mod imp {
    use std::io;
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
    use std::ptr::NonNull;

    use syntropy_core::Entropy;

    use super::super::RegionHandle;
    use super::unix::{Mapped, check, is_owner_only, permissions, size};

    /// The seals a region's memfd carries: its size can never change, and no seal can be removed.
    const SEALS: libc::c_int = libc::F_SEAL_SHRINK | libc::F_SEAL_GROW | libc::F_SEAL_SEAL;

    pub(in super::super) struct Segment {
        map: Mapped,
    }

    /// A new descriptor for `fd`, with or without close-on-exec.
    fn duplicate(fd: libc::c_int, close_on_exec: bool) -> io::Result<OwnedFd> {
        let command = if close_on_exec {
            libc::F_DUPFD_CLOEXEC
        } else {
            libc::F_DUPFD
        };
        // SAFETY: `fcntl` with `F_DUPFD` or `F_DUPFD_CLOEXEC` takes a descriptor and a lowest
        // number, and returns a new descriptor or -1.
        let copy = check(unsafe { libc::fcntl(fd, command, 0) })?;
        // SAFETY: `copy` is the descriptor `fcntl` just made, and nothing else owns it.
        Ok(unsafe { OwnedFd::from_raw_fd(copy) })
    }

    impl Segment {
        /// A memfd has no name, so `entropy` is not needed.
        pub(in super::super) fn create(len: usize, _entropy: &dyn Entropy) -> io::Result<Segment> {
            // SAFETY: a NUL-terminated name and flags; `memfd_create` returns a new descriptor or
            // -1.
            let fd = check(unsafe {
                libc::memfd_create(
                    c"syntropy-runner".as_ptr(),
                    libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
                )
            })?;
            // SAFETY: `fd` is the descriptor `memfd_create` just made, and nothing else owns it.
            let fd = unsafe { OwnedFd::from_raw_fd(fd) };
            // SAFETY: the next three calls take the open descriptor and integers, no pointers.
            unsafe {
                check(libc::fchmod(fd.as_raw_fd(), 0o600))?;
                check(libc::ftruncate(fd.as_raw_fd(), len as libc::off_t))?;
                check(libc::fcntl(fd.as_raw_fd(), libc::F_ADD_SEALS, SEALS))?;
            }
            Ok(Segment {
                map: Mapped::new(fd, len)?,
            })
        }

        pub(in super::super) fn open(handle: &RegionHandle, len: usize) -> io::Result<Segment> {
            let RegionHandle::Fd(inherited) = handle else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "a Linux region is opened by its descriptor",
                ));
            };
            let fd = duplicate(*inherited, true)?;
            // The inherited number stays open in this process; with close-on-exec set, it goes no
            // further, to this process's own children.
            // SAFETY: `fcntl` with `F_SETFD` takes a descriptor and an integer flag, no pointers.
            check(unsafe { libc::fcntl(*inherited, libc::F_SETFD, libc::FD_CLOEXEC) })?;
            if !is_owner_only(&fd)? {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "the region is not owner-only",
                ));
            }
            // A segment whose size could change could shrink under the mapping, so only a sealed
            // memfd is mapped.
            // SAFETY: `fcntl` with `F_GET_SEALS` takes the open descriptor and returns the seals or
            // -1.
            let seals = check(unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_GET_SEALS) })?;
            if seals & SEALS != SEALS {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "the descriptor is not a sealed runner region",
                ));
            }
            if size(&fd)? < len as u64 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "the region is smaller than its layout",
                ));
            }
            Ok(Segment {
                map: Mapped::new(fd, len)?,
            })
        }

        pub(in super::super) fn base(&self) -> NonNull<u8> {
            self.map.base
        }

        pub(in super::super) fn handle(&self) -> RegionHandle {
            RegionHandle::Fd(self.map.fd.as_raw_fd())
        }

        /// The descriptor keeps close-on-exec in this process, so no other child inherits it.
        /// A hook clears the flag in `command`'s child alone, after the fork and before the exec,
        /// and the child finds it at the same number.
        pub(in super::super) fn configure_spawn(
            &self,
            command: &mut std::process::Command,
        ) -> io::Result<RegionHandle> {
            use std::os::unix::process::CommandExt;
            let fd = self.map.fd.as_raw_fd();
            // SAFETY: the hook runs in the child between `fork` and `exec`, where only
            // async-signal-safe calls are allowed. It makes one `fcntl` call, which is one,
            // builds its error without allocating, and touches nothing else.
            unsafe {
                command.pre_exec(move || {
                    if libc::fcntl(fd, libc::F_SETFD, 0) == -1 {
                        Err(io::Error::last_os_error())
                    } else {
                        Ok(())
                    }
                });
            }
            Ok(RegionHandle::Fd(fd))
        }

        pub(in super::super) fn unlink(&self) -> io::Result<()> {
            Ok(())
        }

        pub(in super::super) fn unlink_after_open(&self) -> io::Result<()> {
            Ok(())
        }

        pub(in super::super) fn is_owner_only(&self) -> io::Result<bool> {
            is_owner_only(&self.map.fd)
        }

        pub(in super::super) fn permissions(&self) -> io::Result<String> {
            permissions(&self.map.fd)
        }
    }
}

/// Apple platforms: a POSIX shared-memory object, named until `shm_unlink`.
#[cfg(target_vendor = "apple")]
mod imp {
    use std::ffi::CString;
    use std::io;
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
    use std::ptr::NonNull;

    use syntropy_core::Entropy;

    use super::super::RegionHandle;
    use super::unix::{Mapped, check, is_owner_only, permissions, size};
    use super::{NAME_ATTEMPTS, random_name};

    pub(in super::super) struct Segment {
        map: Mapped,
        name: CString,
    }

    fn shm_open(name: &CString, flags: libc::c_int) -> io::Result<OwnedFd> {
        // SAFETY: a NUL-terminated name, flags and a mode (passed as `c_uint`, the type the
        // variadic argument is read as); `shm_open` returns a new descriptor or -1.
        let fd = check(unsafe { libc::shm_open(name.as_ptr(), flags, 0o600 as libc::c_uint) })?;
        // SAFETY: `fd` is the descriptor `shm_open` just made, and nothing else owns it.
        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }

    impl Segment {
        pub(in super::super) fn create(len: usize, entropy: &dyn Entropy) -> io::Result<Segment> {
            for _ in 0..NAME_ATTEMPTS {
                let name = CString::new(random_name("/syn", entropy)).map_err(io::Error::other)?;
                let fd = match shm_open(&name, libc::O_RDWR | libc::O_CREAT | libc::O_EXCL) {
                    Ok(fd) => fd,
                    Err(error) if error.raw_os_error() == Some(libc::EEXIST) => continue,
                    Err(error) => return Err(error),
                };
                // SAFETY: `ftruncate` takes the open descriptor and a length, no pointers.
                let sized = unsafe { libc::ftruncate(fd.as_raw_fd(), len as libc::off_t) };
                if let Err(error) = check(sized) {
                    // SAFETY: `name` is NUL-terminated; the object was created just above.
                    unsafe { libc::shm_unlink(name.as_ptr()) };
                    return Err(error);
                }
                let map = Mapped::new(fd, len)?;
                return Ok(Segment { map, name });
            }
            Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "every segment name tried was taken",
            ))
        }

        pub(in super::super) fn open(handle: &RegionHandle, len: usize) -> io::Result<Segment> {
            let RegionHandle::Name(name) = handle else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "an Apple region is opened by its name",
                ));
            };
            let name = CString::new(name.as_str()).map_err(io::Error::other)?;
            let fd = shm_open(&name, libc::O_RDWR)?;
            if !is_owner_only(&fd)? {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "the region is not owner-only",
                ));
            }
            if size(&fd)? < len as u64 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "the region is smaller than its layout",
                ));
            }
            Ok(Segment {
                map: Mapped::new(fd, len)?,
                name,
            })
        }

        pub(in super::super) fn base(&self) -> NonNull<u8> {
            self.map.base
        }

        pub(in super::super) fn handle(&self) -> RegionHandle {
            RegionHandle::Name(self.name.to_string_lossy().into_owned())
        }

        /// A runner opens the segment by name, so `command` needs nothing.
        pub(in super::super) fn configure_spawn(
            &self,
            _command: &mut std::process::Command,
        ) -> io::Result<RegionHandle> {
            Ok(self.handle())
        }

        pub(in super::super) fn unlink(&self) -> io::Result<()> {
            // SAFETY: `name` is NUL-terminated.
            match check(unsafe { libc::shm_unlink(self.name.as_ptr()) }) {
                Ok(_) => Ok(()),
                Err(error) if error.raw_os_error() == Some(libc::ENOENT) => Ok(()),
                Err(error) => Err(error),
            }
        }

        /// The runner has mapped the segment, so its name is no longer needed: removing it now
        /// keeps the time another process could open it to the spawn.
        pub(in super::super) fn unlink_after_open(&self) -> io::Result<()> {
            self.unlink()
        }

        pub(in super::super) fn is_owner_only(&self) -> io::Result<bool> {
            is_owner_only(&self.map.fd)
        }

        pub(in super::super) fn permissions(&self) -> io::Result<String> {
            permissions(&self.map.fd)
        }
    }
}

/// Windows: a named file mapping backed by the paging file, in the session's `Local\` namespace,
/// with a DACL that allows only the user who created it.
#[cfg(windows)]
mod imp {
    use std::io;
    use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
    use std::ptr::NonNull;

    use windows_sys::Win32::Foundation::{
        ERROR_ALREADY_EXISTS, GetLastError, HANDLE, HLOCAL, INVALID_HANDLE_VALUE, LocalFree,
    };
    use windows_sys::Win32::Security::Authorization::{
        ConvertSecurityDescriptorToStringSecurityDescriptorW, ConvertSidToStringSidW,
        ConvertStringSecurityDescriptorToSecurityDescriptorW, GetSecurityInfo, SDDL_REVISION_1,
        SE_KERNEL_OBJECT,
    };
    use windows_sys::Win32::Security::{
        ACCESS_ALLOWED_ACE, ACL, ACL_SIZE_INFORMATION, AclSizeInformation,
        DACL_SECURITY_INFORMATION, EqualSid, GetAce, GetAclInformation,
        GetSecurityDescriptorControl, GetTokenInformation, PSECURITY_DESCRIPTOR, PSID,
        SE_DACL_PROTECTED, SECURITY_ATTRIBUTES, TOKEN_QUERY, TOKEN_USER, TokenUser,
    };
    use windows_sys::Win32::System::Memory::{
        CreateFileMappingW, FILE_MAP_ALL_ACCESS, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
        OpenFileMappingW, PAGE_READWRITE, UnmapViewOfFile,
    };
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

    use syntropy_core::Entropy;

    use super::super::RegionHandle;
    use super::{NAME_ATTEMPTS, random_name};

    pub(in super::super) struct Segment {
        mapping: OwnedHandle,
        base: NonNull<u8>,
        name: String,
    }

    // SAFETY: the view is plain shared memory, valid until `drop` unmaps it, and the mapping
    // handle may be used from any thread.
    unsafe impl Send for Segment {}
    // SAFETY: shared access goes through the region's atomics and rings, which synchronise it.
    unsafe impl Sync for Segment {}

    fn wide(text: &str) -> Vec<u16> {
        text.encode_utf16().chain(std::iter::once(0)).collect()
    }

    /// A `LocalAlloc`ed string of UTF-16 code units, read up to its NUL and then freed.
    ///
    /// # Safety
    ///
    /// `text` points to a NUL-terminated UTF-16 string that `LocalAlloc` allocated, and nothing
    /// else frees it.
    unsafe fn take_local_string(text: *mut u16) -> String {
        let mut len = 0;
        // SAFETY: the string is NUL-terminated, so every unit up to the NUL is readable.
        while unsafe { *text.add(len) } != 0 {
            len += 1;
        }
        // SAFETY: the `len` units before the NUL are initialised and readable.
        let units = unsafe { std::slice::from_raw_parts(text, len) };
        let string = String::from_utf16_lossy(units);
        // SAFETY: the caller promises `text` came from `LocalAlloc` and is freed only here.
        unsafe { LocalFree(text as HLOCAL) };
        string
    }

    /// The current user's SID, from this process's token.
    struct UserSid {
        /// The `TOKEN_USER` and the SID it points to. `u64`s keep the `TOKEN_USER` aligned.
        buffer: Vec<u64>,
    }

    impl UserSid {
        fn current() -> io::Result<UserSid> {
            let mut token: HANDLE = std::ptr::null_mut();
            // SAFETY: the pseudo-handle of this process, a query right, and a live output pointer.
            if unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) } == 0 {
                return Err(io::Error::last_os_error());
            }
            // SAFETY: `token` is the handle `OpenProcessToken` just returned, owned only here.
            let token = unsafe { OwnedHandle::from_raw_handle(token.cast()) };
            let mut needed = 0u32;
            // SAFETY: a null buffer of length 0 asks only for the size the token user takes.
            unsafe {
                GetTokenInformation(
                    token.as_raw_handle().cast(),
                    TokenUser,
                    std::ptr::null_mut(),
                    0,
                    &mut needed,
                )
            };
            let mut buffer = vec![0u64; (needed as usize).div_ceil(8).max(1)];
            // SAFETY: the buffer holds at least `needed` bytes, and the call writes at most that.
            if unsafe {
                GetTokenInformation(
                    token.as_raw_handle().cast(),
                    TokenUser,
                    buffer.as_mut_ptr().cast(),
                    (buffer.len() * 8) as u32,
                    &mut needed,
                )
            } == 0
            {
                return Err(io::Error::last_os_error());
            }
            Ok(UserSid { buffer })
        }

        /// The SID, which points into `buffer`, so it lives as long as `self`.
        fn sid(&self) -> PSID {
            // SAFETY: `current` filled the buffer with a `TOKEN_USER`, aligned by the `u64`
            // buffer, whose SID pointer points into the same buffer.
            unsafe { (*self.buffer.as_ptr().cast::<TOKEN_USER>()).User.Sid }
        }

        /// The SID in its string form, such as `S-1-5-21-…`.
        fn text(&self) -> io::Result<String> {
            let mut text: *mut u16 = std::ptr::null_mut();
            // SAFETY: a valid SID and a live output pointer; on success the string is
            // `LocalAlloc`ed.
            if unsafe { ConvertSidToStringSidW(self.sid(), &mut text) } == 0 {
                return Err(io::Error::last_os_error());
            }
            // SAFETY: `ConvertSidToStringSidW` returned a `LocalAlloc`ed, NUL-terminated string.
            Ok(unsafe { take_local_string(text) })
        }
    }

    /// A security descriptor whose protected DACL has one entry: all access for the current user.
    struct OwnerOnly(PSECURITY_DESCRIPTOR);

    impl OwnerOnly {
        fn new() -> io::Result<OwnerOnly> {
            let sddl = wide(&format!("D:P(A;;GA;;;{})", UserSid::current()?.text()?));
            let mut descriptor: PSECURITY_DESCRIPTOR = std::ptr::null_mut();
            // SAFETY: a NUL-terminated SDDL string and a live output pointer; on success the
            // descriptor is `LocalAlloc`ed, and `drop` frees it.
            if unsafe {
                ConvertStringSecurityDescriptorToSecurityDescriptorW(
                    sddl.as_ptr(),
                    SDDL_REVISION_1,
                    &mut descriptor,
                    std::ptr::null_mut(),
                )
            } == 0
            {
                return Err(io::Error::last_os_error());
            }
            Ok(OwnerOnly(descriptor))
        }
    }

    impl Drop for OwnerOnly {
        fn drop(&mut self) {
            // SAFETY: the descriptor came from `LocalAlloc`, and only this frees it.
            unsafe { LocalFree(self.0 as HLOCAL) };
        }
    }

    fn map(mapping: OwnedHandle, len: usize, name: String) -> io::Result<Segment> {
        // SAFETY: a live mapping handle with all access; `len` bytes from offset 0, which the
        // call refuses if the section is smaller.
        let view: MEMORY_MAPPED_VIEW_ADDRESS = unsafe {
            MapViewOfFile(
                mapping.as_raw_handle().cast(),
                FILE_MAP_ALL_ACCESS,
                0,
                0,
                len,
            )
        };
        let base = NonNull::new(view.Value.cast::<u8>()).ok_or_else(io::Error::last_os_error)?;
        Ok(Segment {
            mapping,
            base,
            name,
        })
    }

    impl Segment {
        pub(in super::super) fn create(len: usize, entropy: &dyn Entropy) -> io::Result<Segment> {
            let descriptor = OwnerOnly::new()?;
            let attributes = SECURITY_ATTRIBUTES {
                nLength: std::mem::size_of::<SECURITY_ATTRIBUTES>() as u32,
                lpSecurityDescriptor: descriptor.0,
                bInheritHandle: 0,
            };
            let size = len as u64;
            for _ in 0..NAME_ATTEMPTS {
                let name = random_name("Local\\", entropy);
                let wide_name = wide(&name);
                // SAFETY: the paging file as backing, live security attributes, a read-write
                // section of `size` bytes and a NUL-terminated name. It returns a handle or null.
                let handle = unsafe {
                    CreateFileMappingW(
                        INVALID_HANDLE_VALUE,
                        &attributes,
                        PAGE_READWRITE,
                        (size >> 32) as u32,
                        size as u32,
                        wide_name.as_ptr(),
                    )
                };
                if handle.is_null() {
                    return Err(io::Error::last_os_error());
                }
                // SAFETY: `GetLastError` reads this thread's last error, which the call just set.
                let existed = unsafe { GetLastError() } == ERROR_ALREADY_EXISTS;
                // SAFETY: `handle` is the handle `CreateFileMappingW` just returned, owned only
                // here.
                let mapping = unsafe { OwnedHandle::from_raw_handle(handle.cast()) };
                if existed {
                    // Another mapping has this name: this handle is to it, so drop it and retry.
                    continue;
                }
                return map(mapping, len, name);
            }
            Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "every segment name tried was taken",
            ))
        }

        pub(in super::super) fn open(handle: &RegionHandle, len: usize) -> io::Result<Segment> {
            let RegionHandle::Name(name) = handle else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "a Windows region is opened by its name",
                ));
            };
            let wide_name = wide(name);
            // SAFETY: all access, not inheritable, and a NUL-terminated name. It returns a handle
            // or null.
            let opened = unsafe { OpenFileMappingW(FILE_MAP_ALL_ACCESS, 0, wide_name.as_ptr()) };
            if opened.is_null() {
                return Err(io::Error::last_os_error());
            }
            // SAFETY: `opened` is the handle `OpenFileMappingW` just returned, owned only here.
            let mapping = unsafe { OwnedHandle::from_raw_handle(opened.cast()) };
            if !owner_only(&mapping)? {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "the region is not owner-only",
                ));
            }
            map(mapping, len, name.clone())
        }

        pub(in super::super) fn base(&self) -> NonNull<u8> {
            self.base
        }

        pub(in super::super) fn handle(&self) -> RegionHandle {
            RegionHandle::Name(self.name.clone())
        }

        /// A runner opens the mapping by name, so `command` needs nothing.
        pub(in super::super) fn configure_spawn(
            &self,
            _command: &mut std::process::Command,
        ) -> io::Result<RegionHandle> {
            Ok(self.handle())
        }

        pub(in super::super) fn unlink(&self) -> io::Result<()> {
            Ok(())
        }

        /// A mapping's name lives as long as a handle to it does, so nothing is removed here.
        pub(in super::super) fn unlink_after_open(&self) -> io::Result<()> {
            Ok(())
        }

        pub(in super::super) fn is_owner_only(&self) -> io::Result<bool> {
            owner_only(&self.mapping)
        }

        pub(in super::super) fn permissions(&self) -> io::Result<String> {
            permissions_text(&self.mapping)
        }
    }

    /// The security descriptor of `mapping` with its DACL, `LocalAlloc`ed, and the DACL inside it.
    fn dacl(mapping: &OwnedHandle) -> io::Result<(OwnerOnly, *mut ACL)> {
        let mut descriptor: PSECURITY_DESCRIPTOR = std::ptr::null_mut();
        let mut dacl: *mut ACL = std::ptr::null_mut();
        // SAFETY: a live mapping handle opened with all access (which includes reading its
        // security); null owner, group and SACL outputs; live DACL and descriptor outputs. On
        // success the descriptor is `LocalAlloc`ed and the DACL points into it.
        let status = unsafe {
            GetSecurityInfo(
                mapping.as_raw_handle().cast(),
                SE_KERNEL_OBJECT,
                DACL_SECURITY_INFORMATION,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut dacl,
                std::ptr::null_mut(),
                &mut descriptor,
            )
        };
        if status != 0 {
            return Err(io::Error::from_raw_os_error(status as i32));
        }
        Ok((OwnerOnly(descriptor), dacl))
    }

    /// Whether `mapping`'s DACL is protected and has exactly one entry, which allows the current
    /// user. Read from the DACL itself, not its text form, which abbreviates some SIDs.
    fn owner_only(mapping: &OwnedHandle) -> io::Result<bool> {
        let (descriptor, dacl) = dacl(mapping)?;
        if dacl.is_null() {
            // A null DACL allows everyone.
            return Ok(false);
        }
        let mut control = 0u16;
        let mut revision = 0u32;
        // SAFETY: a live descriptor and two live outputs.
        if unsafe { GetSecurityDescriptorControl(descriptor.0, &mut control, &mut revision) } == 0 {
            return Err(io::Error::last_os_error());
        }
        if control & SE_DACL_PROTECTED == 0 {
            return Ok(false);
        }
        let mut size = ACL_SIZE_INFORMATION {
            AceCount: 0,
            AclBytesInUse: 0,
            AclBytesFree: 0,
        };
        // SAFETY: a live DACL and a live output of the size the class needs.
        if unsafe {
            GetAclInformation(
                dacl,
                (&raw mut size).cast(),
                std::mem::size_of::<ACL_SIZE_INFORMATION>() as u32,
                AclSizeInformation,
            )
        } == 0
        {
            return Err(io::Error::last_os_error());
        }
        if size.AceCount != 1 {
            return Ok(false);
        }
        let mut ace: *mut core::ffi::c_void = std::ptr::null_mut();
        // SAFETY: a live DACL with one entry, index 0, and a live output; on success `ace` points
        // into the DACL.
        if unsafe { GetAce(dacl, 0, &mut ace) } == 0 {
            return Err(io::Error::last_os_error());
        }
        let ace = ace.cast::<ACCESS_ALLOWED_ACE>();
        // SAFETY: `GetAce` returned a live entry, which starts with its header.
        let kind = unsafe { (*ace).Header.AceType };
        // `ACCESS_ALLOWED_ACE_TYPE` is 0 (it lives behind another windows-sys feature).
        if kind != 0 {
            return Ok(false);
        }
        let user = UserSid::current()?;
        // SAFETY: an access-allowed entry's SID starts at its `SidStart` field, inside the DACL,
        // which `descriptor` keeps alive; the user's SID lives in `user`.
        let same = unsafe { EqualSid((&raw const (*ace).SidStart).cast_mut().cast(), user.sid()) };
        Ok(same != 0)
    }

    /// The DACL in its text form, for diagnostics.
    pub(in super::super) fn permissions_text(mapping: &OwnedHandle) -> io::Result<String> {
        let (descriptor, _) = dacl(mapping)?;
        let mut text: *mut u16 = std::ptr::null_mut();
        // SAFETY: a live descriptor and a live output pointer; on success the string is
        // `LocalAlloc`ed.
        if unsafe {
            ConvertSecurityDescriptorToStringSecurityDescriptorW(
                descriptor.0,
                SDDL_REVISION_1,
                DACL_SECURITY_INFORMATION,
                &mut text,
                std::ptr::null_mut(),
            )
        } == 0
        {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: the call returned a `LocalAlloc`ed, NUL-terminated string.
        Ok(unsafe { take_local_string(text) })
    }

    impl Drop for Segment {
        fn drop(&mut self) {
            // SAFETY: `base` is the view `MapViewOfFile` returned, not unmapped since; every
            // reference into it borrows the region, which is being dropped.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: self.base.as_ptr().cast(),
                })
            };
        }
    }
}
