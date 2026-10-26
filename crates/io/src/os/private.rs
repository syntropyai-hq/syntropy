// SPDX-License-Identifier: MIT OR Apache-2.0

//! Owner-only directories and files, for the daemon's state directory and the node key in it
//! (RFC-0030 §2.1–§2.2; DIST-E2-T01).
//!
//! RFC-0030 §2.1 creates the state directory mode 0700, or with an owner-only DACL on Windows.
//! §2.2 writes the node key's seed with `O_EXCL` and mode 0600, then fsyncs the file and the
//! directory. `cluster` and `daemon` may not use `unsafe` (RFC-0001 §4.2 rule 8), so the OS calls
//! live here behind safe functions (OQ-PRC-26):
//!
//! - [`private_dir`] creates a directory only the current user can use, or tightens an existing
//!   one this user owns.
//! - [`create_private_file`] creates a new owner-only file, writes it and fsyncs it.
//! - [`sync_dir`] fsyncs a directory, so a file just created in it survives a crash.
//! - [`is_owner_only`] says whether only the current user can access a file or directory.
//!
//! On Unix, "only the current user" means owned by the effective uid with no group or other mode
//! bits. A Linux POSIX ACL needs no separate check: with an extended ACL the group mode bits are
//! the ACL mask, and a zero mask makes every named entry ineffective. macOS extended ACLs do not
//! show in the mode, so [`is_owner_only`] also reads the file's extended ACL there and refuses
//! any entry that is not a deny entry.
//!
//! On Windows, the objects these functions create carry the current user's SID as owner and a
//! protected DACL with one entry: full control for that SID, inherited by objects and containers
//! in a directory. The owner is set explicitly because an elevated token's default owner is the
//! Administrators group, not the user. [`is_owner_only`] accepts an object only when its owner is
//! the current user and every entry in its DACL allows the current user. An entry for SYSTEM or
//! the Administrators group makes it not owner-only. Both can reach any file anyway, SYSTEM
//! through the backup privilege and administrators by taking ownership, but an entry for them is
//! still a grant this module did not make, and the check stays strict: it reports what the DACL
//! grants, not what a privileged account could take.
//!
//! Other targets return [`io::ErrorKind::Unsupported`] from every function.

use std::io;
use std::path::Path;

/// Creates the directory at `path`, so only the current user can use it.
///
/// Missing parents are created first with default permissions. On Unix the directory is created
/// mode 0700; if it already exists and this user owns it, its mode is set to exactly 0700, through
/// a descriptor opened on it, so a rename between the check and the change cannot redirect the
/// change. On Windows it is created with the owner-only descriptor the module documentation
/// describes; if it already exists and this user owns it, its DACL is replaced with the owner-only
/// one through `SetNamedSecurityInfoW`, unless it is already owner-only and protected. A symbolic
/// link at `path` is followed, and the checks apply to its target.
///
/// # Errors
///
/// [`io::ErrorKind::PermissionDenied`] when the directory exists and another user owns it;
/// [`io::ErrorKind::NotADirectory`] when `path` exists and is not a directory; otherwise the OS
/// error from creating, opening or changing it. On macOS an existing extended ACL is left in place,
/// and [`is_owner_only`] reports it.
pub fn private_dir(path: &Path) -> io::Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    imp::private_dir(path)
}

/// Creates the file at `path`, which must not exist, owner-only, writes all of `bytes` and
/// fsyncs it.
///
/// On Unix the file is opened with `O_CREAT | O_EXCL` and mode 0600, and the mode is then set to
/// exactly 0600 through the descriptor, since the umask can only have cleared bits. On Windows it
/// is created with `CREATE_NEW` and the owner-only descriptor set at creation. `O_EXCL` and
/// `CREATE_NEW` also refuse a symbolic link at `path`. If writing or syncing fails, the file is
/// removed, so a failed call never leaves a truncated file behind.
///
/// The file's directory entry is durable only once the directory is synced too: call [`sync_dir`]
/// on the parent afterwards (RFC-0030 §2.2).
///
/// # Errors
///
/// [`io::ErrorKind::AlreadyExists`] when `path` exists, and the OS error from creating, writing
/// or syncing it otherwise.
pub fn create_private_file(path: &Path, bytes: &[u8]) -> io::Result<()> {
    imp::create_private_file(path, bytes)
}

/// Flushes the directory at `path` to stable storage, so entries just created or removed in it
/// survive a crash.
///
/// On Unix the directory is opened and synced with `sync_all` (`fsync`, or `F_FULLFSYNC` on
/// macOS). On Windows it is opened for writing with `FILE_FLAG_BACKUP_SEMANTICS`, which opening a
/// directory needs, and flushed with `FlushFileBuffers`.
///
/// # Errors
///
/// The OS error from opening or syncing the directory.
pub fn sync_dir(path: &Path) -> io::Result<()> {
    imp::sync_dir(path)
}

/// Whether only the current user can access the file or directory at `path`, by the rules in the
/// module documentation. A symbolic link at `path` is followed.
///
/// # Errors
///
/// The OS error from reading the object's metadata or security descriptor, such as
/// [`io::ErrorKind::NotFound`].
pub fn is_owner_only(path: &Path) -> io::Result<bool> {
    imp::is_owner_only(path)
}

/// The error for a directory another user owns.
#[cfg(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    windows
))]
fn foreign_owner(path: &Path) -> io::Error {
    io::Error::new(
        io::ErrorKind::PermissionDenied,
        format!(
            "{} is owned by another user, so it cannot be made private to this one",
            path.display()
        ),
    )
}

#[cfg(any(target_os = "linux", target_os = "android", target_os = "macos"))]
mod imp {
    use std::fs::{File, OpenOptions, Permissions};
    use std::io::{self, Write};
    use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt};
    use std::path::Path;

    use super::foreign_owner;

    /// The mode of a private directory.
    const DIR_MODE: u32 = 0o700;
    /// The mode of a private file.
    const FILE_MODE: u32 = 0o600;

    fn euid() -> u32 {
        // SAFETY: `geteuid` takes no arguments, touches no memory and cannot fail.
        unsafe { libc::geteuid() }
    }

    /// Opens a directory read-only. `O_DIRECTORY` makes anything else fail at once, so a FIFO at
    /// the path cannot block the open.
    fn open_dir(path: &Path) -> io::Result<File> {
        OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY)
            .open(path)
    }

    pub(super) fn private_dir(path: &Path) -> io::Result<()> {
        match std::fs::DirBuilder::new().mode(DIR_MODE).create(path) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error),
        }
        // Check and change the directory through one descriptor, also when it was just created:
        // the umask may have cleared bits of 0700.
        let dir = open_dir(path)?;
        let metadata = dir.metadata()?;
        if metadata.uid() != euid() {
            return Err(foreign_owner(path));
        }
        if metadata.mode() & 0o7777 != DIR_MODE {
            dir.set_permissions(Permissions::from_mode(DIR_MODE))?;
        }
        Ok(())
    }

    pub(super) fn create_private_file(path: &Path, bytes: &[u8]) -> io::Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(FILE_MODE)
            .open(path)?;
        let written = (|| {
            file.set_permissions(Permissions::from_mode(FILE_MODE))?;
            file.write_all(bytes)?;
            file.sync_all()
        })();
        if written.is_err() {
            drop(file);
            // The file is this call's own, created above; the write's error is the one to report.
            let _ = std::fs::remove_file(path);
        }
        written
    }

    pub(super) fn sync_dir(path: &Path) -> io::Result<()> {
        open_dir(path)?.sync_all()
    }

    pub(super) fn is_owner_only(path: &Path) -> io::Result<bool> {
        let metadata = std::fs::metadata(path)?;
        if metadata.uid() != euid() || metadata.mode() & 0o077 != 0 {
            return Ok(false);
        }
        #[cfg(target_os = "macos")]
        if acl::grants_access(path)? {
            return Ok(false);
        }
        Ok(true)
    }

    /// macOS extended ACLs, which the mode bits do not show (`acl(3)`).
    #[cfg(target_os = "macos")]
    mod acl {
        use std::ffi::{CString, c_char, c_int, c_void};
        use std::io;
        use std::os::unix::ffi::OsStrExt;
        use std::path::Path;

        /// `ACL_TYPE_EXTENDED` in `<sys/acl.h>`.
        const ACL_TYPE_EXTENDED: c_int = 0x0000_0100;
        /// `ACL_FIRST_ENTRY` and `ACL_NEXT_ENTRY`.
        const ACL_FIRST_ENTRY: c_int = 0;
        const ACL_NEXT_ENTRY: c_int = -1;
        /// `ACL_EXTENDED_DENY`, the tag of an entry that only takes access away.
        const ACL_EXTENDED_DENY: c_int = 2;

        // The `acl(3)` functions of libSystem, which the `libc` crate does not declare on macOS.
        unsafe extern "C" {
            fn acl_get_file(path: *const c_char, kind: c_int) -> *mut c_void;
            fn acl_get_entry(acl: *mut c_void, entry_id: c_int, entry: *mut *mut c_void) -> c_int;
            fn acl_get_tag_type(entry: *mut c_void, tag: *mut c_int) -> c_int;
            fn acl_free(object: *mut c_void) -> c_int;
        }

        /// An ACL from `acl_get_file`, freed on drop.
        struct Acl(*mut c_void);

        impl Drop for Acl {
            fn drop(&mut self) {
                // SAFETY: the pointer came from `acl_get_file`, and only this frees it.
                unsafe { acl_free(self.0) };
            }
        }

        /// Whether the extended ACL of `path` has an entry other than a deny entry.
        pub(super) fn grants_access(path: &Path) -> io::Result<bool> {
            let name = CString::new(path.as_os_str().as_bytes()).map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidInput, "the path holds a NUL byte")
            })?;
            // SAFETY: a NUL-terminated path that outlives the call; the result is an ACL that
            // `Acl` frees, or null with errno set.
            let acl = unsafe { acl_get_file(name.as_ptr(), ACL_TYPE_EXTENDED) };
            if acl.is_null() {
                let error = io::Error::last_os_error();
                // ENOENT for an object that exists means it has no extended ACL.
                return match error.raw_os_error() {
                    Some(libc::ENOENT) => Ok(false),
                    _ => Err(error),
                };
            }
            let acl = Acl(acl);
            let mut which = ACL_FIRST_ENTRY;
            loop {
                let mut entry: *mut c_void = std::ptr::null_mut();
                // SAFETY: a live ACL and a live output; on success `entry` points into the ACL.
                if unsafe { acl_get_entry(acl.0, which, &mut entry) } != 0 {
                    // Past the last entry.
                    return Ok(false);
                }
                which = ACL_NEXT_ENTRY;
                let mut tag: c_int = 0;
                // SAFETY: an entry of the live ACL and a live output.
                if unsafe { acl_get_tag_type(entry, &mut tag) } != 0 {
                    return Err(io::Error::last_os_error());
                }
                if tag != ACL_EXTENDED_DENY {
                    return Ok(true);
                }
            }
        }
    }
}

#[cfg(windows)]
mod imp {
    use std::fs::{File, OpenOptions};
    use std::io::{self, Write};
    use std::os::windows::ffi::OsStrExt;
    use std::os::windows::fs::OpenOptionsExt;
    use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
    use std::path::Path;

    use windows_sys::Win32::Foundation::{
        GENERIC_WRITE, HANDLE, HLOCAL, INVALID_HANDLE_VALUE, LocalFree,
    };
    use windows_sys::Win32::Security::Authorization::{
        ConvertSidToStringSidW, ConvertStringSecurityDescriptorToSecurityDescriptorW,
        GetNamedSecurityInfoW, SDDL_REVISION_1, SE_FILE_OBJECT, SetNamedSecurityInfoW,
    };
    use windows_sys::Win32::Security::{
        ACCESS_ALLOWED_ACE, ACL, ACL_SIZE_INFORMATION, AclSizeInformation,
        DACL_SECURITY_INFORMATION, EqualSid, GetAce, GetAclInformation,
        GetSecurityDescriptorControl, GetSecurityDescriptorDacl, GetSecurityDescriptorOwner,
        GetTokenInformation, OBJECT_SECURITY_INFORMATION, OWNER_SECURITY_INFORMATION,
        PROTECTED_DACL_SECURITY_INFORMATION, PSECURITY_DESCRIPTOR, PSID, SE_DACL_PROTECTED,
        SECURITY_ATTRIBUTES, TOKEN_QUERY, TOKEN_USER, TokenUser,
    };
    use windows_sys::Win32::Storage::FileSystem::{
        CREATE_NEW, CreateDirectoryW, CreateFileW, FILE_ATTRIBUTE_NORMAL,
        FILE_FLAG_BACKUP_SEMANTICS,
    };
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

    use super::foreign_owner;

    /// `path` as a NUL-terminated UTF-16 string.
    fn wide(path: &Path) -> io::Result<Vec<u16>> {
        let mut units: Vec<u16> = path.as_os_str().encode_wide().collect();
        if units.contains(&0) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "the path holds a NUL character",
            ));
        }
        units.push(0);
        Ok(units)
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
    pub(super) struct UserSid {
        /// The `TOKEN_USER` and the SID it points to. `u64`s keep the `TOKEN_USER` aligned.
        buffer: Vec<u64>,
    }

    impl UserSid {
        pub(super) fn current() -> io::Result<UserSid> {
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

        /// Whether `sid` is this user's SID. `sid` must be null or a valid SID.
        fn is(&self, sid: PSID) -> bool {
            // SAFETY: both SIDs are valid: the user's lives in `self`, and the caller passes a
            // valid one, checked for null here.
            !sid.is_null() && unsafe { EqualSid(sid, self.sid()) } != 0
        }

        /// The SID in its string form, such as `S-1-5-21-…`.
        pub(super) fn text(&self) -> io::Result<String> {
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

    /// A `LocalAlloc`ed security descriptor, freed on drop.
    pub(super) struct Descriptor(PSECURITY_DESCRIPTOR);

    impl Descriptor {
        /// The descriptor an SDDL string describes.
        pub(super) fn from_sddl(sddl: &str) -> io::Result<Descriptor> {
            let sddl: Vec<u16> = sddl.encode_utf16().chain(std::iter::once(0)).collect();
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
            Ok(Descriptor(descriptor))
        }

        /// The owner-only descriptor for a directory: the user as owner, and a protected DACL
        /// granting the user full control, inherited by objects and containers.
        fn private_dir(user: &UserSid) -> io::Result<Descriptor> {
            let sid = user.text()?;
            Descriptor::from_sddl(&format!("O:{sid}D:P(A;OICI;FA;;;{sid})"))
        }

        /// The owner-only descriptor for a file: the user as owner, and a protected DACL granting
        /// the user full control.
        fn private_file(user: &UserSid) -> io::Result<Descriptor> {
            let sid = user.text()?;
            Descriptor::from_sddl(&format!("O:{sid}D:P(A;;FA;;;{sid})"))
        }

        /// The DACL inside the descriptor, or null for a descriptor with none.
        fn dacl(&self) -> io::Result<*mut ACL> {
            let mut present = 0;
            let mut defaulted = 0;
            let mut dacl: *mut ACL = std::ptr::null_mut();
            // SAFETY: a live descriptor and three live outputs; `dacl` points into the descriptor.
            if unsafe { GetSecurityDescriptorDacl(self.0, &mut present, &mut dacl, &mut defaulted) }
                == 0
            {
                return Err(io::Error::last_os_error());
            }
            Ok(if present == 0 {
                std::ptr::null_mut()
            } else {
                dacl
            })
        }

        /// Whether the descriptor's DACL is protected from inheritance.
        fn is_protected(&self) -> io::Result<bool> {
            let mut control = 0u16;
            let mut revision = 0u32;
            // SAFETY: a live descriptor and two live outputs.
            if unsafe { GetSecurityDescriptorControl(self.0, &mut control, &mut revision) } == 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(control & SE_DACL_PROTECTED != 0)
        }

        /// `SECURITY_ATTRIBUTES` that point at this descriptor, valid while `self` lives.
        fn attributes(&self) -> SECURITY_ATTRIBUTES {
            SECURITY_ATTRIBUTES {
                nLength: std::mem::size_of::<SECURITY_ATTRIBUTES>() as u32,
                lpSecurityDescriptor: self.0,
                bInheritHandle: 0,
            }
        }

        /// Writes the parts of this descriptor that `info` names onto the file or directory
        /// `name`, with `SetNamedSecurityInfoW`.
        pub(super) fn apply(
            &self,
            name: &[u16],
            info: OBJECT_SECURITY_INFORMATION,
        ) -> io::Result<()> {
            let mut owner: PSID = std::ptr::null_mut();
            if info & OWNER_SECURITY_INFORMATION != 0 {
                let mut defaulted = 0;
                // SAFETY: a live descriptor and two live outputs; `owner` points into it.
                if unsafe { GetSecurityDescriptorOwner(self.0, &mut owner, &mut defaulted) } == 0 {
                    return Err(io::Error::last_os_error());
                }
            }
            let dacl = self.dacl()?;
            // SAFETY: a NUL-terminated path; the owner and DACL point into this live descriptor,
            // and the call reads only the parts `info` names; no group or SACL.
            let status = unsafe {
                SetNamedSecurityInfoW(
                    name.as_ptr(),
                    SE_FILE_OBJECT,
                    info,
                    owner,
                    std::ptr::null_mut(),
                    dacl,
                    std::ptr::null(),
                )
            };
            if status != 0 {
                return Err(io::Error::from_raw_os_error(status as i32));
            }
            Ok(())
        }
    }

    impl Drop for Descriptor {
        fn drop(&mut self) {
            // SAFETY: the descriptor came from `LocalAlloc`, and only this frees it.
            unsafe { LocalFree(self.0 as HLOCAL) };
        }
    }

    /// The owner and DACL of a file or directory, as `GetNamedSecurityInfoW` reads them.
    struct Security {
        descriptor: Descriptor,
        owner: PSID,
        dacl: *mut ACL,
    }

    impl Security {
        fn read(name: &[u16]) -> io::Result<Security> {
            let mut descriptor: PSECURITY_DESCRIPTOR = std::ptr::null_mut();
            let mut owner: PSID = std::ptr::null_mut();
            let mut dacl: *mut ACL = std::ptr::null_mut();
            // SAFETY: a NUL-terminated path; live owner, DACL and descriptor outputs, and null
            // group and SACL outputs. On success the descriptor is `LocalAlloc`ed, and the owner
            // and DACL point into it.
            let status = unsafe {
                GetNamedSecurityInfoW(
                    name.as_ptr(),
                    SE_FILE_OBJECT,
                    OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION,
                    &mut owner,
                    std::ptr::null_mut(),
                    &mut dacl,
                    std::ptr::null_mut(),
                    &mut descriptor,
                )
            };
            if status != 0 {
                return Err(io::Error::from_raw_os_error(status as i32));
            }
            Ok(Security {
                descriptor: Descriptor(descriptor),
                owner,
                dacl,
            })
        }

        /// Whether the owner is `user` and every entry of the DACL allows `user`. A null DACL
        /// allows everyone; an empty one allows no one, and the owner keeps only its implicit
        /// rights to read and change the descriptor.
        fn is_owner_only(&self, user: &UserSid) -> io::Result<bool> {
            if !user.is(self.owner) || self.dacl.is_null() {
                return Ok(false);
            }
            let mut size = ACL_SIZE_INFORMATION {
                AceCount: 0,
                AclBytesInUse: 0,
                AclBytesFree: 0,
            };
            // SAFETY: a live DACL, kept alive by `descriptor`, and a live output of the size the
            // class needs.
            if unsafe {
                GetAclInformation(
                    self.dacl,
                    (&raw mut size).cast(),
                    std::mem::size_of::<ACL_SIZE_INFORMATION>() as u32,
                    AclSizeInformation,
                )
            } == 0
            {
                return Err(io::Error::last_os_error());
            }
            for index in 0..size.AceCount {
                let mut ace: *mut core::ffi::c_void = std::ptr::null_mut();
                // SAFETY: a live DACL, an index below its entry count, and a live output; on
                // success `ace` points into the DACL.
                if unsafe { GetAce(self.dacl, index, &mut ace) } == 0 {
                    return Err(io::Error::last_os_error());
                }
                let ace = ace.cast::<ACCESS_ALLOWED_ACE>();
                // SAFETY: `GetAce` returned a live entry, which starts with its header.
                let kind = unsafe { (*ace).Header.AceType };
                // `ACCESS_ALLOWED_ACE_TYPE` is 0 (it lives behind another windows-sys feature).
                // Any other type, a deny or callback or object entry among them, fails the check.
                if kind != 0 {
                    return Ok(false);
                }
                // SAFETY: an access-allowed entry's SID starts at its `SidStart` field, inside
                // the DACL, which `descriptor` keeps alive.
                let sid: PSID = unsafe { (&raw const (*ace).SidStart).cast_mut().cast() };
                if !user.is(sid) {
                    return Ok(false);
                }
            }
            Ok(true)
        }
    }

    pub(super) fn private_dir(path: &Path) -> io::Result<()> {
        let name = wide(path)?;
        let user = UserSid::current()?;
        let descriptor = Descriptor::private_dir(&user)?;
        let attributes = descriptor.attributes();
        // SAFETY: a NUL-terminated path, and attributes whose descriptor `descriptor` keeps alive.
        if unsafe { CreateDirectoryW(name.as_ptr(), &attributes) } != 0 {
            return Ok(());
        }
        let error = io::Error::last_os_error();
        if error.kind() != io::ErrorKind::AlreadyExists {
            return Err(error);
        }
        if !std::fs::metadata(path)?.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::NotADirectory,
                format!("{} exists and is not a directory", path.display()),
            ));
        }
        let current = Security::read(&name)?;
        if !user.is(current.owner) {
            return Err(foreign_owner(path));
        }
        if current.is_owner_only(&user)? && current.descriptor.is_protected()? {
            return Ok(());
        }
        descriptor.apply(
            &name,
            DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
        )
    }

    pub(super) fn create_private_file(path: &Path, bytes: &[u8]) -> io::Result<()> {
        let name = wide(path)?;
        let descriptor = Descriptor::private_file(&UserSid::current()?)?;
        let attributes = descriptor.attributes();
        // SAFETY: a NUL-terminated path, attributes whose descriptor `descriptor` keeps alive, no
        // sharing, and no template file.
        let handle = unsafe {
            CreateFileW(
                name.as_ptr(),
                GENERIC_WRITE,
                0,
                &attributes,
                CREATE_NEW,
                FILE_ATTRIBUTE_NORMAL,
                std::ptr::null_mut(),
            )
        };
        if handle == INVALID_HANDLE_VALUE {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: `handle` is the valid file handle `CreateFileW` just returned, owned only here.
        let mut file = unsafe { File::from_raw_handle(handle.cast()) };
        let written = file.write_all(bytes).and_then(|()| file.sync_all());
        if written.is_err() {
            drop(file);
            // The file is this call's own, created above; the write's error is the one to report.
            let _ = std::fs::remove_file(path);
        }
        written
    }

    pub(super) fn sync_dir(path: &Path) -> io::Result<()> {
        OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
            .open(path)?
            .sync_all()
    }

    pub(super) fn is_owner_only(path: &Path) -> io::Result<bool> {
        let name = wide(path)?;
        let user = UserSid::current()?;
        Security::read(&name)?.is_owner_only(&user)
    }

    /// For the tests: the path as the wide string [`Descriptor::apply`] takes.
    #[cfg(test)]
    pub(super) fn wide_path(path: &Path) -> Vec<u16> {
        wide(path).expect("a test path has no NUL")
    }
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    windows
)))]
mod imp {
    use std::io;
    use std::path::Path;

    fn unsupported() -> io::Error {
        io::Error::new(
            io::ErrorKind::Unsupported,
            "owner-only files are not implemented on this OS",
        )
    }

    pub(super) fn private_dir(_path: &Path) -> io::Result<()> {
        Err(unsupported())
    }

    pub(super) fn create_private_file(_path: &Path, _bytes: &[u8]) -> io::Result<()> {
        Err(unsupported())
    }

    pub(super) fn sync_dir(_path: &Path) -> io::Result<()> {
        Err(unsupported())
    }

    pub(super) fn is_owner_only(_path: &Path) -> io::Result<bool> {
        Err(unsupported())
    }
}

#[cfg(all(
    test,
    any(
        target_os = "linux",
        target_os = "android",
        target_os = "macos",
        windows
    )
))]
mod tests {
    use std::io;
    use std::path::{Path, PathBuf};

    use super::{create_private_file, is_owner_only, private_dir, sync_dir};

    /// A directory under the system temporary directory, removed with its contents on drop.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(name: &str) -> Scratch {
            let path = std::env::temp_dir()
                .join(format!("syntropy-io-private-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&path);
            Scratch(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn private_dir_creates_an_owner_only_directory_and_its_parents() {
        let scratch = Scratch::new("create");
        let dir = scratch.path().join("parent").join("state");
        private_dir(&dir).expect("private_dir");
        assert!(dir.is_dir());
        assert!(
            is_owner_only(&dir).expect("is_owner_only"),
            "the directory is owner-only"
        );
        // A second call on the directory it made changes nothing and succeeds.
        private_dir(&dir).expect("private_dir again");
        assert!(is_owner_only(&dir).expect("is_owner_only"));
    }

    #[test]
    fn create_private_file_writes_an_owner_only_file_and_refuses_an_existing_one() {
        let scratch = Scratch::new("file");
        private_dir(scratch.path()).expect("private_dir");
        let key = scratch.path().join("node.key");
        create_private_file(&key, &[7u8; 32]).expect("create_private_file");
        assert_eq!(std::fs::read(&key).expect("read"), [7u8; 32]);
        assert!(
            is_owner_only(&key).expect("is_owner_only"),
            "the file is owner-only"
        );

        let again = create_private_file(&key, &[9u8; 32]).expect_err("the file exists");
        assert_eq!(again.kind(), io::ErrorKind::AlreadyExists);
        assert_eq!(
            std::fs::read(&key).expect("read"),
            [7u8; 32],
            "the existing file is left as it was"
        );
    }

    #[test]
    fn sync_dir_succeeds_on_a_private_directory_and_a_plain_one() {
        let scratch = Scratch::new("sync");
        let dir = scratch.path().join("state");
        private_dir(&dir).expect("private_dir");
        create_private_file(&dir.join("node.key"), b"seed").expect("create_private_file");
        sync_dir(&dir).expect("sync_dir on the private directory");
        sync_dir(scratch.path()).expect("sync_dir on its parent");
        sync_dir(&std::env::temp_dir()).expect("sync_dir on the temporary directory");
    }

    #[test]
    fn a_missing_path_is_an_error_not_an_answer() {
        let scratch = Scratch::new("missing");
        let error = is_owner_only(&scratch.path().join("absent")).expect_err("no such file");
        assert_eq!(error.kind(), io::ErrorKind::NotFound);
    }

    #[test]
    fn private_dir_refuses_a_file_in_the_way() {
        let scratch = Scratch::new("notdir");
        std::fs::create_dir_all(scratch.path()).expect("mkdir");
        let file = scratch.path().join("state");
        std::fs::write(&file, b"x").expect("write");
        assert!(
            private_dir(&file).is_err(),
            "a file is not made into a directory"
        );
    }

    #[cfg(unix)]
    mod unix {
        use std::fs::Permissions;
        use std::os::unix::fs::PermissionsExt;

        use super::super::{create_private_file, is_owner_only, private_dir};
        use super::Scratch;

        fn mode(path: &std::path::Path) -> u32 {
            std::fs::metadata(path)
                .expect("metadata")
                .permissions()
                .mode()
                & 0o7777
        }

        #[test]
        fn the_modes_are_exactly_0700_and_0600() {
            let scratch = Scratch::new("modes");
            private_dir(scratch.path()).expect("private_dir");
            assert_eq!(mode(scratch.path()), 0o700);
            let key = scratch.path().join("node.key");
            create_private_file(&key, b"seed").expect("create_private_file");
            assert_eq!(mode(&key), 0o600);
        }

        #[test]
        fn a_world_readable_file_is_not_owner_only() {
            let scratch = Scratch::new("loosefile");
            private_dir(scratch.path()).expect("private_dir");
            let file = scratch.path().join("loose");
            std::fs::write(&file, b"x").expect("write");
            for loose in [0o644, 0o640, 0o604, 0o620] {
                std::fs::set_permissions(&file, Permissions::from_mode(loose)).expect("chmod");
                assert!(
                    !is_owner_only(&file).expect("is_owner_only"),
                    "mode {loose:o}"
                );
            }
            std::fs::set_permissions(&file, Permissions::from_mode(0o400)).expect("chmod");
            assert!(
                is_owner_only(&file).expect("is_owner_only"),
                "mode 400 is owner-only"
            );
        }

        #[test]
        fn private_dir_tightens_an_existing_loose_directory() {
            let scratch = Scratch::new("loosedir");
            std::fs::create_dir_all(scratch.path()).expect("mkdir");
            std::fs::set_permissions(scratch.path(), Permissions::from_mode(0o755)).expect("chmod");
            assert!(!is_owner_only(scratch.path()).expect("is_owner_only"));
            private_dir(scratch.path()).expect("private_dir");
            assert_eq!(mode(scratch.path()), 0o700);
            assert!(is_owner_only(scratch.path()).expect("is_owner_only"));
        }

        #[test]
        fn private_dir_refuses_a_directory_another_user_owns() {
            // SAFETY: `geteuid` takes no arguments, touches no memory and cannot fail.
            if unsafe { libc::geteuid() } == 0 {
                eprintln!("skipped: root owns the root directory");
                return;
            }
            let root = std::path::Path::new("/");
            let error = private_dir(root).expect_err("root owns /");
            assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
            assert!(
                error.to_string().contains("owned by another user"),
                "refused by the owner check, before any chmod: {error}"
            );
        }

        /// macOS extended ACLs do not show in the mode bits, so the check reads them.
        #[cfg(target_os = "macos")]
        #[test]
        fn a_macos_acl_entry_that_allows_access_is_not_owner_only() {
            let scratch = Scratch::new("acl");
            private_dir(scratch.path()).expect("private_dir");
            let key = scratch.path().join("node.key");
            create_private_file(&key, b"seed").expect("create_private_file");
            let chmod = |entry: &str| {
                let status = std::process::Command::new("/bin/chmod")
                    .arg("+a")
                    .arg(entry)
                    .arg(&key)
                    .status()
                    .expect("run chmod");
                assert!(status.success(), "chmod +a {entry}");
            };
            chmod("everyone deny write");
            assert!(
                is_owner_only(&key).expect("is_owner_only"),
                "a deny entry takes access away and keeps the file owner-only"
            );
            chmod("everyone allow read");
            assert_eq!(mode(&key), 0o600, "the mode bits do not show the entry");
            assert!(!is_owner_only(&key).expect("is_owner_only"));
        }
    }

    #[cfg(windows)]
    mod windows {
        use windows_sys::Win32::Security::{
            DACL_SECURITY_INFORMATION, OWNER_SECURITY_INFORMATION,
            PROTECTED_DACL_SECURITY_INFORMATION, UNPROTECTED_DACL_SECURITY_INFORMATION,
        };

        use super::super::imp::{Descriptor, UserSid, wide_path};
        use super::super::{create_private_file, is_owner_only, private_dir};
        use super::Scratch;

        fn sid() -> String {
            UserSid::current()
                .expect("user SID")
                .text()
                .expect("SID text")
        }

        /// Sets the DACL `dacl`, in SDDL, on `path`, protected from inheritance.
        fn set_dacl(path: &std::path::Path, dacl: &str) {
            Descriptor::from_sddl(dacl)
                .expect("SDDL")
                .apply(
                    &wide_path(path),
                    DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
                )
                .expect("SetNamedSecurityInfoW");
        }

        #[test]
        fn a_file_that_grants_another_principal_is_not_owner_only() {
            let scratch = Scratch::new("winloose");
            private_dir(scratch.path()).expect("private_dir");
            let file = scratch.path().join("node.key");
            create_private_file(&file, b"seed").expect("create_private_file");
            let me = sid();
            // Everyone (WD), Users (BU), Authenticated Users (AU), SYSTEM (SY) and the
            // Administrators group (BA): each extra grant fails the check.
            for other in ["WD", "BU", "AU", "SY", "BA"] {
                set_dacl(&file, &format!("D:P(A;;FA;;;{me})(A;;FR;;;{other})"));
                assert!(
                    !is_owner_only(&file).expect("is_owner_only"),
                    "a grant to {other}"
                );
            }
            // A deny entry is not an allow entry for the user, so the strict check fails it too.
            set_dacl(&file, &format!("D:P(A;;FA;;;{me})(D;;FW;;;WD)"));
            assert!(
                !is_owner_only(&file).expect("is_owner_only"),
                "a deny entry"
            );
            // Back to the user alone.
            set_dacl(&file, &format!("D:P(A;;FA;;;{me})"));
            assert!(
                is_owner_only(&file).expect("is_owner_only"),
                "the user alone"
            );
        }

        #[test]
        fn a_file_with_inherited_default_permissions_is_not_owner_only() {
            // A file made the ordinary way in the temporary directory inherits its parent's DACL,
            // which grants SYSTEM and the Administrators group as well as the user.
            let scratch = Scratch::new("windefault");
            std::fs::create_dir_all(scratch.path()).expect("mkdir");
            let file = scratch.path().join("plain");
            std::fs::write(&file, b"x").expect("write");
            assert!(!is_owner_only(&file).expect("is_owner_only"));
        }

        #[test]
        fn private_dir_tightens_an_existing_loose_directory() {
            let scratch = Scratch::new("winloosedir");
            std::fs::create_dir_all(scratch.path()).expect("mkdir");
            let me = sid();
            // Make the user the owner, since an elevated token's default owner is the
            // Administrators group, and grant Everyone full control, inherited, unprotected.
            Descriptor::from_sddl(&format!("O:{me}D:(A;OICI;FA;;;WD)(A;OICI;FA;;;{me})"))
                .expect("SDDL")
                .apply(
                    &wide_path(scratch.path()),
                    OWNER_SECURITY_INFORMATION
                        | DACL_SECURITY_INFORMATION
                        | UNPROTECTED_DACL_SECURITY_INFORMATION,
                )
                .expect("loosen");
            assert!(!is_owner_only(scratch.path()).expect("is_owner_only"));
            private_dir(scratch.path()).expect("private_dir");
            assert!(is_owner_only(scratch.path()).expect("is_owner_only"));
        }

        #[test]
        fn private_dir_refuses_a_directory_another_user_owns() {
            // The Windows directory belongs to TrustedInstaller. The owner check refuses it
            // before any change, and the error says so.
            let windows = std::env::var_os("SystemRoot").expect("SystemRoot");
            let error = private_dir(std::path::Path::new(&windows)).expect_err("not ours");
            assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
            assert!(
                error.to_string().contains("owned by another user"),
                "refused by the owner check, before any change: {error}"
            );
        }
    }
}
