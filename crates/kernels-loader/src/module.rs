// SPDX-License-Identifier: MIT OR Apache-2.0

//! Opening a provider library and reading its table (RFC-0040 §10.4).
//!
//! [`Module::load`] opens the library by absolute path, resolves the entry symbol for this
//! core's ABI major version, calls it with the host table and checks the table it gets back just
//! enough to read it safely: the magic, and that every struct is at least as large as the one
//! this build knows. The ordered compatibility check of §10.4 (minor version, build ID, provider
//! IDs against `module.toml`) is COMP-E9-T05's.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use syntropy_core::ErrorClass;
use syntropy_kernels::abi::{
    ABI_MAJOR, ENTRY_SYMBOL, EntryFn, MAGIC, SynModule, SynProviderVtbl, types::SynHeader,
};
use syntropy_kernels::{Error, ErrorCode};

use crate::host::HostServices;
use crate::provider::ModuleProvider;
use crate::sys::RawLibrary;

/// `ModuleMissing` (RFC-0040 §13 code 2): the library cannot be opened.
const MODULE_MISSING: ErrorCode = ErrorCode(2);
/// `ModuleAbi` (RFC-0040 §13 code 4): the library is not a provider library of this ABI.
const MODULE_ABI: ErrorCode = ErrorCode(4);

/// Why a module did not load.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModuleError(pub Error);

impl fmt::Display for ModuleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl std::error::Error for ModuleError {}

impl From<ModuleError> for Error {
    fn from(e: ModuleError) -> Self {
        e.0
    }
}

fn abi_error(message: String) -> ModuleError {
    ModuleError(Error {
        class: ErrorClass::Unsupported,
        code: MODULE_ABI,
        message,
    })
}

/// A loaded provider library.
pub struct Module {
    path: PathBuf,
    header: SynHeader,
    providers: Vec<ModuleProvider>,
}

impl fmt::Debug for Module {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Module")
            .field("path", &self.path)
            .field("header", &self.header)
            .field("providers", &self.providers.len())
            .finish()
    }
}

impl Module {
    /// Opens the provider library at the absolute `path` and wraps its providers.
    ///
    /// The services installed with [`HostServices::install`] are what the module sees as the
    /// core; with none installed, a silent set is used.
    ///
    /// # Errors
    ///
    /// `ModuleMissing` (2) if the library cannot be opened. `ModuleAbi` (4) if it exports no
    /// entry symbol for this ABI major version, the entry call fails, or the table it returns is
    /// not one this build can read.
    pub fn load(path: &Path) -> Result<Self, ModuleError> {
        let lib = RawLibrary::open(path).map_err(|reason| {
            ModuleError(Error {
                class: ErrorClass::Unsupported,
                code: MODULE_MISSING,
                message: format!(
                    "cannot open the provider library {}: {reason}",
                    path.display()
                ),
            })
        })?;
        let lib = Arc::new(lib);
        let entry = lib.symbol(ENTRY_SYMBOL).map_err(|reason| {
            abi_error(format!(
                "{} exports no `{ENTRY_SYMBOL}` (this core is ABI {ABI_MAJOR}): {reason}",
                path.display()
            ))
        })?;
        // SAFETY: the symbol is named for ABI 1 and has the signature `EntryFn` by the ABI's
        // definition; a library that exports it under another type breaks its own contract.
        let entry: EntryFn =
            unsafe { std::mem::transmute::<*mut std::ffi::c_void, EntryFn>(entry) };
        let services = HostServices::current().unwrap_or_else(|| HostServices::silent([0; 32]));
        let host = services.table();
        // SAFETY: `host` is a valid, fully initialised host table that lives through the call.
        let table = unsafe { entry(&raw const host) };
        if table.is_null() {
            return Err(abi_error(format!(
                "{} refused the core's host table: the entry call returned null",
                path.display()
            )));
        }
        // SAFETY: the table is a `SynModule` the module leaked, so it lives for the process; its
        // header is the prefix of the struct, so it is read first and the rest only after the
        // size check below.
        let header = unsafe { std::ptr::read_unaligned(table.cast::<SynHeader>()) };
        if header.magic != MAGIC {
            return Err(abi_error(format!(
                "{}: the table's magic is wrong",
                path.display()
            )));
        }
        if (header.struct_size as usize) < size_of::<SynModule>() {
            return Err(abi_error(format!(
                "{}: the module table is {} bytes, smaller than the {} this core knows",
                path.display(),
                header.struct_size,
                size_of::<SynModule>()
            )));
        }
        // SAFETY: the magic and size checks passed, so the whole `SynModule` is readable.
        let module = unsafe { &*table };
        let providers = read_providers(module, path, &lib)?;
        Ok(Self {
            path: path.to_path_buf(),
            header,
            providers,
        })
    }

    /// The library's path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The module's header, as the module built it.
    #[must_use]
    pub fn header(&self) -> &SynHeader {
        &self.header
    }

    /// The providers the module returned.
    #[must_use]
    pub fn providers(&self) -> &[ModuleProvider] {
        &self.providers
    }

    /// Takes the providers out of the module.
    #[must_use]
    pub fn into_providers(self) -> Vec<ModuleProvider> {
        self.providers
    }
}

fn read_providers(
    module: &SynModule,
    path: &Path,
    lib: &Arc<RawLibrary>,
) -> Result<Vec<ModuleProvider>, ModuleError> {
    let count = module.n_providers as usize;
    if count == 0 {
        return Ok(Vec::new());
    }
    if module.providers.is_null() {
        return Err(abi_error(format!(
            "{}: the provider table is null",
            path.display()
        )));
    }
    // Every table is a `SynProviderVtbl` of the module's build, in one array, so the stride is
    // the first table's `struct_size`.
    // SAFETY: the pointer is non-null and the module reported `count > 0` tables; the first
    // `u32` of a table is its `struct_size`.
    let stride = unsafe { std::ptr::read_unaligned(module.providers.cast::<u32>()) } as usize;
    if stride < size_of::<SynProviderVtbl>() {
        return Err(abi_error(format!(
            "{}: a provider table is {stride} bytes, smaller than the {} this core knows",
            path.display(),
            size_of::<SynProviderVtbl>()
        )));
    }
    (0..count)
        .map(|n| {
            // SAFETY: the module built `count` tables of `stride` bytes each, contiguously.
            let at = unsafe { module.providers.cast::<u8>().add(n * stride) };
            // SAFETY: each table starts with its `struct_size`, which must equal the stride.
            let size = unsafe { std::ptr::read_unaligned(at.cast::<u32>()) } as usize;
            if size != stride {
                return Err(abi_error(format!(
                    "{}: provider table {n} is {size} bytes, not the {stride} of the first",
                    path.display()
                )));
            }
            // SAFETY: the table is at least `size_of::<SynProviderVtbl>()` bytes (checked
            // above), and a table is plain data, so a bitwise copy of our prefix is valid.
            let vtbl = unsafe { std::ptr::read_unaligned(at.cast::<SynProviderVtbl>()) };
            ModuleProvider::new(vtbl, Arc::clone(lib))
        })
        .collect()
}
