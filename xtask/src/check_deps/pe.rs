// SPDX-License-Identifier: MIT OR Apache-2.0

//! A minimal PE reader for rule 7: the DLLs a Windows binary imports, from its import directory
//! and its delay-load directory, and the names it imports from them.
//!
//! It reads PE32 and PE32+ files through the section table. Every offset is bounds-checked, so a
//! truncated or malformed file is an error, never a panic. The work is linear in the file's size:
//! lookup-table entries and name bytes are read against budgets that a real binary, whose tables
//! and names do not overlap, stays well inside, so a crafted file whose tables all point at the
//! same entries is an error, not quadratic time and memory.

use std::cell::Cell;

/// What the Windows loader sees of a binary.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Imports {
    /// The DLLs of the import directory, such as `KERNEL32.dll`, in file order.
    pub dlls: Vec<String>,
    /// The DLLs of the delay-load directory, which the binary loads on first call, in file order.
    pub delayed: Vec<String>,
    /// Every name imported by name from either directory, in file order. Imports by ordinal have
    /// no name, and are left out.
    pub symbols: Vec<String>,
}

/// The data directory entries this reader uses (PE format, "Optional Header Data Directories").
const IMPORT_DIRECTORY: u32 = 1;
const DELAY_IMPORT_DIRECTORY: u32 = 13;
const PE32_MAGIC: u16 = 0x10b;
const PE32_PLUS_MAGIC: u16 = 0x20b;
/// An import directory entry (`IMAGE_IMPORT_DESCRIPTOR`) and a delay-load one
/// (`ImgDelayDescr`).
const IMPORT_ENTRY: u64 = 20;
const DELAY_ENTRY: u64 = 32;
/// A delay-load entry whose `Attributes` lack this bit holds virtual addresses, not RVAs.
const DELAY_RVA_BASED: u32 = 1;
/// The name bytes a file may make the reader copy, beyond its own size: room for a linker that
/// shares the tail of one name with another.
const NAME_SLACK: u64 = 1 << 20;

/// Reads the import and delay-load directories of the PE file `bytes`. A file without them gives
/// an empty [`Imports`].
pub fn read(bytes: &[u8]) -> Result<Imports, String> {
    let pe = Pe::new(bytes)?;
    let mut imports = Imports::default();
    if let Some(table) = pe.directory(IMPORT_DIRECTORY)? {
        for index in 0..pe.max_entries(IMPORT_ENTRY) {
            let at = pe.offset(table)?.saturating_add(index * IMPORT_ENTRY);
            let entry: [u8; 20] = pe.bytes_at(at)?;
            if entry == [0; 20] {
                break;
            }
            let names = pe.u32(at)?;
            let name = pe.u32(at + 12)?;
            let addresses = pe.u32(at + 16)?;
            imports.dlls.push(pe.string(name)?);
            // A binder may overwrite the address table, so the name table is read when there is
            // one.
            let thunks = if names != 0 { names } else { addresses };
            pe.names(thunks, &mut imports.symbols)?;
        }
    }
    if let Some(table) = pe.directory(DELAY_IMPORT_DIRECTORY)? {
        for index in 0..pe.max_entries(DELAY_ENTRY) {
            let at = pe.offset(table)?.saturating_add(index * DELAY_ENTRY);
            let entry: [u8; 32] = pe.bytes_at(at)?;
            if entry == [0; 32] {
                break;
            }
            let rva_based = pe.u32(at)? & DELAY_RVA_BASED != 0;
            let name = pe.delay_address(pe.u32(at + 4)?, rva_based)?;
            let names = pe.delay_address(pe.u32(at + 16)?, rva_based)?;
            imports.delayed.push(pe.string(name)?);
            if names != 0 {
                pe.names(names, &mut imports.symbols)?;
            }
        }
    }
    Ok(imports)
}

struct Pe<'a> {
    bytes: &'a [u8],
    /// PE32+ (64-bit) rather than PE32.
    wide: bool,
    image_base: u64,
    /// Where the data directories start, and how many there are.
    directories: u64,
    directory_count: u32,
    sections: Vec<Section>,
    /// The lookup-table entries still to be read: one per 4 bytes of file, since the tables of a
    /// real binary do not overlap.
    entries_left: Cell<u64>,
    /// The name bytes still to be copied: twice the file's size plus [`NAME_SLACK`].
    name_bytes_left: Cell<u64>,
}

struct Section {
    address: u32,
    virtual_size: u32,
    raw_size: u32,
    raw_offset: u32,
}

impl<'a> Pe<'a> {
    fn new(bytes: &'a [u8]) -> Result<Pe<'a>, String> {
        if bytes.get(..2) != Some(b"MZ".as_slice()) {
            return Err("not a PE file".to_owned());
        }
        let mut pe = Pe {
            bytes,
            wide: false,
            image_base: 0,
            directories: 0,
            directory_count: 0,
            sections: Vec::new(),
            entries_left: Cell::new(bytes.len() as u64 / 4 + 64),
            name_bytes_left: Cell::new(2 * bytes.len() as u64 + NAME_SLACK),
        };
        let header = u64::from(pe.u32(0x3c)?);
        if pe.bytes_at::<4>(header)? != *b"PE\0\0" {
            return Err("the DOS header points at no PE signature".to_owned());
        }
        let coff = header + 4;
        let section_count = pe.u16(coff + 2)?;
        let optional_size = u64::from(pe.u16(coff + 16)?);
        let optional = coff + 20;
        let (base, count_at, directories) = match pe.u16(optional)? {
            PE32_MAGIC => (u64::from(pe.u32(optional + 28)?), 92, 96),
            PE32_PLUS_MAGIC => {
                pe.wide = true;
                (pe.u64(optional + 24)?, 108, 112)
            }
            magic => return Err(format!("unknown optional header magic {magic:#x}")),
        };
        if directories > optional_size {
            return Err("the optional header is too short for its data directories".to_owned());
        }
        pe.image_base = base;
        pe.directories = optional + directories;
        // The directories must fit in the optional header the COFF header sizes.
        let room = u32::try_from((optional_size - directories) / 8).unwrap_or(u32::MAX);
        pe.directory_count = pe.u32(optional + count_at)?.min(room);
        let table = optional + optional_size;
        for index in 0..u64::from(section_count) {
            let at = table + index * 40;
            pe.sections.push(Section {
                virtual_size: pe.u32(at + 8)?,
                address: pe.u32(at + 12)?,
                raw_size: pe.u32(at + 16)?,
                raw_offset: pe.u32(at + 20)?,
            });
        }
        Ok(pe)
    }

    fn bytes_at<const N: usize>(&self, at: u64) -> Result<[u8; N], String> {
        usize::try_from(at)
            .ok()
            .and_then(|at| self.bytes.get(at..at.checked_add(N)?))
            .and_then(|slice| slice.try_into().ok())
            .ok_or_else(|| format!("the file ends before offset {at}"))
    }

    fn u16(&self, at: u64) -> Result<u16, String> {
        self.bytes_at(at).map(u16::from_le_bytes)
    }

    fn u32(&self, at: u64) -> Result<u32, String> {
        self.bytes_at(at).map(u32::from_le_bytes)
    }

    fn u64(&self, at: u64) -> Result<u64, String> {
        self.bytes_at(at).map(u64::from_le_bytes)
    }

    /// The RVA of data directory `index`, or `None` when the binary has none. Only the address
    /// counts: the loader walks the import tables to their terminating entry and does not read the
    /// size, so a directory with an address and a size of 0 is still read.
    fn directory(&self, index: u32) -> Result<Option<u32>, String> {
        if index >= self.directory_count {
            return Ok(None);
        }
        let address = self.u32(self.directories + u64::from(index) * 8)?;
        Ok((address != 0).then_some(address))
    }

    /// An upper bound on the entries of a table, so a table with no terminating entry ends.
    fn max_entries(&self, entry: u64) -> u64 {
        self.bytes.len() as u64 / entry + 1
    }

    /// A delay-load field as an RVA: older linkers wrote virtual addresses.
    fn delay_address(&self, value: u32, rva_based: bool) -> Result<u32, String> {
        if rva_based || value == 0 {
            return Ok(value);
        }
        u64::from(value)
            .checked_sub(self.image_base)
            .and_then(|rva| u32::try_from(rva).ok())
            .ok_or_else(|| format!("the delay-load address {value:#x} is below the image base"))
    }

    /// The file offset of `rva`, which must fall in a section's data in the file.
    fn offset(&self, rva: u32) -> Result<u64, String> {
        for section in &self.sections {
            let size = if section.virtual_size == 0 {
                section.raw_size
            } else {
                section.virtual_size
            };
            let Some(inside) = rva.checked_sub(section.address) else {
                continue;
            };
            if inside < size {
                if inside >= section.raw_size {
                    return Err(format!("RVA {rva:#x} has no data in the file"));
                }
                return Ok(u64::from(section.raw_offset) + u64::from(inside));
            }
        }
        Err(format!("RVA {rva:#x} is in no section"))
    }

    /// The NUL-terminated string at `rva`.
    fn string(&self, rva: u32) -> Result<String, String> {
        let start = usize::try_from(self.offset(rva)?).map_err(|_| "offset overflows")?;
        let tail = self
            .bytes
            .get(start..)
            .ok_or_else(|| format!("the file ends before RVA {rva:#x}"))?;
        let length = tail
            .iter()
            .position(|byte| *byte == 0)
            .ok_or("a name has no terminating NUL")?;
        let left = self.name_bytes_left.get();
        let Some(left) = left.checked_sub(length as u64 + 1) else {
            return Err("the import names overlap more than a linked binary's can".to_owned());
        };
        self.name_bytes_left.set(left);
        String::from_utf8(tail[..length].to_vec()).map_err(|_| "a name is not UTF-8".to_owned())
    }

    /// Appends the names of the lookup table at `rva` to `names`. An entry with the top bit set
    /// imports by ordinal; any other points at a hint and a name.
    fn names(&self, rva: u32, names: &mut Vec<String>) -> Result<(), String> {
        let entry: u64 = if self.wide { 8 } else { 4 };
        let table = self.offset(rva)?;
        for index in 0..self.max_entries(entry) {
            let Some(left) = self.entries_left.get().checked_sub(1) else {
                return Err(
                    "the import lookup tables overlap more than a linked binary's can".into(),
                );
            };
            self.entries_left.set(left);
            let at = table.saturating_add(index * entry);
            let (value, by_ordinal) = if self.wide {
                let value = self.u64(at)?;
                (value, value >> 63 == 1)
            } else {
                let value = u64::from(self.u32(at)?);
                (value, value >> 31 == 1)
            };
            if value == 0 {
                break;
            }
            if by_ordinal {
                continue;
            }
            let hint = u32::try_from(value & 0x7fff_ffff).map_err(|_| "a name RVA overflows")?;
            let name = hint
                .checked_add(2)
                .ok_or_else(|| format!("the name RVA {hint:#x} overflows"))?;
            names.push(self.string(name)?);
        }
        Ok(())
    }
}

/// Builds small PE files for the tests here and in `release`.
#[cfg(test)]
pub mod fixture {
    /// One DLL and the names imported from it, by name or, for `None`, by ordinal.
    pub struct Dll<'a> {
        pub name: &'a str,
        pub imports: &'a [Option<&'a str>],
    }

    /// The RVA of the one section, `.idata`, which the file holds from offset 0x200.
    const SECTION_RVA: u32 = 0x1000;
    const SECTION_OFFSET: usize = 0x200;
    const IMAGE_BASE: u64 = 0x1_4000_0000;

    /// A PE32+ file (or a PE32 one, when `wide` is false) whose import directory lists
    /// `imports` and whose delay-load directory lists `delayed`. With `delay_rva` false, the
    /// delay-load entries hold virtual addresses, as old linkers wrote them.
    pub fn build(wide: bool, imports: &[Dll], delayed: &[Dll], delay_rva: bool) -> Vec<u8> {
        let entry = if wide { 8 } else { 4 };
        let base = if wide { IMAGE_BASE } else { 0x40_0000 };
        // The section: the two directories, then the lookup tables, then the strings.
        let mut data = vec![0u8; (imports.len() + 1) * 20 + (delayed.len() + 1) * 32];
        let rva = |offset: usize| SECTION_RVA + offset as u32;
        let put = |data: &mut Vec<u8>, at: usize, value: u32| {
            data[at..at + 4].copy_from_slice(&value.to_le_bytes());
        };
        let table = |data: &mut Vec<u8>, dll: &Dll| -> (u32, u32) {
            let name = data.len();
            data.extend(dll.name.as_bytes());
            data.push(0);
            let mut hints = Vec::new();
            for import in dll.imports {
                hints.push(import.map(|symbol| {
                    let at = data.len();
                    data.extend([0, 0]);
                    data.extend(symbol.as_bytes());
                    data.push(0);
                    at
                }));
            }
            while !data.len().is_multiple_of(8) {
                data.push(0);
            }
            let lookup = data.len();
            for (ordinal, hint) in hints.iter().enumerate() {
                let value = match hint {
                    Some(at) => u64::from(rva(*at)),
                    None => (1 << (entry * 8 - 1)) | ordinal as u64,
                };
                data.extend(&value.to_le_bytes()[..entry]);
            }
            data.extend(vec![0; entry]);
            (rva(name), rva(lookup))
        };
        for (index, dll) in imports.iter().enumerate() {
            let (name, lookup) = table(&mut data, dll);
            let at = index * 20;
            put(&mut data, at, lookup);
            put(&mut data, at + 12, name);
            put(&mut data, at + 16, lookup);
        }
        let delay_table = (imports.len() + 1) * 20;
        for (index, dll) in delayed.iter().enumerate() {
            let (name, lookup) = table(&mut data, dll);
            let address = |rva: u32| {
                if delay_rva {
                    rva
                } else {
                    (base + u64::from(rva)) as u32
                }
            };
            let at = delay_table + index * 32;
            put(&mut data, at, u32::from(delay_rva));
            put(&mut data, at + 4, address(name));
            put(&mut data, at + 16, address(lookup));
        }

        let mut file = vec![0u8; SECTION_OFFSET];
        file[..2].copy_from_slice(b"MZ");
        let header = 0x40;
        file[0x3c..0x40].copy_from_slice(&(header as u32).to_le_bytes());
        file[header..header + 4].copy_from_slice(b"PE\0\0");
        let coff = header + 4;
        file[coff..coff + 2].copy_from_slice(&(if wide { 0x8664u16 } else { 0x14c }).to_le_bytes());
        file[coff + 2..coff + 4].copy_from_slice(&1u16.to_le_bytes());
        let optional = coff + 20;
        let (count_at, directories) = if wide { (108, 112) } else { (92, 96) };
        let optional_size = directories + 16 * 8;
        file[coff + 16..coff + 18].copy_from_slice(&(optional_size as u16).to_le_bytes());
        let magic: u16 = if wide { 0x20b } else { 0x10b };
        file[optional..optional + 2].copy_from_slice(&magic.to_le_bytes());
        if wide {
            file[optional + 24..optional + 32].copy_from_slice(&base.to_le_bytes());
        } else {
            file[optional + 28..optional + 32].copy_from_slice(&(base as u32).to_le_bytes());
        }
        file[optional + count_at..optional + count_at + 4].copy_from_slice(&16u32.to_le_bytes());
        let mut directory = |index: usize, address: u32, size: u32| {
            let at = optional + directories + index * 8;
            file[at..at + 4].copy_from_slice(&address.to_le_bytes());
            file[at + 4..at + 8].copy_from_slice(&size.to_le_bytes());
        };
        if !imports.is_empty() {
            directory(1, rva(0), ((imports.len() + 1) * 20) as u32);
        }
        if !delayed.is_empty() {
            directory(13, rva(delay_table), ((delayed.len() + 1) * 32) as u32);
        }
        let section = optional + optional_size;
        file[section..section + 6].copy_from_slice(b".idata");
        let size = data.len() as u32;
        file[section + 8..section + 12].copy_from_slice(&size.to_le_bytes());
        file[section + 12..section + 16].copy_from_slice(&SECTION_RVA.to_le_bytes());
        file[section + 16..section + 20].copy_from_slice(&size.to_le_bytes());
        file[section + 20..section + 24].copy_from_slice(&(SECTION_OFFSET as u32).to_le_bytes());
        file.extend(data);
        file
    }
}

#[cfg(test)]
mod tests {
    use super::fixture::{Dll, build};
    use super::*;

    fn kernel32() -> Dll<'static> {
        Dll {
            name: "KERNEL32.dll",
            imports: &[Some("GetLastError"), None, Some("ExitProcess")],
        }
    }

    fn nvcuda() -> Dll<'static> {
        Dll {
            name: "nvcuda.dll",
            imports: &[Some("cuInit")],
        }
    }

    fn vulkan() -> Dll<'static> {
        Dll {
            name: "vulkan-1.dll",
            imports: &[Some("vkCreateInstance")],
        }
    }

    #[test]
    fn rejects_files_that_are_not_pe() {
        assert_eq!(read(b"").unwrap_err(), "not a PE file");
        assert_eq!(read(b"\x7fELF\x02\x01").unwrap_err(), "not a PE file");
        assert!(read(b"MZ").is_err());
        let mut file = build(true, &[kernel32()], &[], true);
        file[0x40] = b'N';
        assert_eq!(
            read(&file).unwrap_err(),
            "the DOS header points at no PE signature"
        );
    }

    #[test]
    fn reads_the_import_directory_of_either_width() {
        for wide in [true, false] {
            let file = build(wide, &[kernel32(), nvcuda()], &[], true);
            assert_eq!(
                read(&file).unwrap(),
                Imports {
                    dlls: vec!["KERNEL32.dll".into(), "nvcuda.dll".into()],
                    delayed: vec![],
                    symbols: vec!["GetLastError".into(), "ExitProcess".into(), "cuInit".into()],
                },
                "wide: {wide}"
            );
        }
    }

    #[test]
    fn reads_the_delay_load_directory_by_rva_and_by_address() {
        // Address-based entries are a PE32 form: their 32-bit fields cannot hold a PE32+ address.
        for (wide, delay_rva) in [(true, true), (false, true), (false, false)] {
            let file = build(wide, &[kernel32()], &[vulkan()], delay_rva);
            assert_eq!(
                read(&file).unwrap(),
                Imports {
                    dlls: vec!["KERNEL32.dll".into()],
                    delayed: vec!["vulkan-1.dll".into()],
                    symbols: vec![
                        "GetLastError".into(),
                        "ExitProcess".into(),
                        "vkCreateInstance".into()
                    ],
                },
                "wide: {wide}, rva: {delay_rva}"
            );
        }
    }

    #[test]
    fn a_binary_with_no_imports_reads_as_empty() {
        assert_eq!(
            read(&build(true, &[], &[], true)).unwrap(),
            Imports::default()
        );
    }

    #[test]
    fn a_truncated_or_corrupted_file_never_panics() {
        let file = build(true, &[kernel32(), nvcuda()], &[vulkan()], true);
        for length in 0..file.len() {
            assert!(read(&file[..length]).is_err(), "{length}");
        }
        for at in 0..file.len() {
            for value in [0x00, 0x7f, 0x80, 0xff] {
                let mut corrupted = file.clone();
                corrupted[at] = value;
                let _ = read(&corrupted);
            }
        }
    }

    /// A crafted file whose import entries all share one long lookup table would take quadratic
    /// time and memory; the entry budget makes it an error, at once.
    #[test]
    fn import_tables_that_all_share_one_table_are_an_error() {
        let many: Vec<String> = (0..1000).map(|i| format!("f{i}")).collect();
        let many: Vec<Option<&str>> = many.iter().map(|name| Some(name.as_str())).collect();
        let names: Vec<String> = (0..300).map(|i| format!("d{i}.dll")).collect();
        let mut dlls = vec![Dll {
            name: "a.dll",
            imports: &many,
        }];
        dlls.extend(names.iter().map(|name| Dll {
            name,
            imports: &[Some("g")],
        }));
        let mut file = build(true, &dlls, &[], true);
        assert_eq!(read(&file).unwrap().symbols.len(), 1300);
        // Point every entry's lookup table at the first DLL's.
        let pe = Pe::new(&file).unwrap();
        let table = pe
            .offset(pe.directory(IMPORT_DIRECTORY).unwrap().unwrap())
            .unwrap() as usize;
        let first: [u8; 4] = file[table..table + 4].try_into().unwrap();
        for index in 1..dlls.len() {
            let at = table + index * IMPORT_ENTRY as usize;
            file[at..at + 4].copy_from_slice(&first);
        }
        let started = std::time::Instant::now();
        let error = read(&file).unwrap_err();
        assert!(error.contains("overlap"), "{error}");
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
    }

    /// The loader reads an import directory by its address alone, so a size of 0 must not hide
    /// the DLLs it names.
    #[test]
    fn an_import_directory_with_a_size_of_0_is_still_read() {
        let mut file = build(true, &[kernel32(), nvcuda()], &[vulkan()], true);
        let directories = Pe::new(&file).unwrap().directories;
        for index in [IMPORT_DIRECTORY, DELAY_IMPORT_DIRECTORY] {
            let size = (directories + u64::from(index) * 8 + 4) as usize;
            file[size..size + 4].copy_from_slice(&[0; 4]);
        }
        let imports = read(&file).unwrap();
        assert_eq!(imports.dlls, ["KERNEL32.dll", "nvcuda.dll"]);
        assert_eq!(imports.delayed, ["vulkan-1.dll"]);
    }

    /// The test binary itself is a PE file on Windows, and imports the Windows kernel.
    #[cfg(windows)]
    #[test]
    fn reads_this_test_binary() {
        let bytes = std::fs::read(std::env::current_exe().unwrap()).unwrap();
        let imports = read(&bytes).unwrap();
        assert!(
            imports
                .dlls
                .iter()
                .any(|dll| dll.eq_ignore_ascii_case("kernel32.dll")),
            "{:?}",
            imports.dlls
        );
        assert!(!imports.symbols.is_empty());
    }
}
