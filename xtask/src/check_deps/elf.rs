// SPDX-License-Identifier: MIT OR Apache-2.0

//! A minimal ELF reader for rules 7 and 10: the shared libraries a binary needs (`DT_NEEDED`), the
//! names in its dynamic symbol table (`.dynsym`) and, for rule 10, the names in its full symbol
//! table (`.symtab`).
//!
//! It reads 32- and 64-bit files of either byte order through the section headers, which
//! `strip` keeps. Every offset is bounds-checked, so a truncated or malformed file is an error,
//! never a panic.

/// What the dynamic linker sees of a binary.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Dynamic {
    /// `DT_NEEDED` entries, such as `libc.so.6`, in file order.
    pub needed: Vec<String>,
    /// Every named entry of `.dynsym`, defined or not, in file order.
    pub symbols: Vec<String>,
    /// The named entries of `.dynsym` the file defines (a section index other than `SHN_UNDEF`):
    /// what it exports to the dynamic linker, in file order.
    pub exports: Vec<String>,
}

const SHT_SYMTAB: u32 = 2;
const SHT_DYNAMIC: u32 = 6;
const SHT_DYNSYM: u32 = 11;
const SHN_UNDEF: u16 = 0;
const DT_NULL: u64 = 0;
const DT_NEEDED: u64 = 1;

/// Reads the dynamic section and dynamic symbols of the ELF file `bytes`. A file without them,
/// such as a static binary, gives an empty [`Dynamic`].
pub fn read(bytes: &[u8]) -> Result<Dynamic, String> {
    let elf = Elf::new(bytes)?;
    let mut dynamic = Dynamic::default();
    for index in 0..elf.section_count()? {
        let section = elf.section(index)?;
        match section.kind {
            SHT_DYNAMIC => {
                let strings = elf.section(section.link)?;
                let entry = if elf.wide { 16 } else { 8 };
                for at in
                    (0..section.size / entry).map(|n| section.offset.saturating_add(n * entry))
                {
                    let tag = elf.word(at)?;
                    if tag == DT_NULL {
                        break;
                    }
                    if tag == DT_NEEDED {
                        let value = elf.word(at.saturating_add(entry / 2))?;
                        dynamic.needed.push(elf.string(&strings, value)?);
                    }
                }
            }
            SHT_DYNSYM => {
                let strings = elf.section(section.link)?;
                let entry = if elf.wide { 24 } else { 16 };
                // Entry 0 is the reserved null symbol.
                for at in
                    (1..section.size / entry).map(|n| section.offset.saturating_add(n * entry))
                {
                    let name = elf.string(&strings, u64::from(elf.u32(at)?))?;
                    if !name.is_empty() {
                        // `st_shndx`: after `st_name`, `st_info` and `st_other` in a 64-bit
                        // entry; after `st_value` and `st_size` too in a 32-bit one.
                        let shndx = elf.u16(at.saturating_add(if elf.wide { 6 } else { 14 }))?;
                        if shndx != SHN_UNDEF {
                            dynamic.exports.push(name.clone());
                        }
                        dynamic.symbols.push(name);
                    }
                }
            }
            _ => {}
        }
    }
    Ok(dynamic)
}

/// Every named entry of the full symbol table (`.symtab`), in file order: the binary's own
/// functions and statics, by their mangled names. A stripped file has no `.symtab`, and gives an
/// empty list.
pub fn symtab(bytes: &[u8]) -> Result<Vec<String>, String> {
    let elf = Elf::new(bytes)?;
    let mut names = Vec::new();
    for index in 0..elf.section_count()? {
        let section = elf.section(index)?;
        if section.kind != SHT_SYMTAB {
            continue;
        }
        let strings = elf.section(section.link)?;
        let entry = if elf.wide { 24 } else { 16 };
        // Entry 0 is the reserved null symbol.
        for at in (1..section.size / entry).map(|n| section.offset.saturating_add(n * entry)) {
            let name = elf.string(&strings, u64::from(elf.u32(at)?))?;
            if !name.is_empty() {
                names.push(name);
            }
        }
    }
    Ok(names)
}

struct Elf<'a> {
    bytes: &'a [u8],
    /// 64-bit (`ELFCLASS64`) rather than 32-bit.
    wide: bool,
    little: bool,
}

struct Section {
    kind: u32,
    offset: u64,
    size: u64,
    link: u64,
}

impl<'a> Elf<'a> {
    fn new(bytes: &'a [u8]) -> Result<Elf<'a>, String> {
        if bytes.get(..4) != Some(b"\x7fELF".as_slice()) {
            return Err("not an ELF file".to_owned());
        }
        let wide = match bytes.get(4) {
            Some(1) => false,
            Some(2) => true,
            _ => return Err("unknown ELF class".to_owned()),
        };
        let little = match bytes.get(5) {
            Some(1) => true,
            Some(2) => false,
            _ => return Err("unknown ELF byte order".to_owned()),
        };
        Ok(Elf {
            bytes,
            wide,
            little,
        })
    }

    fn bytes_at<const N: usize>(&self, at: u64) -> Result<[u8; N], String> {
        usize::try_from(at)
            .ok()
            .and_then(|at| self.bytes.get(at..at.checked_add(N)?))
            .and_then(|slice| slice.try_into().ok())
            .ok_or_else(|| format!("the file ends before offset {at}"))
    }

    fn u16(&self, at: u64) -> Result<u16, String> {
        let raw = self.bytes_at(at)?;
        Ok(if self.little {
            u16::from_le_bytes(raw)
        } else {
            u16::from_be_bytes(raw)
        })
    }

    fn u32(&self, at: u64) -> Result<u32, String> {
        let raw = self.bytes_at(at)?;
        Ok(if self.little {
            u32::from_le_bytes(raw)
        } else {
            u32::from_be_bytes(raw)
        })
    }

    fn u64(&self, at: u64) -> Result<u64, String> {
        let raw = self.bytes_at(at)?;
        Ok(if self.little {
            u64::from_le_bytes(raw)
        } else {
            u64::from_be_bytes(raw)
        })
    }

    /// An address-sized field: 8 bytes in a 64-bit file, 4 in a 32-bit one.
    fn word(&self, at: u64) -> Result<u64, String> {
        if self.wide {
            self.u64(at)
        } else {
            self.u32(at).map(u64::from)
        }
    }

    fn section_table(&self) -> Result<(u64, u64), String> {
        let (offset_at, size_at) = if self.wide {
            (0x28, 0x3a)
        } else {
            (0x20, 0x2e)
        };
        Ok((self.word(offset_at)?, u64::from(self.u16(size_at)?)))
    }

    fn section_count(&self) -> Result<u64, String> {
        let count_at = if self.wide { 0x3c } else { 0x30 };
        let count = u64::from(self.u16(count_at)?);
        let (offset, _) = self.section_table()?;
        if count == 0 && offset != 0 {
            return Err("extended section numbering is not supported".to_owned());
        }
        Ok(count)
    }

    fn section(&self, index: u64) -> Result<Section, String> {
        let (table, entry) = self.section_table()?;
        let at = index
            .checked_mul(entry)
            .and_then(|offset| offset.checked_add(table))
            .ok_or("section header offset overflows")?;
        let (offset_at, size_at, link_at) = if self.wide {
            (24, 32, 40)
        } else {
            (16, 20, 24)
        };
        Ok(Section {
            kind: self.u32(at.saturating_add(4))?,
            offset: self.word(at.saturating_add(offset_at))?,
            size: self.word(at.saturating_add(size_at))?,
            link: u64::from(self.u32(at.saturating_add(link_at))?),
        })
    }

    /// The NUL-terminated string at `offset` in the string table `strings`.
    fn string(&self, strings: &Section, offset: u64) -> Result<String, String> {
        if offset >= strings.size {
            return Err(format!("string offset {offset} is outside its table"));
        }
        let start = strings
            .offset
            .checked_add(offset)
            .and_then(|start| usize::try_from(start).ok())
            .ok_or("string offset overflows")?;
        let end = strings
            .offset
            .checked_add(strings.size)
            .and_then(|end| usize::try_from(end).ok())
            .ok_or("string table overflows")?;
        let table = self
            .bytes
            .get(start..end)
            .ok_or("the file ends inside a string table")?;
        let length = table
            .iter()
            .position(|byte| *byte == 0)
            .ok_or("a string has no terminating NUL")?;
        String::from_utf8(table[..length].to_vec()).map_err(|_| "a name is not UTF-8".to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_files_that_are_not_elf() {
        assert_eq!(read(b"").unwrap_err(), "not an ELF file");
        assert_eq!(read(b"MZ\x90\x00").unwrap_err(), "not an ELF file");
        assert_eq!(read(b"\x7fELF\x03\x01").unwrap_err(), "unknown ELF class");
        assert_eq!(
            read(b"\x7fELF\x02\x03").unwrap_err(),
            "unknown ELF byte order"
        );
        assert!(read(b"\x7fELF\x02\x01").is_err());
    }

    /// The test binary itself is a dynamically linked ELF file on Linux with glibc: it needs
    /// libc and imports `malloc` through the system allocator.
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    #[test]
    fn reads_this_test_binary() {
        let bytes = std::fs::read(std::env::current_exe().unwrap()).unwrap();
        let dynamic = read(&bytes).unwrap();
        assert!(
            dynamic.needed.iter().any(|lib| lib.starts_with("libc.so")),
            "{:?}",
            dynamic.needed
        );
        assert!(dynamic.symbols.iter().any(|symbol| symbol == "malloc"));
    }

    /// Cutting a real binary short anywhere gives an error or a partial read, never a panic.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_truncated_binary_never_panics() {
        let bytes = std::fs::read(std::env::current_exe().unwrap()).unwrap();
        let step = (bytes.len() / 512).max(1);
        for length in (0..bytes.len()).step_by(step).chain(0..256) {
            let _ = read(&bytes[..length]);
            let _ = symtab(&bytes[..length]);
        }
    }

    /// The test binary keeps its `.symtab`, which names this very test function, mangled.
    #[cfg(target_os = "linux")]
    #[test]
    fn the_symbol_table_names_this_test() {
        let bytes = std::fs::read(std::env::current_exe().unwrap()).unwrap();
        let names = symtab(&bytes).unwrap();
        assert!(
            names
                .iter()
                .any(|name| name.contains("the_symbol_table_names_this_test")),
            "{} names",
            names.len()
        );
        assert_eq!(symtab(b"MZ").unwrap_err(), "not an ELF file");
    }
}
