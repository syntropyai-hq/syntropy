// SPDX-License-Identifier: MIT OR Apache-2.0

//! A minimal Mach-O reader for rule 7: the dynamic libraries a macOS binary loads, from its load
//! commands, and the undefined external symbols of its symbol table, which the dynamic loader
//! binds from those libraries.
//!
//! It reads 32- and 64-bit files of either byte order, and universal ("fat") files, whose
//! architectures it reads in turn. Every offset is bounds-checked, so a truncated or malformed
//! file is an error, never a panic. The work is linear in the file's size: names are deduplicated
//! through sets, and the string bytes copied are held to a budget that a real binary, whose names
//! do not overlap, stays well inside.

use std::cell::Cell;
use std::collections::BTreeSet;

/// What the dynamic loader sees of a binary, over every architecture of a universal file.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Loads {
    /// Each library a load command names, with that command, in file order. A library named by
    /// the same command in two architectures is listed once.
    pub dylibs: Vec<Dylib>,
    /// Every undefined external symbol, without the leading `_` of C names, in file order and
    /// listed once.
    pub symbols: Vec<String>,
    /// Every external symbol the file defines in a section, without the leading `_`: what it
    /// exports, in file order and listed once.
    pub exports: Vec<String>,
}

/// One library and the load command that names it.
#[derive(Debug, PartialEq, Eq)]
pub struct Dylib {
    /// The install name, such as `/usr/lib/libSystem.B.dylib` or `@rpath/libfoo.dylib`.
    pub name: String,
    pub command: Command,
}

/// The load commands that make the dynamic loader load a library.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Command {
    Load,
    Weak,
    Reexport,
    Lazy,
    Upward,
}

impl Command {
    /// The command's name in `<mach-o/loader.h>`.
    pub fn name(self) -> &'static str {
        match self {
            Command::Load => "LC_LOAD_DYLIB",
            Command::Weak => "LC_LOAD_WEAK_DYLIB",
            Command::Reexport => "LC_REEXPORT_DYLIB",
            Command::Lazy => "LC_LAZY_LOAD_DYLIB",
            Command::Upward => "LC_LOAD_UPWARD_DYLIB",
        }
    }

    fn from_raw(cmd: u32) -> Option<Command> {
        match cmd {
            0xc => Some(Command::Load),
            0x8000_0018 => Some(Command::Weak),
            0x8000_001f => Some(Command::Reexport),
            0x20 => Some(Command::Lazy),
            0x8000_0023 => Some(Command::Upward),
            _ => None,
        }
    }
}

const MH_MAGIC: u32 = 0xfeed_face;
const MH_MAGIC_64: u32 = 0xfeed_facf;
const FAT_MAGIC: u32 = 0xcafe_babe;
const FAT_MAGIC_64: u32 = 0xcafe_babf;
const LC_SYMTAB: u32 = 0x2;
/// `nlist` type bits: debugging entries, the type mask, undefined, defined in a section, external.
const N_STAB: u8 = 0xe0;
const N_TYPE: u8 = 0x0e;
const N_UNDF: u8 = 0x0;
const N_SECT: u8 = 0x0e;
const N_EXT: u8 = 0x01;
/// More architectures than any universal file holds; a larger count is a corrupt header.
const MAX_ARCHITECTURES: u32 = 64;

/// Whether `bytes` starts with a Mach-O or universal magic number, of either byte order.
pub fn is_mach_o(bytes: &[u8]) -> bool {
    let Some(raw) = bytes.get(..4).and_then(|raw| <[u8; 4]>::try_from(raw).ok()) else {
        return false;
    };
    let (little, big) = (u32::from_le_bytes(raw), u32::from_be_bytes(raw));
    [MH_MAGIC, MH_MAGIC_64].contains(&little)
        || [MH_MAGIC, MH_MAGIC_64, FAT_MAGIC, FAT_MAGIC_64].contains(&big)
}

/// The string bytes a file may make the reader copy, beyond twice its own size.
const NAME_SLACK: u64 = 1 << 20;

/// What a read has already listed, so each library and symbol is listed once in linear time.
#[derive(Default)]
struct Seen {
    dylibs: BTreeSet<(String, &'static str)>,
    symbols: BTreeSet<String>,
    exports: BTreeSet<String>,
}

/// Reads the load commands and undefined symbols of the Mach-O file `bytes`.
pub fn read(bytes: &[u8]) -> Result<Loads, String> {
    let magic = bytes
        .get(..4)
        .and_then(|raw| raw.try_into().ok())
        .map(u32::from_be_bytes)
        .ok_or("not a Mach-O file")?;
    let mut loads = Loads::default();
    let mut seen = Seen::default();
    // The string bytes the read may still copy.
    let budget = Cell::new(2 * bytes.len() as u64 + NAME_SLACK);
    if magic == FAT_MAGIC || magic == FAT_MAGIC_64 {
        let fat = Reader {
            bytes,
            little: false,
            name_bytes_left: &budget,
        };
        let count = fat.u32(4)?;
        if count > MAX_ARCHITECTURES {
            return Err(format!("a universal file with {count} architectures"));
        }
        for index in 0..u64::from(count) {
            let (offset, size) = if magic == FAT_MAGIC {
                let at = 8 + index * 20;
                (u64::from(fat.u32(at + 8)?), u64::from(fat.u32(at + 12)?))
            } else {
                let at = 8 + index * 32;
                (fat.u64(at + 8)?, fat.u64(at + 16)?)
            };
            let slice = offset
                .checked_add(size)
                .and_then(|end| {
                    bytes.get(usize::try_from(offset).ok()?..usize::try_from(end).ok()?)
                })
                .ok_or_else(|| format!("architecture {index} lies outside the file"))?;
            thin(slice, &mut loads, &mut seen, &budget)
                .map_err(|error| format!("architecture {index}: {error}"))?;
        }
    } else {
        thin(bytes, &mut loads, &mut seen, &budget)?;
    }
    Ok(loads)
}

/// Reads one architecture's file into `loads`.
fn thin(
    bytes: &[u8],
    loads: &mut Loads,
    seen: &mut Seen,
    budget: &Cell<u64>,
) -> Result<(), String> {
    let raw: [u8; 4] = bytes
        .get(..4)
        .and_then(|raw| raw.try_into().ok())
        .ok_or("not a Mach-O file")?;
    let (wide, little) = match (u32::from_le_bytes(raw), u32::from_be_bytes(raw)) {
        (MH_MAGIC, _) => (false, true),
        (MH_MAGIC_64, _) => (true, true),
        (_, MH_MAGIC) => (false, false),
        (_, MH_MAGIC_64) => (true, false),
        _ => return Err("not a Mach-O file".to_owned()),
    };
    let file = Reader {
        bytes,
        little,
        name_bytes_left: budget,
    };
    let count = file.u32(16)?;
    let size = u64::from(file.u32(20)?);
    let start: u64 = if wide { 32 } else { 28 };
    let end = start + size;
    let mut at = start;
    for _ in 0..count {
        if at + 8 > end {
            return Err("the load commands overrun their size".to_owned());
        }
        let cmd = file.u32(at)?;
        let cmdsize = u64::from(file.u32(at + 4)?);
        if cmdsize < 8 || at + cmdsize > end {
            return Err(format!("a load command at offset {at} has size {cmdsize}"));
        }
        if let Some(command) = Command::from_raw(cmd) {
            let offset = u64::from(file.u32(at + 8)?);
            if offset < 12 || offset >= cmdsize {
                return Err(format!("a {} names no library", command.name()));
            }
            let name = file.string(at + offset, at + cmdsize)?;
            if seen.dylibs.insert((name.clone(), command.name())) {
                loads.dylibs.push(Dylib { name, command });
            }
        } else if cmd == LC_SYMTAB {
            symbols(&file, at, wide, loads, seen)?;
        }
        at += cmdsize;
    }
    Ok(())
}

/// Appends the undefined external symbols of the `LC_SYMTAB` command at `at` to `loads.symbols`,
/// and the external symbols it defines in a section to `loads.exports`, each one not yet in
/// `seen`.
fn symbols(
    file: &Reader,
    at: u64,
    wide: bool,
    loads: &mut Loads,
    seen: &mut Seen,
) -> Result<(), String> {
    let table = u64::from(file.u32(at + 8)?);
    let count = u64::from(file.u32(at + 12)?);
    let strings = u64::from(file.u32(at + 16)?);
    let strings_end = strings + u64::from(file.u32(at + 20)?);
    let entry: u64 = if wide { 16 } else { 12 };
    for index in 0..count {
        let symbol = table + index * entry;
        let kind = file.bytes_at::<1>(symbol + 4)?[0];
        let external = kind & N_STAB == 0 && kind & N_EXT != 0;
        let undefined = external && kind & N_TYPE == N_UNDF;
        let defined = external && kind & N_TYPE == N_SECT;
        let name_at = u64::from(file.u32(symbol)?);
        if !(undefined || defined) || name_at == 0 {
            continue;
        }
        let start = strings + name_at;
        if start >= strings_end {
            return Err(format!("symbol {index} names a string outside its table"));
        }
        let name = file.string(start, strings_end)?;
        let name = name.strip_prefix('_').map(str::to_owned).unwrap_or(name);
        if name.is_empty() {
            continue;
        }
        if undefined && seen.symbols.insert(name.clone()) {
            loads.symbols.push(name);
        } else if defined && seen.exports.insert(name.clone()) {
            loads.exports.push(name);
        }
    }
    Ok(())
}

struct Reader<'a> {
    bytes: &'a [u8],
    little: bool,
    /// The string bytes the read may still copy, shared by every architecture.
    name_bytes_left: &'a Cell<u64>,
}

impl Reader<'_> {
    fn bytes_at<const N: usize>(&self, at: u64) -> Result<[u8; N], String> {
        usize::try_from(at)
            .ok()
            .and_then(|at| self.bytes.get(at..at.checked_add(N)?))
            .and_then(|slice| slice.try_into().ok())
            .ok_or_else(|| format!("the file ends before offset {at}"))
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

    /// The string from `start` to the first NUL before `end`, or to `end`.
    fn string(&self, start: u64, end: u64) -> Result<String, String> {
        let range = usize::try_from(start)
            .ok()
            .zip(usize::try_from(end).ok())
            .and_then(|(start, end)| self.bytes.get(start..end))
            .ok_or_else(|| format!("the file ends before offset {end}"))?;
        let length = range
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or(range.len());
        let Some(left) = self.name_bytes_left.get().checked_sub(length as u64 + 1) else {
            return Err("the names overlap more than a linked binary's can".to_owned());
        };
        self.name_bytes_left.set(left);
        String::from_utf8(range[..length].to_vec()).map_err(|_| "a name is not UTF-8".to_owned())
    }
}

/// Builds small Mach-O files for the tests here and in `release`.
#[cfg(test)]
pub mod fixture {
    use super::Command;

    /// A thin executable of the given width and byte order that loads `dylibs` and has the
    /// undefined symbols `undefined` and one defined symbol, `_main`.
    pub fn executable(
        wide: bool,
        little: bool,
        dylibs: &[(&str, Command)],
        undefined: &[&str],
    ) -> Vec<u8> {
        let word = |value: u32| {
            if little {
                value.to_le_bytes()
            } else {
                value.to_be_bytes()
            }
        };
        let raw = |command: Command| match command {
            Command::Load => 0xc,
            Command::Weak => 0x8000_0018,
            Command::Reexport => 0x8000_001f,
            Command::Lazy => 0x20,
            Command::Upward => 0x8000_0023,
        };
        let mut commands = Vec::new();
        for (name, command) in dylibs {
            let mut size = 24 + name.len() + 1;
            size = size.div_ceil(8) * 8;
            commands.extend(word(raw(*command)));
            commands.extend(word(size as u32));
            commands.extend(word(24));
            commands.extend([0; 12]);
            commands.extend(name.as_bytes());
            commands.resize(commands.len() + size - 24 - name.len(), 0);
        }
        let header: usize = if wide { 32 } else { 28 };
        let symtab_at = header + commands.len() + 24;
        let entry = if wide { 16 } else { 12 };
        let names: Vec<String> = undefined
            .iter()
            .map(|name| format!("_{name}"))
            .chain(["_main".to_owned()])
            .collect();
        let strings_at = symtab_at + entry * names.len();
        let mut strings = vec![b' ', 0];
        let mut table = Vec::new();
        for (index, name) in names.iter().enumerate() {
            table.extend(word(strings.len() as u32));
            strings.extend(name.as_bytes());
            strings.push(0);
            // The last name, `_main`, is defined in section 1; the others are undefined.
            let defined = index + 1 == names.len();
            table.push(if defined { 0x0f } else { N_UNDF_EXT });
            table.push(u8::from(defined));
            table.extend([0, 0]);
            table.extend(vec![0; entry - 8]);
        }
        commands.extend(word(LC_SYMTAB_RAW));
        commands.extend(word(24));
        commands.extend(word(symtab_at as u32));
        commands.extend(word(names.len() as u32));
        commands.extend(word(strings_at as u32));
        commands.extend(word(strings.len() as u32));

        let magic: u32 = if wide { 0xfeed_facf } else { 0xfeed_face };
        let mut file = Vec::new();
        file.extend(word(magic));
        file.extend(word(if wide { 0x0100_000c } else { 7 }));
        file.extend(word(0));
        file.extend(word(2));
        file.extend(word(dylibs.len() as u32 + 1));
        file.extend(word(commands.len() as u32));
        file.extend(word(0));
        if wide {
            file.extend(word(0));
        }
        file.extend(commands);
        file.extend(table);
        file.extend(strings);
        file
    }

    const LC_SYMTAB_RAW: u32 = 0x2;
    const N_UNDF_EXT: u8 = 0x01;

    /// A universal file holding `slices`, with 64-bit entries when `wide` is set.
    pub fn universal(wide: bool, slices: &[Vec<u8>]) -> Vec<u8> {
        let entry = if wide { 32 } else { 20 };
        let mut offset = (8 + entry * slices.len()).div_ceil(16) * 16;
        let mut file = Vec::new();
        file.extend((if wide { 0xcafe_babfu32 } else { 0xcafe_babe }).to_be_bytes());
        file.extend((slices.len() as u32).to_be_bytes());
        let mut body: Vec<u8> = Vec::new();
        for slice in slices {
            file.extend([0; 8]);
            if wide {
                file.extend((offset as u64).to_be_bytes());
                file.extend((slice.len() as u64).to_be_bytes());
                file.extend([0; 8]);
            } else {
                file.extend((offset as u32).to_be_bytes());
                file.extend((slice.len() as u32).to_be_bytes());
                file.extend([0; 4]);
            }
            body.extend_from_slice(slice);
            offset += slice.len();
        }
        file.resize(file.len().div_ceil(16) * 16, 0);
        file.extend(body);
        file
    }
}

#[cfg(test)]
mod tests {
    use super::fixture::{executable, universal};
    use super::*;

    const SYSTEM: &str = "/usr/lib/libSystem.B.dylib";
    const METAL: &str = "/System/Library/Frameworks/Metal.framework/Versions/A/Metal";
    const VULKAN: &str = "@rpath/libvulkan.1.dylib";

    fn dylib(name: &str, command: Command) -> Dylib {
        Dylib {
            name: name.to_owned(),
            command,
        }
    }

    #[test]
    fn rejects_files_that_are_not_mach_o() {
        assert_eq!(read(b"").unwrap_err(), "not a Mach-O file");
        assert_eq!(read(b"MZ\x90\x00").unwrap_err(), "not a Mach-O file");
        assert_eq!(read(b"\x7fELF\x02\x01").unwrap_err(), "not a Mach-O file");
        assert!(read(b"\xcf\xfa\xed\xfe").is_err());
        // A Java class file shares the universal magic, and its version reads as a huge count.
        assert_eq!(
            read(b"\xca\xfe\xba\xbe\x00\x00\x00\x41").unwrap_err(),
            "a universal file with 65 architectures"
        );
    }

    #[test]
    fn reads_each_load_command_of_either_width_and_byte_order() {
        let dylibs = [
            (SYSTEM, Command::Load),
            (METAL, Command::Weak),
            (VULKAN, Command::Reexport),
            ("@rpath/liblazy.dylib", Command::Lazy),
            ("@rpath/libupward.dylib", Command::Upward),
        ];
        for (wide, little) in [(true, true), (false, true), (true, false), (false, false)] {
            let file = executable(wide, little, &dylibs, &["malloc", "vkCreateInstance"]);
            assert_eq!(
                read(&file).unwrap(),
                Loads {
                    dylibs: dylibs
                        .iter()
                        .map(|&(name, command)| dylib(name, command))
                        .collect(),
                    symbols: vec!["malloc".into(), "vkCreateInstance".into()],
                    exports: vec!["main".into()],
                },
                "wide: {wide}, little: {little}"
            );
        }
    }

    #[test]
    fn reads_every_architecture_of_a_universal_file() {
        let arm = executable(true, true, &[(SYSTEM, Command::Load)], &["malloc"]);
        let intel = executable(
            true,
            true,
            &[(SYSTEM, Command::Load), (VULKAN, Command::Load)],
            &["malloc", "vkCreateInstance"],
        );
        for wide in [false, true] {
            assert_eq!(
                read(&universal(wide, &[arm.clone(), intel.clone()])).unwrap(),
                Loads {
                    dylibs: vec![dylib(SYSTEM, Command::Load), dylib(VULKAN, Command::Load)],
                    symbols: vec!["malloc".into(), "vkCreateInstance".into()],
                    exports: vec!["main".into()],
                },
                "wide: {wide}"
            );
        }
        let mut nested = universal(false, &[universal(false, &[arm])]);
        assert!(read(&nested).unwrap_err().starts_with("architecture 0: "));
        // An architecture past the end of the file.
        nested.truncate(nested.len() - 1);
        assert_eq!(
            read(&nested).unwrap_err(),
            "architecture 0 lies outside the file"
        );
    }

    /// A crafted file whose symbols all name one long string would copy it once per symbol; the
    /// string budget makes that an error, at once.
    #[test]
    fn symbols_that_all_name_one_long_string_are_an_error() {
        let long = "x".repeat(50_000);
        let short: Vec<String> = (0..2000).map(|i| format!("s{i}")).collect();
        let undefined: Vec<&str> = std::iter::once(long.as_str())
            .chain(short.iter().map(String::as_str))
            .collect();
        let mut file = executable(true, true, &[], &undefined);
        assert_eq!(read(&file).unwrap().symbols.len(), 2001);
        // The symbol table follows the header and the one `LC_SYMTAB` command; the long name's
        // string starts at offset 2 of the string table.
        let symtab = 32 + 24;
        for index in 0..undefined.len() {
            let at = symtab + index * 16;
            file[at..at + 4].copy_from_slice(&2u32.to_le_bytes());
        }
        let started = std::time::Instant::now();
        let error = read(&file).unwrap_err();
        assert!(error.contains("overlap"), "{error}");
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
    }

    #[test]
    fn a_truncated_or_corrupted_file_never_panics() {
        let file = universal(
            false,
            &[executable(
                true,
                true,
                &[(SYSTEM, Command::Load), (VULKAN, Command::Weak)],
                &["vkCreateInstance"],
            )],
        );
        for length in 0..file.len() {
            let _ = read(&file[..length]);
        }
        for at in 0..file.len() {
            for value in [0x00, 0x7f, 0x80, 0xff] {
                let mut corrupted = file.clone();
                corrupted[at] = value;
                let _ = read(&corrupted);
            }
        }
    }

    /// The test binary itself is a Mach-O file on macOS, and loads the system library.
    #[cfg(target_os = "macos")]
    #[test]
    fn reads_this_test_binary() {
        let bytes = std::fs::read(std::env::current_exe().unwrap()).unwrap();
        let loads = read(&bytes).unwrap();
        assert!(
            loads.dylibs.iter().any(|dylib| dylib.name == SYSTEM),
            "{:?}",
            loads.dylibs
        );
        assert!(loads.symbols.iter().any(|symbol| symbol == "malloc"));
    }
}
