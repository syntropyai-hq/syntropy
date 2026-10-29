// SPDX-License-Identifier: MIT OR Apache-2.0

//! The dynamic symbol table of the test provider library holds exactly one symbol, the entry
//! point (COMP-E9-T04; RFC-0040 §10.4; ADR-0400 decision 4).
//!
//! - Linux: `nm -D --defined-only` lists one symbol. It is unversioned: rustc's own anonymous
//!   version script, which makes everything but the entry symbol local, cannot be combined with a
//!   named `SYNTROPY_PROVIDER_1` node (see `build.rs`).
//! - Windows: the test reads the PE export directory itself, so it needs no tool from a Visual
//!   Studio prompt. The reader is checked against a synthetic image on every OS.
//! - macOS has no modules, but the same library builds there with an exported-symbols list, and
//!   `nm -gU` lists one symbol.

mod common;

const ENTRY: &str = "syntropy_provider_entry_v1";

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn run(tool: &str, args: &[&str]) -> String {
    let output = std::process::Command::new(tool)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("cannot run `{tool}`: {e}"));
    assert!(
        output.status.success(),
        "`{tool} {}` failed: {}",
        args.join(" "),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// The names `nm` lists, one per line, last field.
fn nm_names(listing: &str) -> Vec<String> {
    listing
        .lines()
        .filter_map(|line| line.split_whitespace().last())
        .map(str::to_owned)
        .collect()
}

#[cfg(target_os = "linux")]
#[test]
fn the_dynamic_symbol_table_holds_exactly_the_entry_symbol() {
    let lib = common::provider_library();
    let path = lib.to_str().expect("a UTF-8 path");
    let names = nm_names(&run("nm", &["-D", "--defined-only", path]));
    assert_eq!(names, [ENTRY], "defined dynamic symbols of {path}");
}

#[cfg(target_os = "macos")]
#[test]
fn the_dynamic_symbol_table_holds_exactly_the_entry_symbol() {
    let lib = common::provider_library();
    let path = lib.to_str().expect("a UTF-8 path");
    // Mach-O prefixes C symbols with an underscore.
    let names = nm_names(&run("nm", &["-gU", path]));
    assert_eq!(names, [format!("_{ENTRY}")], "exported symbols of {path}");
}

#[cfg(windows)]
#[test]
fn the_export_table_holds_exactly_the_entry_symbol() {
    let bytes = std::fs::read(common::provider_library()).expect("read the library");
    assert_eq!(pe_exports(&bytes).expect("a valid PE image"), [ENTRY]);
}

/// The names in a PE image's export directory. An export by ordinal alone counts too: the
/// result lists `<ordinal N>` for each function that has no name.
fn pe_exports(image: &[u8]) -> Result<Vec<String>, String> {
    fn u16_at(b: &[u8], at: usize) -> Result<usize, String> {
        let bytes = b.get(at..at + 2).ok_or("truncated")?;
        Ok(usize::from(u16::from_le_bytes([bytes[0], bytes[1]])))
    }
    fn u32_at(b: &[u8], at: usize) -> Result<usize, String> {
        let bytes = b.get(at..at + 4).ok_or("truncated")?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize)
    }
    let pe = u32_at(image, 0x3C)?;
    if image.get(pe..pe + 4) != Some(b"PE\0\0") {
        return Err("no PE signature".into());
    }
    let sections = u16_at(image, pe + 6)?;
    let optional_size = u16_at(image, pe + 20)?;
    let optional = pe + 24;
    let directories = match u16_at(image, optional)? {
        0x10B => optional + 96,
        0x20B => optional + 112,
        other => return Err(format!("unknown optional header magic {other:#x}")),
    };
    let export_rva = u32_at(image, directories)?;
    let export_size = u32_at(image, directories + 4)?;
    if export_rva == 0 || export_size == 0 {
        return Ok(Vec::new());
    }
    let table = optional + optional_size;
    let to_file = |rva: usize| -> Result<usize, String> {
        for n in 0..sections {
            let at = table + n * 40;
            let (size, virt, raw) = (
                u32_at(image, at + 8)?.max(u32_at(image, at + 16)?),
                u32_at(image, at + 12)?,
                u32_at(image, at + 20)?,
            );
            if (virt..virt + size).contains(&rva) {
                return Ok(raw + (rva - virt));
            }
        }
        Err(format!("RVA {rva:#x} is in no section"))
    };
    let dir = to_file(export_rva)?;
    let functions = u32_at(image, dir + 20)?;
    let names = u32_at(image, dir + 24)?;
    let name_table = to_file(u32_at(image, dir + 32)?)?;
    let mut listed = Vec::new();
    for n in 0..names {
        let at = to_file(u32_at(image, name_table + 4 * n)?)?;
        let end = image
            .get(at..)
            .and_then(|rest| rest.iter().position(|&b| b == 0))
            .ok_or("an unterminated name")?;
        listed.push(String::from_utf8_lossy(&image[at..at + end]).into_owned());
    }
    for ordinal in names..functions {
        listed.push(format!("<ordinal {ordinal}>"));
    }
    Ok(listed)
}

/// A minimal PE32+ image with one section that holds an export directory of the given names.
#[cfg(test)]
fn synthetic_pe(names: &[&str], functions: usize) -> Vec<u8> {
    const SECTION_RVA: usize = 0x1000;
    const SECTION_RAW: usize = 0x200;
    let mut image = vec![0u8; SECTION_RAW];
    image[0x3C..0x40].copy_from_slice(&0x80u32.to_le_bytes());
    image[0x80..0x84].copy_from_slice(b"PE\0\0");
    image[0x80 + 6..0x80 + 8].copy_from_slice(&1u16.to_le_bytes());
    let optional_size = 112 + 16 * 8;
    image[0x80 + 20..0x80 + 22].copy_from_slice(&(optional_size as u16).to_le_bytes());
    let optional = 0x80 + 24;
    image[optional..optional + 2].copy_from_slice(&0x20Bu16.to_le_bytes());
    let section = optional + optional_size;
    assert!(section + 40 <= SECTION_RAW);

    // The section: export directory, then the name pointers, then the names.
    let mut body = vec![0u8; 40];
    let name_ptrs = body.len();
    body.resize(name_ptrs + 4 * names.len(), 0);
    for (n, name) in names.iter().enumerate() {
        let at = SECTION_RVA + body.len();
        body[name_ptrs + 4 * n..name_ptrs + 4 * n + 4].copy_from_slice(&(at as u32).to_le_bytes());
        body.extend_from_slice(name.as_bytes());
        body.push(0);
    }
    body[20..24].copy_from_slice(&(functions as u32).to_le_bytes());
    body[24..28].copy_from_slice(&(names.len() as u32).to_le_bytes());
    body[32..36].copy_from_slice(&((SECTION_RVA + name_ptrs) as u32).to_le_bytes());

    let directories = optional + 112;
    image[directories..directories + 4].copy_from_slice(&(SECTION_RVA as u32).to_le_bytes());
    image[directories + 4..directories + 8].copy_from_slice(&(body.len() as u32).to_le_bytes());
    image[section + 8..section + 12].copy_from_slice(&(body.len() as u32).to_le_bytes());
    image[section + 12..section + 16].copy_from_slice(&(SECTION_RVA as u32).to_le_bytes());
    image[section + 16..section + 20].copy_from_slice(&(body.len() as u32).to_le_bytes());
    image[section + 20..section + 24].copy_from_slice(&(SECTION_RAW as u32).to_le_bytes());
    image.extend_from_slice(&body);
    image
}

#[test]
fn the_pe_reader_lists_named_and_ordinal_only_exports() {
    assert_eq!(pe_exports(&synthetic_pe(&[ENTRY], 1)).unwrap(), [ENTRY]);
    assert_eq!(
        pe_exports(&synthetic_pe(&[ENTRY, "other"], 2)).unwrap(),
        [ENTRY, "other"]
    );
    assert_eq!(
        pe_exports(&synthetic_pe(&[ENTRY], 2)).unwrap(),
        [ENTRY.to_owned(), "<ordinal 1>".to_owned()]
    );
}

#[test]
fn the_pe_reader_refuses_what_is_not_a_pe_image() {
    assert!(pe_exports(&[0u8; 16]).is_err());
    let mut image = synthetic_pe(&[ENTRY], 1);
    image[0x80] = b'X';
    assert!(pe_exports(&image).is_err());
    assert!(pe_exports(&synthetic_pe(&[ENTRY], 1)[..300]).is_err());
}

#[test]
fn nm_names_takes_the_last_field() {
    assert_eq!(
        nm_names("0000000000001130 T syntropy_provider_entry_v1\n"),
        [ENTRY]
    );
}
