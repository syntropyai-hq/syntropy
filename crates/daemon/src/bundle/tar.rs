// SPDX-License-Identifier: MIT OR Apache-2.0

//! A minimal POSIX ustar writer and reader, for the bundle's `tar.zst` (RFC-0053 §6). Regular
//! files only, with names under 100 bytes, mode 0600, owner 0, and one modification time.

/// One file of an archive.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TarFile {
    /// Its path inside the archive.
    pub name: String,
    /// Its bytes.
    pub bytes: Vec<u8>,
}

const BLOCK: usize = 512;

fn octal(out: &mut [u8], value: u64) {
    // `width - 1` octal digits, zero-padded, then NUL.
    let width = out.len();
    let text = format!("{value:0w$o}", w = width - 1);
    out[..width - 1].copy_from_slice(&text.as_bytes()[text.len() - (width - 1)..]);
    out[width - 1] = 0;
}

/// The archive of `files`, each stamped `mtime` seconds after the Unix epoch.
///
/// # Panics
///
/// When a name is 100 bytes or longer, or not ASCII: the bundle's names are fixed by this crate.
#[must_use]
pub fn write(files: &[TarFile], mtime: u64) -> Vec<u8> {
    let mut out = Vec::new();
    for file in files {
        assert!(
            file.name.len() < 100 && file.name.is_ascii(),
            "tar name {}",
            file.name
        );
        let mut header = [0u8; BLOCK];
        header[..file.name.len()].copy_from_slice(file.name.as_bytes());
        octal(&mut header[100..108], 0o600);
        octal(&mut header[108..116], 0);
        octal(&mut header[116..124], 0);
        octal(&mut header[124..136], file.bytes.len() as u64);
        octal(&mut header[136..148], mtime);
        header[156] = b'0';
        header[257..263].copy_from_slice(b"ustar\0");
        header[263..265].copy_from_slice(b"00");
        // The checksum is computed with its own field as spaces.
        header[148..156].fill(b' ');
        let sum: u64 = header.iter().map(|&b| u64::from(b)).sum();
        let text = format!("{sum:06o}");
        header[148..154].copy_from_slice(text.as_bytes());
        header[154] = 0;
        header[155] = b' ';
        out.extend_from_slice(&header);
        out.extend_from_slice(&file.bytes);
        out.resize(out.len().div_ceil(BLOCK) * BLOCK, 0);
    }
    // Two zero blocks end the archive.
    out.resize(out.len() + 2 * BLOCK, 0);
    out
}

/// The regular files of a ustar archive `write` made, in order.
///
/// # Errors
///
/// What is wrong, for an archive that is truncated, has a bad checksum or a non-regular entry.
pub fn read(archive: &[u8]) -> Result<Vec<TarFile>, &'static str> {
    let mut files = Vec::new();
    let mut at = 0;
    loop {
        let header = archive
            .get(at..at + BLOCK)
            .ok_or("the archive ends inside a header")?;
        if header.iter().all(|&b| b == 0) {
            return Ok(files);
        }
        let mut copy = [0u8; BLOCK];
        copy.copy_from_slice(header);
        copy[148..156].fill(b' ');
        let sum: u64 = copy.iter().map(|&b| u64::from(b)).sum();
        if parse_octal(&header[148..156]) != Some(sum) {
            return Err("a header's checksum is wrong");
        }
        if header[156] != b'0' && header[156] != 0 {
            return Err("an entry is not a regular file");
        }
        let name_len = header[..100].iter().position(|&b| b == 0).unwrap_or(100);
        let name = std::str::from_utf8(&header[..name_len])
            .map_err(|_| "a name is not UTF-8")?
            .to_owned();
        let size = usize::try_from(parse_octal(&header[124..136]).ok_or("a size is not octal")?)
            .map_err(|_| "a size is too large")?;
        let start = at + BLOCK;
        let bytes = archive
            .get(start..start + size)
            .ok_or("the archive ends inside a file")?
            .to_vec();
        files.push(TarFile { name, bytes });
        at = start + size.div_ceil(BLOCK) * BLOCK;
    }
}

fn parse_octal(field: &[u8]) -> Option<u64> {
    let text: Vec<u8> = field
        .iter()
        .copied()
        .take_while(|&b| b != 0)
        .filter(|&b| b != b' ')
        .collect();
    if text.is_empty() {
        return None;
    }
    u64::from_str_radix(std::str::from_utf8(&text).ok()?, 8).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_archive_round_trips() {
        let files = vec![
            TarFile {
                name: "MANIFEST.txt".into(),
                bytes: b"hello\n".to_vec(),
            },
            TarFile {
                name: "crashes/a.json".into(),
                bytes: vec![7; 1300],
            },
            TarFile {
                name: "empty".into(),
                bytes: Vec::new(),
            },
        ];
        let archive = write(&files, 1_800_000_000);
        assert_eq!(archive.len() % BLOCK, 0);
        assert_eq!(read(&archive).unwrap(), files);
        // The header layout other tools read: magic, version, type, and an octal size.
        assert_eq!(&archive[257..263], b"ustar\0");
        assert_eq!(&archive[263..265], b"00");
        assert_eq!(archive[156], b'0');
        assert_eq!(&archive[124..136], b"00000000006\0");
        let mut bad = archive.clone();
        bad[0] ^= 1;
        assert_eq!(read(&bad), Err("a header's checksum is wrong"));
        assert!(read(&archive[..600]).is_err());
    }
}
