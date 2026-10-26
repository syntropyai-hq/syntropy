// SPDX-License-Identifier: MIT OR Apache-2.0

//! The memory ranges of the ACPI System Resource Affinity Table (SRAT, ACPI 6.5 §5.2.16),
//! which give each NUMA node's RAM on a multi-node Windows host. Windows reports only each
//! node's free memory (`GetNumaAvailableMemoryNodeEx`).
//!
//! The table is a 36-byte ACPI header, 12 reserved bytes, then structures of `Type` (1 byte)
//! and `Length` (1 byte). A Memory Affinity Structure is type 1, 40 bytes: `Proximity Domain`
//! (u32 at 2), `Base Address` (u64 at 8), `Length` (u64 at 16) and `Flags` (u32 at 28; bit 0
//! enabled, bit 1 hot-pluggable). Ranges that are not enabled, or hot-pluggable (a reserved
//! slot, not installed RAM), are not counted.

use std::collections::BTreeMap;

/// Enabled, non-hot-pluggable bytes per proximity domain.
pub(crate) fn memory_by_domain(table: &[u8]) -> Result<BTreeMap<u32, u64>, String> {
    if table.get(0..4) != Some(b"SRAT") {
        return Err("the table's signature is not SRAT".into());
    }
    let len = table
        .get(4..8)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
        .ok_or("no table length")?;
    let table = table
        .get(..len)
        .ok_or("the table is shorter than its length")?;
    let mut out = BTreeMap::new();
    let mut at = 48;
    while at + 2 <= table.len() {
        let ty = table[at];
        let l = usize::from(table[at + 1]);
        if l < 2 || at + l > table.len() {
            return Err(format!("structure at byte {at} has length {l}"));
        }
        if ty == 1 {
            let s = &table[at..at + l];
            if s.len() < 32 {
                return Err(format!(
                    "memory affinity structure at byte {at} is {l} bytes"
                ));
            }
            let u32_at = |o: usize| u32::from_le_bytes([s[o], s[o + 1], s[o + 2], s[o + 3]]);
            let u64_at = |o: usize| u64::from(u32_at(o)) | (u64::from(u32_at(o + 4)) << 32);
            let flags = u32_at(28);
            if flags & 1 != 0 && flags & 2 == 0 {
                let e = out.entry(u32_at(2)).or_insert(0u64);
                *e = e.saturating_add(u64_at(16));
            }
        }
        at += l;
    }
    Ok(out)
}

#[cfg(test)]
pub(crate) mod build {
    //! Builds SRAT tables for tests.

    /// A table of memory ranges: (proximity domain, length, flags).
    pub fn table(ranges: &[(u32, u64, u32)]) -> Vec<u8> {
        let mut t = b"SRAT".to_vec();
        t.extend_from_slice(&[0; 44]);
        let mut base = 0u64;
        for (pd, len, flags) in ranges {
            let mut s = vec![1u8, 40];
            s.extend_from_slice(&pd.to_le_bytes());
            s.extend_from_slice(&[0; 2]);
            s.extend_from_slice(&base.to_le_bytes());
            s.extend_from_slice(&len.to_le_bytes());
            s.extend_from_slice(&[0; 4]);
            s.extend_from_slice(&flags.to_le_bytes());
            s.extend_from_slice(&[0; 8]);
            t.extend(s);
            base += len;
        }
        // A processor affinity structure, skipped.
        t.extend_from_slice(&[0, 16]);
        t.extend_from_slice(&[0; 14]);
        let n = t.len() as u32;
        t[4..8].copy_from_slice(&n.to_le_bytes());
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sums_enabled_ranges_per_domain() {
        let t = build::table(&[
            (0, 1 << 30, 1),
            (0, 3 << 30, 1),
            (1, 4 << 30, 1),
            (1, 64 << 30, 3), // hot-pluggable: a reserved slot
            (2, 8 << 30, 0),  // disabled
        ]);
        let m = memory_by_domain(&t).unwrap();
        assert_eq!(m.get(&0), Some(&(4 << 30)));
        assert_eq!(m.get(&1), Some(&(4 << 30)));
        assert_eq!(m.get(&2), None);
    }

    #[test]
    fn rejects_bad_tables() {
        assert!(memory_by_domain(b"APIC").is_err());
        let mut t = build::table(&[(0, 1 << 30, 1)]);
        t[49] = 200;
        assert!(memory_by_domain(&t).is_err());
    }
}
