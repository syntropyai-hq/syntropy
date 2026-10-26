// SPDX-License-Identifier: MIT OR Apache-2.0

//! A parser for the `GetLogicalProcessorInformationEx(RelationAll)` buffer, over its bytes.
//!
//! The buffer is a sequence of `SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX` records, each
//! `Size` bytes. The offsets below are the 64-bit layout of `<winnt.h>` (Syntropy's Windows
//! targets are 64-bit, ADR-0020): `Relationship` at 0 and `Size` at 4, then the union at 8.
//!
//! | Relationship | Fields read (offset) |
//! |---|---|
//! | 0 `RelationProcessorCore` | `Flags` (8), `EfficiencyClass` (9), `GroupCount` (30), `GroupMask[]` (32) |
//! | 1 `RelationNumaNode` | `NodeNumber` (8), `GroupCount` (30), `GroupMasks[]` (32) |
//! | 2 `RelationCache` | `Level` (8), `CacheSize` (12), `Type` (16), `GroupCount` (38), `GroupMasks[]` (40) |
//!
//! A `GROUP_AFFINITY` is 16 bytes: an 8-byte `Mask` and a 2-byte `Group`. Before Windows 10
//! 20H2 the NUMA and cache records had one mask and a `GroupCount` of 0, which counts as 1.
//! Other relationships (packages, groups, dies, modules) are skipped.

use std::collections::BTreeSet;

/// A logical processor: (processor group, number within the group).
pub(crate) type Lp = (u16, u8);

/// A physical core.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Core {
    pub efficiency_class: u8,
    pub lps: BTreeSet<Lp>,
}

/// A data or unified cache.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Cache {
    pub level: u8,
    pub size: u64,
    pub lps: BTreeSet<Lp>,
}

/// A NUMA node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Node {
    pub number: u32,
    pub lps: BTreeSet<Lp>,
}

/// The parsed buffer.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Processors {
    pub cores: Vec<Core>,
    pub caches: Vec<Cache>,
    pub nodes: Vec<Node>,
}

fn u8_at(b: &[u8], at: usize) -> Result<u8, String> {
    b.get(at)
        .copied()
        .ok_or_else(|| format!("record ends before offset {at}"))
}

fn u16_at(b: &[u8], at: usize) -> Result<u16, String> {
    let s = b
        .get(at..at + 2)
        .ok_or_else(|| format!("record ends before offset {at}"))?;
    Ok(u16::from_le_bytes([s[0], s[1]]))
}

fn u32_at(b: &[u8], at: usize) -> Result<u32, String> {
    let s = b
        .get(at..at + 4)
        .ok_or_else(|| format!("record ends before offset {at}"))?;
    Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn u64_at(b: &[u8], at: usize) -> Result<u64, String> {
    let s = b
        .get(at..at + 8)
        .ok_or_else(|| format!("record ends before offset {at}"))?;
    let mut a = [0u8; 8];
    a.copy_from_slice(s);
    Ok(u64::from_le_bytes(a))
}

/// The logical processors of `count` group masks starting at `at` (0 counts as 1).
fn masks(rec: &[u8], at: usize, count: u16) -> Result<BTreeSet<Lp>, String> {
    let mut lps = BTreeSet::new();
    for i in 0..usize::from(count.max(1)) {
        let base = at + i * 16;
        let mask = u64_at(rec, base)?;
        let group = u16_at(rec, base + 8)?;
        for bit in 0..64u8 {
            if mask & (1 << bit) != 0 {
                lps.insert((group, bit));
            }
        }
    }
    Ok(lps)
}

/// Parses the buffer.
pub(crate) fn parse(buf: &[u8]) -> Result<Processors, String> {
    let mut out = Processors::default();
    let mut at = 0;
    while at < buf.len() {
        let head = buf.get(at..).unwrap_or_default();
        let rel = u32_at(head, 0)?;
        let size = u32_at(head, 4)? as usize;
        if size < 8 || size > head.len() {
            return Err(format!("record at byte {at} has size {size}"));
        }
        let rec = &head[..size];
        match rel {
            0 => {
                let count = u16_at(rec, 30)?;
                out.cores.push(Core {
                    efficiency_class: u8_at(rec, 9)?,
                    lps: masks(rec, 32, count)?,
                });
            }
            1 => {
                let count = u16_at(rec, 30)?;
                out.nodes.push(Node {
                    number: u32_at(rec, 8)?,
                    lps: masks(rec, 32, count)?,
                });
            }
            2 => {
                // `PROCESSOR_CACHE_TYPE`: 0 unified, 1 instruction, 2 data, 3 trace.
                let ty = u32_at(rec, 16)?;
                if ty == 0 || ty == 2 {
                    let count = u16_at(rec, 38)?;
                    out.caches.push(Cache {
                        level: u8_at(rec, 8)?,
                        size: u64::from(u32_at(rec, 12)?),
                        lps: masks(rec, 40, count)?,
                    });
                }
            }
            _ => {}
        }
        at += size;
    }
    if out.cores.is_empty() {
        return Err("no processor core records".into());
    }
    if out.nodes.is_empty() {
        return Err("no NUMA node records".into());
    }
    Ok(out)
}

#[cfg(test)]
pub(crate) mod build {
    //! Builds buffers in the layout above, for tests.

    use super::Lp;

    fn masks(out: &mut Vec<u8>, lps: &[Lp]) {
        let mut groups = std::collections::BTreeMap::<u16, u64>::new();
        for (g, b) in lps {
            *groups.entry(*g).or_default() |= 1 << b;
        }
        for (g, m) in groups {
            out.extend_from_slice(&m.to_le_bytes());
            out.extend_from_slice(&g.to_le_bytes());
            out.extend_from_slice(&[0; 6]);
        }
    }

    fn groups(lps: &[Lp]) -> u16 {
        let g: std::collections::BTreeSet<u16> = lps.iter().map(|l| l.0).collect();
        g.len() as u16
    }

    fn record(rel: u32, body: Vec<u8>) -> Vec<u8> {
        let mut r = rel.to_le_bytes().to_vec();
        r.extend_from_slice(&((body.len() + 8) as u32).to_le_bytes());
        r.extend(body);
        r
    }

    pub fn core(smt: bool, class: u8, lps: &[Lp]) -> Vec<u8> {
        let mut b = vec![u8::from(smt), class];
        b.extend_from_slice(&[0; 20]);
        b.extend_from_slice(&groups(lps).to_le_bytes());
        masks(&mut b, lps);
        record(0, b)
    }

    pub fn node(number: u32, lps: &[Lp]) -> Vec<u8> {
        let mut b = number.to_le_bytes().to_vec();
        b.extend_from_slice(&[0; 18]);
        b.extend_from_slice(&groups(lps).to_le_bytes());
        masks(&mut b, lps);
        record(1, b)
    }

    pub fn cache(level: u8, ty: u32, size: u32, lps: &[Lp]) -> Vec<u8> {
        let mut b = vec![level, 8];
        b.extend_from_slice(&64u16.to_le_bytes());
        b.extend_from_slice(&size.to_le_bytes());
        b.extend_from_slice(&ty.to_le_bytes());
        b.extend_from_slice(&[0; 18]);
        b.extend_from_slice(&groups(lps).to_le_bytes());
        masks(&mut b, lps);
        record(2, b)
    }

    pub fn package(lps: &[Lp]) -> Vec<u8> {
        let mut b = vec![0, 0];
        b.extend_from_slice(&[0; 20]);
        b.extend_from_slice(&groups(lps).to_le_bytes());
        masks(&mut b, lps);
        record(3, b)
    }
}

#[cfg(test)]
mod tests {
    use super::build::*;
    use super::*;

    #[test]
    fn parses_cores_caches_and_nodes() {
        let mut buf = core(true, 1, &[(0, 0), (0, 1)]);
        buf.extend(core(false, 0, &[(0, 2)]));
        buf.extend(cache(1, 1, 32 * 1024, &[(0, 0), (0, 1)]));
        buf.extend(cache(3, 0, 8 << 20, &[(0, 0), (0, 1), (0, 2)]));
        buf.extend(package(&[(0, 0), (0, 1), (0, 2)]));
        buf.extend(node(0, &[(0, 0), (0, 1), (0, 2)]));
        let p = parse(&buf).unwrap();
        assert_eq!(p.cores.len(), 2);
        assert_eq!(p.cores[0].lps.len(), 2);
        assert_eq!(p.cores[1].efficiency_class, 0);
        // The instruction cache is skipped.
        assert_eq!(p.caches.len(), 1);
        assert_eq!(p.caches[0].size, 8 << 20);
        assert_eq!(p.nodes[0].lps.len(), 3);
    }

    #[test]
    fn multi_group_masks() {
        let lps: Vec<Lp> = (0..64)
            .map(|b| (0, b))
            .chain((0..8).map(|b| (1, b)))
            .collect();
        let mut buf = core(false, 0, &[(1, 7)]);
        buf.extend(node(0, &lps));
        let p = parse(&buf).unwrap();
        assert_eq!(p.nodes[0].lps.len(), 72);
        assert!(p.cores[0].lps.contains(&(1, 7)));
    }

    #[test]
    fn truncated_buffers_are_errors() {
        let mut buf = core(false, 0, &[(0, 0)]);
        buf.extend(node(0, &[(0, 0)]));
        assert!(parse(&buf[..buf.len() - 3]).is_err());
        assert!(parse(&[]).is_err());
        let mut bad = buf.clone();
        bad[4] = 4; // a size below the header
        assert!(parse(&bad).is_err());
        // No NUMA record.
        assert!(parse(&core(false, 0, &[(0, 0)])).is_err());
    }
}
