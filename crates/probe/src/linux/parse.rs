// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pure parsers for the procfs and sysfs formats E1 reads. Each returns `Err(what)` with a
//! short description; the caller adds the path.

use std::collections::BTreeSet;

/// A kernel CPU list such as `0-3,8,10-11`. An empty or blank string is the empty list.
pub fn cpulist(text: &str) -> Result<Vec<u32>, String> {
    let text = text.trim();
    let mut cpus = BTreeSet::new();
    if text.is_empty() {
        return Ok(Vec::new());
    }
    for part in text.split(',') {
        let part = part.trim();
        match part.split_once('-') {
            Some((lo, hi)) => {
                let lo = number::<u32>(lo)?;
                let hi = number::<u32>(hi)?;
                if lo > hi {
                    return Err(format!("range {part} runs backwards"));
                }
                cpus.extend(lo..=hi);
            }
            None => {
                cpus.insert(number::<u32>(part)?);
            }
        }
    }
    Ok(cpus.into_iter().collect())
}

/// A decimal number with surrounding whitespace.
pub fn number<T: std::str::FromStr>(text: &str) -> Result<T, String> {
    let t = text.trim();
    t.parse()
        .map_err(|_| format!("{t:?} is not a whole number in range"))
}

/// A sysfs cache size: `48K`, `2048K`, `1M`, or a bare byte count.
pub fn cache_size(text: &str) -> Result<u64, String> {
    let t = text.trim();
    let (digits, scale) = match t.as_bytes().last() {
        Some(b'K') => (&t[..t.len() - 1], 1u64 << 10),
        Some(b'M') => (&t[..t.len() - 1], 1 << 20),
        Some(b'G') => (&t[..t.len() - 1], 1 << 30),
        _ => (t, 1),
    };
    number::<u64>(digits)?
        .checked_mul(scale)
        .ok_or_else(|| format!("{t:?} overflows"))
}

/// One `Key:   value kB` field of `/proc/meminfo`, or of a per-node `meminfo` (whose lines
/// start `Node <n> `), in bytes. `Ok(None)` when the key is absent.
pub fn meminfo_kb(text: &str, key: &str) -> Result<Option<u64>, String> {
    for line in text.lines() {
        let line = match line.strip_prefix("Node ") {
            // "Node 0 MemFree: ..." → "MemFree: ..."
            Some(rest) => rest.split_once(' ').map_or(rest, |(_, r)| r).trim_start(),
            None => line,
        };
        let Some((k, v)) = line.split_once(':') else {
            continue;
        };
        if k.trim() != key {
            continue;
        }
        let v = v.trim();
        let (digits, unit) = v.split_once(char::is_whitespace).unwrap_or((v, ""));
        let n = number::<u64>(digits).map_err(|e| format!("{key}: {e}"))?;
        return match unit.trim() {
            "kB" => n
                .checked_mul(1024)
                .map(Some)
                .ok_or_else(|| format!("{key} overflows")),
            other => Err(format!("{key}: unit {other:?}, expected kB")),
        };
    }
    Ok(None)
}

/// One logical CPU's entry in `/proc/cpuinfo`: its `key: value` fields.
pub type CpuInfoEntry = Vec<(String, String)>;

/// Splits `/proc/cpuinfo` into per-processor entries. Lines without a colon are ignored; a
/// blank line ends an entry. Trailing system-wide fields (aarch64 `Hardware`, `Model`) form
/// their own entry, which has no `processor` key.
pub fn cpuinfo(text: &str) -> Vec<CpuInfoEntry> {
    let mut entries = Vec::new();
    let mut cur: CpuInfoEntry = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            if !cur.is_empty() {
                entries.push(std::mem::take(&mut cur));
            }
            continue;
        }
        if let Some((k, v)) = line.split_once(':') {
            cur.push((k.trim().to_owned(), v.trim().to_owned()));
        }
    }
    if !cur.is_empty() {
        entries.push(cur);
    }
    entries
}

/// The value of `key` in one cpuinfo entry.
pub fn field<'a>(entry: &'a CpuInfoEntry, key: &str) -> Option<&'a str> {
    entry
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
}

/// The soft limit on the `Max locked memory` row of `/proc/<pid>/limits`, in bytes.
/// `Ok(None)` means `unlimited`.
pub fn memlock_soft(text: &str) -> Result<Option<u64>, String> {
    const ROW: &str = "Max locked memory";
    let line = text
        .lines()
        .find(|l| l.starts_with(ROW))
        .ok_or_else(|| format!("no {ROW:?} row"))?;
    let soft = line[ROW.len()..]
        .split_whitespace()
        .next()
        .ok_or_else(|| format!("{ROW:?} row has no soft limit"))?;
    if soft == "unlimited" {
        Ok(None)
    } else {
        number::<u64>(soft).map(Some)
    }
}

/// The effective capability mask, `CapEff:` in `/proc/<pid>/status`.
pub fn cap_eff(status: &str) -> Result<u64, String> {
    let v = status_field(status, "CapEff").ok_or("no CapEff field")?;
    u64::from_str_radix(v, 16).map_err(|_| format!("CapEff {v:?} is not hex"))
}

/// The value of `key` in `/proc/<pid>/status`.
pub fn status_field<'a>(status: &'a str, key: &str) -> Option<&'a str> {
    status.lines().find_map(|l| {
        let (k, v) = l.split_once(':')?;
        (k == key).then(|| v.trim())
    })
}

/// The process's group IDs from `/proc/<pid>/status`: the effective GID and the
/// supplementary groups.
pub fn groups(status: &str) -> Result<Vec<u32>, String> {
    let mut out = Vec::new();
    let gid = status_field(status, "Gid").ok_or("no Gid field")?;
    // Real, effective, saved, filesystem.
    let eff = gid
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| format!("Gid {gid:?} has no effective GID"))?;
    out.push(number::<u32>(eff)?);
    if let Some(list) = status_field(status, "Groups") {
        for g in list.split_whitespace() {
            out.push(number::<u32>(g)?);
        }
    }
    Ok(out)
}

/// One line of `/proc/<pid>/cgroup`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CgroupLine {
    /// Hierarchy ID; 0 is the cgroup v2 unified hierarchy.
    pub hierarchy: u32,
    /// The v1 controllers bound to the hierarchy; empty for v2.
    pub controllers: Vec<String>,
    /// The process's cgroup path in that hierarchy.
    pub path: String,
}

/// Parses `/proc/<pid>/cgroup`: `hierarchy-ID:controller-list:cgroup-path` per line.
pub fn cgroup_lines(text: &str) -> Result<Vec<CgroupLine>, String> {
    let mut out = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let mut parts = line.splitn(3, ':');
        let (Some(h), Some(c), Some(p)) = (parts.next(), parts.next(), parts.next()) else {
            return Err(format!("line {line:?} has fewer than three fields"));
        };
        out.push(CgroupLine {
            hierarchy: number::<u32>(h)?,
            controllers: c
                .split(',')
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .collect(),
            path: p.to_owned(),
        });
    }
    Ok(out)
}

/// A cgroup memory limit file: v2 `memory.max` (`max` = no limit) or v1
/// `memory.limit_in_bytes` (a value at or above 2^62 is the kernel's "no limit", which it
/// rounds down to a page multiple of `i64::MAX`).
pub fn cgroup_limit(text: &str) -> Result<Option<u64>, String> {
    const V1_UNLIMITED: u64 = 1 << 62;
    let t = text.trim();
    if t == "max" {
        return Ok(None);
    }
    let n = number::<u64>(t)?;
    Ok((n < V1_UNLIMITED).then_some(n))
}

/// The selected value in a sysfs choice file such as `always [madvise] never`.
pub fn bracketed(text: &str) -> Option<&str> {
    let start = text.find('[')?;
    let end = text[start..].find(']')? + start;
    Some(&text[start + 1..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpulist_forms() {
        assert_eq!(cpulist("0-3,8,10-11\n").unwrap(), [0, 1, 2, 3, 8, 10, 11]);
        assert_eq!(cpulist("5").unwrap(), [5]);
        assert_eq!(cpulist("\n").unwrap(), Vec::<u32>::new());
        assert!(cpulist("3-1").is_err());
        assert!(cpulist("0-x").is_err());
        assert!(cpulist("1,,2").is_err());
    }

    #[test]
    fn cache_sizes() {
        assert_eq!(cache_size("48K\n").unwrap(), 48 << 10);
        assert_eq!(cache_size("2M").unwrap(), 2 << 20);
        assert_eq!(cache_size("512").unwrap(), 512);
        assert!(cache_size("K").is_err());
        assert!(cache_size("12Q").is_err());
    }

    #[test]
    fn meminfo_fields() {
        let sys = "MemTotal:       32768000 kB\nMemFree:  100 kB\nMemAvailable:   2000 kB\n";
        assert_eq!(
            meminfo_kb(sys, "MemTotal").unwrap(),
            Some(32_768_000 * 1024)
        );
        assert_eq!(meminfo_kb(sys, "MemAvailable").unwrap(), Some(2000 * 1024));
        assert_eq!(meminfo_kb(sys, "SwapTotal").unwrap(), None);
        let node = "Node 1 MemTotal:  10 kB\nNode 1 MemFree:   7 kB\n";
        assert_eq!(meminfo_kb(node, "MemFree").unwrap(), Some(7 * 1024));
        assert!(meminfo_kb("MemTotal: lots kB\n", "MemTotal").is_err());
        assert!(meminfo_kb("MemTotal: 10 MB\n", "MemTotal").is_err());
        assert!(meminfo_kb("MemTotal: 10\n", "MemTotal").is_err());
    }

    #[test]
    fn limits_rows() {
        let t = "Limit                     Soft Limit           Hard Limit           Units     \n\
                 Max locked memory         8388608              8388608              bytes     \n";
        assert_eq!(memlock_soft(t).unwrap(), Some(8_388_608));
        let u = "Max locked memory         unlimited            unlimited            bytes\n";
        assert_eq!(memlock_soft(u).unwrap(), None);
        assert!(memlock_soft("Max open files 1024 1024 files\n").is_err());
        assert!(memlock_soft("Max locked memory  lots  lots bytes\n").is_err());
    }

    #[test]
    fn status_fields() {
        let s = "Name:\tx\nGid:\t1000\t1001\t1000\t1000\nGroups:\t4 27 100 \nCapEff:\t0000000000004000\n";
        assert_eq!(cap_eff(s).unwrap(), 1 << 14);
        assert_eq!(groups(s).unwrap(), [1001, 4, 27, 100]);
        assert!(cap_eff("CapEff:\tzz\n").is_err());
        assert!(cap_eff("Name:\tx\n").is_err());
    }

    #[test]
    fn cgroup_forms() {
        let v = cgroup_lines("12:memory:/user.slice\n0::/user.slice/a.scope\n").unwrap();
        assert_eq!(v[0].controllers, ["memory"]);
        assert_eq!(v[1].hierarchy, 0);
        assert_eq!(v[1].path, "/user.slice/a.scope");
        assert!(cgroup_lines("garbage\n").is_err());
        assert_eq!(cgroup_limit("max\n").unwrap(), None);
        assert_eq!(cgroup_limit("8589934592\n").unwrap(), Some(8 << 30));
        assert_eq!(cgroup_limit("9223372036854771712\n").unwrap(), None);
        assert_eq!(
            cgroup_limit("4611686018427387903").unwrap(),
            Some((1 << 62) - 1)
        );
        assert!(cgroup_limit("-1").is_err());
    }

    #[test]
    fn bracketed_choice() {
        assert_eq!(bracketed("always [madvise] never\n"), Some("madvise"));
        assert_eq!(bracketed("always madvise never"), None);
    }
}
