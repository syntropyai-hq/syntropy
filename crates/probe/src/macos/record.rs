// SPDX-License-Identifier: MIT OR Apache-2.0

//! Recorded macOS API responses: [`MacRecorder`] wraps a [`MacApi`] and records every call
//! with its response, and renders them as fixture text; [`MacFixture`] answers from that
//! text.
//!
//! The fixture format is one call per line; `#` starts a comment.
//!
//! ```text
//! sysctl <name> u32 <n> | u64 <n> | str <text> | hex <bytes> | absent
//! vm page_size=<n> free=<n> active=<n> inactive=<n> wired=<n> purgeable=<n> speculative=<n> compressor=<n>
//! memlock unlimited | memlock <bytes>
//! power providing <text> | power providing none
//! power source type=<text>;state=<text>;current=<n>;max=<n>;present=<bool>
//! ```
//!
//! A sysctl's value is its exact bytes: `u32` and `u64` are native-endian integers of that
//! width, `str` is the text plus its NUL, `hex` anything else. A `power source` field left out
//! is a key the dictionary does not have.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io;

use super::{MacApi, PowerSourceDesc, PowerSources, VmStats};
use crate::host::parse_hex;

/// One recorded call and its response.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MacCall {
    /// `sysctlbyname`.
    Sysctl {
        /// The name.
        name: String,
        /// The new value passed, if any: a write.
        new: Option<Vec<u8>>,
        /// The value returned.
        value: Option<Vec<u8>>,
    },
    /// `host_statistics64`.
    VmStatistics(VmStats),
    /// `getrlimit(RLIMIT_MEMLOCK)`.
    MemlockLimit(Option<u64>),
    /// The IOKit power sources.
    PowerSources(PowerSources),
}

/// A [`MacApi`] that passes each call to `inner` and records it, the recording fake of
/// PLC-E1-T03's wired-limit criterion.
pub struct MacRecorder<A> {
    inner: A,
    calls: RefCell<Vec<MacCall>>,
}

impl<A: MacApi> MacRecorder<A> {
    /// Records the calls made to `inner`.
    pub fn new(inner: A) -> Self {
        Self {
            inner,
            calls: RefCell::new(Vec::new()),
        }
    }

    /// Every successful call so far, in order. A sysctl write is recorded even when it fails.
    pub fn calls(&self) -> Vec<MacCall> {
        self.calls.borrow().clone()
    }

    /// The names of the sysctls a call passed a new value to.
    pub fn sysctl_writes(&self) -> Vec<String> {
        self.calls
            .borrow()
            .iter()
            .filter_map(|c| match c {
                MacCall::Sysctl {
                    name, new: Some(_), ..
                } => Some(name.clone()),
                _ => None,
            })
            .collect()
    }

    /// The names of the sysctls read, in order.
    pub fn sysctl_reads(&self) -> Vec<String> {
        self.calls
            .borrow()
            .iter()
            .filter_map(|c| match c {
                MacCall::Sysctl {
                    name, new: None, ..
                } => Some(name.clone()),
                _ => None,
            })
            .collect()
    }

    /// The recorded responses as fixture text, which [`MacFixture::parse`] reads back.
    pub fn fixture(&self) -> String {
        let mut out = String::new();
        let mut seen = std::collections::BTreeSet::new();
        for c in self.calls.borrow().iter() {
            match c {
                MacCall::Sysctl {
                    name,
                    new: None,
                    value,
                } => {
                    if seen.insert(name.clone()) {
                        let _ = writeln!(out, "sysctl {name} {}", render_value(value.as_deref()));
                    }
                }
                MacCall::Sysctl { new: Some(_), .. } => {}
                MacCall::VmStatistics(v) => {
                    let _ = writeln!(
                        out,
                        "vm page_size={} free={} active={} inactive={} wired={} purgeable={} speculative={} compressor={}",
                        v.page_size,
                        v.free,
                        v.active,
                        v.inactive,
                        v.wired,
                        v.purgeable,
                        v.speculative,
                        v.compressor
                    );
                }
                MacCall::MemlockLimit(m) => match m {
                    Some(b) => {
                        let _ = writeln!(out, "memlock {b}");
                    }
                    None => out.push_str("memlock unlimited\n"),
                },
                MacCall::PowerSources(p) => {
                    let _ = writeln!(
                        out,
                        "power providing {}",
                        p.providing.as_deref().unwrap_or("none")
                    );
                    for s in &p.sources {
                        let mut f = Vec::new();
                        if let Some(v) = &s.kind {
                            f.push(format!("type={v}"));
                        }
                        if let Some(v) = &s.state {
                            f.push(format!("state={v}"));
                        }
                        if let Some(v) = s.current_capacity {
                            f.push(format!("current={v}"));
                        }
                        if let Some(v) = s.max_capacity {
                            f.push(format!("max={v}"));
                        }
                        if let Some(v) = s.present {
                            f.push(format!("present={v}"));
                        }
                        let _ = writeln!(out, "power source {}", f.join(";"));
                    }
                }
            }
        }
        out
    }

    fn push(&self, c: MacCall) {
        self.calls.borrow_mut().push(c);
    }
}

impl<A: MacApi> MacApi for MacRecorder<A> {
    fn sysctl(&self, name: &str, new: Option<&[u8]>) -> io::Result<Option<Vec<u8>>> {
        let r = self.inner.sysctl(name, new);
        if new.is_some() || r.is_ok() {
            self.push(MacCall::Sysctl {
                name: name.into(),
                new: new.map(<[u8]>::to_vec),
                value: r.as_ref().ok().cloned().flatten(),
            });
        }
        r
    }

    fn vm_statistics(&self) -> io::Result<VmStats> {
        let r = self.inner.vm_statistics()?;
        self.push(MacCall::VmStatistics(r));
        Ok(r)
    }

    fn memlock_limit(&self) -> io::Result<Option<u64>> {
        let r = self.inner.memlock_limit()?;
        self.push(MacCall::MemlockLimit(r));
        Ok(r)
    }

    fn power_sources(&self) -> io::Result<PowerSources> {
        let r = self.inner.power_sources()?;
        self.push(MacCall::PowerSources(r.clone()));
        Ok(r)
    }
}

fn render_value(v: Option<&[u8]>) -> String {
    let Some(b) = v else {
        return "absent".into();
    };
    if b.is_empty() {
        return "hex".into();
    }
    if let Some(text) = b.strip_suffix(&[0])
        && !text.contains(&0)
        && let Ok(s) = std::str::from_utf8(text)
        && !s.contains('\n')
        && s.trim() == s
        && !s.is_empty()
    {
        return format!("str {s}");
    }
    match b.len() {
        4 => format!(
            "u32 {}",
            u32::from_ne_bytes(b.try_into().unwrap_or_default())
        ),
        8 => format!(
            "u64 {}",
            u64::from_ne_bytes(b.try_into().unwrap_or_default())
        ),
        _ => {
            let mut s = String::from("hex ");
            for byte in b {
                let _ = write!(s, "{byte:02x}");
            }
            s
        }
    }
}

/// A [`MacApi`] that answers from recorded fixture text. A call the text does not record
/// fails with `NotFound`; a sysctl write fails with `PermissionDenied`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MacFixture {
    sysctls: BTreeMap<String, Option<Vec<u8>>>,
    vm: Option<VmStats>,
    memlock: Option<Option<u64>>,
    power: Option<PowerSources>,
}

impl MacFixture {
    /// Parses fixture text.
    ///
    /// # Errors
    ///
    /// A line that does not parse, with its number.
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut f = Self::default();
        for (i, raw) in text.lines().enumerate() {
            let line = raw.split_once('#').map_or(raw, |(a, _)| a).trim();
            if line.is_empty() {
                continue;
            }
            let bad = |what: &str| format!("line {}: {what}: {raw}", i + 1);
            let (word, rest) = line.split_once(' ').unwrap_or((line, ""));
            match word {
                "sysctl" => {
                    let mut parts = rest.splitn(3, ' ');
                    let name = parts.next().ok_or_else(|| bad("no name"))?;
                    let kind = parts.next().ok_or_else(|| bad("no type"))?;
                    let val = parts.next().unwrap_or("").trim();
                    let bytes = match kind {
                        "absent" => None,
                        "u32" => Some(
                            val.parse::<u32>()
                                .map_err(|_| bad("u32"))?
                                .to_ne_bytes()
                                .to_vec(),
                        ),
                        "u64" => Some(
                            val.parse::<u64>()
                                .map_err(|_| bad("u64"))?
                                .to_ne_bytes()
                                .to_vec(),
                        ),
                        "str" => {
                            let mut b = val.as_bytes().to_vec();
                            b.push(0);
                            Some(b)
                        }
                        "hex" => Some(parse_hex(val).ok_or_else(|| bad("hex"))?),
                        _ => return Err(bad("unknown type")),
                    };
                    f.sysctls.insert(name.into(), bytes);
                }
                "vm" => {
                    let mut v = VmStats::default();
                    for kv in rest.split_whitespace() {
                        let (k, n) = kv.split_once('=').ok_or_else(|| bad("vm field"))?;
                        let n: u64 = n.parse().map_err(|_| bad("vm number"))?;
                        match k {
                            "page_size" => v.page_size = n,
                            "free" => v.free = n,
                            "active" => v.active = n,
                            "inactive" => v.inactive = n,
                            "wired" => v.wired = n,
                            "purgeable" => v.purgeable = n,
                            "speculative" => v.speculative = n,
                            "compressor" => v.compressor = n,
                            _ => return Err(bad("unknown vm field")),
                        }
                    }
                    f.vm = Some(v);
                }
                "memlock" => {
                    f.memlock = Some(match rest {
                        "unlimited" => None,
                        n => Some(n.parse().map_err(|_| bad("memlock"))?),
                    });
                }
                "power" => {
                    let p = f.power.get_or_insert_with(PowerSources::default);
                    let (what, val) = rest.split_once(' ').unwrap_or((rest, ""));
                    match what {
                        "providing" => {
                            p.providing = (val != "none").then(|| val.to_owned());
                        }
                        "source" => {
                            let mut s = PowerSourceDesc::default();
                            for kv in val.split(';').filter(|kv| !kv.is_empty()) {
                                let (k, v) =
                                    kv.split_once('=').ok_or_else(|| bad("source field"))?;
                                match k {
                                    "type" => s.kind = Some(v.into()),
                                    "state" => s.state = Some(v.into()),
                                    "current" => {
                                        s.current_capacity =
                                            Some(v.parse().map_err(|_| bad("current"))?)
                                    }
                                    "max" => {
                                        s.max_capacity = Some(v.parse().map_err(|_| bad("max"))?)
                                    }
                                    "present" => {
                                        s.present = Some(v.parse().map_err(|_| bad("present"))?)
                                    }
                                    _ => return Err(bad("unknown source field")),
                                }
                            }
                            p.sources.push(s);
                        }
                        _ => return Err(bad("unknown power line")),
                    }
                }
                _ => return Err(bad("unknown call")),
            }
        }
        Ok(f)
    }

    /// Replaces or adds a sysctl's recorded value (`None`: the name does not exist).
    pub fn set_sysctl(&mut self, name: &str, value: Option<Vec<u8>>) -> &mut Self {
        self.sysctls.insert(name.into(), value);
        self
    }

    /// Replaces the recorded power sources.
    pub fn set_power(&mut self, power: PowerSources) -> &mut Self {
        self.power = Some(power);
        self
    }

    /// Replaces the recorded page counts.
    pub fn set_vm(&mut self, vm: VmStats) -> &mut Self {
        self.vm = Some(vm);
        self
    }
}

fn missing(what: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::NotFound,
        format!("the fixture records no {what}"),
    )
}

impl MacApi for MacFixture {
    fn sysctl(&self, name: &str, new: Option<&[u8]>) -> io::Result<Option<Vec<u8>>> {
        if new.is_some() {
            return Err(io::Error::from(io::ErrorKind::PermissionDenied));
        }
        // A name the recording never asked for is one the host does not have.
        Ok(self.sysctls.get(name).cloned().flatten())
    }

    fn vm_statistics(&self) -> io::Result<VmStats> {
        self.vm.ok_or_else(|| missing("host_statistics64"))
    }

    fn memlock_limit(&self) -> io::Result<Option<u64>> {
        self.memlock.ok_or_else(|| missing("getrlimit"))
    }

    fn power_sources(&self) -> io::Result<PowerSources> {
        self.power.clone().ok_or_else(|| missing("power sources"))
    }
}
