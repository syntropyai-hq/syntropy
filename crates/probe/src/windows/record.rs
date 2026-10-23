// SPDX-License-Identifier: MIT OR Apache-2.0

//! Recorded Windows API responses: [`WinRecorder`] wraps a [`WinApi`], records each response
//! and renders them as fixture text; [`WinFixture`] answers from that text.
//!
//! The fixture format is one call per line; `#` starts a comment.
//!
//! ```text
//! cpu arch <x86_64|...>       # anything but x86_64 is refused
//! cpu name <text>
//! cpu flags <name> <name> ...
//! glpi <hex>                    # repeated lines concatenate
//! memory total=<n> available=<n>
//! numa-available <node> <bytes>
//! srat absent | srat <hex>      # repeated lines concatenate
//! proximity <domain> <node>
//! power ac=<n> flag=<n> percent=<n>
//! working-set-ceiling <bytes>
//! version <major> <minor> <build>
//! job none | job job_limit=<n|none> process_limit=<n|none> peak_job=<n> peak_process=<n>
//! ```

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::io;

use super::{WinApi, WinCpu, WinJob, WinMemory, WinPower};
use crate::host::parse_hex;

/// The responses seen so far.
#[derive(Clone, Debug, Default)]
struct Seen {
    cpu: Option<WinCpu>,
    glpi: Option<Vec<u8>>,
    memory: Option<WinMemory>,
    numa: BTreeMap<u16, u64>,
    srat: Option<Option<Vec<u8>>>,
    proximity: BTreeMap<u32, u16>,
    power: Option<WinPower>,
    ceiling: Option<u64>,
    version: Option<(u32, u32, u32)>,
    job: Option<Option<WinJob>>,
}

/// A [`WinApi`] that passes each call to `inner` and records the response.
pub struct WinRecorder<A> {
    inner: A,
    seen: RefCell<Seen>,
}

impl<A: WinApi> WinRecorder<A> {
    /// Records the calls made to `inner`.
    pub fn new(inner: A) -> Self {
        Self {
            inner,
            seen: RefCell::new(Seen::default()),
        }
    }

    /// The recorded responses as fixture text, which [`WinFixture::parse`] reads back.
    pub fn fixture(&self) -> String {
        let s = self.seen.borrow();
        let mut out = String::new();
        if let Some(c) = &s.cpu {
            let _ = writeln!(out, "cpu arch {}", c.arch);
            let _ = writeln!(out, "cpu name {}", c.name);
            let flags: Vec<&str> = c.flags.iter().map(String::as_str).collect();
            let _ = writeln!(out, "cpu flags {}", flags.join(" "));
        }
        if let Some(g) = &s.glpi {
            hex_lines(&mut out, "glpi", g);
        }
        if let Some(m) = s.memory {
            let _ = writeln!(out, "memory total={} available={}", m.total, m.available);
        }
        for (n, b) in &s.numa {
            let _ = writeln!(out, "numa-available {n} {b}");
        }
        match &s.srat {
            Some(Some(t)) => hex_lines(&mut out, "srat", t),
            Some(None) => out.push_str("srat absent\n"),
            None => {}
        }
        for (pd, n) in &s.proximity {
            let _ = writeln!(out, "proximity {pd} {n}");
        }
        if let Some(p) = s.power {
            let _ = writeln!(
                out,
                "power ac={} flag={} percent={}",
                p.ac_line, p.battery_flag, p.battery_percent
            );
        }
        if let Some(c) = s.ceiling {
            let _ = writeln!(out, "working-set-ceiling {c}");
        }
        if let Some((a, b, c)) = s.version {
            let _ = writeln!(out, "version {a} {b} {c}");
        }
        match s.job {
            Some(None) => out.push_str("job none\n"),
            Some(Some(j)) => {
                let opt = |v: Option<u64>| v.map_or("none".to_owned(), |n| n.to_string());
                let _ = writeln!(
                    out,
                    "job job_limit={} process_limit={} peak_job={} peak_process={}",
                    opt(j.job_limit),
                    opt(j.process_limit),
                    j.peak_job,
                    j.peak_process
                );
            }
            None => {}
        }
        out
    }
}

fn hex_lines(out: &mut String, key: &str, bytes: &[u8]) {
    for chunk in bytes.chunks(48) {
        let _ = write!(out, "{key} ");
        for b in chunk {
            let _ = write!(out, "{b:02x}");
        }
        out.push('\n');
    }
}

impl<A: WinApi> WinApi for WinRecorder<A> {
    fn processor_information(&self) -> io::Result<Vec<u8>> {
        let r = self.inner.processor_information()?;
        self.seen.borrow_mut().glpi = Some(r.clone());
        Ok(r)
    }

    fn memory_status(&self) -> io::Result<WinMemory> {
        let r = self.inner.memory_status()?;
        self.seen.borrow_mut().memory = Some(r);
        Ok(r)
    }

    fn numa_available(&self, node: u16) -> io::Result<u64> {
        let r = self.inner.numa_available(node)?;
        self.seen.borrow_mut().numa.insert(node, r);
        Ok(r)
    }

    fn srat(&self) -> io::Result<Option<Vec<u8>>> {
        let r = self.inner.srat()?;
        self.seen.borrow_mut().srat = Some(r.clone());
        Ok(r)
    }

    fn proximity_node(&self, proximity: u32) -> Option<u16> {
        let r = self.inner.proximity_node(proximity)?;
        self.seen.borrow_mut().proximity.insert(proximity, r);
        Some(r)
    }

    fn power_status(&self) -> io::Result<WinPower> {
        let r = self.inner.power_status()?;
        self.seen.borrow_mut().power = Some(r);
        Ok(r)
    }

    fn working_set_ceiling(&self, upper: u64) -> io::Result<u64> {
        let r = self.inner.working_set_ceiling(upper)?;
        self.seen.borrow_mut().ceiling = Some(r);
        Ok(r)
    }

    fn version(&self) -> io::Result<(u32, u32, u32)> {
        let r = self.inner.version()?;
        self.seen.borrow_mut().version = Some(r);
        Ok(r)
    }

    fn cpu(&self) -> io::Result<WinCpu> {
        let r = self.inner.cpu()?;
        self.seen.borrow_mut().cpu = Some(r.clone());
        Ok(r)
    }

    fn job_memory(&self) -> io::Result<Option<WinJob>> {
        let r = self.inner.job_memory()?;
        self.seen.borrow_mut().job = Some(r);
        Ok(r)
    }
}

/// A [`WinApi`] that answers from recorded fixture text. A call the text does not record
/// fails with `NotFound`.
#[derive(Clone, Debug, Default)]
pub struct WinFixture(Seen);

impl WinFixture {
    /// Parses fixture text.
    ///
    /// # Errors
    ///
    /// A line that does not parse, with its number.
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut s = Seen::default();
        let mut glpi = String::new();
        let mut srat = String::new();
        for (i, raw) in text.lines().enumerate() {
            let line = raw.split_once('#').map_or(raw, |(a, _)| a).trim();
            if line.is_empty() {
                continue;
            }
            let bad = |what: &str| format!("line {}: {what}: {raw}", i + 1);
            let num = |t: &str| t.parse::<u64>().map_err(|_| bad("number"));
            let (word, rest) = line.split_once(' ').unwrap_or((line, ""));
            let words: Vec<&str> = rest.split_whitespace().collect();
            match word {
                "cpu" => {
                    let c = s.cpu.get_or_insert_with(WinCpu::default);
                    let (what, val) = rest.split_once(' ').unwrap_or((rest, ""));
                    match what {
                        "arch" => c.arch = val.trim().into(),
                        "name" => c.name = val.trim().into(),
                        "flags" => {
                            c.flags = val
                                .split_whitespace()
                                .map(str::to_owned)
                                .collect::<BTreeSet<_>>();
                        }
                        _ => return Err(bad("unknown cpu line")),
                    }
                }
                "glpi" => glpi.push_str(rest),
                "srat" if rest == "absent" => s.srat = Some(None),
                "srat" => srat.push_str(rest),
                "memory" => {
                    let mut m = WinMemory::default();
                    for kv in &words {
                        match kv.split_once('=') {
                            Some(("total", v)) => m.total = num(v)?,
                            Some(("available", v)) => m.available = num(v)?,
                            _ => return Err(bad("memory field")),
                        }
                    }
                    s.memory = Some(m);
                }
                "numa-available" => {
                    let [n, b] = words[..] else {
                        return Err(bad("numa-available <node> <bytes>"));
                    };
                    let n = u16::try_from(num(n)?).map_err(|_| bad("node"))?;
                    s.numa.insert(n, num(b)?);
                }
                "proximity" => {
                    let [pd, n] = words[..] else {
                        return Err(bad("proximity <domain> <node>"));
                    };
                    let pd = u32::try_from(num(pd)?).map_err(|_| bad("domain"))?;
                    let n = u16::try_from(num(n)?).map_err(|_| bad("node"))?;
                    s.proximity.insert(pd, n);
                }
                "power" => {
                    let mut p = WinPower::default();
                    for kv in &words {
                        let (k, v) = kv.split_once('=').ok_or_else(|| bad("power field"))?;
                        let v = u8::try_from(num(v)?).map_err(|_| bad("power value"))?;
                        match k {
                            "ac" => p.ac_line = v,
                            "flag" => p.battery_flag = v,
                            "percent" => p.battery_percent = v,
                            _ => return Err(bad("power field")),
                        }
                    }
                    s.power = Some(p);
                }
                "working-set-ceiling" => s.ceiling = Some(num(rest)?),
                "job" if rest == "none" => s.job = Some(None),
                "job" => {
                    let mut j = WinJob::default();
                    let opt = |v: &str| -> Result<Option<u64>, String> {
                        if v == "none" {
                            Ok(None)
                        } else {
                            num(v).map(Some)
                        }
                    };
                    for kv in &words {
                        match kv.split_once('=') {
                            Some(("job_limit", v)) => j.job_limit = opt(v)?,
                            Some(("process_limit", v)) => j.process_limit = opt(v)?,
                            Some(("peak_job", v)) => j.peak_job = num(v)?,
                            Some(("peak_process", v)) => j.peak_process = num(v)?,
                            _ => return Err(bad("job field")),
                        }
                    }
                    s.job = Some(Some(j));
                }
                "version" => {
                    let [a, b, c] = words[..] else {
                        return Err(bad("version <major> <minor> <build>"));
                    };
                    let v = |t| u32::try_from(num(t)?).map_err(|_| bad("version"));
                    s.version = Some((v(a)?, v(b)?, v(c)?));
                }
                _ => return Err(bad("unknown call")),
            }
        }
        if !glpi.is_empty() {
            s.glpi = Some(parse_hex(&glpi).ok_or("bad glpi hex")?);
        }
        if !srat.is_empty() {
            s.srat = Some(Some(parse_hex(&srat).ok_or("bad srat hex")?));
        }
        Ok(Self(s))
    }

    /// Replaces the recorded power status.
    pub fn set_power(&mut self, p: WinPower) -> &mut Self {
        self.0.power = Some(p);
        self
    }

    /// Replaces the recorded job-object limits (`None`: the process is in no job).
    pub fn set_job(&mut self, j: Option<WinJob>) -> &mut Self {
        self.0.job = Some(j);
        self
    }

    /// Replaces the recorded CPU architecture.
    pub fn set_cpu_arch(&mut self, arch: &str) -> &mut Self {
        if let Some(c) = &mut self.0.cpu {
            c.arch = arch.into();
        }
        self
    }

    /// Replaces the recorded SRAT (`None`: the firmware has none).
    pub fn set_srat(&mut self, t: Option<Vec<u8>>) -> &mut Self {
        self.0.srat = Some(t);
        self
    }

    /// Replaces the recorded CPU flags.
    pub fn set_cpu_flags<'a>(&mut self, flags: impl IntoIterator<Item = &'a str>) -> &mut Self {
        if let Some(c) = &mut self.0.cpu {
            c.flags = flags.into_iter().map(str::to_owned).collect();
        }
        self
    }
}

fn missing(what: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::NotFound,
        format!("the fixture records no {what}"),
    )
}

impl WinApi for WinFixture {
    fn processor_information(&self) -> io::Result<Vec<u8>> {
        self.0.glpi.clone().ok_or_else(|| missing("glpi"))
    }

    fn memory_status(&self) -> io::Result<WinMemory> {
        self.0.memory.ok_or_else(|| missing("memory"))
    }

    fn numa_available(&self, node: u16) -> io::Result<u64> {
        self.0
            .numa
            .get(&node)
            .copied()
            .ok_or_else(|| missing("numa-available"))
    }

    fn srat(&self) -> io::Result<Option<Vec<u8>>> {
        self.0.srat.clone().ok_or_else(|| missing("srat"))
    }

    fn proximity_node(&self, proximity: u32) -> Option<u16> {
        self.0.proximity.get(&proximity).copied()
    }

    fn power_status(&self) -> io::Result<WinPower> {
        self.0.power.ok_or_else(|| missing("power"))
    }

    fn working_set_ceiling(&self, _upper: u64) -> io::Result<u64> {
        self.0.ceiling.ok_or_else(|| missing("working-set-ceiling"))
    }

    fn version(&self) -> io::Result<(u32, u32, u32)> {
        self.0.version.ok_or_else(|| missing("version"))
    }

    fn cpu(&self) -> io::Result<WinCpu> {
        self.0.cpu.clone().ok_or_else(|| missing("cpu"))
    }

    fn job_memory(&self) -> io::Result<Option<WinJob>> {
        self.0.job.ok_or_else(|| missing("job"))
    }
}
