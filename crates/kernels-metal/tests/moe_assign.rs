// SPDX-License-Identifier: MIT OR Apache-2.0

//! The generated MSL version of the split rule, `shaders/moe_assign.metal`, gives the same codes
//! as the Rust version, `syntropy_kernels::moe::assign_into`, on 10⁶ random inputs
//! (COMP-E1-T11; RFC-0040 §3.4; ADR-0404 item 4).
//!
//! The `syntropy-kernels` tests check that the file is what the one specification prints today.
//! This test runs it on the GPU: it builds `moe_assign_harness.swift` with `swiftc`, which
//! compiles the MSL with the Metal compiler of the default device and runs the
//! `syn_moe_assign` kernel once per case, and compares every code with the Rust version's.
//! It skips, saying so, only when the machine has no Metal device; a missing `swiftc` skips it
//! outside CI and fails it in CI.

#![cfg(target_os = "macos")]

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use syntropy_kernels::moe::mirror::{MOE_CPU, MOE_HIT, MOE_MISS};
use syntropy_kernels::moe::{EligibleBits, LayerSplit, RouteSlot, assign_into};

const SEED: u64 = 0x0E10_0011_5EED_0001;
const CASES: u64 = 1_000_000;
/// Cases per command buffer.
const BATCH: u64 = 20_000;

fn crate_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

/// splitmix64, as the `syntropy-kernels` split-rule tests use it.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// One input of the split rule.
#[derive(Debug)]
struct Case {
    route: Vec<RouteSlot>,
    words: Vec<u32>,
    experts: u32,
    split: LayerSplit,
}

impl Case {
    /// The regimes of the `syntropy-kernels` million-input test: no experts, experts outside
    /// the bitmap, repeated experts, token ties (most counts are 0 to 3), hits and misses,
    /// empty and full bitmaps, zero and maximal budgets, zero and maximal expert sizes, no copy
    /// slots and `u32::MAX` of them.
    fn draw(rng: &mut Rng) -> Self {
        let experts = match rng.below(4) {
            0 => 0,
            1 => 1 + rng.below(8),
            2 => 1 + rng.below(300),
            _ => 1 + rng.below(65536),
        } as u32;
        let n = if rng.below(32) == 0 {
            rng.below(300)
        } else {
            rng.below(40)
        } as usize;
        let mut route: Vec<RouteSlot> = Vec::with_capacity(n);
        for i in 0..n {
            let mut expert = if rng.below(8) == 0 || experts == 0 {
                rng.below(65536)
            } else {
                rng.below(u64::from(experts))
            } as u16;
            if i > 0 && rng.below(16) == 0 {
                expert = route[rng.below(i as u64) as usize].expert;
            }
            let tokens = if rng.below(4) == 0 {
                rng.below(65536)
            } else {
                rng.below(4)
            } as u16;
            let cache_slot = if rng.below(2) == 0 {
                MOE_MISS
            } else if rng.below(16) == 0 {
                MOE_MISS - 1
            } else {
                rng.below(1 << 20) as u32
            };
            route.push(RouteSlot {
                expert,
                tokens,
                cache_slot,
            });
        }
        let density = rng.below(4);
        let words = (0..experts.div_ceil(32))
            .map(|_| match density {
                0 => 0,
                1 => u32::MAX,
                2 => rng.next() as u32,
                _ => (rng.next() & rng.next()) as u32,
            })
            .collect();
        let b_copy = match rng.below(4) {
            0 => 0,
            1 => u32::MAX,
            2 => rng.below(1 << 20) as u32,
            _ => rng.next() as u32,
        };
        let expert_bytes = match rng.below(5) {
            0 => 0,
            1 => 1 + rng.below(1 << 16),
            2 => 1 + rng.below(1 << 24),
            3 => rng.next(),
            _ => u64::from(b_copy),
        };
        let copy_slots = match rng.below(4) {
            0 => 0,
            1 => rng.below(8) as u32,
            2 => u32::MAX,
            _ => rng.below(n as u64 + 1) as u32,
        };
        Self {
            route,
            words,
            experts,
            split: LayerSplit {
                b_copy,
                expert_bytes,
                copy_slots,
            },
        }
    }
}

fn put32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

/// One batch in the harness's input format.
fn encode(cases: &[Case]) -> Vec<u8> {
    let entries: usize = cases.iter().map(|c| c.route.len()).sum();
    let words: usize = cases.iter().map(|c| c.words.len()).sum();
    let mut buf = Vec::new();
    put32(&mut buf, cases.len() as u32);
    put32(&mut buf, entries as u32);
    put32(&mut buf, words as u32);
    let (mut route_off, mut words_off) = (0u32, 0u32);
    for c in cases {
        put32(&mut buf, route_off);
        put32(&mut buf, c.route.len() as u32);
        put32(&mut buf, words_off);
        put32(&mut buf, c.experts);
        put32(&mut buf, c.split.b_copy);
        put32(&mut buf, c.split.copy_slots);
        buf.extend_from_slice(&c.split.expert_bytes.to_le_bytes());
        route_off += c.route.len() as u32;
        words_off += c.words.len() as u32;
    }
    for r in cases.iter().flat_map(|c| &c.route) {
        buf.extend_from_slice(&r.expert.to_le_bytes());
        buf.extend_from_slice(&r.tokens.to_le_bytes());
        put32(&mut buf, r.cache_slot);
    }
    for &w in cases.iter().flat_map(|c| &c.words) {
        put32(&mut buf, w);
    }
    buf
}

fn read32(from: &mut impl Read) -> u32 {
    let mut b = [0; 4];
    from.read_exact(&mut b)
        .expect("the harness stopped: see its stderr above");
    u32::from_le_bytes(b)
}

/// Builds the harness with `swiftc`, or `None` when there is none outside CI.
fn build_harness(dir: &Path) -> Option<PathBuf> {
    let swiftc = std::env::var("SWIFTC").unwrap_or_else(|_| "swiftc".to_owned());
    let found = Command::new(&swiftc)
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success());
    if found.is_none() {
        assert!(
            std::env::var_os("CI").is_none(),
            "no swiftc found in CI (set SWIFTC to one)"
        );
        eprintln!("skipped: no swiftc found (set SWIFTC to use one)");
        return None;
    }
    let exe = dir.join("moe_assign_harness");
    let built = Command::new(&swiftc)
        .arg("-O")
        .arg("-module-cache-path")
        .arg(dir.join("module-cache"))
        .arg(crate_path("tests/moe_assign_harness.swift"))
        .arg("-o")
        .arg(&exe)
        .output()
        .unwrap();
    assert!(
        built.status.success(),
        "{swiftc} rejected the harness:\n{}",
        String::from_utf8_lossy(&built.stderr)
    );
    Some(exe)
}

/// The generated MSL version gives the same code as the Rust version for every entry of 10⁶
/// random inputs, run on this machine's Metal device.
#[test]
fn msl_version_matches_rust_on_a_million_inputs() {
    let dir = std::env::temp_dir().join(format!("syntropy-metal-moe-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let Some(exe) = build_harness(&dir) else {
        return;
    };
    let mut child = Command::new(&exe)
        .arg(crate_path("shaders/moe_assign.metal"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = child.stdout.take().unwrap();
    if read32(&mut stdout) == 0 {
        assert!(child.wait().unwrap().success());
        let _ = std::fs::remove_dir_all(&dir);
        eprintln!("skipped: this machine has no Metal device");
        return;
    }
    let mut name = vec![0; read32(&mut stdout) as usize];
    stdout.read_exact(&mut name).unwrap();
    let name = String::from_utf8_lossy(&name).into_owned();

    let mut rng = Rng(SEED);
    let (mut compared, mut entries) = (0u64, 0u64);
    let (mut hits, mut copies, mut cpus) = (0u64, 0u64, 0u64);
    let mut want = Vec::new();
    while compared < CASES {
        let cases: Vec<Case> = (0..BATCH.min(CASES - compared))
            .map(|_| Case::draw(&mut rng))
            .collect();
        stdin.write_all(&encode(&cases)).unwrap();
        stdin.flush().unwrap();
        let n: usize = cases.iter().map(|c| c.route.len()).sum();
        let mut reply = vec![0; n * 4];
        stdout
            .read_exact(&mut reply)
            .expect("the harness stopped: see its stderr above");
        let mut codes = reply
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b));
        for c in &cases {
            want.clear();
            want.resize(c.route.len(), 0);
            assign_into(
                &c.route,
                EligibleBits::new(&c.words, c.experts),
                c.split,
                &mut want,
            )
            .unwrap();
            let got: Vec<u32> = codes.by_ref().take(c.route.len()).collect();
            assert_eq!(
                got, want,
                "case {compared}: MSL (left) and Rust (right) differ on {c:?}"
            );
            for &code in &want {
                match code {
                    MOE_HIT => hits += 1,
                    MOE_CPU => cpus += 1,
                    _ => copies += 1,
                }
            }
            entries += c.route.len() as u64;
            compared += 1;
        }
    }
    stdin.write_all(&[0; 12]).unwrap();
    drop(stdin);
    assert!(child.wait().unwrap().success(), "the harness failed");
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(compared, CASES);
    assert!(
        hits > 0 && copies > 0 && cpus > 0,
        "every code kind is exercised"
    );
    eprintln!(
        "compared {compared} cases, {entries} entries ({hits} hit, {copies} copy, {cpus} cpu) \
         on {name}"
    );
}
