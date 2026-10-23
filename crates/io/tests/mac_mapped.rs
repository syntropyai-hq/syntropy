// SPDX-License-Identifier: MIT OR Apache-2.0

//! `MacMapped` against real files (RFC-0021 §5 and Testing; TIER-E5-T09): the engine conformance
//! harness (`syntropy_io::conformance`, TIER-E5-T33) with one worker and with several, the
//! engine's kind and capabilities, and the I/O loop driving it. The order of its mapping calls
//! and the residency check are unit tests in `pool::mapped`, where a shim can record the calls.

#![cfg(target_os = "macos")]
#![deny(clippy::undocumented_unsafe_blocks)]

use std::path::PathBuf;

use syntropy_io::conformance::{self, Options, Report};
use syntropy_io::pool::{MacMapped, POOL_THREADS_MACOS_DEFAULT};
use syntropy_io::{
    Bytes, EngineKind, ExtentRef, IoDest, IoEngine, IoLoop, IoOp, IoPriority, IoTicket, LoopConfig,
    OpenMode, Poll, ReadRequest,
};

const BLOCK: u64 = 16384;

#[repr(C, align(16384))]
#[derive(Clone)]
struct Block([u8; BLOCK as usize]);

/// The conformance harness over a generated 8 MiB file, on an engine with `threads` workers
/// built on the harness's thread.
fn conformance_run(name: &str, threads: usize) -> Report {
    let dir = std::env::temp_dir();
    let pid = std::process::id();
    let file = dir.join(format!("syntropy-io-mapped-conf-{name}-{pid}"));
    let scratch = dir.join(format!("syntropy-io-mapped-conf-{name}-{pid}.scratch"));
    conformance::write_test_file(&file, 8 << 20, 0x7009).unwrap();
    let seed = conformance::seed_from_env();
    let result = conformance::check(
        move || MacMapped::new(threads).unwrap(),
        &file,
        &scratch,
        &Options::new(seed),
    );
    let _ = std::fs::remove_file(&file);
    let _ = std::fs::remove_file(&scratch);
    let report = result.unwrap_or_else(|failure| panic!("{failure}"));
    assert_eq!(report.engine, EngineKind::MacMapped);
    assert!(report.reads > 0 && report.writes > 0, "{report:?}");
    eprintln!(
        "{name}: conformance passed ({:?}, seed {:#x}, {} reads, {} writes, {}-byte unit)",
        report.engine, report.seed, report.reads, report.writes, report.unit
    );
    report
}

/// TIER-E5-T09 criterion 1, with a single worker.
#[test]
fn the_conformance_harness_passes_with_one_worker() {
    conformance_run("one", 1);
}

/// TIER-E5-T09 criterion 1, with the default worker count: reads complete out of order and
/// share pages of the mapping.
#[test]
fn the_conformance_harness_passes_with_the_default_workers() {
    conformance_run("default", POOL_THREADS_MACOS_DEFAULT);
}

#[test]
fn the_engine_reports_its_kind_and_depth() {
    let e = MacMapped::new(3).unwrap();
    assert_eq!(e.kind(), EngineKind::MacMapped);
    let caps = e.caps();
    assert_eq!(caps.max_depth, 3);
    assert!(caps.writes && caps.cancel && !caps.fixed_buffers && !caps.fixed_files);
    assert!(caps.direct);
    assert_eq!(e.buffered_reason(), None);
}

#[test]
fn a_mapped_file_is_not_the_flagged_buffered_mode() {
    let path: PathBuf =
        std::env::temp_dir().join(format!("syntropy-io-mapped-mode-{}", std::process::id()));
    conformance::write_test_file(&path, 4 * BLOCK, 1).unwrap();
    let mut e = MacMapped::new(1).unwrap();
    let file = e.open(&path, OpenMode::Read).unwrap();
    assert!(e.caps().direct, "{:?}", e.buffered_reason());
    assert_eq!(e.file_dio(file).map(|d| d.mem), Some(16384));
    drop(e);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn the_io_loop_drives_the_engine() {
    let path = std::env::temp_dir().join(format!("syntropy-io-mapped-loop-{}", std::process::id()));
    conformance::write_test_file(&path, 1 << 20, 2).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let mut e = MacMapped::new(4).unwrap();
    let file = e.open(&path, OpenMode::Read).unwrap();
    let mut mem = vec![Block([0; BLOCK as usize]); 64];
    let base = mem.as_mut_ptr() as u64;
    let config = LoopConfig {
        chunk: Bytes(4 * BLOCK),
        spin: std::time::Duration::ZERO,
        ..LoopConfig::new(1)
    };
    let (mut io, mut clients) = IoLoop::new(e, &config).unwrap();
    let client = &mut clients[0];
    // Eight requests of eight blocks: each is split into two chunks of four.
    for i in 0..8u64 {
        let offset = (i * 3 % 8) * 8 * BLOCK;
        // SAFETY: a range of `mem`, which outlives the loop, one destination per range.
        let dest = unsafe { IoDest::raw(base + i * 8 * BLOCK, Bytes(8 * BLOCK)) };
        client
            .submit(IoOp::Read(ReadRequest {
                ticket: IoTicket(i),
                extent: ExtentRef {
                    file,
                    offset,
                    len: Bytes(8 * BLOCK),
                },
                dest,
                priority: IoPriority::Demand,
                useful_until: None,
            }))
            .unwrap();
    }
    let mut done = Vec::new();
    for _ in 0..1_000_000 {
        let p = io.poll_once();
        while let Ok(c) = client.poll() {
            done.push(c);
        }
        if done.len() == 8 && p == Poll::Idle {
            break;
        }
    }
    drop(io);
    assert_eq!(done.len(), 8);
    let got: Vec<u8> = mem.iter().flat_map(|b| b.0).collect();
    for c in &done {
        let i = c.ticket.0;
        assert_eq!(c.result, Ok(Bytes(8 * BLOCK)));
        let (at, offset) = ((i * 8 * BLOCK) as usize, ((i * 3 % 8) * 8 * BLOCK) as usize);
        let n = 8 * BLOCK as usize;
        assert!(got[at..at + n] == bytes[offset..offset + n]);
    }
    let _ = std::fs::remove_file(&path);
}
