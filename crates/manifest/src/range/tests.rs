// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tests of `RangeSource`, the header-read routine, coalescing and recorded-fixture replay
//! (COMP-E10-T01, RFC-0041 §2, §10).
//!
//! `tests/data/tiny-header.bin` is a synthetic GGUF-shaped header made by [`synthetic_header`];
//! `tests/data/tiny-header.hubrec` is the recording of the requests [`parse_like`] makes
//! against it. Neither comes from a real checkpoint or from the network. To regenerate both
//! after changing the generator or the read pattern, run the tests with
//! `SYNTROPY_BLESS=1`.

use std::cell::Cell;
use std::path::PathBuf;

use proptest::prelude::*;

use super::replay::{Recorder, Recording, Replay};
use super::*;

const BIN: &str = "tests/data/tiny-header.bin";
const REC: &str = "tests/data/tiny-header.hubrec";
/// The first request in the fixture reads 1 KiB, so the header takes several requests.
const INITIAL: u64 = 1024;

fn data(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(name)
}

/// A GGUF-shaped header: magic, version 3, a tensor count of 0, then `n` string-keyed `u32`
/// metadata entries, then filler. It is synthetic and licence-clean.
fn synthetic_header() -> Vec<u8> {
    let n: u64 = 120;
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes());
    out.extend_from_slice(&n.to_le_bytes());
    for i in 0..n {
        let key = format!("syntropy.fixture.key_{i:03}");
        out.extend_from_slice(&(key.len() as u64).to_le_bytes());
        out.extend_from_slice(key.as_bytes());
        out.extend_from_slice(&4u32.to_le_bytes()); // GGUF type id of u32
        out.extend_from_slice(&u32::try_from(i * 7919 % 65_521).unwrap().to_le_bytes());
    }
    out.extend((0u32..600).map(|i| u8::try_from(i % 251).unwrap()));
    out
}

/// Reads the header the way a GGUF parser would, through the one header-read routine, and
/// returns every byte it read, in order.
fn parse_like<S: RangeSource>(reader: &mut HeaderReader<S>) -> Result<Vec<u8>, RangeError> {
    let mut seen = Vec::new();
    let mut at = 0u64;
    let take =
        |reader: &mut HeaderReader<S>, at: &mut u64, n: u64| -> Result<Vec<u8>, RangeError> {
            let b = reader.read(*at, n)?.to_vec();
            *at += n;
            Ok(b)
        };
    for n in [4, 4, 8] {
        seen.extend(take(reader, &mut at, n)?);
    }
    let count_bytes = take(reader, &mut at, 8)?;
    let count = u64::from_le_bytes(count_bytes.clone().try_into().unwrap());
    seen.extend(count_bytes);
    for _ in 0..count {
        let len_bytes = take(reader, &mut at, 8)?;
        let len = u64::from_le_bytes(len_bytes.clone().try_into().unwrap());
        seen.extend(len_bytes);
        for n in [len, 4, 4] {
            seen.extend(take(reader, &mut at, n)?);
        }
    }
    let rest = reader.source().len() - at;
    seen.extend(take(reader, &mut at, rest)?);
    Ok(seen)
}

fn record_fixture() -> Recording {
    let file = FileSource::open(data(BIN)).unwrap();
    let recorder = Recorder::new(&file, "synthetic:tiny-header.bin");
    let mut reader =
        HeaderReader::new(&recorder, HEADER_MAX_BYTES_DEFAULT).with_initial_read(INITIAL);
    parse_like(&mut reader).unwrap();
    recorder.recording()
}

#[test]
fn fixtures_are_the_generator_output_and_its_recording() {
    let bless = std::env::var_os("SYNTROPY_BLESS").is_some();
    if bless {
        std::fs::write(data(BIN), synthetic_header()).unwrap();
        let text = format!(
            "# SPDX-License-Identifier: MIT OR Apache-2.0\n# Synthetic: made by range::tests, not recorded from the network. See src/range/replay.rs for the format.\n{}",
            record_fixture().to_text()
        );
        std::fs::write(data(REC), text).unwrap();
    }
    assert_eq!(std::fs::read(data(BIN)).unwrap(), synthetic_header());
    let checked_in = Recording::parse(&std::fs::read_to_string(data(REC)).unwrap()).unwrap();
    assert_eq!(checked_in, record_fixture());
}

#[test]
fn one_header_routine_reads_identical_bytes_from_a_file_and_a_replayed_fixture() {
    let file = FileSource::open(data(BIN)).unwrap();
    let replay = Replay::parse(&std::fs::read_to_string(data(REC)).unwrap()).unwrap();
    assert_eq!(file.len(), replay.len());

    let mut from_file =
        HeaderReader::new(&file, HEADER_MAX_BYTES_DEFAULT).with_initial_read(INITIAL);
    let mut from_hub =
        HeaderReader::new(&replay, HEADER_MAX_BYTES_DEFAULT).with_initial_read(INITIAL);
    let a = parse_like(&mut from_file).unwrap();
    let b = parse_like(&mut from_hub).unwrap();
    assert_eq!(a, b);
    assert_eq!(a, synthetic_header());
    assert_eq!(from_file.requests(), from_hub.requests());

    // Direct reads agree too, at any offset the recording covers.
    for (off, len) in [(0, 4), (17, 300), (4000, 1000), (0, file.len())] {
        let mut x = vec![0; len as usize];
        let mut y = vec![0; len as usize];
        file.read_at(off, &mut x).unwrap();
        replay.read_at(off, &mut y).unwrap();
        assert_eq!(x, y, "{off}+{len}");
    }
}

// ---- The cap ---------------------------------------------------------------------------------

#[test]
fn reads_past_the_cap_fail_with_header_limit() {
    let file = FileSource::open(data(BIN)).unwrap();
    let mut reader = HeaderReader::new(&file, 1000);
    assert!(reader.read(0, 1000).is_ok());
    let err = reader.read(990, 11).unwrap_err();
    assert_eq!(
        err,
        RangeError::HeaderLimit {
            end: 1001,
            cap: 1000
        }
    );
    assert_eq!((err.code(), err.class()), (Some(5), ErrorClass::Integrity));
    assert!(
        err.to_string().starts_with("HeaderLimit (Formats 5)"),
        "{err}"
    );
    // Overflowing the offset is past any cap.
    assert!(matches!(
        reader.read(u64::MAX, 2),
        Err(RangeError::HeaderLimit { .. })
    ));
    // The same parse that succeeds under the default cap fails under a small one.
    let replay = Replay::parse(&std::fs::read_to_string(data(REC)).unwrap()).unwrap();
    let mut capped = HeaderReader::new(&replay, 2048).with_initial_read(INITIAL);
    assert!(matches!(
        parse_like(&mut capped),
        Err(RangeError::HeaderLimit { cap: 2048, .. })
    ));
}

#[test]
fn the_cap_bounds_every_request_not_just_the_read() {
    // The read-ahead never asks the source for bytes past the cap.
    let src = Counting::new(vec![7; 10_000]);
    let mut reader = HeaderReader::new(&src, 3000).with_initial_read(8192);
    reader.read(0, 10).unwrap();
    assert_eq!(src.max_end.get(), 3000);
}

#[test]
fn reads_past_the_source_are_header_invalid() {
    let file = FileSource::open(data(BIN)).unwrap();
    let n = file.len();
    let mut buf = [0u8; 8];
    let err = file.read_at(n - 4, &mut buf).unwrap_err();
    assert_eq!(
        err,
        RangeError::OutOfRange {
            offset: n - 4,
            len: 8,
            source_len: n
        }
    );
    assert_eq!((err.code(), err.class()), (Some(6), ErrorClass::Integrity));
    let mut reader = HeaderReader::new(&file, HEADER_MAX_BYTES_DEFAULT);
    assert!(matches!(
        reader.read(n, 1),
        Err(RangeError::OutOfRange { .. })
    ));
    // An empty read at the end is fine.
    assert_eq!(reader.read(n, 0).unwrap(), &[] as &[u8]);
}

#[test]
fn a_missing_file_is_an_io_error() {
    let err = FileSource::open(data("tests/data/no-such-file")).unwrap_err();
    assert!(
        matches!(
            err,
            RangeError::Io {
                kind: io::ErrorKind::NotFound,
                ..
            }
        ),
        "{err:?}"
    );
    assert_eq!((err.code(), err.class()), (None, ErrorClass::Integrity));
}

// ---- Coalescing ------------------------------------------------------------------------------

/// A source in memory that counts requests and the furthest byte asked for.
struct Counting {
    bytes: Vec<u8>,
    requests: Cell<u32>,
    max_end: Cell<u64>,
}

impl Counting {
    fn new(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            requests: Cell::new(0),
            max_end: Cell::new(0),
        }
    }
}

impl RangeSource for Counting {
    fn len(&self) -> u64 {
        self.bytes.len() as u64
    }
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), RangeError> {
        check_bounds(offset, buf.len() as u64, self.len())?;
        self.requests.set(self.requests.get() + 1);
        self.max_end
            .set(self.max_end.get().max(offset + buf.len() as u64));
        let o = offset as usize;
        buf.copy_from_slice(&self.bytes[o..o + buf.len()]);
        Ok(())
    }
}

#[test]
fn small_header_reads_are_coalesced_into_few_requests() {
    let src = Counting::new(synthetic_header());
    let mut reader = HeaderReader::new(&src, HEADER_MAX_BYTES_DEFAULT).with_initial_read(INITIAL);
    let bytes = parse_like(&mut reader).unwrap();
    assert_eq!(bytes, synthetic_header());
    // Hundreds of parser reads; the prefix doubles from 1 KiB: 1, 2, 4, then the rest.
    assert_eq!(src.requests.get(), 4);
    assert_eq!(reader.requests(), 4);
    // Re-reading inside the prefix issues nothing.
    reader.read(10, 100).unwrap();
    assert_eq!(src.requests.get(), 4);
    // With the default 4 MiB first read, a small header is one request.
    let src = Counting::new(synthetic_header());
    parse_like(&mut HeaderReader::new(&src, HEADER_MAX_BYTES_DEFAULT)).unwrap();
    assert_eq!(src.requests.get(), 1);
}

#[test]
fn a_failed_request_leaves_the_reader_usable() {
    let replay = Replay::new(Recording {
        source: "synthetic:partial".into(),
        length: 100,
        responses: vec![(0, vec![1; 40])],
    });
    let mut reader = HeaderReader::new(&replay, 1000).with_initial_read(40);
    assert_eq!(reader.read(0, 40).unwrap(), &[1; 40][..]);
    assert_eq!(
        reader.read(30, 20),
        Err(RangeError::NotRecorded {
            offset: 40,
            len: 40
        })
    );
    assert_eq!(reader.read(0, 40).unwrap(), &[1; 40][..]);
}

#[test]
fn coalesce_merges_within_gap_and_size() {
    let r = |a: u64, b: u64| a..b;
    assert_eq!(coalesce(&[], 0, 100), vec![]);
    assert_eq!(
        coalesce(&[r(10, 20), r(0, 10), r(5, 5)], 0, 100),
        vec![r(0, 20)]
    );
    assert_eq!(
        coalesce(&[r(0, 10), r(15, 20)], 4, 100),
        vec![r(0, 10), r(15, 20)]
    );
    assert_eq!(coalesce(&[r(0, 10), r(15, 20)], 5, 100), vec![r(0, 20)]);
    assert_eq!(
        coalesce(&[r(0, 60), r(60, 120)], 0, 100),
        vec![r(0, 60), r(60, 120)]
    );
    assert_eq!(coalesce(&[r(0, 500)], 0, 100), vec![r(0, 500)]);
    assert_eq!(coalesce(&[r(0, 50), r(10, 20)], 0, 100), vec![r(0, 50)]);
}

proptest! {
    #[test]
    fn coalesced_requests_cover_exactly_the_requested_bytes(
        ranges in prop::collection::vec((0u64..2000, 0u64..300), 0..20),
        gap in 0u64..50,
        max_len in 1u64..1000,
    ) {
        let ranges: Vec<Range<u64>> = ranges.into_iter().map(|(a, l)| a..a + l).collect();
        let merged = coalesce(&ranges, gap, max_len);
        // Every requested byte is in exactly one request.
        for r in &ranges {
            for b in r.clone() {
                prop_assert_eq!(merged.iter().filter(|m| m.contains(&b)).count(), 1);
            }
        }
        // Requests are sorted, disjoint, and no larger than needed: a merged request only
        // exceeds `max_len` when one input range alone does.
        for w in merged.windows(2) {
            prop_assert!(w[0].end <= w[1].start);
        }
        let longest = ranges.iter().map(|r| r.end - r.start).max().unwrap_or(0);
        for m in &merged {
            prop_assert!(m.end - m.start <= max_len.max(longest));
        }
        // The same input in another order gives the same requests.
        let mut rev = ranges.clone();
        rev.reverse();
        prop_assert_eq!(coalesce(&rev, gap, max_len), merged);
    }
}

// ---- The recording format --------------------------------------------------------------------

#[test]
fn recordings_round_trip_and_reject_malformed_input() {
    let rec = Recording {
        source: "synthetic:x".into(),
        length: 10,
        responses: vec![(0, vec![0xab, 0x01]), (1, vec![0x01, 0xff])],
    };
    assert_eq!(Recording::parse(&rec.to_text()).unwrap(), rec);

    let cases = [
        ("", "empty"),
        ("nope\n", "expected"),
        ("syntropy-range-recording 1\nlength 4\n", "no `source`"),
        (
            "syntropy-range-recording 1\nsource s\nlength 4\nrange 0-1 abc\n",
            "odd",
        ),
        (
            "syntropy-range-recording 1\nsource s\nlength 4\nrange 0-1 ABCD\n",
            "hex",
        ),
        (
            "syntropy-range-recording 1\nsource s\nlength 4\nrange 0-2 abcd\n",
            "body length",
        ),
        (
            "syntropy-range-recording 1\nsource s\nlength 4\nrange 3-4 abcd\n",
            "passes",
        ),
        (
            "syntropy-range-recording 1\nsource s\nrange 0-1 abcd\n",
            "before `length`",
        ),
        (
            "syntropy-range-recording 1\nsource s\nlength 4\nrange 0-1 abcd\nrange 1-2 ce00\n",
            "disagrees",
        ),
        (
            "syntropy-range-recording 1\nsource s\nlength x\n",
            "bad length",
        ),
        (
            "syntropy-range-recording 1\nsource s\nlength 4\nweird 1\n",
            "unexpected",
        ),
    ];
    for (text, want) in cases {
        let err = Recording::parse(text).unwrap_err();
        assert!(err.to_string().contains(want), "{text:?}: {err}");
        assert_eq!(err.class(), ErrorClass::Internal);
    }
}

#[test]
fn replay_serves_reads_spanning_several_responses_and_refuses_gaps() {
    let replay = Replay::new(Recording {
        source: "synthetic:y".into(),
        length: 20,
        responses: vec![
            (0, (0..8).collect()),
            (6, (6..12).collect()),
            (14, (14..20).collect()),
        ],
    });
    let mut buf = [0u8; 12];
    replay.read_at(0, &mut buf).unwrap();
    assert_eq!(buf.to_vec(), (0..12).collect::<Vec<u8>>());
    let mut gap = [0u8; 4];
    assert_eq!(
        replay.read_at(11, &mut gap),
        Err(RangeError::NotRecorded { offset: 11, len: 4 })
    );
    assert!(matches!(
        replay.read_at(18, &mut gap),
        Err(RangeError::OutOfRange { .. })
    ));
}
