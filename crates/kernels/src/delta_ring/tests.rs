// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tests of the table-delta ring's layout against RFC-0023 §5.4: the header and its generated
//! mirror, which the `syntropy-kernels` tests keep in step.

use std::mem::{offset_of, size_of};
use std::path::{Path, PathBuf};
use std::process::Command;

use proptest::prelude::*;

use super::mirror::{self, DeltaRingHeader, TableDelta};
use super::ring::{DeltaRingLayout, RingError, applied, epoch_capacity, head};
use crate::step::cgen::{self, TABLE_DELTA_RING};

fn crate_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn header() -> cgen::Header {
    let text = std::fs::read_to_string(crate_path(TABLE_DELTA_RING.header)).unwrap();
    cgen::parse_named(TABLE_DELTA_RING.header, &text).unwrap()
}

/// `generated` is what is on disk at `rel`, ignoring CRLF line ends (a Git for Windows
/// checkout). `SYNTROPY_BLESS_GENERATED=1` rewrites it first.
fn assert_current(rel: &str, generated: &str) {
    let path = crate_path(rel);
    if std::env::var_os("SYNTROPY_BLESS_GENERATED").is_some() {
        std::fs::write(&path, generated).unwrap();
    }
    let on_disk = std::fs::read_to_string(&path)
        .unwrap()
        .replace("\r\n", "\n");
    assert!(
        on_disk == generated,
        "{rel} is stale: run `SYNTROPY_BLESS_GENERATED=1 cargo test -p syntropy-kernels \
         delta_ring::`"
    );
}

/// The mirror is what the generator makes of the header today.
#[test]
fn mirror_is_generated_from_the_header() {
    assert_current(
        TABLE_DELTA_RING.mirror,
        &cgen::generate_for(&header(), &TABLE_DELTA_RING),
    );
}

/// The Rust compiler's layout of each mirror equals the C layout the generator computes from
/// the header.
#[test]
fn rust_layout_matches_the_header() {
    let header = header();
    assert_eq!(header.structs.len(), mirror::LAYOUTS.len());
    for (c, rust) in header.structs.iter().zip(mirror::LAYOUTS) {
        let tag = rust.tag;
        assert_eq!(c.c_name, tag);
        assert_eq!(c.size, rust.size, "{tag} size");
        assert_eq!(c.align, rust.align, "{tag} align");
        let c_fields: Vec<(&str, usize)> = c
            .fields
            .iter()
            .map(|f| (f.name.as_str(), f.offset))
            .collect();
        assert_eq!(c_fields, rust.fields.to_vec(), "{tag} members");
    }
}

// ---- The C compiler check --------------------------------------------------------------

/// A C or C++ compiler: `$CC`/`$CXX`, else `cc`/`c++`. In CI (`CI` set) a missing one fails
/// the calling test instead of skipping it.
fn compiler(env: &str, default: &str) -> Option<String> {
    let cmd = std::env::var(env).unwrap_or_else(|_| default.to_owned());
    let found = Command::new(&cmd)
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success());
    if found.is_none() {
        assert!(
            std::env::var_os("CI").is_none(),
            "no {default} compiler found in CI (set {env} to one)"
        );
        eprintln!("skipped: no {default} compiler found (set {env} to use one)");
    }
    found.map(|_| cmd)
}

/// A scratch directory for one test's C files.
fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("syntropy-delta-ring-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Compiles `source` with `-fsyntax-only` as C11 and C++11, with warnings as errors. Returns
/// how many compilers ran.
fn compile_syntax(name: &str, source: &str) -> usize {
    let dir = scratch(name);
    let mut ran = 0;
    for (env, default, file, std) in [
        ("CC", "cc", "check.c", "-std=c11"),
        ("CXX", "c++", "check.cpp", "-std=c++11"),
    ] {
        let Some(cc) = compiler(env, default) else {
            continue;
        };
        let src = dir.join(file);
        std::fs::write(&src, source).unwrap();
        let out = Command::new(&cc)
            .args([std, "-fsyntax-only", "-Wall", "-Wextra", "-Werror", "-I"])
            .arg(crate_path("include"))
            .arg(&src)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{cc} rejected {name}:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        ran += 1;
    }
    let _ = std::fs::remove_dir_all(&dir);
    ran
}

/// `_Static_assert`s of the header's ring macros against [`DeltaRingLayout`] for each capacity.
fn c_macro_check(capacities: &[u32]) -> String {
    let mut out = String::from(
        "#include \"syntropy/table_delta_ring.h\"\n#if defined(__cplusplus)\n\
         #define SYN_ASSERT static_assert\n#else\n\
         #define SYN_ASSERT _Static_assert\n#endif\n",
    );
    for &capacity in capacities {
        let layout = DeltaRingLayout::new(capacity).expect("a power of two");
        out.push_str(&format!(
            "SYN_ASSERT(SYN_DELTA_RING_ENTRIES_OFFSET() == {}u, \"entries offset {capacity}\");\n\
             SYN_ASSERT(SYN_DELTA_RING_SIZE({capacity}u) == {}u, \"size {capacity}\");\n",
            layout.entries_offset(),
            layout.size(),
        ));
    }
    out
}

/// The ring's header layouts, and the entries offset and size the header's macros compute, as a
/// C and a C++ compiler see them.
#[test]
fn c_compiler_agrees_when_present() {
    let layout = cgen::c_layout_check_for(TABLE_DELTA_RING.include, mirror::LAYOUTS);
    let mut ran = compile_syntax("layout", &layout);
    // A capacity of 1, which the layout accepts even though an epoch never gives a ring less
    // than [`mirror::TABLE_DELTA_RING_MIN_CAPACITY`], and a large one [target: RFC-0023 §5.4].
    ran += compile_syntax("macros", &c_macro_check(&[1, 4096, 1 << 20]));
    eprintln!("C layout checks run: {ran}");
}

// ---- The capacity an epoch gives a ring ----------------------------------------------------

/// RFC-0023 §5.4: the next power of two at or above
/// `2 · (sequences · classes · (ceil(T_max / P) + 1) + max_replace · caches)`, minimum 4,096.
#[test]
fn epoch_capacity_follows_the_rfc_0023_bound() {
    // Nothing to carry: the minimum, 4,096 [target: RFC-0023 §5.4's minimum].
    assert_eq!(epoch_capacity(0, 0, 0, 1, 0, 0), Some(4096));
    assert_eq!(epoch_capacity(1, 1, 1, 4, 1, 1), Some(4096));
    // One sequence, one class, a 512-token window in 128-token pages: ceil(512/128) = 4, so the
    // bound is 2 · (1 · 1 · 5 + 0) = 10 [estimate], rounded up to the minimum.
    assert_eq!(epoch_capacity(1, 1, 512, 128, 0, 0), Some(4096));
    // A bound that is exactly a power of two stays: 1 sequence of 1 class with a window of 4095
    // pages of size 1 gives 2 · (4095 + 1) = 8192.
    assert_eq!(epoch_capacity(1, 1, 4095, 1, 0, 0), Some(8192));
    // One entry above that doubles it: 2 · (4096 + 1) = 8194 → 16,384.
    assert_eq!(epoch_capacity(1, 1, 4096, 1, 0, 0), Some(16384));
    // Ceil, not floor: a window of 1 token in pages of 4 is one page.
    assert_eq!(epoch_capacity(1, 1, 1, 4, 0, 0), Some(4096));
    assert_eq!(epoch_capacity(2, 2, 1, 4, 3, 4), Some(4096));
    // DeepSeek-V3-class numbers [estimate]: 64 live sequences, 3 page classes, a 4,096-token
    // window in 128-token pages, 256 replacements in 4 caches: the bound is 14,720.
    assert_eq!(epoch_capacity(64, 3, 4096, 128, 256, 4), Some(16384));
    // A page size of 0 has no bound, and a bound past a `u32` is not a ring capacity.
    assert_eq!(epoch_capacity(1, 1, 1, 0, 0, 0), None);
    assert_eq!(epoch_capacity(0, 0, 0, 0, 0, 0), None);
    assert_eq!(epoch_capacity(u32::MAX, 1, 0, 1, 0, 0), None);
    assert_eq!(epoch_capacity(u32::MAX, u32::MAX, u32::MAX, 1, 0, 0), None);
    // The largest bound that still fits is exactly 2^31; one more entry needs 2^32, which a
    // `u32` capacity cannot hold.
    assert_eq!(epoch_capacity(1, 1, (1 << 30) - 1, 1, 0, 0), Some(1 << 31));
    assert_eq!(epoch_capacity(1, 1, (1 << 30) - 1, 1, 1, 1), None);
    assert_eq!(epoch_capacity(1 << 30, 1, 0, 1, 0, 0), Some(1 << 31));
}

/// Every capacity [`epoch_capacity`] returns is a power of two that the layout accepts, at least
/// the header's minimum, and at least the RFC's bound.
#[test]
fn an_epoch_capacity_is_a_power_of_two_the_layout_accepts() {
    for (s, c, t, p, r, k) in [
        (0u32, 0, 0, 1, 0, 0),
        (1, 1, 1, 1, 0, 0),
        (8, 2, 1024, 256, 64, 2),
        (1024, 8, 8192, 16, 1024, 8),
        (u32::MAX, 1, 1, 1, 1, 1),
        (1, u32::MAX, 1, 1, 1, u32::MAX),
    ] {
        // The bound itself, in `u64`: entries per sequence, plus the replacements.
        let pages = (u64::from(t) + u64::from(p) - 1) / u64::from(p);
        let bound = (u64::from(s) * u64::from(c) * (pages + 1) + u64::from(r) * u64::from(k))
            .checked_mul(2);
        match epoch_capacity(s, c, t, p, r, k) {
            Some(cap) => {
                assert!(cap.is_power_of_two(), "{cap} is not a power of two");
                assert!(
                    cap >= mirror::TABLE_DELTA_RING_MIN_CAPACITY,
                    "{cap} is below the minimum"
                );
                assert!(DeltaRingLayout::new(cap).is_ok());
                let bound = bound.expect("a capacity needs a bound that fits a `u64`");
                assert!(u64::from(cap) >= bound, "{cap} is under the bound {bound}");
            }
            // No capacity: the bound overflows a `u64`, or is above 2^31, so its next power of
            // two is past a `u32`.
            None => assert!(bound.is_none() || bound.unwrap() > u64::from(1u32 << 31)),
        }
    }
}

// ---- The layout, posting and reading -------------------------------------------------------

fn delta(table: u16, index: u32, value: u32) -> TableDelta {
    TableDelta {
        table,
        _pad: 0,
        index,
        value,
    }
}

#[test]
fn layout_parts_follow_the_header() {
    let layout = DeltaRingLayout::new(mirror::TABLE_DELTA_RING_MIN_CAPACITY).unwrap();
    assert_eq!(layout.capacity(), mirror::TABLE_DELTA_RING_MIN_CAPACITY);
    assert_eq!(layout.entries_offset(), size_of::<DeltaRingHeader>());
    assert_eq!(
        layout.size(),
        layout.entries_offset()
            + (mirror::TABLE_DELTA_RING_MIN_CAPACITY as usize) * size_of::<TableDelta>()
    );
}

#[test]
fn a_capacity_must_be_a_power_of_two() {
    for bad in [0, 6, 4095, (1 << 31) + 1, u32::MAX] {
        assert_eq!(DeltaRingLayout::new(bad), Err(RingError::BadCapacity(bad)));
    }
    for good in [1, 2, 4, 4096, 1 << 24] {
        assert!(DeltaRingLayout::new(good).is_ok());
    }
}

#[test]
fn a_buffer_shorter_than_the_layout_is_rejected() {
    let layout = DeltaRingLayout::new(4096).unwrap();
    let need = layout.size();
    let mut buf = vec![0u8; need - 1];
    let small = RingError::BufferTooSmall {
        need,
        have: need - 1,
    };
    assert_eq!(layout.init(&mut buf), Err(small));
    assert_eq!(layout.post(&mut buf, &[]), Err(small));
    assert_eq!(layout.read(&buf, 0, 0), Err(small));
    assert_eq!(layout.echo(&mut buf, 1, 1), Err(small));
    let short_header = RingError::BufferTooSmall {
        need: size_of::<DeltaRingHeader>(),
        have: size_of::<DeltaRingHeader>() - 1,
    };
    let header = &buf[..size_of::<DeltaRingHeader>() - 1];
    assert_eq!(head(header), Err(short_header));
    assert_eq!(applied(header), Err(short_header));
    assert_eq!(buf, vec![0u8; need - 1]);
}

#[test]
fn init_writes_the_header_and_touches_no_entry() {
    let layout = DeltaRingLayout::new(8).unwrap();
    let mut buf = vec![0xa5u8; layout.size()];
    layout.init(&mut buf).unwrap();
    assert_eq!(head(&buf), Ok(0));
    assert_eq!(applied(&buf), Ok((0, 0)));
    let at = offset_of!(DeltaRingHeader, capacity);
    assert_eq!(u32::from_le_bytes(buf[at..at + 4].try_into().unwrap()), 8);
    assert_eq!(
        &buf[offset_of!(DeltaRingHeader, _pad)..layout.entries_offset()],
        &[0; 4]
    );
    assert!(buf[layout.entries_offset()..].iter().all(|&b| b == 0xa5));
}

#[test]
fn post_writes_entries_before_moving_the_head_and_wraps() {
    let layout = DeltaRingLayout::new(4).unwrap();
    let mut buf = vec![0u8; layout.size()];
    layout.init(&mut buf).unwrap();
    let first = [delta(0, 1, 9), delta(0, 2, 10)];
    assert_eq!(layout.post(&mut buf, &first), Ok(2));
    assert_eq!(head(&buf), Ok(2));
    assert_eq!(layout.read(&buf, 0, 2), Ok(first.to_vec()));

    // Five pending deltas in a ring of four: refused, bytes untouched, head unmoved.
    let second = [delta(2, 0, 5), delta(1, 3, 6), delta(0, 0, 7)];
    let snapshot = buf.clone();
    assert_eq!(
        layout.post(&mut buf, &second),
        Err(RingError::Full {
            head: 2,
            applied: 0,
            capacity: 4,
            posting: 3,
        })
    );
    assert_eq!(buf, snapshot);
    assert_eq!(head(&buf), Ok(2));

    // The echo of the step that applied the first batch frees the ring; posting wraps slot 0.
    layout.echo(&mut buf, 2, 7).unwrap();
    assert_eq!(applied(&buf), Ok((2, 7)));
    assert_eq!(layout.post(&mut buf, &second), Ok(5));
    assert_eq!(layout.read(&buf, 2, 5), Ok(second.to_vec()));
    // After the wrap the last `capacity` positions are exactly what the ring holds.
    assert_eq!(
        layout.read(&buf, 1, 5),
        Ok(vec![first[1], second[0], second[1], second[2]])
    );
    // Position 4 lands in slot 0, so the entry the ring held at position 0 is gone: reading it
    // is a range the ring does not hold.
    assert_eq!(
        layout.read(&buf, 0, 1),
        Err(RingError::BadRange {
            from: 0,
            to: 1,
            head: 5,
            capacity: 4,
        })
    );
    assert_eq!(head(&buf), Ok(5));
    assert_eq!(applied(&buf), Ok((2, 7)));
}

#[test]
fn a_full_ring_refuses_to_post_and_is_unchanged() {
    let layout = DeltaRingLayout::new(4).unwrap();
    let mut buf = vec![0xa7u8; layout.size()];
    layout.init(&mut buf).unwrap();
    let fill = [
        delta(0, 0, 1),
        delta(0, 1, 2),
        delta(2, 3, 3),
        delta(1, 0, 4),
    ];
    assert_eq!(layout.post(&mut buf, &fill), Ok(4));
    let snapshot = buf.clone();
    let extra = [delta(3, 9, 5)];
    assert_eq!(
        layout.post(&mut buf, &extra),
        Err(RingError::Full {
            head: 4,
            applied: 0,
            capacity: 4,
            posting: 1,
        })
    );
    assert_eq!(buf, snapshot);
    // Posting nothing is never full, and a batch that fills the ring exactly is not.
    assert_eq!(layout.post(&mut buf, &[]), Ok(4));
    layout.echo(&mut buf, 3, 11).unwrap();
    assert_eq!(layout.post(&mut buf, &extra), Ok(5));
    assert_eq!(layout.read(&buf, 4, 5), Ok(extra.to_vec()));
}

#[test]
fn a_range_the_ring_does_not_hold_is_bad() {
    let layout = DeltaRingLayout::new(4).unwrap();
    let mut buf = vec![0u8; layout.size()];
    layout.init(&mut buf).unwrap();
    let one = [delta(0, 0, 1), delta(0, 1, 2)];
    layout.post(&mut buf, &one).unwrap();
    assert_eq!(layout.read(&buf, 2, 2), Ok(Vec::new()));
    assert_eq!(
        layout.read(&buf, 0, 3),
        Err(RingError::BadRange {
            from: 0,
            to: 3,
            head: 2,
            capacity: 4,
        })
    );
    assert_eq!(
        layout.read(&buf, 3, 2),
        Err(RingError::BadRange {
            from: 3,
            to: 2,
            head: 2,
            capacity: 4,
        })
    );
    // Eight positions posted into a ring of four: only the last four are still in it.
    let pair = [delta(1, 0, 3), delta(1, 1, 4)];
    for step in 12..=14u64 {
        let live = head(&buf).unwrap();
        layout.echo(&mut buf, live, step).unwrap();
        layout.post(&mut buf, &pair).unwrap();
    }
    assert_eq!(head(&buf), Ok(8));
    assert_eq!(
        layout.read(&buf, 0, 8),
        Err(RingError::BadRange {
            from: 0,
            to: 8,
            head: 8,
            capacity: 4,
        })
    );
    assert_eq!(layout.read(&buf, 4, 8).unwrap().len(), 4);
}

fn table_delta() -> impl Strategy<Value = TableDelta> {
    (0u16.., 0u32.., 0u32..).prop_map(|(table, index, value)| TableDelta {
        table,
        _pad: 0,
        index,
        value,
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]
    /// Batches of random deltas posted into a ring of a random capacity read back in the order
    /// they were posted, however many times the ring wraps. A batch that would overflow the ring
    /// is refused as full with its bytes untouched, and the ring is usable again once the echo
    /// shows the device applied what it held.
    #[test]
    fn posted_deltas_read_back_in_order(
        pow in 2u32..8u32,
        sizes in proptest::collection::vec(1usize..40, 1..=64),
        deltas in proptest::collection::vec(table_delta(), 1..=2048),
    ) {
        let capacity = 1u32 << pow;
        let layout = DeltaRingLayout::new(capacity).unwrap();
        let mut buf = vec![0u8; layout.size()];
        layout.init(&mut buf).unwrap();

        let mut posted: Vec<TableDelta> = Vec::new();
        let mut next = 0usize;
        for (step, &size) in sizes.iter().enumerate() {
            if next == deltas.len() {
                break;
            }
            let take = size.min(deltas.len() - next);
            let batch = &deltas[next..next + take];
            let before = head(&buf).unwrap();
            let snapshot = buf.clone();
            match layout.post(&mut buf, batch) {
                Ok(after) => {
                    prop_assert_eq!(after, before + (take as u64));
                    prop_assert_eq!(layout.read(&buf, before, after).unwrap(), batch.to_vec());
                    posted.extend_from_slice(batch);
                    // Everything still in the ring, up to its capacity, is in posting order.
                    let window = posted.len().min(capacity as usize);
                    prop_assert_eq!(
                        layout.read(&buf, after - (window as u64), after).unwrap(),
                        posted[posted.len() - window..].to_vec()
                    );
                    // Every third step echoes the deltas it applied, so the ring fills up
                    // between echoes and a later batch can be refused as full.
                    if (step + 1) % 3 == 0 {
                        layout.echo(&mut buf, after, (step + 1) as u64).unwrap();
                        prop_assert_eq!(applied(&buf).unwrap(), (after, (step + 1) as u64));
                    }
                    // Posting never leaves more than `capacity` deltas waiting.
                    prop_assert!(
                        after - applied(&buf).unwrap().0 <= capacity as u64
                    );
                    next += take;
                }
                Err(err) => {
                    prop_assert_eq!(
                        err,
                        RingError::Full {
                            head: before,
                            applied: applied(&buf).unwrap().0,
                            capacity,
                            posting: take,
                        }
                    );
                    prop_assert_eq!(&buf, &snapshot);
                    prop_assert_eq!(head(&buf), Ok(before));
                    // A batch longer than the ring never fits, however much the device has
                    // applied. With the echo up to the head, one that fits posts next step.
                    if take <= capacity as usize {
                        layout.echo(&mut buf, before, (step + 1) as u64).unwrap();
                        prop_assert_eq!(layout.post(&mut buf, batch), Ok(before + (take as u64)));
                        prop_assert_eq!(
                            layout.read(&buf, before, before + (take as u64)).unwrap(),
                            batch.to_vec()
                        );
                        posted.extend_from_slice(batch);
                        next += take;
                    }
                }
            }
        }
        let live = head(&buf).unwrap();
        prop_assert_eq!(live, posted.len() as u64);
        // A range reaching past the head is a bad range.
        prop_assert_eq!(
            layout.read(&buf, live, live + 1),
            Err(RingError::BadRange {
                from: live,
                to: live + 1,
                head: live,
                capacity,
            })
        );
        // Positions older than the capacity have wrapped out of the ring.
        if posted.len() > capacity as usize {
            prop_assert_eq!(
                layout.read(&buf, 0, live),
                Err(RingError::BadRange {
                    from: 0,
                    to: live,
                    head: live,
                    capacity,
                })
            );
        }
    }
}
