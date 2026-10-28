// SPDX-License-Identifier: MIT OR Apache-2.0

//! What the `SeqControl` decoder holds while it reads hostile frames, against RFC-0060 §10.2's
//! bound for frame decoders: total allocation at most 4 × the input + 1 MiB, unless the format
//! declares a larger size (DIST-E8-T05; #1809's review, B1 and N1).
//!
//! A binary of its own, so its global allocator, which records the peak of the bytes held, sees
//! only this test. Each frame is built before measuring, and only the decode is measured. Each
//! hostile body goes out twice: compressed in one call, whose zstd frame header states the
//! content size, which the decoder refuses before it allocates; and streamed, whose header states
//! none, so the decoder must decompress up to its limit to find out. The allocator sees Rust's
//! heap; libzstd's own decoder context comes from C `malloc`, and its window is bounded by the
//! limit's window log, which the envelope checks (RFC-0031 §9.1).

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use syntropy_core::{
    ExactnessPolicy, InstanceId, NodeId, PlanEpoch, PrefixScope, Priority, SamplingParams,
    SequenceId, SequenceLimits, SequenceOpen, TokenSlice,
};
use syntropy_net::ctl::envelope;
use syntropy_net::frame::{
    Flags, Frame, FrameSpec, FrameType, SeqControlError, SeqControlLimits, WIRE_MINOR,
    encode_to_vec, open_body_limit, sequence_slot_size,
};

/// The system allocator, recording the bytes held and their peak.
struct Peak;

static HELD: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grew(by: usize) {
    let now = HELD.fetch_add(by, Ordering::Relaxed) + by;
    PEAK.fetch_max(now, Ordering::Relaxed);
}

// SAFETY: every method passes its arguments unchanged to `System`, which keeps `GlobalAlloc`'s
// contract, and returns what `System` returns. The only other work is on two atomics, which
// neither allocates nor unwinds nor touches the memory involved.
unsafe impl GlobalAlloc for Peak {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's contract for `alloc`, passed on.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            grew(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the caller's contract for `dealloc`, passed on.
        unsafe { System.dealloc(ptr, layout) };
        HELD.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's contract for `alloc_zeroed`, passed on.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            grew(layout.size());
        }
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the caller's contract for `realloc`, passed on.
        let moved = unsafe { System.realloc(ptr, layout, new_size) };
        if !moved.is_null() {
            HELD.fetch_sub(layout.size(), Ordering::Relaxed);
            grew(new_size);
        }
        moved
    }
}

#[global_allocator]
static ALLOCATOR: Peak = Peak;

/// The most bytes held at once while `f` runs, above what was held when it started.
fn peak_during<R>(f: impl FnOnce() -> R) -> (R, usize) {
    let before = HELD.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let result = f();
    (result, PEAK.load(Ordering::Relaxed).saturating_sub(before))
}

fn seq() -> SequenceId {
    SequenceId {
        ingress: 7,
        boot: 1,
        n: 1,
    }
}

/// `body` compressed in one call, with its size in the zstd frame header, or streamed, without.
fn compress(body: &[u8], streamed: bool) -> Vec<u8> {
    if streamed {
        use std::io::Write;
        let mut encoder = zstd::stream::Encoder::new(Vec::new(), 3).unwrap();
        encoder.write_all(body).unwrap();
        encoder.finish().unwrap()
    } else {
        zstd::bulk::compress(body, 3).unwrap()
    }
}

/// A `SeqControl` frame of `kind`, its body zstd-compressed.
fn compressed_frame(kind: u8, body: &[u8], streamed: bool) -> Vec<u8> {
    seq_control_frame(kind, 1, &compress(body, streamed))
}

/// A `SeqControl` frame of `kind` whose body, encoded as `body_enc` says, is `payload`.
fn seq_control_frame(kind: u8, body_enc: u8, payload: &[u8]) -> Vec<u8> {
    let mut ext = [0u8; 64];
    ext[0] = kind;
    ext[1] = body_enc;
    let spec = FrameSpec {
        wire_minor: WIRE_MINOR,
        ftype: FrameType::SeqControl,
        flags: Flags::default(),
        instance: InstanceId(1),
        epoch: PlanEpoch { term: 1, seq: 1 },
        step: 0,
        chan_seq: 1,
    };
    encode_to_vec(&spec, &ext, payload).unwrap()
}

/// B1's frame: a `Reconcile` whose `recent` declares `zeros` zero tokens, as bytes.
fn reconcile_with_zeros(zeros: u32) -> Vec<u8> {
    reconcile_claiming(zeros, zeros as usize)
}

/// A `Reconcile` body whose `recent` array header claims `claimed` tokens and is followed by
/// `held` zero bytes, each one token.
fn reconcile_claiming(claimed: u32, held: usize) -> Vec<u8> {
    #[derive(serde::Serialize)]
    struct Head {
        seq: SequenceId,
        emitted: u32,
        rng_counter: u64,
    }
    // The three fields as a map of four, then `recent`'s key, its array header and its zeros.
    let mut body = envelope::to_cbor(&Head {
        seq: seq(),
        emitted: 4,
        rng_counter: 99,
    })
    .unwrap();
    body[0] = 0xa4;
    body.extend([0x66, b'r', b'e', b'c', b'e', b'n', b't', 0x9a]);
    body.extend(claimed.to_be_bytes());
    body.resize(body.len() + held, 0);
    body
}

fn open_with_zero_tokens(tokens: usize) -> SequenceOpen {
    SequenceOpen {
        seq: seq(),
        session: None,
        ingress: NodeId([1; 32]),
        prompt: TokenSlice(vec![0; tokens]),
        prefix: None,
        sampling: SamplingParams::default(),
        limits: SequenceLimits::default(),
        exactness: ExactnessPolicy::default(),
        priority: Priority::default(),
        choice_index: None,
        prefill_chunk: 0,
        initial_window: None,
        resume: None,
        prefix_scope: PrefixScope::default(),
        session_pk: None,
        trace: None,
        parity: None,
    }
}

/// One test, so nothing else in the binary allocates while a decode is measured.
#[test]
fn hostile_frames_stay_within_rfc_0060_s10_2s_bound() {
    let slot = sequence_slot_size(128 << 10);
    let mib = 1 << 20;

    for streamed in [false, true] {
        // B1: a 60M-token `Reconcile` in a small frame is refused within 4 × the input + 1 MiB.
        let frame = compressed_frame(7, &reconcile_with_zeros(60_000_000), streamed);
        let (decoded, peak) = peak_during(|| {
            Frame::decode(&frame)
                .unwrap()
                .seq_control(SeqControlLimits::slot(slot))
        });
        assert!(decoded.is_err());
        assert!(
            peak <= 4 * frame.len() + mib,
            "B1: {peak} bytes held for a {}-byte frame",
            frame.len()
        );
        eprintln!(
            "B1, streamed {streamed}: a {}-byte frame, {peak} bytes held at most",
            frame.len()
        );

        // N1: a 15M-token `Open` in a small frame is refused within the size the format declares
        // for an open on this slot, `open_body_limit(slot)`, four times over, + 1 MiB.
        let body = envelope::to_cbor(&open_with_zero_tokens(15_000_000)).unwrap();
        let frame = compressed_frame(1, &body, streamed);
        drop(body);
        let (decoded, peak) = peak_during(|| {
            Frame::decode(&frame)
                .unwrap()
                .seq_control(SeqControlLimits::slot(slot))
        });
        assert!(decoded.is_err());
        let declared = open_body_limit(slot);
        assert!(
            peak <= 4 * declared + mib,
            "N1: {peak} bytes held for a {}-byte frame, declared {declared}",
            frame.len()
        );
        eprintln!(
            "N1, streamed {streamed}: a {}-byte frame, {peak} bytes held at most, {declared} \
             declared",
            frame.len()
        );
    }

    // #1804's review, N1. A plain `Reconcile` whose `recent` claims 2^32 - 1 tokens and holds
    // nine is refused by the envelope's length check, before the body is read. One that holds
    // 900 tokens, as many as fit under the small kinds' 1 KiB ceiling, is refused for having
    // more than 8, and the decoder holds none of them: a `Vec<u32>` would hold 3.6 KB. The bound
    // [chosen] is the ceiling itself.
    for (claimed, held) in [(u32::MAX, 9), (900, 900)] {
        let frame = seq_control_frame(7, 0, &reconcile_claiming(claimed, held));
        let (decoded, peak) = peak_during(|| {
            Frame::decode(&frame)
                .unwrap()
                .seq_control(SeqControlLimits::slot(slot))
        });
        let error = decoded.unwrap_err();
        if claimed == 900 {
            assert_eq!(error, SeqControlError::TooManyRecent { len: 900 });
        }
        assert!(
            peak <= 1 << 10,
            "N1 of #1804: {peak} bytes held for a {}-byte frame claiming {claimed} tokens",
            frame.len()
        );
        eprintln!(
            "N1 of #1804: a {}-byte frame claiming {claimed} tokens and holding {held}: {:?}, \
             {peak} bytes held at most",
            frame.len(),
            error
        );
    }
}
