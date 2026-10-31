// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::{Arc, Mutex};

use syntropy_core::ErrorClass;
use syntropy_kernels::error::codes;
use syntropy_kernels::{Event, Stream, StreamPlan, StreamPriority};

use super::fake::{Call, FakeStreams, Shared, State};
use super::*;

const ALL: [StreamPriority; 5] = [
    StreamPriority::Compute,
    StreamPriority::Collective,
    StreamPriority::CopyIn,
    StreamPriority::CopyOut,
    StreamPriority::Low,
];

fn plan(compute: u8, copy_in: u8, copy_out: u8, collective: u8) -> StreamPlan {
    StreamPlan {
        compute,
        copy_in,
        copy_out,
        collective,
    }
}

/// A fake with Ada's priority range, `[-5, 0]` [measured: `cuCtxGetStreamPriorityRange`,
/// RTX 4060 Ti, driver 595.91.07].
fn fake() -> Shared {
    Arc::new(Mutex::new(State {
        range: (0, -5),
        ..State::default()
    }))
}

fn open(shared: &Shared, p: &StreamPlan) -> DeviceStreams {
    DeviceStreams::open(Box::new(FakeStreams(shared.clone())), p).unwrap()
}

fn log(shared: &Shared) -> Vec<Call> {
    shared.lock().unwrap().log.clone()
}

/// RFC-0040 §4.1: compute highest. On Ada's six levels each class has its own.
#[test]
fn compute_has_the_highest_priority_on_ada() {
    let got: Vec<i32> = ALL.iter().map(|&p| stream_priority(p, 0, -5)).collect();
    assert_eq!(got, [-5, -4, -3, -2, 0]);
}

/// Every range a driver can report: each class gets a level inside it, `Compute` the highest,
/// the order never inverts, and `Compute` is strictly above the rest whenever there are two
/// levels or more.
#[test]
fn priorities_stay_in_range_and_in_order_for_every_range() {
    for least in -8..=8 {
        for greatest in -16..=least {
            let p: Vec<i32> = ALL
                .iter()
                .map(|&c| stream_priority(c, least, greatest))
                .collect();
            assert!(p.iter().all(|&v| (greatest..=least).contains(&v)), "{p:?}");
            assert_eq!(p[0], greatest);
            assert!(p.windows(2).all(|w| w[0] <= w[1]), "{p:?}");
            assert_eq!(p[4], least);
            if least > greatest {
                assert!(p[1..].iter().all(|&v| v > p[0]), "{p:?}");
            }
            // The range's ends may come swapped; the answer is the same.
            let q: Vec<i32> = ALL
                .iter()
                .map(|&c| stream_priority(c, greatest, least))
                .collect();
            assert_eq!(p, q);
        }
    }
}

#[test]
fn a_device_with_one_level_gives_every_class_that_level() {
    for c in ALL {
        assert_eq!(stream_priority(c, 0, 0), 0);
    }
}

#[test]
fn the_plan_lists_its_streams_by_class() {
    use StreamPriority::*;
    assert_eq!(
        planned(&plan(1, 2, 1, 1)),
        [Compute, CopyIn, CopyIn, CopyOut, Collective]
    );
    assert!(planned(&plan(0, 0, 0, 0)).is_empty());
}

/// `open` creates the plan's streams, each with its class's priority; `stream` hands them out
/// once each and then creates more.
#[test]
fn open_creates_the_plan_and_stream_hands_it_out() {
    let f = fake();
    let d = open(&f, &plan(1, 1, 1, 0));
    assert_eq!(
        log(&f),
        [
            Call::CreateStream(-5),
            Call::CreateStream(-3),
            Call::CreateStream(-2)
        ]
    );
    assert_eq!(
        d.priorities(),
        [
            (StreamPriority::Compute, -5),
            (StreamPriority::CopyIn, -3),
            (StreamPriority::CopyOut, -2)
        ]
    );
    let copy_in = d.stream(StreamPriority::CopyIn).unwrap();
    let compute = d.stream(StreamPriority::Compute).unwrap();
    assert_ne!(copy_in, compute);
    assert_eq!(f.lock().unwrap().live(), (3, 0), "no stream created yet");
    let second = d.stream(StreamPriority::Compute).unwrap();
    assert_ne!(second, compute);
    assert_eq!(log(&f).last(), Some(&Call::CreateStream(-5)));
    let low = d.stream(StreamPriority::Low).unwrap();
    assert_eq!(log(&f).last(), Some(&Call::CreateStream(0)));
    assert_eq!(f.lock().unwrap().live(), (5, 0));
    assert_ne!(low, second);
}

/// Events order work across streams: `record` on the copy-in stream and `wait` on the compute
/// stream reach the driver with those streams' handles.
#[test]
fn record_and_wait_cross_streams() {
    let f = fake();
    let d = open(&f, &plan(1, 1, 0, 0));
    let compute = d.stream(StreamPriority::Compute).unwrap();
    let copy_in = d.stream(StreamPriority::CopyIn).unwrap();
    let e = d.event(false).unwrap();
    f.lock().unwrap().log.clear();
    d.record(&copy_in, &e).unwrap();
    d.wait(&compute, &e).unwrap();
    let calls = log(&f);
    let (Call::Record { e: e1, s: s1 }, Call::Wait { s: s2, e: e2 }) = (&calls[0], &calls[1])
    else {
        panic!("{calls:?}")
    };
    assert_eq!(e1, e2);
    assert_ne!(s1, s2);
    // The copy-in stream is the plan's second, created at -3; compute is first, at -5.
    let st = f.lock().unwrap();
    assert_eq!(st.log.len(), 2);
    drop(st);
    assert_eq!(*s1, 0x5000 + 2);
    assert_eq!(*s2, 0x5000 + 1);
}

/// A handle another device issued, or a made-up one, is refused before any driver call.
#[test]
fn a_foreign_or_made_up_handle_is_refused() {
    let (f, g) = (fake(), fake());
    let (a, b) = (open(&f, &plan(1, 0, 0, 0)), open(&g, &plan(1, 0, 0, 0)));
    let sa = a.stream(StreamPriority::Compute).unwrap();
    let ea = a.event(false).unwrap();
    let sb = b.stream(StreamPriority::Compute).unwrap();
    let eb = b.event(false).unwrap();
    f.lock().unwrap().log.clear();
    for err in [
        a.record(&sb, &ea).unwrap_err(),
        a.record(&sa, &eb).unwrap_err(),
        a.wait(&sa, &Event { handle: 7 }).unwrap_err(),
        a.poll(&eb).unwrap_err(),
        a.synchronize(&Stream { handle: u64::MAX }).unwrap_err(),
    ] {
        assert_eq!(err.class, ErrorClass::Internal, "{err}");
    }
    // An index past the end of this device's table.
    let past = Event {
        handle: ea.handle + 1,
    };
    assert!(a.poll(&past).is_err());
    assert!(log(&f).is_empty(), "{:?}", log(&f));
}

#[test]
fn poll_is_the_event_query() {
    let f = fake();
    let d = open(&f, &plan(1, 0, 0, 0));
    let e = d.event(false).unwrap();
    assert!(!d.poll(&e).unwrap());
    f.lock().unwrap().ready = true;
    assert!(d.poll(&e).unwrap());
}

/// `elapsed` reads only timing events: `event`'s have timing off (RFC-0040 §4.3).
#[test]
fn elapsed_needs_two_timing_events() {
    let f = fake();
    let d = open(&f, &plan(1, 0, 0, 0));
    let (plain, t0, t1) = (
        d.event(false).unwrap(),
        d.event(true).unwrap(),
        d.event(true).unwrap(),
    );
    assert!(log(&f).contains(&Call::CreateEvent { timing: false }));
    assert!(log(&f).contains(&Call::CreateEvent { timing: true }));
    f.lock().unwrap().elapsed_ms = 1.5;
    assert_eq!(d.elapsed(&t0, &t1), Some(Nanos(1_500_000)));
    assert_eq!(d.elapsed(&plain, &t1), None);
    assert_eq!(d.elapsed(&t0, &plain), None);
    // Not complete yet: the driver answers NOT_READY.
    f.lock().unwrap().fail = Some(("cuEventElapsedTime", NOT_READY));
    assert_eq!(d.elapsed(&t0, &t1), None);
    f.lock().unwrap().elapsed_ms = -0.25;
    assert_eq!(d.elapsed(&t0, &t1), None);
}

/// RFC-0040 §13: a sticky error is `DeviceSticky` with class `Device`; out of memory is
/// `AllocFailed`; anything else is a provider fault.
#[test]
fn driver_errors_map_to_their_class() {
    let fault = |status| StreamFault {
        call: "cuEventRecord",
        status,
        error: format!("CUDA error {status}"),
    };
    for status in STICKY {
        let e = stream_error(&fault(status));
        assert_eq!(
            (e.class, e.code),
            (ErrorClass::Device, codes::DEVICE_STICKY)
        );
    }
    let e = stream_error(&fault(OUT_OF_MEMORY));
    assert_eq!(
        (e.class, e.code),
        (ErrorClass::Capacity, codes::ALLOC_FAILED)
    );
    for status in [1, 400, 600, 0xFFFF] {
        let e = stream_error(&fault(status));
        assert_eq!(e.class, ErrorClass::Internal, "{status}");
    }
    assert!(stream_error(&fault(700)).message.contains("cuEventRecord"));
}

#[test]
fn a_sticky_error_on_record_reaches_the_caller() {
    let f = fake();
    let d = open(&f, &plan(1, 0, 0, 0));
    let s = d.stream(StreamPriority::Compute).unwrap();
    let e = d.event(false).unwrap();
    f.lock().unwrap().fail = Some(("cuEventRecord", 700));
    let err = d.record(&s, &e).unwrap_err();
    assert_eq!(err.code, codes::DEVICE_STICKY);
    f.lock().unwrap().fail = Some(("cuStreamCreateWithPriority", OUT_OF_MEMORY));
    let err = d.stream(StreamPriority::Low).unwrap_err();
    assert_eq!(err.code, codes::ALLOC_FAILED);
}

fn pairs(n: u64) -> Vec<(u64, u64, u64)> {
    // Every seventh pair is empty.
    (0..n)
        .map(|i| {
            (
                0x1_0000 + i * 64,
                0x9_0000 + i * 64,
                if i % 7 == 3 { 0 } else { 8 + i },
            )
        })
        .collect()
}

/// The copies a log describes, in order, whether issued one by one or in batches.
fn moved(calls: &[Call]) -> Vec<(u64, u64, usize)> {
    calls
        .iter()
        .flat_map(|c| match c {
            Call::Copy {
                dst, src, bytes, ..
            } => vec![(*dst, *src, *bytes)],
            Call::Batch { pairs, .. } => pairs.clone(),
            _ => vec![],
        })
        .collect()
}

/// `copy_batch` moves the same bytes as a loop of copies: one `cuMemcpyBatchAsync` per
/// `BATCH_CHUNK` pairs, in order, skipping empty pairs, where the driver has it; a loop of
/// `cuMemcpyAsync` otherwise.
#[test]
fn copy_batch_moves_what_a_loop_of_copies_moves() {
    let input = pairs(300);
    let expected: Vec<(u64, u64, usize)> = input
        .iter()
        .filter(|p| p.2 > 0)
        .map(|&(d, s, b)| (d, s, b as usize))
        .collect();

    let looped = fake();
    let d = open(&looped, &plan(0, 1, 0, 0));
    let s = d.stream(StreamPriority::CopyIn).unwrap();
    looped.lock().unwrap().log.clear();
    for &(dst, src, bytes) in &input {
        // SAFETY: the fake reads no memory.
        unsafe { d.copy(&s, dst, src, bytes) }.unwrap();
    }
    let by_copy = moved(&log(&looped));
    assert_eq!(by_copy, expected);

    // A driver without the batched entry point: `copy_batch` is the same loop.
    let fallback = fake();
    let d = open(&fallback, &plan(0, 1, 0, 0));
    let s = d.stream(StreamPriority::CopyIn).unwrap();
    fallback.lock().unwrap().log.clear();
    // SAFETY: the fake reads no memory.
    unsafe { d.copy_batch(&s, input.iter().map(|&p| Ok(p))) }.unwrap();
    assert!(!d.batched());
    assert_eq!(log(&fallback), log(&looped));

    let batched = fake();
    batched.lock().unwrap().batched = true;
    let d = open(&batched, &plan(0, 1, 0, 0));
    let s = d.stream(StreamPriority::CopyIn).unwrap();
    batched.lock().unwrap().log.clear();
    // SAFETY: the fake reads no memory.
    unsafe { d.copy_batch(&s, input.iter().map(|&p| Ok(p))) }.unwrap();
    assert!(d.batched());
    let calls = log(&batched);
    let sizes: Vec<usize> = calls
        .iter()
        .map(|c| match c {
            Call::Batch { pairs, .. } => pairs.len(),
            other => panic!("{other:?}"),
        })
        .collect();
    assert_eq!(
        sizes,
        [BATCH_CHUNK, BATCH_CHUNK, expected.len() - 2 * BATCH_CHUNK]
    );
    assert_eq!(moved(&calls), expected);
}

#[test]
fn a_failed_pair_stops_the_batch() {
    let f = fake();
    f.lock().unwrap().batched = true;
    let d = open(&f, &plan(0, 1, 0, 0));
    let s = d.stream(StreamPriority::CopyIn).unwrap();
    f.lock().unwrap().log.clear();
    let bad = Error::new(ErrorClass::Internal, codes::OP_UNSUPPORTED, "bad slice");
    let input = vec![Ok((1, 2, 8)), Err(bad.clone()), Ok((3, 4, 8))];
    // SAFETY: the fake reads no memory.
    let err = unsafe { d.copy_batch(&s, input) }.unwrap_err();
    assert_eq!(err, bad);
    assert!(
        log(&f).is_empty(),
        "nothing is submitted before the slices check out"
    );
}

/// Closing the device drains every stream, destroys the events and then the streams, and
/// leaves nothing live.
#[test]
fn drop_drains_then_destroys_events_then_streams() {
    let f = fake();
    let d = open(&f, &plan(1, 1, 1, 0));
    let _ = d.stream(StreamPriority::Low).unwrap();
    let _ = (d.event(false).unwrap(), d.event(true).unwrap());
    f.lock().unwrap().log.clear();
    drop(d);
    let calls = log(&f);
    assert_eq!(f.lock().unwrap().live(), (0, 0));
    let kind = |c: &Call| match c {
        Call::Sync(_) => 0,
        Call::DestroyEvent(..) => 1,
        Call::DestroyStream(..) => 2,
        other => panic!("{other:?}"),
    };
    let kinds: Vec<u8> = calls.iter().map(kind).collect();
    assert_eq!(kinds, [0, 0, 0, 0, 1, 1, 2, 2, 2, 2]);
}

/// An `open` that fails, at the priority range or at the plan's third stream, leaves no stream
/// behind.
#[test]
fn a_failed_open_destroys_what_it_made() {
    let g = fake();
    let api = Box::new(FakeStreams(g.clone()));
    g.lock().unwrap().fail = Some(("cuCtxGetStreamPriorityRange", 1));
    assert!(DeviceStreams::open(api, &plan(1, 1, 1, 0)).is_err());
    assert_eq!(g.lock().unwrap().live(), (0, 0));

    // A creation failure part-way through the plan.
    struct ThirdFails(FakeStreams, std::cell::Cell<u32>);
    impl StreamApi for ThirdFails {
        fn priority_range(&self) -> Result<(i32, i32), StreamFault> {
            self.0.priority_range()
        }
        fn create_stream(&self, p: i32) -> Result<u64, StreamFault> {
            self.1.set(self.1.get() + 1);
            if self.1.get() == 3 {
                return Err(StreamFault {
                    call: "cuStreamCreateWithPriority",
                    status: OUT_OF_MEMORY,
                    error: "out of memory".into(),
                });
            }
            self.0.create_stream(p)
        }
        fn create_event(&self, t: bool) -> Result<u64, StreamFault> {
            self.0.create_event(t)
        }
        unsafe fn destroy_stream(&self, s: u64) -> Result<(), StreamFault> {
            // SAFETY: forwarded under the same contract.
            unsafe { self.0.destroy_stream(s) }
        }
        unsafe fn destroy_event(&self, e: u64) -> Result<(), StreamFault> {
            // SAFETY: forwarded under the same contract.
            unsafe { self.0.destroy_event(e) }
        }
        unsafe fn record(&self, e: u64, s: u64) -> Result<(), StreamFault> {
            // SAFETY: forwarded under the same contract.
            unsafe { self.0.record(e, s) }
        }
        unsafe fn wait(&self, s: u64, e: u64) -> Result<(), StreamFault> {
            // SAFETY: forwarded under the same contract.
            unsafe { self.0.wait(s, e) }
        }
        unsafe fn query(&self, e: u64) -> Result<bool, StreamFault> {
            // SAFETY: forwarded under the same contract.
            unsafe { self.0.query(e) }
        }
        unsafe fn elapsed_ms(&self, a: u64, b: u64) -> Result<f32, StreamFault> {
            // SAFETY: forwarded under the same contract.
            unsafe { self.0.elapsed_ms(a, b) }
        }
        unsafe fn synchronize(&self, s: u64) -> Result<(), StreamFault> {
            // SAFETY: forwarded under the same contract.
            unsafe { self.0.synchronize(s) }
        }
        unsafe fn copy(&self, s: u64, d: u64, r: u64, b: usize) -> Result<(), StreamFault> {
            // SAFETY: forwarded under the same contract.
            unsafe { self.0.copy(s, d, r, b) }
        }
        fn batched(&self) -> bool {
            false
        }
        unsafe fn copy_batch(
            &self,
            _: u64,
            _: &[u64],
            _: &[u64],
            _: &[usize],
        ) -> Result<(), StreamFault> {
            unreachable!()
        }
    }
    let h = fake();
    let api = Box::new(ThirdFails(FakeStreams(h.clone()), Default::default()));
    let err = DeviceStreams::open(api, &plan(1, 1, 1, 0)).unwrap_err();
    assert_eq!(err.code, codes::ALLOC_FAILED);
    assert_eq!(h.lock().unwrap().live(), (0, 0));
    assert_eq!(
        log(&h)
            .iter()
            .filter(|c| matches!(c, Call::DestroyStream(..)))
            .count(),
        2
    );
}

#[test]
fn an_empty_copy_makes_no_call() {
    let f = fake();
    let d = open(&f, &plan(1, 0, 0, 0));
    let s = d.stream(StreamPriority::Compute).unwrap();
    f.lock().unwrap().log.clear();
    // SAFETY: the fake reads no memory.
    unsafe { d.copy(&s, 1, 2, 0) }.unwrap();
    // SAFETY: as above.
    unsafe { d.copy_batch(&s, [Ok((1, 2, 0))]) }.unwrap();
    assert!(log(&f).is_empty());
}

mod gpu;
