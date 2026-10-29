// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tests of local tracing (PROD-E8-T07; RFC-0053 §3.1–§3.4).

use proptest::prelude::*;
use syntropy_config::ConfigSection;
use syntropy_core::MonoTime;
use syntropy_core::sequence::TraceContext;

use super::config::{ObsSection, ObsTraceSection};
use super::context::{W3C_RANDOM_TRACE_ID, W3C_SAMPLED};
use super::flight::Entry;
use super::*;
use crate::log::{CallsiteId, Level, LogRecord};
use crate::ring::{RecordSink, SinkError};
use crate::telemetry::{self, TelemetryRecord};

/// Entropy from a counter: deterministic, never all zero after the first byte.
struct Counter(std::sync::atomic::AtomicU8);

impl syntropy_core::Entropy for Counter {
    fn fill(&self, buf: &mut [u8]) {
        for b in buf {
            *b = self
                .0
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                .wrapping_add(1);
        }
    }
}

fn entropy() -> Counter {
    Counter(std::sync::atomic::AtomicU8::new(0))
}

fn record() -> SpanRecord {
    SpanRecord {
        trace_id: *b"0123456789abcdef",
        span_id: [1, 2, 3, 4, 5, 6, 7, 8],
        parent_id: [9; 8],
        start: MonoTime(0x0102_0304_0506_0708),
        dur: 0x0a0b_0c0d,
        name: SpanName::Hop.id(),
        attr_count: 4,
        attrs: [1, 2, 3, 0xffff_ffff],
    }
}

// ---- The span model -------------------------------------------------------------------------

#[test]
fn every_name_is_syntropy_crate_op_at_its_id() {
    for (i, name) in SpanName::ALL.iter().enumerate() {
        assert_eq!(usize::from(name.id().0), i, "{name:?}");
        assert_eq!(SpanName::from_id(name.id()), Some(*name));
        let parts: Vec<&str> = name.name().split('.').collect();
        assert_eq!(parts.len(), 3, "{}", name.name());
        assert_eq!(parts[0], "syntropy");
        assert!(["api", "sched", "exec", "net", "cluster"].contains(&parts[1]));
        assert!(name.attrs().len() <= 4, "{name:?}");
    }
    assert_eq!(SpanName::from_id(SpanNameId(9)), None);
    assert_eq!(SpanName::Request.name(), "syntropy.api.request");
    assert_eq!(SpanName::Step.name(), "syntropy.exec.step");
    assert!(span::GEN_AI_ATTRS.contains(&"gen_ai.request.model"));
    assert!(
        span::SYNTROPY_ATTRS
            .iter()
            .all(|a| a.starts_with("syntropy."))
    );
}

/// The record is 64 bytes in memory and on the ring, and its fields sit at the documented
/// offsets, little-endian.
#[test]
fn the_span_record_is_64_bytes_at_the_documented_offsets() {
    assert_eq!(std::mem::size_of::<SpanRecord>(), 64);
    let bytes = record().encode();
    assert_eq!(&bytes[0..16], b"0123456789abcdef");
    assert_eq!(bytes[16..24], [1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(bytes[24..32], [9; 8]);
    assert_eq!(bytes[32..40], [8, 7, 6, 5, 4, 3, 2, 1]);
    assert_eq!(bytes[40..44], [0x0d, 0x0c, 0x0b, 0x0a]);
    assert_eq!(bytes[44..46], [7, 0], "SpanName::Hop is 7");
    assert_eq!(bytes[46..48], [4, 0]);
    assert_eq!(bytes[48..52], [1, 0, 0, 0]);
    assert_eq!(bytes[60..64], [0xff; 4]);
    assert_eq!(SpanRecord::decode(&bytes), Some(record()));
}

/// A span duration saturates at `u32::MAX` nanoseconds (stand-in N1 on #1948).
#[test]
fn a_span_duration_saturates_at_u32_max() {
    let max = u64::from(u32::MAX);
    assert_eq!(SpanRecord::dur_ns(0), 0);
    assert_eq!(SpanRecord::dur_ns(max - 1), u32::MAX - 1);
    assert_eq!(SpanRecord::dur_ns(max), u32::MAX);
    assert_eq!(SpanRecord::dur_ns(max + 1), u32::MAX);
    assert_eq!(SpanRecord::dur_ns(u64::MAX), u32::MAX);
}

#[test]
fn a_bad_span_record_is_refused() {
    let good = record().encode();
    assert_eq!(SpanRecord::decode(&good[..63]), None, "short");
    let mut name = good;
    name[44] = 9;
    assert_eq!(SpanRecord::decode(&name), None, "an unknown name");
    let mut count = good;
    count[46] = 5;
    assert_eq!(SpanRecord::decode(&count), None, "five attributes");
    let mut unused = record();
    unused.attr_count = 2;
    unused.attrs = [1, 2, 0, 0];
    let mut bytes = unused.encode();
    assert_eq!(SpanRecord::decode(&bytes).unwrap().attr_count, 2);
    bytes[52] = 0;
    bytes[56] = 1;
    assert_eq!(
        SpanRecord::decode(&bytes),
        None,
        "a non-zero unused attribute"
    );
}

/// A ring of `capacity` records.
#[derive(Default)]
struct Bounded {
    records: Vec<Vec<u8>>,
    capacity: usize,
}

impl RecordSink for Bounded {
    fn push(&mut self, record: &[u8]) -> Result<(), SinkError> {
        if self.records.len() == self.capacity {
            return Err(SinkError::Full);
        }
        self.records.push(record.to_vec());
        Ok(())
    }

    fn max_record(&self) -> usize {
        64
    }
}

#[test]
fn the_span_writer_drops_and_counts_on_a_full_ring() {
    let mut w = SpanWriter::new(Bounded {
        capacity: 2,
        ..Bounded::default()
    });
    assert!(w.record(&record()));
    assert!(w.record(&record()));
    assert!(!w.record(&record()));
    assert_eq!((w.written(), w.dropped()), (2, 1));
    assert_eq!(SpanRecord::decode(&w.sink_mut().records[0]), Some(record()));
}

#[test]
fn spans_batch_into_telemetry_records_that_fit_a_slot() {
    let records: Vec<SpanRecord> = (0..20u32)
        .map(|i| SpanRecord { dur: i, ..record() })
        .collect();
    let batches = span_batches(&records, 512).unwrap();
    assert!(batches.len() > 1);
    let mut back = Vec::new();
    let mut buf = Vec::new();
    for batch in &batches {
        telemetry::encode(batch, &mut buf).unwrap();
        assert!(buf.len() <= 512, "{} bytes", buf.len());
        match telemetry::decode(&buf).unwrap() {
            TelemetryRecord::Spans { records } => back.extend(records),
            other => panic!("{other:?}"),
        }
    }
    assert_eq!(back, records, "every record, in order");
    assert_eq!(span_batches(&records, 40), None, "no slot holds one record");
    assert_eq!(span_batches(&[], 512), Some(vec![]));
}

// ---- Sampling -------------------------------------------------------------------------------

/// FNV-1a-64 as published: the FNV test suite's vectors for "", "a" and "foobar".
#[test]
fn fnv1a_64_matches_the_published_vectors() {
    assert_eq!(fnv1a_64(b""), 0xcbf2_9ce4_8422_2325);
    assert_eq!(fnv1a_64(b"a"), 0xaf63_dc4c_8601_ec8c);
    assert_eq!(fnv1a_64(b"foobar"), 0x8594_4171_f739_67e8);
}

/// The step hash's own vectors, computed by an independent implementation (Python) when the test
/// was written: `trace_id = 00 01 … 0f`, the step as u64 LE.
#[test]
fn the_step_hash_is_fnv_1a_64_and_pinned() {
    let id: [u8; 16] = core::array::from_fn(|i| i as u8);
    assert_eq!(step_hash(&id, 0), 0x7755_e7f8_592a_d815);
    assert_eq!(step_hash(&id, 1), 0x585b_20ef_4e3b_8df4);
    assert_eq!(step_hash(&id, 63), 0x4336_7973_82f9_648a);
    assert_eq!(step_hash(&id, 1000), 0xfe8d_8ec3_1912_f4ac);
    assert_eq!(step_hash(&[0; 16], 0), 0x81d2_3fd7_003c_2305);
    // 1/64 is the threshold 2^58, exactly.
    assert_eq!(StepSampler::new(1.0 / 64.0).threshold(), 1 << 58);
    assert_eq!(StepSampler::new(0.0).threshold(), 0);
    assert_eq!(StepSampler::new(1.0).threshold(), u64::MAX);
    assert_eq!(StepSampler::new(f64::NAN).threshold(), 0);
    // Over 65,536 steps of one trace, 1,033 fall under 1/64 (1,024 expected): the same count the
    // independent implementation gives.
    let sampler = StepSampler::new(1.0 / 64.0);
    let count = (0..65_536u64)
        .filter(|&s| sampler.sampled(&id, s, StepKind::Prefill, false))
        .count();
    assert_eq!(count, 1_033);
}

#[test]
fn the_first_four_decode_steps_and_a_full_trace_are_always_sampled() {
    let never = StepSampler::new(0.0);
    let id = [7; 16];
    for index in 0..FIRST_DECODE_STEPS {
        assert!(never.sampled(
            &id,
            100 + u64::from(index),
            StepKind::Decode { index },
            false
        ));
    }
    assert!(!never.sampled(&id, 104, StepKind::Decode { index: 4 }, false));
    assert!(
        !never.sampled(&id, 0, StepKind::Prefill, false),
        "prefill is not a decode step"
    );
    for step in 0..500u64 {
        assert!(never.sampled(&id, step, StepKind::Prefill, true));
    }
}

proptest! {
    /// The same (trace ID, step) gives the same decision in replay, from a sampler built apart.
    #[test]
    fn sampling_replays_identically(
        id in any::<[u8; 16]>(),
        step in any::<u64>(),
        index in any::<u32>(),
        p in 0.0f64..=1.0,
    ) {
        let (a, b) = (StepSampler::new(p), StepSampler::new(p));
        for kind in [StepKind::Prefill, StepKind::Decode { index }] {
            prop_assert_eq!(a.sampled(&id, step, kind, false), b.sampled(&id, step, kind, false));
        }
        prop_assert_eq!(step_hash(&id, step), step_hash(&id, step));
    }
}

/// A request with `trace: true` records every step span: ingress marks its sequence context as
/// a full trace, and the head then samples every step.
#[test]
fn trace_true_records_every_step() {
    let ingress = IngressTrace::start(None, true, false, &entropy());
    assert!(ingress.full);
    let seq = ingress.sequence_context([5; 8]);
    assert!(seq.full_trace());
    let sampler = StepSampler::new(1.0 / 64.0);
    for step in 0..1_000u64 {
        let kind = if step < 10 {
            StepKind::Prefill
        } else {
            StepKind::Decode {
                index: (step - 10) as u32,
            }
        };
        let frame = sampler.step_context(&seq, step, kind, [6; 8]).unwrap();
        assert!(frame.sampled() && frame.full_trace(), "step {step}");
        assert_eq!(frame.trace_id, seq.trace_id);
    }
    // Without it, most later steps carry no context.
    let plain = IngressTrace::start(None, false, false, &entropy()).sequence_context([5; 8]);
    let sampled = (14..1_014u64)
        .filter(|&s| {
            sampler
                .step_context(&plain, s, StepKind::Decode { index: s as u32 }, [6; 8])
                .is_some()
        })
        .count();
    assert!(sampled < 100, "{sampled} of 1,000 steps sampled at 1/64");
}

// ---- traceparent ----------------------------------------------------------------------------

const ID: &str = "4bf92f3577b34da6a3ce929d0e0e4736";
const PARENT: &str = "00f067aa0ba902b7";

fn header(flags: &str) -> String {
    format!("00-{ID}-{PARENT}-{flags}")
}

#[test]
fn traceparent_parses_as_w3c_gives_it() {
    let t = Traceparent::parse(&header("01")).unwrap();
    assert_eq!(t.version, 0);
    assert_eq!(t.flags, 1);
    assert_eq!(t.format(), header("01"));
    for bad in [
        format!("00-{}-{PARENT}-01", "0".repeat(32)),
        format!("00-{ID}-{}-01", "0".repeat(16)),
        format!("ff-{ID}-{PARENT}-01"),
        header("01").to_uppercase(),
        format!("00-{ID}-{PARENT}-1"),
        format!("00-{ID}-{PARENT}-01-extra"),
        format!("00-{ID}x-{PARENT}-01"),
        format!("00_{ID}-{PARENT}-01"),
        String::new(),
    ] {
        assert_eq!(Traceparent::parse(&bad), None, "{bad}");
    }
    // A later version may carry more fields after a dash.
    let later = Traceparent::parse(&format!("01-{ID}-{PARENT}-01-whatever")).unwrap();
    assert_eq!(later.version, 1);
    assert_eq!(Traceparent::parse(&format!("01-{ID}-{PARENT}-01x")), None);
}

/// The rows of the criterion: the key decides whether an incoming sampled bit forces a full
/// trace; `02` never does; `03` is `01`.
#[test]
fn only_an_honoured_incoming_bit_0_forces_a_full_trace() {
    let full = |flags: &str, honour: bool| {
        IngressTrace::start(Some(&header(flags)), false, honour, &entropy()).full
    };
    assert!(
        !full("01", false),
        "a sampled traceparent alone does not force a full trace"
    );
    assert!(full("01", true), "with the key set it does");
    assert!(!full("02", false), "02 never enables full tracing");
    assert!(
        !full("02", true),
        "02 never enables full tracing, key set or not"
    );
    assert_eq!(
        full("03", false),
        full("01", false),
        "03 is 01 without the key"
    );
    assert_eq!(full("03", true), full("01", true), "03 is 01 with the key");
    assert!(!full("00", true));
    // The sequence's context carries Syntropy's flags only: a client's bit 0 alone sets nothing.
    let seq =
        IngressTrace::start(Some(&header("03")), false, false, &entropy()).sequence_context([1; 8]);
    assert_eq!(seq.flags, 0);
    let forced =
        IngressTrace::start(Some(&header("01")), false, true, &entropy()).sequence_context([1; 8]);
    assert_eq!(
        forced.flags,
        TraceContext::SAMPLED | TraceContext::FULL_TRACE
    );
}

/// A returned traceparent: bit 0 set, bit 1 exactly when the incoming one had it, never the
/// full-trace bit; and none for a request without one.
#[test]
fn the_returned_traceparent_sets_its_own_flags() {
    for (flags, trace_requested, honour, want) in [
        ("00", false, false, 0x01),
        ("01", false, false, 0x01),
        ("02", false, false, 0x03),
        ("03", false, false, 0x03),
        // A full trace, by request or by an honoured bit 0, does not leak its bit.
        ("00", true, false, 0x01),
        ("01", false, true, 0x01),
        ("03", false, true, 0x03),
    ] {
        let t = IngressTrace::start(Some(&header(flags)), trace_requested, honour, &entropy());
        let back = Traceparent::parse(&t.response_traceparent().unwrap()).unwrap();
        assert_eq!(back.flags, want, "incoming {flags}, full trace {}", t.full);
        assert_eq!(back.flags & W3C_SAMPLED, W3C_SAMPLED);
        assert_eq!(
            back.flags & W3C_RANDOM_TRACE_ID != 0,
            flags == "02" || flags == "03",
            "bit 1 echoes the incoming bit 1 only"
        );
        // The trace continues, under ingress's root span.
        assert_eq!(
            back.trace_id,
            Traceparent::parse(&header(flags)).unwrap().trace_id
        );
        assert_eq!(back.parent_id, t.root_span);
        assert_ne!(
            back.parent_id,
            Traceparent::parse(&header(flags)).unwrap().parent_id
        );
        assert_eq!(
            t.parent_id,
            Traceparent::parse(&header(flags)).unwrap().parent_id
        );
    }
    // No traceparent in, none out; an unparseable one counts as none.
    for incoming in [None, Some("garbage"), Some("00-zz-zz-01")] {
        let t = IngressTrace::start(incoming, true, true, &entropy());
        assert_eq!(t.response_traceparent(), None, "{incoming:?}");
        assert_ne!(t.trace_id, [0; 16]);
        assert_eq!(t.parent_id, [0; 8]);
        // A minted trace-id is random, so exported spans carry bit 1.
        assert_eq!(t.w3c_flags(), W3C_SAMPLED | W3C_RANDOM_TRACE_ID);
    }
}

#[test]
fn ids_are_never_zero_even_from_a_zero_source() {
    struct Zeros;
    impl syntropy_core::Entropy for Zeros {
        fn fill(&self, buf: &mut [u8]) {
            buf.fill(0);
        }
    }
    let t = IngressTrace::start(None, false, false, &Zeros);
    assert_ne!(t.trace_id, [0; 16]);
    assert_ne!(t.root_span, [0; 8]);
}

// ---- The flight recorder --------------------------------------------------------------------

fn log(n: u64) -> LogRecord {
    LogRecord {
        callsite: CallsiteId(1),
        level: Level::Info,
        at: MonoTime(n),
        args: [n, 0, 0, 0],
    }
}

/// The entry sizes the flight recorder's heap estimate rests on (N3 on #1948).
#[test]
fn the_flight_recorder_s_entries_are_72_bytes() {
    assert_eq!(std::mem::size_of::<Entry>(), 72);
    assert_eq!(std::mem::size_of::<LogRecord>(), 48);
}

#[test]
fn the_flight_recorder_keeps_the_newest_within_its_bound() {
    // Room for three spans (192 bytes) and nothing more.
    let mut fr = FlightRecorder::new(200);
    let span = |trace: u8, dur| SpanRecord {
        trace_id: [trace; 16],
        dur,
        ..record()
    };
    fr.push(Entry::Span(span(1, 1)));
    fr.push(Entry::Span(span(2, 2)));
    fr.push(Entry::Span(span(1, 3)));
    assert_eq!((fr.len(), fr.held(), fr.evicted()), (3, 192, 0));
    fr.push(Entry::Log(log(9)));
    // A 48-byte log record pushes out the oldest span.
    assert_eq!((fr.len(), fr.held(), fr.evicted()), (3, 176, 1));
    let durs: Vec<u32> = fr.trace(&[1; 16]).map(|s| s.dur).collect();
    assert_eq!(durs, vec![3], "trace 1's first span was evicted");
    assert_eq!(fr.trace(&[2; 16]).count(), 1);
    assert_eq!(fr.logs().map(|l| l.at.0).collect::<Vec<_>>(), vec![9]);
    // An entry larger than the recorder is not kept.
    let mut tiny = FlightRecorder::new(10);
    tiny.push(Entry::Span(record()));
    assert!(tiny.is_empty());
    assert_eq!(tiny.evicted(), 1);
}

// ---- Configuration --------------------------------------------------------------------------

#[test]
fn the_tracing_keys_have_their_rfc_defaults() {
    use syntropy_config::{DefaultSpec, DefaultValue, Reload, Scope};
    let obs = &ObsSection::SCHEMA;
    let fr = obs.key("flight_recorder").unwrap();
    assert_eq!(fr.path, "obs.flight_recorder");
    assert_eq!(fr.scope, Scope::Node);
    assert!(matches!(
        fr.default,
        DefaultSpec::Value(DefaultValue::Str("64MiB"))
    ));
    let trace = &ObsTraceSection::SCHEMA;
    let sample = trace.key("step_sample").unwrap();
    assert_eq!(sample.path, "obs.trace.step_sample");
    assert!(
        matches!(sample.default, DefaultSpec::Value(DefaultValue::Float(f)) if f == 1.0 / 64.0)
    );
    assert_eq!(sample.reload, Reload::Hot);
    let honour = trace.key("honor_remote_sampled").unwrap();
    assert_eq!(honour.path, "obs.trace.honor_remote_sampled");
    assert!(matches!(
        honour.default,
        DefaultSpec::Value(DefaultValue::Bool(false))
    ));
    assert!(ObsSection::check_defaults().is_empty());
    assert!(ObsTraceSection::check_defaults().is_empty());
    static REGISTRY: syntropy_config::Registry =
        syntropy_config::Registry::new(&[&ObsSection::SCHEMA, &ObsTraceSection::SCHEMA]);
    assert!(REGISTRY.check().is_empty());
}
