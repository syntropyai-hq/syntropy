// SPDX-License-Identifier: MIT OR Apache-2.0

//! The collective and expert-parallel codecs against the golden frames in
//! `tests/golden/collective/` (DIST-E8-T08; RFC-0031 §6.8; RFC-0032 §5.2–§5.4).
//!
//! Each golden frame is written twice: as wire values, which [`spec`] lays out from RFC-0031
//! §6.8's tables, and as the codec's typed values. The tests check that the files match the
//! tables, that each decodes to its typed values and encodes back to the same bytes, that
//! `ROUTER_BCAST` reads per row, that a `StepDescriptor` carries an `Activation` unchanged, that
//! an `EpDispatch` splits into parts whose row limit counts the extension, at RFC-0031 §6.8's
//! figures, and that a frame breaking a rule is refused with a classified error, never
//! `Internal` (RFC-0060 §10.2).
//!
//! `SYNTROPY_BLESS_GOLDEN=1` rewrites the files, as it does the other frames.

use std::collections::BTreeSet;
use std::path::PathBuf;

use proptest::prelude::*;
use syntropy_core::plan::WireDType;
use syntropy_core::{ErrorClass, InstanceId, SequenceId, StageIx};
use syntropy_net::frame::{
    Activation, ActivationHeader, ActivationKind, ActivationSpec, Collective, CollectiveHeader,
    CollectivePayload, CollectivePhase, CollectiveSpec, EpCombine, EpDispatch, EpHeader, EpRow,
    EpRows, Experts, Flags, Frame, FrameError, FrameType, Part, RouterRow, RouterRows, RowLayout,
    SideItems, Span, Spans, TreeArea, WIRE_MINOR, Weights, WireAlgo, code, route_row_len,
};

use super::{REFUSAL_CLASSES, base, epoch, golden_dir, leak, reseal, spec};

/// The step every golden carries, in the base header and the extension.
const STEP: u64 = 30;

/// Thunderbolt 5's RDMA message limit, `TransportCaps::max_frame` on that link (RFC-0031 §2).
const TB5: usize = 16_773_120;

const FIRST_OF_SEQ: u16 = 1 << 3;
const HAS_TRACE: u16 = 1;
const ENCRYPTED: u16 = 1 << 2;

/// What a golden frame decodes to.
#[derive(Clone, Copy, Debug)]
enum Typed {
    Collective(Collective<'static>),
    StepDescriptor(Activation<'static>),
    Dispatch(EpDispatch<'static>),
    Combine(EpCombine<'static>),
}

struct Golden {
    name: &'static str,
    about: &'static str,
    bytes: Vec<u8>,
    /// The channel's wire dtype.
    wire: WireDType,
    flags: u16,
    chan_seq: u64,
    typed: Typed,
}

impl Golden {
    fn spec(&self) -> CollectiveSpec {
        cspec(self.flags, self.chan_seq)
    }
}

fn cspec(flags: u16, chan_seq: u64) -> CollectiveSpec {
    CollectiveSpec {
        wire_minor: WIRE_MINOR,
        flags: Flags(flags),
        instance: InstanceId(super::INSTANCE),
        epoch: epoch(),
        chan_seq,
    }
}

fn dtype_code(dtype: WireDType) -> u8 {
    match dtype {
        WireDType::F32 => 1,
        WireDType::BF16 => 2,
        other => panic!("no golden in {other:?}"),
    }
}

/// RFC-0031 §6.8's `phase` codes.
fn phase_code(phase: CollectivePhase) -> u8 {
    match phase {
        CollectivePhase::AttnAllReduce => 0,
        CollectivePhase::FfnAllReduce => 1,
        CollectivePhase::EntryBcast => 2,
        CollectivePhase::ExitGather => 3,
        CollectivePhase::RouterBcast => 4,
    }
}

/// RFC-0031 §6.8's `algo` codes.
fn algo_code(algo: WireAlgo) -> u8 {
    match algo {
        WireAlgo::OneShot => 0,
        WireAlgo::RingReduceScatter => 1,
        WireAlgo::RingAllGather => 2,
    }
}

fn weights(w: &[f32]) -> Weights<'static> {
    Weights::new(Vec::leak(w.to_vec()))
}

fn experts(e: &[u16]) -> Experts<'static> {
    Experts::new(Vec::leak(e.to_vec()))
}

type RouteRow = (Vec<f32>, Vec<u16>);
type DispatchRow = (u16, Vec<f32>, Vec<u16>);

fn router_rows(rows: &[RouteRow]) -> RouterRows<'static> {
    let typed: Vec<RouterRow<'static>> = rows
        .iter()
        .map(|(w, e)| RouterRow {
            weights: weights(w),
            experts: experts(e),
        })
        .collect();
    RouterRows::new(Vec::leak(typed))
}

fn ep_rows(rows: &[DispatchRow]) -> EpRows<'static> {
    let typed: Vec<EpRow<'static>> = rows
        .iter()
        .map(|(row, w, e)| EpRow {
            row: *row,
            weights: weights(w),
            experts: experts(e),
        })
        .collect();
    EpRows::new(Vec::leak(typed))
}

fn coll_header(
    layer: u16,
    phase: CollectivePhase,
    algo: WireAlgo,
    members: (u16, u16, u16),
    dtype: WireDType,
    elems: (u32, u32),
    rows: u32,
) -> CollectiveHeader {
    CollectiveHeader {
        step: STEP,
        layer,
        phase,
        algo,
        src_member: members.0,
        n_members: members.1,
        ring_step: members.2,
        dtype,
        elems: elems.0,
        elem_offset: elems.1,
        rows,
    }
}

/// A `Collective` golden: its wire values laid out by [`spec`], with `payload` the payload's
/// bytes as RFC-0031 §6.8 gives them.
fn collective(
    name: &'static str,
    about: &'static str,
    (flags, chan_seq): (u16, u64),
    h: CollectiveHeader,
    payload: Vec<u8>,
    typed: CollectivePayload<'static>,
) -> Golden {
    let ext = spec::collective_ext(&spec::CollectiveFields {
        step: h.step,
        layer: h.layer,
        phase: phase_code(h.phase),
        algo: algo_code(h.algo),
        src_member: h.src_member,
        n_members: h.n_members,
        ring_step: h.ring_step,
        dtype: dtype_code(h.dtype),
        elems: h.elems,
        elem_offset: h.elem_offset,
        rows: h.rows,
    });
    Golden {
        name,
        about,
        bytes: spec::frame(&base(0x08, flags, STEP, chan_seq), &ext, &payload),
        wire: h.dtype,
        flags,
        chan_seq,
        typed: Typed::Collective(Collective {
            header: h,
            payload: typed,
        }),
    }
}

fn ep_header(layer: u16, n_rows: u16, dtype: WireDType, part: (u16, u16)) -> EpHeader {
    EpHeader {
        step: STEP,
        layer,
        n_rows,
        dtype,
        part_index: part.0,
        part_count: part.1,
    }
}

fn ep_fields(h: &EpHeader) -> spec::EpFields {
    spec::EpFields {
        step: h.step,
        layer: h.layer,
        n_rows: h.n_rows,
        dtype: dtype_code(h.dtype),
        part_index: h.part_index,
        part_count: h.part_count,
    }
}

fn dispatch(
    name: &'static str,
    about: &'static str,
    chan_seq: u64,
    h: EpHeader,
    hidden: u32,
    rows: &[DispatchRow],
    payload: Vec<u8>,
) -> Golden {
    let ext = spec::ep_ext(&ep_fields(&h), rows);
    Golden {
        name,
        about,
        bytes: spec::frame(&base(0x0E, 0, STEP, chan_seq), &ext, &payload),
        wire: h.dtype,
        flags: 0,
        chan_seq,
        typed: Typed::Dispatch(EpDispatch {
            header: h,
            hidden,
            rows: ep_rows(rows),
            payload: leak(payload),
        }),
    }
}

fn combine(
    name: &'static str,
    about: &'static str,
    chan_seq: u64,
    h: EpHeader,
    hidden: u32,
    payload: Vec<u8>,
) -> Golden {
    let ext = spec::ep_ext(&ep_fields(&h), &[]);
    Golden {
        name,
        about,
        bytes: spec::frame(&base(0x0F, 0, STEP, chan_seq), &ext, &payload),
        wire: h.dtype,
        flags: 0,
        chan_seq,
        typed: Typed::Combine(EpCombine {
            header: h,
            hidden,
            payload: leak(payload),
        }),
    }
}

/// The routing rows of `ROUTER_BCAST`'s golden: k = 2, 1, 3 and 0, so every padding case of
/// the expert IDs occurs, and a row that routes nowhere.
fn router_golden_rows() -> Vec<RouteRow> {
    vec![
        (vec![0.75, 0.25], vec![3, 61]),
        (vec![1.0], vec![255]),
        (vec![0.5, 0.3, 0.2], vec![0, 1, 4095]),
        (vec![], vec![]),
    ]
}

/// The dispatch the part goldens split: 5 F32 rows of hidden 24 (96 bytes at a 128-byte
/// stride), routing to 1, 1, 2, 2 and 1 experts.
fn part_rows() -> Vec<DispatchRow> {
    vec![
        (0, vec![1.0], vec![4]),
        (1, vec![1.0], vec![4]),
        (3, vec![0.7, 0.3], vec![4, 8]),
        (4, vec![0.9, 0.1], vec![8, 4]),
        (6, vec![1.0], vec![8]),
    ]
}

/// The `max_frame` the dispatch part goldens are split under: part 0 takes 3 rows (512 bytes),
/// and a fourth would make it 704.
const PART_MAX_FRAME: usize = 512;

fn part_dispatch() -> EpDispatch<'static> {
    EpDispatch {
        header: ep_header(7, 5, WireDType::F32, (0, 1)),
        hidden: 24,
        rows: ep_rows(&part_rows()),
        payload: leak(spec::f32_rows(5, 24)),
    }
}

/// The step descriptor's `Activation`: a decode of 2 F32 rows of hidden 16, one span.
fn step_activation() -> (Activation<'static>, Vec<u8>) {
    let seq = SequenceId {
        ingress: 0x0102_0304_0506_0708,
        boot: 2,
        n: 5,
    };
    let span = Span {
        seq,
        first_position: 40,
        n_positions: 2,
        row_offset: 0,
        tree: None,
    };
    let header = ActivationHeader {
        from_stage: StageIx(1),
        to_stage: StageIx(2),
        kind: ActivationKind::Decode,
        layout: RowLayout {
            dtype: WireDType::F32,
            streams: 1,
            hidden: 16,
        },
        n_rows: 2,
        part_index: 0,
        part_count: 1,
    };
    let payload = spec::f32_rows(2, 16);
    let ext = spec::activation_ext(
        &spec::ActivationFixed {
            from_stage: 1,
            to_stage: 2,
            kind: 2,
            dtype: 1,
            streams: 1,
            hidden: 16,
            n_rows: 2,
            part_index: 0,
            part_count: 1,
            kind_arg: 0,
            kind_flags: 0,
        },
        &[spec::SpanFields {
            ingress: seq.ingress,
            boot: seq.boot,
            n: seq.n,
            first_position: 40,
            n_positions: 2,
            row_offset: 0,
            tree_offset: 0xFFFF,
            tree_nodes: 0,
        }],
        &[],
        &[],
    );
    let activation = Activation {
        header,
        spans: Spans::new(Vec::leak(vec![span])),
        tree: TreeArea::new(&[]),
        side: SideItems::new(&[]),
        payload: leak(payload.clone()),
    };
    let bytes = spec::frame(&base(0x09, 0, STEP, 11), &ext, &payload);
    (activation, bytes)
}

fn goldens() -> Vec<Golden> {
    use CollectivePhase::*;
    use WireAlgo::*;
    let f32 = WireDType::F32;
    let rr = router_golden_rows();
    let (activation, step_bytes) = step_activation();
    let mut all = vec![
        collective(
            "collective-attn-allreduce",
            "one-shot attention partial of member 1 of 2: 2 rows of hidden 16 in F32",
            (0, 7),
            coll_header(5, AttnAllReduce, OneShot, (1, 2, 0), f32, (32, 0), 2),
            spec::f32_elems(32, 1000),
            CollectivePayload::Elements(leak(spec::f32_elems(32, 1000))),
        ),
        collective(
            "collective-ffn-ring-rs",
            "ring reduce-scatter step 1 of member 2 of 4: a 24-element F32 slice at offset 24",
            (0, 8),
            coll_header(
                5,
                FfnAllReduce,
                RingReduceScatter,
                (2, 4, 1),
                f32,
                (24, 24),
                3,
            ),
            spec::f32_elems(24, 2024),
            CollectivePayload::Elements(leak(spec::f32_elems(24, 2024))),
        ),
        collective(
            "collective-exit-gather-bf16",
            "ring all-gather step 1 of member 1 of 3 on a BF16 channel: 16 elements at offset 32",
            (0, 9),
            coll_header(
                0,
                ExitGather,
                RingAllGather,
                (1, 3, 1),
                WireDType::BF16,
                (16, 32),
                1,
            ),
            spec::bf16_elems(16, 3000),
            CollectivePayload::Elements(leak(spec::bf16_elems(16, 3000))),
        ),
        collective(
            "collective-entry-bcast",
            "the primary's one-shot entry broadcast of 1 row of hidden 16, first of its sequence",
            (FIRST_OF_SEQ, 1),
            coll_header(0, EntryBcast, OneShot, (0, 2, 0), f32, (16, 0), 1),
            spec::f32_elems(16, 4000),
            CollectivePayload::Elements(leak(spec::f32_elems(16, 4000))),
        ),
        collective(
            "collective-router-bcast",
            "ROUTER_BCAST of layer 9 from the primary: 4 rows routing to 2, 1, 3 and 0 experts",
            (0, 10),
            coll_header(9, RouterBcast, OneShot, (0, 2, 0), f32, (6, 0), 4),
            spec::router_payload(&rr),
            CollectivePayload::Router(router_rows(&rr)),
        ),
        Golden {
            name: "step-descriptor",
            about: "a decode of 2 F32 rows of hidden 16 forwarded to a member as a StepDescriptor",
            bytes: step_bytes,
            wire: f32,
            flags: 0,
            chan_seq: 11,
            typed: Typed::StepDescriptor(activation),
        },
        dispatch(
            "ep-dispatch",
            "3 F32 rows of hidden 24 for one owner, routing to 1, 2 and 3 of its experts",
            12,
            ep_header(7, 3, f32, (0, 1)),
            24,
            &[
                (0, vec![1.0], vec![17]),
                (2, vec![0.6, 0.4], vec![3, 900]),
                (5, vec![0.5, 0.25, 0.25], vec![1, 2, 65535]),
            ],
            spec::f32_rows(3, 24),
        ),
        combine(
            "ep-combine",
            "an owner's weighted sums for 3 F32 rows of hidden 24",
            15,
            ep_header(7, 3, f32, (0, 1)),
            24,
            spec::f32_rows(3, 24),
        ),
        combine(
            "ep-combine-bf16-part1",
            "part 1 of 2 of a combine on a BF16 channel: 2 rows of hidden 40 at a 128-byte stride",
            16,
            ep_header(7, 2, WireDType::BF16, (1, 2)),
            40,
            spec::bf16_rows(2, 40),
        ),
    ];
    let rows = part_rows();
    let payload = spec::f32_rows(5, 24);
    all.push(dispatch(
        "ep-dispatch-part0",
        "part 0 of 2 of a 5-row F32 dispatch under a max_frame of 512: rows 0..3",
        13,
        ep_header(7, 3, f32, (0, 2)),
        24,
        &rows[..3],
        payload[..3 * 128].to_vec(),
    ));
    all.push(dispatch(
        "ep-dispatch-part1",
        "part 1 of 2 of the same dispatch: rows 3..5",
        14,
        ep_header(7, 2, f32, (1, 2)),
        24,
        &rows[3..],
        payload[3 * 128..].to_vec(),
    ));
    all
}

fn collective_dir() -> PathBuf {
    golden_dir().join("collective")
}

fn read_golden(name: &str) -> Vec<u8> {
    let path = collective_dir().join(format!("{name}.hex"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    spec::from_hex(&text)
}

#[test]
fn collective_golden_files_match_the_spec_tables() {
    let goldens = goldens();
    if std::env::var_os("SYNTROPY_BLESS_GOLDEN").is_some() {
        std::fs::create_dir_all(collective_dir()).unwrap();
        for golden in &goldens {
            let text = spec::to_hex(golden.name, golden.about, &golden.bytes);
            std::fs::write(collective_dir().join(format!("{}.hex", golden.name)), text).unwrap();
        }
    }
    for golden in &goldens {
        assert_eq!(
            read_golden(golden.name),
            golden.bytes,
            "{} differs from the tables",
            golden.name
        );
    }
    let listed: BTreeSet<String> = goldens.iter().map(|g| format!("{}.hex", g.name)).collect();
    let on_disk: BTreeSet<String> = std::fs::read_dir(collective_dir())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .filter(|name| name.ends_with(".hex"))
        .collect();
    assert_eq!(on_disk, listed);
}

/// Reads a golden frame as its type.
fn read(frame: &Frame<'static>, golden: &Golden) -> Result<Typed, FrameError> {
    Ok(match golden.typed {
        Typed::Collective(_) => Typed::Collective(frame.collective(golden.wire)?),
        Typed::StepDescriptor(_) => Typed::StepDescriptor(frame.step_descriptor(golden.wire)?),
        Typed::Dispatch(d) => Typed::Dispatch(frame.ep_dispatch(golden.wire, d.hidden)?),
        Typed::Combine(c) => Typed::Combine(frame.ep_combine(golden.wire, c.hidden)?),
    })
}

/// Encodes a golden's typed values into `out`.
fn write(golden: &Golden, out: &mut [u8]) -> Result<usize, FrameError> {
    let spec = golden.spec();
    match golden.typed {
        Typed::Collective(c) => c.encode(&spec, out),
        Typed::StepDescriptor(a) => {
            let spec = ActivationSpec {
                wire_minor: WIRE_MINOR,
                flags: Flags(golden.flags),
                instance: spec.instance,
                epoch: spec.epoch,
                step: STEP,
                chan_seq: golden.chan_seq,
            };
            a.encode_step_descriptor(&spec, out)
        }
        Typed::Dispatch(d) => d.encode(&spec, out),
        Typed::Combine(c) => c.encode(&spec, out),
    }
}

fn same(a: &Typed, b: &Typed) -> bool {
    match (a, b) {
        (Typed::Collective(a), Typed::Collective(b)) => a == b,
        (Typed::StepDescriptor(a), Typed::StepDescriptor(b)) => a == b,
        (Typed::Dispatch(a), Typed::Dispatch(b)) => a == b,
        (Typed::Combine(a), Typed::Combine(b)) => a == b,
        _ => false,
    }
}

/// The criterion "golden frames round-trip": each file decodes to its typed values, and those
/// encode to the same bytes, whole and in place in a larger slot. The goldens cover the four
/// frame types, every phase, every algorithm step, and both dtypes.
#[test]
fn collective_goldens_round_trip() {
    let mut ftypes = BTreeSet::new();
    let mut phases = BTreeSet::new();
    let mut algos = BTreeSet::new();
    for golden in goldens() {
        let name = golden.name;
        let bytes = leak(read_golden(name));
        let frame = Frame::decode(bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        ftypes.insert(frame.header.spec.ftype.code());
        assert_eq!(frame.header.spec.step, STEP, "{name}");
        assert_eq!(frame.header.spec.chan_seq, golden.chan_seq, "{name}");
        let decoded = read(&frame, &golden).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(
            same(&decoded, &golden.typed),
            "{name}: {decoded:?} != {:?}",
            golden.typed
        );
        let mut slot = vec![0xAA; bytes.len() + 100];
        assert_eq!(write(&golden, &mut slot), Ok(bytes.len()), "{name}");
        assert_eq!(&slot[..bytes.len()], bytes, "{name}");
        match golden.typed {
            Typed::Collective(c) => {
                assert_eq!(
                    c.encode_to_vec(&golden.spec()).as_deref(),
                    Ok(bytes),
                    "{name}"
                );
                assert_eq!(c.encoded_len(), Ok(bytes.len()), "{name}");
                let mut ext = vec![0xAA; 100];
                assert_eq!(c.write_ext(&mut ext), Ok(64), "{name}");
                assert_eq!(&ext[..64], frame.ext, "{name}");
                phases.insert(phase_code(c.header.phase));
                algos.insert(algo_code(c.header.algo));
            }
            Typed::Dispatch(d) => {
                assert_eq!(
                    d.encode_to_vec(&golden.spec()).as_deref(),
                    Ok(bytes),
                    "{name}"
                );
                assert_eq!(d.encoded_len(), Ok(bytes.len()), "{name}");
            }
            Typed::Combine(c) => {
                assert_eq!(
                    c.encode_to_vec(&golden.spec()).as_deref(),
                    Ok(bytes),
                    "{name}"
                );
                assert_eq!(c.encoded_len(), Ok(bytes.len()), "{name}");
            }
            Typed::StepDescriptor(_) => {}
        }
    }
    assert_eq!(ftypes, BTreeSet::from([0x08, 0x09, 0x0E, 0x0F]));
    assert_eq!(phases, BTreeSet::from([0, 1, 2, 3, 4]));
    assert_eq!(algos, BTreeSet::from([0, 1, 2]));
}

/// The criterion "the `ROUTER_BCAST` payload decodes per row with k experts and weights".
#[test]
fn router_bcast_decodes_per_row_with_k_experts_and_weights() {
    let bytes = read_golden("collective-router-bcast");
    let frame = Frame::decode(&bytes).unwrap();
    let collective = frame.collective(WireDType::F32).unwrap();
    assert_eq!(collective.header.phase, CollectivePhase::RouterBcast);
    let CollectivePayload::Router(rows) = collective.payload else {
        panic!("not routing rows: {:?}", collective.payload);
    };
    assert_eq!(rows.len(), 4);
    let expected = router_golden_rows();
    let mut named = 0;
    for (row, (w, e)) in rows.iter().zip(&expected) {
        assert_eq!(row.k(), e.len());
        assert_eq!(row.weights.len(), row.k());
        let read_w: Vec<f32> = row.weights.iter().collect();
        let read_e: Vec<u16> = row.experts.iter().collect();
        assert_eq!(&read_w, w);
        assert_eq!(&read_e, e);
        named += row.k();
    }
    assert_eq!(collective.header.elems as usize, named);
    // Each row is its head, its weights, then its expert IDs padded to 4: 16 + 12 + 24 + 4.
    assert_eq!(frame.payload.len(), 56);
    assert_eq!(frame.payload[0], 2);
    assert_eq!(&frame.payload[4..8], &0.75f32.to_le_bytes());
    assert_eq!(&frame.payload[12..14], &3u16.to_le_bytes());
    // The weights of every row start 4-byte aligned in the payload, as RFC-0031 §6.8 requires.
    let mut at = 0;
    for (w, _) in &expected {
        assert_eq!((at + 4) % 4, 0);
        at += route_row_len(w.len());
    }
    assert_eq!(at, frame.payload.len());
}

fn activation_goldens() -> Vec<(String, Vec<u8>)> {
    let dir = golden_dir().join("activation");
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "hex"))
        .collect();
    files.sort();
    files
        .into_iter()
        .map(|p| {
            let text = std::fs::read_to_string(&p).unwrap();
            (p.display().to_string(), spec::from_hex(&text))
        })
        .collect()
}

/// RFC-0032 §5.2 step 1: the primary forwards the inbound `Activation` unchanged as a
/// `StepDescriptor`. Every `Activation` golden forwards with its extension and payload bytes
/// untouched, and reads back as the same `Activation`.
#[test]
fn a_step_descriptor_forwards_an_activation_unchanged() {
    let goldens = activation_goldens();
    assert!(goldens.len() >= 8);
    for (name, bytes) in goldens {
        let frame = Frame::decode(&bytes).unwrap();
        let wire = match frame.ext[5] {
            1 => WireDType::F32,
            2 => WireDType::BF16,
            other => panic!("{name}: dtype {other}"),
        };
        let activation = frame.activation(wire).unwrap();
        let mut out = vec![0xAA; bytes.len() + 64];
        let len = frame
            .forward_as_step_descriptor(wire, 99, &mut out)
            .unwrap();
        assert_eq!(len, bytes.len(), "{name}");
        let forwarded = Frame::decode(&out[..len]).unwrap();
        let (a, f) = (frame.header.spec, forwarded.header.spec);
        assert_eq!(f.ftype, FrameType::StepDescriptor, "{name}");
        assert_eq!(f.chan_seq, 99, "{name}");
        assert_eq!(
            (f.wire_minor, f.flags, f.instance, f.epoch, f.step),
            (a.wire_minor, a.flags, a.instance, a.epoch, a.step),
            "{name}"
        );
        assert_eq!(forwarded.ext, frame.ext, "{name}");
        assert_eq!(forwarded.payload, frame.payload, "{name}");
        assert_eq!(forwarded.step_descriptor(wire), Ok(activation), "{name}");
        // Each type is read only as itself.
        assert_eq!(
            forwarded.activation(wire),
            Err(FrameError::NotActivation(FrameType::StepDescriptor))
        );
        assert_eq!(
            frame.step_descriptor(wire),
            Err(FrameError::WrongType {
                expected: FrameType::StepDescriptor,
                found: FrameType::Activation,
            })
        );
    }
    // The step-descriptor golden is the forward of the same frame sent as an `Activation`.
    let (activation, golden) = step_activation();
    let inbound = activation
        .encode_to_vec(&ActivationSpec {
            wire_minor: WIRE_MINOR,
            flags: Flags(0),
            instance: InstanceId(super::INSTANCE),
            epoch: epoch(),
            step: STEP,
            chan_seq: 3,
        })
        .unwrap();
    let inbound = Frame::decode(&inbound).unwrap();
    let mut out = vec![0; golden.len()];
    inbound
        .forward_as_step_descriptor(WireDType::F32, 11, &mut out)
        .unwrap();
    assert_eq!(out, golden);
    assert_eq!(read_golden("step-descriptor"), golden);
    // A malformed inbound frame is refused, not forwarded; and the descriptor's refusals name it.
    assert_eq!(
        inbound.forward_as_step_descriptor(WireDType::BF16, 11, &mut out),
        Err(FrameError::WireDTypeMismatch {
            expected: WireDType::BF16,
            found: WireDType::F32,
        })
    );
    let mut bad = golden.clone();
    bad[64 + 6] = 0; // streams 0
    reseal(&mut bad);
    assert_eq!(
        Frame::decode(&bad).unwrap().step_descriptor(WireDType::F32),
        Err(FrameError::BadField {
            ftype: FrameType::StepDescriptor,
            field: "streams",
            value: 0,
        })
    );
}

/// Encodes every part of `message` and decodes each back.
fn dispatch_parts(
    message: &EpDispatch<'_>,
    max_frame: usize,
) -> Result<Vec<(Part, Vec<u8>)>, FrameError> {
    let mut frames = Vec::new();
    for part in message.parts(max_frame)? {
        let mut out = vec![0; part.frame_len];
        let len = message.encode_part(&part, &cspec(0, 20 + u64::from(part.index)), &mut out)?;
        assert_eq!(len, part.frame_len);
        frames.push((part, out));
    }
    Ok(frames)
}

/// The parts of a dispatch: each fits `max_frame`, and with one more row would not; they cover
/// the rows in order; and each decodes to its rows' routing and hidden states.
fn check_dispatch_parts(message: &EpDispatch<'_>, frames: &[(Part, Vec<u8>)], max_frame: usize) {
    let stride = message.payload.len() / usize::from(message.header.n_rows);
    let all: Vec<EpRow<'_>> = message.rows.iter().collect();
    let mut next = 0u32;
    for (part, bytes) in frames {
        assert_eq!(part.count as usize, frames.len());
        assert_eq!(part.rows.start, next);
        next = part.rows.end;
        assert!(bytes.len() <= max_frame);
        let frame = Frame::decode(bytes).unwrap();
        let decoded = frame
            .ep_dispatch(message.header.dtype, message.hidden)
            .unwrap();
        let (start, end) = (part.rows.start as usize, part.rows.end as usize);
        assert_eq!(decoded.header.part_index, part.index);
        assert_eq!(decoded.header.part_count, part.count);
        assert_eq!(usize::from(decoded.header.n_rows), end - start);
        assert!(decoded.rows.iter().eq(all[start..end].iter().copied()));
        assert_eq!(
            decoded.payload,
            &message.payload[start * stride..end * stride]
        );
        if end < all.len() {
            // One more row would not fit: the part holds as many rows as fit.
            let meta: usize = all[start..=end]
                .iter()
                .map(|r| route_row_len(r.k_local()))
                .sum();
            let longer = 64 + (24 + meta).next_multiple_of(64) + (end - start + 1) * stride;
            assert!(
                longer > max_frame,
                "part {} could take another row",
                part.index
            );
        }
    }
    assert_eq!(next, u32::from(message.header.n_rows));
}

/// `n` rows of `k_local` experts each, row `r` routing to experts `r..r + k_local` with equal
/// weights.
fn uniform_rows(n: usize, k_local: u16) -> Vec<DispatchRow> {
    (0..n)
        .map(|r| {
            let w = vec![1.0 / f32::from(k_local); usize::from(k_local)];
            let e = (0..k_local).map(|i| r as u16 + i).collect();
            (r as u16, w, e)
        })
        .collect()
}

/// The criterion "at hidden 7168 in F32 under the TB5 message limit, dispatch parts hold 584
/// rows at `k_local = 1` and 583 at `k_local = 8` (RFC-0031 §6.8 derivation)": a 600-row
/// dispatch is split by the codec, and the closed form agrees.
#[test]
fn dispatch_parts_hold_584_and_583_rows_at_hidden_7168_under_the_tb5_limit() {
    let payload = vec![0u8; 600 * 7168 * 4];
    for (k_local, rows_per_part) in [(1u16, 584u32), (8, 583)] {
        let rows = uniform_rows(600, k_local);
        let message = EpDispatch {
            header: ep_header(3, 600, WireDType::F32, (0, 1)),
            hidden: 7168,
            rows: ep_rows(&rows),
            payload: &payload,
        };
        let parts: Vec<Part> = message.parts(TB5).unwrap().collect();
        let ranges: Vec<_> = parts.iter().map(|p| p.rows.clone()).collect();
        assert_eq!(
            ranges,
            [0..rows_per_part, rows_per_part..600],
            "k_local {k_local}"
        );
        assert_eq!(
            EpDispatch::max_part_rows(TB5, WireDType::F32, 7168, k_local as u8),
            Ok(rows_per_part)
        );
        // The derivation's frame lengths, from the header area the plan sizes.
        let area = EpDispatch::header_area(rows_per_part, k_local as u8);
        assert_eq!(parts[0].frame_len, area + rows_per_part as usize * 28_672);
        let over = EpDispatch::header_area(rows_per_part + 1, k_local as u8)
            + (rows_per_part as usize + 1) * 28_672;
        assert!(parts[0].frame_len <= TB5 && over > TB5);
        let frames = dispatch_parts(&message, TB5).unwrap();
        check_dispatch_parts(&message, &frames, TB5);
    }
    // RFC-0031 §6.8's bracket at k_local = 8: 583 rows take 16,746,240 bytes, 584 take
    // 16,774,912.
    assert_eq!(EpDispatch::header_area(583, 8) + 583 * 28_672, 16_746_240);
    assert_eq!(EpDispatch::header_area(584, 8) + 584 * 28_672, 16_774_912);
    // A combine part carries no routing rows: 584 rows of 28,672 bytes fit with its 128-byte
    // header area, and 585 do not.
    assert_eq!(EpCombine::HEADER_AREA, 128);
    assert_eq!(EpCombine::max_part_rows(TB5, WireDType::F32, 7168), Ok(584));
}

/// The header slot a plan sizes for EP rows (RFC-0031 §8): the base header and the extension at
/// the plan's rows and `k_local`, padded to 64, the same length the tables' extension takes.
#[test]
fn the_header_area_is_sized_from_the_plan() {
    for (rows, k_local) in [(1u32, 1u8), (512, 1), (512, 8), (583, 8), (40, 255)] {
        let fields = spec::EpFields {
            step: 0,
            layer: 0,
            n_rows: rows as u16,
            dtype: 1,
            part_index: 0,
            part_count: 1,
        };
        let ext = spec::ep_ext(&fields, &uniform_rows(rows as usize, k_local.into()));
        assert_eq!(
            EpDispatch::header_area(rows, k_local),
            64 + ext.len(),
            "{rows} rows at k_local {k_local}"
        );
    }
    // 512 rows at k_local 8: 24 + 512 × 52 = 26,648 bytes of extension, padded to 26,688.
    assert_eq!(EpDispatch::header_area(512, 8), 64 + 26_688);
}

#[test]
fn a_dispatch_splits_into_the_part_goldens() {
    let message = part_dispatch();
    let frames = dispatch_parts(&message, PART_MAX_FRAME).unwrap();
    let parts: Vec<_> = frames
        .iter()
        .map(|(p, _)| (p.index, p.count, p.rows.clone(), p.frame_len))
        .collect();
    assert_eq!(parts, [(0, 2, 0..3, 512), (1, 2, 3..5, 384)]);
    let golden: Vec<_> = goldens()
        .into_iter()
        .filter(|g| g.name.starts_with("ep-dispatch-part"))
        .collect();
    for ((part, bytes), golden) in frames.iter().zip(&golden) {
        let mut want = golden.bytes.clone();
        // The goldens' chan_seq is 13 and 14; the parts here use 20 and 21.
        want[40..48].copy_from_slice(&(20 + u64::from(part.index)).to_le_bytes());
        reseal(&mut want);
        assert_eq!(bytes, &want, "part {}", part.index);
        assert_eq!(read_golden(golden.name), golden.bytes);
    }
    check_dispatch_parts(&message, &frames, PART_MAX_FRAME);
    // Under every limit from one row up, the parts are maximal and rejoin into the dispatch.
    for max_frame in 256..1400 {
        let frames = dispatch_parts(&message, max_frame).unwrap();
        check_dispatch_parts(&message, &frames, max_frame);
    }
    // A dispatch that fits is one part, byte for byte the whole frame.
    let whole = message.encode_to_vec(&cspec(0, 20)).unwrap();
    let frames = dispatch_parts(&message, whole.len()).unwrap();
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].1, whole);
    // One row and its routing need 256 bytes; below that nothing fits. A sender's failure, so
    // `Internal`, with the `FrameTooLarge` code.
    let error = message.parts(255).unwrap_err();
    assert_eq!(
        error,
        FrameError::TooLong {
            what: "EpDispatch part",
            len: 256
        }
    );
    assert_eq!(error.class(), ErrorClass::Internal);
    assert_eq!(error.code(), Some(code::FRAME_TOO_LARGE));
    // Only a whole dispatch is split, and only its own parts are encoded.
    let partial = EpDispatch {
        header: ep_header(7, 5, WireDType::F32, (0, 2)),
        ..message
    };
    assert!(matches!(
        partial.parts(PART_MAX_FRAME),
        Err(FrameError::BadField {
            field: "part_count",
            ..
        })
    ));
    let mut out = vec![0; 2048];
    let stray = Part {
        index: 0,
        count: 1,
        rows: 4..9,
        frame_len: 0,
    };
    assert!(matches!(
        message.encode_part(&stray, &cspec(0, 1), &mut out),
        Err(FrameError::BadField { field: "part", .. })
    ));
}

#[test]
fn a_combine_splits_into_parts_that_rejoin() {
    let payload = spec::f32_rows(5, 24);
    let message = EpCombine {
        header: ep_header(7, 5, WireDType::F32, (0, 1)),
        hidden: 24,
        payload: &payload,
    };
    for max_frame in [256usize, 300, 512, 640, 768, 1000] {
        let mut next = 0;
        let mut joined = Vec::new();
        let parts: Vec<Part> = message.parts(max_frame).unwrap().collect();
        for part in &parts {
            assert_eq!(part.rows.start, next);
            next = part.rows.end;
            let rows = (part.rows.end - part.rows.start) as usize;
            assert_eq!(part.frame_len, 128 + rows * 128);
            assert!(part.frame_len <= max_frame);
            assert!(part.rows.end == 5 || part.frame_len + 128 > max_frame);
            let mut out = vec![0; part.frame_len];
            message.encode_part(part, &cspec(0, 1), &mut out).unwrap();
            let frame = Frame::decode(&out).unwrap();
            let decoded = frame.ep_combine(WireDType::F32, 24).unwrap();
            assert_eq!(
                (decoded.header.part_index, decoded.header.part_count),
                (part.index, parts.len() as u16)
            );
            joined.extend_from_slice(decoded.payload);
        }
        assert_eq!(next, 5);
        assert_eq!(joined, payload);
    }
    assert_eq!(
        message.parts(255).unwrap_err(),
        FrameError::TooLong {
            what: "EpCombine part",
            len: 256
        }
    );
}

/// A golden with `edit` applied to its bytes and its CRCs recomputed, read as its type.
fn refused(name: &str, edit: impl Fn(&mut Vec<u8>)) -> FrameError {
    let golden = goldens().into_iter().find(|g| g.name == name).unwrap();
    let mut bytes = read_golden(name);
    edit(&mut bytes);
    reseal(&mut bytes);
    let bytes = leak(bytes);
    let frame = Frame::decode(bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
    let error = read(&frame, &golden).map(|t| format!("{t:?}")).unwrap_err();
    assert!(
        REFUSAL_CLASSES.contains(&error.class()),
        "{error}: {:?}",
        error.class()
    );
    error
}

/// An edit to a golden's bytes.
type Edit = Box<dyn Fn(&mut Vec<u8>)>;

fn set16(b: &mut [u8], at: usize, v: u16) {
    b[at..at + 2].copy_from_slice(&v.to_le_bytes());
}

fn set32(b: &mut [u8], at: usize, v: u32) {
    b[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

/// A frame that breaks a rule of RFC-0031 §6.4 or §6.8 is refused with the error that names
/// it, in a class RFC-0060 §10.2 allows. Extension offsets are from the frame's start: the
/// extension begins at 64.
#[test]
fn a_frame_breaking_a_rule_is_refused_with_a_classified_error() {
    use FrameType::{Collective as C, EpCombine as EC, EpDispatch as ED};
    let x = 64;
    let bad = |ftype, field, value| FrameError::BadField {
        ftype,
        field,
        value,
    };
    let inc = |ftype, field, value, expected| FrameError::Inconsistent {
        ftype,
        field,
        value,
        expected,
    };
    let cases: Vec<(&str, Edit, FrameError)> = vec![
        (
            "collective-attn-allreduce",
            Box::new(move |b| b[x + 10] = 5),
            bad(C, "phase", 5),
        ),
        (
            "collective-attn-allreduce",
            Box::new(move |b| b[x + 11] = 3),
            bad(C, "algo", 3),
        ),
        (
            "collective-attn-allreduce",
            Box::new(move |b| b[x + 18] = 9),
            bad(C, "dtype", 9),
        ),
        (
            "collective-attn-allreduce",
            Box::new(move |b| b[x + 18] = 2),
            FrameError::WireDTypeMismatch {
                expected: WireDType::F32,
                found: WireDType::BF16,
            },
        ),
        (
            "collective-attn-allreduce",
            Box::new(move |b| set16(b, x + 14, 1)),
            bad(C, "n_members", 1),
        ),
        (
            "collective-attn-allreduce",
            Box::new(move |b| set16(b, x + 12, 2)),
            inc(C, "src_member", 2, 2),
        ),
        (
            "collective-attn-allreduce",
            Box::new(move |b| set16(b, x + 16, 1)),
            bad(C, "ring_step", 1),
        ),
        (
            "collective-attn-allreduce",
            Box::new(move |b| set32(b, x + 20, 31)),
            inc(C, "payload_len", 128, 124),
        ),
        (
            "collective-attn-allreduce",
            Box::new(|b| set16(b, 6, HAS_TRACE)),
            inc(C, "HAS_TRACE", 1, 0),
        ),
        (
            "collective-attn-allreduce",
            Box::new(move |b| b[x] ^= 1),
            inc(C, "step", STEP ^ 1, STEP),
        ),
        (
            "collective-router-bcast",
            Box::new(move |b| b[x + 11] = 1),
            bad(C, "algo", 1),
        ),
        (
            "collective-router-bcast",
            Box::new(move |b| set16(b, x + 12, 1)),
            inc(C, "src_member", 1, 0),
        ),
        (
            "collective-router-bcast",
            Box::new(move |b| set32(b, x + 24, 4)),
            bad(C, "elem_offset", 4),
        ),
        (
            "collective-router-bcast",
            Box::new(move |b| set32(b, x + 20, 7)),
            inc(C, "elems", 7, 6),
        ),
        (
            "collective-router-bcast",
            Box::new(move |b| set32(b, x + 28, 3)),
            inc(C, "payload_len", 56, 52),
        ),
        (
            "collective-router-bcast",
            Box::new(move |b| set32(b, x + 28, 5)),
            inc(C, "payload_len", 56, 60),
        ),
        // Row 0's k raised to 3: row 1 is then read from byte 24, inside the old row 1, where
        // expert 255's low byte gives k = 255.
        (
            "collective-router-bcast",
            Box::new(move |b| b[x + 64] = 3),
            inc(C, "payload_len", 56, 24 + route_row_len(255) as u64),
        ),
        (
            "ep-dispatch",
            Box::new(move |b| b[x + 12] = 3),
            FrameError::WireDTypeMismatch {
                expected: WireDType::F32,
                found: WireDType::F16,
            },
        ),
        (
            "ep-dispatch",
            Box::new(move |b| set16(b, x + 16, 0)),
            bad(ED, "part_count", 0),
        ),
        (
            "ep-dispatch",
            Box::new(move |b| set16(b, x + 14, 1)),
            inc(ED, "part_index", 1, 1),
        ),
        (
            "ep-dispatch",
            Box::new(move |b| set16(b, x + 10, 0)),
            bad(ED, "n_rows", 0),
        ),
        (
            "ep-dispatch",
            Box::new(move |b| set16(b, x + 10, 2)),
            inc(ED, "payload_len", 384, 256),
        ),
        // The last row's k_local set to 0: rows 0 and 1 take 12 + 16 bytes before it.
        (
            "ep-dispatch",
            Box::new(move |b| b[x + 24 + 28 + 2] = 0),
            bad(ED, "k_local", 0),
        ),
        // More routing rows than the extension holds.
        (
            "ep-dispatch",
            Box::new(move |b| {
                set16(b, x + 10, 6);
                b[x + 24 + 2] = 40;
            }),
            FrameError::ExtTooShort {
                ftype: ED,
                len: 128,
                needed: 24 + route_row_len(40),
            },
        ),
        (
            "ep-dispatch",
            Box::new(move |b| b[x + 1] ^= 1),
            inc(ED, "step", STEP ^ 256, STEP),
        ),
        (
            "ep-combine",
            Box::new(move |b| set16(b, x + 10, 4)),
            inc(EC, "payload_len", 384, 512),
        ),
    ];
    for (name, edit, expected) in cases {
        let error = refused(name, edit);
        assert_eq!(error, expected, "{name}");
    }
    // A combine relabelled as a dispatch is read only as a dispatch, and its zeroed extension
    // then gives rows that route nowhere.
    let mut bytes = read_golden("ep-combine");
    bytes[5] = 0x0E;
    reseal(&mut bytes);
    let frame = Frame::decode(&bytes).unwrap();
    assert_eq!(
        frame.ep_combine(WireDType::F32, 24),
        Err(FrameError::WrongType {
            expected: EC,
            found: ED
        })
    );
    assert_eq!(
        frame.ep_dispatch(WireDType::F32, 24),
        Err(bad(ED, "k_local", 0))
    );
    // A dtype this build does not carry, on a channel of that dtype.
    let mut bytes = read_golden("ep-dispatch");
    bytes[64 + 12] = 3;
    reseal(&mut bytes);
    let error = Frame::decode(&bytes)
        .unwrap()
        .ep_dispatch(WireDType::F16, 24)
        .unwrap_err();
    assert_eq!(error, FrameError::UnsupportedDType(WireDType::F16));
    assert_eq!(error.class(), ErrorClass::Unsupported);
    // A dtype mismatch is `Unsupported`, with `WireDTypeMismatch`'s Close code.
    let error = refused("collective-attn-allreduce", |b| b[64 + 18] = 2);
    assert_eq!(error.class(), ErrorClass::Unsupported);
    assert_eq!(error.code(), Some(code::WIRE_DTYPE_MISMATCH));
    // A sealed frame is opened by the AEAD layer first.
    let mut bytes = read_golden("ep-dispatch");
    set16(&mut bytes, 6, ENCRYPTED);
    reseal(&mut bytes);
    assert_eq!(
        Frame::decode(&bytes)
            .unwrap()
            .ep_dispatch(WireDType::F32, 24),
        Err(FrameError::Sealed(ED))
    );
    // `hidden` comes from the plan; a payload of other rows is inconsistent with it.
    let bytes = read_golden("ep-dispatch");
    let frame = Frame::decode(&bytes).unwrap();
    assert_eq!(
        frame.ep_dispatch(WireDType::F32, 40),
        Err(inc(ED, "payload_len", 384, 3 * 192))
    );
    assert_eq!(
        frame.ep_dispatch(WireDType::F32, 0),
        Err(bad(ED, "hidden", 0))
    );
    // An encoder never writes what the decoder refuses.
    let golden = goldens()
        .into_iter()
        .find(|g| g.name == "ep-dispatch")
        .unwrap();
    let Typed::Dispatch(d) = golden.typed else {
        unreachable!()
    };
    let lopsided = [EpRow {
        row: 0,
        weights: weights(&[1.0, 0.0]),
        experts: experts(&[1]),
    }];
    let broken = EpDispatch {
        header: ep_header(7, 1, WireDType::F32, (0, 1)),
        rows: EpRows::new(&lopsided),
        payload: &d.payload[..128],
        ..d
    };
    assert_eq!(
        broken.encode_to_vec(&cspec(0, 1)),
        Err(inc(ED, "weights", 2, 1))
    );
    assert!(matches!(
        d.encode_to_vec(&cspec(ENCRYPTED, 1)),
        Err(FrameError::BadField { field: "flags", .. })
    ));
}

/// The criterion behind "golden frames round-trip": a corrupted byte anywhere in these frames is
/// caught by the base header's CRCs, as in every other frame.
#[test]
fn a_corrupted_byte_anywhere_in_the_collective_goldens_is_detected() {
    for golden in goldens() {
        let bytes = read_golden(golden.name);
        for at in 0..bytes.len() {
            for mask in [1u8, 0x80, 0xFF] {
                let mut corrupt = bytes.clone();
                corrupt[at] ^= mask;
                assert!(
                    Frame::decode(&corrupt).is_err(),
                    "{}: byte {at} ^ {mask:#04x} was accepted",
                    golden.name
                );
            }
        }
    }
}

fn wire_of(bf16: bool) -> WireDType {
    if bf16 {
        WireDType::BF16
    } else {
        WireDType::F32
    }
}

proptest! {
    /// The `Collective` parser on arbitrary extensions and payloads: no panic, every refusal is
    /// classified, and an accepted frame encodes to one that decodes to it again.
    #[test]
    fn arbitrary_collective_bodies_parse_safely(
        ext in proptest::collection::vec(any::<u8>(), 0..96),
        payload in proptest::collection::vec(any::<u8>(), 0..200),
        bf16 in any::<bool>(),
    ) {
        match Collective::parse(&ext, &payload, wire_of(bf16)) {
            Err(error) => {
                prop_assert!(REFUSAL_CLASSES.contains(&error.class()), "{error}: {:?}", error.class());
            }
            Ok(collective) => {
                let spec = cspec(0, 1);
                let bytes = collective.encode_to_vec(&spec).unwrap();
                let frame = Frame::decode(&bytes).unwrap();
                prop_assert_eq!(frame.collective(wire_of(bf16)), Ok(collective));
            }
        }
    }

    /// The EP parsers on arbitrary extensions and payloads, with the same properties.
    #[test]
    fn arbitrary_ep_bodies_parse_safely(
        ext in proptest::collection::vec(any::<u8>(), 0..160),
        payload in proptest::collection::vec(any::<u8>(), 0..400),
        bf16 in any::<bool>(),
        hidden in 0u32..48,
    ) {
        let wire = wire_of(bf16);
        match EpDispatch::parse(&ext, &payload, wire, hidden) {
            Err(error) => {
                prop_assert!(REFUSAL_CLASSES.contains(&error.class()), "{error}: {:?}", error.class());
            }
            Ok(dispatch) => {
                let bytes = dispatch.encode_to_vec(&cspec(0, 1)).unwrap();
                let frame = Frame::decode(&bytes).unwrap();
                prop_assert_eq!(frame.ep_dispatch(wire, hidden), Ok(dispatch));
            }
        }
        match EpCombine::parse(&ext, &payload, wire, hidden) {
            Err(error) => {
                prop_assert!(REFUSAL_CLASSES.contains(&error.class()), "{error}: {:?}", error.class());
            }
            Ok(combine) => {
                let bytes = combine.encode_to_vec(&cspec(0, 1)).unwrap();
                let frame = Frame::decode(&bytes).unwrap();
                prop_assert_eq!(frame.ep_combine(wire, hidden), Ok(combine));
            }
        }
    }

    /// Any dispatch splits under any limit that fits one row into maximal parts that rejoin.
    #[test]
    fn dispatches_split_into_maximal_parts(
        ks in proptest::collection::vec(1u16..12, 1..40),
        hidden in 1u32..64,
        bf16 in any::<bool>(),
        slack in 0usize..2000,
    ) {
        let dtype = wire_of(bf16);
        let rows: Vec<DispatchRow> = ks
            .iter()
            .enumerate()
            .map(|(r, &k)| (r as u16, vec![0.5; usize::from(k)], (0..k).collect()))
            .collect();
        let stride = (hidden as usize * if bf16 { 2 } else { 4 }).next_multiple_of(64);
        let payload: Vec<u8> = (0..rows.len() * stride).map(|i| i as u8).collect();
        let message = EpDispatch {
            header: ep_header(1, rows.len() as u16, dtype, (0, 1)),
            hidden,
            rows: ep_rows(&rows),
            payload: &payload,
        };
        let widest = rows.iter().map(|(_, w, _)| route_row_len(w.len())).max().unwrap();
        let max_frame = 64 + (24 + widest).next_multiple_of(64) + stride + slack;
        let frames = dispatch_parts(&message, max_frame).unwrap();
        check_dispatch_parts(&message, &frames, max_frame);
    }
}
