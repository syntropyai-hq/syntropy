// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `Activation` codec against the golden frames in `tests/golden/activation/` (DIST-E8-T02;
//! RFC-0031 §6.3, §6.4; ADR-0010 item 5; ADR-0019 item 5).
//!
//! Each golden frame is written twice: as wire values, which [`spec`] lays out from RFC-0031
//! §6.3's tables, and as the codec's typed values. The tests check that the files match the
//! tables, that the codec decodes each file to its typed values and encodes those to the same
//! bytes, that a chunk splits into parts that each fit `max_frame` and join back into the chunk,
//! and that a frame breaking a rule is refused with a classified error, never `Internal`
//! (RFC-0060 §10.2).
//!
//! `SYNTROPY_BLESS_GOLDEN=1` rewrites the files, as it does the frames of DIST-E8-T01.

use std::collections::BTreeSet;
use std::path::PathBuf;

use proptest::prelude::*;
use syntropy_core::plan::WireDType;
use syntropy_core::{ErrorClass, InstanceId, SequenceId, SessionId, StageIx};
use syntropy_net::frame::{
    Activation, ActivationHeader, ActivationKind, ActivationSpec, Flags, Frame, FrameError,
    FrameType, Part, RowLayout, SeqCtrl, Side, SideItem, SideItems, Span, SpanTree, Spans,
    TraceContext, TreeArea, TreeNode, WIRE_MINOR, code, tlv,
};

use super::{INSTANCE, REFUSAL_CLASSES, base, epoch, golden_dir, leak, reseal, spec};

/// The sequences the goldens carry.
const Q: [SequenceId; 4] = [
    SequenceId {
        ingress: 0x0102_0304_0506_0708,
        boot: 2,
        n: 77,
    },
    SequenceId {
        ingress: 0x0102_0304_0506_0708,
        boot: 2,
        n: 78,
    },
    SequenceId {
        ingress: 0x0102_0304_0506_0708,
        boot: 2,
        n: 79,
    },
    SequenceId {
        ingress: 0x0102_0304_0506_0708,
        boot: 2,
        n: 80,
    },
];

const FIRST_OF_SEQ: u16 = 1 << 3;
const HAS_TRACE: u16 = 1;

/// A golden frame: the bytes, and the typed values and header fields they encode.
struct Golden {
    name: &'static str,
    about: &'static str,
    bytes: Vec<u8>,
    /// The channel's wire dtype.
    wire: WireDType,
    spec: ActivationSpec,
    header: ActivationHeader,
    spans: Vec<Span>,
    tree: Vec<TreeNode>,
    side: Vec<Side<'static>>,
    payload: Vec<u8>,
}

impl Golden {
    fn activation(&self) -> Activation<'_> {
        Activation {
            header: self.header,
            spans: Spans::new(&self.spans),
            tree: TreeArea::new(&self.tree),
            side: SideItems::new(&self.side),
            payload: &self.payload,
        }
    }
}

fn act_spec(flags: u16, step: u64, chan_seq: u64) -> ActivationSpec {
    ActivationSpec {
        wire_minor: WIRE_MINOR,
        flags: Flags(flags),
        instance: InstanceId(INSTANCE),
        epoch: epoch(),
        step,
        chan_seq,
    }
}

fn header(
    from: u16,
    kind: ActivationKind,
    dtype: WireDType,
    streams: u8,
    hidden: u32,
    n_rows: u32,
) -> ActivationHeader {
    ActivationHeader {
        from_stage: StageIx(from),
        to_stage: StageIx(from + 1),
        kind,
        layout: RowLayout {
            dtype,
            streams,
            hidden,
        },
        n_rows,
        part_index: 0,
        part_count: 1,
    }
}

fn span(seq: SequenceId, first_position: u32, n_positions: u32, row_offset: u32) -> Span {
    Span {
        seq,
        first_position,
        n_positions,
        row_offset,
        tree: None,
    }
}

fn side(span: Option<u16>, item: SideItem<'static>) -> Side<'static> {
    Side { span, item }
}

fn le32s(values: &[u32]) -> &'static [u8] {
    leak(values.iter().flat_map(|v| v.to_le_bytes()).collect())
}

fn le16s(values: &[u16]) -> &'static [u8] {
    leak(values.iter().flat_map(|v| v.to_le_bytes()).collect())
}

/// `kind`, `kind_arg` and `kind_flags` as RFC-0031 §6.3 writes each kind.
fn kind_fields(kind: ActivationKind) -> (u8, u32, u32) {
    match kind {
        ActivationKind::Prefill { chunk, last } => (1, chunk, u32::from(last)),
        ActivationKind::Decode => (2, 0, 0),
        ActivationKind::Verify { window, tree } => (3, window, u32::from(tree)),
        ActivationKind::Replay { id, last, forward } => {
            (4, id, u32::from(last) | (u32::from(forward) << 1))
        }
        ActivationKind::Canary => (5, 0, 0),
    }
}

/// The RFC-0031 §6.4 code.
fn dtype_code(dtype: WireDType) -> u8 {
    match dtype {
        WireDType::F32 => 1,
        WireDType::BF16 => 2,
        WireDType::F16 => 3,
        WireDType::Q8B32 => 4,
        WireDType::F8E4M3B32 => 5,
        other => panic!("no RFC-0031 code for {other:?}"),
    }
}

fn span_fields(s: &Span) -> spec::SpanFields {
    spec::SpanFields {
        ingress: s.seq.ingress,
        boot: s.seq.boot,
        n: s.seq.n,
        first_position: s.first_position,
        n_positions: s.n_positions,
        row_offset: s.row_offset,
        tree_offset: s.tree.map_or(0xFFFF, |t| t.offset),
        tree_nodes: s.tree.map_or(0, |t| t.nodes),
    }
}

/// A side item's `type`, `span` and value, laid out as RFC-0031 §6.3's TLV table gives it.
fn tlv_fields(side: &Side<'_>) -> (u16, u16, Vec<u8>) {
    let (ty, value) = match side.item {
        SideItem::TokenIds(ids) => (1, ids.to_vec()),
        SideItem::RouterHint { layer, k, experts } => {
            let mut v = layer.to_le_bytes().to_vec(); // layer: u16
            v.push(k); //                                k: u8
            v.push(0); //                                pad: u8
            v.extend(experts); //                        per row k × u16
            (2, v)
        }
        SideItem::SeqMeta(cbor) => (3, cbor.to_vec()),
        SideItem::SeqCtrl(ctrl) => {
            let mut v = vec![0u8; 8]; //                 code: u8, pad[7]
            match ctrl {
                SeqCtrl::Cancel => v[0] = 1,
                SeqCtrl::Release => v[0] = 2,
                SeqCtrl::ReleasePersist { session } => {
                    v[0] = 3;
                    v.extend(session.0); //              session: [u8; 16]
                }
                SeqCtrl::Reserved(code) => v[0] = code,
            }
            (4, v)
        }
        SideItem::Trace(t) => {
            let mut v = vec![t.version]; //              version
            v.extend(t.trace_id); //                     trace-id
            v.extend(t.span_id); //                      span-id
            v.push(t.flags); //                          flags
            (5, v)
        }
        SideItem::Truncate { to } => (6, to.to_le_bytes().to_vec()),
        SideItem::DraftConf(conf) => (7, conf.to_vec()),
        SideItem::Other { ty, value } => (ty, value.to_vec()),
    };
    (ty, side.span.unwrap_or(0xFFFF), value)
}

/// The golden's bytes from wire values: RFC-0031 §6.3's fields for the typed content, laid out
/// by [`spec`], with `HAS_TRACE` set when a `TRACE` item is present.
#[allow(clippy::too_many_arguments)]
fn golden(
    name: &'static str,
    about: &'static str,
    wire: WireDType,
    spec: ActivationSpec,
    header: ActivationHeader,
    spans: Vec<Span>,
    tree: Vec<TreeNode>,
    side: Vec<Side<'static>>,
    payload: Vec<u8>,
) -> Golden {
    let (kind, kind_arg, kind_flags) = kind_fields(header.kind);
    let fixed = spec::ActivationFixed {
        from_stage: header.from_stage.0,
        to_stage: header.to_stage.0,
        kind,
        dtype: dtype_code(header.layout.dtype),
        streams: header.layout.streams,
        hidden: header.layout.hidden,
        n_rows: header.n_rows,
        part_index: header.part_index,
        part_count: header.part_count,
        kind_arg,
        kind_flags,
    };
    let nodes: Vec<(u16, u8, u8)> = tree
        .iter()
        .map(|n| (n.parent.unwrap_or(0xFFFF), n.depth, n.flags))
        .collect();
    let ext = spec::activation_ext(
        &fixed,
        &spans.iter().map(span_fields).collect::<Vec<_>>(),
        &nodes,
        &side.iter().map(tlv_fields).collect::<Vec<_>>(),
    );
    let trace = side.iter().any(|s| matches!(s.item, SideItem::Trace(_)));
    let flags = spec.flags.0 | if trace { HAS_TRACE } else { 0 };
    let bytes = spec::frame(&base(0x01, flags, spec.step, spec.chan_seq), &ext, &payload);
    Golden {
        name,
        about,
        bytes,
        wire,
        spec,
        header,
        spans,
        tree,
        side,
        payload,
    }
}

fn array<const N: usize>(seed: u8) -> [u8; N] {
    spec::fill(seed, N).try_into().unwrap()
}

/// The chunk of the part goldens: 5 F32 rows of hidden 16 (64-byte stride), one span, its
/// `SEQ_META` and `TOKEN_IDS`.
fn part_chunk() -> Golden {
    golden(
        "unused",
        "the chunk the part goldens split",
        WireDType::F32,
        act_spec(FIRST_OF_SEQ, 20, 5),
        header(
            0,
            ActivationKind::Prefill {
                chunk: 0,
                last: false,
            },
            WireDType::F32,
            1,
            16,
            5,
        ),
        vec![span(Q[0], 100, 5, 0)],
        vec![],
        vec![
            side(Some(0), SideItem::SeqMeta(leak(spec::fill(0x5A, 16)))),
            side(Some(0), SideItem::TokenIds(le32s(&[1, 2, 3, 4, 5]))),
        ],
        spec::f32_rows(5, 16),
    )
}

/// The `max_frame` the part goldens are split under: part 0 takes 3 rows (448 bytes), and a
/// fourth would make it 512.
const PART_MAX_FRAME: usize = 448;

fn goldens() -> Vec<Golden> {
    let trace = TraceContext {
        version: 0,
        trace_id: array(0x70),
        span_id: array(0x80),
        flags: 1,
    };
    let mut all = vec![
        golden(
            "activation-prefill",
            "Prefill chunk 3, LAST_CHUNK, F32 hidden 24 (96-byte rows, 128-byte stride), SEQ_META, \
             TOKEN_IDS, ROUTER_HINT and TRACE",
            WireDType::F32,
            act_spec(FIRST_OF_SEQ, 12, 1),
            header(
                0,
                ActivationKind::Prefill {
                    chunk: 3,
                    last: true,
                },
                WireDType::F32,
                1,
                24,
                4,
            ),
            vec![span(Q[0], 1536, 4, 0)],
            vec![],
            vec![
                side(Some(0), SideItem::SeqMeta(leak(spec::fill(0x5A, 21)))),
                side(Some(0), SideItem::TokenIds(le32s(&[101, 102, 103, 104]))),
                side(
                    Some(0),
                    SideItem::RouterHint {
                        layer: 12,
                        k: 2,
                        experts: le16s(&[5, 9, 5, 17, 3, 9, 40, 41]),
                    },
                ),
                side(None, SideItem::Trace(trace)),
            ],
            spec::f32_rows(4, 24),
        ),
        golden(
            "activation-decode",
            "Decode of 3 sequences, BF16 with 2 streams of hidden 32, a fourth span without rows \
             carrying ReleasePersist, TOKEN_IDS for every span, TRUNCATE, and KV_HEADROOM passed \
             through",
            WireDType::BF16,
            act_spec(0, 13, 2),
            header(1, ActivationKind::Decode, WireDType::BF16, 2, 32, 3),
            vec![
                span(Q[0], 1600, 1, 0),
                span(Q[1], 40, 1, 1),
                span(Q[2], 7, 1, 2),
                span(Q[3], 300, 0, 3),
            ],
            vec![],
            vec![
                side(None, SideItem::TokenIds(le32s(&[11, 22, 33]))),
                side(Some(1), SideItem::Truncate { to: 40 }),
                side(
                    Some(3),
                    SideItem::SeqCtrl(SeqCtrl::ReleasePersist {
                        session: SessionId(array(0xC0)),
                    }),
                ),
                side(
                    None,
                    SideItem::Other {
                        ty: tlv::KV_HEADROOM,
                        // n = 1, pad[7]; { stage 1, flags 0, tokens 1000, 2000, 3000 }
                        value: leak(
                            [1u8, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
                                .into_iter()
                                .chain(le32s(&[1000, 2000, 3000]).iter().copied())
                                .collect(),
                        ),
                    },
                ),
            ],
            spec::bf16_rows(3, 64),
        ),
        golden(
            "activation-verify-tree",
            "Verify with TREE: window 5, one span of 5 nodes, a prompt-lookup branch, DRAFT_CONF; \
             a 20-byte tree area, so the side area starts 4 bytes past an 8-byte boundary",
            WireDType::F32,
            act_spec(0, 14, 3),
            header(
                0,
                ActivationKind::Verify {
                    window: 5,
                    tree: true,
                },
                WireDType::F32,
                1,
                16,
                5,
            ),
            vec![Span {
                tree: Some(SpanTree {
                    offset: 0,
                    nodes: 5,
                }),
                ..span(Q[0], 1601, 5, 0)
            }],
            vec![
                TreeNode {
                    parent: None,
                    depth: 1,
                    flags: 0,
                },
                TreeNode {
                    parent: Some(0),
                    depth: 2,
                    flags: 0,
                },
                TreeNode {
                    parent: Some(0),
                    depth: 2,
                    flags: TreeNode::PROMPT_LOOKUP,
                },
                TreeNode {
                    parent: Some(1),
                    depth: 3,
                    flags: 0,
                },
                TreeNode {
                    parent: None,
                    depth: 1,
                    flags: TreeNode::PROMPT_LOOKUP,
                },
            ],
            vec![side(
                Some(0),
                SideItem::DraftConf(le16s(&[0x3C00, 0x3800, 0x3400, 0x3000, 0x2C00])),
            )],
            spec::f32_rows(5, 16),
        ),
        golden(
            "activation-verify-chain",
            "Verify of a chain: window 4 (γ = 3), BF16 hidden 64, TOKEN_IDS",
            WireDType::BF16,
            act_spec(0, 15, 4),
            header(
                0,
                ActivationKind::Verify {
                    window: 4,
                    tree: false,
                },
                WireDType::BF16,
                1,
                64,
                4,
            ),
            vec![span(Q[1], 41, 4, 0)],
            vec![],
            vec![side(Some(0), SideItem::TokenIds(le32s(&[5, 6, 7, 8])))],
            spec::bf16_rows(4, 64),
        ),
        golden(
            "activation-replay",
            "Replay id 9 with LAST and FORWARD, F32 hidden 16, from stage 1 to stage 2",
            WireDType::F32,
            act_spec(0, 7, 1),
            header(
                1,
                ActivationKind::Replay {
                    id: 9,
                    last: true,
                    forward: true,
                },
                WireDType::F32,
                1,
                16,
                2,
            ),
            vec![span(Q[2], 0, 2, 0)],
            vec![],
            vec![],
            spec::f32_rows(2, 16),
        ),
        golden(
            "activation-canary",
            "Canary: one F32 row of hidden 16, no side items",
            WireDType::F32,
            act_spec(0, 1, 1),
            header(2, ActivationKind::Canary, WireDType::F32, 1, 16, 1),
            vec![span(Q[3], 0, 1, 0)],
            vec![],
            vec![],
            spec::f32_rows(1, 16),
        ),
    ];
    // The two parts of `part_chunk` under a 448-byte `max_frame`, written out by hand.
    let chunk = part_chunk();
    let rows =
        |range: std::ops::Range<usize>| chunk.payload[range.start * 64..range.end * 64].to_vec();
    all.push(golden(
        "activation-prefill-part0",
        "part 0 of 2 of a 5-row chunk under a 448-byte max_frame: rows 0..3 with SEQ_META and \
         their TOKEN_IDS",
        WireDType::F32,
        act_spec(FIRST_OF_SEQ, 20, 5),
        ActivationHeader {
            n_rows: 3,
            part_index: 0,
            part_count: 2,
            ..chunk.header
        },
        vec![span(Q[0], 100, 3, 0)],
        vec![],
        vec![
            side(Some(0), SideItem::SeqMeta(leak(spec::fill(0x5A, 16)))),
            side(Some(0), SideItem::TokenIds(le32s(&[1, 2, 3]))),
        ],
        rows(0..3),
    ));
    all.push(golden(
        "activation-prefill-part1",
        "part 1 of 2: rows 3..5, the span moved to position 103 at row 0, their TOKEN_IDS only",
        WireDType::F32,
        act_spec(0, 20, 6),
        ActivationHeader {
            n_rows: 2,
            part_index: 1,
            part_count: 2,
            ..chunk.header
        },
        vec![span(Q[0], 103, 2, 0)],
        vec![],
        vec![side(Some(0), SideItem::TokenIds(le32s(&[4, 5])))],
        rows(3..5),
    ));
    all
}

fn activation_dir() -> PathBuf {
    golden_dir().join("activation")
}

fn read_golden(name: &str) -> Vec<u8> {
    let path = activation_dir().join(format!("{name}.hex"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    spec::from_hex(&text)
}

#[test]
fn activation_golden_files_match_the_spec_tables() {
    let goldens = goldens();
    if std::env::var_os("SYNTROPY_BLESS_GOLDEN").is_some() {
        std::fs::create_dir_all(activation_dir()).unwrap();
        for golden in &goldens {
            let text = spec::to_hex(golden.name, golden.about, &golden.bytes);
            std::fs::write(activation_dir().join(format!("{}.hex", golden.name)), text).unwrap();
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
    let on_disk: BTreeSet<String> = std::fs::read_dir(activation_dir())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .filter(|name| name.ends_with(".hex"))
        .collect();
    assert_eq!(on_disk, listed);
}

/// The criterion "golden frames per kind (Prefill, Decode, Verify, Replay, Canary)
/// round-trip": each file decodes to its typed values, and those encode to the same bytes,
/// whole, in place in a larger slot, and as the extension alone.
#[test]
fn activation_goldens_round_trip_for_every_kind() {
    let mut kinds = BTreeSet::new();
    let mut dtypes = BTreeSet::new();
    for golden in goldens() {
        let name = golden.name;
        let bytes = read_golden(name);
        let frame = Frame::decode(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(frame.header.spec.ftype, FrameType::Activation, "{name}");
        let decoded = frame
            .activation(golden.wire)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(decoded, golden.activation(), "{name}");
        // `HAS_TRACE` follows from the items: the spec leaves it clear.
        assert_eq!(golden.spec.flags.0 & HAS_TRACE, 0);
        assert_eq!(
            golden.activation().encode_to_vec(&golden.spec),
            Ok(bytes.clone()),
            "{name}"
        );
        let mut slot = vec![0xAA; bytes.len() + 100];
        assert_eq!(
            golden.activation().encode(&golden.spec, &mut slot),
            Ok(bytes.len()),
            "{name}"
        );
        assert_eq!(&slot[..bytes.len()], &bytes[..], "{name}");
        let mut ext = vec![0xAA; frame.ext.len() + 64];
        assert_eq!(
            golden.activation().write_ext(&mut ext),
            Ok(frame.ext.len()),
            "{name}"
        );
        assert_eq!(&ext[..frame.ext.len()], frame.ext, "{name}");
        assert_eq!(golden.activation().ext_len(), Ok(frame.ext.len()), "{name}");
        // Rows at the stride, without its padding.
        let layout = decoded.header.layout;
        let (row_len, stride) = (
            layout.row_len().unwrap() as usize,
            layout.stride().unwrap() as usize,
        );
        assert_eq!(stride % 64, 0);
        assert_eq!(decoded.rows().count(), decoded.header.n_rows as usize);
        for (r, row) in decoded.rows().enumerate() {
            assert_eq!(
                row,
                &golden.payload[r * stride..r * stride + row_len],
                "{name}"
            );
        }
        kinds.insert(kind_fields(decoded.header.kind).0);
        dtypes.insert(dtype_code(layout.dtype));
    }
    assert_eq!(kinds, BTreeSet::from([1, 2, 3, 4, 5]));
    assert_eq!(dtypes, BTreeSet::from([1, 2]));
}

/// F32 rows read back as the values written, padding excluded: hidden 24 is 96 bytes of a
/// 128-byte stride.
#[test]
fn f32_rows_hold_their_values_at_a_64_byte_stride() {
    let bytes = read_golden("activation-prefill");
    let frame = Frame::decode(&bytes).unwrap();
    let activation = frame.activation(WireDType::F32).unwrap();
    assert_eq!(activation.payload.len(), 4 * 128);
    for r in 0..4u32 {
        let row = activation.row(r).unwrap();
        assert_eq!(row.len(), 96);
        for (i, value) in row.as_chunks::<4>().0.iter().enumerate() {
            let value = f32::from_le_bytes(*value);
            assert_eq!(value, (r as usize * 1000 + i) as f32);
        }
        assert!(
            activation.payload[r as usize * 128 + 96..(r as usize + 1) * 128]
                .iter()
                .all(|b| *b == 0)
        );
    }
    assert_eq!(activation.row(4), None);
}

fn encode_parts(
    chunk: &Activation<'_>,
    max_frame: usize,
    specs: impl Fn(&Part) -> ActivationSpec,
) -> Result<Vec<(Part, Vec<u8>)>, FrameError> {
    let mut frames = Vec::new();
    for part in chunk.parts(max_frame)? {
        let mut out = vec![0; part.frame_len];
        let len = chunk.encode_part(&part, &specs(&part), &mut out)?;
        assert_eq!(len, part.frame_len);
        frames.push((part, out));
    }
    Ok(frames)
}

#[test]
fn a_chunk_splits_into_the_part_goldens() {
    let chunk = part_chunk();
    let chunk = chunk.activation();
    let frames = encode_parts(&chunk, PART_MAX_FRAME, |part| {
        if part.index == 0 {
            act_spec(FIRST_OF_SEQ, 20, 5)
        } else {
            act_spec(0, 20, 6)
        }
    })
    .unwrap();
    let parts: Vec<(u16, u16, std::ops::Range<u32>, usize)> = frames
        .iter()
        .map(|(p, _)| (p.index, p.count, p.rows.clone(), p.frame_len))
        .collect();
    assert_eq!(parts, [(0, 2, 0..3, 448), (1, 2, 3..5, 320)]);
    assert_eq!(frames[0].1, read_golden("activation-prefill-part0"));
    assert_eq!(frames[1].1, read_golden("activation-prefill-part1"));
    check_parts(&chunk, &frames, WireDType::F32).unwrap();
    // One byte less, and part 0 holds two rows.
    let rows: Vec<_> = chunk
        .parts(PART_MAX_FRAME - 1)
        .unwrap()
        .map(|p| p.rows)
        .collect();
    assert_eq!(rows, [0..2, 2..5]);
    // A chunk that fits is one part, byte for byte the whole frame.
    let whole = chunk.encode_to_vec(&act_spec(FIRST_OF_SEQ, 20, 5)).unwrap();
    let frames = encode_parts(&chunk, whole.len(), |_| act_spec(FIRST_OF_SEQ, 20, 5)).unwrap();
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].1, whole);
    // A part's one row and extension need 320 bytes; below that nothing fits. A sender's
    // failure, so `Internal`, with the `FrameTooLarge` code.
    let error = chunk.parts(319).unwrap_err();
    assert_eq!(
        error,
        FrameError::TooLong {
            what: "activation part",
            len: 320
        }
    );
    assert_eq!(error.class(), ErrorClass::Internal);
    assert_eq!(error.code(), Some(code::FRAME_TOO_LARGE));
}

/// The criterion "a 16 MiB chunk splits into 2 parts under the TB5 limit (RFC-0031
/// §Performance)": 512 F32 rows at hidden 8192 are 16,777,216 bytes, over Thunderbolt 5's
/// 16,773,120-byte message limit (RFC-0031 §2, `TransportCaps::max_frame`).
#[test]
fn a_16_mib_chunk_splits_into_2_parts_under_the_tb5_limit() {
    const TB5_MAX_FRAME: usize = 16_773_120;
    const ROWS: u32 = 512;
    const STRIDE: usize = 8192 * 4;
    let payload: Vec<u8> = (0..ROWS as usize * STRIDE)
        .map(|i| (i / STRIDE) as u8 ^ i as u8)
        .collect();
    assert_eq!(payload.len(), 16 * 1024 * 1024);
    let ids: Vec<u8> = (1..=ROWS).flat_map(u32::to_le_bytes).collect();
    let meta = spec::fill(0x5A, 64);
    let spans = [span(Q[0], 0, ROWS, 0)];
    let side = [
        Side {
            span: Some(0),
            item: SideItem::SeqMeta(&meta),
        },
        Side {
            span: Some(0),
            item: SideItem::TokenIds(&ids),
        },
    ];
    let chunk = Activation {
        header: header(
            0,
            ActivationKind::Prefill {
                chunk: 0,
                last: true,
            },
            WireDType::F32,
            1,
            8192,
            ROWS,
        ),
        spans: Spans::new(&spans),
        tree: TreeArea::new(&[]),
        side: SideItems::new(&side),
        payload: &payload,
    };
    let whole = 64 + chunk.ext_len().unwrap() + payload.len();
    assert!(whole > TB5_MAX_FRAME, "{whole}");
    let frames = encode_parts(&chunk, TB5_MAX_FRAME, |part| {
        act_spec(0, 30, 1 + u64::from(part.index))
    })
    .unwrap();
    assert_eq!(frames.len(), 2);
    for (part, bytes) in &frames {
        assert!(part.frame_len <= TB5_MAX_FRAME, "{part:?}");
        assert_eq!(bytes.len(), part.frame_len);
    }
    // Part 0 takes every row that fits: 511 × 32 KiB with a 64-byte header and a 2,240-byte
    // extension is 16,746,752 bytes; a 512th row would pass the limit.
    assert_eq!(frames[0].0.rows, 0..511);
    assert_eq!(frames[0].0.frame_len, 16_746_752);
    assert_eq!(frames[1].0.rows, 511..512);
    check_parts(&chunk, &frames, WireDType::F32).unwrap();
}

/// What a frame says about each of its rows, and the items that are not per row.
#[derive(Default)]
struct Joined<'x> {
    /// Per row: its sequence and position.
    rows: Vec<(SequenceId, u32)>,
    /// Per row: the token ids the frame's `TOKEN_IDS` items give it.
    ids: Vec<Vec<u32>>,
    /// Per row: `(layer, k, experts)` from each `ROUTER_HINT`.
    hints: Vec<Vec<(u16, u8, Vec<u8>)>>,
    /// The other items, with the sequence of their span and the part they came in.
    once: Vec<(Option<SequenceId>, SideItem<'x>, usize)>,
    /// The sequence of each span without rows, and the part it came in.
    empty: Vec<(SequenceId, usize)>,
    /// The first part each sequence appears in.
    first: Vec<(SequenceId, usize)>,
}

impl<'x> Joined<'x> {
    fn add(&mut self, activation: &Activation<'x>, part: usize) {
        let base = self.rows.len();
        let n_rows = activation.header.n_rows as usize;
        let spans: Vec<Span> = activation.spans.iter().collect();
        for s in &spans {
            for r in 0..s.n_positions {
                self.rows.push((s.seq, s.first_position + r));
            }
            if s.n_positions == 0 {
                self.empty.push((s.seq, part));
            }
            if !self.first.iter().any(|(seq, _)| *seq == s.seq) {
                self.first.push((s.seq, part));
            }
        }
        self.ids.resize(base + n_rows, Vec::new());
        self.hints.resize(base + n_rows, Vec::new());
        for side in activation.side.iter() {
            let rows = match side.span {
                Some(j) => {
                    let s = spans[usize::from(j)];
                    s.row_offset as usize..(s.row_offset + s.n_positions) as usize
                }
                None => 0..n_rows,
            };
            match side.item {
                SideItem::TokenIds(ids) => {
                    for (r, id) in rows.zip(ids.as_chunks::<4>().0) {
                        self.ids[base + r].push(u32::from_le_bytes(*id));
                    }
                }
                SideItem::RouterHint { layer, k, experts } => {
                    let per_row = 2 * usize::from(k);
                    for (i, r) in rows.enumerate() {
                        let row = experts[i * per_row..(i + 1) * per_row].to_vec();
                        self.hints[base + r].push((layer, k, row));
                    }
                }
                item => {
                    let seq = side.span.map(|j| spans[usize::from(j)].seq);
                    self.once.push((seq, item, part));
                }
            }
        }
    }
}

/// Checks the parts of `chunk` join back into it: they tile its rows in order; each is a frame
/// with the chunk's fixed fields and its own rows' payload; together they give every row the
/// same sequence, position, token ids and router hints; and every other item and every span
/// without rows travels once, in the first part that carries its sequence, or in part 0
/// (RFC-0031 §6.3, revised 2026-10-22).
fn check_parts(
    chunk: &Activation<'_>,
    frames: &[(Part, Vec<u8>)],
    wire: WireDType,
) -> Result<(), TestCaseError> {
    let stride = chunk.header.layout.stride().unwrap() as usize;
    let mut joined = Joined::default();
    let mut decoded = Vec::new();
    let mut next = 0;
    for (index, (part, bytes)) in frames.iter().enumerate() {
        prop_assert_eq!(usize::from(part.index), index);
        prop_assert_eq!(usize::from(part.count), frames.len());
        prop_assert_eq!(part.rows.start, next);
        next = part.rows.end;
        let frame = Frame::decode(bytes).map_err(|e| TestCaseError::fail(e.to_string()))?;
        let activation = frame
            .activation(wire)
            .map_err(|e| TestCaseError::fail(format!("part {index}: {e}")))?;
        prop_assert_eq!(
            activation.header,
            ActivationHeader {
                n_rows: part.rows.end - part.rows.start,
                part_index: part.index,
                part_count: part.count,
                ..chunk.header
            }
        );
        let rows = part.rows.start as usize * stride..part.rows.end as usize * stride;
        prop_assert_eq!(activation.payload, &chunk.payload[rows]);
        decoded.push(activation);
    }
    prop_assert_eq!(next, chunk.header.n_rows);
    for (index, activation) in decoded.iter().enumerate() {
        joined.add(activation, index);
    }
    let mut whole = Joined::default();
    whole.add(chunk, 0);
    prop_assert_eq!(&joined.rows, &whole.rows);
    prop_assert_eq!(&joined.ids, &whole.ids);
    prop_assert_eq!(&joined.hints, &whole.hints);
    let expected_part = |seq: &Option<SequenceId>| match seq {
        None => 0,
        Some(seq) => joined
            .first
            .iter()
            .find(|(s, _)| s == seq)
            .map_or(usize::MAX, |(_, part)| *part),
    };
    for (seq, _, part) in &joined.once {
        prop_assert_eq!(*part, expected_part(seq), "{:?}", seq);
    }
    for (seq, part) in &joined.empty {
        prop_assert_eq!(*part, expected_part(&Some(*seq)));
    }
    let mut once: Vec<(Option<SequenceId>, SideItem<'_>)> =
        joined.once.iter().map(|(s, i, _)| (*s, *i)).collect();
    for (seq, item, _) in &whole.once {
        let at = once.iter().position(|(s, i)| s == seq && i == item);
        prop_assert!(at.is_some(), "{:?} {:?} is in no part", seq, item);
        once.swap_remove(at.unwrap_or(0));
    }
    prop_assert!(once.is_empty(), "items no chunk has: {:?}", once);
    let mut empty: Vec<SequenceId> = joined.empty.iter().map(|(s, _)| *s).collect();
    let mut expected: Vec<SequenceId> = whole.empty.iter().map(|(s, _)| *s).collect();
    empty.sort();
    expected.sort();
    prop_assert_eq!(empty, expected);
    Ok(())
}

/// The criterion "a dtype mismatch with `ChannelSpec::wire` is rejected": with
/// `WireDTypeMismatch`, 0x0206, classed `Unsupported` (RFC-0031 §12, revised 2026-10-22), never
/// `Internal`.
#[test]
fn a_dtype_mismatch_with_the_channel_wire_is_rejected() {
    for golden in goldens() {
        let bytes = read_golden(golden.name);
        let frame = Frame::decode(&bytes).unwrap();
        for wire in [
            WireDType::F32,
            WireDType::BF16,
            WireDType::F16,
            WireDType::Q8B32,
            WireDType::F8E4M3B32,
        ] {
            let result = frame.activation(wire);
            if wire == golden.wire {
                assert!(result.is_ok(), "{}", golden.name);
                continue;
            }
            let error = result.unwrap_err();
            assert_eq!(
                error,
                FrameError::WireDTypeMismatch {
                    expected: wire,
                    found: golden.wire
                },
                "{}",
                golden.name
            );
            assert_eq!(error.code(), Some(code::WIRE_DTYPE_MISMATCH));
            assert_eq!(code::WIRE_DTYPE_MISMATCH, 0x0206);
            assert_eq!(error.class(), ErrorClass::Unsupported);
        }
    }
}

/// A frame as wire values, to break one rule at a time.
struct Raw {
    ftype: u8,
    flags: u16,
    fixed: spec::ActivationFixed,
    spans: Vec<spec::SpanFields>,
    tree: Vec<(u16, u8, u8)>,
    tlvs: Vec<(u16, u16, Vec<u8>)>,
    payload: Vec<u8>,
}

impl Raw {
    fn ext(&self) -> Vec<u8> {
        spec::activation_ext(&self.fixed, &self.spans, &self.tree, &self.tlvs)
    }

    fn frame(&self, ext: &[u8]) -> Vec<u8> {
        spec::frame(&base(self.ftype, self.flags, 1, 1), ext, &self.payload)
    }

    fn bytes(&self) -> Vec<u8> {
        self.frame(&self.ext())
    }
}

fn raw_span(n: u32, first_position: u32, n_positions: u32, row_offset: u32) -> spec::SpanFields {
    spec::SpanFields {
        ingress: Q[0].ingress,
        boot: Q[0].boot,
        n,
        first_position,
        n_positions,
        row_offset,
        tree_offset: 0xFFFF,
        tree_nodes: 0,
    }
}

fn ids(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Decode, F32 hidden 16: two spans of one row each, `TOKEN_IDS` for every span.
fn raw_decode() -> Raw {
    Raw {
        ftype: 0x01,
        flags: 0,
        fixed: spec::ActivationFixed {
            from_stage: 0,
            to_stage: 1,
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
        spans: vec![raw_span(1, 10, 1, 0), raw_span(2, 20, 1, 1)],
        tree: vec![],
        tlvs: vec![(tlv::TOKEN_IDS, 0xFFFF, ids(&[7, 8]))],
        payload: spec::f32_rows(2, 16),
    }
}

/// Verify with `TREE`, F32 hidden 16: one span of a 3-node chain, `DRAFT_CONF`.
fn raw_tree() -> Raw {
    let mut raw = raw_decode();
    raw.fixed.kind = 3;
    raw.fixed.kind_arg = 3;
    raw.fixed.kind_flags = 1;
    raw.fixed.n_rows = 3;
    raw.spans = vec![spec::SpanFields {
        tree_offset: 0,
        tree_nodes: 3,
        ..raw_span(1, 50, 3, 0)
    }];
    raw.tree = vec![(0xFFFF, 1, 0), (0, 2, 0), (1, 3, 0)];
    raw.tlvs = vec![(tlv::DRAFT_CONF, 0, vec![0; 6])];
    raw.payload = spec::f32_rows(3, 16);
    raw
}

fn field(field: &'static str, value: u64) -> FrameError {
    FrameError::BadField {
        ftype: FrameType::Activation,
        field,
        value,
    }
}

fn inconsistent(field: &'static str, value: u64, expected: u64) -> FrameError {
    FrameError::Inconsistent {
        ftype: FrameType::Activation,
        field,
        value,
        expected,
    }
}

/// One refusal case: `raw` with `edit` applied, read on a `wire` channel, refused with `error`.
fn case(
    name: &'static str,
    mut raw: Raw,
    edit: fn(&mut Raw),
    wire: WireDType,
    error: FrameError,
) -> (&'static str, Vec<u8>, WireDType, FrameError) {
    edit(&mut raw);
    (name, raw.bytes(), wire, error)
}

/// Every rule the codec checks, broken one at a time, is refused with the error that names it,
/// in a class RFC-0060 §10.2 allows.
#[test]
fn a_frame_breaking_a_rule_is_refused_with_a_classified_error() {
    let tlv_ext = |raw: &Raw, edit: fn(&mut Vec<u8>)| {
        let mut ext = raw.ext();
        edit(&mut ext);
        raw.frame(&ext)
    };
    let (f32, decode, tree) = (WireDType::F32, raw_decode, raw_tree);
    let mut frames = vec![
        case(
            "BF16 on an F32 channel",
            decode(),
            |r| r.fixed.dtype = 2,
            f32,
            FrameError::WireDTypeMismatch {
                expected: WireDType::F32,
                found: WireDType::BF16,
            },
        ),
        case(
            "F32 on a BF16 channel",
            decode(),
            |_| {},
            WireDType::BF16,
            FrameError::WireDTypeMismatch {
                expected: WireDType::BF16,
                found: WireDType::F32,
            },
        ),
        case(
            "the F16 container",
            decode(),
            |r| r.fixed.dtype = 3,
            WireDType::F16,
            FrameError::UnsupportedDType(WireDType::F16),
        ),
        case(
            "the Q8B32 container",
            decode(),
            |r| r.fixed.dtype = 4,
            WireDType::Q8B32,
            FrameError::UnsupportedDType(WireDType::Q8B32),
        ),
        case(
            "the F8E4M3B32 container",
            decode(),
            |r| r.fixed.dtype = 5,
            WireDType::F8E4M3B32,
            FrameError::UnsupportedDType(WireDType::F8E4M3B32),
        ),
        case(
            "dtype 0",
            decode(),
            |r| r.fixed.dtype = 0,
            f32,
            field("dtype", 0),
        ),
        case(
            "dtype 6",
            decode(),
            |r| r.fixed.dtype = 6,
            f32,
            field("dtype", 6),
        ),
        case(
            "kind 0",
            decode(),
            |r| r.fixed.kind = 0,
            f32,
            field("kind", 0),
        ),
        case(
            "kind 6",
            decode(),
            |r| r.fixed.kind = 6,
            f32,
            field("kind", 6),
        ),
        case(
            "no streams",
            decode(),
            |r| r.fixed.streams = 0,
            f32,
            field("streams", 0),
        ),
        case(
            "hidden 0",
            decode(),
            |r| r.fixed.hidden = 0,
            f32,
            field("hidden", 0),
        ),
        case(
            "no parts",
            decode(),
            |r| r.fixed.part_count = 0,
            f32,
            field("part_count", 0),
        ),
        case(
            "part 1 of 1",
            decode(),
            |r| r.fixed.part_index = 1,
            f32,
            field("part_index", 1),
        ),
        case(
            "a gap between spans",
            decode(),
            |r| r.spans[1].row_offset = 2,
            f32,
            inconsistent("row_offset", 2, 1),
        ),
        case(
            "spans short of n_rows",
            decode(),
            |r| {
                r.fixed.n_rows = 3;
                r.payload = spec::f32_rows(3, 16);
            },
            f32,
            inconsistent("n_rows", 3, 2),
        ),
        case(
            "a payload one byte short",
            decode(),
            |r| {
                r.payload.pop();
            },
            f32,
            inconsistent("payload_len", 127, 128),
        ),
        case(
            "rows without the stride's padding",
            decode(),
            |r| {
                r.fixed.hidden = 24;
                r.payload = vec![0; 2 * 96];
            },
            f32,
            inconsistent("payload_len", 192, 256),
        ),
        case(
            "TOKEN_IDS short of the rows",
            decode(),
            |r| r.tlvs[0].2 = ids(&[7]),
            f32,
            inconsistent("TOKEN_IDS len", 4, 8),
        ),
        case(
            "TOKEN_IDS past a span's rows",
            decode(),
            |r| r.tlvs[0].1 = 1,
            f32,
            inconsistent("TOKEN_IDS len", 8, 4),
        ),
        case(
            "a TLV naming a span the frame lacks",
            decode(),
            |r| r.tlvs[0].1 = 2,
            f32,
            field("TLV span", 2),
        ),
        case(
            "ROUTER_HINT short of k per row",
            decode(),
            |r| {
                r.tlvs = vec![(tlv::ROUTER_HINT, 0, vec![3, 0, 2, 0, 9, 0])];
            },
            f32,
            inconsistent("ROUTER_HINT len", 2, 4),
        ),
        case(
            "ROUTER_HINT without its head",
            decode(),
            |r| {
                r.tlvs = vec![(tlv::ROUTER_HINT, 0, vec![3, 0, 2])];
            },
            f32,
            field("ROUTER_HINT len", 3),
        ),
        case(
            "TRACE of 25 bytes",
            decode(),
            |r| {
                r.flags = HAS_TRACE;
                r.tlvs = vec![(tlv::TRACE, 0xFFFF, vec![0; 25])];
            },
            f32,
            field("TRACE len", 25),
        ),
        case(
            "TRUNCATE of 8 bytes",
            decode(),
            |r| r.tlvs = vec![(tlv::TRUNCATE, 0, vec![0; 8])],
            f32,
            field("TRUNCATE len", 8),
        ),
        case(
            "SEQ_CTRL code 0",
            decode(),
            |r| r.tlvs = vec![(tlv::SEQ_CTRL, 0, vec![0; 8])],
            f32,
            field("SEQ_CTRL code", 0),
        ),
        case(
            "SEQ_CTRL code 6",
            decode(),
            |r| {
                r.tlvs = vec![(tlv::SEQ_CTRL, 0, vec![6, 0, 0, 0, 0, 0, 0, 0])];
            },
            f32,
            field("SEQ_CTRL code", 6),
        ),
        case(
            "Cancel with a session",
            decode(),
            |r| {
                let mut v = vec![1u8];
                v.resize(24, 0);
                r.tlvs = vec![(tlv::SEQ_CTRL, 0, v)];
            },
            f32,
            field("SEQ_CTRL len", 24),
        ),
        case(
            "ReleasePersist without its session",
            decode(),
            |r| {
                r.tlvs = vec![(tlv::SEQ_CTRL, 0, vec![3, 0, 0, 0, 0, 0, 0, 0])];
            },
            f32,
            field("SEQ_CTRL len", 8),
        ),
        case(
            "an empty SEQ_CTRL",
            decode(),
            |r| r.tlvs = vec![(tlv::SEQ_CTRL, 0, vec![])],
            f32,
            field("SEQ_CTRL len", 0),
        ),
        case(
            "DRAFT_CONF on a span without a tree",
            decode(),
            |r| {
                r.tlvs = vec![(tlv::DRAFT_CONF, 0, vec![0; 2])];
            },
            f32,
            field("DRAFT_CONF span", 0),
        ),
        case(
            "DRAFT_CONF for every span of a frame without a tree",
            decode(),
            |r| {
                r.tlvs = vec![(tlv::DRAFT_CONF, 0xFFFF, vec![0; 4])];
            },
            f32,
            field("DRAFT_CONF span", 0xFFFF),
        ),
        case(
            "a span with nodes but no tree offset",
            decode(),
            |r| r.spans[0].tree_nodes = 1,
            f32,
            field("tree_nodes", 1),
        ),
        case(
            "a tree in a decode frame",
            decode(),
            |r| {
                r.spans[0].tree_offset = 0;
                r.spans[0].tree_nodes = 1;
                r.tree = vec![(0xFFFF, 1, 0)];
            },
            f32,
            field("tree_offset", 0),
        ),
        case(
            "a tree in a verify frame without TREE",
            tree(),
            |r| r.fixed.kind_flags = 0,
            f32,
            field("tree_offset", 0),
        ),
        case(
            "a node that is its own parent",
            tree(),
            |r| r.tree[1].0 = 1,
            f32,
            field("parent", 1),
        ),
        case(
            "a parent after its child",
            tree(),
            |r| r.tree[1].0 = 2,
            f32,
            field("parent", 2),
        ),
        case(
            "fewer nodes than rows",
            tree(),
            |r| {
                r.spans[0].tree_nodes = 2;
                r.tlvs[0].2 = vec![0; 4];
            },
            f32,
            inconsistent("tree_nodes", 2, 3),
        ),
        case(
            "a node no span owns",
            tree(),
            |r| r.tree.push((0xFFFF, 1, 0)),
            f32,
            inconsistent("tree_len", 16, 12),
        ),
        case(
            "a tree that does not start the area",
            tree(),
            |r| r.spans[0].tree_offset = 1,
            f32,
            inconsistent("tree_offset", 1, 0),
        ),
        case(
            "DRAFT_CONF short of the nodes",
            tree(),
            |r| r.tlvs[0].2 = vec![0; 4],
            f32,
            inconsistent("DRAFT_CONF len", 4, 6),
        ),
        case(
            "HAS_TRACE without a TRACE",
            decode(),
            |r| r.flags = HAS_TRACE,
            f32,
            inconsistent("HAS_TRACE", 1, 0),
        ),
        case(
            "a TRACE without HAS_TRACE",
            decode(),
            |r| {
                r.tlvs.push((tlv::TRACE, 0xFFFF, vec![0; 26]));
            },
            f32,
            inconsistent("HAS_TRACE", 0, 1),
        ),
        case(
            "a Token frame",
            decode(),
            |r| r.ftype = 0x02,
            f32,
            FrameError::NotActivation(FrameType::Token),
        ),
    ];
    // Edits to the extension's bytes, past what the wire values can say.
    let raw = raw_decode();
    frames.push((
        "a TLV running past the side area",
        // The TLV at side offset 0 claims 64 value bytes of a 16-byte area.
        tlv_ext(&raw, |ext| ext[128 + 4] = 64),
        WireDType::F32,
        inconsistent("side_len", 16, 72),
    ));
    frames.push((
        "a side area of 20 bytes",
        tlv_ext(&raw, |ext| ext[36] = 20),
        WireDType::F32,
        field("side_len", 20),
    ));
    frames.push((
        "a tree area of 2 bytes",
        tlv_ext(&raw, |ext| ext[32] = 2),
        WireDType::F32,
        field("tree_len", 2),
    ));
    frames.push((
        "areas past the extension",
        tlv_ext(&raw, |ext| ext[36] = 80),
        WireDType::F32,
        FrameError::ExtTooShort {
            ftype: FrameType::Activation,
            len: 192,
            needed: 208,
        },
    ));
    frames.push((
        "no extension",
        raw.frame(&[]),
        WireDType::F32,
        FrameError::ExtTooShort {
            ftype: FrameType::Activation,
            len: 0,
            needed: 64,
        },
    ));
    for (name, bytes, wire, expected) in frames {
        let frame = Frame::decode(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        let error = frame.activation(wire).expect_err(name);
        assert_eq!(error, expected, "{name}");
        assert!(
            REFUSAL_CLASSES.contains(&error.class()),
            "{name}: {error} is {:?}",
            error.class()
        );
    }
    // The base frames themselves are accepted, so each refusal is its edit's doing.
    for raw in [raw_decode(), raw_tree()] {
        let bytes = raw.bytes();
        assert!(
            Frame::decode(&bytes)
                .unwrap()
                .activation(WireDType::F32)
                .is_ok()
        );
    }
}

/// Reserved fields, value padding, bytes past the side area, `kind_arg` and reserved
/// `kind_flags` bits of a decode frame, item types this codec does not type, and the reserved
/// `SEQ_CTRL` codes 4 and 5 are all accepted and carry no meaning (RFC-0031 §6, §6.3, §10).
#[test]
fn reserved_bytes_and_later_additions_are_ignored() {
    let raw = raw_decode();
    let plain = raw.bytes();
    let plain = Frame::decode(&plain).unwrap();
    let plain = plain.activation(WireDType::F32).unwrap();
    type ExtEdit = fn(&mut Vec<u8>);
    let edits: [(&str, ExtEdit); 4] = [
        ("reserved fixed bytes", |ext| {
            ext[7] = 0xEE;
            ext[22] = 0xEE;
            ext[23] = 0xEE;
            ext[40..64].fill(0xEE);
        }),
        ("a decode kind_arg and reserved kind_flags", |ext| {
            ext[24..32].fill(0xEE);
        }),
        ("a later minor's trailing extension fields", |ext| {
            ext.extend([0xEE; 64]);
        }),
        ("the extension's padding", |ext| {
            ext[160..192].fill(0xEE);
        }),
    ];
    for (name, edit) in edits {
        let mut ext = raw.ext();
        edit(&mut ext);
        let bytes = raw.frame(&ext);
        let frame = Frame::decode(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        let activation = frame
            .activation(WireDType::F32)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(activation, plain, "{name}");
    }
    // Value padding proper: a 4-byte TRUNCATE padded with 0xEE.
    let mut raw = raw_decode();
    raw.tlvs = vec![(tlv::TRUNCATE, 1, 9u32.to_le_bytes().to_vec())];
    let mut ext = raw.ext();
    ext[128 + 12..128 + 16].fill(0xEE);
    let bytes = raw.frame(&ext);
    let frame = Frame::decode(&bytes).unwrap();
    let side: Vec<Side<'_>> = frame
        .activation(WireDType::F32)
        .unwrap()
        .side
        .iter()
        .collect();
    assert_eq!(
        side,
        [Side {
            span: Some(1),
            item: SideItem::Truncate { to: 9 }
        }]
    );
    // Types this codec does not type pass through, and the reserved control codes parse.
    let mut raw = raw_decode();
    raw.tlvs = vec![
        (tlv::KV_HEADROOM, 0xFFFF, vec![1; 24]),
        (tlv::SESSION_SAVED, 0xFFFF, vec![2; 56]),
        (16, 0, vec![3; 5]),
        (0xFFFE, 1, vec![]),
        (tlv::SEQ_CTRL, 0, vec![4, 0, 0, 0, 0, 0, 0, 0]),
        (tlv::SEQ_CTRL, 1, vec![5]),
    ];
    let bytes = raw.bytes();
    let frame = Frame::decode(&bytes).unwrap();
    let activation = frame.activation(WireDType::F32).unwrap();
    let items: Vec<SideItem<'_>> = activation.side.iter().map(|s| s.item).collect();
    assert_eq!(
        items,
        [
            SideItem::Other {
                ty: tlv::KV_HEADROOM,
                value: &[1; 24]
            },
            SideItem::Other {
                ty: tlv::SESSION_SAVED,
                value: &[2; 56]
            },
            SideItem::Other {
                ty: 16,
                value: &[3; 5]
            },
            SideItem::Other {
                ty: 0xFFFE,
                value: &[]
            },
            SideItem::SeqCtrl(SeqCtrl::Reserved(4)),
            SideItem::SeqCtrl(SeqCtrl::Reserved(5)),
        ]
    );
    // Re-encoded, the reserved codes take the 8-byte form.
    let again = activation.encode_to_vec(&act_spec(0, 1, 1)).unwrap();
    let frame = Frame::decode(&again).unwrap();
    assert_eq!(frame.activation(WireDType::F32), Ok(activation));
}

/// A sender cannot write what a receiver would refuse: the encoder checks the same rules.
#[test]
fn the_encoder_refuses_what_the_decoder_would() {
    let golden = &goldens()[0];
    let short = &golden.payload[..golden.payload.len() - 1];
    let activation = Activation {
        payload: short,
        ..golden.activation()
    };
    assert_eq!(
        activation.encode_to_vec(&golden.spec),
        Err(inconsistent("payload_len", 511, 512))
    );
    let disguised = [Side {
        span: None,
        item: SideItem::Other {
            ty: tlv::TOKEN_IDS,
            value: &[0; 16],
        },
    }];
    let activation = Activation {
        side: SideItems::new(&disguised),
        ..golden.activation()
    };
    assert_eq!(
        activation.encode_to_vec(&golden.spec),
        Err(field("TLV type", u64::from(tlv::TOKEN_IDS)))
    );
    let reserved = [Side {
        span: None,
        item: SideItem::SeqCtrl(SeqCtrl::Reserved(7)),
    }];
    let activation = Activation {
        side: SideItems::new(&reserved),
        ..golden.activation()
    };
    assert_eq!(
        activation.encode_to_vec(&golden.spec),
        Err(field("SEQ_CTRL code", 7))
    );
    let f16 = Activation {
        header: ActivationHeader {
            layout: RowLayout {
                dtype: WireDType::F16,
                ..golden.header.layout
            },
            ..golden.header
        },
        ..golden.activation()
    };
    assert_eq!(
        f16.encode_to_vec(&golden.spec),
        Err(FrameError::UnsupportedDType(WireDType::F16))
    );
    // A part of a chunk already in parts, and a part of a tree frame, are refused.
    let part = goldens().remove(6);
    assert_eq!(part.header.part_count, 2);
    assert_eq!(
        part.activation().parts(1 << 20).map(|p| p.len()),
        Err(field("part_count", 2))
    );
    let tree = goldens().remove(2);
    assert!(tree.header.kind.is_tree());
    let len = tree.bytes.len();
    assert_eq!(tree.activation().parts(len).map(|p| p.len()), Ok(1));
    assert_eq!(
        tree.activation().parts(len - 1).map(|p| p.len()),
        Err(FrameError::TooLong {
            what: "verify frame with a tree",
            len
        })
    );
    let half = Part {
        index: 0,
        count: 2,
        rows: 0..2,
        frame_len: len,
    };
    let mut out = vec![0; len];
    assert_eq!(
        tree.activation().encode_part(&half, &tree.spec, &mut out),
        Err(field("part", 0))
    );
}

/// The criterion "a corrupted byte anywhere is detected", for the `Activation` goldens.
#[test]
fn a_corrupted_activation_byte_is_detected() {
    for golden in goldens() {
        let bytes = read_golden(golden.name);
        for at in 0..bytes.len() {
            for mask in [1, 2, 4, 8, 16, 32, 64, 128, 0xFF] {
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

/// A side item a generated chunk carries, with its own bytes.
#[derive(Clone, Debug)]
enum Owned {
    TokenIds(Vec<u8>),
    RouterHint(u16, u8, Vec<u8>),
    SeqMeta(Vec<u8>),
    SeqCtrl(SeqCtrl),
    Trace(TraceContext),
    Truncate(u32),
    Other(u16, Vec<u8>),
}

impl Owned {
    fn item(&self) -> SideItem<'_> {
        match self {
            Owned::TokenIds(ids) => SideItem::TokenIds(ids),
            Owned::RouterHint(layer, k, experts) => SideItem::RouterHint {
                layer: *layer,
                k: *k,
                experts,
            },
            Owned::SeqMeta(cbor) => SideItem::SeqMeta(cbor),
            Owned::SeqCtrl(ctrl) => SideItem::SeqCtrl(*ctrl),
            Owned::Trace(trace) => SideItem::Trace(*trace),
            Owned::Truncate(to) => SideItem::Truncate { to: *to },
            Owned::Other(ty, value) => SideItem::Other { ty: *ty, value },
        }
    }
}

/// A valid chunk from a few numbers: the kind, the layout, each span's rows and which items it
/// has (bit0 `TOKEN_IDS`, bit1 `ROUTER_HINT`, bit2 `SEQ_META`, bit3 `TRUNCATE`, bit4
/// `SEQ_CTRL`), and the items for every span (bit0 `TRACE`, bit1 `TOKEN_IDS`, bit2 an item of a
/// type this codec passes through).
struct Generated {
    header: ActivationHeader,
    spans: Vec<Span>,
    items: Vec<(Option<u16>, Owned)>,
    payload: Vec<u8>,
}

fn generate(
    kind: u8,
    bf16: bool,
    streams: u8,
    hidden: u32,
    spans: &[(u32, u8)],
    all: u8,
) -> Generated {
    let kind = match kind % 4 {
        0 => ActivationKind::Prefill {
            chunk: u32::from(all),
            last: all & 1 != 0,
        },
        1 => ActivationKind::Decode,
        2 => ActivationKind::Replay {
            id: 3,
            last: false,
            forward: true,
        },
        _ => ActivationKind::Canary,
    };
    let dtype = if bf16 {
        WireDType::BF16
    } else {
        WireDType::F32
    };
    let mut out_spans = Vec::new();
    let mut items = Vec::new();
    let mut row = 0;
    for (j, &(n, bits)) in spans.iter().enumerate() {
        let seq = SequenceId {
            ingress: 9,
            boot: 1,
            n: j as u32,
        };
        let position = 1000 * j as u32;
        out_spans.push(span(seq, position, n, row));
        let ids: Vec<u8> = (0..n).flat_map(|r| (row + r).to_le_bytes()).collect();
        let j = Some(j as u16);
        if bits & 1 != 0 {
            items.push((j, Owned::TokenIds(ids)));
        }
        if bits & 2 != 0 {
            let k = 1 + bits % 3;
            let experts = spec::fill(bits, 2 * usize::from(k) * n as usize);
            items.push((j, Owned::RouterHint(u16::from(bits), k, experts)));
        }
        if bits & 4 != 0 {
            items.push((j, Owned::SeqMeta(spec::fill(bits, usize::from(bits % 20)))));
        }
        if bits & 8 != 0 {
            items.push((j, Owned::Truncate(position)));
        }
        if bits & 16 != 0 {
            let ctrl = match bits % 3 {
                0 => SeqCtrl::Cancel,
                1 => SeqCtrl::Release,
                _ => SeqCtrl::ReleasePersist {
                    session: SessionId(array(bits)),
                },
            };
            items.push((j, Owned::SeqCtrl(ctrl)));
        }
        row += n;
    }
    if all & 1 != 0 {
        items.push((
            None,
            Owned::Trace(TraceContext {
                version: 0,
                trace_id: array(all),
                span_id: array(all),
                flags: 1,
            }),
        ));
    }
    if all & 2 != 0 {
        let ids: Vec<u8> = (0..row).flat_map(|r| (100 + r).to_le_bytes()).collect();
        items.push((None, Owned::TokenIds(ids)));
    }
    if all & 4 != 0 {
        items.push((
            None,
            Owned::Other(
                tlv::KV_HEADROOM + u16::from(all % 9),
                spec::fill(all, usize::from(all % 13)),
            ),
        ));
    }
    let elements = usize::from(streams) * hidden as usize;
    let payload = if bf16 {
        spec::bf16_rows(row as usize, elements)
    } else {
        spec::f32_rows(row as usize, elements)
    };
    Generated {
        header: header(0, kind, dtype, streams, hidden, row),
        spans: out_spans,
        items,
        payload,
    }
}

proptest! {
    /// The `Activation` parser on arbitrary extensions and payloads: no panic, every refusal is
    /// classified, and an accepted frame encodes to one that decodes to it again.
    #[test]
    fn arbitrary_activation_bodies_parse_safely(
        ext in proptest::collection::vec(any::<u8>(), 0..400),
        payload in proptest::collection::vec(any::<u8>(), 0..300),
        bf16 in any::<bool>(),
    ) {
        let wire = if bf16 { WireDType::BF16 } else { WireDType::F32 };
        match Activation::parse(&ext, &payload, wire) {
            Err(error) => {
                prop_assert!(REFUSAL_CLASSES.contains(&error.class()), "{error}: {:?}", error.class());
            }
            Ok(activation) => {
                let bytes = activation.encode_to_vec(&act_spec(0, 1, 1)).unwrap();
                let frame = Frame::decode(&bytes).unwrap();
                prop_assert_eq!(frame.activation(wire), Ok(activation));
            }
        }
    }

    /// Golden `Activation` frames with header, extension or payload bytes changed and both CRCs
    /// recomputed, so the changes reach the codec: no panic, every refusal is classified, and an
    /// accepted frame re-encodes to one that decodes to it again.
    #[test]
    fn resealed_activation_mutations_parse_safely(
        which in any::<usize>(),
        edits in proptest::collection::vec((any::<usize>(), any::<u8>()), 1..6),
    ) {
        let goldens = goldens();
        let golden = &goldens[which % goldens.len()];
        let mut bytes = read_golden(golden.name);
        for (at, value) in edits {
            let at = at % bytes.len();
            if (4..48).contains(&at) || at >= 64 {
                bytes[at] = value;
            }
        }
        reseal(&mut bytes);
        let Ok(frame) = Frame::decode(&bytes) else { return Ok(()) };
        match frame.activation(golden.wire) {
            Err(error) => {
                prop_assert!(REFUSAL_CLASSES.contains(&error.class()), "{error}: {:?}", error.class());
            }
            Ok(activation) => {
                let header = frame.header.spec;
                let spec = ActivationSpec {
                    wire_minor: header.wire_minor,
                    flags: header.flags,
                    instance: header.instance,
                    epoch: header.epoch,
                    step: header.step,
                    chan_seq: header.chan_seq,
                };
                let again = activation.encode_to_vec(&spec).unwrap();
                let frame = Frame::decode(&again).unwrap();
                prop_assert_eq!(frame.activation(golden.wire), Ok(activation));
            }
        }
    }

    /// Generated chunks round-trip through a frame, and split under any `max_frame` into parts
    /// that each fit and join back into the chunk, or are refused because one row cannot fit.
    #[test]
    fn generated_chunks_round_trip_and_split(
        kind in any::<u8>(),
        bf16 in any::<bool>(),
        streams in 1u8..4,
        hidden in 1u32..48,
        spans in proptest::collection::vec((0u32..6, any::<u8>()), 0..5),
        all in any::<u8>(),
        room in 0usize..4096,
    ) {
        let generated = generate(kind, bf16, streams, hidden, &spans, all);
        let side: Vec<Side<'_>> = generated
            .items
            .iter()
            .map(|(span, owned)| Side { span: *span, item: owned.item() })
            .collect();
        let chunk = Activation {
            header: generated.header,
            spans: Spans::new(&generated.spans),
            tree: TreeArea::new(&[]),
            side: SideItems::new(&side),
            payload: &generated.payload,
        };
        let wire = generated.header.layout.dtype;
        let whole = chunk.encode_to_vec(&act_spec(0, 1, 1)).unwrap();
        let frame = Frame::decode(&whole).unwrap();
        prop_assert_eq!(frame.activation(wire), Ok(chunk));
        let max_frame = 192 + room;
        match encode_parts(&chunk, max_frame, |part| act_spec(0, 1, 1 + u64::from(part.index))) {
            Ok(frames) => {
                for (part, _) in &frames {
                    prop_assert!(part.frame_len <= max_frame, "{part:?} over {max_frame}");
                }
                if whole.len() <= max_frame {
                    prop_assert_eq!(frames.len(), 1);
                    prop_assert_eq!(&frames[0].1, &whole);
                }
                check_parts(&chunk, &frames, wire)?;
            }
            Err(FrameError::TooLong { what: "activation part", len }) => {
                prop_assert!(len > max_frame);
            }
            Err(error) => return Err(TestCaseError::fail(error.to_string())),
        }
    }
}
