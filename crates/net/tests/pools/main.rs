// SPDX-License-Identifier: MIT OR Apache-2.0

//! Header areas built in place and multi-part prefill landing (DIST-E7-T12; RFC-0031 §8, §6.3),
//! with frames from `net`'s own codec.

use syntropy_core::plan::WireDType;
use syntropy_core::{InstanceId, PlanEpoch, SequenceId, StageIx};
use syntropy_net::frame::{
    self, Activation, ActivationHeader, ActivationKind, ActivationSpec, Flags, Frame, HEADER_LEN,
    RowLayout, Side, SideItem, SideItems, Span, Spans, TreeArea, WIRE_MINOR,
};
use syntropy_net::transport::buffer::{BufferPool, PoolId};
use syntropy_net::transport::landing::Landing;
use syntropy_net::transport::sizing::{HeaderLimits, header_area};

const SEQ: SequenceId = SequenceId {
    ingress: 0x5359_4e00,
    boot: 3,
    n: 7,
};

fn spec(part: u16) -> ActivationSpec {
    ActivationSpec {
        wire_minor: WIRE_MINOR,
        flags: Flags(if part == 0 { Flags::FIRST_OF_SEQ } else { 0 }),
        instance: InstanceId(0x5359_4e46),
        epoch: PlanEpoch { term: 3, seq: 9 },
        step: 12,
        chan_seq: u64::from(part) + 1,
    }
}

fn header(n_rows: u32, hidden: u32) -> ActivationHeader {
    ActivationHeader {
        from_stage: StageIx(0),
        to_stage: StageIx(1),
        kind: ActivationKind::Prefill {
            chunk: 0,
            last: true,
        },
        layout: RowLayout {
            dtype: WireDType::F32,
            streams: 1,
            hidden,
        },
        n_rows,
        part_index: 0,
        part_count: 1,
    }
}

/// `rows` F32 rows of `hidden` elements at their stride, each element its row and column.
fn rows(rows: u32, hidden: u32) -> Vec<u8> {
    let layout = header(rows, hidden).layout;
    let stride = layout.stride().expect("F32") as usize;
    let mut payload = vec![0u8; rows as usize * stride];
    for r in 0..rows as usize {
        for c in 0..hidden as usize {
            let value = (r * 1000 + c) as f32;
            payload[r * stride + 4 * c..r * stride + 4 * c + 4]
                .copy_from_slice(&value.to_le_bytes());
        }
    }
    payload
}

fn le32s(values: impl Iterator<Item = u32>) -> Vec<u8> {
    values.flat_map(u32::to_le_bytes).collect()
}

/// RFC-0031 §8: a 512-row prefill frame with `TOKEN_IDS` and `ROUTER_HINT` at k = 8 builds its
/// base header and extension in place, in a slot the size of its plan-sized header area.
#[test]
fn a_512_row_prefill_header_is_built_in_place_in_its_plan_sized_area() {
    const ROWS: u32 = 512;
    let limits = HeaderLimits {
        max_rows: ROWS,
        max_spans: 1,
        token_ids: true,
        router_k: Some(8),
        ..HeaderLimits::default()
    };
    let area = header_area(&limits);
    let token_ids = le32s(0..ROWS);
    let experts: Vec<u8> = (0..ROWS * 8)
        .flat_map(|i| (i as u16 % 256).to_le_bytes())
        .collect();
    let side = [
        Side {
            span: Some(0),
            item: SideItem::TokenIds(&token_ids),
        },
        Side {
            span: Some(0),
            item: SideItem::RouterHint {
                layer: 3,
                k: 8,
                experts: &experts,
            },
        },
    ];
    let spans = [Span {
        seq: SEQ,
        first_position: 0,
        n_positions: ROWS,
        row_offset: 0,
        tree: None,
    }];
    let payload = rows(ROWS, 16);
    let chunk = Activation {
        header: header(ROWS, 16),
        spans: Spans::new(&spans),
        tree: TreeArea::new(&[]),
        side: SideItems::new(&side),
        payload: &payload,
    };
    let ext_len = chunk.ext_len().expect("a valid chunk");
    assert_eq!(ext_len, 10_368, "the codec's extension at these limits");
    assert!(HEADER_LEN + ext_len <= area, "{ext_len} + 64 > {area}");

    // The area is one slot of a header pool; base header and extension are written there.
    let pool = BufferPool::heap(PoolId(1), area, 2).expect("a header pool");
    let mut slot = pool.take().expect("a free header slot");
    let (head, rest) = slot.bytes_mut().split_at_mut(HEADER_LEN);
    let written = chunk.write_ext(rest).expect("the extension fits the area");
    assert_eq!(written, ext_len);
    let spec = spec(0);
    let base = frame::encode_header(
        &frame::FrameSpec {
            wire_minor: spec.wire_minor,
            ftype: frame::FrameType::Activation,
            flags: spec.flags,
            instance: spec.instance,
            epoch: spec.epoch,
            step: spec.step,
            chan_seq: spec.chan_seq,
        },
        &rest[..written],
        &payload,
    )
    .expect("a base header");
    head.copy_from_slice(&base);
    let header_area = slot.freeze();

    // The wire carries the area's bytes, then the payload from its own slot: they decode to
    // the chunk.
    let mut wire = header_area.bytes()[..HEADER_LEN + written].to_vec();
    wire.extend_from_slice(&payload);
    let frame = Frame::decode(&wire).expect("the frame decodes");
    let decoded = frame.activation(WireDType::F32).expect("an activation");
    assert_eq!(decoded.header, chunk.header);
    assert_eq!(decoded.payload, &payload[..]);
    assert_eq!(decoded.side.iter().count(), 2);
}

/// RFC-0031 §8 and §6.3: a chunk's parts land at consecutive offsets of the stage's input, and
/// the parts' payloads put back together are the chunk's.
#[test]
fn a_chunks_parts_land_at_consecutive_offsets() {
    const ROWS: u32 = 7;
    const HIDDEN: u32 = 16;
    let payload = rows(ROWS, HIDDEN);
    let spans = [Span {
        seq: SEQ,
        first_position: 100,
        n_positions: ROWS,
        row_offset: 0,
        tree: None,
    }];
    let chunk = Activation {
        header: header(ROWS, HIDDEN),
        spans: Spans::new(&spans),
        tree: TreeArea::new(&[]),
        side: SideItems::new(&[]),
        payload: &payload,
    };
    // A frame limit that fits two rows and the extension, so the chunk splits into four parts.
    let max_frame = HEADER_LEN + 128 + 2 * 64;
    let parts: Vec<_> = chunk.parts(max_frame).expect("parts").collect();
    assert_eq!(parts.len(), 4, "{parts:?}");
    let mut input = vec![0u8; payload.len()];
    let mut landing = Landing::new(input.len() as u64);
    for (i, part) in parts.iter().enumerate() {
        let mut out = vec![0; part.frame_len];
        chunk
            .encode_part(part, &spec(part.index), &mut out)
            .expect("encode a part");
        let frame = Frame::decode(&out).expect("a part decodes");
        let activation = frame.activation(WireDType::F32).expect("an activation");
        let landed = landing
            .land(&activation.header, activation.payload.len() as u64)
            .expect("the part lands");
        assert_eq!(landed.rows.start, u64::from(part.rows.start), "part {i}");
        assert_eq!(landed.rows.end, u64::from(part.rows.end), "part {i}");
        assert_eq!(landed.last, i == parts.len() - 1);
        input[landed.bytes.start as usize..landed.bytes.end as usize]
            .copy_from_slice(activation.payload);
    }
    assert_eq!(input, payload, "the parts landed back into the chunk");

    // Out of order: part 1 first is refused, and the landing starts over at the next part 0.
    let mut out_of_order = Landing::new(input.len() as u64);
    let second = &parts[1];
    let mut bytes = vec![0; second.frame_len];
    chunk
        .encode_part(second, &spec(1), &mut bytes)
        .expect("encode");
    let frame = Frame::decode(&bytes).expect("decode");
    let activation = frame.activation(WireDType::F32).expect("activation");
    let error = out_of_order
        .land(&activation.header, activation.payload.len() as u64)
        .unwrap_err();
    assert_eq!(error.class, syntropy_core::ErrorClass::InvalidRequest);
    let mut first = vec![0; parts[0].frame_len];
    chunk
        .encode_part(&parts[0], &spec(0), &mut first)
        .expect("encode");
    let frame = Frame::decode(&first).expect("decode");
    let activation = frame.activation(WireDType::F32).expect("activation");
    assert!(
        out_of_order
            .land(&activation.header, activation.payload.len() as u64)
            .is_ok()
    );
}

#[test]
fn a_part_whose_payload_is_not_its_rows_is_refused() {
    let header = header(3, 16);
    let stride = header.layout.stride().expect("F32");
    let mut landing = Landing::new(3 * stride);
    assert!(landing.land(&header, 3 * stride - 1).is_err());
    assert!(landing.land(&header, 3 * stride).is_ok());
}

/// RFC-0031 §8: a channel side's slot memory is `(recv_slots + 4 + retention) × slot_size`, the
/// four being the receives the channel core keeps for control frames (§7.1), and a core takes a
/// pool of that size as its receive pool.
#[test]
fn a_channel_pool_holds_its_receives_control_receives_and_retention() {
    use syntropy_core::Nanos;
    use syntropy_net::transport::core::{
        ACK_TIMEOUTS_TO_FAULT, CONTROL_RECEIVES, CRC_STORM_COUNT, MAX_CONTROL_FRAME,
    };
    use syntropy_net::transport::sizing::{channel_bytes, channel_slots};
    use syntropy_net::transport::{
        ChannelCore, ChannelKey, ChannelPurpose, CoreConfig, CreditCadence, SendConfig,
    };

    let (recv_slots, retention, slot_size) = (8u16, 8u16, 4096usize);
    let slots = channel_slots(recv_slots, retention);
    assert_eq!(slots, 8 + u32::from(CONTROL_RECEIVES) + 8);
    let pool = BufferPool::heap(PoolId(9), slot_size, slots).expect("a channel pool");
    assert_eq!(
        pool.slots() as usize * pool.slot_size(),
        channel_bytes(recv_slots, retention, slot_size)
    );
    assert_eq!(
        channel_bytes(recv_slots, retention, slot_size),
        (8 + 4 + 8) * 4096
    );
    assert!(slot_size >= MAX_CONTROL_FRAME);
    let core = ChannelCore::new(CoreConfig {
        key: ChannelKey {
            instance: InstanceId(1),
            epoch: PlanEpoch { term: 1, seq: 1 },
            purpose: ChannelPurpose::Activations,
            from: StageIx(0),
            to: StageIx(1),
            lane: 0,
        },
        wire_minor: WIRE_MINOR,
        cadence: CreditCadence::EveryFrame,
        alpha_p99: Nanos(1_000_000),
        recv_slots,
        slot_size: slot_size as u32,
        pool,
        send: Some(SendConfig {
            peer_recv_slots: recv_slots,
            peer_slot_size: slot_size as u32,
            retention,
        }),
        ack_timeouts_to_fault: ACK_TIMEOUTS_TO_FAULT,
        crc_storm_count: CRC_STORM_COUNT,
    });
    assert!(core.is_ok(), "{core:?}");
}

/// Review of #1833, B1: every span of a frame carries `TOKEN_IDS`, `ROUTER_HINT` (k = 8),
/// `SEQ_META` (10 B) and `TRUNCATE` (4 B), and the frame a `TRACE`. Over 1 to 512 spans of a
/// 512-row frame, the codec's base header and extension fit the plan-sized area.
#[test]
fn every_frame_within_its_limits_fits_its_header_area_over_1_to_512_spans() {
    use syntropy_net::frame::TraceContext;
    const ROWS: u32 = 512;
    let seq_meta = [0x5Au8; 10];
    let token_ids = le32s(0..ROWS);
    let experts: Vec<u8> = (0..ROWS * 8)
        .flat_map(|i| (i as u16).to_le_bytes())
        .collect();
    let payload = rows(ROWS, 16);
    let mut spans_tried = Vec::new();
    for n_spans in [1u32, 2, 4, 8, 16, 32, 64, 128, 256, 512] {
        let per = ROWS / n_spans;
        let spans: Vec<Span> = (0..n_spans)
            .map(|i| Span {
                seq: SequenceId {
                    ingress: 7,
                    boot: 1,
                    n: i,
                },
                first_position: 0,
                n_positions: per,
                row_offset: i * per,
                tree: None,
            })
            .collect();
        let mut side = Vec::new();
        for i in 0..n_spans {
            let rows = (i * per) as usize..((i + 1) * per) as usize;
            let span = Some(i as u16);
            side.push(Side {
                span,
                item: SideItem::TokenIds(&token_ids[rows.start * 4..rows.end * 4]),
            });
            side.push(Side {
                span,
                item: SideItem::RouterHint {
                    layer: 1,
                    k: 8,
                    experts: &experts[rows.start * 16..rows.end * 16],
                },
            });
            side.push(Side {
                span,
                item: SideItem::SeqMeta(&seq_meta),
            });
            side.push(Side {
                span,
                item: SideItem::Truncate { to: 3 },
            });
        }
        side.push(Side {
            span: None,
            item: SideItem::Trace(TraceContext {
                version: 0,
                trace_id: [1; 16],
                span_id: [2; 8],
                flags: 1,
            }),
        });
        let chunk = Activation {
            header: header(ROWS, 16),
            spans: Spans::new(&spans),
            tree: TreeArea::new(&[]),
            side: SideItems::new(&side),
            payload: &payload,
        };
        let needed = HEADER_LEN + chunk.ext_len().expect("a valid chunk");
        let area = header_area(&HeaderLimits {
            max_rows: ROWS,
            max_spans: n_spans,
            token_ids: true,
            router_k: Some(8),
            // `SEQ_META` and `TRUNCATE`: 10 + 4 bytes of values.
            items_per_span: 2,
            item_bytes_per_span: 14,
            // A `TRACE` is 26 bytes of value (RFC-0031 §6.3).
            frame_items: 1,
            frame_item_bytes: 26,
            ..HeaderLimits::default()
        });
        assert!(
            needed <= area,
            "{n_spans} spans need {needed} B, the area is {area} B"
        );
        spans_tried.push((n_spans, needed, area));
    }
    assert_eq!(spans_tried.len(), 10);
}

/// Review of #1833, N3: a part that would end past the stage input's bytes is refused.
#[test]
fn a_part_past_the_inputs_capacity_is_refused() {
    let stride = header(2, 16).layout.stride().expect("F32");
    let mut landing = Landing::new(5 * stride);
    let part = |index: u16, rows: u32| ActivationHeader {
        part_index: index,
        part_count: 3,
        ..header(rows, 16)
    };
    assert!(landing.land(&part(0, 2), 2 * stride).is_ok());
    assert!(landing.land(&part(1, 2), 2 * stride).is_ok());
    let error = landing.land(&part(2, 2), 2 * stride).unwrap_err();
    assert_eq!(error.class, syntropy_core::ErrorClass::InvalidRequest);
    // It starts over at the next part 0.
    assert!(landing.land(&part(0, 2), 2 * stride).is_ok());
}

/// A part's stride is the peer's: a part 0 whose rows fit the input but whose wider layout does
/// not is refused, and so is one whose rows times stride would overflow, without a panic.
#[test]
fn a_part_whose_layout_overruns_the_inputs_bytes_is_refused() {
    let planned = header(4, 16).layout.stride().expect("F32");
    let mut landing = Landing::new(4 * planned);
    let wide = header(2, 64);
    let wide_stride = wide.layout.stride().expect("F32");
    assert!(2 * wide_stride > 4 * planned);
    let error = landing.land(&wide, 2 * wide_stride).unwrap_err();
    assert_eq!(error.class, syntropy_core::ErrorClass::InvalidRequest);
    let huge = ActivationHeader {
        layout: RowLayout {
            streams: u8::MAX,
            ..header(u32::MAX, u32::MAX).layout
        },
        ..header(u32::MAX, u32::MAX)
    };
    assert!(
        u64::from(u32::MAX)
            .checked_mul(huge.layout.stride().expect("F32"))
            .is_none()
    );
    for payload_len in [0, 64, u64::MAX] {
        let error = landing.land(&huge, payload_len).unwrap_err();
        assert_eq!(error.class, syntropy_core::ErrorClass::InvalidRequest);
    }
    // The planned layout still lands.
    assert!(landing.land(&header(4, 16), 4 * planned).is_ok());
}
