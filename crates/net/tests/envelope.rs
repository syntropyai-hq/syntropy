// SPDX-License-Identifier: MIT OR Apache-2.0

//! The control envelope (RFC-0031 §9.1, DIST-E8-T12's acceptance criteria).
//!
//! - Identical state serialises to identical bytes across runs and platforms: golden vectors over
//!   test types that exercise each encoding rule, checked in here and compared on every OS the
//!   tests run on.
//! - Unknown fields are ignored.
//! - Decode is total over arbitrary bytes: arbitrary input never panics and returns a classified
//!   error.

use std::collections::BTreeMap;

use proptest::prelude::*;
use serde::{Deserialize, Serialize, Serializer};
use syntropy_core::ErrorClass;
use syntropy_net::ctl::envelope::{
    self, COMPRESS_ABOVE, DecodeError, EncodeError, Encoding, LEN_BYTES, MAX_BODY,
};

/// A value that touches every encoding rule: integers at each width boundary, signed and
/// unsigned; strings and byte arrays; `Option`; a float; nested structs; a `BTreeMap` with
/// integer keys, including negative ones, whose `Ord` order differs from RFC 8949's bytewise
/// order; a `BTreeMap` with string keys of different lengths; and each enum shape.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct State {
    small: u8,
    widths: Vec<u64>,
    signed: Vec<i64>,
    name: String,
    raw: [u8; 4],
    maybe: Option<u32>,
    nothing: Option<u32>,
    ratio: f64,
    inner: Inner,
    by_signed: BTreeMap<i64, bool>,
    by_name: BTreeMap<String, u16>,
    kinds: Vec<Kind>,
    pair: (u8, String),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Inner {
    term: u64,
    flag: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
enum Kind {
    Unit,
    Newtype(u32),
    Tuple(u8, u8),
    Named { id: u64 },
}

fn state() -> State {
    State {
        small: 7,
        widths: vec![
            0,
            23,
            24,
            255,
            256,
            65_535,
            65_536,
            u64::from(u32::MAX),
            u64::from(u32::MAX) + 1,
            u64::MAX,
        ],
        signed: vec![-1, -24, -25, -256, -257, i64::MIN, i64::MAX],
        name: "node-a".to_owned(),
        raw: [0xde, 0xad, 0xbe, 0xef],
        maybe: Some(42),
        nothing: None,
        ratio: 1.5,
        inner: Inner {
            term: 9,
            flag: true,
        },
        by_signed: BTreeMap::from([(-2, true), (-1, false), (0, true), (1, false)]),
        by_name: BTreeMap::from([
            ("b".to_owned(), 1),
            ("aa".to_owned(), 2),
            ("a".to_owned(), 3),
        ]),
        kinds: vec![
            Kind::Unit,
            Kind::Newtype(5),
            Kind::Tuple(1, 2),
            Kind::Named { id: 300 },
        ],
        pair: (3, "x".to_owned()),
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// `state()`'s CBOR body. A change here changes the bytes every control message is made of, and
/// the event log's hash chain with them: it needs a reason in the change that makes it.
const GOLDEN_BODY: &str = concat!(
    "ad65736d616c6c07667769647468738a0017181818ff19010019ffff1a000100001affffffff1b00",
    "000001000000001bffffffffffffffff667369676e6564872037381838ff3901003b7fffffffffff",
    "ffff1b7fffffffffffffff646e616d65666e6f64652d61637261778418de18ad18be18ef656d6179",
    "6265182a676e6f7468696e67f665726174696ff93e0065696e6e6572a2647465726d0964666c6167",
    "f56962795f7369676e6564a421f520f400f501f46762795f6e616d65a36161036261610261620165",
    "6b696e64738464556e6974a1674e65777479706505a1655475706c65820102a1654e616d6564a162",
    "696419012c647061697282036178",
);

/// The first five bytes of `state()` as a stream message: `len` (255, little-endian) and `enc` (0,
/// plain CBOR).
const GOLDEN_FRAME_PREFIX: &str = "ff00000000";

#[test]
fn the_same_state_gives_the_golden_bytes() {
    let body = envelope::to_cbor(&state()).unwrap();
    assert_eq!(hex(&body), GOLDEN_BODY);
    let frame = envelope::encode(&state()).unwrap();
    assert_eq!(hex(&frame[..LEN_BYTES + 1]), GOLDEN_FRAME_PREFIX);
    assert_eq!(&frame[LEN_BYTES + 1..], body.as_slice());
    assert_eq!(envelope::decode::<State>(&frame).unwrap(), state());
}

#[test]
fn map_keys_follow_btreemap_order_not_rfc_8949_bytewise_order() {
    // RFC-0031 §9.1 sorts by the key's `Ord`: -2, -1, 0, 1 and "a", "aa", "b". RFC 8949 §4.2.1
    // would put 0 (0x00) before -1 (0x20), and "b" (0x61 62) before "aa" (0x62 61 61).
    let body = envelope::to_cbor(&BTreeMap::from([(1i64, 0u8), (-1, 0), (0, 0), (-2, 0)])).unwrap();
    assert_eq!(hex(&body), "a42100200000000100");
    let body = envelope::to_cbor(&BTreeMap::from([("b", 0u8), ("aa", 0), ("a", 0)])).unwrap();
    assert_eq!(hex(&body), "a3616100626161006162 00".replace(' ', ""));
}

#[test]
fn integers_take_their_shortest_form() {
    for (value, expected) in [
        (0u64, "00"),
        (23, "17"),
        (24, "1818"),
        (255, "18ff"),
        (256, "190100"),
        (65_535, "19ffff"),
        (65_536, "1a00010000"),
        (u64::from(u32::MAX), "1affffffff"),
        (u64::from(u32::MAX) + 1, "1b0000000100000000"),
    ] {
        assert_eq!(
            hex(&envelope::to_cbor(&value).unwrap()),
            expected,
            "{value}"
        );
    }
    for (value, expected) in [(-1i64, "20"), (-24, "37"), (-25, "3818"), (-257, "390100")] {
        assert_eq!(
            hex(&envelope::to_cbor(&value).unwrap()),
            expected,
            "{value}"
        );
    }
}

/// A sequence whose size serde cannot know in advance: ciborium would write it with an
/// indefinite length.
struct UnknownLength(Vec<u32>);

impl Serialize for UnknownLength {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.0.iter().filter(|_| true))
    }
}

#[test]
fn an_indefinite_length_is_refused_not_sent() {
    let error = envelope::to_cbor(&UnknownLength(vec![1, 2, 3])).unwrap_err();
    assert_eq!(error, EncodeError::NotDeterministic { at: 0 });
    assert_eq!(error.class(), ErrorClass::Internal);
    assert!(envelope::encode(&UnknownLength(vec![1])).is_err());
}

/// `Inner` as a newer build might send it, with a field this build lacks.
#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct InnerV2 {
    term: u64,
    flag: bool,
    lease_ms: u32,
}

/// `Inner` as a newer build reads it: the new field has a default.
#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct InnerWithDefault {
    term: u64,
    flag: bool,
    #[serde(default)]
    lease_ms: u32,
}

#[test]
fn unknown_fields_are_ignored_and_missing_ones_take_their_default() {
    let newer = envelope::encode(&InnerV2 {
        term: 3,
        flag: true,
        lease_ms: 500,
    })
    .unwrap();
    assert_eq!(
        envelope::decode::<Inner>(&newer).unwrap(),
        Inner {
            term: 3,
            flag: true
        }
    );
    let older = envelope::encode(&Inner {
        term: 3,
        flag: true,
    })
    .unwrap();
    assert_eq!(
        envelope::decode::<InnerWithDefault>(&older).unwrap(),
        InnerWithDefault {
            term: 3,
            flag: true,
            lease_ms: 0
        }
    );
}

/// A body of `len` bytes of CBOR: a text string of 'a's, whose head takes `head` bytes.
fn text_of_body_len(len: usize) -> String {
    let head = match len {
        0..=24 => 1,
        25..=257 => 2,
        258..=65_538 => 3,
        _ => 5,
    };
    "a".repeat(len - head)
}

#[test]
fn bodies_over_64_kib_are_compressed_and_the_rest_are_not() {
    let at_limit = text_of_body_len(COMPRESS_ABOVE);
    assert_eq!(envelope::to_cbor(&at_limit).unwrap().len(), COMPRESS_ABOVE);
    let frame = envelope::encode(&at_limit).unwrap();
    assert_eq!(frame[LEN_BYTES], Encoding::Cbor as u8);
    assert_eq!(envelope::decode::<String>(&frame).unwrap(), at_limit);

    let over = text_of_body_len(COMPRESS_ABOVE + 1);
    let frame = envelope::encode(&over).unwrap();
    assert_eq!(frame[LEN_BYTES], Encoding::ZstdCbor as u8);
    assert!(frame.len() < 1024, "{} bytes", frame.len());
    assert_eq!(envelope::decode::<String>(&frame).unwrap(), over);
    // Compression is deterministic too, within one zstd version.
    assert_eq!(envelope::encode(&over).unwrap(), frame);
}

#[test]
fn a_body_over_64_mib_is_refused_on_both_sides() {
    let over = text_of_body_len(MAX_BODY + 1);
    assert_eq!(
        envelope::encode(&over).unwrap_err(),
        EncodeError::TooLarge { len: MAX_BODY + 1 }
    );
    // A `len` field that declares more is refused before anything is read or allocated.
    let declared = u32::try_from(MAX_BODY + 2).unwrap().to_le_bytes();
    assert_eq!(
        envelope::message_len(declared).unwrap_err(),
        DecodeError::TooLarge { len: MAX_BODY + 1 }
    );
    // A small compressed body that expands past the limit is cut off at the limit.
    let bomb = zstd::bulk::compress(&vec![0u8; MAX_BODY + 1], 3).unwrap();
    let mut frame = u32::try_from(1 + bomb.len())
        .unwrap()
        .to_le_bytes()
        .to_vec();
    frame.push(Encoding::ZstdCbor as u8);
    frame.extend_from_slice(&bomb);
    assert_eq!(
        envelope::decode::<String>(&frame).unwrap_err(),
        DecodeError::TooLarge { len: MAX_BODY + 1 }
    );
}

#[test]
fn frames_are_checked_against_their_len_field() {
    let frame = envelope::encode(&Inner {
        term: 1,
        flag: false,
    })
    .unwrap();
    assert!(matches!(
        envelope::decode::<Inner>(&frame[..frame.len() - 1]),
        Err(DecodeError::Truncated { .. })
    ));
    let mut longer = frame.clone();
    longer.push(0);
    assert_eq!(
        envelope::decode::<Inner>(&longer).unwrap_err(),
        DecodeError::TrailingBytes { extra: 1 }
    );
    let mut unknown = frame.clone();
    unknown[LEN_BYTES] = 7;
    let error = envelope::decode::<Inner>(&unknown).unwrap_err();
    assert_eq!(error, DecodeError::UnknownEncoding(7));
    assert_eq!(error.class(), ErrorClass::Unsupported);
    assert_eq!(
        envelope::message_len([0, 0, 0, 0]).unwrap_err(),
        DecodeError::Truncated { needed: 1, got: 0 }
    );
}

#[test]
fn datagrams_carry_a_tag_and_respect_the_path_limit() {
    let inner = Inner {
        term: 2,
        flag: true,
    };
    let datagram = envelope::encode_datagram(9, &inner, 1200).unwrap();
    assert_eq!(datagram[0], 9);
    assert_eq!(
        &datagram[1..],
        envelope::to_cbor(&inner).unwrap().as_slice()
    );
    assert_eq!(
        envelope::decode_datagram::<Inner>(&datagram).unwrap(),
        (9, inner.clone())
    );
    assert_eq!(
        envelope::split_datagram(&datagram).unwrap(),
        (9, &datagram[1..])
    );
    let error = envelope::encode_datagram(9, &inner, 4).unwrap_err();
    assert_eq!(
        error,
        EncodeError::DatagramTooLarge {
            len: datagram.len(),
            limit: 4
        }
    );
    assert!(envelope::decode_datagram::<Inner>(&[]).is_err());
}

#[test]
fn a_mismatched_type_is_an_invalid_request_and_damage_is_integrity() {
    let frame = envelope::encode(&"text").unwrap();
    let error = envelope::decode::<Inner>(&frame).unwrap_err();
    assert_eq!(error.class(), ErrorClass::InvalidRequest, "{error}");
    let mut damaged = envelope::encode(&state()).unwrap();
    damaged.truncate(damaged.len() - 3);
    let len = u32::try_from(damaged.len() - LEN_BYTES).unwrap();
    damaged[..LEN_BYTES].copy_from_slice(&len.to_le_bytes());
    let error = envelope::decode::<State>(&damaged).unwrap_err();
    assert_eq!(error.class(), ErrorClass::Integrity, "{error}");
}

fn classified(error: &DecodeError) -> bool {
    matches!(
        error.class(),
        ErrorClass::Integrity | ErrorClass::InvalidRequest | ErrorClass::Unsupported
    )
}

fn arb_kind() -> impl Strategy<Value = Kind> {
    prop_oneof![
        Just(Kind::Unit),
        any::<u32>().prop_map(Kind::Newtype),
        (any::<u8>(), any::<u8>()).prop_map(|(a, b)| Kind::Tuple(a, b)),
        any::<u64>().prop_map(|id| Kind::Named { id }),
    ]
}

prop_compose! {
    fn arb_state()(
        small in any::<u8>(),
        widths in prop::collection::vec(any::<u64>(), 0..8),
        signed in prop::collection::vec(any::<i64>(), 0..8),
        name in ".{0,24}",
        raw in any::<[u8; 4]>(),
        maybe in any::<Option<u32>>(),
        ratio in -1.0e9f64..1.0e9,
        term in any::<u64>(),
        flag in any::<bool>(),
        by_signed in prop::collection::btree_map(any::<i64>(), any::<bool>(), 0..6),
        by_name in prop::collection::btree_map(".{0,6}", any::<u16>(), 0..6),
        kinds in prop::collection::vec(arb_kind(), 0..5),
        pair in (any::<u8>(), ".{0,4}"),
    ) -> State {
        State {
            small, widths, signed, name, raw, maybe, nothing: None, ratio,
            inner: Inner { term, flag }, by_signed, by_name, kinds, pair,
        }
    }
}

proptest! {
    /// Decode is total over arbitrary bytes: whole frames, the bytes after `len`, datagrams and
    /// bare CBOR bodies all return a value or a classified error, and never panic.
    #[test]
    fn decode_is_total_over_arbitrary_bytes(bytes in prop::collection::vec(any::<u8>(), 0..512)) {
        for error in [
            envelope::decode::<State>(&bytes).err(),
            envelope::decode_body::<State>(&bytes).err(),
            envelope::decode_datagram::<State>(&bytes).err(),
            envelope::from_cbor::<State>(&bytes).err(),
            envelope::from_cbor::<BTreeMap<String, Vec<u64>>>(&bytes).err(),
        ]
        .into_iter()
        .flatten()
        {
            prop_assert!(classified(&error), "{error}");
        }
    }

    /// The same with a valid `len` and a known `enc` byte, so the body parsers see arbitrary
    /// input rather than stopping at the frame checks.
    #[test]
    fn decode_is_total_over_arbitrary_bodies(
        compressed in any::<bool>(),
        body in prop::collection::vec(any::<u8>(), 0..512),
    ) {
        let mut frame = u32::try_from(1 + body.len()).unwrap().to_le_bytes().to_vec();
        frame.push(if compressed { Encoding::ZstdCbor as u8 } else { Encoding::Cbor as u8 });
        frame.extend_from_slice(&body);
        if let Err(error) = envelope::decode::<State>(&frame) {
            prop_assert!(classified(&error), "{error}");
        }
    }

    /// Valid frames with bytes flipped decode to a value or a classified error.
    #[test]
    fn damaged_frames_are_classified(
        state in arb_state(),
        flips in prop::collection::vec((any::<prop::sample::Index>(), any::<u8>()), 1..4),
    ) {
        let mut frame = envelope::encode(&state).unwrap();
        for (at, byte) in flips {
            let i = at.index(frame.len());
            frame[i] ^= byte | 1;
        }
        if let Err(error) = envelope::decode::<State>(&frame) {
            prop_assert!(classified(&error), "{error}");
        }
    }

    /// Every state round-trips, and encoding it twice gives the same bytes.
    #[test]
    fn states_round_trip_and_encode_deterministically(state in arb_state()) {
        let frame = envelope::encode(&state).unwrap();
        prop_assert_eq!(envelope::encode(&state).unwrap(), frame.clone());
        prop_assert_eq!(envelope::decode::<State>(&frame).unwrap(), state);
    }
}
