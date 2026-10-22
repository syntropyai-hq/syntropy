// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `EmbeddingResult` codec against the golden frames in `tests/golden/embedding/`
//! (DIST-E8-T15; RFC-0031 §6.14, RFC-0050 amendment A5).
//!
//! The acceptance criteria this file checks:
//! - golden `EmbeddingResult` frames round-trip with `dims`, `pooling`, `normalised` and
//!   `prompt_tokens` recorded. The goldens are laid out from RFC-0031 §6.14's table by [`spec`],
//!   independently of the codec. They include a 4096-dimension F32 vector split across two parts
//!   of a 16 KiB slot, which the codec's splitter must produce byte for byte and its assembler
//!   must put back together;
//! - hostile input is refused with an error that names the frame, in a class RFC-0060 §10.2
//!   allows, and never panics;
//! - the encoder refuses to write the frame for a channel whose negotiated capabilities lack bit
//!   14 `EMBEDDING`, so a peer without it never receives one.
//!
//! `SYNTROPY_BLESS_GOLDEN=1` rewrites the files, as it does the other goldens.

use std::collections::BTreeSet;
use std::path::PathBuf;

use proptest::prelude::*;
use syntropy_core::{ErrorClass, InstanceId, SequenceId};
use syntropy_net::frame::{
    EmbeddingResult, EmbeddingSpec, Flags, Frame, FrameError, FrameType, Pooling, WIRE_MINOR,
    assemble, capability,
};

use super::{INSTANCE, REFUSAL_CLASSES, base, epoch, golden_dir, leak, reseal, spec};

/// The ingress node the results go to.
const X: u64 = 0x0102_0304_0506_0708;

/// The slot of the `Tokens` channel the 4096-dimension golden is split for.
const SLOT: usize = 16 << 10;

fn seq(n: u32) -> SequenceId {
    SequenceId {
        ingress: X,
        boot: 2,
        n,
    }
}

fn spec_for(step: u64, chan_seq: u64) -> EmbeddingSpec {
    EmbeddingSpec {
        wire_minor: WIRE_MINOR,
        flags: Flags::default(),
        instance: InstanceId(INSTANCE),
        epoch: epoch(),
        step,
        chan_seq,
    }
}

/// A golden frame: its bytes, and the `EmbeddingResult` they decode to.
struct Golden {
    name: &'static str,
    about: &'static str,
    bytes: Vec<u8>,
    spec: EmbeddingSpec,
    result: EmbeddingResult<'static>,
}

/// The golden's bytes from wire values, laid out by [`spec`].
fn golden(
    name: &'static str,
    about: &'static str,
    spec: EmbeddingSpec,
    result: EmbeddingResult<'static>,
) -> Golden {
    let ext = spec::embedding_ext(&spec::EmbeddingFields {
        ingress: result.seq.ingress,
        boot: result.seq.boot,
        n: result.seq.n,
        dims: result.dims,
        dtype: 1,
        pooling: match result.pooling {
            Pooling::LastToken => 1,
            Pooling::Mean => 2,
            Pooling::Cls => 3,
        },
        normalised: u8::from(result.normalised),
        prompt_tokens: result.prompt_tokens,
        part_index: result.part_index,
        part_count: result.part_count,
    });
    let bytes = spec::frame(
        &base(0x12, 0, spec.step, spec.chan_seq),
        &ext,
        result.vector,
    );
    Golden {
        name,
        about,
        bytes,
        spec,
        result,
    }
}

/// A whole result for sequence `n`.
fn whole(
    n: u32,
    dims: u32,
    pooling: Pooling,
    normalised: bool,
    prompt_tokens: u32,
) -> EmbeddingResult<'static> {
    EmbeddingResult {
        seq: seq(n),
        dims,
        pooling,
        normalised,
        prompt_tokens,
        part_index: 0,
        part_count: 1,
        vector: leak(spec::f32_elems(dims as usize, 1)),
    }
}

/// The 4096-dimension vector, whole.
fn big() -> EmbeddingResult<'static> {
    whole(9, 4096, Pooling::Mean, true, 512)
}

/// Part `index` of [`big`] in a 16 KiB slot, laid out by hand: 16,384 bytes less the header and
/// extension leave room for 4,064 elements, so part 0 holds elements 0..4064 and part 1 the last
/// 32.
fn big_part(index: u16) -> EmbeddingResult<'static> {
    let (start, end) = if index == 0 { (0, 4064) } else { (4064, 4096) };
    EmbeddingResult {
        part_index: index,
        part_count: 2,
        vector: leak(spec::f32_elems(end - start, 1 + start)),
        ..big()
    }
}

fn goldens() -> Vec<Golden> {
    vec![
        golden(
            "embedding-mean",
            "a mean-pooled, normalised 8-dimension vector for a 12-token prompt, in one part",
            spec_for(30, 1),
            whole(3, 8, Pooling::Mean, true, 12),
        ),
        golden(
            "embedding-last-token",
            "a last-token 4-dimension vector, not normalised, for a 1-token prompt",
            spec_for(31, 2),
            whole(4, 4, Pooling::LastToken, false, 1),
        ),
        golden(
            "embedding-cls",
            "a CLS-pooled, normalised 3-dimension vector for a 70,000-token prompt",
            spec_for(32, 3),
            whole(5, 3, Pooling::Cls, true, 70_000),
        ),
        golden(
            "embedding-4096-part0",
            "part 0 of 2 of a mean-pooled 4096-dimension F32 vector in a 16 KiB slot: elements \
             0..4064",
            spec_for(33, 4),
            big_part(0),
        ),
        golden(
            "embedding-4096-part1",
            "part 1 of 2 of the 4096-dimension vector: elements 4064..4096",
            spec_for(33, 5),
            big_part(1),
        ),
    ]
}

fn embedding_dir() -> PathBuf {
    golden_dir().join("embedding")
}

fn read_golden(name: &str) -> Vec<u8> {
    let path = embedding_dir().join(format!("{name}.hex"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    spec::from_hex(&text)
}

#[test]
fn embedding_golden_files_match_the_spec_tables() {
    let goldens = goldens();
    if std::env::var_os("SYNTROPY_BLESS_GOLDEN").is_some() {
        std::fs::create_dir_all(embedding_dir()).unwrap();
        for golden in &goldens {
            let text = spec::to_hex(golden.name, golden.about, &golden.bytes);
            std::fs::write(embedding_dir().join(format!("{}.hex", golden.name)), text).unwrap();
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
    let on_disk: BTreeSet<String> = std::fs::read_dir(embedding_dir())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .filter(|name| name.ends_with(".hex"))
        .collect();
    assert_eq!(on_disk, listed);
}

/// The criterion "golden frames round-trip with `dims`, `pooling`, `normalised` and
/// `prompt_tokens` recorded".
#[test]
fn embedding_goldens_round_trip() {
    for golden in goldens() {
        let name = golden.name;
        let bytes = read_golden(name);
        let frame = Frame::decode(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            frame.header.spec.ftype,
            FrameType::EmbeddingResult,
            "{name}"
        );
        let result = frame
            .embedding_result()
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(result, golden.result, "{name}");
        assert_eq!(
            golden
                .result
                .encode_to_vec(&golden.spec, capability::EMBEDDING),
            Ok(bytes.clone()),
            "{name}"
        );
        let mut slot = vec![0xAA; bytes.len() + 64];
        assert_eq!(
            golden
                .result
                .encode(&golden.spec, capability::EMBEDDING, &mut slot),
            Ok(bytes.len()),
            "{name}"
        );
        assert_eq!(&slot[..bytes.len()], &bytes[..], "{name}");
    }
    // The fields are where §6.14 puts them, read back from the mean golden's bytes.
    let bytes = read_golden("embedding-mean");
    let ext = &bytes[64..128];
    assert_eq!(u32::from_le_bytes(ext[16..20].try_into().unwrap()), 8); // dims
    assert_eq!((ext[20], ext[21], ext[22]), (1, 2, 1)); // F32, mean, normalised
    assert_eq!(u32::from_le_bytes(ext[24..28].try_into().unwrap()), 12); // prompt_tokens
    assert_eq!(&ext[28..32], &[0, 0, 1, 0]); // part 0 of 1
    let result = Frame::decode(&bytes).unwrap().embedding_result().unwrap();
    let elements: Vec<f32> = result.elements().collect();
    assert_eq!(elements, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

/// A 4096-dimension F32 vector, 16 KiB, does not fit a 16 KiB slot beside the header and the
/// extension: the splitter writes it as the two golden parts, and the receiver's assembler,
/// given them in either order, gives back the whole vector.
#[test]
fn a_vector_larger_than_the_slot_splits_into_the_golden_parts() {
    let whole = big();
    assert!(whole.frame_len() > SLOT);
    let parts: Vec<EmbeddingResult<'static>> = whole.parts(SLOT).unwrap().collect();
    assert_eq!(parts.len(), 2);
    for (part, name) in parts
        .iter()
        .zip(["embedding-4096-part0", "embedding-4096-part1"])
    {
        let chan_seq = 4 + u64::from(part.part_index);
        let bytes = part.encode_to_vec(&spec_for(33, chan_seq), capability::EMBEDDING);
        assert_eq!(bytes, Ok(read_golden(name)), "{name}");
        assert!(part.frame_len() <= SLOT);
    }
    assert_eq!(parts[0].frame_len(), SLOT);
    let expected: Vec<f32> = whole.elements().collect();
    let goldens: Vec<Vec<u8>> = ["embedding-4096-part1", "embedding-4096-part0"]
        .into_iter()
        .map(read_golden)
        .collect();
    let received: Vec<EmbeddingResult<'_>> = goldens
        .iter()
        .map(|bytes| Frame::decode(bytes).unwrap().embedding_result().unwrap())
        .collect();
    assert_eq!(assemble(&received), Ok(expected.clone()));
    assert_eq!(assemble(&parts), Ok(expected));
    // A vector that fits is one part, the whole result.
    let small = whole_small();
    assert_eq!(small.parts(SLOT).unwrap().collect::<Vec<_>>(), [small]);
    assert_eq!(assemble(&[small]), Ok(small.elements().collect()));
}

fn whole_small() -> EmbeddingResult<'static> {
    whole(3, 8, Pooling::Mean, true, 12)
}

/// The criterion "the encoder refuses a channel without bit 14 `EMBEDDING`": a channel that did
/// not negotiate the capability never receives the frame, whatever else it negotiated.
#[test]
fn the_encoder_refuses_a_channel_without_the_embedding_capability() {
    let result = whole_small();
    let refused = FrameError::CapabilityNotNegotiated {
        ftype: FrameType::EmbeddingResult,
        capability: 1 << 14,
    };
    for capabilities in [0, !capability::EMBEDDING, 0b1011] {
        assert_eq!(
            result.encode_to_vec(&spec_for(30, 1), capabilities),
            Err(refused.clone()),
            "{capabilities:#x}"
        );
        let mut out = vec![0; 1024];
        assert_eq!(
            result.encode(&spec_for(30, 1), capabilities, &mut out),
            Err(refused.clone())
        );
    }
    assert_eq!(refused.class(), ErrorClass::Unsupported);
    assert_eq!(
        refused.to_string(),
        "EmbeddingResult needs capability bit 14, which the channel did not negotiate"
    );
    assert!(
        result
            .encode_to_vec(&spec_for(30, 1), capability::EMBEDDING | 0b1011)
            .is_ok()
    );
    assert_eq!(capability::EMBEDDING, 1 << 14);
}

fn field(field: &'static str, value: u64) -> FrameError {
    FrameError::BadField {
        ftype: FrameType::EmbeddingResult,
        field,
        value,
    }
}

fn inconsistent(field: &'static str, value: u64, expected: u64) -> FrameError {
    FrameError::Inconsistent {
        ftype: FrameType::EmbeddingResult,
        field,
        value,
        expected,
    }
}

/// An `EmbeddingResult` frame as wire values, to break one rule at a time.
struct Raw {
    ftype: u8,
    flags: u16,
    fields: spec::EmbeddingFields,
    ext_len: usize,
    payload: Vec<u8>,
}

impl Raw {
    fn new() -> Raw {
        Raw {
            ftype: 0x12,
            flags: 0,
            fields: spec::EmbeddingFields {
                ingress: X,
                boot: 2,
                n: 3,
                dims: 8,
                dtype: 1,
                pooling: 2,
                normalised: 1,
                prompt_tokens: 12,
                part_index: 0,
                part_count: 1,
            },
            ext_len: 64,
            payload: spec::f32_elems(8, 1),
        }
    }

    fn bytes(&self) -> Vec<u8> {
        let mut ext = spec::embedding_ext(&self.fields);
        ext.resize(self.ext_len, 0);
        spec::frame(&base(self.ftype, self.flags, 30, 1), &ext, &self.payload)
    }
}

fn case(edit: fn(&mut Raw)) -> Vec<u8> {
    let mut raw = Raw::new();
    edit(&mut raw);
    raw.bytes()
}

/// Every rule the codec checks, broken one at a time, is refused with the error that names it,
/// in a class RFC-0060 §10.2 allows, and never `Internal`.
#[test]
fn an_embedding_frame_breaking_a_rule_is_refused_with_a_classified_error() {
    let cases: Vec<(&str, Vec<u8>, FrameError)> = vec![
        (
            "a dtype other than F32",
            case(|r| r.fields.dtype = 2),
            field("dtype", 2),
        ),
        (
            "pooling 0",
            case(|r| r.fields.pooling = 0),
            field("pooling", 0),
        ),
        (
            "pooling 4",
            case(|r| r.fields.pooling = 4),
            field("pooling", 4),
        ),
        (
            "normalised 2",
            case(|r| r.fields.normalised = 2),
            field("normalised", 2),
        ),
        (
            "dims 0",
            case(|r| {
                r.fields.dims = 0;
                r.payload = vec![];
            }),
            field("dims", 0),
        ),
        (
            "part_count 0, what the bytes held while they were reserved",
            case(|r| r.fields.part_count = 0),
            field("part_count", 0),
        ),
        (
            "part_index at part_count",
            case(|r| {
                r.fields.part_index = 2;
                r.fields.part_count = 2;
            }),
            field("part_index", 2),
        ),
        (
            "a payload that is not whole elements",
            case(|r| {
                r.fields.part_count = 2;
                r.payload.truncate(6);
            }),
            inconsistent("payload_len", 6, 32),
        ),
        (
            "an empty part",
            case(|r| {
                r.fields.part_count = 2;
                r.payload.clear();
            }),
            inconsistent("payload_len", 0, 32),
        ),
        (
            "a one-part result short of dims",
            case(|r| r.payload.truncate(28)),
            inconsistent("payload_len", 28, 32),
        ),
        (
            "a part longer than the whole vector",
            case(|r| {
                r.fields.part_count = 2;
                r.payload = spec::f32_elems(9, 1);
            }),
            inconsistent("payload_len", 36, 32),
        ),
        (
            "HAS_TRACE on a frame with no TLV area",
            case(|r| r.flags = 1),
            inconsistent("HAS_TRACE", 1, 0),
        ),
        (
            "an extension shorter than its fields",
            case(|r| r.ext_len = 0),
            FrameError::ExtTooShort {
                ftype: FrameType::EmbeddingResult,
                len: 0,
                needed: 32,
            },
        ),
        (
            "a Token frame read as an EmbeddingResult",
            case(|r| r.ftype = 0x02),
            FrameError::WrongType {
                expected: FrameType::EmbeddingResult,
                found: FrameType::Token,
            },
        ),
    ];
    for (name, bytes, error) in cases {
        let frame = Frame::decode(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        let refused = frame.embedding_result().expect_err(name);
        assert_eq!(refused, error, "{name}");
        assert!(
            REFUSAL_CLASSES.contains(&refused.class()),
            "{name}: {:?}",
            refused.class()
        );
        assert!(
            refused.to_string().contains("EmbeddingResult"),
            "{name}: {refused}"
        );
    }
    // A sealed frame is opened first.
    let mut sealed = case(|r| r.flags = 1 << 2);
    let tag = [0u8; 16];
    // Give the sealed frame a payload that holds the AEAD tag, then fix its CRCs.
    sealed.extend_from_slice(&tag);
    let len = (sealed.len() - 128) as u32;
    sealed[52..56].copy_from_slice(&len.to_le_bytes());
    reseal(&mut sealed);
    let frame = Frame::decode(&sealed).unwrap();
    assert_eq!(
        frame.embedding_result(),
        Err(FrameError::Sealed(FrameType::EmbeddingResult))
    );
}

/// The reserved byte at offset 23, the padding, and an extension a later minor makes longer are
/// ignored.
#[test]
fn reserved_bytes_and_later_additions_are_ignored() {
    let plain = Raw::new().bytes();
    let expected = Frame::decode(&plain).unwrap().embedding_result().unwrap();
    let mut raw = Raw::new();
    raw.ext_len = 128;
    let mut bytes = raw.bytes();
    bytes[64 + 23] = 0xEE; // reserved
    // The padding, and a second 64 bytes of later fields.
    bytes[64 + 32..64 + 128].fill(0x5A);
    reseal(&mut bytes);
    let frame = Frame::decode(&bytes).unwrap();
    assert_eq!(frame.embedding_result(), Ok(expected));
}

/// The encoder checks what the decoder checks, so it never writes a frame a receiver refuses.
#[test]
fn the_encoder_refuses_what_the_decoder_would() {
    let good = whole_small();
    let edits: Vec<(&str, EmbeddingResult<'static>, FrameError)> = vec![
        (
            "dims 0",
            EmbeddingResult { dims: 0, ..good },
            field("dims", 0),
        ),
        (
            "part_count 0",
            EmbeddingResult {
                part_count: 0,
                ..good
            },
            field("part_count", 0),
        ),
        (
            "part_index at part_count",
            EmbeddingResult {
                part_index: 1,
                ..good
            },
            field("part_index", 1),
        ),
        (
            "a vector of 7 elements for dims 8",
            EmbeddingResult {
                vector: &good.vector[..28],
                ..good
            },
            inconsistent("payload_len", 28, 32),
        ),
        (
            "a vector that is not whole elements",
            EmbeddingResult {
                vector: &good.vector[..30],
                part_count: 2,
                ..good
            },
            inconsistent("payload_len", 30, 32),
        ),
    ];
    for (name, result, error) in edits {
        assert_eq!(
            result.encode_to_vec(&spec_for(30, 1), capability::EMBEDDING),
            Err(error.clone()),
            "{name}"
        );
    }
    let traced = EmbeddingSpec {
        flags: Flags(Flags::HAS_TRACE),
        ..spec_for(30, 1)
    };
    assert_eq!(
        good.encode_to_vec(&traced, capability::EMBEDDING),
        Err(inconsistent("HAS_TRACE", 1, 0))
    );
    // Only a whole result is split, and a slot too small for one element is refused.
    assert_eq!(big_part(0).parts(SLOT).err(), Some(field("part_count", 2)));
    assert!(matches!(
        good.parts(128),
        Err(FrameError::TooLong {
            what: "embedding part",
            ..
        })
    ));
    assert_eq!(good.parts(132).map(|parts| parts.len()), Ok(8));
}

/// The receiver acts only on a whole result: missing, repeated and disagreeing parts are refused.
#[test]
fn assembly_needs_every_part_once_and_in_agreement() {
    let (a, b) = (big_part(0), big_part(1));
    assert_eq!(assemble(&[]), Err(inconsistent("part_count", 0, 1)));
    assert_eq!(assemble(&[a]), Err(inconsistent("part_count", 1, 2)));
    assert_eq!(assemble(&[a, a]), Err(field("part_index", 0)));
    let other_seq = EmbeddingResult { seq: seq(10), ..b };
    assert_eq!(assemble(&[a, other_seq]), Err(field("seq", 1)));
    let other_pooling = EmbeddingResult {
        pooling: Pooling::Cls,
        ..b
    };
    assert_eq!(assemble(&[a, other_pooling]), Err(field("pooling", 1)));
    let short = EmbeddingResult {
        vector: &b.vector[..64],
        ..b
    };
    assert_eq!(assemble(&[a, short]), Err(inconsistent("dims", 4080, 4096)));
}

/// A flipped bit anywhere in a golden is caught: by the CRCs, or by the checks they cover. The
/// small goldens are flipped at every bit; the two 16 KiB parts are left to the property test.
#[test]
fn a_corrupted_embedding_byte_is_detected() {
    for golden in goldens()
        .into_iter()
        .filter(|g| !g.name.starts_with("embedding-4096"))
    {
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

proptest! {
    /// The parser on arbitrary extensions and payloads: no panic, every refusal is classified,
    /// and an accepted frame encodes to one that decodes to it again.
    #[test]
    fn arbitrary_embedding_bodies_parse_safely(
        ext in proptest::collection::vec(any::<u8>(), 0..96),
        payload in proptest::collection::vec(any::<u8>(), 0..64),
    ) {
        match EmbeddingResult::parse(&ext, &payload) {
            Err(error) => {
                prop_assert!(REFUSAL_CLASSES.contains(&error.class()), "{error}: {:?}", error.class());
            }
            Ok(result) => {
                let bytes = result.encode_to_vec(&spec_for(1, 1), capability::EMBEDDING).unwrap();
                let frame = Frame::decode(&bytes).unwrap();
                prop_assert_eq!(frame.embedding_result(), Ok(result));
            }
        }
    }

    /// A golden with arbitrary edits to its header fields, extension and payload, its CRCs then
    /// fixed so the frame stays self-consistent: decoding never panics, and a refusal of the
    /// typed frame is classified.
    #[test]
    fn resealed_embedding_mutations_decode_safely(
        which in any::<usize>(),
        edits in proptest::collection::vec((any::<usize>(), any::<u8>()), 1..6),
    ) {
        let goldens = goldens();
        let mut bytes = read_golden(goldens[which % goldens.len()].name);
        for (at, value) in edits {
            // Leave magic, the lengths and the CRCs alone.
            let at = at % bytes.len();
            if (4..48).contains(&at) || at >= 64 {
                bytes[at] = value;
            }
        }
        reseal(&mut bytes);
        if let Ok(frame) = Frame::decode(&bytes)
            && let Err(error) = frame.embedding_result()
        {
            prop_assert!(REFUSAL_CLASSES.contains(&error.class()), "{error}: {:?}", error.class());
        }
    }

    /// Any whole vector splits into parts that each fit the slot and assemble back to it.
    #[test]
    fn any_vector_splits_and_assembles(dims in 1u32..6000, slot in 132usize..40_000) {
        let whole = EmbeddingResult {
            vector: leak(spec::f32_elems(dims as usize, 7)),
            ..whole(1, dims, Pooling::LastToken, false, 3)
        };
        match whole.parts(slot) {
            Ok(parts) => {
                let parts: Vec<_> = parts.collect();
                for part in &parts {
                    prop_assert!(part.frame_len() <= slot);
                    prop_assert!(part.validate().is_ok());
                }
                prop_assert_eq!(assemble(&parts), Ok(whole.elements().collect::<Vec<_>>()));
            }
            Err(error) => prop_assert!(
                matches!(error, FrameError::TooLong { what: "part count", .. }),
                "{error}"
            ),
        }
    }
}
