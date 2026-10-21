// SPDX-License-Identifier: MIT OR Apache-2.0

//! The control and join envelope (RFC-0031 §9.1, ADR-0305 items 1–3). Owner: RFC-0031.
//!
//! On `syntropy/ctl/1` and `syntropy/join/1` streams, each message is
//!
//! ```text
//! | len: u32 (bytes after this field) | enc: u8 (0 = CBOR, 1 = zstd(CBOR)) | body |
//! ```
//!
//! and a datagram is `| tag: u8 | CBOR body |`, with no length field. Integers are little-endian,
//! as everywhere in RFC-0031.
//!
//! **The body is deterministic CBOR (RFC 8949) from serde**, so identical state gives identical
//! bytes, which the event log's hash chain relies on (RFC-0032 §6.1):
//! - integers and lengths take their shortest form;
//! - every array, map, byte string and text string has a definite length;
//! - struct fields are encoded in declaration order;
//! - map keys come in the order the map iterates them, which for a `BTreeMap` is sorted by the
//!   key's `Ord`. That is the order RFC-0031 §9.1 specifies. It is not RFC 8949 §4.2.1's
//!   bytewise order of the encoded keys: the two differ for negative integers and for strings of
//!   different lengths. A map whose iteration order is not fixed, such as a `HashMap`, must not
//!   appear in a control message.
//!
//! [`to_cbor`] checks its own output and refuses anything that breaks the first two rules. serde
//! asks for an indefinite length when a sequence's size is unknown in advance, as
//! `collect_seq` over a filtered iterator does, and the encoder then returns
//! [`EncodeError::NotDeterministic`] rather than send it.
//!
//! **Evolution within a major** (ADR-0305 item 3): decoders ignore map keys they don't know, so a
//! field a newer build adds, with `#[serde(default)]`, does not stop an older build reading the
//! message. Types never use `deny_unknown_fields`.
//!
//! **Size.** A CBOR body over 64 KiB is zstd-compressed at level 3. No body, before or after
//! compression, may exceed 64 MiB; larger state travels on the blob plane by digest. The decoder
//! also refuses a zstd window over 64 MiB, so decompressing one message takes at most the limit
//! plus one window.
//!
//! **Decoding is total.** Every function that reads bytes returns a [`DecodeError`] for any input
//! it cannot accept, never panics, and checks every declared length against the limit and the
//! bytes present before it allocates. [`DecodeError::class`] is `Integrity`, `InvalidRequest` or
//! `Unsupported` (RFC-0060 §10.2), never `Internal`. The `control_msgs` fuzz target
//! (QUAL-E10-T04) wraps [`decode`].

use std::fmt;
use std::io::Read;

use serde::Serialize;
use serde::de::DeserializeOwned;
use syntropy_core::ErrorClass;

/// The largest body a message may carry, before or after compression: 64 MiB.
pub const MAX_BODY: usize = 64 << 20;

/// A CBOR body longer than this is compressed: 64 KiB.
pub const COMPRESS_ABOVE: usize = 64 << 10;

/// The zstd level for compressed bodies.
pub const ZSTD_LEVEL: i32 = 3;

/// The largest zstd window a compressed body may ask the decoder for, as a power of two: 2^26
/// bytes, [`MAX_BODY`] itself. A frame never needs a window larger than its content, and content
/// over the limit is refused anyway, so a larger window would only cost the reader memory.
/// zstd's own default allows 2^27.
pub const ZSTD_WINDOW_LOG_MAX: u32 = 26;
const _: () = assert!(1 << ZSTD_WINDOW_LOG_MAX == MAX_BODY);

/// The bytes of the `len` field.
pub const LEN_BYTES: usize = 4;

/// The `enc` byte: how the body is encoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Encoding {
    /// Plain CBOR.
    Cbor = 0,
    /// zstd-compressed CBOR.
    ZstdCbor = 1,
}

impl Encoding {
    /// The encoding the `enc` byte names, if it names one.
    pub fn from_byte(byte: u8) -> Option<Encoding> {
        match byte {
            0 => Some(Encoding::Cbor),
            1 => Some(Encoding::ZstdCbor),
            _ => None,
        }
    }
}

/// Why a value could not be encoded.
#[derive(Debug, PartialEq, Eq)]
pub enum EncodeError {
    /// The CBOR body, or the compressed body, is over [`MAX_BODY`].
    TooLarge { len: usize },
    /// A datagram would be longer than the path allows, so it is not sent.
    DatagramTooLarge { len: usize, limit: usize },
    /// The value's `Serialize` produced an indefinite length or a non-shortest integer, at this
    /// byte offset of the body. It is a bug in that type.
    NotDeterministic { at: usize },
    /// serde or the CBOR writer refused the value.
    Cbor(String),
    /// zstd failed.
    Compress(String),
}

impl EncodeError {
    /// The error class (RFC-0001 §5.12).
    pub fn class(&self) -> ErrorClass {
        match self {
            EncodeError::TooLarge { .. } | EncodeError::DatagramTooLarge { .. } => {
                ErrorClass::Capacity
            }
            EncodeError::NotDeterministic { .. }
            | EncodeError::Cbor(_)
            | EncodeError::Compress(_) => ErrorClass::Internal,
        }
    }
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EncodeError::TooLarge { len } => {
                write!(f, "a {len}-byte body is over the {MAX_BODY}-byte limit")
            }
            EncodeError::DatagramTooLarge { len, limit } => {
                write!(
                    f,
                    "a {len}-byte datagram is over the path's {limit}-byte limit"
                )
            }
            EncodeError::NotDeterministic { at } => write!(
                f,
                "the body is not deterministic CBOR at byte {at}: an indefinite length or a \
                 non-shortest integer"
            ),
            EncodeError::Cbor(message) => write!(f, "CBOR encoding failed: {message}"),
            EncodeError::Compress(message) => write!(f, "zstd compression failed: {message}"),
        }
    }
}

impl std::error::Error for EncodeError {}

/// Why bytes could not be decoded. Every variant is a classified rejection (RFC-0060 §10.2).
#[derive(Debug, PartialEq, Eq)]
pub enum DecodeError {
    /// Fewer bytes than the format or a length field needs.
    Truncated { needed: usize, got: usize },
    /// Bytes left over after one complete message.
    TrailingBytes { extra: usize },
    /// A declared or decompressed body over [`MAX_BODY`].
    TooLarge { len: usize },
    /// An `enc` byte this build does not know.
    UnknownEncoding(u8),
    /// The compressed body is not valid zstd.
    Decompress(String),
    /// The body is not well-formed CBOR.
    Malformed(String),
    /// Well-formed CBOR that does not match the expected type, or nested too deeply.
    Mismatch(String),
}

impl DecodeError {
    /// The error class (RFC-0001 §5.12): `Integrity` for damaged bytes, `InvalidRequest` for a
    /// message that breaks the protocol, `Unsupported` for an encoding this build lacks.
    pub fn class(&self) -> ErrorClass {
        match self {
            DecodeError::Truncated { .. }
            | DecodeError::TrailingBytes { .. }
            | DecodeError::Decompress(_)
            | DecodeError::Malformed(_) => ErrorClass::Integrity,
            DecodeError::TooLarge { .. } | DecodeError::Mismatch(_) => ErrorClass::InvalidRequest,
            DecodeError::UnknownEncoding(_) => ErrorClass::Unsupported,
        }
    }
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeError::Truncated { needed, got } => {
                write!(f, "truncated: {needed} bytes needed, {got} present")
            }
            DecodeError::TrailingBytes { extra } => {
                write!(f, "{extra} bytes after the end of the message")
            }
            DecodeError::TooLarge { len } => {
                write!(f, "a {len}-byte body is over the {MAX_BODY}-byte limit")
            }
            DecodeError::UnknownEncoding(byte) => write!(f, "unknown body encoding {byte}"),
            DecodeError::Decompress(message) => write!(f, "zstd decompression failed: {message}"),
            DecodeError::Malformed(message) => write!(f, "malformed CBOR: {message}"),
            DecodeError::Mismatch(message) => {
                write!(f, "CBOR does not match the expected type: {message}")
            }
        }
    }
}

impl std::error::Error for DecodeError {}

/// `value` as deterministic CBOR: the body of a message or a datagram.
pub fn to_cbor<T: Serialize + ?Sized>(value: &T) -> Result<Vec<u8>, EncodeError> {
    let mut body = Vec::new();
    ciborium::into_writer(value, &mut body)
        .map_err(|error| EncodeError::Cbor(error.to_string()))?;
    if let Err(at) = check_deterministic(&body) {
        return Err(EncodeError::NotDeterministic { at });
    }
    Ok(body)
}

/// Reads one CBOR value of type `T` that fills `body` exactly.
pub fn from_cbor<T: DeserializeOwned>(body: &[u8]) -> Result<T, DecodeError> {
    if body.len() > MAX_BODY {
        return Err(DecodeError::TooLarge { len: body.len() });
    }
    // Every length the CBOR declares must fit in the bytes present, so the reader never
    // allocates for more than the input holds.
    check_lengths(body)?;
    let mut rest = body;
    let value = ciborium::from_reader(&mut rest).map_err(|error| match error {
        ciborium::de::Error::Io(error) => DecodeError::Malformed(error.to_string()),
        ciborium::de::Error::Syntax(at) => DecodeError::Malformed(format!("at byte {at}")),
        ciborium::de::Error::Semantic(_, message) => DecodeError::Mismatch(message),
        ciborium::de::Error::RecursionLimitExceeded => {
            DecodeError::Mismatch("nested too deeply".to_owned())
        }
    })?;
    if !rest.is_empty() {
        return Err(DecodeError::TrailingBytes { extra: rest.len() });
    }
    Ok(value)
}

/// `value` as one stream message: `len | enc | body`, compressed when the CBOR body is over
/// [`COMPRESS_ABOVE`].
pub fn encode<T: Serialize + ?Sized>(value: &T) -> Result<Vec<u8>, EncodeError> {
    let cbor = to_cbor(value)?;
    if cbor.len() > MAX_BODY {
        return Err(EncodeError::TooLarge { len: cbor.len() });
    }
    let (encoding, body) = if cbor.len() > COMPRESS_ABOVE {
        let compressed = zstd::bulk::compress(&cbor, ZSTD_LEVEL)
            .map_err(|error| EncodeError::Compress(error.to_string()))?;
        if compressed.len() > MAX_BODY {
            return Err(EncodeError::TooLarge {
                len: compressed.len(),
            });
        }
        (Encoding::ZstdCbor, compressed)
    } else {
        (Encoding::Cbor, cbor)
    };
    let len =
        u32::try_from(1 + body.len()).map_err(|_| EncodeError::TooLarge { len: body.len() })?;
    let mut frame = Vec::with_capacity(LEN_BYTES + 1 + body.len());
    frame.extend_from_slice(&len.to_le_bytes());
    frame.push(encoding as u8);
    frame.extend_from_slice(&body);
    Ok(frame)
}

/// The number of bytes that follow a message's `len` field, from those four bytes. A stream
/// reader calls this before it reads or allocates the rest.
pub fn message_len(prefix: [u8; LEN_BYTES]) -> Result<usize, DecodeError> {
    // `as usize` widens: a u32 always fits in usize on every target RFC-0001 supports.
    let len = u32::from_le_bytes(prefix) as usize;
    if len == 0 {
        return Err(DecodeError::Truncated { needed: 1, got: 0 });
    }
    if len - 1 > MAX_BODY {
        return Err(DecodeError::TooLarge { len: len - 1 });
    }
    Ok(len)
}

/// Reads one stream message of type `T` that fills `frame` exactly: `len | enc | body`.
pub fn decode<T: DeserializeOwned>(frame: &[u8]) -> Result<T, DecodeError> {
    let Some((prefix, rest)) = frame.split_first_chunk::<LEN_BYTES>() else {
        return Err(DecodeError::Truncated {
            needed: LEN_BYTES,
            got: frame.len(),
        });
    };
    let len = message_len(*prefix)?;
    match rest.len().cmp(&len) {
        std::cmp::Ordering::Less => Err(DecodeError::Truncated {
            needed: len,
            got: rest.len(),
        }),
        std::cmp::Ordering::Greater => Err(DecodeError::TrailingBytes {
            extra: rest.len() - len,
        }),
        std::cmp::Ordering::Equal => decode_body(rest),
    }
}

/// Reads one message from the bytes after its `len` field: `enc | body`.
pub fn decode_body<T: DeserializeOwned>(enc_and_body: &[u8]) -> Result<T, DecodeError> {
    let Some((&enc, body)) = enc_and_body.split_first() else {
        return Err(DecodeError::Truncated { needed: 1, got: 0 });
    };
    if body.len() > MAX_BODY {
        return Err(DecodeError::TooLarge { len: body.len() });
    }
    match Encoding::from_byte(enc) {
        Some(Encoding::Cbor) => from_cbor(body),
        Some(Encoding::ZstdCbor) => from_cbor(&decompress(body)?),
        None => Err(DecodeError::UnknownEncoding(enc)),
    }
}

/// `value` as a datagram, `tag | body`, or an error if it would be longer than `limit`, the
/// path's datagram size: such a datagram is not sent (RFC-0031 §9.1).
pub fn encode_datagram<T: Serialize + ?Sized>(
    tag: u8,
    value: &T,
    limit: usize,
) -> Result<Vec<u8>, EncodeError> {
    let body = to_cbor(value)?;
    let len = 1 + body.len();
    if len > limit {
        return Err(EncodeError::DatagramTooLarge { len, limit });
    }
    let mut datagram = Vec::with_capacity(len);
    datagram.push(tag);
    datagram.extend_from_slice(&body);
    Ok(datagram)
}

/// A datagram's tag and its CBOR body, for a caller that picks the body's type by tag.
pub fn split_datagram(datagram: &[u8]) -> Result<(u8, &[u8]), DecodeError> {
    datagram
        .split_first()
        .map(|(&tag, body)| (tag, body))
        .ok_or(DecodeError::Truncated { needed: 1, got: 0 })
}

/// A datagram's tag, and its body read as `T`.
pub fn decode_datagram<T: DeserializeOwned>(datagram: &[u8]) -> Result<(u8, T), DecodeError> {
    let (tag, body) = split_datagram(datagram)?;
    Ok((tag, from_cbor(body)?))
}

/// The decompressed body, refused once it passes [`MAX_BODY`]. A small compressed body cannot
/// make the reader hold more than the limit and one window:
/// - a frame header that declares content over the limit is refused before decoding starts;
/// - a frame that asks for a window over 2^[`ZSTD_WINDOW_LOG_MAX`] bytes fails to decode;
/// - the output grows by doubling from 64 KiB, its capacity never past `MAX_BODY + 1` bytes, and
///   reading stops at that byte.
fn decompress(body: &[u8]) -> Result<Vec<u8>, DecodeError> {
    let failed = |error: std::io::Error| DecodeError::Decompress(error.to_string());
    if let Ok(Some(declared)) = zstd::zstd_safe::get_frame_content_size(body)
        && declared > MAX_BODY as u64
    {
        return Err(DecodeError::TooLarge {
            len: usize::try_from(declared).unwrap_or(usize::MAX),
        });
    }
    let mut decoder = zstd::stream::read::Decoder::with_buffer(body).map_err(failed)?;
    decoder
        .window_log_max(ZSTD_WINDOW_LOG_MAX)
        .map_err(failed)?;
    let mut out = Vec::new();
    let mut filled = 0;
    loop {
        if filled == out.len() {
            if filled > MAX_BODY {
                return Err(DecodeError::TooLarge { len: filled });
            }
            let grown = (out.len() * 2).clamp(COMPRESS_ABOVE, MAX_BODY + 1);
            out.reserve_exact(grown - out.len());
            out.resize(grown, 0);
        }
        match decoder.read(&mut out[filled..]) {
            Ok(0) => break,
            Ok(read) => filled += read,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => return Err(failed(error)),
        }
    }
    out.truncate(filled);
    Ok(out)
}

/// One CBOR item head (RFC 8949 §3): the major type, the additional information and the
/// argument, and the offset after the head.
struct Head {
    major: u8,
    info: u8,
    argument: u64,
    end: usize,
}

/// Reads the head at `at`. `None` if the bytes end inside it or the additional information is
/// reserved (28 to 30).
fn head(bytes: &[u8], at: usize) -> Option<Head> {
    let first = *bytes.get(at)?;
    let (major, info) = (first >> 5, first & 0x1f);
    let width = match info {
        0..=23 | 31 => 0,
        24 => 1,
        25 => 2,
        26 => 4,
        27 => 8,
        _ => return None,
    };
    let start = at + 1;
    let field = bytes.get(start..start.checked_add(width)?)?;
    let argument = if width == 0 {
        u64::from(info)
    } else {
        field
            .iter()
            .fold(0u64, |acc, byte| (acc << 8) | u64::from(*byte))
    };
    Some(Head {
        major,
        info,
        argument,
        end: start + width,
    })
}

/// Walks every item in `bytes` and calls `check` on each head; `items` counts the items still to
/// read. Fails with the offset of the first head `check` refuses, or of the first head that is
/// cut short or reserved. Iterative, so nesting depth costs no stack.
fn walk(bytes: &[u8], mut check: impl FnMut(&Head, usize) -> bool) -> Result<usize, usize> {
    let mut at = 0;
    let mut items: u64 = 1;
    while items > 0 {
        items -= 1;
        let Some(head) = head(bytes, at) else {
            return Err(at);
        };
        if !check(&head, at) {
            return Err(at);
        }
        at = head.end;
        let indefinite = head.info == 31;
        match head.major {
            // Byte and text strings: skip the content.
            2 | 3 if !indefinite => {
                let len = usize::try_from(head.argument).map_err(|_| at)?;
                at = at
                    .checked_add(len)
                    .filter(|end| *end <= bytes.len())
                    .ok_or(at)?;
            }
            // Arrays and maps: their items follow.
            4 if !indefinite => items = items.checked_add(head.argument).ok_or(at)?,
            5 if !indefinite => {
                let pairs = head.argument.checked_mul(2).ok_or(at)?;
                items = items.checked_add(pairs).ok_or(at)?;
            }
            // A tag is followed by its one item.
            6 => items += 1,
            _ => {}
        }
        // Each item takes at least one byte, so more items than bytes left cannot be valid.
        if items > (bytes.len() - at.min(bytes.len())) as u64 {
            return Err(at);
        }
    }
    Ok(at)
}

/// Checks the deterministic rules [`to_cbor`] promises: definite lengths, and integer and length
/// arguments in their shortest form. Fails with the offset of the first item that breaks one.
fn check_deterministic(body: &[u8]) -> Result<(), usize> {
    let end = walk(body, |head, _| {
        let shortest = match head.info {
            24 => head.argument >= 24,
            25 => head.argument > 0xff,
            26 => head.argument > 0xffff,
            27 => head.argument > 0xffff_ffff,
            _ => true,
        };
        // Floats (major 7, 25 to 27) carry bits, not an integer, so the rule does not apply.
        let float = head.major == 7 && (25..=27).contains(&head.info);
        head.info != 31 && (shortest || float)
    })?;
    if end == body.len() { Ok(()) } else { Err(end) }
}

/// Checks that every length `body` declares fits in the bytes present, before the reader trusts
/// any of them: each string's length against the bytes left, and each array's or map's item
/// count against the bytes left, since every item takes at least one byte. The scan reads every
/// head in order, so it also covers the items inside indefinite-length containers, which peers
/// may send. RFC-0060 §10.2 asks every parser to check declared sizes against the input before
/// allocating. ciborium 0.2 reads strings incrementally and so does not over-allocate on its
/// own, but this check keeps the rule from depending on that.
fn check_lengths(body: &[u8]) -> Result<(), DecodeError> {
    let malformed =
        |at: usize| DecodeError::Malformed(format!("a length at byte {at} runs past the end"));
    let mut at = 0;
    while at < body.len() {
        let Some(head) = head(body, at) else {
            return Err(DecodeError::Malformed(format!(
                "a bad item head at byte {at}"
            )));
        };
        let left = (body.len() - head.end) as u64;
        let fits = match head.major {
            2 | 3 if head.info != 31 => head.argument <= left,
            4 if head.info != 31 => head.argument <= left,
            5 if head.info != 31 => head.argument.checked_mul(2).is_some_and(|n| n <= left),
            _ => true,
        };
        if !fits {
            return Err(malformed(at));
        }
        at = head.end;
        if matches!(head.major, 2 | 3) && head.info != 31 {
            // `fits` holds, so the content is inside `body`.
            at += head.argument as usize;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heads_read_their_argument_big_endian_as_rfc_8949_says() {
        let h = head(&[0x19, 0x01, 0x02], 0).unwrap();
        assert_eq!((h.major, h.info, h.argument, h.end), (0, 25, 0x0102, 3));
        assert!(head(&[0x19, 0x01], 0).is_none(), "cut short");
        assert!(head(&[0x1c], 0).is_none(), "reserved");
    }

    #[test]
    fn non_shortest_integers_and_indefinite_lengths_are_not_deterministic() {
        assert_eq!(check_deterministic(&[0x17]), Ok(()), "23 in the head");
        assert_eq!(check_deterministic(&[0x18, 0x18]), Ok(()), "24 in one byte");
        assert_eq!(check_deterministic(&[0x18, 0x17]), Err(0), "23 in a byte");
        assert_eq!(
            check_deterministic(&[0x19, 0x00, 0xff]),
            Err(0),
            "255 in two bytes"
        );
        assert_eq!(
            check_deterministic(&[0x9f, 0x01, 0xff]),
            Err(0),
            "indefinite array"
        );
        assert_eq!(
            check_deterministic(&[0x82, 0x01, 0x18, 0x02]),
            Err(2),
            "the second element"
        );
        // A half float carries bits, so its width is not an integer argument.
        assert_eq!(check_deterministic(&[0xf9, 0x3c, 0x00]), Ok(()));
        assert_eq!(check_deterministic(&[0x01, 0x01]), Err(1), "two items");
    }

    #[test]
    fn declared_lengths_must_fit_the_bytes_present() {
        // A text string claiming 2^62 bytes, then three bytes.
        let mut huge = vec![0x7b];
        huge.extend_from_slice(&(1u64 << 62).to_be_bytes());
        huge.extend_from_slice(b"abc");
        assert!(matches!(
            check_lengths(&huge),
            Err(DecodeError::Malformed(_))
        ));
        // An array claiming more items than bytes remain.
        assert!(check_lengths(&[0x9a, 0x00, 0x10, 0x00, 0x00]).is_err());
        assert!(check_lengths(&[0x83, 0x01, 0x02, 0x03]).is_ok());
        // The same claim inside an indefinite-length array is caught too.
        let mut nested = vec![0x9f];
        nested.extend_from_slice(&huge);
        nested.push(0xff);
        assert!(check_lengths(&nested).is_err());
        assert!(check_lengths(&[0x9f, 0x63, b'a', b'b', b'c', 0xff]).is_ok());
    }

    #[test]
    fn every_decode_error_is_a_classified_rejection() {
        let allowed = [
            ErrorClass::Integrity,
            ErrorClass::InvalidRequest,
            ErrorClass::Unsupported,
        ];
        for error in [
            DecodeError::Truncated { needed: 1, got: 0 },
            DecodeError::TrailingBytes { extra: 1 },
            DecodeError::TooLarge { len: 1 },
            DecodeError::UnknownEncoding(9),
            DecodeError::Decompress(String::new()),
            DecodeError::Malformed(String::new()),
            DecodeError::Mismatch(String::new()),
        ] {
            assert!(allowed.contains(&error.class()), "{error}");
        }
    }
}
