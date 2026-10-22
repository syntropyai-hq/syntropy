// SPDX-License-Identifier: MIT OR Apache-2.0

//! The canonical serialisation of `core` values (RFC-0054 §4.5).
//!
//! [`to_bytes`] writes any `Serialize` value as one byte string that depends only on the value,
//! never on the host, the build or a map's iteration order. It is the contract behind
//! `InstancePolicy::canonical_bytes`; the digest is BLAKE3 over these bytes, computed by the
//! crate that links BLAKE3, because `core` links no hasher. The encoding is self-delimiting,
//! and each value starts with a tag byte:
//!
//! | Value | Tag | Then |
//! |---|---|---|
//! | unit, `None`, unit struct | `0x00` | nothing |
//! | `Some(v)` | `0x01` | `v` |
//! | `bool` | `0x02` | `0x00` or `0x01` |
//! | unsigned integer, any width | `0x03` | `u64`, little-endian |
//! | signed integer, any width | `0x04` | `i64`, little-endian |
//! | `f32` | `0x05` | its bits, little-endian; every NaN as `0x7fc0_0000`, and -0.0 as +0.0 |
//! | `f64` | `0x06` | its bits, little-endian; every NaN as `0x7ff8_0000_0000_0000`, and -0.0 as +0.0 |
//! | string, `char` | `0x07` | length as `u64` LE, then UTF-8 |
//! | bytes | `0x08` | length as `u64` LE, then the bytes |
//! | sequence, tuple, tuple struct | `0x09` | count as `u64` LE, then each element |
//! | map | `0x0a` | count as `u64` LE, then each key and value, sorted by the key's encoding |
//! | struct | `0x0b` | count as `u64` LE, then each field's name as a string and its value, sorted by name (byte order) |
//! | enum variant | `0x0c` | the variant's name as a string, then its content as above |
//!
//! Integers widen to 64 bits, so changing a field's width does not change the encoding of a
//! value that fits both. Struct fields and enum variants are encoded by name, never by
//! position: fields are sorted by name, so reordering a struct's declaration leaves its
//! encoding unchanged. Floats that compare equal encode equally, except that every NaN is
//! one value. A field that serde skips is absent, not encoded as empty. A field added later
//! with `skip_serializing_if` at its default therefore leaves every older value's encoding,
//! and so its digest, unchanged. A newtype struct encodes as its content, so `#[serde(transparent)]`
//! wrappers cost nothing. Encoding is not human-readable, so `bitflags` types encode as their
//! bits. 128-bit integers are refused.

use std::fmt;

use serde::Serialize;
use serde::ser::{self, SerializeMap, SerializeSeq, SerializeStruct};

const UNIT: u8 = 0x00;
const SOME: u8 = 0x01;
const BOOL: u8 = 0x02;
const UINT: u8 = 0x03;
const INT: u8 = 0x04;
const F32: u8 = 0x05;
const F64: u8 = 0x06;
const STR: u8 = 0x07;
const BYTES: u8 = 0x08;
const SEQ: u8 = 0x09;
const MAP: u8 = 0x0a;
const STRUCT: u8 = 0x0b;
const VARIANT: u8 = 0x0c;

/// A value the canonical encoding refuses: a 128-bit integer, or an error from a `Serialize`
/// implementation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanonError(String);

impl fmt::Display for CanonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "canonical encoding failed: {}", self.0)
    }
}

impl std::error::Error for CanonError {}

impl ser::Error for CanonError {
    fn custom<T: fmt::Display>(msg: T) -> Self {
        CanonError(msg.to_string())
    }
}

/// The canonical encoding of `value`.
pub fn to_bytes<T: Serialize + ?Sized>(value: &T) -> Result<Vec<u8>, CanonError> {
    let mut out = Vec::new();
    value.serialize(&mut Encoder { out: &mut out })?;
    Ok(out)
}

struct Encoder<'a> {
    out: &'a mut Vec<u8>,
}

impl Encoder<'_> {
    fn len(&mut self, len: usize) {
        self.out.extend_from_slice(&(len as u64).to_le_bytes());
    }

    fn str(&mut self, s: &str) {
        self.out.push(STR);
        self.len(s.len());
        self.out.extend_from_slice(s.as_bytes());
    }

    fn uint(&mut self, v: u64) {
        self.out.push(UINT);
        self.out.extend_from_slice(&v.to_le_bytes());
    }

    fn int(&mut self, v: i64) {
        self.out.push(INT);
        self.out.extend_from_slice(&v.to_le_bytes());
    }

    fn variant(&mut self, name: &str) {
        self.out.push(VARIANT);
        self.str(name);
    }
}

/// Elements of a sequence or the fields of a struct: counted as they arrive, because serde's
/// length hint may count skipped fields.
/// A struct's fields are kept apart, by name, and sorted when it ends.
struct Counted<'a> {
    out: &'a mut Vec<u8>,
    tag: u8,
    count: u64,
    body: Vec<u8>,
    fields: Vec<(&'static str, Vec<u8>)>,
}

impl<'a> Counted<'a> {
    fn new(out: &'a mut Vec<u8>, tag: u8) -> Self {
        Counted {
            out,
            tag,
            count: 0,
            body: Vec::new(),
            fields: Vec::new(),
        }
    }

    fn element<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), CanonError> {
        self.count += 1;
        value.serialize(&mut Encoder {
            out: &mut self.body,
        })
    }

    fn field<T: Serialize + ?Sized>(
        &mut self,
        name: &'static str,
        value: &T,
    ) -> Result<(), CanonError> {
        self.count += 1;
        let mut bytes = Vec::new();
        let mut enc = Encoder { out: &mut bytes };
        enc.str(name);
        value.serialize(&mut enc)?;
        self.fields.push((name, bytes));
        Ok(())
    }

    fn finish(mut self) {
        self.out.push(self.tag);
        self.out.extend_from_slice(&self.count.to_le_bytes());
        self.out.extend_from_slice(&self.body);
        // Names are unique within a struct, so the sort never compares two values.
        self.fields
            .sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
        for (_, bytes) in self.fields {
            self.out.extend_from_slice(&bytes);
        }
    }
}

/// A map's entries, sorted by their keys' encodings when it ends.
struct Entries<'a> {
    out: &'a mut Vec<u8>,
    entries: Vec<(Vec<u8>, Vec<u8>)>,
    key: Option<Vec<u8>>,
}

impl<'a> ser::Serializer for &'a mut Encoder<'_> {
    type Ok = ();
    type Error = CanonError;
    type SerializeSeq = Counted<'a>;
    type SerializeTuple = Counted<'a>;
    type SerializeTupleStruct = Counted<'a>;
    type SerializeTupleVariant = Counted<'a>;
    type SerializeMap = Entries<'a>;
    type SerializeStruct = Counted<'a>;
    type SerializeStructVariant = Counted<'a>;

    fn is_human_readable(&self) -> bool {
        false
    }

    fn serialize_bool(self, v: bool) -> Result<(), CanonError> {
        self.out.extend_from_slice(&[BOOL, u8::from(v)]);
        Ok(())
    }

    fn serialize_i8(self, v: i8) -> Result<(), CanonError> {
        self.int(v.into());
        Ok(())
    }

    fn serialize_i16(self, v: i16) -> Result<(), CanonError> {
        self.int(v.into());
        Ok(())
    }

    fn serialize_i32(self, v: i32) -> Result<(), CanonError> {
        self.int(v.into());
        Ok(())
    }

    fn serialize_i64(self, v: i64) -> Result<(), CanonError> {
        self.int(v);
        Ok(())
    }

    fn serialize_u8(self, v: u8) -> Result<(), CanonError> {
        self.uint(v.into());
        Ok(())
    }

    fn serialize_u16(self, v: u16) -> Result<(), CanonError> {
        self.uint(v.into());
        Ok(())
    }

    fn serialize_u32(self, v: u32) -> Result<(), CanonError> {
        self.uint(v.into());
        Ok(())
    }

    fn serialize_u64(self, v: u64) -> Result<(), CanonError> {
        self.uint(v);
        Ok(())
    }

    fn serialize_i128(self, _: i128) -> Result<(), CanonError> {
        Err(ser::Error::custom(
            "128-bit integers have no canonical form",
        ))
    }

    fn serialize_u128(self, _: u128) -> Result<(), CanonError> {
        Err(ser::Error::custom(
            "128-bit integers have no canonical form",
        ))
    }

    fn serialize_f32(self, v: f32) -> Result<(), CanonError> {
        // `+ 0.0` turns -0.0 into +0.0 and leaves every other value as it is.
        let bits = if v.is_nan() {
            0x7fc0_0000
        } else {
            (v + 0.0).to_bits()
        };
        self.out.push(F32);
        self.out.extend_from_slice(&bits.to_le_bytes());
        Ok(())
    }

    fn serialize_f64(self, v: f64) -> Result<(), CanonError> {
        let bits = if v.is_nan() {
            0x7ff8_0000_0000_0000
        } else {
            (v + 0.0).to_bits()
        };
        self.out.push(F64);
        self.out.extend_from_slice(&bits.to_le_bytes());
        Ok(())
    }

    fn serialize_char(self, v: char) -> Result<(), CanonError> {
        self.str(v.encode_utf8(&mut [0; 4]));
        Ok(())
    }

    fn serialize_str(self, v: &str) -> Result<(), CanonError> {
        self.str(v);
        Ok(())
    }

    fn serialize_bytes(self, v: &[u8]) -> Result<(), CanonError> {
        self.out.push(BYTES);
        self.len(v.len());
        self.out.extend_from_slice(v);
        Ok(())
    }

    fn serialize_none(self) -> Result<(), CanonError> {
        self.out.push(UNIT);
        Ok(())
    }

    fn serialize_some<T: Serialize + ?Sized>(self, value: &T) -> Result<(), CanonError> {
        self.out.push(SOME);
        value.serialize(self)
    }

    fn serialize_unit(self) -> Result<(), CanonError> {
        self.out.push(UNIT);
        Ok(())
    }

    fn serialize_unit_struct(self, _: &'static str) -> Result<(), CanonError> {
        self.out.push(UNIT);
        Ok(())
    }

    fn serialize_unit_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
    ) -> Result<(), CanonError> {
        self.variant(variant);
        self.out.push(UNIT);
        Ok(())
    }

    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        value: &T,
    ) -> Result<(), CanonError> {
        value.serialize(self)
    }

    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<(), CanonError> {
        self.variant(variant);
        value.serialize(self)
    }

    fn serialize_seq(self, _: Option<usize>) -> Result<Counted<'a>, CanonError> {
        Ok(Counted::new(self.out, SEQ))
    }

    fn serialize_tuple(self, _: usize) -> Result<Counted<'a>, CanonError> {
        Ok(Counted::new(self.out, SEQ))
    }

    fn serialize_tuple_struct(self, _: &'static str, _: usize) -> Result<Counted<'a>, CanonError> {
        Ok(Counted::new(self.out, SEQ))
    }

    fn serialize_tuple_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        _: usize,
    ) -> Result<Counted<'a>, CanonError> {
        self.variant(variant);
        Ok(Counted::new(self.out, SEQ))
    }

    fn serialize_map(self, _: Option<usize>) -> Result<Entries<'a>, CanonError> {
        Ok(Entries {
            out: self.out,
            entries: Vec::new(),
            key: None,
        })
    }

    fn serialize_struct(self, _: &'static str, _: usize) -> Result<Counted<'a>, CanonError> {
        Ok(Counted::new(self.out, STRUCT))
    }

    fn serialize_struct_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        _: usize,
    ) -> Result<Counted<'a>, CanonError> {
        self.variant(variant);
        Ok(Counted::new(self.out, STRUCT))
    }
}

impl SerializeSeq for Counted<'_> {
    type Ok = ();
    type Error = CanonError;

    fn serialize_element<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), CanonError> {
        self.element(value)
    }

    fn end(self) -> Result<(), CanonError> {
        self.finish();
        Ok(())
    }
}

impl ser::SerializeTuple for Counted<'_> {
    type Ok = ();
    type Error = CanonError;

    fn serialize_element<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), CanonError> {
        self.element(value)
    }

    fn end(self) -> Result<(), CanonError> {
        self.finish();
        Ok(())
    }
}

impl ser::SerializeTupleStruct for Counted<'_> {
    type Ok = ();
    type Error = CanonError;

    fn serialize_field<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), CanonError> {
        self.element(value)
    }

    fn end(self) -> Result<(), CanonError> {
        self.finish();
        Ok(())
    }
}

impl ser::SerializeTupleVariant for Counted<'_> {
    type Ok = ();
    type Error = CanonError;

    fn serialize_field<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), CanonError> {
        self.element(value)
    }

    fn end(self) -> Result<(), CanonError> {
        self.finish();
        Ok(())
    }
}

impl SerializeStruct for Counted<'_> {
    type Ok = ();
    type Error = CanonError;

    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        name: &'static str,
        value: &T,
    ) -> Result<(), CanonError> {
        self.field(name, value)
    }

    fn skip_field(&mut self, _: &'static str) -> Result<(), CanonError> {
        Ok(())
    }

    fn end(self) -> Result<(), CanonError> {
        self.finish();
        Ok(())
    }
}

impl ser::SerializeStructVariant for Counted<'_> {
    type Ok = ();
    type Error = CanonError;

    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        name: &'static str,
        value: &T,
    ) -> Result<(), CanonError> {
        self.field(name, value)
    }

    fn skip_field(&mut self, _: &'static str) -> Result<(), CanonError> {
        Ok(())
    }

    fn end(self) -> Result<(), CanonError> {
        self.finish();
        Ok(())
    }
}

impl SerializeMap for Entries<'_> {
    type Ok = ();
    type Error = CanonError;

    fn serialize_key<T: Serialize + ?Sized>(&mut self, key: &T) -> Result<(), CanonError> {
        self.key = Some(to_bytes(key)?);
        Ok(())
    }

    fn serialize_value<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), CanonError> {
        let key = self
            .key
            .take()
            .ok_or_else(|| ser::Error::custom("a map value came before its key"))?;
        self.entries.push((key, to_bytes(value)?));
        Ok(())
    }

    fn end(mut self) -> Result<(), CanonError> {
        self.entries.sort();
        self.out.push(MAP);
        self.out
            .extend_from_slice(&(self.entries.len() as u64).to_le_bytes());
        for (key, value) in self.entries {
            self.out.extend_from_slice(&key);
            self.out.extend_from_slice(&value);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    // A `HashMap` with `RandomState`, on purpose: `maps_are_sorted_whatever_their_order` shows
    // that a map's order, random from run to run, does not reach the encoding. A listed exception
    // to the determinism lints (`xtask/src/determinism.rs`).
    #[allow(
        clippy::disallowed_types,
        reason = "the test feeds the encoder a randomly ordered map"
    )]
    use std::collections::HashMap;

    use super::*;

    #[derive(Serialize)]
    struct Wide {
        a: u64,
    }

    #[derive(Serialize)]
    struct Narrow {
        a: u8,
    }

    #[derive(Serialize)]
    struct Skips {
        a: u8,
        #[serde(skip_serializing_if = "Option::is_none")]
        b: Option<u8>,
    }

    #[derive(Serialize)]
    enum E {
        A,
        B(u8),
        C { x: u8 },
    }

    #[test]
    fn pinned_bytes() {
        assert_eq!(to_bytes(&()).unwrap(), [UNIT]);
        assert_eq!(to_bytes(&true).unwrap(), [BOOL, 1]);
        assert_eq!(
            to_bytes(&-2i8).unwrap(),
            [INT, 0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff]
        );
        assert_eq!(
            to_bytes(&"ab").unwrap(),
            [STR, 2, 0, 0, 0, 0, 0, 0, 0, b'a', b'b']
        );
        assert_eq!(
            to_bytes(&Some(1u8)).unwrap(),
            [SOME, UINT, 1, 0, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(
            to_bytes(&E::A).unwrap(),
            [VARIANT, STR, 1, 0, 0, 0, 0, 0, 0, 0, b'A', UNIT]
        );
        assert_eq!(
            to_bytes(&E::B(1)).unwrap(),
            [
                VARIANT, STR, 1, 0, 0, 0, 0, 0, 0, 0, b'B', UINT, 1, 0, 0, 0, 0, 0, 0, 0
            ]
        );
        let c = to_bytes(&E::C { x: 1 }).unwrap();
        assert_eq!(
            &c[..12],
            [VARIANT, STR, 1, 0, 0, 0, 0, 0, 0, 0, b'C', STRUCT]
        );
    }

    #[test]
    fn integer_width_does_not_matter() {
        assert_eq!(
            to_bytes(&Wide { a: 7 }).unwrap(),
            to_bytes(&Narrow { a: 7 }).unwrap()
        );
    }

    #[test]
    fn skipped_fields_are_absent() {
        assert_eq!(
            to_bytes(&Skips { a: 7, b: None }).unwrap(),
            to_bytes(&Narrow { a: 7 }).unwrap()
        );
        assert_ne!(
            to_bytes(&Skips { a: 7, b: Some(0) }).unwrap(),
            to_bytes(&Narrow { a: 7 }).unwrap()
        );
    }

    #[test]
    #[allow(
        clippy::disallowed_types,
        reason = "the test feeds the encoder a randomly ordered map"
    )]
    fn maps_are_sorted_whatever_their_order() {
        let mut hash = HashMap::new();
        let mut tree = BTreeMap::new();
        for k in ["b", "a", "c", "aa"] {
            hash.insert(k, k.len());
            tree.insert(k, k.len());
        }
        assert_eq!(to_bytes(&hash).unwrap(), to_bytes(&tree).unwrap());
    }

    #[test]
    fn nan_is_canonical() {
        let quiet = f32::NAN;
        let other = f32::from_bits(0xffc0_0001);
        assert!(other.is_nan());
        assert_eq!(to_bytes(&quiet).unwrap(), to_bytes(&other).unwrap());
        assert_eq!(to_bytes(&f64::NAN).unwrap(), to_bytes(&-f64::NAN).unwrap());
    }

    #[test]
    fn negative_zero_is_zero() {
        assert_eq!(to_bytes(&-0.0f32).unwrap(), to_bytes(&0.0f32).unwrap());
        assert_eq!(to_bytes(&-0.0f64).unwrap(), to_bytes(&0.0f64).unwrap());
        assert_ne!(to_bytes(&-1.0f32).unwrap(), to_bytes(&1.0f32).unwrap());
    }

    #[derive(Serialize)]
    struct Ab {
        a: u8,
        b: u8,
    }

    #[derive(Serialize)]
    struct Ba {
        b: u8,
        a: u8,
    }

    #[test]
    fn fields_are_sorted_by_name() {
        assert_eq!(
            to_bytes(&Ab { a: 1, b: 2 }).unwrap(),
            to_bytes(&Ba { b: 2, a: 1 }).unwrap()
        );
        assert_ne!(
            to_bytes(&Ab { a: 1, b: 2 }).unwrap(),
            to_bytes(&Ab { a: 2, b: 1 }).unwrap()
        );
    }

    #[test]
    fn refuses_128_bit_integers() {
        assert!(to_bytes(&1u128).is_err());
        assert!(to_bytes(&1i128).is_err());
    }
}
