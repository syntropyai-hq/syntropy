// SPDX-License-Identifier: MIT OR Apache-2.0

//! The parity tap types through a binary serde format (RFC-0031 §6.5): every float keeps its IEEE
//! bit pattern, NaN payloads, signed zeros, subnormals and infinities included.
//!
//! The workspace has no binary serde format, so this file carries a minimal one. It is not
//! self-describing (`deserialize_any` fails, as in postcard), it reports
//! `is_human_readable() == false`, and it writes fixed-width little-endian values, length-prefixed
//! sequences, structs as their fields in order and enums as a `u32` variant index. It supports the
//! shapes the tap types use and fails on anything else.

use std::fmt;

use proptest::prelude::*;
use serde::de::{self, DeserializeOwned, IntoDeserializer, Visitor};
use serde::ser::{self, Impossible};
use serde::{Deserializer, Serialize, Serializer};
use syntropy_core::{
    ForceMode, LogitTap, ParityTapSpec, TapMode, TapRow, TapValues, TeacherStreamId,
};

#[derive(Debug)]
struct Error(String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}
impl ser::Error for Error {
    fn custom<T: fmt::Display>(msg: T) -> Self {
        Error(msg.to_string())
    }
}
impl de::Error for Error {
    fn custom<T: fmt::Display>(msg: T) -> Self {
        Error(msg.to_string())
    }
}

fn unsupported<T>(what: &str) -> Result<T, Error> {
    Err(Error(format!("the test format does not support {what}")))
}

fn encode<T: Serialize>(value: &T) -> Vec<u8> {
    let mut out = Vec::new();
    value.serialize(Ser(&mut out)).expect("encode");
    out
}

fn decode<T: DeserializeOwned>(bytes: &[u8]) -> T {
    let mut de = De { input: bytes };
    let value = T::deserialize(&mut de).expect("decode");
    assert!(de.input.is_empty(), "{} trailing bytes", de.input.len());
    value
}

struct Ser<'a>(&'a mut Vec<u8>);

impl Ser<'_> {
    fn put(&mut self, bytes: &[u8]) {
        self.0.extend_from_slice(bytes);
    }
}

impl<'a> Serializer for Ser<'a> {
    type Ok = ();
    type Error = Error;
    type SerializeSeq = Ser<'a>;
    type SerializeTuple = Ser<'a>;
    type SerializeTupleStruct = Ser<'a>;
    type SerializeTupleVariant = Ser<'a>;
    type SerializeMap = Impossible<(), Error>;
    type SerializeStruct = Ser<'a>;
    type SerializeStructVariant = Ser<'a>;

    fn is_human_readable(&self) -> bool {
        false
    }
    fn serialize_bool(mut self, v: bool) -> Result<(), Error> {
        self.put(&[u8::from(v)]);
        Ok(())
    }
    fn serialize_i8(mut self, v: i8) -> Result<(), Error> {
        self.put(&v.to_le_bytes());
        Ok(())
    }
    fn serialize_i16(mut self, v: i16) -> Result<(), Error> {
        self.put(&v.to_le_bytes());
        Ok(())
    }
    fn serialize_i32(mut self, v: i32) -> Result<(), Error> {
        self.put(&v.to_le_bytes());
        Ok(())
    }
    fn serialize_i64(mut self, v: i64) -> Result<(), Error> {
        self.put(&v.to_le_bytes());
        Ok(())
    }
    fn serialize_u8(mut self, v: u8) -> Result<(), Error> {
        self.put(&[v]);
        Ok(())
    }
    fn serialize_u16(mut self, v: u16) -> Result<(), Error> {
        self.put(&v.to_le_bytes());
        Ok(())
    }
    fn serialize_u32(mut self, v: u32) -> Result<(), Error> {
        self.put(&v.to_le_bytes());
        Ok(())
    }
    fn serialize_u64(mut self, v: u64) -> Result<(), Error> {
        self.put(&v.to_le_bytes());
        Ok(())
    }
    fn serialize_f32(mut self, v: f32) -> Result<(), Error> {
        self.put(&v.to_bits().to_le_bytes());
        Ok(())
    }
    fn serialize_f64(mut self, v: f64) -> Result<(), Error> {
        self.put(&v.to_bits().to_le_bytes());
        Ok(())
    }
    fn serialize_char(self, _: char) -> Result<(), Error> {
        unsupported("char")
    }
    fn serialize_str(mut self, v: &str) -> Result<(), Error> {
        self.put(&(v.len() as u64).to_le_bytes());
        self.put(v.as_bytes());
        Ok(())
    }
    fn serialize_bytes(self, _: &[u8]) -> Result<(), Error> {
        unsupported("bytes")
    }
    fn serialize_none(mut self) -> Result<(), Error> {
        self.put(&[0]);
        Ok(())
    }
    fn serialize_some<T: ?Sized + Serialize>(mut self, value: &T) -> Result<(), Error> {
        self.put(&[1]);
        value.serialize(self)
    }
    fn serialize_unit(self) -> Result<(), Error> {
        Ok(())
    }
    fn serialize_unit_struct(self, _: &'static str) -> Result<(), Error> {
        Ok(())
    }
    fn serialize_unit_variant(
        self,
        _: &'static str,
        index: u32,
        _: &'static str,
    ) -> Result<(), Error> {
        self.serialize_u32(index)
    }
    fn serialize_newtype_struct<T: ?Sized + Serialize>(
        self,
        _: &'static str,
        value: &T,
    ) -> Result<(), Error> {
        value.serialize(self)
    }
    fn serialize_newtype_variant<T: ?Sized + Serialize>(
        mut self,
        _: &'static str,
        index: u32,
        _: &'static str,
        value: &T,
    ) -> Result<(), Error> {
        self.put(&index.to_le_bytes());
        value.serialize(self)
    }
    fn serialize_seq(mut self, len: Option<usize>) -> Result<Ser<'a>, Error> {
        let Some(len) = len else {
            return unsupported("a sequence of unknown length");
        };
        self.put(&(len as u64).to_le_bytes());
        Ok(self)
    }
    fn serialize_tuple(self, _: usize) -> Result<Ser<'a>, Error> {
        Ok(self)
    }
    fn serialize_tuple_struct(self, _: &'static str, _: usize) -> Result<Ser<'a>, Error> {
        Ok(self)
    }
    fn serialize_tuple_variant(
        mut self,
        _: &'static str,
        index: u32,
        _: &'static str,
        _: usize,
    ) -> Result<Ser<'a>, Error> {
        self.put(&index.to_le_bytes());
        Ok(self)
    }
    fn serialize_map(self, _: Option<usize>) -> Result<Impossible<(), Error>, Error> {
        unsupported("maps")
    }
    fn serialize_struct(self, _: &'static str, _: usize) -> Result<Ser<'a>, Error> {
        Ok(self)
    }
    fn serialize_struct_variant(
        mut self,
        _: &'static str,
        index: u32,
        _: &'static str,
        _: usize,
    ) -> Result<Ser<'a>, Error> {
        self.put(&index.to_le_bytes());
        Ok(self)
    }
}

impl ser::SerializeSeq for Ser<'_> {
    type Ok = ();
    type Error = Error;
    fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Error> {
        value.serialize(Ser(&mut *self.0))
    }
    fn end(self) -> Result<(), Error> {
        Ok(())
    }
}
impl ser::SerializeTuple for Ser<'_> {
    type Ok = ();
    type Error = Error;
    fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Error> {
        value.serialize(Ser(&mut *self.0))
    }
    fn end(self) -> Result<(), Error> {
        Ok(())
    }
}
impl ser::SerializeTupleStruct for Ser<'_> {
    type Ok = ();
    type Error = Error;
    fn serialize_field<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Error> {
        value.serialize(Ser(&mut *self.0))
    }
    fn end(self) -> Result<(), Error> {
        Ok(())
    }
}
impl ser::SerializeTupleVariant for Ser<'_> {
    type Ok = ();
    type Error = Error;
    fn serialize_field<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Error> {
        value.serialize(Ser(&mut *self.0))
    }
    fn end(self) -> Result<(), Error> {
        Ok(())
    }
}
impl ser::SerializeStruct for Ser<'_> {
    type Ok = ();
    type Error = Error;
    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        _: &'static str,
        value: &T,
    ) -> Result<(), Error> {
        value.serialize(Ser(&mut *self.0))
    }
    fn end(self) -> Result<(), Error> {
        Ok(())
    }
}
impl ser::SerializeStructVariant for Ser<'_> {
    type Ok = ();
    type Error = Error;
    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        _: &'static str,
        value: &T,
    ) -> Result<(), Error> {
        value.serialize(Ser(&mut *self.0))
    }
    fn end(self) -> Result<(), Error> {
        Ok(())
    }
}

struct De<'de> {
    input: &'de [u8],
}

impl<'de> De<'de> {
    fn take<const N: usize>(&mut self) -> Result<[u8; N], Error> {
        if self.input.len() < N {
            return Err(Error("unexpected end of input".to_owned()));
        }
        let (head, rest) = self.input.split_at(N);
        self.input = rest;
        Ok(head.try_into().expect("N bytes"))
    }
    fn len(&mut self) -> Result<usize, Error> {
        usize::try_from(u64::from_le_bytes(self.take()?)).map_err(|e| Error(e.to_string()))
    }
}

/// A sequence, tuple or struct of `left` more elements.
struct Counted<'a, 'de> {
    de: &'a mut De<'de>,
    left: usize,
}

impl<'de> de::SeqAccess<'de> for Counted<'_, 'de> {
    type Error = Error;
    fn next_element_seed<T: de::DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>, Error> {
        if self.left == 0 {
            return Ok(None);
        }
        self.left -= 1;
        seed.deserialize(&mut *self.de).map(Some)
    }
    fn size_hint(&self) -> Option<usize> {
        Some(self.left)
    }
}

impl<'de> de::EnumAccess<'de> for &mut De<'de> {
    type Error = Error;
    type Variant = Self;
    fn variant_seed<V: de::DeserializeSeed<'de>>(self, seed: V) -> Result<(V::Value, Self), Error> {
        let index = u32::from_le_bytes(self.take()?);
        let value = seed.deserialize(IntoDeserializer::<Error>::into_deserializer(index))?;
        Ok((value, self))
    }
}

impl<'de> de::VariantAccess<'de> for &mut De<'de> {
    type Error = Error;
    fn unit_variant(self) -> Result<(), Error> {
        Ok(())
    }
    fn newtype_variant_seed<T: de::DeserializeSeed<'de>>(self, seed: T) -> Result<T::Value, Error> {
        seed.deserialize(self)
    }
    fn tuple_variant<V: Visitor<'de>>(self, len: usize, visitor: V) -> Result<V::Value, Error> {
        visitor.visit_seq(Counted {
            de: self,
            left: len,
        })
    }
    fn struct_variant<V: Visitor<'de>>(
        self,
        fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Error> {
        visitor.visit_seq(Counted {
            de: self,
            left: fields.len(),
        })
    }
}

macro_rules! fixed {
    ($method:ident, $visit:ident, $t:ty) => {
        fn $method<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
            visitor.$visit(<$t>::from_le_bytes(self.take()?))
        }
    };
}

impl<'de> Deserializer<'de> for &mut De<'de> {
    type Error = Error;

    fn is_human_readable(&self) -> bool {
        false
    }
    fn deserialize_any<V: Visitor<'de>>(self, _: V) -> Result<V::Value, Error> {
        unsupported("self-describing reads")
    }
    fn deserialize_bool<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        let [b] = self.take()?;
        visitor.visit_bool(b != 0)
    }
    fixed!(deserialize_i8, visit_i8, i8);
    fixed!(deserialize_i16, visit_i16, i16);
    fixed!(deserialize_i32, visit_i32, i32);
    fixed!(deserialize_i64, visit_i64, i64);
    fixed!(deserialize_u8, visit_u8, u8);
    fixed!(deserialize_u16, visit_u16, u16);
    fixed!(deserialize_u32, visit_u32, u32);
    fixed!(deserialize_u64, visit_u64, u64);
    fn deserialize_f32<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        visitor.visit_f32(f32::from_bits(u32::from_le_bytes(self.take()?)))
    }
    fn deserialize_f64<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        visitor.visit_f64(f64::from_bits(u64::from_le_bytes(self.take()?)))
    }
    fn deserialize_char<V: Visitor<'de>>(self, _: V) -> Result<V::Value, Error> {
        unsupported("char")
    }
    fn deserialize_str<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        let len = self.len()?;
        if self.input.len() < len {
            return Err(Error("unexpected end of input".to_owned()));
        }
        let (head, rest) = self.input.split_at(len);
        self.input = rest;
        visitor.visit_borrowed_str(std::str::from_utf8(head).map_err(|e| Error(e.to_string()))?)
    }
    fn deserialize_string<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        self.deserialize_str(visitor)
    }
    fn deserialize_bytes<V: Visitor<'de>>(self, _: V) -> Result<V::Value, Error> {
        unsupported("bytes")
    }
    fn deserialize_byte_buf<V: Visitor<'de>>(self, _: V) -> Result<V::Value, Error> {
        unsupported("bytes")
    }
    fn deserialize_option<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        match self.take::<1>()? {
            [0] => visitor.visit_none(),
            _ => visitor.visit_some(self),
        }
    }
    fn deserialize_unit<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        visitor.visit_unit()
    }
    fn deserialize_unit_struct<V: Visitor<'de>>(
        self,
        _: &'static str,
        visitor: V,
    ) -> Result<V::Value, Error> {
        visitor.visit_unit()
    }
    fn deserialize_newtype_struct<V: Visitor<'de>>(
        self,
        _: &'static str,
        visitor: V,
    ) -> Result<V::Value, Error> {
        visitor.visit_newtype_struct(self)
    }
    fn deserialize_seq<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        let left = self.len()?;
        visitor.visit_seq(Counted { de: self, left })
    }
    fn deserialize_tuple<V: Visitor<'de>>(self, len: usize, visitor: V) -> Result<V::Value, Error> {
        visitor.visit_seq(Counted {
            de: self,
            left: len,
        })
    }
    fn deserialize_tuple_struct<V: Visitor<'de>>(
        self,
        _: &'static str,
        len: usize,
        visitor: V,
    ) -> Result<V::Value, Error> {
        visitor.visit_seq(Counted {
            de: self,
            left: len,
        })
    }
    fn deserialize_map<V: Visitor<'de>>(self, _: V) -> Result<V::Value, Error> {
        unsupported("maps")
    }
    fn deserialize_struct<V: Visitor<'de>>(
        self,
        _: &'static str,
        fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Error> {
        visitor.visit_seq(Counted {
            de: self,
            left: fields.len(),
        })
    }
    fn deserialize_enum<V: Visitor<'de>>(
        self,
        _: &'static str,
        _: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Error> {
        visitor.visit_enum(self)
    }
    fn deserialize_identifier<V: Visitor<'de>>(self, _: V) -> Result<V::Value, Error> {
        unsupported("identifiers")
    }
    fn deserialize_ignored_any<V: Visitor<'de>>(self, _: V) -> Result<V::Value, Error> {
        unsupported("skipped values")
    }
}

/// Every bit pattern, NaN payloads included.
fn any_f32() -> impl Strategy<Value = f32> {
    any::<u32>().prop_map(f32::from_bits)
}

fn any_f64() -> impl Strategy<Value = f64> {
    any::<u64>().prop_map(f64::from_bits)
}

fn bits(row: &TapRow) -> (u32, u64, u64, Vec<u32>, Vec<u32>) {
    let (top, tail) = match &row.values {
        TapValues::Gather { top, tail } => (
            top.iter().map(|v| v.to_bits()).collect(),
            tail.iter().map(|v| v.to_bits()).collect(),
        ),
        _ => (Vec::new(), Vec::new()),
    };
    (
        row.margin.to_bits(),
        row.lse.to_bits(),
        row.lse_complement.to_bits(),
        top,
        tail,
    )
}

proptest! {
    /// Every float of a tap row comes back with its bit pattern, whatever it is.
    #[test]
    fn every_tap_float_keeps_its_bits(
        margin in any_f32(),
        lse in any_f64(),
        lse_complement in any_f64(),
        top in prop::collection::vec(any_f32(), 0..6),
        tail in prop::collection::vec(any_f32(), 0..6),
    ) {
        let row = TapRow {
            pos: 9,
            argmax: 4,
            margin,
            lse,
            lse_complement,
            values: TapValues::Gather { top, tail },
        };
        let tap = LogitTap { k: 6, tail: 6, rows: vec![row.clone()] };
        let back: LogitTap = decode(&encode(&tap));
        prop_assert_eq!((back.k, back.tail, back.rows.len()), (6, 6, 1));
        prop_assert_eq!((back.rows[0].pos, back.rows[0].argmax), (9, 4));
        prop_assert_eq!(bits(&back.rows[0]), bits(&row));
    }
}

/// The special values RFC-0031 §6.5 names, by bit pattern: NaN payloads of both signs,
/// infinities, signed zeros and subnormals. None of them is written as a string here.
#[test]
fn special_floats_are_written_as_their_bits() {
    let specials_f32 = [
        f32::from_bits(0x7fc0_1234), // quiet NaN with a payload
        f32::from_bits(0xffa0_0001), // negative signalling-pattern NaN
        f32::INFINITY,
        f32::NEG_INFINITY,
        -0.0,
        f32::from_bits(1), // smallest subnormal
    ];
    let row = TapRow {
        pos: 0,
        argmax: 0,
        margin: f32::from_bits(0x7fc0_1234),
        lse: f64::from_bits(0x7ff8_0000_0000_beef),
        lse_complement: -0.0,
        values: TapValues::Gather {
            top: specials_f32.to_vec(),
            tail: vec![],
        },
    };
    let bytes = encode(&row);
    assert!(
        !bytes.windows(3).any(|w| w == b"nan" || w == b"inf"),
        "a binary format must not use the human-readable names"
    );
    let back: TapRow = decode(&bytes);
    assert_eq!(bits(&back), bits(&row));
}

/// The other types round-trip through the binary format too.
#[test]
fn specs_and_full_local_taps_round_trip() {
    for spec in [
        ParityTapSpec {
            stream: TeacherStreamId(1),
            mode: ForceMode::Prefill,
            tap: TapMode::FullLocal,
        },
        ParityTapSpec {
            stream: TeacherStreamId(u64::MAX),
            mode: ForceMode::Verify {
                prefill_len: 3,
                gamma: 8,
                accept: 0.5,
                seed: 7,
            },
            tap: TapMode::Gather { k: 256, tail: 64 },
        },
        ParityTapSpec {
            stream: TeacherStreamId(2),
            mode: ForceMode::Decode { prefill_len: 9 },
            tap: TapMode::FullLocal,
        },
    ] {
        assert_eq!(decode::<ParityTapSpec>(&encode(&spec)), spec);
    }
    let tap = LogitTap {
        k: 0,
        tail: 0,
        rows: vec![TapRow {
            pos: 1,
            argmax: 2,
            margin: 0.25,
            lse: 3.5,
            lse_complement: 0.0,
            values: TapValues::FullLocal { shm_offset: 4096 },
        }],
    };
    assert_eq!(decode::<LogitTap>(&encode(&tap)), tap);
}
