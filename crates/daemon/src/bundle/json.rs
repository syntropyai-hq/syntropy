// SPDX-License-Identifier: MIT OR Apache-2.0

//! A record as a JSON value, for the bundle's projection (RFC-0053 §6).
//!
//! `serde_json` refuses a map whose keys are not strings, which the cluster state has (a
//! `NodeId`, a `(NodeId, PlanEpoch)` tuple). This serializer writes such a key as text. It
//! reports itself human-readable, as `serde_json` does, so a type with two forms writes its text
//! one: an `IpAddr` or `SocketAddr` is `"203.0.113.7"` or `"[2001:db8::5]:9000"`, never a byte
//! array, and the address redactor sees it. A compact serializer such as CBOR's writes those
//! types as integer arrays, which no text scan finds.
//!
//! A key that is not a string becomes its text when it serialises to one, lower-case hex when
//! it is an array of bytes, and its compact JSON otherwise. Bytes are lower-case hex. Enums are
//! externally tagged, as `serde_json` writes them.

use std::fmt;

use serde::Serialize;
use serde::ser;
use serde_json::{Map, Number, Value};

/// Why a record could not be written.
#[derive(Debug)]
pub(crate) struct Error(String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl ser::Error for Error {
    fn custom<T: fmt::Display>(message: T) -> Error {
        Error(message.to_string())
    }
}

/// `value` as JSON, human-readable, with every map key as text.
pub(crate) fn to_value<T: Serialize + ?Sized>(value: &T) -> Result<Value, Error> {
    value.serialize(Ser)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// A key's text (module docs).
fn key_text(key: Value) -> String {
    match key {
        Value::String(s) => s,
        Value::Array(items)
            if !items.is_empty() && items.iter().all(|v| v.as_u64().is_some_and(|n| n <= 255)) =>
        {
            items
                .iter()
                .map(|v| format!("{:02x}", v.as_u64().unwrap_or(0)))
                .collect()
        }
        other => other.to_string(),
    }
}

struct Ser;

impl ser::Serializer for Ser {
    type Ok = Value;
    type Error = Error;
    type SerializeSeq = Seq;
    type SerializeTuple = Seq;
    type SerializeTupleStruct = Seq;
    type SerializeTupleVariant = Seq;
    type SerializeMap = Obj;
    type SerializeStruct = Obj;
    type SerializeStructVariant = Obj;

    fn is_human_readable(&self) -> bool {
        true
    }

    fn serialize_bool(self, v: bool) -> Result<Value, Error> {
        Ok(Value::Bool(v))
    }
    fn serialize_i8(self, v: i8) -> Result<Value, Error> {
        Ok(Value::from(v))
    }
    fn serialize_i16(self, v: i16) -> Result<Value, Error> {
        Ok(Value::from(v))
    }
    fn serialize_i32(self, v: i32) -> Result<Value, Error> {
        Ok(Value::from(v))
    }
    fn serialize_i64(self, v: i64) -> Result<Value, Error> {
        Ok(Value::from(v))
    }
    fn serialize_i128(self, v: i128) -> Result<Value, Error> {
        Ok(i64::try_from(v).map_or_else(|_| Value::String(v.to_string()), Value::from))
    }
    fn serialize_u8(self, v: u8) -> Result<Value, Error> {
        Ok(Value::from(v))
    }
    fn serialize_u16(self, v: u16) -> Result<Value, Error> {
        Ok(Value::from(v))
    }
    fn serialize_u32(self, v: u32) -> Result<Value, Error> {
        Ok(Value::from(v))
    }
    fn serialize_u64(self, v: u64) -> Result<Value, Error> {
        Ok(Value::from(v))
    }
    fn serialize_u128(self, v: u128) -> Result<Value, Error> {
        Ok(u64::try_from(v).map_or_else(|_| Value::String(v.to_string()), Value::from))
    }
    fn serialize_f32(self, v: f32) -> Result<Value, Error> {
        self.serialize_f64(f64::from(v))
    }
    fn serialize_f64(self, v: f64) -> Result<Value, Error> {
        Ok(Number::from_f64(v).map_or(Value::Null, Value::Number))
    }
    fn serialize_char(self, v: char) -> Result<Value, Error> {
        Ok(Value::String(v.to_string()))
    }
    fn serialize_str(self, v: &str) -> Result<Value, Error> {
        Ok(Value::String(v.to_owned()))
    }
    fn serialize_bytes(self, v: &[u8]) -> Result<Value, Error> {
        Ok(Value::String(hex(v)))
    }
    fn serialize_none(self) -> Result<Value, Error> {
        Ok(Value::Null)
    }
    fn serialize_some<T: Serialize + ?Sized>(self, value: &T) -> Result<Value, Error> {
        value.serialize(Ser)
    }
    fn serialize_unit(self) -> Result<Value, Error> {
        Ok(Value::Null)
    }
    fn serialize_unit_struct(self, _name: &'static str) -> Result<Value, Error> {
        Ok(Value::Null)
    }
    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _index: u32,
        variant: &'static str,
    ) -> Result<Value, Error> {
        Ok(Value::String(variant.to_owned()))
    }
    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        _name: &'static str,
        value: &T,
    ) -> Result<Value, Error> {
        value.serialize(Ser)
    }
    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        _name: &'static str,
        _index: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<Value, Error> {
        let mut object = Map::new();
        object.insert(variant.to_owned(), value.serialize(Ser)?);
        Ok(Value::Object(object))
    }
    fn serialize_seq(self, len: Option<usize>) -> Result<Seq, Error> {
        Ok(Seq {
            variant: None,
            items: Vec::with_capacity(len.unwrap_or(0)),
        })
    }
    fn serialize_tuple(self, len: usize) -> Result<Seq, Error> {
        self.serialize_seq(Some(len))
    }
    fn serialize_tuple_struct(self, _name: &'static str, len: usize) -> Result<Seq, Error> {
        self.serialize_seq(Some(len))
    }
    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Seq, Error> {
        Ok(Seq {
            variant: Some(variant),
            items: Vec::with_capacity(len),
        })
    }
    fn serialize_map(self, _len: Option<usize>) -> Result<Obj, Error> {
        Ok(Obj {
            variant: None,
            map: Map::new(),
            key: None,
        })
    }
    fn serialize_struct(self, _name: &'static str, _len: usize) -> Result<Obj, Error> {
        self.serialize_map(None)
    }
    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _index: u32,
        variant: &'static str,
        _len: usize,
    ) -> Result<Obj, Error> {
        Ok(Obj {
            variant: Some(variant),
            map: Map::new(),
            key: None,
        })
    }
}

/// An array, or a tuple variant's payload.
struct Seq {
    variant: Option<&'static str>,
    items: Vec<Value>,
}

impl Seq {
    fn push<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Error> {
        self.items.push(value.serialize(Ser)?);
        Ok(())
    }

    fn done(self) -> Value {
        let array = Value::Array(self.items);
        match self.variant {
            None => array,
            Some(variant) => {
                let mut object = Map::new();
                object.insert(variant.to_owned(), array);
                Value::Object(object)
            }
        }
    }
}

impl ser::SerializeSeq for Seq {
    type Ok = Value;
    type Error = Error;
    fn serialize_element<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Error> {
        self.push(value)
    }
    fn end(self) -> Result<Value, Error> {
        Ok(self.done())
    }
}

impl ser::SerializeTuple for Seq {
    type Ok = Value;
    type Error = Error;
    fn serialize_element<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Error> {
        self.push(value)
    }
    fn end(self) -> Result<Value, Error> {
        Ok(self.done())
    }
}

impl ser::SerializeTupleStruct for Seq {
    type Ok = Value;
    type Error = Error;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Error> {
        self.push(value)
    }
    fn end(self) -> Result<Value, Error> {
        Ok(self.done())
    }
}

impl ser::SerializeTupleVariant for Seq {
    type Ok = Value;
    type Error = Error;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Error> {
        self.push(value)
    }
    fn end(self) -> Result<Value, Error> {
        Ok(self.done())
    }
}

/// An object: a map, a struct, or a struct variant's payload.
struct Obj {
    variant: Option<&'static str>,
    map: Map<String, Value>,
    key: Option<String>,
}

impl Obj {
    fn done(self) -> Value {
        let object = Value::Object(self.map);
        match self.variant {
            None => object,
            Some(variant) => {
                let mut outer = Map::new();
                outer.insert(variant.to_owned(), object);
                Value::Object(outer)
            }
        }
    }
}

impl ser::SerializeMap for Obj {
    type Ok = Value;
    type Error = Error;
    fn serialize_key<T: Serialize + ?Sized>(&mut self, key: &T) -> Result<(), Error> {
        self.key = Some(key_text(key.serialize(Ser)?));
        Ok(())
    }
    fn serialize_value<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Error> {
        let key = self
            .key
            .take()
            .ok_or_else(|| Error("a map value came before its key".to_owned()))?;
        self.map.insert(key, value.serialize(Ser)?);
        Ok(())
    }
    fn end(self) -> Result<Value, Error> {
        Ok(self.done())
    }
}

impl ser::SerializeStruct for Obj {
    type Ok = Value;
    type Error = Error;
    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), Error> {
        self.map.insert(key.to_owned(), value.serialize(Ser)?);
        Ok(())
    }
    fn end(self) -> Result<Value, Error> {
        Ok(self.done())
    }
}

impl ser::SerializeStructVariant for Obj {
    type Ok = Value;
    type Error = Error;
    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), Error> {
        self.map.insert(key.to_owned(), value.serialize(Ser)?);
        Ok(())
    }
    fn end(self) -> Result<Value, Error> {
        Ok(self.done())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::net::{IpAddr, SocketAddr};

    use serde_json::json;

    use super::*;

    #[derive(Serialize)]
    enum E {
        Unit,
        New(u8),
        Tup(u8, u8),
        Strukt { a: u8 },
    }

    #[test]
    fn records_write_as_serde_json_writes_them_and_keys_become_text() {
        let mut by_tuple = BTreeMap::new();
        by_tuple.insert(([1u8; 2], 7u64), "x");
        let mut by_bytes = BTreeMap::new();
        by_bytes.insert([0xabu8; 3], 1);
        let value = to_value(&(
            E::Unit,
            E::New(1),
            E::Tup(2, 3),
            E::Strukt { a: 4 },
            by_tuple,
            by_bytes,
            Some(5u8),
            None::<u8>,
            u128::MAX,
        ))
        .unwrap();
        assert_eq!(
            value,
            json!([
                "Unit",
                {"New": 1},
                {"Tup": [2, 3]},
                {"Strukt": {"a": 4}},
                {"[[1,1],7]": "x"},
                {"ababab": 1},
                5,
                null,
                u128::MAX.to_string(),
            ])
        );
    }

    /// The review of #1967, B2: typed addresses come out as text, which the redactor scans.
    #[test]
    fn addresses_are_written_as_text() {
        let ip: IpAddr = "203.0.113.77".parse().unwrap();
        let sock: SocketAddr = "[2001:db8::5]:9000".parse().unwrap();
        let mut keyed = BTreeMap::new();
        keyed.insert(sock, ip);
        assert_eq!(
            to_value(&(ip, sock, keyed)).unwrap(),
            json!(["203.0.113.77", "[2001:db8::5]:9000", {"[2001:db8::5]:9000": "203.0.113.77"}])
        );
    }
}
