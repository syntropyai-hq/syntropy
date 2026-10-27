// SPDX-License-Identifier: MIT OR Apache-2.0

//! No message skips a field when it is written (RFC-0001 §18.20).
//!
//! Postcard reads a struct's fields by position, so a field that `skip_serializing_if` leaves out
//! shifts everything after it, and the other side misreads or refuses the message. A type the
//! IPC carries may skip a field only in the canonical encoding, through
//! `syntropy_core::canon::absent_if_none`. This walks every fixture message, including a plan
//! with every optional field unset, through a serializer that records each skipped field by
//! type and name. The walk sees only the values it is given; QUAL-E17-T15's schema fingerprint
//! is the check over the types themselves.

use std::fmt;

use serde::Serialize;
use serde::ser;

use super::{daemon_msgs, runner_msgs};

/// Every field skipped while writing a value, as `Type.field`.
#[derive(Default)]
struct Skips(Vec<String>);

#[derive(Debug)]
struct Never;

impl fmt::Display for Never {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("the skip walk does not fail")
    }
}

impl std::error::Error for Never {}

impl ser::Error for Never {
    fn custom<T: fmt::Display>(_: T) -> Never {
        Never
    }
}

/// The serializer: every value is accepted, and nested values are walked.
struct Walk<'a>(&'a mut Skips);

/// A compound value's elements, walked in turn; a struct's skipped fields are recorded under
/// its type's name.
struct Parts<'a> {
    skips: &'a mut Skips,
    ty: &'static str,
}

impl Parts<'_> {
    fn walk<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Never> {
        value.serialize(Walk(self.skips))
    }

    fn skip(&mut self, field: &'static str) -> Result<(), Never> {
        self.skips.0.push(format!("{}.{field}", self.ty));
        Ok(())
    }
}

macro_rules! accept {
    ($($method:ident($($arg:ty),*);)*) => {
        $(fn $method(self, $(_: $arg),*) -> Result<(), Never> { Ok(()) })*
    };
}

impl<'a> ser::Serializer for Walk<'a> {
    type Ok = ();
    type Error = Never;
    type SerializeSeq = Parts<'a>;
    type SerializeTuple = Parts<'a>;
    type SerializeTupleStruct = Parts<'a>;
    type SerializeTupleVariant = Parts<'a>;
    type SerializeMap = Parts<'a>;
    type SerializeStruct = Parts<'a>;
    type SerializeStructVariant = Parts<'a>;

    accept! {
        serialize_bool(bool);
        serialize_i8(i8);
        serialize_i16(i16);
        serialize_i32(i32);
        serialize_i64(i64);
        serialize_i128(i128);
        serialize_u8(u8);
        serialize_u16(u16);
        serialize_u32(u32);
        serialize_u64(u64);
        serialize_u128(u128);
        serialize_f32(f32);
        serialize_f64(f64);
        serialize_char(char);
        serialize_str(&str);
        serialize_bytes(&[u8]);
        serialize_none();
        serialize_unit();
        serialize_unit_struct(&'static str);
        serialize_unit_variant(&'static str, u32, &'static str);
    }

    fn serialize_some<T: Serialize + ?Sized>(self, value: &T) -> Result<(), Never> {
        value.serialize(self)
    }

    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        value: &T,
    ) -> Result<(), Never> {
        value.serialize(self)
    }

    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
        value: &T,
    ) -> Result<(), Never> {
        value.serialize(self)
    }

    fn serialize_seq(self, _: Option<usize>) -> Result<Parts<'a>, Never> {
        Ok(Parts {
            skips: self.0,
            ty: "seq",
        })
    }

    fn serialize_tuple(self, _: usize) -> Result<Parts<'a>, Never> {
        self.serialize_seq(None)
    }

    fn serialize_tuple_struct(self, ty: &'static str, _: usize) -> Result<Parts<'a>, Never> {
        Ok(Parts { skips: self.0, ty })
    }

    fn serialize_tuple_variant(
        self,
        ty: &'static str,
        _: u32,
        _: &'static str,
        _: usize,
    ) -> Result<Parts<'a>, Never> {
        Ok(Parts { skips: self.0, ty })
    }

    fn serialize_map(self, _: Option<usize>) -> Result<Parts<'a>, Never> {
        self.serialize_seq(None)
    }

    fn serialize_struct(self, ty: &'static str, _: usize) -> Result<Parts<'a>, Never> {
        Ok(Parts { skips: self.0, ty })
    }

    fn serialize_struct_variant(
        self,
        ty: &'static str,
        _: u32,
        variant: &'static str,
        _: usize,
    ) -> Result<Parts<'a>, Never> {
        // `Type::Variant`, leaked once per variant the walk meets: a test's handful.
        let ty: &'static str = Box::leak(format!("{ty}::{variant}").into_boxed_str());
        Ok(Parts { skips: self.0, ty })
    }

    fn collect_str<T: fmt::Display + ?Sized>(self, _: &T) -> Result<(), Never> {
        Ok(())
    }
}

impl ser::SerializeSeq for Parts<'_> {
    type Ok = ();
    type Error = Never;
    fn serialize_element<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Never> {
        self.walk(value)
    }
    fn end(self) -> Result<(), Never> {
        Ok(())
    }
}

impl ser::SerializeTuple for Parts<'_> {
    type Ok = ();
    type Error = Never;
    fn serialize_element<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Never> {
        self.walk(value)
    }
    fn end(self) -> Result<(), Never> {
        Ok(())
    }
}

impl ser::SerializeTupleStruct for Parts<'_> {
    type Ok = ();
    type Error = Never;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Never> {
        self.walk(value)
    }
    fn end(self) -> Result<(), Never> {
        Ok(())
    }
}

impl ser::SerializeTupleVariant for Parts<'_> {
    type Ok = ();
    type Error = Never;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Never> {
        self.walk(value)
    }
    fn end(self) -> Result<(), Never> {
        Ok(())
    }
}

impl ser::SerializeMap for Parts<'_> {
    type Ok = ();
    type Error = Never;
    fn serialize_key<T: Serialize + ?Sized>(&mut self, key: &T) -> Result<(), Never> {
        self.walk(key)
    }
    fn serialize_value<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Never> {
        self.walk(value)
    }
    fn end(self) -> Result<(), Never> {
        Ok(())
    }
}

impl ser::SerializeStruct for Parts<'_> {
    type Ok = ();
    type Error = Never;
    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        _: &'static str,
        value: &T,
    ) -> Result<(), Never> {
        self.walk(value)
    }
    fn skip_field(&mut self, field: &'static str) -> Result<(), Never> {
        self.skip(field)
    }
    fn end(self) -> Result<(), Never> {
        Ok(())
    }
}

impl ser::SerializeStructVariant for Parts<'_> {
    type Ok = ();
    type Error = Never;
    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        _: &'static str,
        value: &T,
    ) -> Result<(), Never> {
        self.walk(value)
    }
    fn skip_field(&mut self, field: &'static str) -> Result<(), Never> {
        self.skip(field)
    }
    fn end(self) -> Result<(), Never> {
        Ok(())
    }
}

/// The fields `value` skips when written.
fn skipped<T: Serialize>(value: &T) -> Vec<String> {
    let mut skips = Skips::default();
    let _: Result<(), Never> = value.serialize(Walk(&mut skips));
    skips.0
}

#[test]
fn no_message_skips_a_field() {
    for msg in daemon_msgs() {
        assert_eq!(skipped(&msg), Vec::<String>::new(), "{msg:?}");
    }
    for msg in runner_msgs() {
        assert_eq!(skipped(&msg), Vec::<String>::new(), "{msg:?}");
    }
}

#[test]
fn the_walk_sees_a_skipped_field() {
    #[derive(Serialize)]
    struct Skipping {
        kept: u8,
        #[serde(skip_serializing_if = "Option::is_none")]
        gone: Option<u8>,
    }
    #[derive(Serialize)]
    enum Outer {
        Inner { values: Vec<Skipping> },
    }
    let value = Outer::Inner {
        values: vec![
            Skipping {
                kept: 1,
                gone: Some(2),
            },
            Skipping {
                kept: 3,
                gone: None,
            },
        ],
    };
    assert_eq!(skipped(&value), ["Skipping.gone"]);
    // The canonical-only skip writes the field here.
    assert_eq!(
        skipped(&syntropy_core::ApproxParams::default()),
        Vec::<String>::new()
    );
}
