// SPDX-License-Identifier: MIT OR Apache-2.0

//! GGUF metadata values and [`MetaView`], the typed view over a file's key-value pairs
//! (RFC-0041 §2.1 "Metadata"; RFC-0043 §5.1).
//!
//! A family module reads hyperparameters through `MetaView` (RFC-0043 §5.1). The accessors
//! widen where no value can change: [`MetaView::u64`] takes any unsigned integer and any
//! non-negative signed one, [`MetaView::i64`] any integer that fits, [`MetaView::f64`] F32 or
//! F64. Anything else is a [`MetaError`] naming the key, never a silent default. The view over
//! `config.json` that shares these accessors comes with COMP-E16-T01.

use std::collections::BTreeMap;
use std::fmt;

/// A GGUF value type, with its ID in the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ValueType {
    /// `u8`, ID 0.
    U8,
    /// `i8`, ID 1.
    I8,
    /// `u16`, ID 2.
    U16,
    /// `i16`, ID 3.
    I16,
    /// `u32`, ID 4.
    U32,
    /// `i32`, ID 5.
    I32,
    /// `f32`, ID 6.
    F32,
    /// `bool` (one byte, 0 or 1), ID 7.
    Bool,
    /// A string: `u64` length, then that many bytes, ID 8.
    Str,
    /// An array: element type, `u64` count, then the elements, ID 9.
    Array,
    /// `u64`, ID 10.
    U64,
    /// `i64`, ID 11.
    I64,
    /// `f64`, ID 12.
    F64,
}

impl ValueType {
    /// The type with file ID `id`.
    #[must_use]
    pub fn from_id(id: u32) -> Option<Self> {
        Some(match id {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::Str,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            _ => return None,
        })
    }

    /// Its file ID.
    #[must_use]
    pub fn id(self) -> u32 {
        match self {
            Self::U8 => 0,
            Self::I8 => 1,
            Self::U16 => 2,
            Self::I16 => 3,
            Self::U32 => 4,
            Self::I32 => 5,
            Self::F32 => 6,
            Self::Bool => 7,
            Self::Str => 8,
            Self::Array => 9,
            Self::U64 => 10,
            Self::I64 => 11,
            Self::F64 => 12,
        }
    }

    /// The bytes one element takes, for the fixed-size types.
    #[must_use]
    pub fn fixed_size(self) -> Option<u64> {
        match self {
            Self::U8 | Self::I8 | Self::Bool => Some(1),
            Self::U16 | Self::I16 => Some(2),
            Self::U32 | Self::I32 | Self::F32 => Some(4),
            Self::U64 | Self::I64 | Self::F64 => Some(8),
            Self::Str | Self::Array => None,
        }
    }
}

/// A metadata value. Strings are kept as bytes: GGUF does not require UTF-8, and token
/// strings in a vocabulary need not be.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// `u8`.
    U8(u8),
    /// `i8`.
    I8(i8),
    /// `u16`.
    U16(u16),
    /// `i16`.
    I16(i16),
    /// `u32`.
    U32(u32),
    /// `i32`.
    I32(i32),
    /// `f32`.
    F32(f32),
    /// `bool`.
    Bool(bool),
    /// A string's bytes.
    Str(Box<[u8]>),
    /// An array.
    Array(Array),
    /// `u64`.
    U64(u64),
    /// `i64`.
    I64(i64),
    /// `f64`.
    F64(f64),
}

impl Value {
    /// Its type.
    #[must_use]
    pub fn value_type(&self) -> ValueType {
        match self {
            Self::U8(_) => ValueType::U8,
            Self::I8(_) => ValueType::I8,
            Self::U16(_) => ValueType::U16,
            Self::I16(_) => ValueType::I16,
            Self::U32(_) => ValueType::U32,
            Self::I32(_) => ValueType::I32,
            Self::F32(_) => ValueType::F32,
            Self::Bool(_) => ValueType::Bool,
            Self::Str(_) => ValueType::Str,
            Self::Array(_) => ValueType::Array,
            Self::U64(_) => ValueType::U64,
            Self::I64(_) => ValueType::I64,
            Self::F64(_) => ValueType::F64,
        }
    }

    /// The value as an `i128`, for any integer type.
    fn as_int(&self) -> Option<i128> {
        Some(match *self {
            Self::U8(v) => v.into(),
            Self::I8(v) => v.into(),
            Self::U16(v) => v.into(),
            Self::I16(v) => v.into(),
            Self::U32(v) => v.into(),
            Self::I32(v) => v.into(),
            Self::U64(v) => v.into(),
            Self::I64(v) => v.into(),
            _ => return None,
        })
    }
}

/// An array value: one vector per element type. GGUF arrays are homogeneous and never nested.
#[derive(Clone, Debug, PartialEq)]
pub enum Array {
    /// `u8` elements.
    U8(Vec<u8>),
    /// `i8` elements.
    I8(Vec<i8>),
    /// `u16` elements.
    U16(Vec<u16>),
    /// `i16` elements.
    I16(Vec<i16>),
    /// `u32` elements.
    U32(Vec<u32>),
    /// `i32` elements.
    I32(Vec<i32>),
    /// `f32` elements.
    F32(Vec<f32>),
    /// `bool` elements.
    Bool(Vec<bool>),
    /// String elements, as bytes.
    Str(Vec<Box<[u8]>>),
    /// `u64` elements.
    U64(Vec<u64>),
    /// `i64` elements.
    I64(Vec<i64>),
    /// `f64` elements.
    F64(Vec<f64>),
}

impl Array {
    /// The element type.
    #[must_use]
    pub fn element_type(&self) -> ValueType {
        match self {
            Self::U8(_) => ValueType::U8,
            Self::I8(_) => ValueType::I8,
            Self::U16(_) => ValueType::U16,
            Self::I16(_) => ValueType::I16,
            Self::U32(_) => ValueType::U32,
            Self::I32(_) => ValueType::I32,
            Self::F32(_) => ValueType::F32,
            Self::Bool(_) => ValueType::Bool,
            Self::Str(_) => ValueType::Str,
            Self::U64(_) => ValueType::U64,
            Self::I64(_) => ValueType::I64,
            Self::F64(_) => ValueType::F64,
        }
    }

    /// The element count.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::U8(v) => v.len(),
            Self::I8(v) => v.len(),
            Self::U16(v) => v.len(),
            Self::I16(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::Bool(v) => v.len(),
            Self::Str(v) => v.len(),
            Self::U64(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::F64(v) => v.len(),
        }
    }

    /// Whether it has no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A metadata value of the wrong type or range for the accessor that read it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetaError {
    /// The key.
    pub key: String,
    /// What the accessor accepts.
    pub expected: &'static str,
    /// What the file holds: the value's type, or the value when only its range is wrong.
    pub found: String,
}

impl fmt::Display for MetaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "metadata key `{}` holds {}, expected {}",
            self.key, self.found, self.expected
        )
    }
}

impl std::error::Error for MetaError {}

/// The typed view over a GGUF file's key-value metadata, in file order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MetaView {
    entries: Vec<(String, Value)>,
    index: BTreeMap<String, usize>,
}

impl MetaView {
    /// An empty view.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds `key`; returns `false`, and changes nothing, when the key is already present.
    pub(crate) fn insert(&mut self, key: String, value: Value) -> bool {
        if self.index.contains_key(&key) {
            return false;
        }
        self.index.insert(key.clone(), self.entries.len());
        self.entries.push((key, value));
        true
    }

    /// Whether `key` is present.
    #[must_use]
    pub fn contains(&self, key: &str) -> bool {
        self.index.contains_key(key)
    }

    /// The number of keys.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether there are no keys.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The pairs in file order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// The value of `key`, of any type.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.index.get(key).map(|&i| &self.entries[i].1)
    }

    fn err(key: &str, expected: &'static str, found: String) -> MetaError {
        MetaError {
            key: key.to_owned(),
            expected,
            found,
        }
    }

    fn type_err(key: &str, expected: &'static str, v: &Value) -> MetaError {
        Self::err(key, expected, format!("a {:?}", v.value_type()))
    }

    /// `key` as an unsigned integer: any unsigned type, or a signed one holding a value ≥ 0.
    ///
    /// # Errors
    ///
    /// [`MetaError`] for a non-integer or a negative value.
    pub fn u64(&self, key: &str) -> Result<Option<u64>, MetaError> {
        const EXPECTED: &str = "a non-negative integer";
        let Some(v) = self.get(key) else {
            return Ok(None);
        };
        let n = v.as_int().ok_or_else(|| Self::type_err(key, EXPECTED, v))?;
        u64::try_from(n)
            .map(Some)
            .map_err(|_| Self::err(key, EXPECTED, n.to_string()))
    }

    /// `key` as a `u32`: as [`u64`](Self::u64), and at most `u32::MAX`.
    ///
    /// # Errors
    ///
    /// [`MetaError`] for a non-integer or a value outside `u32`.
    pub fn u32(&self, key: &str) -> Result<Option<u32>, MetaError> {
        match self.u64(key)? {
            None => Ok(None),
            Some(n) => u32::try_from(n)
                .map(Some)
                .map_err(|_| Self::err(key, "an integer in u32's range", n.to_string())),
        }
    }

    /// `key` as a signed integer: any integer type holding a value that fits `i64`.
    ///
    /// # Errors
    ///
    /// [`MetaError`] for a non-integer or a `u64` above `i64::MAX`.
    pub fn i64(&self, key: &str) -> Result<Option<i64>, MetaError> {
        const EXPECTED: &str = "an integer in i64's range";
        let Some(v) = self.get(key) else {
            return Ok(None);
        };
        let n = v.as_int().ok_or_else(|| Self::type_err(key, EXPECTED, v))?;
        i64::try_from(n)
            .map(Some)
            .map_err(|_| Self::err(key, EXPECTED, n.to_string()))
    }

    /// `key` as a float: F32 (widened exactly) or F64.
    ///
    /// # Errors
    ///
    /// [`MetaError`] for any other type.
    pub fn f64(&self, key: &str) -> Result<Option<f64>, MetaError> {
        match self.get(key) {
            None => Ok(None),
            Some(Value::F32(x)) => Ok(Some(f64::from(*x))),
            Some(Value::F64(x)) => Ok(Some(*x)),
            Some(v) => Err(Self::type_err(key, "an F32 or F64", v)),
        }
    }

    /// `key` as a bool.
    ///
    /// # Errors
    ///
    /// [`MetaError`] for any other type.
    pub fn bool(&self, key: &str) -> Result<Option<bool>, MetaError> {
        match self.get(key) {
            None => Ok(None),
            Some(Value::Bool(b)) => Ok(Some(*b)),
            Some(v) => Err(Self::type_err(key, "a bool", v)),
        }
    }

    /// `key` as a UTF-8 string.
    ///
    /// # Errors
    ///
    /// [`MetaError`] for another type or a string that is not UTF-8.
    pub fn str(&self, key: &str) -> Result<Option<&str>, MetaError> {
        match self.get(key) {
            None => Ok(None),
            Some(Value::Str(s)) => std::str::from_utf8(s)
                .map(Some)
                .map_err(|_| Self::err(key, "a UTF-8 string", "a string that is not UTF-8".into())),
            Some(v) => Err(Self::type_err(key, "a string", v)),
        }
    }

    /// `key` as an array of any element type.
    ///
    /// # Errors
    ///
    /// [`MetaError`] for a value that is not an array.
    pub fn array(&self, key: &str) -> Result<Option<&Array>, MetaError> {
        match self.get(key) {
            None => Ok(None),
            Some(Value::Array(a)) => Ok(Some(a)),
            Some(v) => Err(Self::type_err(key, "an array", v)),
        }
    }

    /// `general.architecture`, the architecture string a family is resolved by
    /// (RFC-0043 §5.1).
    ///
    /// # Errors
    ///
    /// As [`str`](Self::str).
    pub fn architecture(&self) -> Result<Option<&str>, MetaError> {
        self.str("general.architecture")
    }
}
