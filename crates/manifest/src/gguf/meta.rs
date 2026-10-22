// SPDX-License-Identifier: MIT OR Apache-2.0

//! GGUF metadata values and [`MetaView`], the typed view over a file's key-value pairs
//! (RFC-0041 §2.1 "Metadata"; RFC-0043 §5.1).
//!
//! [`MetaView`] keeps the metadata's bytes as the file has them and decodes values on access
//! ([`ValueRef`], [`ArrayRef`]); [`Value`] and [`Array`] are owned copies.
//!
//! A family module reads hyperparameters through `MetaView` (RFC-0043 §5.1). The accessors
//! widen where no value can change: [`MetaView::u64`] takes any unsigned integer and any
//! non-negative signed one, [`MetaView::i64`] any integer that fits, [`MetaView::f64`] F32 or
//! F64. Anything else is a [`MetaError`] naming the key, never a silent default. The view over
//! `config.json` that shares these accessors comes with COMP-E16-T01.

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
    /// String elements, as bytes, packed into one buffer.
    Str(StrArray),
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

/// The elements of an owned string array, packed: every string's bytes in one buffer, and
/// each string's end offset. [`ArrayRef::to_array`] makes one; the reader itself keeps string
/// arrays as bytes in the header buffer and allocates nothing per string.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StrArray {
    bytes: Vec<u8>,
    ends: Vec<usize>,
}

impl StrArray {
    /// An empty array with room for `n` strings of `bytes` bytes in total.
    #[must_use]
    pub fn with_capacity(n: usize, bytes: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(bytes),
            ends: Vec::with_capacity(n),
        }
    }

    /// Appends a string.
    pub fn push(&mut self, s: &[u8]) {
        self.bytes.extend_from_slice(s);
        self.ends.push(self.bytes.len());
    }

    /// The number of strings.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ends.len()
    }

    /// Whether there are no strings.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ends.is_empty()
    }

    /// String `i`'s bytes.
    #[must_use]
    pub fn get(&self, i: usize) -> Option<&[u8]> {
        let end = *self.ends.get(i)?;
        let start = if i == 0 { 0 } else { self.ends[i - 1] };
        self.bytes.get(start..end)
    }

    /// The strings in order.
    pub fn iter(&self) -> impl Iterator<Item = &[u8]> {
        (0..self.len()).filter_map(|i| self.get(i))
    }

    /// The heap bytes it holds: its buffers' capacities.
    #[must_use]
    pub fn heap_bytes(&self) -> usize {
        self.bytes.capacity() + self.ends.capacity() * std::mem::size_of::<usize>()
    }
}

impl<'a> FromIterator<&'a [u8]> for StrArray {
    fn from_iter<I: IntoIterator<Item = &'a [u8]>>(iter: I) -> Self {
        let mut out = Self::default();
        for s in iter {
            out.push(s);
        }
        out
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

/// Reads a little-endian `N`-byte field at `at`; zeros when out of range, which a validated
/// buffer never is.
fn field<const N: usize>(buf: &[u8], at: usize) -> [u8; N] {
    buf.get(at..at.saturating_add(N))
        .and_then(|b| b.try_into().ok())
        .unwrap_or([0; N])
}

fn rd_u32(buf: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(field(buf, at))
}

fn rd_len(buf: &[u8], at: usize) -> usize {
    usize::try_from(u64::from_le_bytes(field(buf, at))).unwrap_or(usize::MAX)
}

/// The GGUF string at `at`: its bytes, and the position after it.
fn rd_str(buf: &[u8], at: usize) -> (&[u8], usize) {
    let len = rd_len(buf, at);
    let start = at.saturating_add(8);
    let end = start.saturating_add(len);
    (buf.get(start..end).unwrap_or(&[]), end)
}

/// A borrowed metadata value, decoded on access from the header bytes the reader validated.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ValueRef<'a> {
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
    Str(&'a [u8]),
    /// An array.
    Array(ArrayRef<'a>),
    /// `u64`.
    U64(u64),
    /// `i64`.
    I64(i64),
    /// `f64`.
    F64(f64),
}

impl<'a> ValueRef<'a> {
    /// Decodes the value of type `ty` at `at`; returns it and the position after it.
    fn decode(buf: &'a [u8], ty: ValueType, at: usize) -> (Self, usize) {
        let next = |n: usize| at.saturating_add(n);
        match ty {
            ValueType::U8 => (Self::U8(u8::from_le_bytes(field(buf, at))), next(1)),
            ValueType::I8 => (Self::I8(i8::from_le_bytes(field(buf, at))), next(1)),
            ValueType::U16 => (Self::U16(u16::from_le_bytes(field(buf, at))), next(2)),
            ValueType::I16 => (Self::I16(i16::from_le_bytes(field(buf, at))), next(2)),
            ValueType::U32 => (Self::U32(rd_u32(buf, at)), next(4)),
            ValueType::I32 => (Self::I32(i32::from_le_bytes(field(buf, at))), next(4)),
            ValueType::F32 => (Self::F32(f32::from_le_bytes(field(buf, at))), next(4)),
            ValueType::Bool => (Self::Bool(field::<1>(buf, at)[0] == 1), next(1)),
            ValueType::U64 => (Self::U64(u64::from_le_bytes(field(buf, at))), next(8)),
            ValueType::I64 => (Self::I64(i64::from_le_bytes(field(buf, at))), next(8)),
            ValueType::F64 => (Self::F64(f64::from_le_bytes(field(buf, at))), next(8)),
            ValueType::Str => {
                let (s, end) = rd_str(buf, at);
                (Self::Str(s), end)
            }
            ValueType::Array => {
                let elem = ValueType::from_id(rd_u32(buf, at)).unwrap_or(ValueType::U8);
                let n = rd_len(buf, at.saturating_add(4));
                let start = at.saturating_add(12);
                let end = match elem.fixed_size() {
                    Some(size) => start.saturating_add(n.saturating_mul(size as usize)),
                    None => (0..n).fold(start, |p, _| rd_str(buf, p).1),
                };
                let data = buf.get(start..end).unwrap_or(&[]);
                (Self::Array(ArrayRef { elem, n, data }), end)
            }
        }
    }

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

    /// An owned copy.
    #[must_use]
    pub fn to_value(&self) -> Value {
        match *self {
            Self::U8(v) => Value::U8(v),
            Self::I8(v) => Value::I8(v),
            Self::U16(v) => Value::U16(v),
            Self::I16(v) => Value::I16(v),
            Self::U32(v) => Value::U32(v),
            Self::I32(v) => Value::I32(v),
            Self::F32(v) => Value::F32(v),
            Self::Bool(v) => Value::Bool(v),
            Self::Str(s) => Value::Str(s.into()),
            Self::Array(a) => Value::Array(a.to_array()),
            Self::U64(v) => Value::U64(v),
            Self::I64(v) => Value::I64(v),
            Self::F64(v) => Value::F64(v),
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

/// A borrowed array value: its element type, its length and its bytes, decoded on access.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArrayRef<'a> {
    elem: ValueType,
    n: usize,
    data: &'a [u8],
}

impl<'a> ArrayRef<'a> {
    /// The element type.
    #[must_use]
    pub fn element_type(&self) -> ValueType {
        self.elem
    }

    /// The element count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether it has no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// The elements in order.
    pub fn iter(&self) -> impl Iterator<Item = ValueRef<'a>> + use<'a> {
        let (data, elem) = (self.data, self.elem);
        let mut at = 0usize;
        (0..self.n).map(move |_| {
            let (v, next) = ValueRef::decode(data, elem, at);
            at = next;
            v
        })
    }

    /// The elements as byte strings, when they are strings.
    #[must_use]
    pub fn strings(&self) -> Option<impl Iterator<Item = &'a [u8]> + use<'a>> {
        (self.elem == ValueType::Str).then(|| {
            self.iter().map(|v| match v {
                ValueRef::Str(s) => s,
                _ => &[][..],
            })
        })
    }

    /// An owned copy.
    #[must_use]
    pub fn to_array(&self) -> Array {
        fn all<'a, T>(a: &ArrayRef<'a>, f: impl Fn(ValueRef<'a>) -> Option<T>) -> Vec<T> {
            a.iter().filter_map(f).collect()
        }
        match self.elem {
            ValueType::U8 => Array::U8(all(self, |v| match v {
                ValueRef::U8(x) => Some(x),
                _ => None,
            })),
            ValueType::I8 => Array::I8(all(self, |v| match v {
                ValueRef::I8(x) => Some(x),
                _ => None,
            })),
            ValueType::U16 => Array::U16(all(self, |v| match v {
                ValueRef::U16(x) => Some(x),
                _ => None,
            })),
            ValueType::I16 => Array::I16(all(self, |v| match v {
                ValueRef::I16(x) => Some(x),
                _ => None,
            })),
            ValueType::U32 => Array::U32(all(self, |v| match v {
                ValueRef::U32(x) => Some(x),
                _ => None,
            })),
            ValueType::I32 => Array::I32(all(self, |v| match v {
                ValueRef::I32(x) => Some(x),
                _ => None,
            })),
            ValueType::F32 => Array::F32(all(self, |v| match v {
                ValueRef::F32(x) => Some(x),
                _ => None,
            })),
            ValueType::Bool => Array::Bool(all(self, |v| match v {
                ValueRef::Bool(x) => Some(x),
                _ => None,
            })),
            ValueType::U64 => Array::U64(all(self, |v| match v {
                ValueRef::U64(x) => Some(x),
                _ => None,
            })),
            ValueType::I64 => Array::I64(all(self, |v| match v {
                ValueRef::I64(x) => Some(x),
                _ => None,
            })),
            ValueType::F64 => Array::F64(all(self, |v| match v {
                ValueRef::F64(x) => Some(x),
                _ => None,
            })),
            ValueType::Str | ValueType::Array => {
                Array::Str(self.strings().into_iter().flatten().collect())
            }
        }
    }
}

/// The typed view over a GGUF file's key-value metadata.
///
/// It holds the metadata's bytes as the file has them, validated by the reader, with one
/// position per pair and a sorted index for lookup: 8 bytes per pair beyond the input, against
/// at least 13 input bytes per pair. Values are decoded on access ([`ValueRef`]), so a
/// vocabulary of 2²⁴ strings costs its bytes and nothing per string.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MetaView {
    buf: Vec<u8>,
    /// Each pair's position in `buf`, in file order.
    pos: Vec<u32>,
    /// Indices into `pos`, sorted by key.
    sorted: Vec<u32>,
}

impl MetaView {
    /// An empty view.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A view over `buf`, the key-value pairs the reader validated, with each pair's position.
    /// On a duplicate key it returns the position of the pair that repeats it.
    pub(crate) fn from_parts(buf: Vec<u8>, pos: Vec<u32>) -> Result<Self, u32> {
        let mut view = Self {
            buf,
            sorted: (0..u32::try_from(pos.len()).unwrap_or(u32::MAX)).collect(),
            pos,
        };
        let mut sorted = std::mem::take(&mut view.sorted);
        sorted.sort_unstable_by(|&a, &b| view.key_bytes(a).cmp(view.key_bytes(b)).then(a.cmp(&b)));
        if let Some(w) = sorted
            .windows(2)
            .find(|w| view.key_bytes(w[0]) == view.key_bytes(w[1]))
        {
            return Err(view.pos[w[1] as usize]);
        }
        view.sorted = sorted;
        Ok(view)
    }

    /// The heap bytes it holds: its buffers' capacities.
    #[must_use]
    pub fn heap_bytes(&self) -> usize {
        self.buf.capacity() + 4 * (self.pos.capacity() + self.sorted.capacity())
    }

    fn key_bytes(&self, i: u32) -> &[u8] {
        let at = self.pos.get(i as usize).map_or(usize::MAX, |&p| p as usize);
        rd_str(&self.buf, at).0
    }

    fn entry(&self, i: u32) -> (&str, ValueRef<'_>) {
        let at = self.pos.get(i as usize).map_or(usize::MAX, |&p| p as usize);
        let (key, after) = rd_str(&self.buf, at);
        let ty = ValueType::from_id(rd_u32(&self.buf, after)).unwrap_or(ValueType::U8);
        let (value, _) = ValueRef::decode(&self.buf, ty, after.saturating_add(4));
        (std::str::from_utf8(key).unwrap_or(""), value)
    }

    fn find(&self, key: &str) -> Option<u32> {
        self.sorted
            .binary_search_by(|&i| self.key_bytes(i).cmp(key.as_bytes()))
            .ok()
            .map(|j| self.sorted[j])
    }

    /// Whether `key` is present.
    #[must_use]
    pub fn contains(&self, key: &str) -> bool {
        self.find(key).is_some()
    }

    /// The number of keys.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pos.len()
    }

    /// Whether there are no keys.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pos.is_empty()
    }

    /// The pairs in file order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, ValueRef<'_>)> {
        (0..u32::try_from(self.len()).unwrap_or(u32::MAX)).map(|i| self.entry(i))
    }

    /// The value of `key`, of any type.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<ValueRef<'_>> {
        self.find(key).map(|i| self.entry(i).1)
    }

    fn err(key: &str, expected: &'static str, found: String) -> MetaError {
        MetaError {
            key: key.to_owned(),
            expected,
            found,
        }
    }

    fn type_err(key: &str, expected: &'static str, v: &ValueRef<'_>) -> MetaError {
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
        let n = v
            .as_int()
            .ok_or_else(|| Self::type_err(key, EXPECTED, &v))?;
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
        let n = v
            .as_int()
            .ok_or_else(|| Self::type_err(key, EXPECTED, &v))?;
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
            Some(ValueRef::F32(x)) => Ok(Some(f64::from(x))),
            Some(ValueRef::F64(x)) => Ok(Some(x)),
            Some(v) => Err(Self::type_err(key, "an F32 or F64", &v)),
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
            Some(ValueRef::Bool(b)) => Ok(Some(b)),
            Some(v) => Err(Self::type_err(key, "a bool", &v)),
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
            Some(ValueRef::Str(s)) => std::str::from_utf8(s)
                .map(Some)
                .map_err(|_| Self::err(key, "a UTF-8 string", "a string that is not UTF-8".into())),
            Some(v) => Err(Self::type_err(key, "a string", &v)),
        }
    }

    /// `key` as an array of any element type.
    ///
    /// # Errors
    ///
    /// [`MetaError`] for a value that is not an array.
    pub fn array(&self, key: &str) -> Result<Option<ArrayRef<'_>>, MetaError> {
        match self.get(key) {
            None => Ok(None),
            Some(ValueRef::Array(a)) => Ok(Some(a)),
            Some(v) => Err(Self::type_err(key, "an array", &v)),
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
