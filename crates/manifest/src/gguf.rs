// SPDX-License-Identifier: MIT OR Apache-2.0

//! The GGUF reader (RFC-0041 §2.1; ADR-0005 decision 1; ADR-0001 decision 4).
//!
//! It parses a GGUF header from untrusted bytes, in safe Rust, through a [`HeaderReader`]
//! over any [`RangeSource`], so the same code reads a local file, a sparse blob or a Hub file.
//!
//! - **Versions.** GGUF v2 and v3, little-endian. A v1 file, a big-endian file (its version
//!   field reads byte-swapped) and any other version fail with `UnsupportedContainer`
//!   (RFC-0041 §10 code 2). Input that does not start with the magic `GGUF` is
//!   `HeaderInvalid` (6).
//! - **Limits** ([`Limits`]): at most 2²⁰ tensors, 2¹⁶ key-value pairs, 64 MiB per string,
//!   2²⁴ elements per array, and a header of at most `store.header_max_bytes`. Each is checked
//!   before the bytes it covers are read or allocated. Any excess is `HeaderLimit` (5), class
//!   `Integrity`.
//! - **Reading.** Every read goes through the one header-read routine, which starts with the
//!   first 4 MiB and extends by range reads (RFC-0041 §7.2). The parser never asks for a byte
//!   past the end of the tensor-info table: the data section is never parsed.
//! - **Alignment.** `general.alignment` must be a `u32` power of two; the default is 32. The
//!   data section starts at the aligned end of the tensor-info table.
//! - **Tensors.** At most 4 dimensions; every `ne` fits `i64`; the first dimension is a whole
//!   number of blocks; the element count fits `i64`. Tensors are laid out as ggml lays them
//!   and requires them: in table order, the first at offset 0 and each next one at the aligned
//!   end of the one before, so none overlap and each is aligned. Every tensor's bytes lie
//!   inside the file. Any breach is `HeaderInvalid` (6).
//! - **Types.** Each type ID maps to `QuantScheme::Ggml` through [`ggml_type_from_id`]; any
//!   other ID is `UnknownQuant` (4) naming it.
//! - **Metadata.** Key-value pairs in file order, with typed accessors ([`MetaView`]). Keys
//!   and tensor names are UTF-8 and unique; arrays are not nested; a bool is 0 or 1.
//! - **Split files** ([`GgufModel`], [`SplitName`]): every split's header is read, `split.no`
//!   and `split.count` must match the set, `split.tensors.count` the total, and names are
//!   unique across splits. Each tensor carries its file index.
//!
//! The reader is stricter than ggml's in three places: keys and tensor names must be UTF-8, a
//! bool must be 0 or 1, and every tensor's bytes must be inside the file. It accepts tensor
//! names longer than ggml's 63-byte limit; the ggml provider refuses such a file when it loads
//! it. On every file both readers accept, they agree on each tensor's name, type, shape and
//! offset (the differential corpus in the tests).

use std::collections::BTreeSet;
use std::fmt;

use syntropy_core::{ErrorClass, GgmlType, QuantScheme};

use crate::range::{HEADER_MAX_BYTES_DEFAULT, HeaderReader, RangeError, RangeSource};

mod meta;
mod split;
mod types;

#[cfg(test)]
pub(crate) mod synth;
#[cfg(test)]
mod tests;

pub use meta::{Array, MetaError, MetaView, StrArray, Value, ValueType};
pub use split::{GgufModel, SPLIT_COUNT, SPLIT_NO, SPLIT_TENSORS_COUNT, SplitName};
pub use types::{block_geometry, ggml_type_from_id, ggml_type_id, ggml_type_name};

/// The first four bytes of every GGUF file.
pub const MAGIC: [u8; 4] = *b"GGUF";
/// The alignment when `general.alignment` is absent.
pub const DEFAULT_ALIGNMENT: u64 = 32;
/// The key that sets the alignment.
pub const ALIGNMENT_KEY: &str = "general.alignment";
/// The most dimensions a tensor may have (ggml's `GGML_MAX_DIMS`).
pub const MAX_DIMS: u32 = 4;

/// The parser limits of RFC-0041 §2.1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    /// The most tensors in one file: 2²⁰.
    pub tensors: u64,
    /// The most key-value pairs in one file: 2¹⁶.
    pub kv_pairs: u64,
    /// The most bytes in one string (keys and names included): 64 MiB.
    pub string_bytes: u64,
    /// The most elements in one array: 2²⁴.
    pub array_len: u64,
    /// `store.header_max_bytes`: the most bytes a header may span, 1 GiB by default.
    pub header_bytes: u64,
}

impl Limits {
    /// RFC-0041 §2.1's limits, with `store.header_max_bytes` at its default.
    pub const DEFAULT: Self = Self {
        tensors: 1 << 20,
        kv_pairs: 1 << 16,
        string_bytes: 64 << 20,
        array_len: 1 << 24,
        header_bytes: HEADER_MAX_BYTES_DEFAULT,
    };
}

impl Default for Limits {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Which limit a header exceeds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Limit {
    /// [`Limits::tensors`].
    Tensors,
    /// [`Limits::kv_pairs`].
    KvPairs,
    /// [`Limits::string_bytes`].
    StringBytes,
    /// [`Limits::array_len`].
    ArrayLen,
    /// [`Limits::header_bytes`]: `store.header_max_bytes`.
    HeaderBytes,
}

/// Why a container is unsupported (RFC-0041 §10 code 2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unsupported {
    /// GGUF v1.
    Version1,
    /// A big-endian GGUF: the version field reads byte-swapped.
    BigEndian {
        /// The version, read big-endian.
        version: u32,
    },
    /// A version other than 1, 2 or 3.
    UnknownVersion(u32),
}

/// A failed GGUF read, with RFC-0041 §10's code where it names one.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GgufError {
    /// `UnsupportedContainer`, `(Formats, 2)`, class `Unsupported`.
    UnsupportedContainer(Unsupported),
    /// `UnknownQuant`, `(Formats, 4)`, class `Unsupported`: a tensor's ggml type ID is not one
    /// Syntropy reads.
    UnknownQuant {
        /// The tensor.
        tensor: String,
        /// Its type ID.
        type_id: u32,
    },
    /// `HeaderLimit`, `(Formats, 5)`, class `Integrity`: a parser limit is exceeded.
    HeaderLimit {
        /// The limit.
        limit: Limit,
        /// The header's value.
        value: u64,
        /// The limit's value.
        max: u64,
    },
    /// `HeaderInvalid`, `(Formats, 6)`, class `Integrity`: the header is malformed.
    HeaderInvalid {
        /// The byte offset in the file where the reader found it.
        offset: u64,
        /// What is wrong.
        reason: String,
    },
    /// `HeaderInvalid`, `(Formats, 6)`, class `Integrity`: the files of a split set do not
    /// agree.
    SplitInvalid {
        /// What is wrong.
        reason: String,
    },
    /// An error in one file of a split set.
    InFile {
        /// The file's index in the set.
        file: u16,
        /// The error.
        error: Box<GgufError>,
    },
    /// The source failed a read for a reason with no `Formats` code of its own.
    Read(RangeError),
}

impl GgufError {
    /// The `Formats` code RFC-0041 §10 gives it, if any.
    #[must_use]
    pub fn code(&self) -> Option<u16> {
        match self {
            Self::UnsupportedContainer(_) => Some(2),
            Self::UnknownQuant { .. } => Some(4),
            Self::HeaderLimit { .. } => Some(5),
            Self::HeaderInvalid { .. } | Self::SplitInvalid { .. } => Some(6),
            Self::InFile { error, .. } => error.code(),
            Self::Read(e) => e.code(),
        }
    }

    /// Its class (RFC-0001 §10).
    #[must_use]
    pub fn class(&self) -> ErrorClass {
        match self {
            Self::UnsupportedContainer(_) | Self::UnknownQuant { .. } => ErrorClass::Unsupported,
            Self::HeaderLimit { .. } | Self::HeaderInvalid { .. } | Self::SplitInvalid { .. } => {
                ErrorClass::Integrity
            }
            Self::InFile { error, .. } => error.class(),
            Self::Read(e) => e.class(),
        }
    }

    fn invalid(offset: u64, reason: impl Into<String>) -> Self {
        Self::HeaderInvalid {
            offset,
            reason: reason.into(),
        }
    }

    fn from_range(e: RangeError) -> Self {
        match e {
            RangeError::HeaderLimit { end, cap } => Self::HeaderLimit {
                limit: Limit::HeaderBytes,
                value: end,
                max: cap,
            },
            RangeError::OutOfRange {
                offset,
                len,
                source_len,
            } => Self::invalid(
                offset,
                format!("truncated: {len} bytes at {offset} pass the file's end at {source_len}"),
            ),
            other => Self::Read(other),
        }
    }
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedContainer(why) => {
                write!(f, "UnsupportedContainer (Formats 2): ")?;
                match why {
                    Unsupported::Version1 => write!(f, "GGUF v1 is not supported; v2 and v3 are"),
                    Unsupported::BigEndian { version } => {
                        write!(f, "big-endian GGUF (v{version}) is not supported")
                    }
                    Unsupported::UnknownVersion(v) => {
                        write!(f, "GGUF version {v} is not supported; v2 and v3 are")
                    }
                }
            }
            Self::UnknownQuant { tensor, type_id } => match ggml_type_name(*type_id) {
                Some(name) => write!(
                    f,
                    "UnknownQuant (Formats 4): tensor `{tensor}` has ggml type {name} (ID {type_id}), which Syntropy does not read as a weight"
                ),
                None => write!(
                    f,
                    "UnknownQuant (Formats 4): tensor `{tensor}` has unknown ggml type ID {type_id}"
                ),
            },
            Self::HeaderLimit { limit, value, max } => write!(
                f,
                "HeaderLimit (Formats 5): {limit:?} is {value}, past the limit of {max}"
            ),
            Self::HeaderInvalid { offset, reason } => {
                write!(f, "HeaderInvalid (Formats 6) at byte {offset}: {reason}")
            }
            Self::SplitInvalid { reason } => {
                write!(f, "HeaderInvalid (Formats 6), split set: {reason}")
            }
            Self::InFile { file, error } => write!(f, "split file {file}: {error}"),
            Self::Read(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for GgufError {}

/// One tensor's entry in the tensor-info table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GgufTensor {
    /// Its name.
    pub name: String,
    /// Its dimensions in ggml's order: `ne[0]` is the innermost (contiguous) one.
    pub ne: Vec<u64>,
    /// Its ggml type.
    pub ggml_type: GgmlType,
    /// Its offset from the start of the data section, as stored.
    pub offset: u64,
    /// Its offset from the start of its file: the data section's start plus
    /// [`offset`](Self::offset).
    pub start: u64,
    /// Its byte length.
    pub bytes: u64,
    /// The index of its file in a split set; 0 for a single file.
    pub file: u16,
}

impl GgufTensor {
    /// Its quantization scheme (RFC-0041 §2.1 "Types").
    #[must_use]
    pub fn scheme(&self) -> QuantScheme {
        QuantScheme::Ggml(self.ggml_type)
    }

    /// Its element count.
    #[must_use]
    pub fn elements(&self) -> u64 {
        self.ne.iter().product()
    }
}

/// A parsed GGUF header.
#[derive(Clone, Debug, PartialEq)]
pub struct GgufHeader {
    /// 2 or 3.
    pub version: u32,
    /// The alignment in force.
    pub alignment: u64,
    /// The key-value metadata.
    pub metadata: MetaView,
    /// The tensor-info table, in file order.
    pub tensors: Vec<GgufTensor>,
    /// The end of the tensor-info table: the header's length in bytes.
    pub header_bytes: u64,
    /// The start of the data section: `header_bytes` rounded up to the alignment.
    pub data_start: u64,
    /// The file's length.
    pub file_len: u64,
}

impl GgufHeader {
    /// Reads the header of `source`, with `store.header_max_bytes` taken from `limits`.
    ///
    /// # Errors
    ///
    /// A [`GgufError`] with RFC-0041 §10's code.
    pub fn read<S: RangeSource>(source: S, limits: &Limits) -> Result<Self, GgufError> {
        Self::read_from(&mut HeaderReader::new(source, limits.header_bytes), limits)
    }

    /// Reads the header through `reader`. The tighter of the reader's cap and
    /// `limits.header_bytes` bounds the header.
    ///
    /// # Errors
    ///
    /// A [`GgufError`] with RFC-0041 §10's code.
    pub fn read_from<S: RangeSource>(
        reader: &mut HeaderReader<S>,
        limits: &Limits,
    ) -> Result<Self, GgufError> {
        Parser {
            reader,
            at: 0,
            limits,
        }
        .header()
    }
}

/// Rounds `n` up to a multiple of `align`, a power of two.
fn pad(n: u64, align: u64) -> Option<u64> {
    n.checked_add(align - 1).map(|x| x & !(align - 1))
}

struct Parser<'a, S> {
    reader: &'a mut HeaderReader<S>,
    at: u64,
    limits: &'a Limits,
}

macro_rules! read_le {
    ($name:ident, $t:ty) => {
        fn $name(&mut self) -> Result<$t, GgufError> {
            let b = self.bytes(std::mem::size_of::<$t>() as u64)?;
            let mut a = [0u8; std::mem::size_of::<$t>()];
            a.copy_from_slice(b);
            Ok(<$t>::from_le_bytes(a))
        }
    };
}

impl<S: RangeSource> Parser<'_, S> {
    fn bytes(&mut self, n: u64) -> Result<&[u8], GgufError> {
        let at = self.at;
        let end = at.checked_add(n).ok_or(GgufError::HeaderLimit {
            limit: Limit::HeaderBytes,
            value: u64::MAX,
            max: self.limits.header_bytes,
        })?;
        if end > self.limits.header_bytes {
            return Err(GgufError::HeaderLimit {
                limit: Limit::HeaderBytes,
                value: end,
                max: self.limits.header_bytes,
            });
        }
        self.at = end;
        self.reader.read(at, n).map_err(GgufError::from_range)
    }

    read_le!(u8, u8);
    read_le!(i8, i8);
    read_le!(u16, u16);
    read_le!(i16, i16);
    read_le!(u32, u32);
    read_le!(i32, i32);
    read_le!(u64, u64);
    read_le!(i64, i64);
    read_le!(f32, f32);
    read_le!(f64, f64);

    fn bool(&mut self) -> Result<bool, GgufError> {
        let at = self.at;
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            b => Err(GgufError::invalid(
                at,
                format!("bool holds {b}, not 0 or 1"),
            )),
        }
    }

    fn count(&mut self, limit: Limit, max: u64) -> Result<u64, GgufError> {
        let n = self.u64()?;
        if n > max {
            return Err(GgufError::HeaderLimit {
                limit,
                value: n,
                max,
            });
        }
        Ok(n)
    }

    fn string(&mut self) -> Result<Box<[u8]>, GgufError> {
        let n = self.count(Limit::StringBytes, self.limits.string_bytes)?;
        Ok(self.bytes(n)?.into())
    }

    fn utf8(&mut self, what: &str) -> Result<String, GgufError> {
        let at = self.at;
        let s = self.string()?;
        String::from_utf8(s.into_vec())
            .map_err(|_| GgufError::invalid(at, format!("{what} is not UTF-8")))
    }

    fn value_type(&mut self) -> Result<ValueType, GgufError> {
        let at = self.at;
        let id = self.u32()?;
        ValueType::from_id(id)
            .ok_or_else(|| GgufError::invalid(at, format!("unknown value type {id}")))
    }

    fn value(&mut self, ty: ValueType) -> Result<Value, GgufError> {
        Ok(match ty {
            ValueType::U8 => Value::U8(self.u8()?),
            ValueType::I8 => Value::I8(self.i8()?),
            ValueType::U16 => Value::U16(self.u16()?),
            ValueType::I16 => Value::I16(self.i16()?),
            ValueType::U32 => Value::U32(self.u32()?),
            ValueType::I32 => Value::I32(self.i32()?),
            ValueType::F32 => Value::F32(self.f32()?),
            ValueType::Bool => Value::Bool(self.bool()?),
            ValueType::Str => Value::Str(self.string()?),
            ValueType::Array => Value::Array(self.array()?),
            ValueType::U64 => Value::U64(self.u64()?),
            ValueType::I64 => Value::I64(self.i64()?),
            ValueType::F64 => Value::F64(self.f64()?),
        })
    }

    /// Reads `n` fixed-size elements in one read and decodes each with `f`.
    fn fixed<T, const N: usize>(
        &mut self,
        n: u64,
        f: impl Fn([u8; N]) -> T,
    ) -> Result<Vec<T>, GgufError> {
        // n ≤ the array limit (2²⁴ by default) and N ≤ 8, so this cannot overflow; the read
        // is bounded by the header cap before anything is allocated.
        let total = n.saturating_mul(N as u64);
        let b = self.bytes(total)?;
        Ok(b.as_chunks::<N>().0.iter().map(|c| f(*c)).collect())
    }

    fn array(&mut self) -> Result<Array, GgufError> {
        let at = self.at;
        let ty = self.value_type()?;
        let n = self.count(Limit::ArrayLen, self.limits.array_len)?;
        Ok(match ty {
            ValueType::U8 => Array::U8(self.fixed(n, u8::from_le_bytes)?),
            ValueType::I8 => Array::I8(self.fixed(n, i8::from_le_bytes)?),
            ValueType::U16 => Array::U16(self.fixed(n, u16::from_le_bytes)?),
            ValueType::I16 => Array::I16(self.fixed(n, i16::from_le_bytes)?),
            ValueType::U32 => Array::U32(self.fixed(n, u32::from_le_bytes)?),
            ValueType::I32 => Array::I32(self.fixed(n, i32::from_le_bytes)?),
            ValueType::F32 => Array::F32(self.fixed(n, f32::from_le_bytes)?),
            ValueType::U64 => Array::U64(self.fixed(n, u64::from_le_bytes)?),
            ValueType::I64 => Array::I64(self.fixed(n, i64::from_le_bytes)?),
            ValueType::F64 => Array::F64(self.fixed(n, f64::from_le_bytes)?),
            ValueType::Bool => {
                let start = self.at;
                let raw = self.fixed(n, |[b]: [u8; 1]| b)?;
                let mut out = Vec::with_capacity(raw.len());
                for (i, b) in raw.into_iter().enumerate() {
                    match b {
                        0 => out.push(false),
                        1 => out.push(true),
                        _ => {
                            return Err(GgufError::invalid(
                                start + i as u64,
                                format!("bool holds {b}, not 0 or 1"),
                            ));
                        }
                    }
                }
                Array::Bool(out)
            }
            ValueType::Str => Array::Str(self.str_array(n)?),
            ValueType::Array => return Err(GgufError::invalid(at, "nested arrays")),
        })
    }

    /// Reads `n` strings into one packed [`StrArray`]. A first pass reads only the lengths,
    /// checking each against the string limit and the source, so the second pass allocates
    /// exactly: at most the array's own bytes in the file, whatever `n` claims.
    fn str_array(&mut self, n: u64) -> Result<StrArray, GgufError> {
        let start = self.at;
        let mut total = 0u64;
        for _ in 0..n {
            let len = self.count(Limit::StringBytes, self.limits.string_bytes)?;
            self.bytes(len)?;
            total += len;
        }
        // Every element was read, so `n` and `total` fit in memory already.
        let mut out = StrArray::with_capacity(
            usize::try_from(n).unwrap_or(usize::MAX),
            usize::try_from(total).unwrap_or(usize::MAX),
        );
        self.at = start;
        for _ in 0..n {
            let len = self.u64()?;
            out.push(self.bytes(len)?);
        }
        Ok(out)
    }

    fn version(&mut self) -> Result<u32, GgufError> {
        let magic = self.bytes(4)?;
        if magic != MAGIC {
            return Err(GgufError::invalid(0, "not a GGUF file: bad magic"));
        }
        let v = self.u32()?;
        match v {
            2 | 3 => Ok(v),
            1 => Err(GgufError::UnsupportedContainer(Unsupported::Version1)),
            _ if v & 0xffff == 0 && (1..=3).contains(&v.swap_bytes()) => {
                Err(GgufError::UnsupportedContainer(Unsupported::BigEndian {
                    version: v.swap_bytes(),
                }))
            }
            _ => Err(GgufError::UnsupportedContainer(
                Unsupported::UnknownVersion(v),
            )),
        }
    }

    fn header(mut self) -> Result<GgufHeader, GgufError> {
        let version = self.version()?;
        let n_tensors = self.count(Limit::Tensors, self.limits.tensors)?;
        let n_kv = self.count(Limit::KvPairs, self.limits.kv_pairs)?;

        let mut metadata = MetaView::new();
        for _ in 0..n_kv {
            let at = self.at;
            let key = self.utf8("a metadata key")?;
            let ty = self.value_type()?;
            let value = self.value(ty)?;
            if metadata.contains(&key) {
                return Err(GgufError::invalid(at, format!("duplicate key `{key}`")));
            }
            metadata.insert(key, value);
        }

        let alignment = match metadata.get(ALIGNMENT_KEY) {
            None => DEFAULT_ALIGNMENT,
            Some(Value::U32(a)) if a.is_power_of_two() => u64::from(*a),
            Some(v) => {
                return Err(GgufError::invalid(
                    self.at,
                    format!("`{ALIGNMENT_KEY}` must be a u32 power of two, found {v:?}"),
                ));
            }
        };

        let mut tensors = Vec::new();
        let mut names = BTreeSet::new();
        let mut expected = 0u64;
        for _ in 0..n_tensors {
            let at = self.at;
            let t = self.tensor_info(alignment)?;
            if !names.insert(t.name.clone()) {
                return Err(GgufError::invalid(
                    at,
                    format!("duplicate tensor `{}`", t.name),
                ));
            }
            if t.offset != expected {
                return Err(GgufError::invalid(
                    at,
                    format!(
                        "tensor `{}` is at data offset {}, expected {expected}: tensors must follow each other in table order, each at the aligned end of the one before",
                        t.name, t.offset
                    ),
                ));
            }
            expected = t
                .offset
                .checked_add(t.bytes)
                .and_then(|end| pad(end, alignment))
                .ok_or_else(|| GgufError::invalid(at, "tensor data passes 2⁶⁴ bytes"))?;
            tensors.push(t);
        }

        let header_bytes = self.at;
        let data_start = pad(header_bytes, alignment)
            .ok_or_else(|| GgufError::invalid(header_bytes, "data section passes 2⁶⁴ bytes"))?;
        let file_len = self.reader.source().len();
        for t in &mut tensors {
            let end = data_start
                .checked_add(t.offset)
                .and_then(|s| s.checked_add(t.bytes).map(|e| (s, e)));
            match end {
                Some((start, end)) if end <= file_len => t.start = start,
                _ => {
                    return Err(GgufError::invalid(
                        header_bytes,
                        format!(
                            "tensor `{}` ({} bytes at data offset {}) passes the file's end at {file_len}",
                            t.name, t.bytes, t.offset
                        ),
                    ));
                }
            }
        }

        Ok(GgufHeader {
            version,
            alignment,
            metadata,
            tensors,
            header_bytes,
            data_start,
            file_len,
        })
    }

    fn tensor_info(&mut self, alignment: u64) -> Result<GgufTensor, GgufError> {
        let at = self.at;
        let name = self.utf8("a tensor name")?;
        let n_dims = self.u32()?;
        if n_dims > MAX_DIMS {
            return Err(GgufError::invalid(
                at,
                format!("tensor `{name}` has {n_dims} dimensions, more than {MAX_DIMS}"),
            ));
        }
        let mut ne = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            let d = self.u64()?;
            if d > i64::MAX as u64 {
                return Err(GgufError::invalid(
                    at,
                    format!("tensor `{name}` has a negative dimension"),
                ));
            }
            ne.push(d);
        }
        let type_id = self.u32()?;
        let offset = self.u64()?;
        let Some(ggml_type) = ggml_type_from_id(type_id) else {
            return Err(GgufError::UnknownQuant {
                tensor: name,
                type_id,
            });
        };
        let Some((block, block_bytes)) = block_geometry(ggml_type) else {
            return Err(GgufError::UnknownQuant {
                tensor: name,
                type_id,
            });
        };
        let ne0 = ne.first().copied().unwrap_or(1);
        if ne0 % block != 0 {
            return Err(GgufError::invalid(
                at,
                format!(
                    "tensor `{name}`: its first dimension {ne0} is not a whole number of {ggml_type:?} blocks of {block}"
                ),
            ));
        }
        let elements = ne
            .iter()
            .try_fold(1u64, |acc, &d| acc.checked_mul(d))
            .filter(|&n| n <= i64::MAX as u64)
            .ok_or_else(|| {
                GgufError::invalid(at, format!("tensor `{name}`'s element count passes 2⁶³"))
            })?;
        let bytes = (elements / block)
            .checked_mul(block_bytes)
            .ok_or_else(|| GgufError::invalid(at, format!("tensor `{name}`'s size passes 2⁶⁴")))?;
        if offset % alignment != 0 {
            return Err(GgufError::invalid(
                at,
                format!(
                    "tensor `{name}` is at data offset {offset}, not a multiple of the alignment {alignment}"
                ),
            ));
        }
        Ok(GgufTensor {
            name,
            ne,
            ggml_type,
            offset,
            start: 0,
            bytes,
            file: 0,
        })
    }
}
