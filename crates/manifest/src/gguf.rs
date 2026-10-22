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
//! - **Memory.** The header keeps the metadata and tensor-info bytes as the file has them,
//!   copied once, with a 4-byte position per entry and a 4-byte sort index, and decodes on
//!   access ([`MetaView`], [`GgufHeader::tensors`]). Nothing is allocated per entry or per
//!   string, so a parse's peak heap use, with the header-read prefix, stays within 4× its input
//!   (`tests/gguf_alloc.rs` measures the worst shapes).
//! - **Split files** ([`GgufModel`], [`SplitName`]): every split's header is read, `split.no`
//!   and `split.count` must match the set, `split.tensors.count` the total, and names are
//!   unique across splits. Each tensor carries its file index.
//!
//! The reader is stricter than ggml's in three places: keys and tensor names must be UTF-8, a
//! bool must be 0 or 1, and every tensor's bytes must be inside the file. It accepts tensor
//! names longer than ggml's 63-byte limit; the ggml provider refuses such a file when it loads
//! it. On every file both readers accept, they agree on each tensor's name, type, shape and
//! offset (the differential corpus in the tests).

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

pub use meta::{Array, ArrayRef, MetaError, MetaView, StrArray, Value, ValueRef, ValueType};
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

/// One tensor's entry in the tensor-info table, decoded from the header bytes on access.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GgufTensor<'a> {
    /// Its name.
    pub name: &'a str,
    ne: [u64; MAX_DIMS as usize],
    n_dims: u8,
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

impl GgufTensor<'_> {
    /// Its dimensions in ggml's order: `ne[0]` is the innermost (contiguous) one.
    #[must_use]
    pub fn ne(&self) -> &[u64] {
        &self.ne[..usize::from(self.n_dims)]
    }

    /// Its quantization scheme (RFC-0041 §2.1 "Types").
    #[must_use]
    pub fn scheme(&self) -> QuantScheme {
        QuantScheme::Ggml(self.ggml_type)
    }

    /// Its element count.
    #[must_use]
    pub fn elements(&self) -> u64 {
        self.ne().iter().product()
    }
}

/// A tensor's byte length from its type and shape; `None` when it is not a whole number of
/// blocks or overflows.
fn tensor_bytes(t: GgmlType, ne: &[u64]) -> Option<(u64, u64)> {
    let (block, block_bytes) = block_geometry(t)?;
    let elements = ne
        .iter()
        .try_fold(1u64, |acc, &d| acc.checked_mul(d))
        .filter(|&n| n <= i64::MAX as u64)?;
    Some((elements, (elements / block).checked_mul(block_bytes)?))
}

/// A parsed GGUF header.
///
/// It keeps the tensor-info table's bytes as the file has them, validated, with one position
/// per tensor: 4 bytes per tensor beyond the input, against at least 25 input bytes per entry.
/// [`tensors`](Self::tensors) decodes each entry on access.
#[derive(Clone, Debug, PartialEq)]
pub struct GgufHeader {
    /// 2 or 3.
    pub version: u32,
    /// The alignment in force.
    pub alignment: u64,
    /// The key-value metadata.
    pub metadata: MetaView,
    /// The end of the tensor-info table: the header's length in bytes.
    pub header_bytes: u64,
    /// The start of the data section: `header_bytes` rounded up to the alignment.
    pub data_start: u64,
    /// The file's length.
    pub file_len: u64,
    /// The tensor-info table's bytes.
    info: Vec<u8>,
    /// Each entry's position in `info`, in file order.
    info_pos: Vec<u32>,
    /// The file's index in a split set.
    file: u16,
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

    /// The number of tensors.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.info_pos.len()
    }

    /// Tensor `i`, in file order.
    #[must_use]
    pub fn tensor(&self, i: usize) -> Option<GgufTensor<'_>> {
        let at = *self.info_pos.get(i)? as usize;
        Some(self.decode(at))
    }

    /// The tensors in file order.
    pub fn tensors(&self) -> impl ExactSizeIterator<Item = GgufTensor<'_>> {
        self.info_pos.iter().map(|&p| self.decode(p as usize))
    }

    /// The heap bytes it holds beyond the metadata: its buffers' capacities.
    #[must_use]
    pub fn heap_bytes(&self) -> usize {
        self.metadata.heap_bytes() + self.info.capacity() + 4 * self.info_pos.capacity()
    }

    pub(crate) fn set_file(&mut self, file: u16) {
        self.file = file;
    }

    fn name_at(&self, at: usize) -> &str {
        let len = rd_len(&self.info, at);
        let s = self
            .info
            .get(at.saturating_add(8)..at.saturating_add(8).saturating_add(len))
            .unwrap_or(&[]);
        std::str::from_utf8(s).unwrap_or("")
    }

    /// Decodes the entry at `at`, which the reader validated.
    fn decode(&self, at: usize) -> GgufTensor<'_> {
        let name = self.name_at(at);
        let mut p = at + 8 + name.len();
        let n_dims = rd_u32(&self.info, p).min(MAX_DIMS);
        p += 4;
        let mut ne = [1u64; MAX_DIMS as usize];
        for d in ne.iter_mut().take(n_dims as usize) {
            *d = rd_u64(&self.info, p);
            p += 8;
        }
        let ggml_type = ggml_type_from_id(rd_u32(&self.info, p)).unwrap_or(GgmlType::F32);
        let offset = rd_u64(&self.info, p + 4);
        let n_dims = n_dims as u8;
        let bytes = tensor_bytes(ggml_type, &ne[..usize::from(n_dims)]).map_or(0, |b| b.1);
        GgufTensor {
            name,
            ne,
            n_dims,
            ggml_type,
            offset,
            start: self.data_start.saturating_add(offset),
            bytes,
            file: self.file,
        }
    }
}

fn field<const N: usize>(buf: &[u8], at: usize) -> [u8; N] {
    buf.get(at..at.saturating_add(N))
        .and_then(|b| b.try_into().ok())
        .unwrap_or([0; N])
}

fn rd_u32(buf: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(field(buf, at))
}

fn rd_u64(buf: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(field(buf, at))
}

fn rd_len(buf: &[u8], at: usize) -> usize {
    usize::try_from(rd_u64(buf, at)).unwrap_or(usize::MAX)
}

/// Rounds `n` up to a multiple of `align`, a power of two.
fn pad(n: u64, align: u64) -> Option<u64> {
    n.checked_add(align - 1).map(|x| x & !(align - 1))
}

/// The fewest bytes a key-value pair takes: an empty key's length, a type and a one-byte value.
const MIN_KV_BYTES: u64 = 13;
/// The fewest bytes a tensor-info entry takes: an empty name's length, `n_dims` of 0, a type
/// and an offset.
const MIN_TENSOR_BYTES: u64 = 24;

struct Parser<'a, S> {
    reader: &'a mut HeaderReader<S>,
    at: u64,
    limits: &'a Limits,
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

    fn u8(&mut self) -> Result<u8, GgufError> {
        Ok(self.bytes(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, GgufError> {
        let b = self.bytes(4)?;
        Ok(rd_u32(b, 0))
    }

    fn u64(&mut self) -> Result<u64, GgufError> {
        let b = self.bytes(8)?;
        Ok(rd_u64(b, 0))
    }

    /// The bytes left before the header must end: the source's end or the cap.
    fn remaining(&self) -> u64 {
        self.reader
            .source()
            .len()
            .min(self.limits.header_bytes)
            .saturating_sub(self.at)
    }

    /// A capacity for `n` entries of at least `min_bytes` each: no more than the bytes left
    /// could hold, whatever `n` claims.
    fn capacity(&self, n: u64, min_bytes: u64) -> usize {
        usize::try_from(n.min(self.remaining() / min_bytes)).unwrap_or(0)
    }

    /// A position inside a region starting at `start`, as the `u32` the indexes hold.
    fn position(&self, start: u64) -> Result<u32, GgufError> {
        u32::try_from(self.at - start).map_err(|_| GgufError::HeaderLimit {
            limit: Limit::HeaderBytes,
            value: self.at,
            max: u64::from(u32::MAX),
        })
    }

    fn bool(&mut self) -> Result<(), GgufError> {
        let at = self.at;
        match self.u8()? {
            0 | 1 => Ok(()),
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

    /// Checks a string and steps over it; returns its length.
    fn string(&mut self) -> Result<u64, GgufError> {
        let n = self.count(Limit::StringBytes, self.limits.string_bytes)?;
        self.bytes(n)?;
        Ok(n)
    }

    /// Checks a UTF-8 string and steps over it.
    fn utf8(&mut self, what: &str) -> Result<(), GgufError> {
        let at = self.at;
        let n = self.count(Limit::StringBytes, self.limits.string_bytes)?;
        if std::str::from_utf8(self.bytes(n)?).is_err() {
            return Err(GgufError::invalid(at, format!("{what} is not UTF-8")));
        }
        Ok(())
    }

    fn value_type(&mut self) -> Result<ValueType, GgufError> {
        let at = self.at;
        let id = self.u32()?;
        ValueType::from_id(id)
            .ok_or_else(|| GgufError::invalid(at, format!("unknown value type {id}")))
    }

    /// Checks a value of type `ty` and steps over it. Nothing is allocated.
    fn value(&mut self, ty: ValueType) -> Result<(), GgufError> {
        match ty {
            ValueType::Bool => self.bool(),
            ValueType::Str => self.string().map(drop),
            ValueType::Array => self.array(),
            fixed => {
                self.bytes(fixed.fixed_size().unwrap_or(0))?;
                Ok(())
            }
        }
    }

    fn array(&mut self) -> Result<(), GgufError> {
        let at = self.at;
        let ty = self.value_type()?;
        let n = self.count(Limit::ArrayLen, self.limits.array_len)?;
        match ty {
            ValueType::Array => Err(GgufError::invalid(at, "nested arrays")),
            ValueType::Str => {
                for _ in 0..n {
                    self.string()?;
                }
                Ok(())
            }
            ValueType::Bool => {
                let start = self.at;
                let bad = self.bytes(n)?.iter().position(|&b| b > 1);
                match bad {
                    None => Ok(()),
                    Some(i) => Err(GgufError::invalid(
                        start + i as u64,
                        "bool holds a value other than 0 or 1",
                    )),
                }
            }
            fixed => {
                // n ≤ the array limit and the size ≤ 8; the read is bounded by the source
                // and the header cap.
                self.bytes(n.saturating_mul(fixed.fixed_size().unwrap_or(0)))?;
                Ok(())
            }
        }
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

    /// The bytes `[start, self.at)`, copied once into an exactly sized buffer.
    fn region(&mut self, start: u64) -> Result<Vec<u8>, GgufError> {
        let end = self.at;
        self.at = start;
        let out = self.bytes(end - start)?.to_vec();
        self.at = end;
        Ok(out)
    }

    fn header(mut self) -> Result<GgufHeader, GgufError> {
        let version = self.version()?;
        let n_tensors = self.count(Limit::Tensors, self.limits.tensors)?;
        let n_kv = self.count(Limit::KvPairs, self.limits.kv_pairs)?;

        // Pass over the key-value pairs, checking each and recording its position; then keep
        // their bytes, copied once.
        let kv_start = self.at;
        let mut kv_pos = Vec::with_capacity(self.capacity(n_kv, MIN_KV_BYTES));
        for _ in 0..n_kv {
            kv_pos.push(self.position(kv_start)?);
            self.utf8("a metadata key")?;
            let ty = self.value_type()?;
            self.value(ty)?;
        }
        let kv_buf = self.region(kv_start)?;
        let metadata = MetaView::from_parts(kv_buf, kv_pos).map_err(|pos| {
            GgufError::invalid(kv_start + u64::from(pos), "duplicate key".to_owned())
        })?;

        let alignment = match metadata.get(ALIGNMENT_KEY) {
            None => DEFAULT_ALIGNMENT,
            Some(ValueRef::U32(a)) if a.is_power_of_two() => u64::from(a),
            Some(v) => {
                return Err(GgufError::invalid(
                    self.at,
                    format!("`{ALIGNMENT_KEY}` must be a u32 power of two, found {v:?}"),
                ));
            }
        };

        let info_start = self.at;
        let mut info_pos = Vec::with_capacity(self.capacity(n_tensors, MIN_TENSOR_BYTES));
        let mut expected = 0u64;
        for _ in 0..n_tensors {
            let at = self.at;
            info_pos.push(self.position(info_start)?);
            let (offset, bytes) = self.tensor_info(alignment)?;
            if offset != expected {
                return Err(GgufError::invalid(
                    at,
                    format!(
                        "a tensor is at data offset {offset}, expected {expected}: tensors must follow each other in table order, each at the aligned end of the one before"
                    ),
                ));
            }
            expected = offset
                .checked_add(bytes)
                .and_then(|end| pad(end, alignment))
                .ok_or_else(|| GgufError::invalid(at, "tensor data passes 2⁶⁴ bytes"))?;
        }
        let header_bytes = self.at;
        let info = self.region(info_start)?;
        let data_start = pad(header_bytes, alignment)
            .ok_or_else(|| GgufError::invalid(header_bytes, "data section passes 2⁶⁴ bytes"))?;
        let file_len = self.reader.source().len();

        let header = GgufHeader {
            version,
            alignment,
            metadata,
            header_bytes,
            data_start,
            file_len,
            info,
            info_pos,
            file: 0,
        };

        // Names are unique: sort the entries by name and compare neighbours.
        let mut by_name: Vec<u32> =
            (0..u32::try_from(header.tensor_count()).unwrap_or(0)).collect();
        let name = |i: u32| header.name_at(header.info_pos[i as usize] as usize);
        by_name.sort_unstable_by(|&a, &b| name(a).cmp(name(b)).then(a.cmp(&b)));
        if let Some(w) = by_name.windows(2).find(|w| name(w[0]) == name(w[1])) {
            let pos = u64::from(header.info_pos[w[1] as usize]);
            return Err(GgufError::invalid(
                info_start + pos,
                format!("duplicate tensor `{}`", name(w[1])),
            ));
        }
        drop(by_name);

        // The tensors are contiguous, so the last one ends the data: every tensor's bytes are
        // inside the file when its bytes are.
        if let Some(last) = header.tensors().last() {
            let end = data_start
                .checked_add(last.offset)
                .and_then(|s| s.checked_add(last.bytes));
            if !end.is_some_and(|e| e <= file_len) {
                return Err(GgufError::invalid(
                    header_bytes,
                    format!(
                        "tensor `{}` ({} bytes at data offset {}) passes the file's end at {file_len}",
                        last.name, last.bytes, last.offset
                    ),
                ));
            }
        }
        Ok(header)
    }

    /// Checks one tensor-info entry and steps over it; returns its offset and byte length.
    fn tensor_info(&mut self, alignment: u64) -> Result<(u64, u64), GgufError> {
        let at = self.at;
        self.utf8("a tensor name")?;
        let name_end = self.at;
        let name = |p: &mut Self| -> String {
            let len = name_end - at - 8;
            p.reader
                .read(at + 8, len)
                .map(|b| String::from_utf8_lossy(b).into_owned())
                .unwrap_or_default()
        };
        let n_dims = self.u32()?;
        if n_dims > MAX_DIMS {
            let name = name(self);
            return Err(GgufError::invalid(
                at,
                format!("tensor `{name}` has {n_dims} dimensions, more than {MAX_DIMS}"),
            ));
        }
        let mut ne = [1u64; MAX_DIMS as usize];
        for d in ne.iter_mut().take(n_dims as usize) {
            *d = self.u64()?;
            if *d > i64::MAX as u64 {
                let name = name(self);
                return Err(GgufError::invalid(
                    at,
                    format!("tensor `{name}` has a negative dimension"),
                ));
            }
        }
        let ne = &ne[..n_dims as usize];
        let type_id = self.u32()?;
        let offset = self.u64()?;
        let Some(ggml_type) = ggml_type_from_id(type_id) else {
            return Err(GgufError::UnknownQuant {
                tensor: name(self),
                type_id,
            });
        };
        let Some((block, _)) = block_geometry(ggml_type) else {
            return Err(GgufError::UnknownQuant {
                tensor: name(self),
                type_id,
            });
        };
        let ne0 = ne.first().copied().unwrap_or(1);
        if ne0 % block != 0 {
            let name = name(self);
            return Err(GgufError::invalid(
                at,
                format!(
                    "tensor `{name}`: its first dimension {ne0} is not a whole number of {ggml_type:?} blocks of {block}"
                ),
            ));
        }
        let Some((_, bytes)) = tensor_bytes(ggml_type, ne) else {
            let name = name(self);
            return Err(GgufError::invalid(
                at,
                format!("tensor `{name}`'s element count or size overflows"),
            ));
        };
        if offset % alignment != 0 {
            let name = name(self);
            return Err(GgufError::invalid(
                at,
                format!(
                    "tensor `{name}` is at data offset {offset}, not a multiple of the alignment {alignment}"
                ),
            ));
        }
        Ok((offset, bytes))
    }
}
