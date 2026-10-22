// SPDX-License-Identifier: MIT OR Apache-2.0

//! The safetensors reader (RFC-0041 §2.2; ADR-0005 decision 1; COMP-E10-T04).
//!
//! - [`read_header`] reads one file's container through a [`RangeSource`]: the 8-byte
//!   little-endian header length (at most [`HEADER_MAX`], 100 MiB), then the JSON header, by
//!   the shared [`HeaderReader`]: two requests, the first 8 bytes and then the header
//!   (RFC-0041 §7.2). No byte past the header is read.
//! - It checks every tensor: its `dtype` is one this reader supports, its `shape` is a list of
//!   non-negative integers, its `data_offsets` lie inside the file's data section, its length
//!   is `numel × element size`, and no two tensors share a byte. Any failure is
//!   [`SafetensorsError::HeaderInvalid`] (`(Formats, 6)`), except an unknown or unsupported
//!   dtype, which is [`SafetensorsError::UnsupportedContainer`] (`(Formats, 2)`), and a header
//!   past the limit, which is [`SafetensorsError::HeaderLimit`] (`(Formats, 5)`).
//! - [`index`] reads the sharded index (`model.safetensors.index.json`) and every shard's
//!   header, and checks that the index and the shards agree.
//! - [`side`] names the side files (`config.json`, `generation_config.json`, the tokenizer
//!   files) and gives each its source-file role.
//!
//! This task supports `F32`, `F16` and `BF16`. The other dtypes of RFC-0041 §2.2 are
//! COMP-E10-T10; until then a tensor in one of them is `UnsupportedContainer` naming the dtype.
//! The JSON is parsed by `serde_json` into a strict value that refuses repeated keys
//! (the private `json` module). Everything here is safe Rust (ADR-0001 decision 4).

use std::fmt;

use syntropy_core::{DType, ErrorClass};

use crate::range::{HEADER_MAX_BYTES_DEFAULT, HeaderReader, RangeError, RangeSource};

pub mod index;
mod json;
pub mod side;

#[cfg(test)]
mod tests;

use json::Json;

/// The largest header length the reader accepts: 100 MiB (RFC-0041 §2.2).
pub const HEADER_MAX: u64 = 100 << 20;

/// The key of the free-form string map in the header.
pub const METADATA_KEY: &str = "__metadata__";

/// A failure to read a safetensors container or its index (RFC-0041 §10).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SafetensorsError {
    /// `UnsupportedContainer`, `(Formats, 2)`, class `Unsupported`: a dtype this reader does
    /// not read.
    UnsupportedContainer {
        /// The tensor.
        tensor: String,
        /// The dtype as the header spells it.
        dtype: String,
    },
    /// `HeaderLimit`, `(Formats, 5)`, class `Integrity`: the header length passes
    /// [`HEADER_MAX`] or `store.header_max_bytes`.
    HeaderLimit {
        /// The declared header length, or the file length for an index.
        len: u64,
        /// The limit it passes.
        cap: u64,
    },
    /// `HeaderInvalid`, `(Formats, 6)`, class `Integrity`: the container is malformed, an
    /// offset is out of range, two tensors overlap, or a length does not match its shape.
    HeaderInvalid {
        /// What is wrong.
        reason: String,
    },
    /// A read failed.
    Range(RangeError),
}

impl SafetensorsError {
    /// The `Formats` code RFC-0041 §10 gives it, if any.
    #[must_use]
    pub fn code(&self) -> Option<u16> {
        match self {
            Self::UnsupportedContainer { .. } => Some(2),
            Self::HeaderLimit { .. } => Some(5),
            Self::HeaderInvalid { .. } => Some(6),
            Self::Range(e) => e.code(),
        }
    }

    /// Its class (RFC-0001 §10).
    #[must_use]
    pub fn class(&self) -> ErrorClass {
        match self {
            Self::UnsupportedContainer { .. } => ErrorClass::Unsupported,
            Self::HeaderLimit { .. } | Self::HeaderInvalid { .. } => ErrorClass::Integrity,
            Self::Range(e) => e.class(),
        }
    }

    pub(crate) fn invalid(reason: impl Into<String>) -> Self {
        Self::HeaderInvalid {
            reason: reason.into(),
        }
    }
}

impl From<RangeError> for SafetensorsError {
    fn from(e: RangeError) -> Self {
        match e {
            // An out-of-range read of a header is the header's fault: it is HeaderInvalid.
            RangeError::OutOfRange { .. } => Self::invalid(e.to_string()),
            e => Self::Range(e),
        }
    }
}

impl fmt::Display for SafetensorsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedContainer { tensor, dtype } => write!(
                f,
                "UnsupportedContainer (Formats 2): tensor {tensor:?} has dtype {dtype:?}, which \
                 this reader does not read (it reads F32, F16 and BF16)"
            ),
            Self::HeaderLimit { len, cap } => write!(
                f,
                "HeaderLimit (Formats 5): a header of {len} bytes passes the limit of {cap} bytes"
            ),
            Self::HeaderInvalid { reason } => write!(f, "HeaderInvalid (Formats 6): {reason}"),
            Self::Range(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for SafetensorsError {}

/// The dtype a header names, for the dtypes this reader supports.
#[must_use]
pub fn dtype_of(name: &str) -> Option<DType> {
    match name {
        "F32" => Some(DType::F32),
        "F16" => Some(DType::F16),
        "BF16" => Some(DType::BF16),
        _ => None,
    }
}

/// The bytes one element of `dtype` takes, for the whole-byte dtypes this reader supports.
fn element_bytes(dtype: DType) -> u64 {
    u64::from(dtype.bits() / 8)
}

/// One tensor of a safetensors file, validated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SafetensorsTensor {
    /// The name as the header spells it.
    pub name: String,
    /// The element type.
    pub dtype: DType,
    /// The shape, outermost dimension first; empty for a scalar.
    pub shape: Vec<u64>,
    /// The first byte, as an offset into the file (the header's `data_offsets[0]` plus the data
    /// section's start).
    pub offset: u64,
    /// The length in bytes: `numel × element size`.
    pub len: u64,
}

/// One safetensors file's header, validated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SafetensorsHeader {
    /// The file's length.
    pub file_len: u64,
    /// Where the data section starts: 8 plus the header length.
    pub data_start: u64,
    /// `__metadata__`, in header order; empty when absent.
    pub metadata: Vec<(String, String)>,
    /// The tensors, in header order.
    pub tensors: Vec<SafetensorsTensor>,
}

impl SafetensorsHeader {
    /// The tensor named `name`.
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&SafetensorsTensor> {
        self.tensors.iter().find(|t| t.name == name)
    }
}

/// Reads and validates one safetensors file's header from `source`, with `header_max_bytes`
/// the `store.header_max_bytes` in force ([`HEADER_MAX_BYTES_DEFAULT`] by default).
///
/// # Errors
///
/// [`SafetensorsError`], as the module documentation lists.
pub fn read_header<S: RangeSource>(
    source: S,
    header_max_bytes: u64,
) -> Result<SafetensorsHeader, SafetensorsError> {
    let file_len = source.len();
    let mut reader = HeaderReader::new(source, header_max_bytes).with_initial_read(8);
    if file_len < 8 {
        return Err(SafetensorsError::invalid(format!(
            "the file is {file_len} bytes, shorter than the 8-byte header length"
        )));
    }
    let mut word = [0u8; 8];
    reader.read_into(0, &mut word)?;
    let n = u64::from_le_bytes(word);
    let cap = HEADER_MAX.min(header_max_bytes.saturating_sub(8));
    if n > cap {
        return Err(SafetensorsError::HeaderLimit { len: n, cap });
    }
    let data_start = 8 + n;
    if data_start > file_len {
        return Err(SafetensorsError::invalid(format!(
            "the header of {n} bytes passes the file's end at {file_len}"
        )));
    }
    let bytes = reader.read(8, n)?;
    parse_header(bytes, file_len)
}

/// [`read_header`] with the default `store.header_max_bytes`.
///
/// # Errors
///
/// As [`read_header`].
pub fn read_header_default<S: RangeSource>(
    source: S,
) -> Result<SafetensorsHeader, SafetensorsError> {
    read_header(source, HEADER_MAX_BYTES_DEFAULT)
}

/// Validates the JSON header `bytes` of a file of `file_len` bytes whose data section starts
/// right after it.
///
/// # Errors
///
/// As [`read_header`].
pub fn parse_header(bytes: &[u8], file_len: u64) -> Result<SafetensorsHeader, SafetensorsError> {
    let n = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    let data_start = n
        .checked_add(8)
        .filter(|s| *s <= file_len)
        .ok_or_else(|| SafetensorsError::invalid("the header passes the file's end"))?;
    let data_len = file_len - data_start;
    if bytes.first() != Some(&b'{') {
        return Err(SafetensorsError::invalid(
            "the header does not start with '{'",
        ));
    }
    let json = Json::parse(bytes)
        .map_err(|e| SafetensorsError::invalid(format!("the header is not valid JSON: {e}")))?;
    let Json::Object(entries) = json else {
        return Err(SafetensorsError::invalid("the header is not a JSON object"));
    };
    let mut metadata = Vec::new();
    let mut tensors = Vec::with_capacity(entries.len());
    // (start, end, index) in the data section, for the overlap check.
    let mut spans: Vec<(u64, u64, usize)> = Vec::with_capacity(entries.len());
    for (name, value) in entries {
        if name == METADATA_KEY {
            metadata = parse_metadata(value)?;
            continue;
        }
        let (dtype, shape, begin, end) = parse_entry(&name, value)?;
        if begin > end || end > data_len {
            return Err(SafetensorsError::invalid(format!(
                "tensor {name:?}: data_offsets [{begin}, {end}] lie outside the data section of \
                 {data_len} bytes"
            )));
        }
        let numel = shape
            .iter()
            .try_fold(1u64, |acc, d| acc.checked_mul(*d))
            .ok_or_else(|| {
                SafetensorsError::invalid(format!("tensor {name:?}: the element count overflows"))
            })?;
        let expect = numel.checked_mul(element_bytes(dtype)).ok_or_else(|| {
            SafetensorsError::invalid(format!("tensor {name:?}: the byte length overflows"))
        })?;
        if end - begin != expect {
            return Err(SafetensorsError::invalid(format!(
                "tensor {name:?}: data_offsets hold {} bytes, but shape {shape:?} of {dtype:?} \
                 needs {expect}",
                end - begin
            )));
        }
        spans.push((begin, end, tensors.len()));
        tensors.push(SafetensorsTensor {
            name,
            dtype,
            shape,
            offset: data_start + begin,
            len: expect,
        });
    }
    spans.sort_unstable();
    // Only tensors with bytes can overlap; an empty tensor holds no byte.
    let mut prev: Option<(u64, usize)> = None;
    for &(start, end, i) in spans.iter().filter(|(s, e, _)| s < e) {
        if let Some((prev_end, j)) = prev
            && start < prev_end
        {
            return Err(SafetensorsError::invalid(format!(
                "tensors {:?} and {:?} overlap",
                tensors[j].name, tensors[i].name
            )));
        }
        prev = Some((end, i));
    }
    Ok(SafetensorsHeader {
        file_len,
        data_start,
        metadata,
        tensors,
    })
}

fn parse_metadata(value: Json) -> Result<Vec<(String, String)>, SafetensorsError> {
    let Json::Object(entries) = value else {
        return Err(SafetensorsError::invalid(format!(
            "{METADATA_KEY} is {}, not an object",
            value.kind()
        )));
    };
    entries
        .into_iter()
        .map(|(k, v)| match v {
            Json::Str(s) => Ok((k, s)),
            other => Err(SafetensorsError::invalid(format!(
                "{METADATA_KEY}.{k} is {}, not a string",
                other.kind()
            ))),
        })
        .collect()
}

/// One tensor entry: its dtype, shape and the two data offsets.
fn parse_entry(name: &str, value: Json) -> Result<(DType, Vec<u64>, u64, u64), SafetensorsError> {
    let bad = |what: &str| SafetensorsError::invalid(format!("tensor {name:?}: {what}"));
    let Json::Object(fields) = value else {
        return Err(bad(&format!(
            "the entry is {}, not an object",
            value.kind()
        )));
    };
    let (mut dtype, mut shape, mut offsets) = (None, None, None);
    for (key, v) in fields {
        match key.as_str() {
            "dtype" => dtype = Some(v),
            "shape" => shape = Some(v),
            "data_offsets" => offsets = Some(v),
            other => return Err(bad(&format!("unknown field {other:?}"))),
        }
    }
    let dtype = match dtype.ok_or_else(|| bad("no dtype"))? {
        Json::Str(s) => dtype_of(&s).ok_or(SafetensorsError::UnsupportedContainer {
            tensor: name.to_owned(),
            dtype: s,
        })?,
        other => return Err(bad(&format!("dtype is {}, not a string", other.kind()))),
    };
    let shape = uints(shape.ok_or_else(|| bad("no shape"))?)
        .ok_or_else(|| bad("shape is not a list of non-negative integers"))?;
    let offsets = uints(offsets.ok_or_else(|| bad("no data_offsets"))?)
        .ok_or_else(|| bad("data_offsets is not a list of non-negative integers"))?;
    let [begin, end] = offsets[..] else {
        return Err(bad(&format!(
            "data_offsets has {} entries, not 2",
            offsets.len()
        )));
    };
    Ok((dtype, shape, begin, end))
}

/// The array of non-negative integers `value` holds, if it is one.
fn uints(value: Json) -> Option<Vec<u64>> {
    let Json::Array(items) = value else {
        return None;
    };
    items
        .into_iter()
        .map(|v| match v {
            Json::Uint(n) => Some(n),
            _ => None,
        })
        .collect()
}
