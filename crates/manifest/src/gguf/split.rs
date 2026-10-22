// SPDX-License-Identifier: MIT OR Apache-2.0

//! Split GGUF files (RFC-0041 §2.1 "Split files").
//!
//! A model too large for one file is written as `<prefix>-NNNNN-of-MMMMM.gguf`, with `NNNNN`
//! counting from 1. Each split is a complete GGUF file with its own header and data section,
//! and carries `split.no` (counting from 0), `split.count` and `split.tensors.count`.
//! [`GgufModel`] reads every split's header and checks that they form one set.

use super::{GgufError, GgufHeader, GgufTensor, Limits, MetaView};
use crate::range::{HeaderReader, RangeSource};

/// The key holding a split's index, counting from 0.
pub const SPLIT_NO: &str = "split.no";
/// The key holding the number of splits.
pub const SPLIT_COUNT: &str = "split.count";
/// The key holding the number of tensors across all splits.
pub const SPLIT_TENSORS_COUNT: &str = "split.tensors.count";

/// A split file's name: `<prefix>-NNNNN-of-MMMMM.gguf`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SplitName {
    /// The name before `-NNNNN-of-MMMMM.gguf`.
    pub prefix: String,
    /// The split's number, counting from 1.
    pub no: u16,
    /// The number of splits.
    pub count: u16,
}

impl SplitName {
    /// Parses a file name; `None` when it is not a split name or its numbers are out of
    /// range (`no` is 1 to `count`).
    #[must_use]
    pub fn parse(file_name: &str) -> Option<Self> {
        let stem = file_name.strip_suffix(".gguf")?;
        // "-NNNNN-of-MMMMM" is 15 bytes.
        let cut = stem.len().checked_sub(15)?;
        let (prefix, tail) = (stem.get(..cut)?, stem.get(cut..)?);
        let tail = tail.as_bytes();
        let digits = |b: &[u8]| -> Option<u16> {
            if b.len() == 5 && b.iter().all(u8::is_ascii_digit) {
                std::str::from_utf8(b).ok()?.parse().ok()
            } else {
                None
            }
        };
        if tail[0] != b'-' || &tail[6..10] != b"-of-" {
            return None;
        }
        let no = digits(&tail[1..6])?;
        let count = digits(&tail[10..15])?;
        if prefix.is_empty() || no == 0 || no > count {
            return None;
        }
        Some(Self {
            prefix: prefix.to_owned(),
            no,
            count,
        })
    }

    /// The file name of split `no` (counting from 1) of this set.
    #[must_use]
    pub fn file_name(&self, no: u16) -> String {
        format!("{}-{no:05}-of-{:05}.gguf", self.prefix, self.count)
    }

    /// Every file name of this set, in split order.
    #[must_use]
    pub fn file_names(&self) -> Vec<String> {
        (1..=self.count).map(|n| self.file_name(n)).collect()
    }
}

/// A model's GGUF files: one file, or every split of a split set, in split order.
#[derive(Clone, Debug, PartialEq)]
pub struct GgufModel {
    files: Vec<GgufHeader>,
}

impl GgufModel {
    /// Each file's header, in split order (never empty). Each tensor's
    /// [`file`](GgufTensor::file) is its index here.
    #[must_use]
    pub fn files(&self) -> &[GgufHeader] {
        &self.files
    }

    /// Reads every file's header. `sources` are the set's files in split order: a single file,
    /// or the files [`SplitName::file_names`] lists.
    ///
    /// # Errors
    ///
    /// A file's error, wrapped in [`GgufError::InFile`] for a split set, or
    /// [`GgufError::SplitInvalid`] when the files do not form one set.
    pub fn read<S: RangeSource>(
        sources: impl IntoIterator<Item = S>,
        limits: &Limits,
    ) -> Result<Self, GgufError> {
        let mut readers: Vec<_> = sources
            .into_iter()
            .map(|s| HeaderReader::new(s, limits.header_bytes))
            .collect();
        Self::read_from(&mut readers, limits)
    }

    /// As [`read`](Self::read), through header readers the caller set up.
    ///
    /// # Errors
    ///
    /// As [`read`](Self::read).
    pub fn read_from<S: RangeSource>(
        readers: &mut [HeaderReader<S>],
        limits: &Limits,
    ) -> Result<Self, GgufError> {
        let invalid = |reason: String| GgufError::SplitInvalid { reason };
        let n = readers.len();
        if n == 0 {
            return Err(invalid("no files".into()));
        }
        let n16 = u16::try_from(n).map_err(|_| invalid(format!("{n} files, more than 65535")))?;
        let wrap = |file: u16, error: GgufError| {
            if n == 1 {
                error
            } else {
                GgufError::InFile {
                    file,
                    error: Box::new(error),
                }
            }
        };

        let mut files = Vec::with_capacity(n);
        let mut total = 0u64;
        for (i, reader) in (0u16..).zip(readers.iter_mut()) {
            let mut h = GgufHeader::read_from(reader, limits).map_err(|e| wrap(i, e))?;
            let meta_err = |e: super::MetaError| wrap(i, invalid(e.to_string()));
            let no = h.metadata.u64(SPLIT_NO).map_err(meta_err)?;
            let count = h.metadata.u64(SPLIT_COUNT).map_err(meta_err)?;
            match (no, count) {
                (None, None) if n == 1 => {}
                (Some(no), Some(count)) if no == u64::from(i) && count == u64::from(n16) => {}
                _ => {
                    return Err(wrap(
                        i,
                        invalid(format!(
                            "file {i} of {n} has `{SPLIT_NO}` = {no:?} and `{SPLIT_COUNT}` = {count:?}; expected {i} and {n}"
                        )),
                    ));
                }
            }
            h.set_file(i);
            total += h.tensor_count() as u64;
            files.push(h);
        }
        // Names are unique across the set (each file's own are checked already): sort every
        // (file, entry) pair by name, 8 bytes per tensor, and compare neighbours.
        if n > 1 {
            let mut all: Vec<(u16, u32)> = Vec::with_capacity(usize::try_from(total).unwrap_or(0));
            for (i, h) in (0u16..).zip(&files) {
                all.extend((0..u32::try_from(h.tensor_count()).unwrap_or(0)).map(|j| (i, j)));
            }
            let name = |&(f, j): &(u16, u32)| {
                files[usize::from(f)]
                    .tensor(j as usize)
                    .map_or("", |t| t.name)
            };
            all.sort_unstable_by(|a, b| name(a).cmp(name(b)).then(a.cmp(b)));
            if let Some(w) = all.windows(2).find(|w| name(&w[0]) == name(&w[1])) {
                return Err(wrap(
                    w[1].0,
                    invalid(format!(
                        "tensor `{}` is in more than one split",
                        name(&w[1])
                    )),
                ));
            }
        }
        for (i, h) in (0u16..).zip(&files) {
            let declared = h
                .metadata
                .u64(SPLIT_TENSORS_COUNT)
                .map_err(|e| wrap(i, invalid(e.to_string())))?;
            if let Some(d) = declared
                && d != total
            {
                return Err(wrap(
                    i,
                    invalid(format!(
                        "`{SPLIT_TENSORS_COUNT}` is {d}, but the set holds {total} tensors"
                    )),
                ));
            }
        }
        Ok(Self { files })
    }

    /// The metadata: the first file's, which holds the model's keys.
    #[must_use]
    pub fn metadata(&self) -> &MetaView {
        &self.files[0].metadata
    }

    /// Every tensor of every file, in split order and then file order.
    pub fn tensors(&self) -> impl Iterator<Item = GgufTensor<'_>> {
        self.files.iter().flat_map(GgufHeader::tensors)
    }
}
