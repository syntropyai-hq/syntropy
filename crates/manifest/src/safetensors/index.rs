// SPDX-License-Identifier: MIT OR Apache-2.0

//! The sharded index, `model.safetensors.index.json` (RFC-0041 §2.2, "Sharding").
//!
//! The index maps each tensor name to the shard file that holds it. [`read_index`] reads and
//! checks it; [`read_sharded`] then reads every shard's header and checks that the two agree:
//! every indexed tensor is in its shard, every shard tensor is indexed to that shard, and no
//! name is in two shards. A shard name must be a plain file name in the index's directory, so
//! an index cannot point a reader at another path. The index is read whole, bounded by
//! [`HEADER_MAX`](super::HEADER_MAX) like a header.

use super::json::Json;
use super::{HEADER_MAX, SafetensorsError, SafetensorsHeader, read_header};
use crate::range::{HeaderReader, RangeSource};

/// The index's file name.
pub const INDEX_FILE: &str = "model.safetensors.index.json";

/// A sharded index, checked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardIndex {
    /// `metadata.total_size`, when the index gives it. Recorded, not checked: writers differ
    /// on what it counts.
    pub total_size: Option<u64>,
    /// `weight_map`, in index order: (tensor, shard file).
    pub weight_map: Vec<(String, String)>,
    /// The shard files, each once, in order of first appearance in `weight_map`.
    pub shards: Vec<String>,
}

/// A sharded checkpoint's headers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardedHeaders {
    /// The index.
    pub index: ShardIndex,
    /// Each shard's header, in [`ShardIndex::shards`] order.
    pub shards: Vec<(String, SafetensorsHeader)>,
}

impl ShardedHeaders {
    /// The shard file and header that hold `tensor`.
    #[must_use]
    pub fn locate(&self, tensor: &str) -> Option<(&str, &super::SafetensorsTensor)> {
        self.shards
            .iter()
            .find_map(|(f, h)| h.tensor(tensor).map(|t| (f.as_str(), t)))
    }
}

/// Whether `name` is a plain file name: not empty, not `.` or `..`, and free of path
/// separators and NUL.
#[must_use]
pub fn is_plain_file_name(name: &str) -> bool {
    !name.is_empty()
        && name != "."
        && name != ".."
        && !name.contains(['/', '\\', '\0'])
        && !name.contains(':')
}

/// Reads and checks the index from `source`.
///
/// # Errors
///
/// [`SafetensorsError::HeaderLimit`] when the file passes the limit,
/// [`SafetensorsError::HeaderInvalid`] when it is malformed, and read failures.
pub fn read_index<S: RangeSource>(
    source: S,
    header_max_bytes: u64,
) -> Result<ShardIndex, SafetensorsError> {
    let len = source.len();
    let cap = HEADER_MAX.min(header_max_bytes);
    if len > cap {
        return Err(SafetensorsError::HeaderLimit { len, cap });
    }
    let mut reader = HeaderReader::new(source, header_max_bytes).with_initial_read(len.max(1));
    let bytes = reader.read(0, len)?;
    parse_index(bytes)
}

/// Checks the index text `bytes`.
///
/// # Errors
///
/// [`SafetensorsError::HeaderInvalid`] when it is malformed.
pub fn parse_index(bytes: &[u8]) -> Result<ShardIndex, SafetensorsError> {
    let bad = |what: String| SafetensorsError::invalid(format!("{INDEX_FILE}: {what}"));
    let json = Json::parse(bytes).map_err(|e| bad(format!("not valid JSON: {e}")))?;
    let Json::Object(top) = json else {
        return Err(bad("not a JSON object".into()));
    };
    let mut total_size = None;
    let mut weight_map = None;
    for (key, value) in top {
        match (key.as_str(), value) {
            ("metadata", Json::Object(meta)) => {
                for (k, v) in meta {
                    if k == "total_size" {
                        match v {
                            Json::Uint(n) => total_size = Some(n),
                            other => {
                                return Err(bad(format!(
                                    "metadata.total_size is {}, not a non-negative integer",
                                    other.kind()
                                )));
                            }
                        }
                    }
                }
            }
            ("metadata", other) => {
                return Err(bad(format!("metadata is {}, not an object", other.kind())));
            }
            ("weight_map", Json::Object(map)) => weight_map = Some(map),
            ("weight_map", other) => {
                return Err(bad(format!(
                    "weight_map is {}, not an object",
                    other.kind()
                )));
            }
            (other, _) => return Err(bad(format!("unknown key {other:?}"))),
        }
    }
    let map = weight_map.ok_or_else(|| bad("no weight_map".into()))?;
    let mut out = Vec::with_capacity(map.len());
    let mut shards: Vec<String> = Vec::new();
    for (tensor, file) in map {
        let Json::Str(file) = file else {
            return Err(bad(format!(
                "weight_map.{tensor} is {}, not a file name",
                file.kind()
            )));
        };
        if !is_plain_file_name(&file) {
            return Err(bad(format!(
                "weight_map.{tensor} names {file:?}, which is not a plain file name"
            )));
        }
        if !shards.contains(&file) {
            shards.push(file.clone());
        }
        out.push((tensor, file));
    }
    Ok(ShardIndex {
        total_size,
        weight_map: out,
        shards,
    })
}

/// Reads the index from `index_source`, then every shard's header through `open`, which
/// opens a shard by its file name, and checks that they agree.
///
/// # Errors
///
/// As [`read_index`] and [`read_header`](super::read_header), `open`'s errors, and
/// [`SafetensorsError::HeaderInvalid`] when the index and the shards disagree.
pub fn read_sharded<S, T, F>(
    index_source: S,
    mut open: F,
    header_max_bytes: u64,
) -> Result<ShardedHeaders, SafetensorsError>
where
    S: RangeSource,
    T: RangeSource,
    F: FnMut(&str) -> Result<T, SafetensorsError>,
{
    let index = read_index(index_source, header_max_bytes)?;
    let mut shards = Vec::with_capacity(index.shards.len());
    for file in &index.shards {
        let header = read_header(open(file)?, header_max_bytes)?;
        shards.push((file.clone(), header));
    }
    check_agreement(&index, &shards)?;
    Ok(ShardedHeaders { index, shards })
}

fn check_agreement(
    index: &ShardIndex,
    shards: &[(String, SafetensorsHeader)],
) -> Result<(), SafetensorsError> {
    // Ordered maps (RFC-0060 §2.4 rule T3), though only lookups and inserts read them.
    use std::collections::BTreeMap;
    let indexed: BTreeMap<&str, &str> = index
        .weight_map
        .iter()
        .map(|(t, f)| (t.as_str(), f.as_str()))
        .collect();
    let mut holder: BTreeMap<&str, &str> = BTreeMap::new();
    for (file, header) in shards {
        for t in &header.tensors {
            if let Some(other) = holder.insert(t.name.as_str(), file.as_str()) {
                return Err(SafetensorsError::invalid(format!(
                    "tensor {:?} is in both {other:?} and {file:?}",
                    t.name
                )));
            }
            match indexed.get(t.name.as_str()) {
                Some(f) if f == file => {}
                Some(f) => {
                    return Err(SafetensorsError::invalid(format!(
                        "tensor {:?} is in {file:?}, but the index places it in {f:?}",
                        t.name
                    )));
                }
                None => {
                    return Err(SafetensorsError::invalid(format!(
                        "tensor {:?} in {file:?} is not in the index",
                        t.name
                    )));
                }
            }
        }
    }
    for (tensor, file) in &index.weight_map {
        if !holder.contains_key(tensor.as_str()) {
            return Err(SafetensorsError::invalid(format!(
                "the index places tensor {tensor:?} in {file:?}, which does not hold it"
            )));
        }
    }
    Ok(())
}
