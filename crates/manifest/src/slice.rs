// SPDX-License-Identifier: MIT OR Apache-2.0

//! Expert slicing: a stacked expert tensor becomes one record per expert, each with its own
//! extent in the source file (RFC-0041 §2.1 "Expert tensors", §2.2; COMP-E10-T03).
//!
//! A family's classifier marks a stacked expert tensor and says how many experts it holds
//! (`Classification::Classified { stacked_experts: Some(n), .. }`, RFC-0043 §4). This module
//! slices it, for the census builder (RFC-0041 §3 step 5, COMP-E10-T07) and the chunk boundaries
//! (RFC-0041 §6.1, COMP-E12-T01), which start a chunk at every expert:
//! - **GGUF** ([`gguf_experts`]): the expert index is the outermost dimension, `ne[last]`, as in
//!   `blk.L.ffn_{gate,up,down,gate_up}_exps.weight` with `ne[2] = n_expert`. Slice `e` is the
//!   extent (`offset + e · nb[last]`, `nb[last]`), where `nb` is ggml's byte stride of each
//!   dimension: `nb[0]` is the type's block bytes, `nb[1] = nb[0] · ne[0] / block` and
//!   `nb[i] = nb[i−1] · ne[i−1]` after that. For the three-dimensional tensors of RFC-0041 §2.1
//!   that is `nb[2]`.
//! - **safetensors** ([`safetensors_experts`]): the expert index is dimension 0, the outermost.
//!   Slice `e` starts `e` per-expert lengths after the tensor's first byte.
//!
//! Each slice is named `<tensor>[e]`, takes the tensor's shape without the expert dimension, and
//! has its file index from the tensor. The slices of a tensor tile its bytes exactly: in order,
//! with no gap or overlap, from its first byte to its last. A tensor whose expert dimension or
//! byte length does not divide into the experts the classifier named is `HeaderInvalid` (6).

use std::fmt;

use syntropy_core::{ErrorClass, ExpertIx, GgmlType};

use crate::gguf::{GgufTensor, block_geometry};
use crate::safetensors::SafetensorsTensor;

/// Where a tensor's bytes are: a file of the model, and a byte range in it (RFC-0001 §5.3
/// `SourceExtent`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SourceExtent {
    /// The file's index among the model's source files: a GGUF split, or a safetensors shard.
    pub file: u16,
    /// The first byte, as an offset into the file.
    pub offset: u64,
    /// The length in bytes.
    pub len: u64,
}

/// One expert's slice of a stacked expert tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertSlice {
    /// `<tensor>[e]`, as RFC-0041 §2.1 names it.
    pub name: String,
    /// The expert.
    pub expert: ExpertIx,
    /// The tensor's shape without the expert dimension, in the source's own order: innermost
    /// first for GGUF, outermost first for safetensors.
    pub shape: Vec<u64>,
    /// Its bytes.
    pub extent: SourceExtent,
}

/// Why a stacked expert tensor cannot be sliced. Every case is `HeaderInvalid` (RFC-0041 §10
/// code 6): the file's shape or length disagrees with the experts its family named.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SliceError {
    /// The tensor.
    pub tensor: String,
    /// What is wrong.
    pub reason: String,
}

impl SliceError {
    fn new(tensor: &str, reason: impl Into<String>) -> Self {
        Self {
            tensor: tensor.to_owned(),
            reason: reason.into(),
        }
    }

    /// The `Formats` code RFC-0041 §10 gives it: 6, `HeaderInvalid`.
    #[must_use]
    pub fn code(&self) -> u16 {
        6
    }

    /// Its class (RFC-0001 §10): `Integrity`.
    #[must_use]
    pub fn class(&self) -> ErrorClass {
        ErrorClass::Integrity
    }
}

impl fmt::Display for SliceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "the stacked expert tensor `{}` cannot be sliced: {}",
            self.tensor, self.reason
        )
    }
}

impl std::error::Error for SliceError {}

/// ggml's byte stride of each dimension of a tensor of type `t` and shape `ne` (innermost
/// first), or `None` when the innermost dimension is not a whole number of blocks, the type has
/// no block geometry, or a stride overflows.
#[must_use]
pub fn ggml_strides(t: GgmlType, ne: &[u64]) -> Option<Vec<u64>> {
    let (block, block_bytes) = block_geometry(t)?;
    let first = *ne.first()?;
    if block == 0 || first % block != 0 {
        return None;
    }
    let mut nb = Vec::with_capacity(ne.len());
    nb.push(block_bytes);
    if ne.len() > 1 {
        nb.push(block_bytes.checked_mul(first / block)?);
    }
    for i in 2..ne.len() {
        nb.push(nb[i - 1].checked_mul(ne[i - 1])?);
    }
    Some(nb)
}

/// The slices of the GGUF tensor `tensor`, which its family says stacks `n_expert` experts with
/// the expert index outermost. Slice `e` is (`start + e · nb[last]`, `nb[last]`).
///
/// # Errors
///
/// [`SliceError`] when the tensor has fewer than two dimensions, its outermost dimension is not
/// `n_expert`, its strides cannot be computed, or the experts do not tile its bytes exactly.
pub fn gguf_experts(
    tensor: &GgufTensor<'_>,
    n_expert: u16,
) -> Result<Vec<ExpertSlice>, SliceError> {
    let ne = tensor.ne();
    let name = tensor.name;
    let Some((&outer, inner)) = ne.split_last() else {
        return Err(SliceError::new(name, "it has no dimensions"));
    };
    if inner.is_empty() {
        return Err(SliceError::new(
            name,
            "it has one dimension, so no expert dimension beside the expert's own",
        ));
    }
    if outer != u64::from(n_expert) {
        return Err(SliceError::new(
            name,
            format!("its outermost dimension is {outer}, and its family names {n_expert} experts"),
        ));
    }
    let nb = ggml_strides(tensor.ggml_type, ne).ok_or_else(|| {
        SliceError::new(
            name,
            "its innermost dimension is not a whole number of its type's blocks",
        )
    })?;
    let stride = *nb.last().expect("a stride per dimension");
    slices(
        name,
        tensor.file,
        tensor.start,
        tensor.bytes,
        stride,
        n_expert,
        inner,
    )
}

/// The slices of the safetensors tensor `tensor`, in file `file`, which its family says stacks
/// `n_expert` experts in dimension 0.
///
/// # Errors
///
/// [`SliceError`] when the tensor has no dimension, its dimension 0 is not `n_expert`, or one
/// expert's bytes are not a whole number of bytes or do not tile the tensor exactly.
pub fn safetensors_experts(
    tensor: &SafetensorsTensor,
    file: u16,
    n_expert: u16,
) -> Result<Vec<ExpertSlice>, SliceError> {
    let name = tensor.name.as_str();
    let Some((&outer, inner)) = tensor.shape.split_first() else {
        return Err(SliceError::new(
            name,
            "it is a scalar, with no expert dimension",
        ));
    };
    if outer != u64::from(n_expert) {
        return Err(SliceError::new(
            name,
            format!("its dimension 0 is {outer}, and its family names {n_expert} experts"),
        ));
    }
    let stride = tensor.dtype.byte_len(inner).map_err(|_| {
        SliceError::new(
            name,
            "one expert's elements are not a whole number of bytes",
        )
    })?;
    slices(
        name,
        file,
        tensor.offset,
        tensor.len,
        stride,
        n_expert,
        inner,
    )
}

/// `n_expert` slices of `stride` bytes from `start`, which must tile `bytes` exactly.
fn slices(
    name: &str,
    file: u16,
    start: u64,
    bytes: u64,
    stride: u64,
    n_expert: u16,
    shape: &[u64],
) -> Result<Vec<ExpertSlice>, SliceError> {
    if n_expert == 0 {
        return Err(SliceError::new(name, "its family names no experts"));
    }
    let total = stride.checked_mul(u64::from(n_expert));
    if total != Some(bytes) {
        return Err(SliceError::new(
            name,
            format!("{n_expert} experts of {stride} bytes do not make its {bytes} bytes"),
        ));
    }
    Ok((0..n_expert)
        .map(|e| ExpertSlice {
            name: format!("{name}[{e}]"),
            expert: ExpertIx(e),
            shape: shape.to_vec(),
            extent: SourceExtent {
                file,
                offset: start + u64::from(e) * stride,
                len: stride,
            },
        })
        .collect())
}

#[cfg(test)]
mod tests;
