// SPDX-License-Identifier: MIT OR Apache-2.0

//! COMP-E10-T03's tests: on synthetic GGUF and safetensors files, the slices of each stacked
//! expert tensor tile it exactly, and a GGUF slice's extent is `offset + e · nb[2]`, with `nb[2]`
//! computed here from ggml's stride rule rather than taken from the code under test.

use proptest::prelude::*;
use syntropy_core::{ExpertIx, GgmlType};

use super::*;
use crate::gguf::synth::Synth;
use crate::gguf::{GgufHeader, Limits, Value, ggml_type_id};
use crate::range::{RangeError, RangeSource, check_bounds};
use crate::safetensors::{SafetensorsDtype, parse_header};

/// A file in memory.
struct Mem(Vec<u8>);

impl RangeSource for Mem {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), RangeError> {
        check_bounds(offset, buf.len() as u64, self.len())?;
        let at = offset as usize;
        buf.copy_from_slice(&self.0[at..at + buf.len()]);
        Ok(())
    }
}

/// The ggml types the synthetic writer and the reader both know.
fn gguf_types() -> Vec<GgmlType> {
    GgmlType::ALL
        .into_iter()
        .filter(|t| block_geometry(*t).is_some() && ggml_type_id(*t).is_some())
        .collect()
}

/// `nb[2]` of a `[ne0, ne1, n]` tensor of type `t`, by ggml's rule: `nb[0]` is the block's
/// bytes, `nb[1] = nb[0] · ne0 / block`, `nb[2] = nb[1] · ne1`.
fn nb2(t: GgmlType, ne0: u64, ne1: u64) -> u64 {
    let (block, block_bytes) = block_geometry(t).unwrap();
    block_bytes * (ne0 / block) * ne1
}

/// One stacked tensor of a synthetic GGUF: its type, `ne[0]` in blocks, `ne[1]`, the experts.
#[derive(Clone, Debug)]
struct Stack {
    t: GgmlType,
    blocks: u64,
    rows: u64,
    experts: u16,
}

fn stack() -> impl Strategy<Value = Stack> {
    (
        prop::sample::select(gguf_types()),
        1u64..4,
        1u64..6,
        1u16..10,
    )
        .prop_map(|(t, blocks, rows, experts)| Stack {
            t,
            blocks,
            rows,
            experts,
        })
}

/// Checks the slices of one tensor against its extent, `nb`, its shape and its name.
fn check_tiling(
    slices: &[ExpertSlice],
    name: &str,
    file: u16,
    start: u64,
    bytes: u64,
    stride: u64,
    shape: &[u64],
) -> Result<(), TestCaseError> {
    prop_assert_eq!(u64::try_from(slices.len()).unwrap() * stride, bytes);
    let mut at = start;
    for (e, slice) in slices.iter().enumerate() {
        let e16 = u16::try_from(e).unwrap();
        prop_assert_eq!(slice.expert, ExpertIx(e16));
        prop_assert_eq!(&slice.name, &format!("{name}[{e}]"));
        prop_assert_eq!(&slice.shape, &shape.to_vec());
        prop_assert_eq!(slice.extent.file, file);
        // Tiling: each slice starts where the one before ended.
        prop_assert_eq!(slice.extent.offset, at);
        // The rule: offset + e · stride, with stride's length.
        prop_assert_eq!(slice.extent.offset, start + u64::from(e16) * stride);
        prop_assert_eq!(slice.extent.len, stride);
        at += slice.extent.len;
    }
    // And the last ends where the tensor ends.
    prop_assert_eq!(at, start + bytes);
    Ok(())
}

proptest! {
    /// GGUF: on a synthetic file with dense tensors around one or more stacked ones, at any
    /// alignment, the slices tile each stacked tensor exactly, and slice `e`'s extent is
    /// (`offset + e · nb[2]`, `nb[2]`).
    #[test]
    fn gguf_slices_tile_each_stack_at_offset_plus_e_nb2(
        stacks in prop::collection::vec(stack(), 1..4),
        alignment in prop::sample::select(vec![32u32, 64, 128, 256]),
        dense_rows in 1u64..4,
    ) {
        let mut synth = Synth::new(3)
            .kv("general.alignment", Value::U32(alignment))
            .tensor("token_embd.weight", &[32, dense_rows], GgmlType::F32);
        for (l, s) in stacks.iter().enumerate() {
            let (block, _) = block_geometry(s.t).unwrap();
            synth = synth
                .tensor(&format!("blk.{l}.attn_norm.weight"), &[32], GgmlType::F32)
                .tensor(
                    &format!("blk.{l}.ffn_up_exps.weight"),
                    &[s.blocks * block, s.rows, u64::from(s.experts)],
                    s.t,
                );
        }
        let file = synth.build();
        let file_len = file.len() as u64;
        let header = GgufHeader::read(Mem(file), &Limits::DEFAULT).unwrap();
        for (l, s) in stacks.iter().enumerate() {
            let name = format!("blk.{l}.ffn_up_exps.weight");
            let tensor = header.tensors().find(|t| t.name == name).unwrap();
            let slices = gguf_experts(&tensor, s.experts).unwrap();
            let (block, _) = block_geometry(s.t).unwrap();
            let ne0 = s.blocks * block;
            let stride = nb2(s.t, ne0, s.rows);
            check_tiling(
                &slices,
                &name,
                tensor.file,
                tensor.start,
                tensor.bytes,
                stride,
                &[ne0, s.rows],
            )?;
            prop_assert!(tensor.start + tensor.bytes <= file_len);
        }
    }

    /// safetensors: on a synthetic file with a dense tensor beside a stacked one, the slices
    /// tile it exactly, each one expert's bytes after the one before, from the tensor's first
    /// byte.
    #[test]
    fn safetensors_slices_tile_each_stack_along_dimension_0(
        dtype in prop::sample::select(vec![
            SafetensorsDtype::F32,
            SafetensorsDtype::F16,
            SafetensorsDtype::BF16,
            SafetensorsDtype::F8E4M3,
            SafetensorsDtype::U8,
            SafetensorsDtype::F4E2M1,
        ]),
        experts in 1u16..10,
        rows in 1u64..6,
        pairs in 1u64..5,
        file_ix in 0u16..4,
        dense_first in any::<bool>(),
    ) {
        // An even last dimension, so an F4 expert is a whole number of bytes.
        let inner = [rows, pairs * 2];
        let shape = [u64::from(experts), inner[0], inner[1]];
        let len = dtype.byte_len(&shape).unwrap();
        let dense = 16u64;
        let (stack_at, dense_at) = if dense_first { (dense, 0) } else { (0, len) };
        let json = format!(
            "{{\"dense.weight\":{{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[{},{}]}},\
             \"model.layers.0.mlp.experts.gate_up_proj\":{{\"dtype\":\"{}\",\"shape\":[{},{},{}],\
             \"data_offsets\":[{},{}]}}}}",
            dense_at,
            dense_at + dense,
            dtype.name(),
            shape[0],
            shape[1],
            shape[2],
            stack_at,
            stack_at + len,
        );
        let file_len = 8 + json.len() as u64 + len + dense;
        let header = parse_header(json.as_bytes(), file_len).unwrap();
        let tensor = header.tensor("model.layers.0.mlp.experts.gate_up_proj").unwrap();
        let slices = safetensors_experts(tensor, file_ix, experts).unwrap();
        let stride = dtype.byte_len(&inner).unwrap();
        check_tiling(
            &slices,
            &tensor.name,
            file_ix,
            tensor.offset,
            tensor.len,
            stride,
            &inner,
        )?;
        prop_assert_eq!(tensor.offset, header.data_start + stack_at);
    }
}

/// The three-dimensional case RFC-0041 §2.1 writes out, by hand: `ne = [64, 4, 3]` in Q8_0
/// (32 weights, 34 bytes per block) has `nb = [34, 68, 272]`, so expert 2 starts 544 bytes in.
#[test]
fn a_q8_0_stack_slices_at_its_nb2() {
    let nb = ggml_strides(GgmlType::Q8_0, &[64, 4, 3]).unwrap();
    assert_eq!(nb, [34, 68, 272]);
    let file = Synth::new(3)
        .tensor("blk.0.ffn_down_exps.weight", &[64, 4, 3], GgmlType::Q8_0)
        .build();
    let header = GgufHeader::read(Mem(file), &Limits::DEFAULT).unwrap();
    let tensor = header.tensor(0).unwrap();
    let slices = gguf_experts(&tensor, 3).unwrap();
    let offsets: Vec<u64> = slices
        .iter()
        .map(|s| s.extent.offset - tensor.start)
        .collect();
    assert_eq!(offsets, [0, 272, 544]);
    assert_eq!(slices[2].name, "blk.0.ffn_down_exps.weight[2]");
    assert_eq!(slices[2].shape, [64, 4]);
}

/// A tensor whose shape or length disagrees with the experts its family names is
/// `HeaderInvalid` (6), `Integrity`, naming the tensor.
#[test]
fn a_tensor_that_does_not_divide_into_its_experts_is_refused() {
    let file = Synth::new(3)
        .tensor("blk.0.ffn_up_exps.weight", &[32, 4, 8], GgmlType::F16)
        .tensor("output_norm.weight", &[32], GgmlType::F32)
        .build();
    let header = GgufHeader::read(Mem(file), &Limits::DEFAULT).unwrap();
    let stacked = header.tensor(0).unwrap();
    let flat = header.tensor(1).unwrap();
    for (error, says) in [
        (
            gguf_experts(&stacked, 4).unwrap_err(),
            "outermost dimension is 8",
        ),
        (
            gguf_experts(&stacked, 0).unwrap_err(),
            "outermost dimension is 8",
        ),
        (gguf_experts(&flat, 32).unwrap_err(), "one dimension"),
    ] {
        assert_eq!(error.code(), 6);
        assert_eq!(error.class(), ErrorClass::Integrity);
        assert!(error.to_string().contains(says), "{error}");
        assert!(error.to_string().contains(&error.tensor), "{error}");
    }

    // safetensors: the wrong expert count, a scalar, and an F4 expert of an odd element count.
    let json = "{\"a\":{\"dtype\":\"F4\",\"shape\":[2,3],\"data_offsets\":[0,3]},\
                \"s\":{\"dtype\":\"F32\",\"shape\":[],\"data_offsets\":[3,7]}}";
    let header = parse_header(json.as_bytes(), 8 + json.len() as u64 + 7).unwrap();
    let odd = header.tensor("a").unwrap();
    let scalar = header.tensor("s").unwrap();
    for (error, says) in [
        (
            safetensors_experts(odd, 0, 3).unwrap_err(),
            "dimension 0 is 2",
        ),
        (
            safetensors_experts(odd, 0, 2).unwrap_err(),
            "whole number of bytes",
        ),
        (safetensors_experts(scalar, 0, 1).unwrap_err(), "scalar"),
    ] {
        assert_eq!(error.code(), 6);
        assert!(error.to_string().contains(says), "{error}");
    }
}

/// `nb` is refused for an innermost dimension that is not a whole number of blocks, and for a
/// stride that overflows.
#[test]
fn strides_need_whole_blocks_and_no_overflow() {
    assert_eq!(ggml_strides(GgmlType::Q8_0, &[48, 2]), None);
    assert_eq!(ggml_strides(GgmlType::F32, &[]), None);
    assert_eq!(ggml_strides(GgmlType::F32, &[4]), Some(vec![4]));
    assert_eq!(ggml_strides(GgmlType::F32, &[u64::MAX, 2, 2]), None);
}
