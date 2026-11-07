// SPDX-License-Identifier: MIT OR Apache-2.0

use smallvec::SmallVec;
use syntropy_core::resource::GgmlBackend;
use syntropy_core::{DType, GgmlType, QuantScheme};
use syntropy_kernels::{ActKind, GatedAct, LayoutVersion, OpKey, OpKind, ShapeClass, ShapeSig};

use super::{Mapping, WeightType, capture_safe, mapping};

const F32: QuantScheme = QuantScheme::Plain(DType::F32);
const Q8: QuantScheme = QuantScheme::Ggml(GgmlType::Q8_0);
const BF16: QuantScheme = QuantScheme::Plain(DType::BF16);

fn key(op: OpKind, schemes: &[QuantScheme]) -> OpKey {
    OpKey {
        op,
        schemes: SmallVec::from_slice(schemes),
        layout: LayoutVersion::SOURCE,
        variant: 0,
    }
}

fn map(op: OpKind, schemes: &[QuantScheme], class: ShapeClass, dims: &[u32]) -> Mapping {
    mapping(&key(op, schemes), &ShapeSig::new(class, dims)).unwrap()
}

#[test]
fn each_mapped_kind_reads_its_dims_in_the_registry_order() {
    let d = ShapeClass::Decode;
    let w = |s| WeightType::of(&s).unwrap();
    assert_eq!(
        map(OpKind::Gemv, &[Q8], d, &[16, 64]),
        Mapping::MatMul {
            weight: w(Q8),
            n_out: 16,
            k_in: 64
        }
    );
    assert_eq!(
        map(OpKind::Logits, &[F32], d, &[1000, 64]),
        Mapping::MatMul {
            weight: w(F32),
            n_out: 1000,
            k_in: 64
        }
    );
    assert_eq!(
        map(
            OpKind::GateUpAct {
                act: GatedAct::Silu
            },
            &[BF16],
            d,
            &[16, 64]
        ),
        Mapping::GateUpSilu {
            weight: w(BF16),
            n_out: 16,
            k_in: 64
        }
    );
    assert_eq!(
        map(OpKind::RmsNorm, &[F32], d, &[64]),
        Mapping::RmsNorm { hidden: 64 }
    );
    assert_eq!(
        map(OpKind::Act(ActKind::Silu), &[], d, &[9]),
        Mapping::Silu { n: 9 }
    );
    assert_eq!(
        map(OpKind::Add, &[F32, F32, F32], d, &[9]),
        Mapping::Add { n: 9 }
    );
    assert_eq!(map(OpKind::Mul, &[F32], d, &[9]), Mapping::Mul { n: 9 });
    assert_eq!(
        map(OpKind::RowGather, &[Q8], d, &[1000, 64]),
        Mapping::RowGather {
            table: w(Q8),
            table_rows: 1000,
            width: 64
        }
    );
    assert_eq!(
        map(OpKind::Convert, &[BF16, F32, F32], d, &[9]),
        Mapping::Convert {
            from: DType::BF16,
            to: DType::F32,
            n: 9
        }
    );
}

#[test]
fn elementwise_ops_need_f32_schemes() {
    let s = ShapeSig::new(ShapeClass::Decode, &[8]);
    for op in [
        OpKind::Add,
        OpKind::Mul,
        OpKind::RmsNorm,
        OpKind::Act(ActKind::Silu),
    ] {
        assert!(mapping(&key(op, &[F32]), &s).is_some(), "{op:?}");
        assert_eq!(mapping(&key(op, &[BF16]), &s), None, "{op:?}");
        assert_eq!(mapping(&key(op, &[F32, F32, Q8]), &s), None, "{op:?}");
    }
}

#[test]
fn a_row_gather_needs_whole_blocks_per_row() {
    let s = ShapeSig::new(ShapeClass::Decode, &[10, 48]);
    assert_eq!(mapping(&key(OpKind::RowGather, &[Q8]), &s), None);
    assert!(mapping(&key(OpKind::RowGather, &[F32]), &s).is_some());
}

#[test]
fn an_empty_bucket_is_not_mapped() {
    let mut s = ShapeSig::new(ShapeClass::Decode, &[8]);
    s.rows = 0;
    assert_eq!(mapping(&key(OpKind::Add, &[F32]), &s), None);
}

/// RFC-0040 §8.2: on CUDA and HIP, `Gemm`, and `mul_mat` over float weights or at prefill and
/// batch buckets, may reach vendor BLAS and are not capture-safe; quantised `Gemv`, `Logits` and
/// `GateUpAct` at decode and verify buckets are.
#[test]
fn capture_safety_follows_section_8_2() {
    let decode = ShapeClass::Decode;
    let verify = ShapeClass::Verify { window: 4 };
    let prefill = ShapeClass::Prefill { chunk: 256 };
    let stream = ShapeClass::PrefillStream { chunk: 256 };
    let batch = ShapeClass::Batch { rows: 8 };
    let gate = OpKind::GateUpAct {
        act: GatedAct::Silu,
    };
    let mm = |op, s: QuantScheme, class| (op, map(op, &[s], class, &[16, 64]), class);
    for backend in [GgmlBackend::Cuda, GgmlBackend::Hip] {
        for (op, m, class) in [
            mm(OpKind::Gemv, Q8, decode),
            mm(OpKind::Gemv, Q8, verify),
            mm(OpKind::Logits, Q8, decode),
            mm(gate, Q8, verify),
        ] {
            assert!(
                capture_safe(backend, op, &m, class),
                "{backend:?} {op:?} {class:?}"
            );
        }
        for (op, m, class) in [
            mm(OpKind::Gemm, Q8, decode),
            mm(OpKind::Gemm, Q8, prefill),
            mm(OpKind::Gemv, F32, decode),
            mm(OpKind::Gemv, BF16, verify),
            mm(OpKind::Logits, F32, decode),
            mm(gate, BF16, decode),
            mm(OpKind::Gemv, Q8, prefill),
            mm(OpKind::Gemv, Q8, stream),
            mm(OpKind::Logits, Q8, batch),
            mm(gate, Q8, batch),
        ] {
            assert!(
                !capture_safe(backend, op, &m, class),
                "{backend:?} {op:?} {class:?}"
            );
        }
        // No `mul_mat`: capture-safe at any bucket.
        let rms = map(OpKind::RmsNorm, &[F32], prefill, &[64]);
        assert!(capture_safe(backend, OpKind::RmsNorm, &rms, prefill));
        let gather = map(OpKind::RowGather, &[F32], prefill, &[10, 64]);
        assert!(capture_safe(backend, OpKind::RowGather, &gather, prefill));
    }
    // Other backends have no vendor BLAS in ggml's path.
    for backend in [
        GgmlBackend::Cpu,
        GgmlBackend::Metal,
        GgmlBackend::Vulkan,
        GgmlBackend::Sycl,
    ] {
        let (op, m, class) = mm(OpKind::Gemm, F32, prefill);
        assert!(capture_safe(backend, op, &m, class), "{backend:?}");
    }
}

#[test]
fn float_weights_are_float() {
    for (s, float) in [
        (F32, true),
        (BF16, true),
        (QuantScheme::Plain(DType::F16), true),
        (QuantScheme::Ggml(GgmlType::BF16), true),
        (Q8, false),
        (QuantScheme::Ggml(GgmlType::Q4_K), false),
    ] {
        assert_eq!(WeightType::of(&s).unwrap().is_float(), float, "{s:?}");
    }
    assert_eq!(WeightType::of(&QuantScheme::Plain(DType::I8)), None);
    assert_eq!(WeightType::of(&QuantScheme::Ggml(GgmlType::MXFP4)), None);
}
