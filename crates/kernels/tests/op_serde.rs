// SPDX-License-Identifier: MIT OR Apache-2.0

//! Serde round trips of the op types the selection cache and `OpCert` store (COMP-E1-T02;
//! RFC-0040 §6.5, RFC-0060 §6.4): `OpKind`, `OpKey`, `LayoutVersion`, `Fusions`,
//! `ShapeClass` and `ShapeSig`. JSON stands in for any self-describing format.

use std::fmt::Debug;

use proptest::prelude::*;
use serde::{Serialize, de::DeserializeOwned};
use smallvec::SmallVec;
use syntropy_core::{DType, GgmlType, QuantScheme};
use syntropy_kernels::*;

fn round_trip<T>(value: &T) -> Result<(), TestCaseError>
where
    T: Serialize + DeserializeOwned + PartialEq + Debug,
{
    let json = serde_json::to_string(value).expect("encodes");
    let back: T = serde_json::from_str(&json).expect("decodes");
    prop_assert_eq!(&back, value);
    Ok(())
}

fn op_kind() -> impl Strategy<Value = OpKind> {
    prop::sample::select(vec![
        OpKind::Gemv,
        OpKind::GateUpAct {
            act: GatedAct::ClampedSilu,
        },
        OpKind::MoeRoute {
            norm: RouteNorm::Sigmoid,
        },
        OpKind::AttnDecode(AttnKind::SlidingGqa { sinks: true }),
        OpKind::AttnPartial(AttnKind::Mla),
        OpKind::AttnIndex(IndexKind::TokenTopK),
        OpKind::ScanPrefill(ScanKind::Mamba2),
        OpKind::Rope(RopeKind::Yarn),
        OpKind::Act(ActKind::Gelu),
        OpKind::Pool(PoolKind::LastToken),
        OpKind::KvUnpack,
        OpKind::HiddenTap,
    ])
}

fn scheme() -> impl Strategy<Value = QuantScheme> {
    prop::sample::select(vec![
        QuantScheme::Plain(DType::F32),
        QuantScheme::Plain(DType::BF16),
        QuantScheme::Ggml(GgmlType::Q4_K),
        QuantScheme::Mxfp4 { block: 32 },
    ])
}

fn op_key() -> impl Strategy<Value = OpKey> {
    (
        op_kind(),
        prop::collection::vec(scheme(), 0..=3),
        any::<u16>(),
        0u16..64,
    )
        .prop_map(|(op, schemes, layout, variant)| OpKey {
            op,
            schemes: SmallVec::from_vec(schemes),
            layout: LayoutVersion(layout),
            variant,
        })
}

fn shape_class() -> impl Strategy<Value = ShapeClass> {
    prop_oneof![
        Just(ShapeClass::Decode),
        any::<u16>().prop_map(|window| ShapeClass::Verify { window }),
        any::<u32>().prop_map(|rows| ShapeClass::Batch { rows }),
        any::<u32>().prop_map(|chunk| ShapeClass::Prefill { chunk }),
        any::<u32>().prop_map(|chunk| ShapeClass::PrefillStream { chunk }),
    ]
}

fn shape_sig() -> impl Strategy<Value = ShapeSig> {
    (shape_class(), prop::collection::vec(any::<u32>(), 0..=8))
        .prop_map(|(class, dims)| ShapeSig::new(class, &dims))
}

proptest! {
    #[test]
    fn op_kind_round_trips(v in op_kind()) { round_trip(&v)?; }

    #[test]
    fn op_key_round_trips(v in op_key()) { round_trip(&v)?; }

    #[test]
    fn fusions_round_trip(bits in 0u16..64) {
        round_trip(&Fusions::from_bits_truncate(bits))?;
    }

    #[test]
    fn shape_class_round_trips(v in shape_class()) { round_trip(&v)?; }

    #[test]
    fn shape_sig_round_trips(v in shape_sig()) { round_trip(&v)?; }
}
