// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tests of the boundary computation, the plan check and the partition against RFC-0043 §6.1,
//! §10 and RFC-0060 §7.2 check 8 (COMP-E15-T06).
//!
//! The graphs come from `ir::tests`' generator: GQA layers with dense or MoE FFNs, one or two
//! residual streams, n-gram layers that read token IDs, a per-layer input, and MTP modules.
//! `nano_shaped` builds an 8-layer graph with every one of those features, the size RFC-0060
//! §7.2 gives `nano` (at most 8 layers, so every pair is checked).

use proptest::prelude::*;
use proptest::sample::Index;
use smallvec::smallvec;
use syntropy_core::DType;

use super::*;
use crate::ir::tests::{Fixture, Params, build, params};
use crate::ir::{Value, ValueType};

const HIDDEN: u32 = 64;

fn none() -> BoundaryFacts<'static> {
    BoundaryFacts::default()
}

/// Installs the computed boundary set in the fixture's graph.
fn with_computed(mut f: Fixture, facts: &BoundaryFacts<'_>) -> (Fixture, BoundaryAnalysis) {
    let a = compute(&f.graph, facts).expect("computable");
    f.graph.boundaries = a.set.clone();
    (f, a)
}

/// RFC-0060 §7.2 check 8's property for one set of cuts: the parts' programs are the graph's
/// programs, each once and in order, and only residual streams and side data the boundary
/// declares cross each cut.
fn check_partition(f: &Fixture, cuts: &[LayerIx]) -> Result<(), TestCaseError> {
    let g = &f.graph;
    let parts = partition(g, cuts).map_err(|e| TestCaseError::fail(e.to_string()))?;
    prop_assert_eq!(parts.len(), cuts.len() + 1);
    let union: Vec<ProgramId> = parts.iter().flat_map(|p| p.programs.clone()).collect();
    let whole: Vec<ProgramId> = programs(g).into_iter().map(|(id, _)| id).collect();
    prop_assert_eq!(union, whole);
    for (k, &c) in cuts.iter().enumerate() {
        let b = usize::from(c.0);
        let side = g.boundaries.side[b];
        prop_assert_eq!(&parts[k].sends, &crossing(g, b));
        for &v in &parts[k].sends {
            let value = &g.values[v.0 as usize];
            let ok = match value.kind {
                ValueKind::Residual => value.ty.dtype == g.residual.dtype,
                ValueKind::TokenIds => side.token_ids,
                ValueKind::LayerInput => side.layer_inputs > 0,
                _ => false,
            };
            prop_assert!(
                ok,
                "value {} of kind {:?} crosses cut {}",
                v.0,
                value.kind,
                b
            );
        }
        // The inbound frame is exactly the outbound frame of the stage before.
        prop_assert_eq!(&parts[k + 1].receives, &parts[k].sends);
    }
    prop_assert!(parts.last().unwrap().sends.is_empty());
    prop_assert!(parts[0].receives.is_empty());
    Ok(())
}

/// Every two-stage and three-stage split at legal interior boundaries.
fn check_every_legal_pair(f: &Fixture) -> Result<usize, TestCaseError> {
    let n = f.graph.layers.len();
    let legal: Vec<LayerIx> = (1..n)
        .filter(|&b| f.graph.boundaries.legal[b])
        .map(|b| LayerIx(u16::try_from(b).unwrap()))
        .collect();
    let mut checked = 1;
    check_partition(f, &[])?;
    for (i, &a) in legal.iter().enumerate() {
        check_partition(f, &[a])?;
        checked += 1;
        for &b in &legal[i + 1..] {
            check_partition(f, &[a, b])?;
            checked += 1;
        }
    }
    Ok(checked)
}

/// The generator's own side data: token IDs up to the last n-gram layer, the per-layer input
/// up to its layer, 4 bytes per element.
fn expected_side(p: &Params, n: usize) -> Vec<SideData> {
    let last_ngram = p.layers.iter().rposition(|&(_, ngram)| ngram);
    (0..=n)
        .map(|bd| {
            let token_ids = last_ngram.is_some_and(|k| bd <= k);
            let layer_inputs = if p.per_layer.is_some_and(|k| bd <= k) {
                HIDDEN
            } else {
                0
            };
            SideData {
                token_ids,
                layer_inputs,
                bytes_per_token: u32::from(token_ids) * 4 + layer_inputs * 4,
            }
        })
        .collect()
}

proptest! {
    #[test]
    fn computed_boundaries_follow_liveness_and_pass_rule_4(p in params()) {
        let f = build(&p);
        let n = f.graph.layers.len();
        let (f, a) = with_computed(f, &none());
        // Only residual streams and side data cross, so every boundary is legal.
        prop_assert_eq!(&a.set.legal, &vec![true; n + 1]);
        prop_assert!(a.illegal.iter().all(Option::is_none));
        prop_assert_eq!(&a.set.side, &expected_side(&p, n));
        prop_assert_eq!(f.validate(), Ok(()));
    }

    #[test]
    fn every_legal_pair_partitions_the_graph(p in params(), at in any::<Index>()) {
        let mut f = build(&p);
        let n = f.graph.layers.len();
        // Make one interior boundary illegal by liveness when the graph has one.
        if n >= 2 {
            let l = at.index(n - 1);
            let att = f.layers[l].att;
            f.graph.layers[l + 1].body.inputs.push(att);
        }
        let (f, a) = with_computed(f, &none());
        if n >= 2 {
            let l = at.index(n - 1);
            prop_assert!(!a.set.legal[l + 1]);
            prop_assert_eq!(a.illegal[l + 1], Some(Illegal::Crossing(f.layers[l].att)));
        }
        prop_assert_eq!(f.validate(), Ok(()));
        check_every_legal_pair(&f)?;
    }
}

// ---- The nano-shaped graph -----------------------------------------------------------------

/// 8 layers: MoE and dense FFNs, two residual streams, n-gram layers 0 and 3 (token IDs cross
/// boundaries 0–3), a per-layer input for layer 5 (it crosses boundaries 0–5), two MTP
/// modules; and one liveness crossing: layer 7 reads layer 6's attention output.
fn nano_shaped() -> Fixture {
    let mut f = build(&Params {
        layers: vec![
            (false, true),
            (true, false),
            (false, false),
            (true, true),
            (false, false),
            (true, false),
            (false, false),
            (true, false),
        ],
        streams: 2,
        mtp: 2,
        per_layer: Some(5),
        legal: 0,
    });
    let att = f.layers[6].att;
    f.graph.layers[7].body.inputs.push(att);
    f
}

const KV: [KvShare; 1] = [KvShare {
    reader: LayerIx(3),
    source: LayerIx(1),
}];
const ATOMIC: [LayerIx; 1] = [LayerIx(5)];

fn nano_facts() -> BoundaryFacts<'static> {
    BoundaryFacts {
        kv_shared: &KV,
        atomic: &ATOMIC,
    }
}

#[test]
fn the_nano_shaped_graph_gets_the_expected_boundaries() {
    let f = nano_shaped();
    let att6 = f.layers[6].att;
    let (f, a) = with_computed(f, &nano_facts());
    assert_eq!(
        a.set.legal,
        vec![true, true, false, false, true, true, false, false, true]
    );
    assert_eq!(a.illegal[2], Some(Illegal::KvShared(KV[0])));
    assert_eq!(a.illegal[3], Some(Illegal::KvShared(KV[0])));
    assert_eq!(a.illegal[6], Some(Illegal::Atomic(LayerIx(5))));
    assert_eq!(a.illegal[7], Some(Illegal::Crossing(att6)));
    // Token IDs (I32, 4 bytes) up to boundary 3; the per-layer input (64 F32, 256 bytes) up
    // to boundary 5.
    let bytes: Vec<u32> = a.set.side.iter().map(|s| s.bytes_per_token).collect();
    assert_eq!(bytes, vec![260, 260, 260, 260, 256, 256, 0, 0, 0]);
    assert_eq!(a.set.side[3].layer_inputs, HIDDEN);
    assert!(a.set.side[3].token_ids && !a.set.side[4].token_ids);
    assert_eq!(f.validate(), Ok(()));
    assert!(
        a.illegal[2]
            .unwrap()
            .to_string()
            .contains("reads the KV of layer 1")
    );
}

#[test]
fn every_legal_pair_on_the_nano_shaped_graph_partitions_it() {
    let (f, _) = with_computed(nano_shaped(), &nano_facts());
    // Interior legal boundaries 1, 4, 5: the single stage, 3 two-stage and 3 three-stage
    // splits.
    assert_eq!(check_every_legal_pair(&f).unwrap(), 7);
    let parts = partition(&f.graph, &[LayerIx(4)]).unwrap();
    assert_eq!(parts[0].programs[0], ProgramId::Prologue);
    assert_eq!(parts[1].programs.last(), Some(&ProgramId::Mtp(1)));
    // Boundary 4 carries the two residual streams and the per-layer input, not token IDs.
    assert!(!parts[1].receives.contains(&f.tok));
    assert_eq!(parts[1].receives.len(), 3);
}

fn stages(starts: &[u16], total: u16) -> Vec<Range<LayerIx>> {
    let mut ends: Vec<u16> = starts[1..].to_vec();
    ends.push(total);
    starts
        .iter()
        .zip(ends)
        .map(|(&s, e)| LayerIx(s)..LayerIx(e))
        .collect()
}

#[test]
fn a_plan_cutting_at_an_illegal_boundary_is_rejected() {
    let (f, _) = with_computed(nano_shaped(), &nano_facts());
    let set = &f.graph.boundaries;
    let check = |starts: &[u16]| check_cuts(set, 2, &stages(starts, 10));
    assert_eq!(check(&[0]), Ok(()));
    assert_eq!(check(&[0, 4, 8]), Ok(()));
    assert_eq!(check(&[0, 1, 5]), Ok(()));
    for (starts, at, cause) in [
        (&[0, 2][..], 2, CutCause::NotLegal),    // KV sharing
        (&[0, 4, 6][..], 6, CutCause::NotLegal), // atomic pair
        (&[0, 7][..], 7, CutCause::NotLegal),    // a local value crosses
        (&[0, 4, 9][..], 9, CutCause::MtpSeparated),
        (&[0, 5, 4][..], 4, CutCause::OutOfRange),
        (&[0, 10][..], 10, CutCause::OutOfRange),
    ] {
        let err = check(starts).unwrap_err();
        assert_eq!(
            err,
            BoundaryIllegal {
                boundary: LayerIx(at),
                cause
            },
            "{starts:?}"
        );
    }
    // Without MTP modules a stage cannot start at the last boundary.
    assert_eq!(
        check_cuts(set, 0, &stages(&[0, 8], 8)).unwrap_err().cause,
        CutCause::OutOfRange
    );
    let err = check(&[0, 2]).unwrap_err();
    assert_eq!(
        (BoundaryIllegal::CODE, BoundaryIllegal::CLASS),
        (5, ErrorClass::Internal)
    );
    assert!(
        err.to_string().starts_with("BoundaryIllegal (Graph 5)"),
        "{err}"
    );
}

#[test]
fn partition_rejects_bad_cuts() {
    let (f, _) = with_computed(nano_shaped(), &nano_facts());
    let cut = |c: &[u16]| {
        let c: Vec<LayerIx> = c.iter().map(|&x| LayerIx(x)).collect();
        partition(&f.graph, &c)
            .map(|_| ())
            .map_err(|e| (e.boundary.0, e.cause))
    };
    assert_eq!(cut(&[2]), Err((2, CutCause::NotLegal)));
    assert_eq!(cut(&[5, 4]), Err((4, CutCause::OutOfRange)));
    assert_eq!(cut(&[0]), Err((0, CutCause::OutOfRange)));
    // Boundary 8 ends the decoder layers; with MTP modules the last stage may hold only the
    // epilogue and the MTP modules. Inside the MTP modules is never a cut.
    assert_eq!(cut(&[8]), Ok(()));
    assert_eq!(cut(&[9]), Err((9, CutCause::MtpSeparated)));
    assert_eq!(cut(&[10]), Err((10, CutCause::OutOfRange)));
    assert_eq!(cut(&[1, 4, 5]), Ok(()));
}

// ---- Family bugs ---------------------------------------------------------------------------

fn small() -> Fixture {
    build(&Params {
        layers: vec![(false, false), (false, false), (false, false)],
        streams: 2,
        mtp: 0,
        per_layer: None,
        legal: 0,
    })
}

#[test]
fn a_local_value_across_an_end_boundary_is_a_family_bug() {
    // With two streams the embedding output is a local value; layer 0 reading it crosses
    // boundary 0.
    let mut f = small();
    let e = f.graph.prologue.nodes[0].outputs[0];
    assert_eq!(f.graph.values[e.0 as usize].kind, ValueKind::Local);
    f.graph.layers[0].body.inputs.push(e);
    assert_eq!(
        compute(&f.graph, &none()),
        Err(BoundaryError::EndBoundaryCrossed {
            boundary: 0,
            value: e
        })
    );
    // The last boundary: the epilogue reads layer 2's attention output.
    let mut f = small();
    let att = f.layers[2].att;
    f.graph.epilogue.inputs.push(att);
    assert_eq!(
        compute(&f.graph, &none()),
        Err(BoundaryError::EndBoundaryCrossed {
            boundary: 3,
            value: att
        })
    );
}

#[test]
fn invalid_facts_are_rejected() {
    let f = small();
    for s in [
        KvShare {
            reader: LayerIx(1),
            source: LayerIx(1),
        },
        KvShare {
            reader: LayerIx(0),
            source: LayerIx(1),
        },
        KvShare {
            reader: LayerIx(3),
            source: LayerIx(0),
        },
    ] {
        let facts = BoundaryFacts {
            kv_shared: &[s],
            atomic: &[],
        };
        assert_eq!(
            compute(&f.graph, &facts),
            Err(BoundaryError::KvShareInvalid(s))
        );
    }
    let facts = BoundaryFacts {
        kv_shared: &[],
        atomic: &[LayerIx(2)],
    };
    assert_eq!(
        compute(&f.graph, &facts),
        Err(BoundaryError::AtomicInvalid(LayerIx(2)))
    );
    let ok = BoundaryFacts {
        kv_shared: &[],
        atomic: &[LayerIx(1)],
    };
    assert_eq!(
        compute(&f.graph, &ok).unwrap().set.legal,
        vec![true, true, false, true]
    );
}

#[test]
fn side_data_needs_a_fixed_size_per_token() {
    let mut f = build(&Params {
        layers: vec![(false, false), (false, false)],
        streams: 1,
        mtp: 0,
        per_layer: Some(1),
        legal: 0,
    });
    let pl = f
        .graph
        .values
        .iter()
        .position(|v| v.kind == ValueKind::LayerInput)
        .unwrap();
    f.graph.values[pl] = Value {
        ty: ValueType {
            dtype: DType::F32,
            dims: smallvec![Dim::Rows, Dim::Ctx],
        },
        kind: ValueKind::LayerInput,
    };
    let v = ValueId(u32::try_from(pl).unwrap());
    assert_eq!(
        compute(&f.graph, &none()),
        Err(BoundaryError::SideDataShape(v))
    );
    assert!(
        BoundaryError::SideDataShape(v)
            .to_string()
            .contains("context-length")
    );
}

#[test]
fn crossing_past_the_last_boundary_is_empty() {
    let f = small();
    assert!(crossing(&f.graph, 4).is_empty());
    // Boundary 1 of a two-stream graph carries both residual streams.
    assert_eq!(crossing(&f.graph, 1).len(), 2);
}

#[test]
fn check_cuts_and_partition_apply_one_rule() {
    for mtp in [0u8, 2] {
        let mut f = nano_shaped();
        if mtp == 0 {
            f.graph.mtp.clear();
        }
        let (f, _) = with_computed(f, &nano_facts());
        let total = 8 + u16::from(mtp);
        for b in 0..=11u16 {
            let plan = check_cuts(
                &f.graph.boundaries,
                u16::from(mtp),
                &[LayerIx(0)..LayerIx(b), LayerIx(b)..LayerIx(total)],
            );
            let parts = partition(&f.graph, &[LayerIx(b)]).map(|_| ());
            assert_eq!(plan, parts, "boundary {b} with {mtp} MTP modules");
        }
    }
}

#[test]
fn a_stage_forwards_side_data_it_does_not_read() {
    // Cuts at 1 and 4: layers 1–3 do not read the per-layer input of layer 5, but stage 1
    // must still receive it and pass it on.
    let (f, _) = with_computed(nano_shaped(), &nano_facts());
    let pl = ValueId(
        u32::try_from(
            f.graph
                .values
                .iter()
                .position(|v| v.kind == ValueKind::LayerInput)
                .unwrap(),
        )
        .unwrap(),
    );
    let parts = partition(&f.graph, &[LayerIx(1), LayerIx(4)]).unwrap();
    let reads_it = |id: &ProgramId| match id {
        ProgramId::Layer(l) => f.graph.layers[usize::from(*l)].body.inputs.contains(&pl),
        _ => false,
    };
    assert!(!parts[1].programs.iter().any(reads_it));
    assert!(parts[1].receives.contains(&pl));
    assert!(parts[1].sends.contains(&pl));
    assert!(parts[2].receives.contains(&pl));
}

#[test]
fn layer_inputs_is_the_width_per_remaining_layer() {
    // Two per-layer inputs of 64 F32 each, for layers 1 and 2: boundaries 0 and 1 carry both,
    // boundary 2 the second only. `layer_inputs` stays 64; the bytes count both.
    let mut f = build(&Params {
        layers: vec![(false, false), (false, false), (false, false)],
        streams: 1,
        mtp: 0,
        per_layer: Some(1),
        legal: 0,
    });
    let tok = f.tok;
    let second = ValueId(u32::try_from(f.graph.values.len()).unwrap());
    f.graph.values.push(Value {
        ty: ValueType {
            dtype: DType::F32,
            dims: smallvec![Dim::Rows, Dim::Const(HIDDEN)],
        },
        kind: ValueKind::LayerInput,
    });
    let mut node = f.graph.prologue.nodes[0].clone();
    node.inputs = smallvec![tok];
    node.outputs = smallvec![second];
    f.graph.prologue.nodes.push(node);
    f.graph.layers[2].body.inputs.push(second);
    let a = compute(&f.graph, &none()).unwrap();
    let got: Vec<(u32, u32)> = a
        .set
        .side
        .iter()
        .map(|s| (s.layer_inputs, s.bytes_per_token))
        .collect();
    assert_eq!(got, vec![(64, 512), (64, 512), (64, 256), (0, 0)]);

    // A wider second input has no single width per remaining layer.
    f.graph.values[second.0 as usize].ty.dims = smallvec![Dim::Rows, Dim::Const(2 * HIDDEN)];
    assert_eq!(
        compute(&f.graph, &none()),
        Err(BoundaryError::LayerInputWidths {
            boundary: 0,
            value: second
        })
    );
}
