// SPDX-License-Identifier: MIT OR Apache-2.0

use proptest::prelude::*;

use super::*;
use crate::plan::PlanEpoch;
use crate::resource::CpuIsa;

/// A stand-in for BLAKE3, which `core` does not link: FNV-1a 64 under four seeds. The BLAKE3
/// digests and their golden values are `syntropy_manifest::identity`'s tests.
fn stand_in(bytes: &[u8]) -> Blake3 {
    let mut out = [0u8; 32];
    for (lane, word) in out.as_chunks_mut::<8>().0.iter_mut().enumerate() {
        let mut h = 0xcbf2_9ce4_8422_2325_u64 ^ lane as u64;
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        word.copy_from_slice(&h.to_le_bytes());
    }
    Blake3(out)
}

fn model(n: u8) -> ModelId {
    ModelId(Blake3([n; 32]))
}

fn groups(ids: &[u32]) -> Vec<TensorGroupId> {
    ids.iter().copied().map(TensorGroupId).collect()
}

fn chunk(n: u8) -> ChunkId {
    ChunkId(Blake3([n; 32]))
}

#[test]
fn the_digest_preimage_is_the_documented_bytes() {
    let bytes = ShardSpec::digest_preimage(model(0xab), &groups(&[7, 2, 7, 0x0102_0304]));
    let mut want = b"syntropy/shard/v1".to_vec();
    want.extend_from_slice(&[0xab; 32]);
    want.extend_from_slice(&3u64.to_le_bytes());
    want.extend_from_slice(&[2, 0, 0, 0, 7, 0, 0, 0, 4, 3, 2, 1]);
    assert_eq!(bytes, want);
    assert_eq!(SHARD_DIGEST_DOMAIN.len(), 17);
}

#[test]
fn a_spec_holds_its_groups_ascending_and_once() {
    let spec = ShardSpec::new_with(
        model(1),
        NodeId([9; 32]),
        groups(&[5, 1, 5, 3]),
        vec![chunk(2), chunk(1)],
        stand_in,
    );
    assert_eq!(spec.groups, groups(&[1, 3, 5]));
    assert_eq!(
        spec.chunks,
        vec![chunk(2), chunk(1)],
        "chunks are kept as given"
    );
    assert_eq!(
        spec.digest,
        stand_in(&ShardSpec::digest_preimage(model(1), &spec.groups))
    );
    assert!(spec.digest_is_consistent(stand_in));
}

#[test]
fn an_inconsistent_spec_is_caught() {
    let spec = ShardSpec::new_with(model(1), NodeId([9; 32]), groups(&[1, 3]), vec![], stand_in);
    let mut tampered = spec.clone();
    tampered.groups.push(TensorGroupId(4));
    assert!(!tampered.digest_is_consistent(stand_in));
    let mut unsorted = spec.clone();
    unsorted.groups.reverse();
    unsorted.digest = stand_in(&ShardSpec::digest_preimage(model(1), &unsorted.groups));
    assert!(
        !unsorted.digest_is_consistent(stand_in),
        "groups must be ascending"
    );
    let mut repeated = spec;
    repeated.groups = groups(&[1, 1, 3]);
    assert!(!repeated.digest_is_consistent(stand_in), "each group once");
}

#[test]
fn layout_version_source_is_zero_and_the_default() {
    assert_eq!(LayoutVersion::SOURCE, LayoutVersion(0));
    assert_eq!(LayoutVersion::default(), LayoutVersion::SOURCE);
}

#[test]
fn the_types_round_trip_through_postcard() {
    let spec = ShardSpec::new_with(
        model(3),
        NodeId([4; 32]),
        groups(&[0, u32::MAX]),
        vec![chunk(5), chunk(6)],
        stand_in,
    );
    let key = PackKey::for_shard(
        &spec,
        DeviceClass::Cpu { isa: CpuIsa::X86V3 },
        LayoutVersion(u16::MAX),
    );
    let bytes = postcard::to_allocvec(&spec).unwrap();
    assert_eq!(postcard::from_bytes::<ShardSpec>(&bytes).unwrap(), spec);
    let bytes = postcard::to_allocvec(&key).unwrap();
    assert_eq!(postcard::from_bytes::<PackKey>(&bytes).unwrap(), key);
}

fn group_ids() -> impl Strategy<Value = Vec<u32>> {
    prop::collection::vec(0u32..64, 0..24)
}

proptest! {
    /// The digest is a function of the model and the set of groups: not their order, not a
    /// repeat, not the node and not the chunks.
    #[test]
    fn the_digest_is_a_pure_function_of_model_and_group_set(
        ids in group_ids(),
        seed in any::<u64>(),
        m in any::<u8>(),
        node_a in any::<[u8; 32]>(),
        node_b in any::<[u8; 32]>(),
        chunks in prop::collection::vec(any::<u8>(), 0..8),
    ) {
        let mut shuffled = ids.clone();
        // A deterministic shuffle from `seed`, plus one repeat.
        let mut state = seed | 1;
        for i in (1..shuffled.len()).rev() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            shuffled.swap(i, (state % (i as u64 + 1)) as usize);
        }
        if let Some(&first) = ids.first() {
            shuffled.push(first);
        }
        let a = ShardSpec::new_with(model(m), NodeId(node_a), groups(&ids), vec![], stand_in);
        let b = ShardSpec::new_with(
            model(m),
            NodeId(node_b),
            groups(&shuffled),
            chunks.into_iter().map(chunk).collect(),
            stand_in,
        );
        prop_assert_eq!(a.digest, b.digest);
        prop_assert_eq!(&a.groups, &b.groups);
        // And twice over the same inputs gives the same spec.
        let again = ShardSpec::new_with(model(m), NodeId(node_a), groups(&ids), vec![], stand_in);
        prop_assert_eq!(a, again);
    }

    /// Different group sets, or different models, write different preimages, so BLAKE3 keeps
    /// them apart.
    #[test]
    fn different_shards_have_different_preimages(
        a in group_ids(),
        b in group_ids(),
        ma in any::<u8>(),
        mb in any::<u8>(),
    ) {
        let set = |ids: &[u32]| {
            let mut ids = ids.to_vec();
            ids.sort_unstable();
            ids.dedup();
            ids
        };
        let same = set(&a) == set(&b) && ma == mb;
        prop_assert_eq!(
            ShardSpec::digest_preimage(model(ma), &groups(&a))
                == ShardSpec::digest_preimage(model(mb), &groups(&b)),
            same
        );
    }

    /// RFC-0001 §5.6: an epoch change that leaves a node's shard unchanged keeps its `PackKey`,
    /// and a change of the shard, class or layout-table version changes it. The epoch is no
    /// input: two plans at different epochs that give the node the same groups, listed in any
    /// order, give the same key.
    #[test]
    fn an_epoch_change_with_an_unchanged_shard_keeps_the_pack_key(
        ids in group_ids(),
        extra in 64u32..128,
        term in any::<u64>(),
        seq in any::<u64>(),
        layout in any::<u16>(),
    ) {
        let class = DeviceClass::Cuda { sm: 89 };
        let plan = |epoch: PlanEpoch, ids: &[u32]| {
            // A plan at `epoch` lists the node's groups; its order is the plan's own.
            let mut listed = groups(ids);
            if epoch.seq % 2 == 1 {
                listed.reverse();
            }
            let spec = ShardSpec::new_with(model(7), NodeId([1; 32]), listed, vec![], stand_in);
            PackKey::for_shard(&spec, class, LayoutVersion(layout))
        };
        let before = PlanEpoch { term, seq };
        let after = PlanEpoch { term: term.wrapping_add(1), seq: seq.wrapping_add(1) };
        let key = plan(before, &ids);
        prop_assert_eq!(key, plan(after, &ids));
        let mut grown = ids.clone();
        grown.push(extra);
        prop_assert_ne!(key, plan(after, &grown));
        let spec = ShardSpec::new_with(model(7), NodeId([1; 32]), groups(&ids), vec![], stand_in);
        prop_assert_ne!(key, PackKey::for_shard(&spec, DeviceClass::Cuda { sm: 90 }, LayoutVersion(layout)));
        prop_assert_ne!(
            key,
            PackKey::for_shard(&spec, class, LayoutVersion(layout.wrapping_add(1)))
        );
    }
}
