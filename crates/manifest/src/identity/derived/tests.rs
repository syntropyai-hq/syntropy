// SPDX-License-Identifier: MIT OR Apache-2.0

use proptest::prelude::*;
use syntropy_core::resource::DeviceClass;
use syntropy_core::{PackKey, PlanEpoch};

use super::*;
use crate::chunk::{CHUNK_BYTES, EvidenceForm, chunk_file};
use crate::identity::FileDigest;

fn hex(digest: Blake3) -> String {
    digest.0.iter().map(|b| format!("{b:02x}")).collect()
}

fn cid(n: u8) -> ChunkId {
    ChunkId(Blake3([n; 32]))
}

fn ids(n: u8) -> Vec<ChunkId> {
    (1..=n).map(cid).collect()
}

/// Golden values, computed apart from this crate: by a pure-Python BLAKE3 that matches the
/// published BLAKE3 test vectors, over the module docs' bytes. ID `i` is 32 bytes of value `i`.
#[test]
fn tensor_hash_goldens() {
    let goldens = [
        (
            0,
            "cfde5e141551f5c23fb52cbb512f96563e1ba73b9cbe7249737b0a18674c3ff0",
        ),
        (
            1,
            "404117df021da1f5ccd47fec902549848d11256959c9435fd98b11ef8ea3388d",
        ),
        (
            2,
            "b8a370295a6f797b8833dc51bc1425614adf64e6067ab983ff3efea140517373",
        ),
        (
            3,
            "8014cb2ea80d9bc09b01bb92328534fe6b87db7577e85471c860463f54ae0e6e",
        ),
        (
            4,
            "3ae379e02a5ad5dc3c4d6131ea1accb36658dab17f0b7aba26dcb792487aabb9",
        ),
        (
            5,
            "8d316c43f58a7bb6ec34a6b35674f38f11fa2f27e3d80542859d4be2b2ead99a",
        ),
        (
            7,
            "32e851053506756ae989d1bd6339b0e9a99d111fda1e8eff5bafa289ed6756fe",
        ),
        (
            8,
            "0ca3a0495aed69311497775abc4ffc008675052811c92696f3622ff797e267e3",
        ),
        (
            9,
            "ea27cfc6b0074c7d6bc8adf967f5a009ec5c86d7a776d9a2b1ff5863241f41e4",
        ),
    ];
    for (n, want) in goldens {
        assert_eq!(hex(tensor_hash(&ids(n))), want, "{n} chunk IDs");
    }
}

/// The tree's shape, spelled out for three and five leaves: the left subtree takes the largest
/// power of two below the count, and an odd leaf is never duplicated.
#[test]
fn the_tree_splits_at_the_largest_power_of_two() {
    let leaf = |id: ChunkId| tagged(&[&[LEAF], &id.0.0]);
    let node = |l: Blake3, r: Blake3| tagged(&[&[NODE], &l.0, &r.0]);
    let [a, b, c, d, e] = [cid(1), cid(2), cid(3), cid(4), cid(5)];
    assert_eq!(
        tensor_hash(&[a, b, c]),
        node(node(leaf(a), leaf(b)), leaf(c))
    );
    assert_eq!(
        tensor_hash(&[a, b, c, d, e]),
        node(
            node(node(leaf(a), leaf(b)), node(leaf(c), leaf(d))),
            leaf(e)
        )
    );
    assert_ne!(tensor_hash(&[a, b, c]), tensor_hash(&[a, b, c, c]));
    assert_eq!(TENSOR_HASH_DOMAIN.len(), 23);
}

#[test]
fn shard_digest_goldens() {
    let model = ModelId(Blake3([0xab; 32]));
    let groups = [7, 2, 7, 0x0102_0304].map(TensorGroupId);
    let want = "391f03729b637909170d6b9061b7e9f292ece02b990e466f3f4e7707ee54fa45";
    assert_eq!(hex(shard_digest(model, &groups)), want);
    let spec = shard_spec(model, NodeId([3; 32]), groups.to_vec(), vec![cid(9)]);
    assert_eq!(hex(spec.digest), want);
    assert!(spec.digest_is_consistent(blake3_digest));
    assert_eq!(
        hex(shard_digest(ModelId(Blake3([0x11; 32])), &[])),
        "1bc4545e9b62bad3ecf787ee09c5aea2800a09243517a017af8b5ecd570457b9"
    );
}

#[test]
fn extent_key_goldens() {
    let [a, b] = [cid(1).0, cid(2).0];
    assert_eq!(
        hex(ExtentKey::of(&[a, b], LayoutVersion(3), LayoutVersion(1)).0),
        "c5a7c6d67fd9af71ee5a31723471428176c73a228a03fdd2f9ed21772d6437ce"
    );
    assert_eq!(
        hex(ExtentKey::of(&[b, a], LayoutVersion(3), LayoutVersion(1)).0),
        "5a71f47c6ffa93a37170b0cb05a414235e1e5ab37237ea315dc5e42298da0fa7"
    );
    assert_eq!(
        hex(ExtentKey::of(&[a], LayoutVersion::SOURCE, LayoutVersion(7)).0),
        "0fa8e5298070eaf4eaa4d7ecab7ba1df170d6c99bd1689a1fc1796f7bf91fbea"
    );
    assert_eq!(EXTENT_KEY_DOMAIN.len(), 18);
}

/// A 20 MiB file whose data section starts at 4096, with tensors at 4096 (9 MiB), 9 MiB + 4096
/// (an auxiliary extent of 64 KiB) and 10 MiB, and every chunk committed with ID = its index + 1.
fn table() -> ChunkTable {
    const MIB: u64 = 1 << 20;
    let len = 20 * MIB;
    let chunks = chunk_file(len, 4096, [4096, 9 * MIB + 4096, 10 * MIB]).unwrap();
    let mut table = ChunkTable::new(FileDigest([7; 32]), len, &chunks);
    for (i, record) in table.records.iter_mut().enumerate() {
        record.committed = Some((Blake3([i as u8 + 1; 32]), EvidenceForm::Provisional));
    }
    table
}

#[test]
fn a_table_gives_the_hash_of_the_chunks_that_start_in_each_extent() {
    const MIB: u64 = 1 << 20;
    let table = table();
    let starts: Vec<u64> = table.records.iter().map(|r| r.chunk.offset).collect();
    // The header chunk, then the 9 MiB tensor in 4 MiB steps, the 64 KiB auxiliary extent, then
    // the last 10 MiB.
    assert_eq!(
        starts,
        [
            0,
            4096,
            4096 + CHUNK_BYTES,
            4096 + 2 * CHUNK_BYTES,
            9 * MIB + 4096,
            10 * MIB,
            14 * MIB,
            18 * MIB
        ]
    );
    let id = |i: u8| ChunkId(Blake3([i + 1; 32]));
    let weight = 4096..9 * MIB + 4096;
    let aux = 9 * MIB + 4096..9 * MIB + 4096 + 65536;
    assert_eq!(
        table.tensor_hash(std::slice::from_ref(&weight)),
        Ok(Some(tensor_hash(&[id(1), id(2), id(3)])))
    );
    // The auxiliary extent's chunk runs to the next tensor, padding included.
    assert_eq!(
        table.tensor_hash(&[weight.clone(), aux.clone()]),
        Ok(Some(tensor_hash(&[id(1), id(2), id(3), id(4)])))
    );
    // Order is the caller's: source, then auxiliary.
    assert_eq!(
        table.tensor_hash(&[aux.clone(), weight.clone()]),
        Ok(Some(tensor_hash(&[id(4), id(1), id(2), id(3)])))
    );
    // An empty extent covers nothing.
    assert_eq!(
        table.tensor_hash(&[10 * MIB..10 * MIB, weight.clone()]),
        table.tensor_hash(std::slice::from_ref(&weight))
    );
    assert_eq!(table.tensor_hash(&[]), Ok(Some(tensor_hash(&[]))));
}

#[test]
fn the_hash_is_unset_until_every_chunk_is_committed() {
    const MIB: u64 = 1 << 20;
    let mut table = table();
    table.records[2].committed = None;
    assert_eq!(
        table.tensor_hash(std::slice::from_ref(&(4096..9 * MIB + 4096))),
        Ok(None)
    );
    // A tensor whose chunks are all committed still has its hash.
    assert!(
        table
            .tensor_hash(std::slice::from_ref(&(10 * MIB..20 * MIB)))
            .unwrap()
            .is_some()
    );
}

#[test]
fn extents_that_do_not_fit_the_table_are_refused() {
    const MIB: u64 = 1 << 20;
    let table = table();
    let err = table
        .tensor_hash(std::slice::from_ref(&(4097..9 * MIB)))
        .unwrap_err();
    assert_eq!(err, TensorHashError::NotAChunkStart { offset: 4097 });
    assert_eq!(err.code(), 6);
    assert_eq!(
        table.tensor_hash(std::slice::from_ref(&(10 * MIB..20 * MIB + 1))),
        Err(TensorHashError::PastEnd {
            end: 20 * MIB + 1,
            file_len: 20 * MIB
        })
    );
}

fn chunk_ids() -> impl Strategy<Value = Vec<ChunkId>> {
    prop::collection::vec(any::<[u8; 32]>().prop_map(|b| ChunkId(Blake3(b))), 0..40)
}

proptest! {
    /// Each identity is a pure function of its inputs: the same inputs give the same value, and
    /// a change to any input the identity covers changes it.
    #[test]
    fn tensor_hash_is_pure_and_commits_to_every_id_and_its_place(
        list in chunk_ids(),
        at in any::<prop::sample::Index>(),
        flip in 0u8..32,
        extra in any::<[u8; 32]>(),
    ) {
        prop_assert_eq!(tensor_hash(&list), tensor_hash(&list.clone()));
        let mut longer = list.clone();
        longer.push(ChunkId(Blake3(extra)));
        prop_assert_ne!(tensor_hash(&list), tensor_hash(&longer));
        if !list.is_empty() {
            let i = at.index(list.len());
            let mut changed = list.clone();
            changed[i].0.0[usize::from(flip)] ^= 1;
            prop_assert_ne!(tensor_hash(&list), tensor_hash(&changed));
            let j = (i + 1) % list.len();
            if list[i] != list[j] {
                let mut swapped = list.clone();
                swapped.swap(i, j);
                prop_assert_ne!(tensor_hash(&list), tensor_hash(&swapped));
            }
        }
    }

    #[test]
    fn extent_key_is_pure_and_commits_to_hashes_layout_and_version(
        hashes in prop::collection::vec(any::<[u8; 32]>().prop_map(Blake3), 0..8),
        layout in any::<u16>(),
        version in any::<u16>(),
        extra in any::<[u8; 32]>(),
    ) {
        let key = ExtentKey::of(&hashes, LayoutVersion(layout), LayoutVersion(version));
        prop_assert_eq!(key, ExtentKey::of(&hashes.clone(), LayoutVersion(layout), LayoutVersion(version)));
        prop_assert_ne!(key, ExtentKey::of(&hashes, LayoutVersion(layout.wrapping_add(1)), LayoutVersion(version)));
        prop_assert_ne!(key, ExtentKey::of(&hashes, LayoutVersion(layout), LayoutVersion(version.wrapping_add(1))));
        // The layout ID and the version are not interchangeable.
        if layout != version {
            prop_assert_ne!(key, ExtentKey::of(&hashes, LayoutVersion(version), LayoutVersion(layout)));
        }
        let mut more = hashes.clone();
        more.push(Blake3(extra));
        prop_assert_ne!(key, ExtentKey::of(&more, LayoutVersion(layout), LayoutVersion(version)));
    }

    /// The digest is BLAKE3 over `core`'s preimage, and the `PackKey` built on it survives an
    /// epoch change that leaves the shard's groups as they were, in whatever order the new plan
    /// lists them; a different group set gives a different key.
    #[test]
    fn an_epoch_change_with_an_unchanged_shard_keeps_the_pack_key(
        groups in prop::collection::vec(0u32..512, 0..32),
        extra in 512u32..1024,
        term in any::<u64>(),
        seq in any::<u64>(),
    ) {
        let model = ModelId(Blake3([5; 32]));
        let class = DeviceClass::Cuda { sm: 120 };
        let key_at = |epoch: PlanEpoch, ids: &[u32]| {
            let mut listed: Vec<TensorGroupId> = ids.iter().copied().map(TensorGroupId).collect();
            if epoch.seq % 2 == 1 {
                listed.reverse();
            }
            let spec = shard_spec(model, NodeId([2; 32]), listed, vec![]);
            prop_assert_eq!(spec.digest, shard_digest(model, &spec.groups));
            Ok(PackKey::for_shard(&spec, class, LayoutVersion(1)))
        };
        let before = PlanEpoch { term, seq };
        let after = PlanEpoch { term: term.wrapping_add(1), seq: seq.wrapping_add(1) };
        let key = key_at(before, &groups)?;
        prop_assert_eq!(key, key_at(after, &groups)?);
        let mut grown = groups.clone();
        grown.push(extra);
        prop_assert_ne!(key, key_at(after, &grown)?);
    }
}
