// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hash containers with a fixed-key hasher (RFC-0060 §2.4, rule T3). Owner: RFC-0060.
//!
//! Crates reachable from a protocol core or the solver use `BTreeMap` and `BTreeSet`, or
//! [`DetHashMap`] and [`DetHashSet`], never a `HashMap` with `RandomState`. A `DetHashMap` built by
//! the same sequence of inserts iterates in the same order in every process. Rule T4 still holds:
//! nothing whose output reaches a plan, a log entry or a wire frame iterates over a hash container
//! whose order depends on insertion history. Sort at the boundary.
//!
//! [`DetHasher`] is SipHash-1-3 with a fixed all-zero key, implemented here instead of taken from
//! `std`. The standard library does not promise that its default hasher keeps the same algorithm
//! across releases, and the simulator's committed regression seeds replay forever (ADR-0600
//! decision 7), so a toolchain update must not change how a `DetHashMap` iterates. Integers hash
//! as little-endian bytes, and `usize` and `isize` hash as 64-bit values, so a key hashes the same
//! on every host. With a fixed key, the hasher gives no protection against keys chosen to collide.

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasher, Hasher};

/// A `HashMap` with the fixed-key [`DetHasher`]. Build one with `DetHashMap::default()`.
pub type DetHashMap<K, V> = HashMap<K, V, DetState>;

/// A `HashSet` with the fixed-key [`DetHasher`]. Build one with `DetHashSet::default()`.
pub type DetHashSet<T> = HashSet<T, DetState>;

/// Builds [`DetHasher`]s, every one with the same key.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DetState;

impl BuildHasher for DetState {
    type Hasher = DetHasher;

    fn build_hasher(&self) -> DetHasher {
        DetHasher::default()
    }
}

/// SipHash-1-3 with an all-zero key.
#[derive(Clone, Debug)]
pub struct DetHasher(Sip<1, 3>);

impl Default for DetHasher {
    fn default() -> DetHasher {
        DetHasher(Sip::new(0, 0))
    }
}

impl Hasher for DetHasher {
    fn write(&mut self, bytes: &[u8]) {
        self.0.write(bytes);
    }

    fn write_u8(&mut self, n: u8) {
        self.0.write(&[n]);
    }

    fn write_u16(&mut self, n: u16) {
        self.0.write(&n.to_le_bytes());
    }

    fn write_u32(&mut self, n: u32) {
        self.0.write(&n.to_le_bytes());
    }

    fn write_u64(&mut self, n: u64) {
        self.0.write(&n.to_le_bytes());
    }

    fn write_u128(&mut self, n: u128) {
        self.0.write(&n.to_le_bytes());
    }

    fn write_usize(&mut self, n: usize) {
        self.write_u64(n as u64);
    }

    fn write_isize(&mut self, n: isize) {
        self.write_u64(n as i64 as u64);
    }

    fn finish(&self) -> u64 {
        self.0.finish()
    }
}

/// SipHash with `C` compression rounds and `D` finalisation rounds, over a byte stream that may
/// arrive in pieces. Each 8-byte word is read little-endian.
#[derive(Clone, Debug)]
struct Sip<const C: usize, const D: usize> {
    v: [u64; 4],
    /// Bytes not yet in a full word, in the low `pending` bytes, little-endian.
    tail: u64,
    pending: usize,
    /// Total bytes written; its low byte goes into the last block.
    length: u64,
}

impl<const C: usize, const D: usize> Sip<C, D> {
    fn new(k0: u64, k1: u64) -> Sip<C, D> {
        Sip {
            v: [
                k0 ^ 0x736f_6d65_7073_6575,
                k1 ^ 0x646f_7261_6e64_6f6d,
                k0 ^ 0x6c79_6765_6e65_7261,
                k1 ^ 0x7465_6462_7974_6573,
            ],
            tail: 0,
            pending: 0,
            length: 0,
        }
    }

    fn round(v: &mut [u64; 4]) {
        v[0] = v[0].wrapping_add(v[1]);
        v[1] = v[1].rotate_left(13) ^ v[0];
        v[0] = v[0].rotate_left(32);
        v[2] = v[2].wrapping_add(v[3]);
        v[3] = v[3].rotate_left(16) ^ v[2];
        v[0] = v[0].wrapping_add(v[3]);
        v[3] = v[3].rotate_left(21) ^ v[0];
        v[2] = v[2].wrapping_add(v[1]);
        v[1] = v[1].rotate_left(17) ^ v[2];
        v[2] = v[2].rotate_left(32);
    }

    fn compress(v: &mut [u64; 4], word: u64) {
        v[3] ^= word;
        for _ in 0..C {
            Sip::<C, D>::round(v);
        }
        v[0] ^= word;
    }

    fn write(&mut self, mut bytes: &[u8]) {
        self.length = self.length.wrapping_add(bytes.len() as u64);
        // Top up a partial word first.
        while self.pending > 0 && !bytes.is_empty() {
            self.tail |= u64::from(bytes[0]) << (8 * self.pending);
            self.pending += 1;
            bytes = &bytes[1..];
            if self.pending == 8 {
                Sip::<C, D>::compress(&mut self.v, self.tail);
                self.tail = 0;
                self.pending = 0;
            }
        }
        if bytes.is_empty() {
            return;
        }
        // No partial word is left here, so whole words come straight from `bytes`.
        let (words, rest) = bytes.as_chunks::<8>();
        for word in words {
            Sip::<C, D>::compress(&mut self.v, u64::from_le_bytes(*word));
        }
        for (at, byte) in rest.iter().enumerate() {
            self.tail |= u64::from(*byte) << (8 * at);
        }
        self.pending = rest.len();
    }

    fn finish(&self) -> u64 {
        let mut v = self.v;
        let last = (self.length << 56) | self.tail;
        Sip::<C, D>::compress(&mut v, last);
        v[2] ^= 0xff;
        for _ in 0..D {
            Sip::<C, D>::round(&mut v);
        }
        v[0] ^ v[1] ^ v[2] ^ v[3]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SipHash-2-4 with the reference key 00 01 .. 0f.
    fn sip24_reference(message: &[u8]) -> u64 {
        let mut sip = Sip::<2, 4>::new(0x0706_0504_0302_0100, 0x0f0e_0d0c_0b0a_0908);
        sip.write(message);
        sip.finish()
    }

    #[test]
    fn siphash_2_4_matches_the_reference_vectors() {
        // The SipHash paper's worked example: message 00 01 .. 0e.
        let message: Vec<u8> = (0..15).collect();
        assert_eq!(sip24_reference(&message), 0xa129_ca61_49be_45e5);
        // The first vector of the reference implementation's table: the empty message.
        assert_eq!(sip24_reference(&[]), 0x726f_db47_dd0e_0e31);
    }

    /// The deprecated `std::hash::SipHasher` is SipHash-2-4 and is not going to change, so it is
    /// an independent check of the rounds and of byte streams split across calls.
    #[test]
    #[allow(deprecated)]
    fn siphash_2_4_matches_std_on_split_writes() {
        let message: Vec<u8> = (0..=255u8).cycle().take(200).collect();
        for length in [0, 1, 7, 8, 9, 15, 16, 17, 63, 64, 65, 200] {
            for split in [0, 1, 3, 8, 11] {
                let split = split.min(length);
                let (k0, k1) = (0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210);
                let mut ours = Sip::<2, 4>::new(k0, k1);
                ours.write(&message[..split]);
                ours.write(&message[split..length]);
                let mut theirs = std::hash::SipHasher::new_with_keys(k0, k1);
                theirs.write(&message[..length]);
                assert_eq!(
                    ours.finish(),
                    theirs.finish(),
                    "length {length}, split {split}"
                );
            }
        }
    }

    #[test]
    fn writes_split_anywhere_hash_the_same() {
        let message: Vec<u8> = (0..40).collect();
        let whole = {
            let mut hasher = DetHasher::default();
            hasher.write(&message);
            hasher.finish()
        };
        for cuts in [[1, 2], [7, 9], [8, 16], [3, 30], [0, 40]] {
            let mut hasher = DetHasher::default();
            hasher.write(&message[..cuts[0]]);
            hasher.write(&message[cuts[0]..cuts[1]]);
            hasher.write(&message[cuts[1]..]);
            assert_eq!(hasher.finish(), whole, "{cuts:?}");
        }
    }

    /// Pinned values. A change here changes how every `DetHashMap` iterates, and so what a
    /// replayed simulator seed does: it needs a reason in the change that makes it.
    #[test]
    fn the_hash_of_a_key_is_pinned() {
        assert_eq!(DetState.hash_one(0u64), PINNED_ZERO);
        assert_eq!(DetState.hash_one(42u64), PINNED_42);
        assert_eq!(DetState.hash_one("syntropy"), PINNED_STR);
    }

    const PINNED_ZERO: u64 = 0xbd60_acb6_58c7_9e45;
    const PINNED_42: u64 = 0x7b3e_724b_36eb_df51;
    const PINNED_STR: u64 = 0xc309_b18c_81a0_b3a8;

    #[test]
    fn integers_hash_the_same_on_every_host() {
        let as_bytes = |bytes: &[u8]| {
            let mut hasher = DetHasher::default();
            hasher.write(bytes);
            hasher.finish()
        };
        let hashed = |value: &dyn Fn(&mut DetHasher)| {
            let mut hasher = DetHasher::default();
            value(&mut hasher);
            hasher.finish()
        };
        let n = 0x0102_0304_0506_0708_u64;
        assert_eq!(hashed(&|h| h.write_u64(n)), as_bytes(&n.to_le_bytes()));
        assert_eq!(hashed(&|h| h.write_usize(7)), hashed(&|h| h.write_u64(7)));
        assert_eq!(
            hashed(&|h| h.write_isize(-1)),
            hashed(&|h| h.write_u64(u64::MAX))
        );
        assert_eq!(
            hashed(&|h| h.write_u32(0xdead_beef)),
            as_bytes(&[0xef, 0xbe, 0xad, 0xde])
        );
        assert_eq!(hashed(&|h| h.write_u16(0x0102)), as_bytes(&[0x02, 0x01]));
        assert_eq!(hashed(&|h| h.write_u8(9)), as_bytes(&[9]));
        assert_eq!(hashed(&|h| h.write_u128(1)), as_bytes(&1u128.to_le_bytes()));
    }

    #[test]
    fn every_built_hasher_starts_from_the_same_key() {
        let key = ("node", 7u32, [1u8, 2, 3]);
        assert_eq!(DetState.hash_one(key), DetState.hash_one(key));
        let mut first = DetState.build_hasher();
        let mut second = DetState.build_hasher();
        first.write_u64(7);
        second.write_u64(7);
        assert_eq!(first.finish(), second.finish());
    }

    #[test]
    fn the_same_inserts_iterate_in_the_same_order() {
        let build = || {
            let mut map: DetHashMap<u64, u64> = DetHashMap::default();
            for n in 0..1_000u64 {
                map.insert(n.wrapping_mul(0x9e37_79b9_7f4a_7c15), n);
            }
            map.into_iter().collect::<Vec<_>>()
        };
        assert_eq!(build(), build());

        let set = |words: &[&str]| {
            let mut set: DetHashSet<String> = DetHashSet::default();
            set.extend(words.iter().map(|w| (*w).to_owned()));
            set.into_iter().collect::<Vec<_>>()
        };
        let words = ["membership", "election", "log", "transition", "repair"];
        assert_eq!(set(&words), set(&words));
    }

    #[test]
    fn the_containers_behave_as_hash_maps_and_sets() {
        let mut map: DetHashMap<&str, u32> = DetHashMap::default();
        map.insert("a", 1);
        map.insert("b", 2);
        *map.entry("a").or_insert(0) += 10;
        assert_eq!(map.get("a"), Some(&11));
        assert_eq!(map.len(), 2);
        let mut set: DetHashSet<u32> = DetHashSet::default();
        assert!(set.insert(5));
        assert!(!set.insert(5));
        assert!(set.contains(&5));
    }
}
