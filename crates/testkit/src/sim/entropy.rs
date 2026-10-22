// SPDX-License-Identifier: MIT OR Apache-2.0

//! The simulator's seeded entropy (RFC-0060 §2.1 rules T1 and T2, §8.2).
//!
//! A [`SimEntropy`] is a ChaCha20 keystream: 20 rounds, the 32-byte key built from a `u64` seed,
//! a 64-bit block counter starting at zero and a 64-bit nonce of zero. Its bytes are defined by
//! the cipher, not by the host, so a seed gives the same bytes on every platform and in every
//! process. `tests/sim_entropy.rs` holds golden vectors for it.
//!
//! [`Entropy::fill`] hands out the keystream in order, however the calls split it: two fills of
//! 3 and 5 bytes take the same bytes as one fill of 8. That keeps a stream stable when the code
//! that reads it changes how many bytes it asks for at a time, as long as it reads them in the
//! same order.
//!
//! Rule T2 is why this type lives here. Production key material comes from `OsEntropy`, and the
//! simulator substitutes this stream. `testkit` is a dev-dependency only (RFC-0001 §4.2 rule 9),
//! so no production build can reach this type: `tests/dev_only.rs` checks that it cannot be named
//! from a crate's non-test code.

use std::fmt;
use std::sync::{Mutex, PoisonError};

use chacha20::ChaCha20Rng;
use chacha20::rand_core::{Rng, SeedableRng};
use syntropy_core::Entropy;

/// Bytes in one ChaCha20 block.
const BLOCK: usize = 64;

/// A seeded ChaCha20 stream, the simulator's only source of randomness.
pub struct SimEntropy {
    seed: u64,
    stream: Mutex<Stream>,
}

/// The keystream and the read position in its current block.
struct Stream {
    rng: ChaCha20Rng,
    block: [u8; BLOCK],
    /// Bytes of `block` already handed out; `BLOCK` when the next fill needs a new block.
    used: usize,
}

impl SimEntropy {
    /// The stream for `seed`. Its ChaCha20 key is `seed` as eight little-endian bytes followed by
    /// 24 zero bytes, so seed 0 is the all-zero key of RFC 8439 Appendix A.1.
    pub fn new(seed: u64) -> SimEntropy {
        let mut key = [0u8; 32];
        key[..8].copy_from_slice(&seed.to_le_bytes());
        SimEntropy {
            seed,
            stream: Mutex::new(Stream {
                rng: ChaCha20Rng::from_seed(key),
                block: [0u8; BLOCK],
                used: BLOCK,
            }),
        }
    }

    /// The seed this stream was built from.
    pub fn seed(&self) -> u64 {
        self.seed
    }
}

impl Stream {
    fn fill(&mut self, mut buf: &mut [u8]) {
        while !buf.is_empty() {
            if self.used == BLOCK {
                // Whole blocks only: the generator drops the unread bytes of a partly read
                // 32-bit word, which would make the stream depend on how fills are split.
                self.rng.fill_bytes(&mut self.block);
                self.used = 0;
            }
            let take = buf.len().min(BLOCK - self.used);
            let (head, rest) = buf.split_at_mut(take);
            head.copy_from_slice(&self.block[self.used..self.used + take]);
            self.used += take;
            buf = rest;
        }
    }
}

impl Entropy for SimEntropy {
    fn fill(&self, buf: &mut [u8]) {
        // Nothing in `Stream::fill` panics, so a poisoned lock cannot hide a half-updated stream.
        self.stream
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .fill(buf);
    }
}

impl fmt::Debug for SimEntropy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SimEntropy")
            .field("seed", &self.seed)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `len` bytes of `seed`'s stream, read in one fill.
    fn read(seed: u64, len: usize) -> Vec<u8> {
        let mut bytes = vec![0u8; len];
        SimEntropy::new(seed).fill(&mut bytes);
        bytes
    }

    #[test]
    fn the_stream_does_not_depend_on_how_fills_split_it() {
        let whole = read(9, 700);
        // Splits that start, end and cross 4-byte words, 64-byte blocks and the generator's
        // 256-byte buffer.
        for sizes in [
            &[
                1usize, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 371,
            ][..],
            &[3, 61, 64, 128, 1, 255, 188],
            &[64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 60],
            &[0, 700, 0],
        ] {
            assert_eq!(sizes.iter().sum::<usize>(), 700, "{sizes:?}");
            let entropy = SimEntropy::new(9);
            let mut split = Vec::new();
            for &size in sizes {
                let mut part = vec![0u8; size];
                entropy.fill(&mut part);
                split.extend_from_slice(&part);
            }
            assert_eq!(split, whole, "{sizes:?}");
        }
    }

    #[test]
    fn u64_reads_the_next_eight_bytes_little_endian() {
        let bytes = read(4, 24);
        let entropy = SimEntropy::new(4);
        let mut first = [0u8; 3];
        entropy.fill(&mut first);
        let mut expected = [0u8; 8];
        expected.copy_from_slice(&bytes[3..11]);
        assert_eq!(entropy.u64(), u64::from_le_bytes(expected));
        expected.copy_from_slice(&bytes[11..19]);
        assert_eq!(entropy.u64(), u64::from_le_bytes(expected));
    }

    #[test]
    fn the_same_seed_gives_the_same_stream() {
        assert_eq!(read(77, 300), read(77, 300));
    }

    #[test]
    fn different_seeds_give_different_streams() {
        let streams = [0, 1, 2, 1 << 8, 1 << 56, u64::MAX].map(|seed| read(seed, 64));
        for (i, a) in streams.iter().enumerate() {
            for b in &streams[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }

    #[test]
    fn a_source_is_shared_as_a_trait_object() {
        fn send_sync<T: Send + Sync>() {}
        send_sync::<SimEntropy>();
        let expected = read(5, 16);
        let entropy: std::sync::Arc<dyn Entropy> = std::sync::Arc::new(SimEntropy::new(5));
        let mut first = [0u8; 8];
        entropy.fill(&mut first);
        let second = std::thread::spawn({
            let entropy = std::sync::Arc::clone(&entropy);
            move || entropy.u64()
        })
        .join()
        .unwrap();
        assert_eq!(first[..], expected[..8]);
        assert_eq!(second.to_le_bytes()[..], expected[8..]);
    }

    #[test]
    fn debug_shows_the_seed_and_not_the_stream() {
        let text = format!("{:?}", SimEntropy::new(12));
        assert_eq!(text, "SimEntropy { seed: 12, .. }");
        assert_eq!(SimEntropy::new(12).seed(), 12);
    }
}
