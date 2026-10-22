// SPDX-License-Identifier: MIT OR Apache-2.0

//! The commit-then-reveal fields of LAN pairing and the codes both sides derive from them
//! (RFC-0030 §6.2, §6.7; RFC-0034 §6.5; ADR-0302 revision of 2026-10-20 item 1).
//!
//! The exchange runs on `syntropy/join/1` after TLS has authenticated both node keys. Its pairing
//! fields travel in this order:
//!
//! | Step | Message | Field | Bytes |
//! |---|---|---|---|
//! | 1 | joiner → keyholder `PairRequest` | `commit` = [`Commitment::new`]`(joiner, nonce_c)` | 32 |
//! | 2 | keyholder → joiner `PairChallenge` | `nonce_k` | 16 |
//! | 3 | joiner → keyholder `PairReveal` | `nonce_c`, sent only after `PairChallenge` has arrived | 16 |
//!
//! The keyholder checks the reveal with [`Commitment::opens`] and ends the pairing with
//! `Rejected { Denied }` if it fails. Both sides then show the [`Sas`] of [`SasInputs`]. Neither
//! side picks its nonce after learning the other's, so an attacker relaying between a joiner and
//! a keyholder makes the two codes match with probability about 10⁻⁶ per attempt.
//!
//! DIST-E5-T14 froze these fields and derivations for DIST-E5-T03, which builds the messages,
//! their other fields and their wire encoding. The golden vectors are in
//! `crates/cluster/tests/pairing/vectors.txt`.

use std::fmt;

use syntropy_core::NodeId;

/// The domain string of the commitment.
pub const COMMIT_CONTEXT: &[u8] = b"syntropy-sas-commit-v1";

/// The domain string of the SAS.
pub const SAS_CONTEXT: &[u8] = b"syntropy-sas-v1";

/// The number of distinct codes: six decimal digits.
pub const SAS_CODES: u32 = 1_000_000;

/// A pairing nonce: 16 bytes from the CSPRNG, drawn fresh by each side for each pairing.
pub type Nonce = [u8; 16];

/// `PairRequest.commit`: `BLAKE3("syntropy-sas-commit-v1" ‖ joiner ‖ nonce_c)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Commitment(pub [u8; 32]);

impl Commitment {
    /// The joiner's commitment to `nonce_c`. `joiner` is its node ID, the key TLS authenticated.
    pub fn new(joiner: &NodeId, nonce_c: &Nonce) -> Commitment {
        let mut hasher = blake3::Hasher::new();
        hasher.update(COMMIT_CONTEXT);
        hasher.update(&joiner.0);
        hasher.update(nonce_c);
        Commitment(*hasher.finalize().as_bytes())
    }

    /// The keyholder's check of `PairReveal`: whether `nonce_c` opens this commitment for the
    /// joiner it authenticated. Every input is public once revealed, so the comparison need not
    /// run in constant time.
    pub fn opens(&self, joiner: &NodeId, nonce_c: &Nonce) -> bool {
        Commitment::new(joiner, nonce_c) == *self
    }
}

/// Everything the SAS covers, named by role so the two node IDs cannot be swapped by accident.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SasInputs {
    /// The cluster's ID, its Ed25519 public key (RFC-0034 §4), as `PairChallenge` carries it.
    /// DIST-E2-T06 gives it a type.
    pub cluster_id: [u8; 32],
    /// The joining node's ID.
    pub joiner: NodeId,
    /// The keyholder's node ID.
    pub keyholder: NodeId,
    /// The joiner's nonce, from `PairReveal`.
    pub nonce_c: Nonce,
    /// The keyholder's nonce, from `PairChallenge`.
    pub nonce_k: Nonce,
}

impl SasInputs {
    /// `BLAKE3("syntropy-sas-v1" ‖ cluster_id ‖ joiner ‖ keyholder ‖ nonce_c ‖ nonce_k)`. Every
    /// field has a fixed length, so the concatenation is unambiguous.
    pub fn digest(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(SAS_CONTEXT);
        hasher.update(&self.cluster_id);
        hasher.update(&self.joiner.0);
        hasher.update(&self.keyholder.0);
        hasher.update(&self.nonce_c);
        hasher.update(&self.nonce_k);
        *hasher.finalize().as_bytes()
    }

    /// The code both sides show.
    pub fn sas(&self) -> Sas {
        Sas::from_digest(&self.digest())
    }
}

/// The short authentication string: a number below 10⁶, shown as two groups of three digits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Sas(u32);

impl Sas {
    /// `u32_le(digest[0..4]) mod 10⁶`. The reduction favours the 967,296 lowest codes by one
    /// part in 4,294: each code has probability between 4,294 / 2³² and 4,295 / 2³², both
    /// 10⁻⁶ to within 0.03 %.
    pub fn from_digest(digest: &[u8; 32]) -> Sas {
        let word = u32::from_le_bytes([digest[0], digest[1], digest[2], digest[3]]);
        Sas(word % SAS_CODES)
    }

    /// The code as a number, `0..1_000_000`.
    pub fn value(self) -> u32 {
        self.0
    }
}

/// `482 917`; leading zeros are kept, so every code shows six digits.
impl fmt::Display for Sas {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:03} {:03}", self.0 / 1000, self.0 % 1000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs() -> SasInputs {
        SasInputs {
            cluster_id: [0x11; 32],
            joiner: NodeId([0x22; 32]),
            keyholder: NodeId([0x33; 32]),
            nonce_c: [0x44; 16],
            nonce_k: [0x55; 16],
        }
    }

    /// Flips each bit of each input in turn: 1,024 variants.
    fn bit_flips(base: SasInputs) -> Vec<SasInputs> {
        let mut out = Vec::new();
        for bit in 0..256 {
            let mut v = base;
            v.cluster_id[bit / 8] ^= 1 << (bit % 8);
            out.push(v);
            let mut v = base;
            v.joiner.0[bit / 8] ^= 1 << (bit % 8);
            out.push(v);
            let mut v = base;
            v.keyholder.0[bit / 8] ^= 1 << (bit % 8);
            out.push(v);
        }
        for bit in 0..128 {
            let mut v = base;
            v.nonce_c[bit / 8] ^= 1 << (bit % 8);
            out.push(v);
            let mut v = base;
            v.nonce_k[bit / 8] ^= 1 << (bit % 8);
            out.push(v);
        }
        out
    }

    #[test]
    fn any_change_to_a_key_or_a_nonce_changes_the_sas() {
        let base = inputs();
        let flips = bit_flips(base);
        assert_eq!(flips.len(), 3 * 256 + 2 * 128);
        for v in flips {
            assert_ne!(v.digest(), base.digest(), "{v:?}");
            // A different digest can still give the same code, with probability 10⁻⁶; for these
            // fixed inputs none does.
            assert_ne!(v.sas(), base.sas(), "{v:?}");
        }
    }

    #[test]
    fn swapping_the_roles_changes_the_sas() {
        let base = inputs();
        let swapped = SasInputs {
            joiner: base.keyholder,
            keyholder: base.joiner,
            ..base
        };
        assert_ne!(swapped.sas(), base.sas());
        let swapped_nonces = SasInputs {
            nonce_c: base.nonce_k,
            nonce_k: base.nonce_c,
            ..base
        };
        assert_ne!(swapped_nonces.sas(), base.sas());
    }

    #[test]
    fn a_commitment_opens_only_with_its_nonce_and_joiner() {
        let joiner = NodeId([7; 32]);
        let nonce = [9; 16];
        let commit = Commitment::new(&joiner, &nonce);
        assert!(commit.opens(&joiner, &nonce));
        let mut other_nonce = nonce;
        other_nonce[15] ^= 1;
        assert!(!commit.opens(&joiner, &other_nonce));
        let mut other_joiner = joiner;
        other_joiner.0[0] ^= 1;
        assert!(!commit.opens(&other_joiner, &nonce));
    }

    #[test]
    fn the_code_is_the_low_word_mod_a_million_shown_as_six_digits() {
        let mut digest = [0xEE; 32];
        digest[..4].copy_from_slice(&1_000_123u32.to_le_bytes());
        assert_eq!(Sas::from_digest(&digest).value(), 123);
        assert_eq!(Sas::from_digest(&digest).to_string(), "000 123");
        digest[..4].copy_from_slice(&u32::MAX.to_le_bytes());
        // 4,294,967,295 mod 10⁶.
        assert_eq!(Sas::from_digest(&digest).value(), 967_295);
        assert_eq!(Sas::from_digest(&digest).to_string(), "967 295");
        digest[..4].copy_from_slice(&482_917u32.to_le_bytes());
        assert_eq!(Sas::from_digest(&digest).to_string(), "482 917");
    }
}
