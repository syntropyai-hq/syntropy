// SPDX-License-Identifier: MIT OR Apache-2.0

//! DIST-E5-T14: the commit-then-reveal pairing exchange, modelled between a joiner, an attacker
//! relay and a keyholder, and the golden vectors of its fields and SAS (RFC-0030 §6.2, §6.7;
//! RFC-0034 §6.5 and Testing; ADR-0302 revision of 2026-10-20 item 1).
//!
//! The statistical runs are seeded, so each gives the same result on every run. Against the
//! committed exchange the attacker plays the strategies of [`model::Order`]: the two that reveal
//! honestly alternate over 10⁷ attempts, and the one that grinds a cheating reveal runs 20 times.
//! Against the first draft it plays keyholder-first in 20 trials. The default tests run a prefix
//! of each: 10⁴ honest attempts, one cheating attempt and one first-draft trial. The full runs are
//! ignored here and run release-optimised (RFC-0060 §1.3's `test-release` profile) by
//! `cargo xtask test pairing-grind`, because the dev profile builds `blake3` without
//! optimisation: one hash takes about 12.7 µs there and 0.2 µs release-optimised.
//!
//! Runtimes [measured, 2026-10-22, AMD Ryzen 9 5950X under WSL2, toolchain 1.99.1, each test
//! on one thread]: in `test-release`, 4.4 s for the 20 first-draft trials and 14.5 s for the
//! committed run, whose 20 cheating reveals spend 2.5 × 10⁷ hashes besides the 10⁷ honest
//! attempts; 14.2 s together, because the harness runs the two in parallel. In the dev profile
//! the smoke runs take 4.9 s: the first-draft trial's 370,734 hashes and the cheating reveal's
//! 247,508, in parallel.

mod model;
mod spec;

use std::path::PathBuf;

use model::{Denied, Exchange, Joiner, Keyholder, Order, Reveal, SplitMix64, attempt};
use syntropy_cluster::pairing::{Commitment, SasInputs};
use syntropy_core::NodeId;

/// "Within 10⁷ hashes": the attacker's budget per nonce.
const BUDGET: u64 = 10_000_000;
/// "At most 25 times in 10⁷ attempts".
const ATTEMPTS: u64 = 10_000_000;
const BOUND: u64 = 25;
/// Cheating-reveal attempts in the full run.
const CHEATS: u64 = 20;
/// First-draft trials in the full run.
const TRIALS: u64 = 20;
/// The smoke runs: a prefix of each full run.
const SMOKE_ATTEMPTS: u64 = 10_000;
const SMOKE_BOUND: u64 = 1;
const SMOKE_CHEATS: u64 = 1;
const SMOKE_TRIALS: u64 = 1;

const COMMITTED_SEED: u64 = 0xC0DE_0000_0000_0001;
const CHEAT_SEED: u64 = 0xC4EA_0000_0000_0001;
const FIRST_DRAFT_SEED: u64 = 0xD4AF_0000_0000_0001;

/// P(X > k) for X ~ Poisson(λ), summed term by term above k.
fn poisson_tail_above(lambda: f64, k: u64) -> f64 {
    let mut term = (-lambda).exp();
    for i in 1..=k + 1 {
        term *= lambda / i as f64;
    }
    let mut tail = 0.0;
    let mut i = k + 1;
    while term > 1e-300 && i < k + 10_000 {
        tail += term;
        i += 1;
        term *= lambda / i as f64;
    }
    tail
}

/// The committed exchange: `attempts` relayed pairings alternating the two honest-reveal
/// strategies, then `cheats` pairings with a cheating reveal. Returns the matches, the denials
/// and the hashes each cheating reveal took to find its match.
fn committed_outcomes(attempts: u64, cheats: u64) -> (u64, u64, Vec<u64>) {
    let mut matches = 0;
    let mut denials = 0;
    let mut rng = SplitMix64(COMMITTED_SEED);
    for i in 0..attempts {
        let order = if i % 2 == 0 {
            Order::KeyholderFirst
        } else {
            Order::JoinerFirst
        };
        let outcome = attempt(Exchange::Committed, order, BUDGET, &mut rng);
        // Given what each of these orders shows it, the attacker never holds a nonce whose code
        // it can compute while the other code is fixed, so it has nothing to grind.
        assert_eq!(
            outcome.hashes, 0,
            "attempt {i} ({order:?}): the attacker ground"
        );
        assert!(!outcome.denied, "attempt {i} ({order:?}): denied");
        matches += u64::from(outcome.matched);
    }
    let mut cheat_hashes = Vec::new();
    let mut rng = SplitMix64(CHEAT_SEED);
    for i in 0..cheats {
        let outcome = attempt(Exchange::Committed, Order::CheatingReveal, BUDGET, &mut rng);
        // The attacker found a reveal whose code matches, so only the commitment check stops it.
        assert!(
            outcome.hit,
            "cheat {i}: no matching reveal within {BUDGET} hashes"
        );
        matches += u64::from(outcome.matched);
        denials += u64::from(outcome.denied);
        cheat_hashes.push(outcome.hashes);
    }
    (matches, denials, cheat_hashes)
}

/// `trials` relayed pairings against the first draft, in the keyholder-first order. Each must
/// match within the budget; returns the hashes each took.
fn first_draft_hashes(trials: u64) -> Vec<u64> {
    let mut rng = SplitMix64(FIRST_DRAFT_SEED);
    (0..trials)
        .map(|t| {
            let outcome = attempt(
                Exchange::FirstDraft,
                Order::KeyholderFirst,
                BUDGET,
                &mut rng,
            );
            assert!(!outcome.denied, "trial {t}: denied");
            assert!(
                outcome.matched,
                "trial {t}: no match within {BUDGET} hashes"
            );
            assert!(outcome.hashes <= BUDGET);
            outcome.hashes
        })
        .collect()
}

#[test]
fn the_bounds_follow_from_poisson_tails_below_1e_4() {
    // The row's derivation: 10⁷ attempts at 10⁻⁶ is a Poisson mean of 10, and P(X > 25) < 10⁻⁴.
    let full = poisson_tail_above(ATTEMPTS as f64 * 1e-6, BOUND);
    assert!(full < 1e-4, "{full}");
    assert!(full > 1e-5, "{full}");
    // The smoke prefix: mean 0.01, P(X > 1) < 10⁻⁴.
    let smoke = poisson_tail_above(SMOKE_ATTEMPTS as f64 * 1e-6, SMOKE_BOUND);
    assert!(smoke < 1e-4, "{smoke}");
    // A first-draft trial, or a cheating reveal, needs more than 10⁷ hashes with probability
    // (1 − 10⁻⁶)^10⁷ ≈ e⁻¹⁰.
    let miss = (1.0f64 - 1e-6).powf(BUDGET as f64);
    assert!(
        miss < 5e-5 && (TRIALS + CHEATS) as f64 * miss < 2e-3,
        "{miss}"
    );
}

#[test]
fn smoke_the_first_draft_is_ground() {
    let hashes = first_draft_hashes(SMOKE_TRIALS);
    eprintln!("first draft, smoke: hashes to match per trial {hashes:?}");
}

#[test]
fn smoke_the_committed_exchange_is_not_ground() {
    let (matches, denials, cheat_hashes) = committed_outcomes(SMOKE_ATTEMPTS, SMOKE_CHEATS);
    eprintln!(
        "committed, smoke: {matches} matches in {SMOKE_ATTEMPTS} honest attempts and \
         {SMOKE_CHEATS} cheating reveals, {denials} denied; hashes per cheat {cheat_hashes:?}"
    );
    assert!(matches <= SMOKE_BOUND, "{matches} matches");
    assert_eq!(denials, SMOKE_CHEATS, "a cheating reveal was accepted");
}

#[test]
#[ignore = "full run, 4.4 s release-optimised [measured]: cargo xtask test pairing-grind"]
fn the_first_draft_is_ground_within_ten_million_hashes_in_every_trial() {
    let hashes = first_draft_hashes(TRIALS);
    eprintln!(
        "first draft: {TRIALS} trials, hashes to match per trial {hashes:?}, most {}",
        hashes.iter().max().unwrap()
    );
}

#[test]
#[ignore = "full run, 14.5 s release-optimised [measured]: cargo xtask test pairing-grind"]
fn the_committed_exchange_matches_at_most_25_times_in_ten_million_attempts() {
    let (matches, denials, cheat_hashes) = committed_outcomes(ATTEMPTS, CHEATS);
    eprintln!(
        "committed: {matches} matches in {ATTEMPTS} honest attempts and {CHEATS} cheating \
         reveals, {denials} denied; hashes per cheat {cheat_hashes:?}"
    );
    assert!(matches <= BOUND, "{matches} matches");
    assert_eq!(denials, CHEATS, "a cheating reveal was accepted");
}

#[test]
fn only_the_keyholder_first_order_grinds_the_first_draft() {
    let mut rng = SplitMix64(7);
    for order in [Order::JoinerFirst, Order::CheatingReveal] {
        let outcome = attempt(Exchange::FirstDraft, order, BUDGET, &mut rng);
        assert_eq!(outcome.hashes, 0, "{order:?}");
        assert!(!outcome.denied, "{order:?}");
    }
}

/// N pairs directly with K, with no relay.
fn honest(exchange: Exchange, v: &spec::Vector) -> (Joiner, Keyholder) {
    let mut joiner = Joiner::new(exchange, NodeId(v.joiner), v.nonce_c);
    let mut keyholder = Keyholder::new(exchange, NodeId(v.keyholder), v.cluster_id, v.nonce_k);
    let challenge = keyholder.on_request(joiner.id, joiner.request()).unwrap();
    let reveal = joiner.on_challenge(keyholder.id, challenge);
    if exchange == Exchange::Committed {
        keyholder.on_reveal(joiner.id, reveal.unwrap()).unwrap();
    } else {
        assert!(reveal.is_none());
    }
    (joiner, keyholder)
}

#[test]
fn an_honest_pairing_shows_the_vectors_code_on_both_sides() {
    for v in spec::vectors() {
        for exchange in [Exchange::FirstDraft, Exchange::Committed] {
            let (joiner, keyholder) = honest(exchange, &v);
            let code = joiner.sas().unwrap();
            assert_eq!(keyholder.sas(), Some(code), "{} {exchange:?}", v.name);
            assert_eq!(code.value(), spec::sas(&v), "{} {exchange:?}", v.name);
        }
    }
}

#[test]
fn a_reveal_that_does_not_open_its_commitment_ends_the_pairing() {
    let v = &spec::vectors()[1];
    let wrong_nonce = |keyholder: &mut Keyholder, joiner: &Joiner| {
        let mut nonce_c = v.nonce_c;
        nonce_c[0] ^= 1;
        keyholder.on_reveal(joiner.id, Reveal { nonce_c })
    };
    let wrong_peer = |keyholder: &mut Keyholder, _: &Joiner| {
        keyholder.on_reveal(NodeId([0xAB; 32]), Reveal { nonce_c: v.nonce_c })
    };
    for tamper in [
        &wrong_nonce as &dyn Fn(&mut Keyholder, &Joiner) -> _,
        &wrong_peer,
    ] {
        let mut joiner = Joiner::new(Exchange::Committed, NodeId(v.joiner), v.nonce_c);
        let mut keyholder = Keyholder::new(
            Exchange::Committed,
            NodeId(v.keyholder),
            v.cluster_id,
            v.nonce_k,
        );
        let challenge = keyholder.on_request(joiner.id, joiner.request()).unwrap();
        joiner.on_challenge(keyholder.id, challenge).unwrap();
        assert_eq!(tamper(&mut keyholder, &joiner), Err(Denied));
        assert_eq!(keyholder.sas(), None);
        // The pairing is over: the right reveal no longer helps.
        assert_eq!(
            keyholder.on_reveal(joiner.id, Reveal { nonce_c: v.nonce_c }),
            Err(Denied)
        );
    }
}

#[test]
fn a_request_of_the_other_exchange_is_denied() {
    let v = &spec::vectors()[1];
    for (keyholder_side, joiner_side) in [
        (Exchange::Committed, Exchange::FirstDraft),
        (Exchange::FirstDraft, Exchange::Committed),
    ] {
        let joiner = Joiner::new(joiner_side, NodeId(v.joiner), v.nonce_c);
        let mut keyholder =
            Keyholder::new(keyholder_side, NodeId(v.keyholder), v.cluster_id, v.nonce_k);
        assert!(matches!(
            keyholder.on_request(joiner.id, joiner.request()),
            Err(Denied)
        ));
    }
}

fn vectors_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/pairing/vectors.txt")
}

/// The committed file equals the spec's rendering. `SYNTROPY_BLESS_GOLDEN=1` rewrites it, which
/// is a change to the pairing protocol.
#[test]
fn the_vectors_file_matches_the_spec() {
    let text = spec::render(&spec::vectors());
    let path = vectors_path();
    if std::env::var("SYNTROPY_BLESS_GOLDEN").as_deref() == Ok("1") {
        std::fs::write(&path, &text).unwrap();
    }
    let file = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .replace("\r\n", "\n");
    assert_eq!(file, text, "{} differs from the spec", path.display());
}

#[test]
fn the_library_derives_the_vectors() {
    for v in spec::vectors() {
        let joiner = NodeId(v.joiner);
        let commit_input = spec::commit_input(&v);
        assert_eq!(commit_input.len(), 22 + 32 + 16);
        let commit = Commitment::new(&joiner, &v.nonce_c);
        assert_eq!(
            commit.0,
            *blake3::hash(&commit_input).as_bytes(),
            "{}",
            v.name
        );
        assert!(commit.opens(&joiner, &v.nonce_c));
        let inputs = SasInputs {
            cluster_id: v.cluster_id,
            joiner,
            keyholder: NodeId(v.keyholder),
            nonce_c: v.nonce_c,
            nonce_k: v.nonce_k,
        };
        let sas_input = spec::sas_input(&v);
        assert_eq!(sas_input.len(), 15 + 3 * 32 + 2 * 16);
        assert_eq!(
            inputs.digest(),
            *blake3::hash(&sas_input).as_bytes(),
            "{}",
            v.name
        );
        let code = spec::sas(&v);
        assert_eq!(inputs.sas().value(), code, "{}", v.name);
        assert_eq!(
            inputs.sas().to_string(),
            format!("{:03} {:03}", code / 1000, code % 1000)
        );
    }
}
