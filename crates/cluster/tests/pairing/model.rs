// SPDX-License-Identifier: MIT OR Apache-2.0

//! A pure model of LAN pairing between a joiner, an attacker relay and a keyholder (RFC-0034
//! Testing, "The pairing SAS cannot be ground"). It runs the exchange as first drafted, where
//! `PairRequest` carried `nonce_c` itself, and as frozen, where it carries the commitment and
//! `PairReveal` carries the nonce after the challenge (RFC-0030 §6.2).
//!
//! Only the pairing fields travel. TLS is modelled by passing each receiver the authenticated
//! node ID of its peer: the attacker can relay and choose its own nonces, but cannot claim
//! another node's key. The joiner believes the attacker is a keyholder of a lookalike cluster,
//! and the keyholder sees the attacker as a joiner. The attack succeeds when the joiner's code
//! equals the code the keyholder shows for the attacker's request, because the user then picks
//! that code on the keyholder and approves the attacker.

use syntropy_cluster::pairing::{Commitment, Nonce, Sas, SasInputs};
use syntropy_core::NodeId;

/// SplitMix64: a small deterministic generator, so every trial is reproducible from its seed.
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn nonce(&mut self) -> Nonce {
        let mut out = [0; 16];
        out[..8].copy_from_slice(&self.next_u64().to_le_bytes());
        out[8..].copy_from_slice(&self.next_u64().to_le_bytes());
        out
    }

    pub fn key(&mut self) -> [u8; 32] {
        let mut out = [0; 32];
        for chunk in out.chunks_mut(8) {
            chunk.copy_from_slice(&self.next_u64().to_le_bytes());
        }
        out
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exchange {
    /// As first drafted: `PairRequest` carries `nonce_c`, and the keyholder answers with
    /// `nonce_k` after seeing it.
    FirstDraft,
    /// As frozen: `PairRequest` carries the commitment, and `PairReveal` carries `nonce_c` only
    /// after `PairChallenge` has arrived.
    Committed,
}

/// `PairRequest`'s pairing field.
#[derive(Clone, Copy, Debug)]
pub enum Request {
    FirstDraft { nonce_c: Nonce },
    Committed { commit: Commitment },
}

/// `PairChallenge`'s pairing fields.
#[derive(Clone, Copy, Debug)]
pub struct Challenge {
    pub cluster_id: [u8; 32],
    pub nonce_k: Nonce,
}

/// `PairReveal`'s pairing field.
#[derive(Clone, Copy, Debug)]
pub struct Reveal {
    pub nonce_c: Nonce,
}

/// `Rejected { Denied }`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Denied;

/// The joining node: it draws `nonce_c` before anything else happens.
pub struct Joiner {
    exchange: Exchange,
    pub id: NodeId,
    nonce_c: Nonce,
    sas: Option<Sas>,
}

impl Joiner {
    pub fn new(exchange: Exchange, id: NodeId, nonce_c: Nonce) -> Joiner {
        Joiner {
            exchange,
            id,
            nonce_c,
            sas: None,
        }
    }

    pub fn request(&self) -> Request {
        match self.exchange {
            Exchange::FirstDraft => Request::FirstDraft {
                nonce_c: self.nonce_c,
            },
            Exchange::Committed => Request::Committed {
                commit: Commitment::new(&self.id, &self.nonce_c),
            },
        }
    }

    /// `peer` is the node TLS authenticated as the sender of the challenge. The joiner's code
    /// is fixed here; in the committed exchange it reveals its nonce only now.
    pub fn on_challenge(&mut self, peer: NodeId, challenge: Challenge) -> Option<Reveal> {
        self.sas = Some(
            SasInputs {
                cluster_id: challenge.cluster_id,
                joiner: self.id,
                keyholder: peer,
                nonce_c: self.nonce_c,
                nonce_k: challenge.nonce_k,
            }
            .sas(),
        );
        match self.exchange {
            Exchange::FirstDraft => None,
            Exchange::Committed => Some(Reveal {
                nonce_c: self.nonce_c,
            }),
        }
    }

    /// The code the joiner shows.
    pub fn sas(&self) -> Option<Sas> {
        self.sas
    }
}

/// A keyholder of the real cluster: it draws `nonce_k` only when a request arrives.
pub struct Keyholder {
    exchange: Exchange,
    pub id: NodeId,
    cluster_id: [u8; 32],
    nonce_k: Nonce,
    pending: Option<(NodeId, Request)>,
    sas: Option<Sas>,
}

impl Keyholder {
    pub fn new(exchange: Exchange, id: NodeId, cluster_id: [u8; 32], nonce_k: Nonce) -> Keyholder {
        Keyholder {
            exchange,
            id,
            cluster_id,
            nonce_k,
            pending: None,
            sas: None,
        }
    }

    fn inputs(&self, joiner: NodeId, nonce_c: Nonce) -> SasInputs {
        SasInputs {
            cluster_id: self.cluster_id,
            joiner,
            keyholder: self.id,
            nonce_c,
            nonce_k: self.nonce_k,
        }
    }

    /// `peer` is the node TLS authenticated as the requester. A request of the other exchange
    /// is denied.
    pub fn on_request(&mut self, peer: NodeId, request: Request) -> Result<Challenge, Denied> {
        match (self.exchange, request) {
            (Exchange::FirstDraft, Request::FirstDraft { nonce_c }) => {
                self.sas = Some(self.inputs(peer, nonce_c).sas());
            }
            (Exchange::Committed, Request::Committed { .. }) => {}
            _ => return Err(Denied),
        }
        self.pending = Some((peer, request));
        Ok(Challenge {
            cluster_id: self.cluster_id,
            nonce_k: self.nonce_k,
        })
    }

    /// The commitment check: a reveal that does not open the requester's commitment ends the
    /// pairing.
    pub fn on_reveal(&mut self, peer: NodeId, reveal: Reveal) -> Result<(), Denied> {
        let Some((joiner, Request::Committed { commit })) = self.pending.take() else {
            return Err(Denied);
        };
        if peer != joiner || !commit.opens(&joiner, &reveal.nonce_c) {
            return Err(Denied);
        }
        self.sas = Some(self.inputs(joiner, reveal.nonce_c).sas());
        Ok(())
    }

    /// The code the keyholder shows, among two decoys, for the pending request.
    pub fn sas(&self) -> Option<Sas> {
        self.sas
    }
}

/// The order in which the attacker serves its two connections.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Order {
    /// Pair with the keyholder first, then answer the joiner: the order that breaks the first
    /// draft, because the keyholder's code is fixed before the attacker picks `nonce_k`.
    KeyholderFirst,
    /// Answer the joiner first, then pair with the keyholder.
    JoinerFirst,
}

/// The result of one relayed pairing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Outcome {
    /// The two codes matched, so the user's pick approves the attacker.
    pub matched: bool,
    /// BLAKE3 evaluations the attacker spent grinding.
    pub hashes: u64,
}

/// What the attacker grinds toward: the code to match, and the code a candidate nonce gives.
type Aim<'a> = (Sas, &'a dyn Fn(&Nonce) -> Sas);

/// An attacker between a joiner and a keyholder, with its own node key and a lookalike cluster.
pub struct Attacker {
    pub id: NodeId,
    pub cluster_id: [u8; 32],
    /// The most hashes it spends on one nonce.
    pub budget: u64,
}

impl Attacker {
    /// Picks the next nonce to send. When the attacker knows the code to match and can compute
    /// the code this nonce fixes, it grinds: it draws candidates until one matches or the budget
    /// runs out. Otherwise any value is as good as another, and it draws one.
    fn choose(&self, rng: &mut SplitMix64, hashes: &mut u64, aim: Option<Aim<'_>>) -> Nonce {
        let Some((target, code_for)) = aim else {
            return rng.nonce();
        };
        for _ in 0..self.budget {
            let candidate = rng.nonce();
            *hashes += 1;
            if code_for(&candidate) == target {
                return candidate;
            }
        }
        // Out of budget: an unevaluated nonce, as good as any other.
        rng.nonce()
    }

    /// The attacker's own request to the keyholder, for its own `nonce_c`.
    fn request(&self, exchange: Exchange, nonce_c: Nonce) -> Request {
        match exchange {
            Exchange::FirstDraft => Request::FirstDraft { nonce_c },
            Exchange::Committed => Request::Committed {
                commit: Commitment::new(&self.id, &nonce_c),
            },
        }
    }

    /// One relayed pairing attempt. The attacker grinds wherever the exchange lets it.
    pub fn relay(
        &self,
        exchange: Exchange,
        order: Order,
        joiner: &mut Joiner,
        keyholder: &mut Keyholder,
        rng: &mut SplitMix64,
    ) -> Result<Outcome, Denied> {
        let mut hashes = 0;
        let request = joiner.request();
        // The joiner's nonce, if the request shows it. A commitment hides it: recovering it is a
        // preimage search, 2⁻¹²⁸ per hash.
        let known_nonce_c = match request {
            Request::FirstDraft { nonce_c } => Some(nonce_c),
            Request::Committed { .. } => None,
        };
        let joiner_id = joiner.id;
        let joiner_code = |nonce_c: Nonce, nonce_k: &Nonce| {
            SasInputs {
                cluster_id: self.cluster_id,
                joiner: joiner_id,
                keyholder: self.id,
                nonce_c,
                nonce_k: *nonce_k,
            }
            .sas()
        };
        match order {
            Order::KeyholderFirst => {
                // Pair with the keyholder as a joiner, honestly, so its reveal opens.
                let own_nonce_c = self.choose(rng, &mut hashes, None);
                let challenge =
                    keyholder.on_request(self.id, self.request(exchange, own_nonce_c))?;
                if exchange == Exchange::Committed {
                    keyholder.on_reveal(
                        self.id,
                        Reveal {
                            nonce_c: own_nonce_c,
                        },
                    )?;
                }
                // The keyholder's code is fixed, and the attacker knows every input to it. The
                // `nonce_k` it sends the joiner fixes the joiner's code, which it can compute
                // only if it knows the joiner's nonce.
                let nonce_k = match known_nonce_c {
                    Some(nonce_c) => {
                        hashes += 1;
                        let target = SasInputs {
                            cluster_id: challenge.cluster_id,
                            joiner: self.id,
                            keyholder: keyholder.id,
                            nonce_c: own_nonce_c,
                            nonce_k: challenge.nonce_k,
                        }
                        .sas();
                        let code_for = |nonce_k: &Nonce| joiner_code(nonce_c, nonce_k);
                        self.choose(rng, &mut hashes, Some((target, &code_for)))
                    }
                    None => self.choose(rng, &mut hashes, None),
                };
                joiner.on_challenge(
                    self.id,
                    Challenge {
                        cluster_id: self.cluster_id,
                        nonce_k,
                    },
                );
            }
            Order::JoinerFirst => {
                // Nothing on the keyholder's side is fixed yet, so there is no code to aim at.
                let nonce_k = self.choose(rng, &mut hashes, None);
                joiner.on_challenge(
                    self.id,
                    Challenge {
                        cluster_id: self.cluster_id,
                        nonce_k,
                    },
                );
                // The joiner's code is now fixed. The keyholder's code depends on the nonce it
                // draws after the attacker's request, so the attacker cannot compute it while
                // choosing its own nonce.
                let own_nonce_c = self.choose(rng, &mut hashes, None);
                keyholder.on_request(self.id, self.request(exchange, own_nonce_c))?;
                if exchange == Exchange::Committed {
                    keyholder.on_reveal(
                        self.id,
                        Reveal {
                            nonce_c: own_nonce_c,
                        },
                    )?;
                }
            }
        }
        let matched = joiner.sas().is_some() && joiner.sas() == keyholder.sas();
        Ok(Outcome { matched, hashes })
    }
}

/// One attempt with fresh parties drawn from `rng`.
pub fn attempt(
    exchange: Exchange,
    order: Order,
    budget: u64,
    rng: &mut SplitMix64,
) -> Result<Outcome, Denied> {
    let mut joiner = Joiner::new(exchange, NodeId(rng.key()), rng.nonce());
    let mut keyholder = Keyholder::new(exchange, NodeId(rng.key()), rng.key(), rng.nonce());
    let attacker = Attacker {
        id: NodeId(rng.key()),
        cluster_id: rng.key(),
        budget,
    };
    attacker.relay(exchange, order, &mut joiner, &mut keyholder, rng)
}
