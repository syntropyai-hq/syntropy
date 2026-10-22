// SPDX-License-Identifier: MIT OR Apache-2.0

//! An explicit-state model of coordinator election, the lease, log replication and split heal
//! (RFC-0032 §6.7, §6.10, §7.1–§7.6; ADR-0304 and its revisions; RFC-0034 §8.1, §8.3).
//!
//! What is abstracted:
//! - **Time.** A follower's lease is a flag. It goes live when a beat or append from its
//!   coordinator arrives, and expires only when the follower is cut off from the coordinator,
//!   the coordinator is down, or a message loss is spent on it. A claim's window closes once
//!   every claim and reply of that claim has been delivered or lost.
//! - **Signatures.** A signature is a flag that only the signer can set. TLS is modelled by
//!   giving each receiver the true sender of every message. A byzantine member can send anything
//!   it can sign itself, but cannot forge another node's signature: its forged proofs and
//!   authority entries carry `sig_ok = false`.
//! - **Replication.** An append carries the coordinator's whole log and its current
//!   `ElectionProof`, so it is also the beat and the snapshot (§6.7). A follower whose log is a
//!   prefix of it takes the new entries after `validate()`; one whose log diverges replaces it
//!   only after `validate_snapshot()`.
//! - **Replies.** A voter's reply to a claim, and a follower's `LogAck` (with its `Endorse`) to
//!   an append, travel on the stream of the message they answer and arrive with it. Losing the
//!   claim or the append loses the reply too.
//! - **Rank.** A lower node index ranks higher; operator preference and capability class only
//!   reorder the same rule.
//! - **Planned handover** (§7.7) is not modelled; DIST-E10-T04 builds it.
//!
//! Ghost state, which no node reads, records what really happened: the elections nodes actually
//! won, the coordinator each node accepted at each term, the commands acknowledged with
//! `replicated = true`, and the violations the transition oracle in [`crate::check`] finds.

use std::collections::{BTreeMap, BTreeSet};

/// A node, by index. A lower index ranks higher.
pub type Ix = u8;

/// `term = (counter << 24) | tag24(node)`, ordered by counter, then tag (§7.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Term {
    pub counter: u8,
    pub tag: u8,
}

/// A coordinator's `ElectionProof` (§7.5): its signed election and the members that acked its
/// claim or endorsed its election, as a bit mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Proof {
    pub node: Ix,
    pub term: Term,
    /// The election's signature verifies under `node`'s key. Only `node` can make it true.
    pub sig_ok: bool,
    /// Members other than `node` whose `Ack` or `Endorse` the proof carries.
    pub support: u8,
}

impl Proof {
    /// At least one `Ack` or `Endorse` from a member other than the elected node.
    pub fn supported(&self) -> bool {
        self.support & !(1 << self.node) != 0
    }

    /// The heal order: `(supported, counter, tag)`, highest first, then the lower node ID.
    pub fn heal_key(&self) -> (bool, u8, u8, std::cmp::Reverse<Ix>) {
        (
            self.supported(),
            self.term.counter,
            self.term.tag,
            std::cmp::Reverse(self.node),
        )
    }

    pub fn beats(&self, other: &Proof) -> bool {
        self.heal_key() > other.heal_key()
    }
}

/// A log entry, reduced to what election, replication and authority need.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Entry {
    /// `CoordinatorElected`, a signed election.
    Elected { node: Ix, term: Term },
    /// The effect of a journalled command, by `CommandId`.
    Command { id: u8 },
    /// A keyholder-signed authority entry (`ApiKeyCreated`, `KeyholderChanged`, …). A forged one
    /// has `sig_ok = false`.
    Authority { sig_ok: bool },
    /// `NodeRevoked`, carrying a keyholder's `SignedRevocation`.
    Revoked { node: Ix },
}

/// The counter of the last `CoordinatorElected` in `log`, 0 before any.
pub fn last_counter(log: &[Entry]) -> u8 {
    log.iter()
        .rev()
        .find_map(|e| match e {
            Entry::Elected { term, .. } => Some(term.counter),
            _ => None,
        })
        .unwrap_or(0)
}

/// The last `CoordinatorElected` in `log`.
pub fn last_elected(log: &[Entry]) -> Option<(Ix, Term)> {
    log.iter().rev().find_map(|e| match e {
        Entry::Elected { node, term } => Some((*node, *term)),
        _ => None,
    })
}

/// The term under which entry `at` of `log` was appended: that of the last election before it.
pub fn term_at(log: &[Entry], at: usize) -> Option<Term> {
    log[..at].iter().rev().find_map(|e| match e {
        Entry::Elected { term, .. } => Some(*term),
        _ => None,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ReplyKind {
    Ack,
    LeaseLive,
    Better,
    Stale,
}

/// A message in flight. `from` is the true sender: TLS authenticates it.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Msg {
    Claim {
        from: Ix,
        to: Ix,
        term: Term,
    },
    /// `LogAppend`, which also serves as the beat and the snapshot.
    Append {
        from: Ix,
        to: Ix,
        term: Term,
        log: Vec<Entry>,
        proof: Proof,
    },
    /// A fencing reply: the sender's coordinator's proof, or `None` for a bare term.
    Fence {
        from: Ix,
        to: Ix,
        proof: Option<Proof>,
    },
    /// A command sent to the coordinator.
    Command {
        from: Ix,
        to: Ix,
        id: u8,
    },
    /// A keyholder's `Revoke` command, sent to the coordinator.
    RevokeCmd {
        from: Ix,
        to: Ix,
        node: Ix,
    },
    /// A `SignedRevocation` pushed or synced outside the log (RFC-0034 §8.1).
    Revocation {
        from: Ix,
        to: Ix,
        node: Ix,
    },
}

impl Msg {
    pub fn ends(&self) -> (Ix, Ix) {
        match *self {
            Msg::Claim { from, to, .. }
            | Msg::Append { from, to, .. }
            | Msg::Fence { from, to, .. }
            | Msg::Command { from, to, .. }
            | Msg::RevokeCmd { from, to, .. }
            | Msg::Revocation { from, to, .. } => (from, to),
        }
    }
}

/// One node's state.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Node {
    pub alive: bool,
    pub log: Vec<Entry>,
    /// The coordinator it follows, itself when it is the coordinator.
    pub coordinator: Option<Ix>,
    /// `current_term`: the term of the coordinator it follows.
    pub term: Option<Term>,
    /// Its coordinator's proof, as last received, or its own when it is the coordinator.
    pub proof: Option<Proof>,
    /// It hears its coordinator within the lease. A coordinator hears itself.
    pub lease_live: bool,
    /// Its claim in progress: the term and the voters that acked.
    pub claim: Option<(Term, u8)>,
    pub claims_left: u8,
    /// Claims seen, as `(counter, claimant)`, for `Better`.
    pub seen: BTreeSet<(u8, Ix)>,
    /// The command journal: submitted commands not yet retired.
    pub journal: BTreeSet<u8>,
    /// The local deny list, as a bit mask of revoked nodes.
    pub denied: u8,
    /// Coordinators, by term, whose appends it refuses after they failed `validate()`.
    pub refused: BTreeSet<(Ix, Term)>,
    /// Coordinator only: commands appended and not yet replied to.
    pub pending: BTreeSet<u8>,
    /// Keyholder only: the revocation it signed, kept and re-submitted for good.
    pub kh_revoked: Option<Ix>,
}

impl Node {
    fn new(claims: u8) -> Node {
        Node {
            alive: true,
            log: Vec::new(),
            coordinator: None,
            term: None,
            proof: None,
            lease_live: false,
            claim: None,
            claims_left: claims,
            seen: BTreeSet::new(),
            journal: BTreeSet::new(),
            denied: 0,
            refused: BTreeSet::new(),
            pending: BTreeSet::new(),
            kh_revoked: None,
        }
    }

    pub fn is_coordinator(&self, me: Ix) -> bool {
        self.coordinator == Some(me)
    }

    pub fn denies(&self, node: Ix) -> bool {
        self.denied & (1 << node) != 0
    }

    pub fn has_command(&self, id: u8) -> bool {
        self.log.contains(&Entry::Command { id })
    }
}

/// Violations the transition oracle records, as bits. No node reads them.
pub mod flag {
    pub const TWO_AT_ONE_TERM: u16 = 1 << 0;
    pub const ABORTED_CLAIM_DEPOSED: u16 = 1 << 1;
    pub const FORGED_APPLIED: u16 = 1 << 2;
    pub const FORGED_SNAPSHOT_ADOPTED: u16 = 1 << 3;
    pub const FORGED_TERM_DEPOSED: u16 = 1 << 4;
    pub const UNSUPPORTED_DEPOSED_SUPPORTED: u16 = 1 << 5;
    pub const HEAL_UNDECIDED: u16 = 1 << 6;
}

/// Situations a scenario must reach, so its properties are not vacuous, as bits.
pub mod seen {
    pub const SPLIT_ELECTED_EQUAL_COUNTERS: u16 = 1 << 0;
    pub const REVOCATION_OMITTED_BUT_HELD: u16 = 1 << 1;
    pub const CLAIM_ABORTED_AFTER_ACK: u16 = 1 << 2;
    pub const FORGED_REFUSED: u16 = 1 << 3;
    pub const FORGED_SNAPSHOT_REFUSED: u16 = 1 << 4;
    pub const FORGED_TERM_IGNORED: u16 = 1 << 5;
    pub const SUPPORTED_BEAT_UNSUPPORTED: u16 = 1 << 6;
    pub const REPLICATED_ACK: u16 = 1 << 7;
    pub const HEALED_AFTER_SPLIT_ELECTION: u16 = 1 << 8;
    pub const NO_ROLE_CLAIM_IGNORED: u16 = 1 << 9;
    pub const CLAIM_ABORTED: u16 = 1 << 10;
    pub const LONE_SURVIVOR_LEADS: u16 = 1 << 11;
}

/// The whole system's state.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct State {
    pub nodes: Vec<Node>,
    pub net: BTreeSet<Msg>,
    /// Cut links, each as `(lower, higher)`.
    pub cut: BTreeSet<(Ix, Ix)>,
    pub splits_left: u8,
    pub losses_left: u8,
    pub crashes_left: u8,
    pub forges_left: u8,
    pub revoke_issued: bool,
    /// Ghost: elections actually won, as `(node, term)`.
    pub elections: BTreeSet<(Ix, Term)>,
    /// Ghost: per node, the coordinator it accepted at each term.
    pub accepted: Vec<BTreeMap<Term, Ix>>,
    /// Ghost: commands a coordinator acknowledged with `replicated = true`.
    pub acked_replicated: BTreeSet<u8>,
    /// Ghost: a split happened, then elections at the same counter on both sides.
    pub split_elected: BTreeSet<(Ix, Term)>,
    /// Ghost: voters that acked a claim in progress, as `(claimant, claim term, voter)`, with
    /// the coordinator and term the voter followed when it acked.
    pub ackers: BTreeMap<(Ix, Term, Ix), (Ix, Term)>,
    /// Ghost: voters that acked a claim which then aborted, with the coordinator and term they
    /// followed. Each must still accept that coordinator's next valid beat.
    pub aborted_ackers: BTreeMap<Ix, (Ix, Term)>,
    /// Ghost: violations ([`flag`]) and situations reached ([`seen`]).
    pub flags: u16,
    pub reached: u16,
}

/// A transition.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Action {
    Deliver(Msg),
    Drop(Msg),
    /// The lease of a follower runs out.
    Expire(Ix),
    Claim(Ix),
    /// The claim window closes and the claimant takes office.
    Win(Ix),
    /// The coordinator appends to, or beats, a follower.
    Replicate(Ix, Ix),
    Resubmit(Ix, u8),
    /// A keyholder re-submits the revocation it signed, which it keeps for good.
    ResubmitRevocation(Ix),
    Split(u8),
    Heal,
    Crash(Ix),
    /// The scenario's keyholder signs the revocation and pushes it.
    Revoke,
    /// A holder of a revocation syncs it to a peer that lacks it (`deny_digest`).
    Gossip(Ix, Ix, Ix),
    /// A crashed daemon restarts with its disk; it never resumes as coordinator (§7.1).
    Restart(Ix),
    /// The byzantine coordinator appends a forged authority entry.
    ForgeAuthority,
    /// The byzantine member answers a node with a bare term (`None`) or a forged proof.
    ForgeTerm(Ix, bool),
}

/// The scenario a model checks.
#[derive(Clone, Debug)]
pub struct Config {
    pub name: &'static str,
    pub n: u8,
    /// Holds a certificate with `COORDINATOR`.
    pub role: Vec<bool>,
    pub keyholder: Vec<bool>,
    /// `tag24(node)`; two equal tags model a collision.
    pub tags: Vec<u8>,
    pub byzantine: Option<Ix>,
    /// Node 0 starts as the supported coordinator at counter 1; otherwise no one leads.
    pub start_led: bool,
    pub max_counter: u8,
    pub claims_per_node: u8,
    /// The partitions a split may make, each as the links it cuts, `(lower, higher)`.
    pub splits: Vec<Vec<(Ix, Ix)>>,
    pub split_budget: u8,
    pub loss_budget: u8,
    /// Nodes that may crash, and how many may.
    pub crashable: Vec<Ix>,
    pub crash_budget: u8,
    /// A crashed node restarts with its disk: its log, journal, deny list and signed records
    /// survive (crash-recover). Otherwise a crash is the loss of the node and its disk.
    pub restart: bool,
    /// The reply rule before OQ-NET-33: the originator's own ack also counts toward
    /// `replicated = true`. Only the regression test that shows why the rule changed sets it.
    pub originator_ack_counts: bool,
    /// Commands, as `(id, originator)`.
    pub commands: Vec<(u8, Ix)>,
    /// `(keyholder, node)` the keyholder revokes.
    pub revocation: Option<(Ix, Ix)>,
    pub forge_budget: u8,
    /// The byzantine coordinator may append a forged authority entry.
    pub forge_authority: bool,
    /// The byzantine member may answer with bare terms or forged proofs.
    pub forge_terms: bool,
    /// Every split keeps the nodes connected, and nothing is lost or crashes: the property
    /// "a partial partition never yields two coordinators that a single node accepts" applies.
    pub partial_partition: bool,
    /// The situations ([`seen`]) the scenario must reach.
    pub must_reach: u16,
    /// A property this scenario is expected to fail, with the plan record that explains it.
    pub expected_failure: Option<&'static str>,
    /// The most states the checker generates; beyond it the run stops and reports the bound.
    pub state_cap: usize,
}

/// The links a split cuts between `side` and the other nodes of an `n`-node cluster.
pub fn side(n: u8, side: &[Ix]) -> Vec<(Ix, Ix)> {
    let mut cut = Vec::new();
    for a in 0..n {
        for b in a + 1..n {
            if side.contains(&a) != side.contains(&b) {
                cut.push((a, b));
            }
        }
    }
    cut
}

/// The model of one scenario.
#[derive(Clone, Debug)]
pub struct ElectionModel {
    pub cfg: Config,
}

pub fn bit(node: Ix) -> u8 {
    1 << node
}

impl ElectionModel {
    pub fn honest(&self, node: Ix) -> bool {
        self.cfg.byzantine != Some(node)
    }

    pub fn tag(&self, node: Ix) -> u8 {
        self.cfg.tags[node as usize]
    }

    /// Whether `a` and `b` can exchange messages now.
    pub fn link(&self, s: &State, a: Ix, b: Ix) -> bool {
        let (lo, hi) = (a.min(b), a.max(b));
        s.nodes[a as usize].alive
            && s.nodes[b as usize].alive
            && (a == b || !s.cut.contains(&(lo, hi)))
    }

    /// Whether a message in flight can arrive: the receiver is up and the link is not cut. A
    /// message sent before its sender crashed still arrives.
    pub fn deliverable(&self, s: &State, msg: &Msg) -> bool {
        let (from, to) = msg.ends();
        let (lo, hi) = (from.min(to), from.max(to));
        s.nodes[to as usize].alive && (from == to || !s.cut.contains(&(lo, hi)))
    }

    fn send(&self, s: &mut State, msg: Msg) {
        let (from, to) = msg.ends();
        if self.link(s, from, to) {
            s.net.insert(msg);
        }
    }

    /// What every node checks of a proof: the signature, a `COORDINATOR` certificate, and that
    /// the elected node is not on the verifier's deny list (§7.5).
    pub fn verifies(&self, s: &State, verifier: Ix, proof: &Proof) -> bool {
        proof.sig_ok
            && self.cfg.role[proof.node as usize]
            && !s.nodes[verifier as usize].denies(proof.node)
    }

    /// `validate()` of one entry by `node` (RFC-0034 §8.3).
    fn valid_entry(&self, entry: &Entry) -> bool {
        match *entry {
            Entry::Elected { node, .. } => self.cfg.role[node as usize],
            Entry::Authority { sig_ok } => sig_ok,
            Entry::Revoked { .. } | Entry::Command { .. } => true,
        }
    }

    /// `validate_snapshot()`: every authority-derived record verifies, and the coordinator record
    /// matches the proof (§6.7).
    fn valid_snapshot(&self, log: &[Entry], proof: &Proof) -> bool {
        log.iter().all(|e| self.valid_entry(e))
            && last_elected(log) == Some((proof.node, proof.term))
    }

    /// The term a claimant would claim at now.
    pub fn next_term(&self, s: &State, node: Ix) -> Term {
        Term {
            counter: last_counter(&s.nodes[node as usize].log) + 1,
            tag: self.tag(node),
        }
    }

    /// May `node` claim now? Only within the scenario's bounds.
    pub fn may_claim(&self, s: &State, node: Ix) -> bool {
        let n = &s.nodes[node as usize];
        n.alive
            && !n.lease_live
            && n.claim.is_none()
            && !n.is_coordinator(node)
            && (self.cfg.role[node as usize] || !self.honest(node))
            && n.claims_left > 0
            && self.next_term(s, node).counter <= self.cfg.max_counter
    }

    /// Whether `c`'s beats reach `f`: `c` is up and reachable, and coordinates at the term `f`
    /// follows. Otherwise `f`'s lease runs out as time passes.
    fn beats_reach(&self, s: &State, c: Ix, f: Ix) -> bool {
        let leader = &s.nodes[c as usize];
        self.link(s, f, c) && leader.is_coordinator(c) && leader.term == s.nodes[f as usize].term
    }

    /// A claim's window is open while any of its claims or replies is in flight.
    fn claim_in_flight(&self, s: &State, node: Ix, term: Term) -> bool {
        s.net.iter().any(|m| match *m {
            Msg::Claim { from, term: t, .. } => from == node && t == term,
            _ => false,
        })
    }

    /// Would a fresh append from coordinator `c` change anything at `f`?
    pub fn append_needed(&self, s: &State, c: Ix, f: Ix) -> bool {
        let (cn, fn_) = (&s.nodes[c as usize], &s.nodes[f as usize]);
        if !self.honest(c) && cn.term.is_some_and(|t| fn_.refused.contains(&(c, t))) {
            return false;
        }
        !(fn_.coordinator == Some(c)
            && fn_.term == cn.term
            && fn_.log == cn.log
            && fn_.lease_live
            && fn_.proof == cn.proof)
    }

    fn append_in_flight(&self, s: &State, c: Ix, f: Ix) -> bool {
        s.net
            .iter()
            .any(|m| matches!(*m, Msg::Append { from, to, .. } if from == c && to == f))
    }

    pub fn initial(&self) -> State {
        let n = self.cfg.n as usize;
        let mut nodes: Vec<Node> = (0..n)
            .map(|_| Node::new(self.cfg.claims_per_node))
            .collect();
        let mut accepted = vec![BTreeMap::new(); n];
        let mut elections = BTreeSet::new();
        if self.cfg.start_led {
            let term = Term {
                counter: 1,
                tag: self.tag(0),
            };
            let support = (1..n as Ix).fold(0u8, |m, i| m | bit(i));
            let proof = Proof {
                node: 0,
                term,
                sig_ok: true,
                support,
            };
            for (i, node) in nodes.iter_mut().enumerate() {
                node.log = vec![Entry::Elected { node: 0, term }];
                node.coordinator = Some(0);
                node.term = Some(term);
                node.proof = Some(proof);
                node.lease_live = true;
                accepted[i].insert(term, 0);
            }
            elections.insert((0, term));
        }
        let mut s = State {
            nodes,
            net: BTreeSet::new(),
            cut: BTreeSet::new(),
            splits_left: self.cfg.split_budget,
            losses_left: self.cfg.loss_budget,
            crashes_left: self.cfg.crash_budget,
            forges_left: self.cfg.forge_budget,
            revoke_issued: false,
            elections,
            accepted,
            acked_replicated: BTreeSet::new(),
            split_elected: BTreeSet::new(),
            ackers: BTreeMap::new(),
            aborted_ackers: BTreeMap::new(),
            flags: 0,
            reached: 0,
        };
        // Each command is journalled and sent to the coordinator at the start.
        for &(id, origin) in &self.cfg.commands {
            s.nodes[origin as usize].journal.insert(id);
            self.resubmit(&mut s, origin, id);
        }
        s
    }

    /// The protocol's own actions, without faults: what still has to happen for the system to
    /// settle.
    pub fn protocol_actions(&self, s: &State, out: &mut Vec<Action>) {
        for m in &s.net {
            if self.deliverable(s, m) {
                out.push(Action::Deliver(m.clone()));
            }
        }
        for i in 0..self.cfg.n {
            let node = &s.nodes[i as usize];
            if !node.alive {
                continue;
            }
            // A follower cut off from its coordinator, or whose coordinator is down, loses its
            // lease as time passes.
            if node.lease_live
                && let Some(c) = node.coordinator
                && c != i
                && !self.beats_reach(s, c, i)
            {
                out.push(Action::Expire(i));
            }
            if self.may_claim(s, i) {
                out.push(Action::Claim(i));
            }
            if let Some((term, _)) = node.claim
                && !self.claim_in_flight(s, i, term)
            {
                out.push(Action::Win(i));
            }
            if node.is_coordinator(i) {
                for f in 0..self.cfg.n {
                    if f != i
                        && self.link(s, i, f)
                        && !self.append_in_flight(s, i, f)
                        && self.append_needed(s, i, f)
                    {
                        out.push(Action::Replicate(i, f));
                    }
                }
            }
            // A journalled command is re-sent while its coordinator lacks it: dedup by
            // `CommandId` answers any other re-send with the original reply (§6.3, §6.10).
            for &id in &node.journal {
                if !node.has_command(id)
                    && let Some(c) = node.coordinator
                    && self.link(s, i, c)
                    && !s.nodes[c as usize].has_command(id)
                    && (c == i || !s.net.contains(&Msg::Command { from: i, to: c, id }))
                    && (c != i || node.is_coordinator(i))
                {
                    out.push(Action::Resubmit(i, id));
                }
            }
            if let Some(target) = node.kh_revoked
                && !node.log.contains(&Entry::Revoked { node: target })
                && let Some(c) = node.coordinator
                && c != i
                && self.link(s, i, c)
                && !s.nodes[c as usize]
                    .log
                    .contains(&Entry::Revoked { node: target })
                && !s.net.contains(&Msg::RevokeCmd {
                    from: i,
                    to: c,
                    node: target,
                })
            {
                out.push(Action::ResubmitRevocation(i));
            }
            for x in 0..self.cfg.n {
                if node.denies(x) {
                    for peer in 0..self.cfg.n {
                        let msg = Msg::Revocation {
                            from: i,
                            to: peer,
                            node: x,
                        };
                        if peer != i
                            && peer != x
                            && self.link(s, i, peer)
                            && !s.nodes[peer as usize].denies(x)
                            && !s.net.contains(&msg)
                        {
                            out.push(Action::Gossip(i, peer, x));
                        }
                    }
                }
            }
        }
        if self.cfg.restart {
            for i in 0..self.cfg.n {
                if !s.nodes[i as usize].alive {
                    out.push(Action::Restart(i));
                }
            }
        }
        if let Some((kh, _)) = self.cfg.revocation
            && !s.revoke_issued
            && s.nodes[kh as usize].alive
        {
            out.push(Action::Revoke);
        }
    }

    /// Faults and adversary moves: splits, heals, losses, crashes and forgeries.
    pub fn fault_actions(&self, s: &State, out: &mut Vec<Action>) {
        if s.losses_left > 0 {
            for m in &s.net {
                out.push(Action::Drop(m.clone()));
            }
            for i in 0..self.cfg.n {
                let node = &s.nodes[i as usize];
                if node.alive
                    && node.lease_live
                    && node
                        .coordinator
                        .is_some_and(|c| c != i && self.beats_reach(s, c, i))
                {
                    out.push(Action::Expire(i));
                }
            }
        }
        if s.cut.is_empty() && s.splits_left > 0 {
            for k in 0..self.cfg.splits.len() {
                out.push(Action::Split(k as u8));
            }
        }
        if !s.cut.is_empty() {
            out.push(Action::Heal);
        }
        if s.crashes_left > 0 {
            for &c in &self.cfg.crashable {
                if s.nodes[c as usize].alive {
                    out.push(Action::Crash(c));
                }
            }
        }
        if let Some(b) = self.cfg.byzantine
            && s.forges_left > 0
            && s.nodes[b as usize].alive
        {
            if self.cfg.forge_authority && s.nodes[b as usize].is_coordinator(b) {
                out.push(Action::ForgeAuthority);
            }
            for to in 0..self.cfg.n {
                if self.cfg.forge_terms && to != b && self.link(s, b, to) {
                    out.push(Action::ForgeTerm(to, false));
                    out.push(Action::ForgeTerm(to, true));
                }
            }
        }
    }

    /// Takes office: pulls entries from the reachable member with the longest log that extends
    /// its own, then appends its election (§7.3, "Up to date").
    fn win(&self, s: &mut State, y: Ix) {
        let Some((term, acks)) = s.nodes[y as usize].claim.take() else {
            return;
        };
        let mine = s.nodes[y as usize].log.clone();
        let mut best = mine.clone();
        for m in 0..self.cfg.n {
            if m != y && self.link(s, y, m) {
                let theirs = &s.nodes[m as usize].log;
                if theirs.len() > best.len()
                    && theirs.starts_with(&mine)
                    && theirs.iter().all(|e| self.valid_entry(e))
                {
                    best = theirs.clone();
                }
            }
        }
        if last_counter(&best) + 1 != term.counter {
            // Someone else was elected meanwhile: the claim is stale.
            s.nodes[y as usize].log = best;
            return;
        }
        let me = &mut s.nodes[y as usize];
        me.log = best;
        me.log.push(Entry::Elected { node: y, term });
        let proof = Proof {
            node: y,
            term,
            sig_ok: true,
            support: acks,
        };
        me.coordinator = Some(y);
        me.term = Some(term);
        me.proof = Some(proof);
        me.lease_live = true;
        me.pending.clear();
        let log = me.log.clone();
        for e in log {
            if let Entry::Revoked { node } = e {
                s.nodes[y as usize].denied |= bit(node);
            }
        }
        s.elections.insert((y, term));
        if !s.cut.is_empty() {
            s.split_elected.insert((y, term));
            let equal = s
                .split_elected
                .iter()
                .any(|&(z, t)| z != y && t.counter == term.counter);
            if equal {
                s.reached |= seen::SPLIT_ELECTED_EQUAL_COUNTERS;
            }
        }
        self.record_accept(s, y, term, y);
        self.resubmit_journal(s, y);
    }

    pub fn record_accept(&self, s: &mut State, node: Ix, term: Term, coordinator: Ix) {
        let prev = s.accepted[node as usize].insert(term, coordinator);
        if prev.is_some_and(|p| p != coordinator) {
            s.flags |= flag::TWO_AT_ONE_TERM;
        }
    }

    /// Appends a command's entry at coordinator `c`, once.
    fn coordinator_append_command(&self, s: &mut State, c: Ix, id: u8) {
        let node = &mut s.nodes[c as usize];
        if !node.has_command(id) {
            node.log.push(Entry::Command { id });
        }
        node.pending.insert(id);
    }

    /// Follows `c` at `term` with `log` and `proof`, after the caller has checked it may.
    fn follow(&self, s: &mut State, f: Ix, c: Ix, log: Vec<Entry>, proof: Proof) {
        let snapshot_term = proof.term;
        let term_changed = s.nodes[f as usize].term != Some(proof.term);
        let replaced = !log.starts_with(&s.nodes[f as usize].log);
        {
            let node = &mut s.nodes[f as usize];
            if replaced {
                // A snapshot from a coordinator at a later term retires the journal entries it
                // covers (§6.10).
                let retire: Vec<u8> = node
                    .journal
                    .iter()
                    .copied()
                    .filter(|&id| {
                        log.iter()
                            .position(|e| *e == Entry::Command { id })
                            .is_some_and(|at| term_at(&log, at).is_some_and(|t| t < snapshot_term))
                    })
                    .collect();
                for id in retire {
                    node.journal.remove(&id);
                }
            }
            node.log = log;
            node.coordinator = Some(c);
            node.term = Some(proof.term);
            node.proof = Some(proof);
            node.lease_live = true;
            node.claim = None;
            node.pending.clear();
            let revoked: Vec<Ix> = node
                .log
                .iter()
                .filter_map(|e| match e {
                    Entry::Revoked { node } => Some(*node),
                    _ => None,
                })
                .collect();
            for x in revoked {
                node.denied |= bit(x);
            }
        }
        self.record_accept(s, f, proof.term, c);
        s.aborted_ackers.remove(&f);
        if s.nodes[f as usize].denies(c) {
            self.stop_following(s, f);
            return;
        }
        if term_changed || replaced {
            self.resubmit_journal(s, f);
        }
    }

    /// After a change of coordinator term, or a snapshot replacement, a node re-submits every
    /// journalled command its log lacks (§6.10). A command lost on the way is retried by
    /// [`Action::Resubmit`].
    fn resubmit_journal(&self, s: &mut State, o: Ix) {
        let missing: Vec<u8> = s.nodes[o as usize]
            .journal
            .iter()
            .copied()
            .filter(|&id| !s.nodes[o as usize].has_command(id))
            .collect();
        for id in missing {
            self.resubmit(s, o, id);
        }
    }

    /// Treats the coordinator's lease as expired and stops following it.
    fn stop_following(&self, s: &mut State, f: Ix) {
        s.aborted_ackers.remove(&f);
        let node = &mut s.nodes[f as usize];
        node.coordinator = None;
        node.term = None;
        node.proof = None;
        node.lease_live = false;
        node.pending.clear();
    }

    fn deliver(&self, s: &mut State, msg: Msg) {
        s.net.remove(&msg);
        match msg {
            Msg::Claim { from, to, term } => self.on_claim(s, to, from, term),
            Msg::Append {
                from,
                to,
                term,
                log,
                proof,
            } => self.on_append(s, to, from, term, log, proof),
            Msg::Fence { from: _, to, proof } => self.on_fence(s, to, proof),
            Msg::Command { from: _, to, id } => {
                if s.nodes[to as usize].is_coordinator(to) {
                    self.coordinator_append_command(s, to, id);
                }
            }
            Msg::RevokeCmd { from: _, to, node } => {
                let c = &mut s.nodes[to as usize];
                // A byzantine coordinator omits its own revocation from the log.
                if c.is_coordinator(to)
                    && (self.honest(to) || node != to)
                    && !c.log.contains(&Entry::Revoked { node })
                {
                    c.log.push(Entry::Revoked { node });
                    c.denied |= bit(node);
                }
            }
            Msg::Revocation { from, to, node } => self.on_revocation(s, to, from, node),
        }
    }

    fn on_claim(&self, s: &mut State, v: Ix, y: Ix, term: Term) {
        let voter = &s.nodes[v as usize];
        // An unsigned claim, or one from a node without `COORDINATOR` or revoked, is ignored.
        if !self.cfg.role[y as usize] || voter.denies(y) {
            if !self.cfg.role[y as usize] {
                s.reached |= seen::NO_ROLE_CLAIM_IGNORED;
            }
            return;
        }
        let kind = if term.counter != last_counter(&voter.log) + 1 {
            ReplyKind::Stale
        } else if voter.lease_live {
            ReplyKind::LeaseLive
        } else if voter.seen.iter().any(|&(c, z)| c == term.counter && z < y) {
            ReplyKind::Better
        } else {
            ReplyKind::Ack
        };
        s.nodes[v as usize].seen.insert((term.counter, y));
        // The reply travels on the same request stream; it is lost with the claim or not at all.
        if self.link(s, v, y) {
            self.on_reply(s, y, v, term, kind);
        }
    }

    /// A `Stale` claimant aborts; its coordinator's next append brings its log up to date (§6.7).
    fn on_reply(&self, s: &mut State, y: Ix, v: Ix, term: Term, kind: ReplyKind) {
        let claimant = &mut s.nodes[y as usize];
        let Some((t, acks)) = claimant.claim else {
            return;
        };
        if t != term {
            return;
        }
        match kind {
            ReplyKind::Ack => claimant.claim = Some((t, acks | bit(v))),
            ReplyKind::LeaseLive | ReplyKind::Better | ReplyKind::Stale => {
                claimant.claim = None;
                s.reached |= seen::CLAIM_ABORTED;
                if acks != 0 {
                    s.reached |= seen::CLAIM_ABORTED_AFTER_ACK;
                }
            }
        }
    }

    /// The follower's `LogAck`, with an `Endorse` when it accepted a new election, reaching the
    /// coordinator on the same stream as the append it answers.
    fn log_ack(&self, s: &mut State, f: Ix, c: Ix, term: Term, len: u8, endorse: bool) {
        if !self.link(s, f, c) {
            return;
        }
        let coordinator = &mut s.nodes[c as usize];
        if coordinator.is_coordinator(c) && coordinator.term == Some(term) {
            if endorse && let Some(p) = coordinator.proof.as_mut() {
                p.support |= bit(f);
            }
            // The first ack that holds a waiting command's entry, from a `COORDINATOR`-eligible
            // member other than the coordinator and the command's originator, answers it with
            // `replicated = true` (§6.3, revised 2026-10-22, OQ-NET-33). A command that never
            // gets one is answered `replicated = false` at `log.ack_timeout`, which changes
            // nothing here.
            if !self.cfg.role[f as usize] {
                return;
            }
            let originated_by_f = |id: u8| {
                self.cfg
                    .commands
                    .iter()
                    .any(|&(i, origin)| i == id && origin == f)
            };
            let log = coordinator.log.clone();
            let covered: Vec<u8> = coordinator
                .pending
                .iter()
                .copied()
                .filter(|&id| self.cfg.originator_ack_counts || !originated_by_f(id))
                .filter(|&id| {
                    log.iter()
                        .position(|e| *e == Entry::Command { id })
                        .is_some_and(|at| at < len as usize)
                })
                .collect();
            for id in covered {
                s.nodes[c as usize].pending.remove(&id);
                s.acked_replicated.insert(id);
                s.reached |= seen::REPLICATED_ACK;
            }
        }
    }

    fn on_append(&self, s: &mut State, f: Ix, c: Ix, term: Term, log: Vec<Entry>, proof: Proof) {
        let node = &s.nodes[f as usize];
        let my_proof = node.proof;
        let fence = |s: &mut State| {
            if let Some(p) = s.nodes[f as usize].proof {
                self.send(
                    s,
                    Msg::Fence {
                        from: f,
                        to: c,
                        proof: Some(p),
                    },
                );
            }
        };
        if node.denies(c) {
            return;
        }
        // The proof must be the sender's own, at the append's term, and verify.
        if proof.node != c || proof.term != term || !self.verifies(s, f, &proof) {
            s.reached |= seen::FORGED_TERM_IGNORED;
            fence(s);
            return;
        }
        if node.refused.contains(&(c, term)) {
            fence(s);
            return;
        }
        let following = node.coordinator == Some(c) && node.term == Some(term);
        let extends = log.starts_with(&node.log);
        if following && extends {
            let new_ok = log[node.log.len()..].iter().all(|e| self.valid_entry(e));
            if !new_ok {
                // `validate()` failed: refuse every further append from this coordinator at
                // this term, and treat its lease as expired (§6.3).
                s.reached |= seen::FORGED_REFUSED;
                s.nodes[f as usize].refused.insert((c, term));
                self.stop_following(s, f);
                return;
            }
            let len = log.len() as u8;
            self.follow(s, f, c, log, proof);
            self.log_ack(s, f, c, term, len, false);
            return;
        }
        // A new election, or a heal: an election that extends this node's log at the next
        // counter is accepted once its own coordinator's lease is gone; any other switch needs a
        // proof that beats its coordinator's under the heal order (§7.3, §7.5).
        // The election this node last applied, kept with its proof, is what a newcomer must
        // beat, whether or not its coordinator still coordinates.
        let current = my_proof.filter(|p| !node.denies(p.node));
        let am_coordinator = node.is_coordinator(f);
        let normal_election = extends
            && term.counter == last_counter(&node.log) + 1
            && last_elected(&log) == Some((c, term))
            && (current.is_none() || !node.lease_live)
            && !am_coordinator;
        let heal_switch = current.is_none_or(|p| proof.beats(&p)) && !following;
        let resync = following && !extends;
        if !(normal_election || heal_switch || resync) {
            if current.is_some_and(|p| p.supported() && !proof.supported() && proof.term > p.term) {
                s.reached |= seen::SUPPORTED_BEAT_UNSUPPORTED;
            }
            fence(s);
            return;
        }
        if !extends && !self.valid_snapshot(&log, &proof) {
            // `validate_snapshot()` failed: refuse it, report `CoordinatorMisbehaved` and treat
            // the sender's lease as expired.
            s.reached |= seen::FORGED_SNAPSHOT_REFUSED;
            s.nodes[f as usize].refused.insert((c, term));
            if following {
                self.stop_following(s, f);
            }
            return;
        }
        if extends && !log[node.log.len()..].iter().all(|e| self.valid_entry(e)) {
            s.reached |= seen::FORGED_REFUSED;
            s.nodes[f as usize].refused.insert((c, term));
            return;
        }
        let len = log.len() as u8;
        let endorse = !following;
        self.follow(s, f, c, log, proof);
        self.log_ack(s, f, c, term, len, endorse);
    }

    fn on_fence(&self, s: &mut State, c: Ix, proof: Option<Proof>) {
        let node = &s.nodes[c as usize];
        if !node.is_coordinator(c) || !self.honest(c) {
            return;
        }
        let Some(p) = proof else {
            // A bare term is ignored and logged as `ForgedTerm`.
            s.reached |= seen::FORGED_TERM_IGNORED;
            return;
        };
        let Some(mine) = node.proof else {
            return;
        };
        if !self.verifies(s, c, &p) {
            s.reached |= seen::FORGED_TERM_IGNORED;
            return;
        }
        // Its own earlier election never deposes it: it cannot step down to follow itself.
        if mine.beats(&p) || p == mine || p.node == c {
            if mine.supported() && !p.supported() && p.term.counter > mine.term.counter {
                s.reached |= seen::SUPPORTED_BEAT_UNSUPPORTED;
            }
            return;
        }
        // A verified winning proof: step down, follow the winner and sync with it.
        let n = &mut s.nodes[c as usize];
        n.coordinator = Some(p.node);
        n.term = Some(p.term);
        n.proof = Some(p);
        n.lease_live = false;
        n.pending.clear();
        self.record_accept(s, c, p.term, p.node);
    }

    fn on_revocation(&self, s: &mut State, f: Ix, from: Ix, node: Ix) {
        // A revocation verifies when its signer is a keyholder not itself revoked. The target
        // never applies its own revocation, and a byzantine target ignores it.
        // The signature is the keyholder's, whoever forwards it.
        let _ = from;
        let keyholder = self.cfg.revocation.map(|(k, _)| k);
        let signer_ok = keyholder
            .is_some_and(|k| self.cfg.keyholder[k as usize] && !s.nodes[f as usize].denies(k));
        if !signer_ok || f == node {
            return;
        }
        let n = &mut s.nodes[f as usize];
        n.denied |= bit(node);
        if n.coordinator == Some(node) {
            if !s.nodes[node as usize]
                .log
                .contains(&Entry::Revoked { node })
            {
                s.reached |= seen::REVOCATION_OMITTED_BUT_HELD;
            }
            self.stop_following(s, f);
        }
    }

    pub fn apply(&self, s: &mut State, action: Action) {
        match action {
            Action::Deliver(m) => self.deliver(s, m),
            Action::Drop(m) => {
                s.net.remove(&m);
                s.losses_left -= 1;
            }
            Action::Expire(f) => {
                let node = &s.nodes[f as usize];
                let silent = node.coordinator.is_some_and(|c| !self.beats_reach(s, c, f));
                if !silent {
                    s.losses_left -= 1;
                }
                s.nodes[f as usize].lease_live = false;
            }
            Action::Claim(y) => {
                let term = self.next_term(s, y);
                let node = &mut s.nodes[y as usize];
                node.claim = Some((term, 0));
                node.claims_left -= 1;
                node.seen.insert((term.counter, y));
                for v in 0..self.cfg.n {
                    if v != y {
                        self.send(
                            s,
                            Msg::Claim {
                                from: y,
                                to: v,
                                term,
                            },
                        );
                    }
                }
            }
            Action::Win(y) => self.win(s, y),
            Action::Replicate(c, f) => {
                let node = &s.nodes[c as usize];
                let (Some(term), Some(proof)) = (node.term, node.proof) else {
                    return;
                };
                let log = node.log.clone();
                self.send(
                    s,
                    Msg::Append {
                        from: c,
                        to: f,
                        term,
                        log,
                        proof,
                    },
                );
            }
            Action::Resubmit(o, id) => self.resubmit(s, o, id),
            Action::ResubmitRevocation(k) => {
                let node = &s.nodes[k as usize];
                if let (Some(target), Some(c)) = (node.kh_revoked, node.coordinator) {
                    self.send(
                        s,
                        Msg::RevokeCmd {
                            from: k,
                            to: c,
                            node: target,
                        },
                    );
                }
            }
            Action::Split(k) => {
                s.splits_left -= 1;
                s.cut.extend(self.cfg.splits[k as usize].iter().copied());
                let cut = s.cut.clone();
                s.net.retain(|m| {
                    let (a, b) = m.ends();
                    !cut.contains(&(a.min(b), a.max(b)))
                });
            }
            Action::Heal => {
                s.cut.clear();
                if !s.split_elected.is_empty() {
                    s.reached |= seen::HEALED_AFTER_SPLIT_ELECTION;
                }
            }
            Action::Restart(r) => {
                // The log, the record of the last `CoordinatorElected` it applied with its proof,
                // the journal and the deny list are on disk. A former coordinator does not
                // resume office: it follows, or claims anew (§7.1).
                let node = &mut s.nodes[r as usize];
                node.alive = true;
                if node.coordinator == Some(r) {
                    node.coordinator = None;
                }
                node.lease_live = false;
                node.claim = None;
                node.seen.clear();
                node.pending.clear();
            }
            Action::Crash(c) => {
                s.crashes_left -= 1;
                s.nodes[c as usize].alive = false;
                s.net.retain(|m| m.ends().1 != c);
            }
            Action::Revoke => {
                s.revoke_issued = true;
                let (k, target) = self.cfg.revocation.expect("a revocation scenario");
                let node = &mut s.nodes[k as usize];
                node.denied |= bit(target);
                node.kh_revoked = Some(target);
                if node.coordinator == Some(target) {
                    self.stop_following(s, k);
                }
                for peer in 0..self.cfg.n {
                    if peer != k && peer != target {
                        self.send(
                            s,
                            Msg::Revocation {
                                from: k,
                                to: peer,
                                node: target,
                            },
                        );
                    }
                }
            }
            Action::Gossip(a, b, x) => self.send(
                s,
                Msg::Revocation {
                    from: a,
                    to: b,
                    node: x,
                },
            ),
            Action::ForgeAuthority => {
                s.forges_left -= 1;
                let b = self.cfg.byzantine.expect("a byzantine scenario");
                s.nodes[b as usize]
                    .log
                    .push(Entry::Authority { sig_ok: false });
            }
            Action::ForgeTerm(to, with_proof) => {
                s.forges_left -= 1;
                let b = self.cfg.byzantine.expect("a byzantine scenario");
                let forged = Proof {
                    node: b,
                    term: Term {
                        counter: self.cfg.max_counter + 1,
                        tag: u8::MAX,
                    },
                    sig_ok: false,
                    support: u8::MAX,
                };
                if with_proof {
                    let log = s.nodes[b as usize].log.clone();
                    self.send(
                        s,
                        Msg::Append {
                            from: b,
                            to,
                            term: forged.term,
                            log,
                            proof: forged,
                        },
                    );
                    self.send(
                        s,
                        Msg::Fence {
                            from: b,
                            to,
                            proof: Some(forged),
                        },
                    );
                } else {
                    self.send(
                        s,
                        Msg::Fence {
                            from: b,
                            to,
                            proof: None,
                        },
                    );
                }
            }
        }
    }

    /// Sends a journalled command to the coordinator, or appends it when this node leads.
    fn resubmit(&self, s: &mut State, o: Ix, id: u8) {
        let node = &s.nodes[o as usize];
        match node.coordinator {
            Some(c) if c == o && node.is_coordinator(o) => {
                self.coordinator_append_command(s, o, id)
            }
            Some(c) if c != o => self.send(s, Msg::Command { from: o, to: c, id }),
            _ => {}
        }
    }
}
