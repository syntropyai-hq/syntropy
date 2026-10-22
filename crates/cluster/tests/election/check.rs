// SPDX-License-Identifier: MIT OR Apache-2.0

//! The properties of RFC-0032 Testing's "Election safety and liveness" row, one named
//! [`Property`] each, and the transition oracle that records violations no single state shows.

use stateright::{Model, Property};

use crate::model::{
    Action, ElectionModel, Entry, Ix, Msg, Node, Proof, State, Term, bit, flag, seen,
};

impl Model for ElectionModel {
    type State = State;
    type Action = Action;

    fn init_states(&self) -> Vec<State> {
        vec![self.initial()]
    }

    fn actions(&self, s: &State, out: &mut Vec<Action>) {
        self.protocol_actions(s, out);
        self.fault_actions(s, out);
    }

    fn next_state(&self, s: &State, action: Action) -> Option<State> {
        let mut next = s.clone();
        self.apply(&mut next, action.clone());
        oracle(self, s, &action, &mut next);
        // Only the situations this scenario must reach are kept, so the others do not split
        // otherwise equal states.
        next.reached &= self.cfg.must_reach;
        (next != *s).then_some(next)
    }

    fn properties(&self) -> Vec<Property<Self>> {
        PROPERTIES
            .iter()
            .map(|&(name, condition)| Property::always(name, condition))
            .chain([Property::sometimes(REACHES, reaches)])
            .collect()
    }
}

/// The scenario's situations, reached on some path, so its properties are not vacuous.
pub const REACHES: &str = "the scenario reaches its situations";

fn reaches(m: &ElectionModel, s: &State) -> bool {
    s.reached & m.cfg.must_reach == m.cfg.must_reach
}

type Condition = fn(&ElectionModel, &State) -> bool;

/// The 13 properties, by their names in RFC-0032 Testing.
pub const PROPERTIES: [(&str, Condition); 13] = [
    (TWO_AT_ONE_TERM, |_, s| s.flags & flag::TWO_AT_ONE_TERM == 0),
    ("after heal every node's chain converges", |m, s| {
        !settled(m, s) || converged(m, s).is_some()
    }),
    ("a lone survivor elects itself", lone_survivor),
    (
        "a partial partition never yields two coordinators that a single node accepts",
        partial_partition,
    ),
    (
        "an aborted claim never deposes a live coordinator",
        |_, s| s.flags & flag::ABORTED_CLAIM_DEPOSED == 0,
    ),
    (
        "a node without COORDINATOR never becomes coordinator",
        no_role_never_leads,
    ),
    (
        "a forged authority entry is never applied by an honest node",
        |m, s| s.flags & flag::FORGED_APPLIED == 0 && honest_logs_clean(m, s),
    ),
    (
        "a snapshot whose authority records fail verification is never adopted",
        |_, s| s.flags & flag::FORGED_SNAPSHOT_ADOPTED == 0,
    ),
    (
        "a revocation of the coordinator stops its followers even when the coordinator omits it from the log",
        revoked_never_followed,
    ),
    ("a heal at equal counters converges", |m, s| {
        s.flags & flag::HEAL_UNDECIDED == 0
            && (!equal_counter_split(s) || !settled(m, s) || converged(m, s).is_some())
    }),
    (
        "a bare or forged term, from any member, never deposes a coordinator",
        |_, s| s.flags & flag::FORGED_TERM_DEPOSED == 0,
    ),
    (
        "an unsupported self-election never deposes a supported coordinator",
        |_, s| s.flags & flag::UNSUPPORTED_DEPOSED_SUPPORTED == 0,
    ),
    (COMMANDS_SURVIVE, commands_survive),
];

/// The first property. Two claimants share a term only when their 24-bit tags collide, and the
/// heal order then sends the tie to the lower node ID (RFC-0032 §7.2, §7.5), so a node that
/// followed the other switches within that term: the tag-collision scenario expects it to fail.
pub const TWO_AT_ONE_TERM: &str = "no two coordinators accepted by one node at the same term";

/// The 13th property, which a scenario that loses nodes with their disks is expected to fail
/// (OQ-NET-33).
pub const COMMANDS_SURVIVE: &str = "every command acknowledged with replicated = true, or kept in a live originator's journal, appears in the converged log";

/// Honest nodes, by index.
fn honest(m: &ElectionModel) -> impl Iterator<Item = Ix> + '_ {
    (0..m.cfg.n).filter(|&i| m.honest(i))
}

/// The protocol has nothing left to do: no message can be delivered and no protocol action is
/// enabled. Faults may still be possible; they are not the protocol's work.
pub fn quiescent(m: &ElectionModel, s: &State) -> bool {
    let mut actions = Vec::new();
    m.protocol_actions(s, &mut actions);
    actions.is_empty()
}

/// A state in which liveness is judged: the network is whole, the protocol is quiescent, and no
/// honest node that needs an election is stopped only by the scenario's bounds on claims. A node
/// with a live honest coordinator in reach needs no election: that coordinator brings it in.
pub fn settled(m: &ElectionModel, s: &State) -> bool {
    let coordinator_in_reach = |i: Ix| {
        honest(m).any(|c| {
            let n = &s.nodes[c as usize];
            n.alive && n.is_coordinator(c) && m.link(s, i, c)
        })
    };
    s.cut.is_empty()
        && quiescent(m, s)
        && !honest(m).any(|i| {
            let n = &s.nodes[i as usize];
            n.alive
                && m.cfg.role[i as usize]
                && !n.lease_live
                && n.claim.is_none()
                && !n.is_coordinator(i)
                && !m.may_claim(s, i)
                && !coordinator_in_reach(i)
        })
}

/// One coordinator that every live honest node follows, at its term, with its log and a live
/// lease. Returns it.
pub fn converged(m: &ElectionModel, s: &State) -> Option<Ix> {
    let live: Vec<Ix> = honest(m).filter(|&i| s.nodes[i as usize].alive).collect();
    let c = s.nodes[*live.first()? as usize].coordinator?;
    let leader = &s.nodes[c as usize];
    (leader.alive && leader.is_coordinator(c)).then_some(())?;
    live.iter()
        .all(|&i| {
            let n = &s.nodes[i as usize];
            n.coordinator == Some(c) && n.term == leader.term && n.log == leader.log && n.lease_live
        })
        .then_some(c)
}

fn lone_survivor(m: &ElectionModel, s: &State) -> bool {
    let live: Vec<Ix> = (0..m.cfg.n)
        .filter(|&i| s.nodes[i as usize].alive)
        .collect();
    match live.as_slice() {
        [only] if m.cfg.role[*only as usize] && settled(m, s) => {
            s.nodes[*only as usize].is_coordinator(*only)
        }
        _ => true,
    }
}

/// With a split that keeps everyone connected through some member, and no loss or crash, no node
/// accepts any coordinator but the one it started with.
fn partial_partition(m: &ElectionModel, s: &State) -> bool {
    !m.cfg.partial_partition || honest(m).all(|i| s.accepted[i as usize].values().all(|&c| c == 0))
}

fn no_role_never_leads(m: &ElectionModel, s: &State) -> bool {
    honest(m).all(|i| {
        s.nodes[i as usize]
            .coordinator
            .is_none_or(|c| m.cfg.role[c as usize])
            && s.accepted[i as usize]
                .values()
                .all(|&c| m.cfg.role[c as usize])
    })
}

fn honest_logs_clean(m: &ElectionModel, s: &State) -> bool {
    honest(m).all(|i| {
        !s.nodes[i as usize]
            .log
            .contains(&Entry::Authority { sig_ok: false })
    })
}

fn revoked_never_followed(m: &ElectionModel, s: &State) -> bool {
    honest(m).all(|i| {
        let n = &s.nodes[i as usize];
        !n.alive || n.coordinator.is_none_or(|c| c == i || !n.denies(c))
    })
}

/// Two nodes were elected at one counter on the two sides of a split.
fn equal_counter_split(s: &State) -> bool {
    s.split_elected.iter().any(|&(a, ta)| {
        s.split_elected
            .iter()
            .any(|&(b, tb)| a != b && ta.counter == tb.counter)
    })
}

fn commands_survive(m: &ElectionModel, s: &State) -> bool {
    if !settled(m, s) {
        return true;
    }
    let Some(c) = converged(m, s) else {
        return true;
    };
    let log = &s.nodes[c as usize].log;
    let journalled = (0..m.cfg.n)
        .filter(|&i| s.nodes[i as usize].alive)
        .flat_map(|i| s.nodes[i as usize].journal.iter().copied());
    s.acked_replicated
        .iter()
        .copied()
        .chain(journalled)
        .all(|id| log.contains(&Entry::Command { id }))
}

/// Records what no single state shows: how each honest node's log and coordinator changed in
/// this step, and why.
pub fn oracle(m: &ElectionModel, before: &State, action: &Action, after: &mut State) {
    let forged = Entry::Authority { sig_ok: false };
    // Voters that ack a claim, with what they followed when they acked; when the claim ends
    // without a win, they become aborted ackers (§7.3: "aborted claims change nothing").
    if let Action::Deliver(Msg::Claim { from, to, term }) = action {
        let (b, a) = (&before.nodes[*from as usize], &after.nodes[*from as usize]);
        let acked = |n: &Node| {
            n.claim
                .is_some_and(|(t, acks)| t == *term && acks & bit(*to) != 0)
        };
        let voter = &before.nodes[*to as usize];
        if !acked(b)
            && acked(a)
            && let (Some(c), Some(t)) = (voter.coordinator, voter.term)
            && c != *from
        {
            after.ackers.insert((*from, *term, *to), (c, t));
        }
    }
    for y in 0..m.cfg.n {
        if let Some((term, _)) = before.nodes[y as usize].claim
            && after.nodes[y as usize].claim.is_none()
        {
            let won = after.elections.contains(&(y, term));
            let ended: Vec<(Ix, Term, Ix)> = after
                .ackers
                .keys()
                .copied()
                .filter(|&(c, t, _)| c == y && t == term)
                .collect();
            for key in ended {
                let context = after.ackers.remove(&key).expect("listed");
                if !won {
                    after.aborted_ackers.insert(key.2, context);
                }
            }
        }
    }
    // The heal order must decide between any two different elections (§7.5); a tie leaves both
    // coordinators standing, and the heal never converges.
    let undecided =
        |p: &Proof, q: &Proof| (p.node, p.term) != (q.node, q.term) && !p.beats(q) && !q.beats(p);
    match action {
        Action::Deliver(Msg::Fence {
            to, proof: Some(p), ..
        })
        | Action::Deliver(Msg::Append { to, proof: p, .. }) => {
            if let Some(mine) = before.nodes[*to as usize].proof
                && m.verifies(before, *to, p)
                && undecided(p, &mine)
            {
                after.flags |= flag::HEAL_UNDECIDED;
            }
        }
        _ => {}
    }
    if let Action::Deliver(Msg::Append {
        from,
        to,
        term,
        log,
        proof,
    }) = action
        && before.aborted_ackers.get(to) == Some(&(*from, *term))
    {
        let voter = &before.nodes[*to as usize];
        let valid = proof.sig_ok
            && m.cfg.role[*from as usize]
            && before.elections.contains(&(*from, *term))
            && !voter.denies(*from)
            && log.starts_with(&voter.log)
            && log[voter.log.len()..].iter().all(|e| *e != forged);
        let after_node = &after.nodes[*to as usize];
        if valid && (after_node.coordinator != Some(*from) || after_node.term != Some(*term)) {
            // A voter that acked an aborted claim refused its live coordinator's beat.
            after.flags |= flag::ABORTED_CLAIM_DEPOSED;
        }
    }
    for i in 0..m.cfg.n {
        if !m.honest(i) {
            continue;
        }
        let (b, a) = (&before.nodes[i as usize], &after.nodes[i as usize]);
        if !b.log.contains(&forged) && a.log.contains(&forged) {
            after.flags |= if a.log.starts_with(&b.log) {
                flag::FORGED_APPLIED
            } else {
                flag::FORGED_SNAPSHOT_ADOPTED
            };
        }
        // A deposition: the node heard a coordinator, or was one, and no longer follows it.
        let Some(old) = b.coordinator.filter(|_| b.lease_live) else {
            continue;
        };
        if a.coordinator == Some(old) && a.term == b.term {
            if let Action::Deliver(Msg::Append {
                from, term, log, ..
            }) = action
                && *from == old
                && Some(*term) == b.term
                && log.starts_with(&b.log)
                && log[b.log.len()..].iter().all(|e| *e != forged)
                && !a.lease_live
            {
                // A valid beat from its own coordinator, at its own term, refused.
                after.flags |= flag::ABORTED_CLAIM_DEPOSED;
            }
            continue;
        }
        match action {
            Action::Deliver(Msg::Claim { .. }) => {
                after.flags |= flag::ABORTED_CLAIM_DEPOSED;
            }
            Action::Deliver(Msg::Fence { proof: None, .. }) => {
                after.flags |= flag::FORGED_TERM_DEPOSED;
            }
            Action::Deliver(Msg::Fence { proof: Some(p), .. })
            | Action::Deliver(Msg::Append { proof: p, .. }) => {
                let switched_to = a.coordinator;
                if switched_to == Some(p.node) {
                    let genuine = p.sig_ok
                        && m.cfg.role[p.node as usize]
                        && after.elections.contains(&(p.node, p.term));
                    if !genuine {
                        after.flags |= flag::FORGED_TERM_DEPOSED;
                    }
                    if b.proof.is_some_and(|old| old.supported()) && !p.supported() {
                        after.flags |= flag::UNSUPPORTED_DEPOSED_SUPPORTED;
                    }
                }
            }
            _ => {}
        }
    }
    let live: Vec<Ix> = (0..m.cfg.n)
        .filter(|&i| after.nodes[i as usize].alive)
        .collect();
    if let [only] = live.as_slice()
        && after.nodes[*only as usize].is_coordinator(*only)
    {
        after.reached |= seen::LONE_SURVIVOR_LEADS;
    }
}
