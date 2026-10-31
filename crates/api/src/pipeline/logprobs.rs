// SPDX-License-Identifier: MIT OR Apache-2.0

//! What ingress asks the tail for when a request wants the chosen tokens' logprobs alone, as
//! OpenAI's `logprobs: true` without `top_logprobs` does (RFC-0050 §3.1's note on
//! `top_logprobs = 0`; RFC-0031 §6.5, OQ-NET-35).
//!
//! `SamplingParams::logprobs = Some(0)` asks for them. A `Token` entry carries them as
//! `logprob_k = 0x80`, which a peer built before capability bit 16 `CHOSEN_LOGPROBS` would
//! misread, so `net` writes it only on a channel that negotiated the bit. The entry crosses two
//! of the tail's channels: the copy to the ingress, and the ring-return copy to the head, which
//! carries the logprobs too. Ingress asks for `Some(0)` only when every one of them that crosses
//! `net` has the bit ([`TailChannels::ask`]). Otherwise it asks for one alternative and drops it
//! before answering ([`answer_logprobs`]). The chosen token's logprob is the same value either
//! way, so the response stays exact.

use syntropy_core::LogprobBlock;
use syntropy_net::frame::capability;

/// The tail's channels that a `Token` entry with logprobs travels, each as its negotiated
/// capabilities (RFC-0031 §10), or `None` where the entry does not cross `net`. The default is
/// a tail in the ingress's own process.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct TailChannels {
    /// The tail's channel to this ingress. `None` when the tail runs in this ingress's process,
    /// so its entries arrive as `core` types.
    pub to_ingress: Option<u64>,
    /// The tail's channel to the head, which the ring-return copy travels. `None` when the tail
    /// is the head, or runs in the head's process.
    pub to_head: Option<u64>,
}

impl TailChannels {
    /// Whether every channel the entry crosses negotiated capability bit 16 `CHOSEN_LOGPROBS`.
    #[must_use]
    pub fn chosen_logprobs(&self) -> bool {
        [self.to_ingress, self.to_head]
            .into_iter()
            .flatten()
            .all(|caps| caps & capability::CHOSEN_LOGPROBS != 0)
    }

    /// `SamplingParams::logprobs` to ask the tail for, for a request that asked for
    /// `requested`: `Some(0)` becomes `Some(1)` unless [`TailChannels::chosen_logprobs`];
    /// everything else is asked as it came.
    #[must_use]
    pub fn ask(&self, requested: Option<u8>) -> Option<u8> {
        match requested {
            Some(0) if !self.chosen_logprobs() => Some(1),
            other => other,
        }
    }
}

/// The logprobs that answer a request that asked for `requested`, from the tail's `block`. A
/// request for the chosen tokens' logprobs alone that [`TailChannels::ask`] asked as one
/// alternative gets the block without its alternatives; any other block is returned as it came.
/// PROD-E2-T06 writes the result into the response.
#[must_use]
pub fn answer_logprobs(requested: Option<u8>, mut block: LogprobBlock) -> LogprobBlock {
    if requested == Some(0) && block.k > 0 {
        block.k = 0;
        block.top_ids.clear();
        block.top_logprobs.clear();
    }
    block
}

#[cfg(test)]
mod tests {
    use super::*;

    const BIT: u64 = capability::CHOSEN_LOGPROBS;
    /// Every other bit, so a test that drops bit 16 keeps the rest.
    const OTHERS: u64 = !BIT;

    fn remote(to_ingress: u64, to_head: u64) -> TailChannels {
        TailChannels {
            to_ingress: Some(to_ingress),
            to_head: Some(to_head),
        }
    }

    /// DIST-E8-T18's criterion: `Some(0)` only when the tail's channels to the ingress and to
    /// the head both have bit 16; the one-alternative rewrite with the bit missing at the
    /// ingress only, and at the head only.
    #[test]
    fn some_0_needs_bit_16_on_both_channels() {
        assert_eq!(remote(BIT, BIT).ask(Some(0)), Some(0));
        assert_eq!(
            remote(OTHERS, BIT | 1).ask(Some(0)),
            Some(1),
            "ingress only"
        );
        assert_eq!(remote(BIT | 1, OTHERS).ask(Some(0)), Some(1), "head only");
        assert_eq!(remote(0, 0).ask(Some(0)), Some(1));
    }

    /// A channel the entry does not cross needs no bit: a tail in the ingress's process, or a
    /// tail that is the head.
    #[test]
    fn a_channel_in_process_needs_no_bit() {
        assert_eq!(TailChannels::default().ask(Some(0)), Some(0));
        let tail_is_head = TailChannels {
            to_ingress: Some(BIT),
            to_head: None,
        };
        assert_eq!(tail_is_head.ask(Some(0)), Some(0));
        let head_remote = TailChannels {
            to_ingress: None,
            to_head: Some(OTHERS),
        };
        assert_eq!(head_remote.ask(Some(0)), Some(1));
    }

    /// Only `Some(0)` is rewritten.
    #[test]
    fn other_requests_are_asked_as_they_came() {
        for tail in [remote(0, 0), remote(BIT, BIT), TailChannels::default()] {
            for requested in [None, Some(1), Some(5), Some(20)] {
                assert_eq!(tail.ask(requested), requested, "{tail:?} {requested:?}");
            }
        }
    }

    /// The rewritten request's alternative is dropped; the chosen logprobs stay bit for bit.
    #[test]
    fn the_alternative_is_dropped_for_a_chosen_only_request() {
        let asked = LogprobBlock {
            k: 1,
            chosen: vec![-0.5, f32::NEG_INFINITY],
            top_ids: vec![7, 8],
            top_logprobs: vec![-1.0, -2.0],
        };
        let answered = answer_logprobs(Some(0), asked.clone());
        assert_eq!(answered.k, 0);
        assert!(answered.top_ids.is_empty() && answered.top_logprobs.is_empty());
        assert!(answered.is_consistent());
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&answered.chosen), bits(&asked.chosen));
        // A request for alternatives keeps them, and a block the tail sent as asked is unchanged.
        assert_eq!(answer_logprobs(Some(1), asked.clone()), asked);
        let chosen_only = LogprobBlock {
            k: 0,
            top_ids: vec![],
            top_logprobs: vec![],
            ..asked
        };
        assert_eq!(
            bits(&answer_logprobs(Some(0), chosen_only.clone()).chosen),
            bits(&chosen_only.chosen)
        );
    }
}
