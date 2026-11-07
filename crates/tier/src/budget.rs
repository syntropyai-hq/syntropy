// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pin ceilings, the VRAM split and the ledger check (RFC-0020 §5, §8.1, §12). This task adds
//! the ledger check (TIER-E1-T02); the ceilings and the split come in TIER-E2-T01.

#![forbid(unsafe_code)]

use syntropy_core::error::codes;
use syntropy_core::ledger::LedgerGrant;
use syntropy_core::{Error, ErrorClass, RetryHint};

use crate::arena::ArenaPlan;

/// The ledger check of `Fetching` (RFC-0020 §12; OQ-IO-27): every tier's
/// [`TierBudgetLine::reserved`](crate::arena::TierBudgetLine::reserved) must be at most the
/// bytes `grant` gives that tier. A tier with no grant line is granted nothing, so any bytes
/// on it fail. The headroom is not counted: it is what the tier keeps free, not what the
/// runner holds.
///
/// # Errors
///
/// `TIER_LEDGER_EXCEEDED` (`Tier 0x0701`), class `Capacity`, for the first tier in the plan's
/// order that is over its grant, with the tier, the reserved and the granted bytes in its
/// context. The coordinator re-plans with the ledger's figures (RFC-0020 §15), so the retry
/// hint is `OnNewPlan`.
#[expect(
    clippy::result_large_err,
    reason = "core::Error is RFC-0001 §5.12's struct with its fields inline; the ledger check fails once, in Fetching"
)]
pub fn check_ledger(ap: &ArenaPlan, grant: &LedgerGrant) -> Result<(), Error> {
    for (tier, line) in &ap.per_tier {
        let granted = grant.granted(*tier);
        if line.reserved > granted {
            return Err(Error::new(
                ErrorClass::Capacity,
                codes::tier::LEDGER_EXCEEDED,
                format!(
                    "the arena plan reserves {} bytes on tier {} but the ledger grants {}",
                    line.reserved.0, tier.ordinal, granted.0
                ),
            )
            .with_retry(RetryHint::OnNewPlan)
            .with_context("tier", tier.ordinal.to_string())
            .with_context("node", format!("{:016x}", tier.node.short()))
            .with_context("reserved", line.reserved.0.to_string())
            .with_context("granted", granted.0.to_string()));
        }
    }
    Ok(())
}
