// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `Config` error codes of RFC-0054 §8, and the edit-distance suggestion for an unknown key.
//!
//! The codes are stable and never reused. RFC-0001 §5.12's `Error` carries them as
//! `(Config, n)`; until `core` defines that type, [`Code::number`] and [`Code::name`] are the
//! contract, and the renderer prints `config/0003`.

use std::fmt;

use crate::registry::Registry;
use crate::value::ValueErrorKind;

/// An error code in the `Config` domain (RFC-0054 §8).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Code {
    /// 1: the file is not valid TOML.
    Syntax,
    /// 2: a value has the wrong type or shape for its key.
    WrongType,
    /// 3: no section declares the key.
    UnknownKey,
    /// 4: a value lies outside its key's range.
    OutOfRange,
    /// 5: the unit syntax of §3 is wrong.
    BadUnit,
    /// 6: a selector does not parse, or is not allowed where it appears.
    BadSelector,
    /// 7: a regular expression does not compile or exceeds its limits.
    BadRegex,
    /// 8: an approximate knob, or `default_fidelity = "approximate"`, without
    /// `fidelity = "approximate"`.
    KnobWithoutApproximate,
    /// 9: a key appears where its scope does not allow it.
    ScopeMismatch,
    /// 10: a listener's bind, key or TLS settings are not allowed together.
    InsecureListener,
    /// 11: a path does not exist, or cannot be read or written as the key needs.
    PathUnusable,
    /// 12: a tensor override selects part of a tensor group.
    OverrideSplitsGroup,
    /// 13: the coordinator rejected a cluster configuration proposal.
    ClusterProposalRejected,
    /// 14: a reload applied some keys and deferred others (informational).
    ReloadPartiallyDeferred,
    /// 15: a conformance manifest's signer is not allowed.
    ManifestSignerNotAllowed,
    /// 16: an alias has the name of a registered model.
    AliasShadowsModel,
}

impl Code {
    /// Every code, in number order.
    pub const ALL: [Code; 16] = [
        Code::Syntax,
        Code::WrongType,
        Code::UnknownKey,
        Code::OutOfRange,
        Code::BadUnit,
        Code::BadSelector,
        Code::BadRegex,
        Code::KnobWithoutApproximate,
        Code::ScopeMismatch,
        Code::InsecureListener,
        Code::PathUnusable,
        Code::OverrideSplitsGroup,
        Code::ClusterProposalRejected,
        Code::ReloadPartiallyDeferred,
        Code::ManifestSignerNotAllowed,
        Code::AliasShadowsModel,
    ];

    /// The code's number in the `Config` domain.
    pub const fn number(self) -> u16 {
        match self {
            Code::Syntax => 1,
            Code::WrongType => 2,
            Code::UnknownKey => 3,
            Code::OutOfRange => 4,
            Code::BadUnit => 5,
            Code::BadSelector => 6,
            Code::BadRegex => 7,
            Code::KnobWithoutApproximate => 8,
            Code::ScopeMismatch => 9,
            Code::InsecureListener => 10,
            Code::PathUnusable => 11,
            Code::OverrideSplitsGroup => 12,
            Code::ClusterProposalRejected => 13,
            Code::ReloadPartiallyDeferred => 14,
            Code::ManifestSignerNotAllowed => 15,
            Code::AliasShadowsModel => 16,
        }
    }

    /// The code's name as RFC-0054 §8 writes it, such as `unknown_key`.
    pub const fn name(self) -> &'static str {
        match self {
            Code::Syntax => "syntax",
            Code::WrongType => "wrong_type",
            Code::UnknownKey => "unknown_key",
            Code::OutOfRange => "out_of_range",
            Code::BadUnit => "bad_unit",
            Code::BadSelector => "bad_selector",
            Code::BadRegex => "bad_regex",
            Code::KnobWithoutApproximate => "knob_without_approximate",
            Code::ScopeMismatch => "scope_mismatch",
            Code::InsecureListener => "insecure_listener",
            Code::PathUnusable => "path_unusable",
            Code::OverrideSplitsGroup => "override_splits_group",
            Code::ClusterProposalRejected => "cluster_proposal_rejected",
            Code::ReloadPartiallyDeferred => "reload_partially_deferred",
            Code::ManifestSignerNotAllowed => "manifest_signer_not_allowed",
            Code::AliasShadowsModel => "alias_shadows_model",
        }
    }
}

/// `config/0003`, as an error's header prints it.
impl fmt::Display for Code {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "config/{:04}", self.number())
    }
}

/// The code a rejected value gets in phase 3.
impl From<ValueErrorKind> for Code {
    fn from(kind: ValueErrorKind) -> Self {
        match kind {
            ValueErrorKind::OutOfRange => Code::OutOfRange,
            ValueErrorKind::BadUnit => Code::BadUnit,
            ValueErrorKind::BadSelector => Code::BadSelector,
            ValueErrorKind::BadRegex => Code::BadRegex,
            ValueErrorKind::SecretUnavailable => Code::PathUnusable,
            _ => Code::WrongType,
        }
    }
}

/// The registered key closest to `path`, for an unknown key's "did you mean" (RFC-0054 §8).
///
/// Distance is Levenshtein over the whole path's characters. A suggestion is offered only within
/// a third of the length of `path`'s last segment, and at least one edit: a shared table prefix
/// such as `example.` would otherwise let a different word in the last segment count as a typo.
/// Ties go to the first key in registration order.
pub fn suggest(registry: &Registry, path: &str) -> Option<&'static str> {
    let last = path.rsplit('.').next().unwrap_or(path).chars().count();
    let limit = (last / 3).max(1);
    let mut best: Option<(&'static str, usize)> = None;
    for key in registry.keys() {
        let distance = levenshtein(path, key.path);
        if distance == 0 || distance > limit {
            continue;
        }
        if best.is_none_or(|(_, d)| distance < d) {
            best = Some((key.path, distance));
        }
    }
    best.map(|(path, _)| path)
}

/// Edit distance between two strings, in characters.
fn levenshtein(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut diagonal = row[0];
        row[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let above = row[j + 1];
            row[j + 1] = (above + 1)
                .min(row[j] + 1)
                .min(diagonal + usize::from(ca != cb));
            diagonal = above;
        }
    }
    row[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_and_names_match_rfc_0054() {
        let expected = [
            (1, "syntax"),
            (2, "wrong_type"),
            (3, "unknown_key"),
            (4, "out_of_range"),
            (5, "bad_unit"),
            (6, "bad_selector"),
            (7, "bad_regex"),
            (8, "knob_without_approximate"),
            (9, "scope_mismatch"),
            (10, "insecure_listener"),
            (11, "path_unusable"),
            (12, "override_splits_group"),
            (13, "cluster_proposal_rejected"),
            (14, "reload_partially_deferred"),
            (15, "manifest_signer_not_allowed"),
            (16, "alias_shadows_model"),
        ];
        let actual: Vec<(u16, &str)> = Code::ALL.iter().map(|c| (c.number(), c.name())).collect();
        assert_eq!(actual, expected);
        assert_eq!(Code::UnknownKey.to_string(), "config/0003");
        assert_eq!(Code::AliasShadowsModel.to_string(), "config/0016");
    }

    #[test]
    fn edit_distance() {
        assert_eq!(levenshtein("ram_headroon", "ram_headroom"), 1);
        assert_eq!(levenshtein("", "abc"), 3);
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("same", "same"), 0);
    }
}
