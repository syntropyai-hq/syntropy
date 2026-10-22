// SPDX-License-Identifier: MIT OR Apache-2.0

//! RFC-0034 §14's keys: `security.*`, `integrity.*`, and RFC-0034's keys inside other
//! sections' prefixes: `rdma.encrypt_untrusted` (RFC-0031's `rdma.*`), `invite.require_confirm`
//! (RFC-0030's `invite.*`) and `models.signatures` and `models.trusted_signers` (`models.*`,
//! which RFC-0043 shares).

use serde::Deserialize;
use syntropy_config::{ConfigSection, Duration, HoldsSecret};

use super::FEATURE_LEVEL_V1;

/// Where node and session keys are kept (RFC-0034 §4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, HoldsSecret)]
#[serde(rename_all = "snake_case")]
pub enum Keystore {
    /// Wrapped by the OS keystore where the platform offers one, else a file.
    Auto,
    /// A mode-0600 file in the state directory.
    File,
}

/// How much of a loaded pack is re-hashed (RFC-0034 §10.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, HoldsSecret)]
#[serde(rename_all = "snake_case")]
pub enum VerifyOnLoad {
    /// A random 1 % of extents, plus every extent the warm-up canary touches.
    Sample,
    /// Every extent.
    Full,
    /// None.
    Off,
}

/// Whether model signatures are checked (RFC-0034 §10.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, HoldsSecret)]
#[serde(rename_all = "snake_case")]
pub enum Signatures {
    /// Never checked.
    Off,
    /// Checked when the model carries one.
    VerifyIfPresent,
    /// Required, from a signer in `models.trusted_signers`.
    Require,
}

/// Keys, certificates and trusted interfaces (RFC-0034 §4, §5, §8, §9). Keyholder authority
/// for the cluster-scope keys under RFC-0054 §2.3 step 10.
#[derive(Clone, Debug, PartialEq, Deserialize, ConfigSection)]
#[config(section = "security", scope = Node, owner = "RFC-0034")]
pub struct SecuritySection {
    /// Where node and session keys are kept.
    #[config(reload = Node, default = "auto")]
    pub keystore: Keystore,
    /// How long an admission certificate is valid (RFC-0034 §5.3).
    #[config(reload = Hot, default = "180d", scope = Cluster, introduced_at = FEATURE_LEVEL_V1)]
    pub cert_lifetime: Duration,
    /// The clock skew certificate checks tolerate (RFC-0034 §5.4).
    #[config(reload = Hot, default = "10m", scope = Cluster, introduced_at = FEATURE_LEVEL_V1)]
    pub clock_skew: Duration,
    /// After a cluster key rotation, how long certificates signed by the old key are still
    /// accepted (RFC-0034 §8.2).
    #[config(reload = Hot, default = "7d", scope = Cluster, introduced_at = FEATURE_LEVEL_V1)]
    pub rotation_window: Duration,
    /// Interfaces or CIDRs trusted for plaintext RoCEv2 (RFC-0034 §9.2; ADR-0011 decision 4).
    #[config(reload = Node, default = [])]
    pub trusted_fabric: Vec<String>,
}

/// RFC-0034's key inside RFC-0031's `rdma.*` (RFC-0054 §5.1).
#[derive(Clone, Debug, PartialEq, Deserialize, ConfigSection)]
#[config(section = "rdma", shared, scope = Node, owner = "RFC-0034")]
pub struct RdmaSecuritySection {
    /// Encrypt RDMA on interfaces outside `trusted_fabric` (M4; RFC-0034 §9.3).
    #[config(reload = Node, default = false)]
    pub encrypt_untrusted: bool,
}

/// RFC-0034's key inside RFC-0030's `invite.*`.
#[derive(Clone, Debug, PartialEq, Deserialize, ConfigSection)]
#[config(section = "invite", shared, scope = Cluster, owner = "RFC-0034")]
pub struct InviteSecuritySection {
    /// Whether an invite-token join also needs a keyholder's confirmation (RFC-0034 §7).
    #[config(reload = Hot, default = false, introduced_at = FEATURE_LEVEL_V1)]
    pub require_confirm: bool,
}

/// Load-time integrity checks (RFC-0034 §10.2).
#[derive(Clone, Debug, PartialEq, Deserialize, ConfigSection)]
#[config(section = "integrity", scope = Node, owner = "RFC-0034")]
pub struct IntegritySection {
    /// How much of a loaded pack is re-hashed.
    #[config(reload = Hot, default = "sample")]
    pub verify_on_load: VerifyOnLoad,
}

/// RFC-0034's keys inside `models.*`, which RFC-0043 shares (RFC-0054 §5.1).
#[derive(Clone, Debug, PartialEq, Deserialize, ConfigSection)]
#[config(section = "models", shared, scope = Node, owner = "RFC-0034")]
pub struct ModelsSecuritySection {
    /// Whether model signatures are checked.
    #[config(reload = Hot, default = "verify_if_present")]
    pub signatures: Signatures,
    /// The identities or keys `require` accepts.
    #[config(reload = Hot, default = [])]
    pub trusted_signers: Vec<String>,
}
