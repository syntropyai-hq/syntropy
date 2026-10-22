// SPDX-License-Identifier: MIT OR Apache-2.0

//! Whether a config value's type holds a secret anywhere inside it (RFC-0054 §10.1, G7).
//!
//! `#[derive(ConfigSection)]` records `<T as HoldsSecret>::HOLDS_SECRET` for every key in
//! [`KeySchema::holds_secret`](crate::KeySchema::holds_secret). `config show`, flag handling
//! and the diagnostics read that flag to decide what may print. They never read the type's
//! name, which a struct, a map or an array of tables would hide.
//!
//! Every key's type must implement the trait, so a type the trait has not been told about fails
//! to compile rather than print. A struct or enum used as a value derives it:
//!
//! ```
//! use serde::Deserialize;
//! use syntropy_config::{HoldsSecret, Secret};
//!
//! /// One exporter header.
//! #[derive(Deserialize, HoldsSecret)]
//! struct Header {
//!     name: String,
//!     value: Secret<String>,
//! }
//!
//! assert!(<Vec<Header> as HoldsSecret>::HOLDS_SECRET);
//! assert!(!<Vec<String> as HoldsSecret>::HOLDS_SECRET);
//! ```

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, SocketAddrV4, SocketAddrV6};
use std::path::PathBuf;

use zeroize::Zeroize;

use crate::value::{
    ByteSize, DeviceApi, DeviceSelector, Duration, Fraction, LayerRange, ModelSelector,
    NodeSelector, Rate, Regex, Reserve, Secret, SecretSource, TierSelector,
};

/// Whether values of this type can hold a secret anywhere inside them. Implement it with
/// `#[derive(HoldsSecret)]` for a struct or enum: the derive ORs the flags of every field.
pub trait HoldsSecret {
    /// `true` when some value of the type holds a [`Secret`] or a [`SecretSource`].
    const HOLDS_SECRET: bool;
}

macro_rules! no_secret {
    ($($ty:ty),* $(,)?) => {
        $(impl HoldsSecret for $ty {
            const HOLDS_SECRET: bool = false;
        })*
    };
}

no_secret!(
    bool,
    char,
    i8,
    i16,
    i32,
    i64,
    i128,
    isize,
    u8,
    u16,
    u32,
    u64,
    u128,
    usize,
    f32,
    f64,
    String,
    PathBuf,
    IpAddr,
    Ipv4Addr,
    Ipv6Addr,
    SocketAddr,
    SocketAddrV4,
    SocketAddrV6,
    ByteSize,
    Duration,
    Fraction,
    Reserve,
    Rate,
    Regex,
    DeviceApi,
    DeviceSelector,
    LayerRange,
    ModelSelector,
    NodeSelector,
    TierSelector,
);

impl<T: Zeroize> HoldsSecret for Secret<T> {
    const HOLDS_SECRET: bool = true;
}

impl HoldsSecret for SecretSource {
    const HOLDS_SECRET: bool = true;
}

macro_rules! like_element {
    ($($ty:ident),*) => {
        $(impl<T: HoldsSecret> HoldsSecret for $ty<T> {
            const HOLDS_SECRET: bool = T::HOLDS_SECRET;
        })*
    };
}

like_element!(Option, Box, Vec, VecDeque, BTreeSet);

impl<T: HoldsSecret, S> HoldsSecret for HashSet<T, S> {
    const HOLDS_SECRET: bool = T::HOLDS_SECRET;
}

impl<T: HoldsSecret, const N: usize> HoldsSecret for [T; N] {
    const HOLDS_SECRET: bool = T::HOLDS_SECRET;
}

impl<K: HoldsSecret, V: HoldsSecret> HoldsSecret for BTreeMap<K, V> {
    const HOLDS_SECRET: bool = K::HOLDS_SECRET || V::HOLDS_SECRET;
}

impl<K: HoldsSecret, V: HoldsSecret, S> HoldsSecret for HashMap<K, V, S> {
    const HOLDS_SECRET: bool = K::HOLDS_SECRET || V::HOLDS_SECRET;
}

/// The flags of the common shapes, checked when the crate compiles.
const _: () = {
    assert!(<Secret<String> as HoldsSecret>::HOLDS_SECRET);
    assert!(<Option<SecretSource> as HoldsSecret>::HOLDS_SECRET);
    assert!(<BTreeMap<String, Secret<String>> as HoldsSecret>::HOLDS_SECRET);
    assert!(<Vec<Option<Secret<String>>> as HoldsSecret>::HOLDS_SECRET);
    assert!(!<BTreeMap<String, Vec<ByteSize>> as HoldsSecret>::HOLDS_SECRET);
    assert!(!<Option<Duration> as HoldsSecret>::HOLDS_SECRET);
};
