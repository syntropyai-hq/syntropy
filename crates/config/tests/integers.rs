// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integer keys at their type's bounds (#1758; RFC-0054 §8 phase 3).
//!
//! A value outside a key's range is `out_of_range` (code 4), and an integer that does not fit
//! the key's integer type is outside its range. A value of another kind stays `wrong_type`
//! (code 2). TOML integers are 64-bit and signed, so `u64::MAX` and the values past `i64`'s
//! bounds cannot be written; the `u64` and `i64` rows stop at what TOML holds.

// The section exists for its schema; its fields are never read.
#![allow(dead_code)]

use serde::Deserialize;
use syntropy_config::{
    Code, ConfigFile, ConfigSection, Env, Flags, Layers, Os, Registry, ResolvedConfig,
};

/// A key of each integer width, and an optional and a list one.
#[derive(Debug, Deserialize, ConfigSection)]
#[config(section = "n", scope = Node, owner = "RFC-0054")]
struct N {
    /// A u8.
    #[config(reload = Hot, default = 0)]
    a_u8: u8,
    /// A u16.
    #[config(reload = Hot, default = 0)]
    a_u16: u16,
    /// A u32.
    #[config(reload = Hot, default = 0)]
    a_u32: u32,
    /// A u64.
    #[config(reload = Hot, default = 0)]
    a_u64: u64,
    /// A usize.
    #[config(reload = Hot, default = 0)]
    a_usize: usize,
    /// An i8.
    #[config(reload = Hot, default = 0)]
    an_i8: i8,
    /// An i16.
    #[config(reload = Hot, default = 0)]
    an_i16: i16,
    /// An i32.
    #[config(reload = Hot, default = 0)]
    an_i32: i32,
    /// An i64.
    #[config(reload = Hot, default = 0)]
    an_i64: i64,
    /// An optional u32.
    #[config(reload = Hot, default = unset)]
    #[serde(default)]
    maybe: Option<u32>,
    /// Some u16s.
    #[config(reload = Hot, default = [])]
    many: Vec<u16>,
}

static REGISTRY: Registry = Registry::new(&[&N::SCHEMA]);

/// The code `key = toml` is rejected with, or `None` when it loads.
fn code(key: &str, toml: &str) -> Option<Code> {
    let files = [ConfigFile::new(
        "/etc/syntropy/config.toml",
        format!("[n]\n{key} = {toml}\n"),
    )];
    let env = Env::from_pairs::<&str, &str>(Os::Linux, []);
    let resolved = ResolvedConfig::resolve(
        &REGISTRY,
        Layers {
            os: Os::Linux,
            files: &files,
            env: &env,
            flags: &Flags::new(),
        },
    );
    match resolved {
        Ok(config) => {
            config
                .section::<N>()
                .unwrap_or_else(|e| panic!("{key} = {toml} loaded but does not build: {e}"));
            None
        }
        Err(rejected) => rejected.errors[0].code,
    }
}

/// Each width at its bounds and one past each, as (key, value, the code it is rejected with).
const BOUNDS: &[(&str, &str, Option<Code>)] = &[
    ("a_u8", "-1", Some(Code::OutOfRange)),
    ("a_u8", "0", None),
    ("a_u8", "255", None),
    ("a_u8", "256", Some(Code::OutOfRange)),
    ("a_u16", "-1", Some(Code::OutOfRange)),
    ("a_u16", "0", None),
    ("a_u16", "65535", None),
    ("a_u16", "65536", Some(Code::OutOfRange)),
    ("a_u32", "-1", Some(Code::OutOfRange)),
    ("a_u32", "0", None),
    ("a_u32", "4294967295", None),
    ("a_u32", "4294967296", Some(Code::OutOfRange)),
    ("a_u64", "-1", Some(Code::OutOfRange)),
    ("a_u64", "0", None),
    ("a_u64", "9223372036854775807", None),
    ("a_usize", "-1", Some(Code::OutOfRange)),
    ("a_usize", "0", None),
    ("a_usize", "9223372036854775807", None),
    ("an_i8", "-129", Some(Code::OutOfRange)),
    ("an_i8", "-128", None),
    ("an_i8", "127", None),
    ("an_i8", "128", Some(Code::OutOfRange)),
    ("an_i16", "-32769", Some(Code::OutOfRange)),
    ("an_i16", "-32768", None),
    ("an_i16", "32767", None),
    ("an_i16", "32768", Some(Code::OutOfRange)),
    ("an_i32", "-2147483649", Some(Code::OutOfRange)),
    ("an_i32", "-2147483648", None),
    ("an_i32", "2147483647", None),
    ("an_i32", "2147483648", Some(Code::OutOfRange)),
    ("an_i64", "-9223372036854775808", None),
    ("an_i64", "9223372036854775807", None),
    ("maybe", "-1", Some(Code::OutOfRange)),
    ("maybe", "4294967296", Some(Code::OutOfRange)),
    ("maybe", "7", None),
    ("many", "[1, 65536]", Some(Code::OutOfRange)),
    ("many", "[-1]", Some(Code::OutOfRange)),
    ("many", "[0, 65535]", None),
];

/// Values of another kind, which stay `wrong_type`.
const OTHER_KINDS: &[(&str, &str, Option<Code>)] = &[
    ("a_u32", "\"7\"", Some(Code::WrongType)),
    ("a_u32", "1.5", Some(Code::WrongType)),
    ("a_u32", "true", Some(Code::WrongType)),
    ("a_u8", "[1]", Some(Code::WrongType)),
    ("an_i64", "1979-05-27", Some(Code::WrongType)),
    ("maybe", "\"x\"", Some(Code::WrongType)),
    ("many", "[\"a\"]", Some(Code::WrongType)),
];

#[test]
fn an_integer_outside_its_type_is_out_of_range() {
    for &(key, toml, expected) in BOUNDS {
        assert_eq!(code(key, toml), expected, "{key} = {toml}");
    }
}

#[test]
fn a_value_of_another_kind_is_still_the_wrong_type() {
    for &(key, toml, expected) in OTHER_KINDS {
        assert_eq!(code(key, toml), expected, "{key} = {toml}");
    }
}
