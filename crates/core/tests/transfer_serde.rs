// SPDX-License-Identifier: MIT OR Apache-2.0

//! Serde properties of the transfer types (RFC-0020 §3, RFC-0021 §2).
//!
//! Every type round-trips through a serde format unchanged, and every record decodes when it
//! carries fields this build does not know. JSON stands in for any self-describing format.

use std::fmt::Debug;

use proptest::prelude::*;
use serde::Serialize;
use serde::de::DeserializeOwned;
use syntropy_core::{Bytes, ExtentRef, FileIx, HostRegion, StepLayer};

fn file_ix() -> impl Strategy<Value = FileIx> {
    any::<u16>().prop_map(FileIx)
}

fn extent_ref() -> impl Strategy<Value = ExtentRef> {
    (file_ix(), any::<u64>(), any::<u64>()).prop_map(|(file, offset, len)| ExtentRef {
        file,
        offset,
        len: Bytes(len),
    })
}

fn step_layer() -> impl Strategy<Value = StepLayer> {
    (any::<u64>(), any::<u16>()).prop_map(|(step, layer)| StepLayer { step, layer })
}

fn host_region() -> impl Strategy<Value = HostRegion> {
    (any::<u64>(), any::<u64>()).prop_map(|(addr, len)| HostRegion {
        addr,
        len: Bytes(len),
    })
}

fn round_trip<T: Serialize + DeserializeOwned + PartialEq + Debug>(value: &T) {
    let json = serde_json::to_string(value).expect("serialize");
    let back: T = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(&back, value, "round trip through {json}");
}

/// Encodes `value`, adds an unknown field to the record, and decodes.
fn decodes_with_future_fields<T: Serialize + DeserializeOwned + PartialEq + Debug>(value: &T) {
    let mut json = serde_json::to_value(value).expect("serialize");
    json.as_object_mut().expect("a record").insert(
        "field_from_a_future_build".to_owned(),
        serde_json::json!({ "nested": [1, 2, 3], "flag": true }),
    );
    let back: T = serde_json::from_value(json.clone())
        .unwrap_or_else(|e| panic!("decode with unknown fields failed: {e}\n{json}"));
    assert_eq!(&back, value);
}

proptest! {
    #[test]
    fn rt_file_ix(value in file_ix()) {
        round_trip(&value);
    }

    #[test]
    fn rt_extent_ref(value in extent_ref()) {
        round_trip(&value);
    }

    #[test]
    fn rt_step_layer(value in step_layer()) {
        round_trip(&value);
    }

    #[test]
    fn rt_host_region(value in host_region()) {
        round_trip(&value);
    }

    #[test]
    fn ff_extent_ref(value in extent_ref()) {
        decodes_with_future_fields(&value);
    }

    #[test]
    fn ff_step_layer(value in step_layer()) {
        decodes_with_future_fields(&value);
    }

    #[test]
    fn ff_host_region(value in host_region()) {
        decodes_with_future_fields(&value);
    }
}

/// `FileIx` is a transparent newtype: it encodes as the bare index.
#[test]
fn file_ix_encodes_as_its_index() {
    assert_eq!(serde_json::to_string(&FileIx(7)).unwrap(), "7");
}
