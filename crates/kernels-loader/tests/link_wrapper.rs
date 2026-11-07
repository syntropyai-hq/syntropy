// SPDX-License-Identifier: MIT OR Apache-2.0

//! `ci/link/provider-cc`, the Linux linker driver `.cargo/config.toml` sets (COMP-E9-T19;
//! RFC-0040 §10.4): run with a stand-in driver that records its arguments, it swaps rustc's
//! version script for `provider.map` only when that script exports exactly the entry symbol, and
//! passes every argument through otherwise, spaces included.

#![cfg(target_os = "linux")]

mod common;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn wrapper() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../ci/link/provider-cc")
}

/// A stand-in driver that writes each argument it gets on a line of `args`.
fn recording_driver(dir: &Path) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let driver = dir.join("cc");
    let args = dir.join("args");
    fs::write(
        &driver,
        format!(
            "#!/bin/sh\nfor a do printf '%s\\n' \"$a\"; done > '{}'\n",
            args.display()
        ),
    )
    .unwrap();
    fs::set_permissions(&driver, fs::Permissions::from_mode(0o755)).unwrap();
    driver
}

/// Runs the wrapper over `args` with the stand-in driver and returns what the driver got.
fn run(dir: &Path, args: &[&str]) -> Vec<String> {
    let status = Command::new(wrapper())
        .args(args)
        .env("SYNTROPY_LINK_CC", recording_driver(dir))
        .status()
        .expect("run the wrapper");
    assert!(status.success());
    fs::read_to_string(dir.join("args"))
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect()
}

/// A version script in rustc's form, exporting `globals`.
fn rustc_script(dir: &Path, name: &str, globals: &[&str]) -> String {
    let mut text = String::from("{\n  global:\n");
    for global in globals {
        text.push_str(&format!("    {global};\n"));
    }
    text.push_str("\n  local:\n    *;\n};\n");
    let path = dir.join(name);
    fs::write(&path, text).unwrap();
    format!("-Wl,--version-script={}", path.display())
}

#[test]
fn a_provider_librarys_script_is_swapped_and_other_arguments_pass_through() {
    let dir = common::TempDir::new("link-swap");
    let script = rustc_script(&dir.0, "provider", &["syntropy_provider_entry_v1"]);
    let got = run(&dir.0, &["-o", "lib a.so", &script, "-Wl,-Bsymbolic", ""]);
    let map = wrapper().with_file_name("provider.map");
    assert_eq!(
        got,
        [
            "-o".to_owned(),
            "lib a.so".to_owned(),
            format!("-Wl,--version-script={}", map.display()),
            "-Wl,-Bsymbolic".to_owned(),
            String::new(),
        ]
    );
}

#[test]
fn any_other_script_is_passed_unchanged() {
    let dir = common::TempDir::new("link-keep");
    for (name, globals) in [
        ("two", &["syntropy_provider_entry_v1", "other"][..]),
        ("prefixed", &["xsyntropy_provider_entry_v1"][..]),
        ("none", &[][..]),
        ("plain", &["some_c_api"][..]),
    ] {
        let script = rustc_script(&dir.0, name, globals);
        assert_eq!(
            run(&dir.0, &[&script]),
            std::slice::from_ref(&script),
            "{name}"
        );
    }
    let missing = "-Wl,--version-script=/nonexistent/list";
    assert_eq!(run(&dir.0, &[missing]), [missing]);
}

#[test]
fn a_missing_driver_is_named() {
    let output = Command::new(wrapper())
        .arg("-v")
        .env("SYNTROPY_LINK_CC", "/nonexistent/cc")
        .output()
        .expect("run the wrapper");
    assert_eq!(output.status.code(), Some(127));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("/nonexistent/cc"), "{stderr}");
}
