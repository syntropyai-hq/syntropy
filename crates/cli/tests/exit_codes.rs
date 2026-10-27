// SPDX-License-Identifier: MIT OR Apache-2.0

//! PROD-E6-T01's exit-code tests (RFC-0051 §2.2): the binary's exit code for usage errors, no
//! daemon, commands not built yet, the hidden modes, and each error class a daemon returns.

mod common;

use common::{Fake, closed_addr, code, finish, syntropy};

#[test]
fn help_and_version_flags_exit_0_and_a_usage_error_exits_2() {
    assert_eq!(code(&finish(&mut syntropy(&["--help"]))), 0);
    assert_eq!(code(&finish(&mut syntropy(&["--version"]))), 0);
    // Every command takes `--version`, as clap's propagated flag did.
    let sub = finish(&mut syntropy(&["config", "check", "--version"]));
    assert_eq!(code(&sub), 0, "{}", String::from_utf8_lossy(&sub.stderr));
    assert!(String::from_utf8_lossy(&sub.stdout).starts_with("syntropy "));
    assert_eq!(code(&finish(&mut syntropy(&["cluster", "--help"]))), 0);
    let unknown = finish(&mut syntropy(&["frobnicate"]));
    assert_eq!(code(&unknown), 2);
    // No command at all is a usage error too.
    assert_eq!(code(&finish(&mut syntropy(&[]))), 2);
    // A hidden mode's malformed argument is a usage error.
    let bad = finish(&mut syntropy(&["__runner", "--ipc", "nonsense"]));
    assert_eq!(code(&bad), 2, "{}", String::from_utf8_lossy(&bad.stderr));
}

#[test]
fn no_daemon_reachable_exits_3_and_names_what_was_tried() {
    let addr = closed_addr();
    let output = finish(syntropy(&["version"]).env("SYNTROPY_HOST", format!("http://{addr}")));
    assert_eq!(code(&output), 3);
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.starts_with("error[cli/no_daemon]: no Syntropy daemon is reachable"),
        "{stderr}"
    );
    assert!(
        stderr.contains(&format!("tried http://{addr} (SYNTROPY_HOST)")),
        "{stderr}"
    );
    assert!(
        stderr.contains("hint: check that the daemon at SYNTROPY_HOST runs"),
        "{stderr}"
    );
    assert!(output.stdout.is_empty());
}

#[test]
fn a_command_not_built_yet_exits_5_and_says_so() {
    for args in [
        &["pull", "qwen3:8b"][..],
        &["cluster", "status"],
        &["ls", "--json"],
    ] {
        let output = finish(&mut syntropy(args));
        assert_eq!(code(&output), 5, "{args:?}");
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(
            stderr.starts_with("error[cli/not_built]: `syntropy "),
            "{stderr}"
        );
        assert!(
            stderr.contains("is not available in this build yet"),
            "{stderr}"
        );
    }
    // An alias names the command it stands for.
    let output = finish(&mut syntropy(&["models"]));
    assert!(String::from_utf8_lossy(&output.stderr).contains("`syntropy ls`"));
}

#[test]
fn the_hidden_modes_exit_1_until_they_are_built() {
    let runner = finish(&mut syntropy(&[
        "__runner",
        "--ipc",
        "socket:/nonexistent/cmd.sock",
    ]));
    assert_eq!(code(&runner), 1);
    assert!(String::from_utf8_lossy(&runner.stderr).contains("runner mode is not available"));
    let selftest = finish(&mut syntropy(&["__selftest", "--root", "/nonexistent"]));
    assert_eq!(code(&selftest), 1);
    assert!(String::from_utf8_lossy(&selftest.stderr).contains("self-test is not available"));
    // Hidden: neither shows in the help.
    let help = finish(&mut syntropy(&["--help"]));
    let help = String::from_utf8(help.stdout).unwrap();
    assert!(
        !help.contains("__runner") && !help.contains("__selftest"),
        "{help}"
    );
}

/// Each error class the daemon can return, as its native error body, and the exit code RFC-0051
/// §2.2 gives it.
#[test]
fn each_error_class_from_the_daemon_exits_with_its_code() {
    let cases = [
        (400, "InvalidRequest", "invalid_request", 5),
        (404, "InvalidRequest", "model_not_found", 4),
        (400, "Unsupported", "unsupported_feature", 5),
        (400, "Capacity", "model_does_not_fit", 6),
        (503, "Transient", "busy", 1),
        (502, "Link", "link_down", 1),
        (500, "Device", "device_lost", 1),
        (500, "Integrity", "chunk_hash_mismatch", 8),
        (500, "Correctness", "canary_failed", 1),
        (500, "Drift", "drift", 1),
        (504, "Timeout", "timed_out", 9),
        (401, "Auth", "local_auth_required", 7),
        (499, "Cancelled", "cancelled", 130),
        (500, "Internal", "bug", 1),
    ];
    for (status, class, error_code, exit) in cases {
        let body = format!(
            r#"{{"error":{{"class":"{class}","code":"{error_code}","message":"the daemon says no"}}}}"#
        );
        // The client probes the address, then sends the request: two connections.
        let fake = Fake::start(status, body, 2);
        let output =
            finish(syntropy(&["version"]).env("SYNTROPY_HOST", format!("http://{}", fake.addr)));
        let stderr = String::from_utf8(output.stderr.clone()).unwrap();
        assert_eq!(code(&output), exit, "{class} {error_code}: {stderr}");
        assert_eq!(
            stderr.trim_end(),
            format!("error[api/{error_code}]: the daemon says no"),
            "{class}"
        );
        assert!(output.stdout.is_empty());
    }
}

#[test]
fn a_remote_daemon_gets_the_api_key_as_a_bearer_token() {
    let fake = Fake::start(
        200,
        serde_json::to_string(&common::test_version()).unwrap(),
        2,
    );
    let output = finish(
        syntropy(&["version", "--json"])
            .env("SYNTROPY_HOST", format!("http://{}", fake.addr))
            .env("SYNTROPY_API_KEY", "sk-test-key"),
    );
    assert_eq!(
        code(&output),
        0,
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let head = fake.heads.recv().unwrap();
    assert!(
        head.starts_with("GET /v1/syntropy/version HTTP/1.1\r\n"),
        "{head}"
    );
    assert!(
        head.to_ascii_lowercase()
            .contains("authorization: bearer sk-test-key\r\n"),
        "{head}"
    );
}

#[test]
fn an_answer_that_is_not_the_api_type_exits_1() {
    let fake = Fake::start(200, r#"{"not":"a version"}"#.to_owned(), 2);
    let output =
        finish(syntropy(&["version"]).env("SYNTROPY_HOST", format!("http://{}", fake.addr)));
    assert_eq!(code(&output), 1);
    assert!(
        String::from_utf8_lossy(&output.stderr).starts_with("error[cli/bad_response]"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn syntropy_host_with_tls_or_a_bad_url_exits_5() {
    for host in ["https://box:7433", "box:7433", "http://10.0.0.2:7433"] {
        let output = finish(syntropy(&["version"]).env("SYNTROPY_HOST", host));
        assert_eq!(code(&output), 5, "{host}");
    }
}

/// A hostile daemon's text never reaches the terminal raw: serde's error for a bad answer quotes
/// the daemon's strings, and every field of the rendered error is escaped (B2 of the review of
/// #1919).
#[test]
fn a_hostile_daemon_s_escape_sequences_never_reach_the_terminal() {
    // A valid build, and one whose version is not a version: serde's error quotes it.
    let build = serde_json::to_string(&common::test_version().build).unwrap();
    let mut bad = serde_json::to_value(&common::test_version().build).unwrap();
    bad["version"] = "\u{1b}[31mEVIL\u{202e}".into();
    let bad = serde_json::to_string(&bad).unwrap();
    let bodies = [
        // An unknown variant: an OSC title set, a BEL and a screen clear.
        format!(
            r#"{{"build":{build},"protocols":{{"wire":1,"ipc":1}},"support_tier":"\u001b]0;PWNED\u0007\u001b[2J"}}"#
        ),
        // A malformed version inside `build`, with a colour and a right-to-left override.
        format!(r#"{{"build":{bad},"protocols":{{"wire":1,"ipc":1}},"support_tier":"tier1"}}"#),
    ];
    for body in bodies {
        let fake = Fake::start(200, body, 2);
        let output =
            finish(syntropy(&["version"]).env("SYNTROPY_HOST", format!("http://{}", fake.addr)));
        assert_eq!(code(&output), 1);
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(stderr.starts_with("error[cli/bad_response]"), "{stderr}");
        assert!(
            !stderr.contains('\u{1b}') && !stderr.contains('\u{7}') && !stderr.contains('\u{202e}'),
            "raw control characters on stderr: {stderr:?}"
        );
        assert!(stderr.contains("\\u{1b}"), "{stderr}");
    }
    // A version that parses, with escapes in its pre-release part, is printed escaped.
    let mut odd = serde_json::to_value(common::test_version()).unwrap();
    odd["build"]["version"] = "0.0.0-x\u{1b}]0;PWNED\u{7}".into();
    let fake = Fake::start(200, odd.to_string(), 2);
    let output =
        finish(syntropy(&["version"]).env("SYNTROPY_HOST", format!("http://{}", fake.addr)));
    assert_eq!(
        code(&output),
        0,
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        !stdout.contains('\u{1b}') && !stdout.contains('\u{7}'),
        "{stdout:?}"
    );
    assert!(stdout.contains("0.0.0-x\\u{1b}]0;PWNED\\u{7}"), "{stdout}");
    // An API error body whose code and message carry the same.
    let body =
        r#"{"error":{"class":"Internal","code":"x\u001b[2J","message":"m\u001b]0;t\u0007"}}"#;
    let fake = Fake::start(500, body.to_owned(), 2);
    let output =
        finish(syntropy(&["version"]).env("SYNTROPY_HOST", format!("http://{}", fake.addr)));
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert_eq!(
        stderr.trim_end(),
        "error[api/x\\u{1b}[2J]: m\\u{1b}]0;t\\u{7}"
    );
}

/// Plain HTTP to a host beyond this machine is refused before any connection, so the API key
/// never crosses the network in the clear (B3 of the review of #1919).
#[test]
fn plain_http_beyond_loopback_is_refused_and_the_key_is_never_sent() {
    for host in [
        "http://192.0.2.1:7433",
        "http://box.example:7433",
        "http://[2001:db8::1]",
    ] {
        let output = finish(
            syntropy(&["version"])
                .env("SYNTROPY_HOST", host)
                .env("SYNTROPY_API_KEY", "sk-never-sent"),
        );
        assert_eq!(code(&output), 5, "{host}");
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(stderr.contains("in the clear"), "{stderr}");
        assert!(stderr.contains("ssh -L 7433:127.0.0.1:7433"), "{stderr}");
        assert!(!stderr.contains("sk-never-sent"), "{stderr}");
    }
}

/// An answer over the client's size limit is refused, not read into memory (N1 of the review of
/// #1919).
#[test]
fn an_answer_over_the_size_limit_exits_1() {
    let body = format!(
        "\"{}\"",
        "a".repeat(syntropy_cli::client::MAX_RESPONSE + 1024)
    );
    let fake = Fake::start(200, body, 2);
    let output =
        finish(syntropy(&["version"]).env("SYNTROPY_HOST", format!("http://{}", fake.addr)));
    assert_eq!(code(&output), 1);
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("over the client's 4.00 MiB limit"),
        "{stderr}"
    );
}
