// SPDX-License-Identifier: MIT OR Apache-2.0

//! PROD-E6-T01's exit-code tests (RFC-0051 §2.2): the binary's exit code for usage errors, no
//! daemon, commands not built yet, the hidden modes, and each error class a daemon returns.

mod common;

use common::{Fake, closed_addr, code, finish, syntropy};

#[test]
fn help_and_version_flags_exit_0_and_a_usage_error_exits_2() {
    assert_eq!(code(&finish(&mut syntropy(&["--help"]))), 0);
    assert_eq!(code(&finish(&mut syntropy(&["--version"]))), 0);
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
    for host in ["https://box:7433", "box:7433"] {
        let output = finish(syntropy(&["version"]).env("SYNTROPY_HOST", host));
        assert_eq!(code(&output), 5, "{host}");
    }
}
