// SPDX-License-Identifier: MIT OR Apache-2.0

//! Golden tests of `set` and `unset` on a file with comments and odd spacing: each edit's output
//! in full, and a line diff showing that only the target lines changed (PROD-E1-T12). Then
//! `write_atomic`'s rename, permissions and symbolic-link refusal.

use super::*;

/// A file as a person writes it: comments above, beside and below keys, indentation, spacing
/// around `=` and none, a dotted key at the top, an inline table, blank lines.
const FILE: &str = "\
# My node.   Edited by hand.
config_version = 1
swim.period = \"1s\"   # gossip

[api]   # the API
  # where to listen
  bind   =   [\"127.0.0.1:7499\"]   # loopback only
  tls = \"auto\"

# discovery, tuned
[discovery]
port=7999
interfaces = { include = [\"eth*\"],   exclude = [\"docker*\"] }

# end of file
";

/// The lines `old` has and `new` lacks, and the lines `new` has and `old` lacks, by a
/// longest-common-subsequence diff.
fn diff<'a>(old: &'a str, new: &'a str) -> (Vec<&'a str>, Vec<&'a str>) {
    let a: Vec<&str> = old.lines().collect();
    let b: Vec<&str> = new.lines().collect();
    let mut lcs = vec![vec![0usize; b.len() + 1]; a.len() + 1];
    for i in (0..a.len()).rev() {
        for j in (0..b.len()).rev() {
            lcs[i][j] = if a[i] == b[j] {
                lcs[i + 1][j + 1] + 1
            } else {
                lcs[i + 1][j].max(lcs[i][j + 1])
            };
        }
    }
    let (mut i, mut j) = (0, 0);
    let (mut removed, mut added) = (Vec::new(), Vec::new());
    while i < a.len() || j < b.len() {
        if i < a.len() && j < b.len() && a[i] == b[j] {
            i += 1;
            j += 1;
        } else if j < b.len() && (i == a.len() || lcs[i][j + 1] >= lcs[i + 1][j]) {
            added.push(b[j]);
            j += 1;
        } else {
            removed.push(a[i]);
            i += 1;
        }
    }
    (removed, added)
}

/// `edited` is `FILE` with exactly `removed` lines gone and `added` lines new.
fn only_lines(edited: &str, removed: &[&str], added: &[&str]) {
    let (r, a) = diff(FILE, edited);
    assert_eq!((r.as_slice(), a.as_slice()), (removed, added), "\n{edited}");
}

#[test]
fn replacing_a_value_keeps_its_spacing_and_comment() {
    let edited = set(FILE, "api.bind", r#"["0.0.0.0:7499", "[::]:7499"]"#).unwrap();
    only_lines(
        &edited,
        &[r#"  bind   =   ["127.0.0.1:7499"]   # loopback only"#],
        &[r#"  bind   =   ["0.0.0.0:7499", "[::]:7499"]   # loopback only"#],
    );
    let edited = set(FILE, "discovery.port", "8000").unwrap();
    only_lines(&edited, &["port=7999"], &["port=8000"]);
    // A dotted key at the top of the file.
    let edited = set(FILE, "swim.period", "2s").unwrap();
    only_lines(
        &edited,
        &[r#"swim.period = "1s"   # gossip"#],
        &[r#"swim.period = "2s"   # gossip"#],
    );
}

#[test]
fn a_bare_word_is_a_string_and_toml_is_toml() {
    let edited = set(FILE, "api.tls", "off").unwrap();
    only_lines(&edited, &[r#"  tls = "auto""#], &[r#"  tls = "off""#]);
    let edited = set(FILE, "api.tls", r#""auto""#).unwrap();
    assert_eq!(edited, FILE);
    // A number meant as a string is quoted on the command line.
    let edited = set(FILE, "api.tls", r#""8080""#).unwrap();
    only_lines(&edited, &[r#"  tls = "auto""#], &[r#"  tls = "8080""#]);
    // Text that would smuggle a second key in is one string.
    let edited = set(FILE, "api.tls", "1\nevil = 2").unwrap();
    assert!(edited.contains(r#"  tls = "1\nevil = 2""#), "{edited}");
}

#[test]
fn a_new_key_goes_after_its_table_s_last_key_indented_like_it() {
    let edited = set(FILE, "api.local_socket", "false").unwrap();
    only_lines(&edited, &[], &["  local_socket = false"]);
    let at = edited.find("local_socket").unwrap();
    assert!(at > edited.find("tls = ").unwrap() && at < edited.find("# discovery").unwrap());
    // A key under a table the file has, through a table it does not: a dotted key in it.
    let edited = set(FILE, "api.tls_files.cert", "/etc/cert.pem").unwrap();
    only_lines(&edited, &[], &[r#"  tls_files.cert = "/etc/cert.pem""#]);
    // A dotted table at the top gains a dotted key.
    let edited = set(FILE, "swim.suspect", "5").unwrap();
    only_lines(&edited, &[], &["swim.suspect = 5"]);
}

#[test]
fn a_key_under_no_table_of_the_file_gets_a_new_table_at_the_end() {
    let edited = set(FILE, "link.failover_timeout", "10s").unwrap();
    only_lines(&edited, &[], &["[link]", r#"failover_timeout = "10s""#, ""]);
    assert!(edited.ends_with("\n[link]\nfailover_timeout = \"10s\"\n\n# end of file\n"));
    let edited = set(FILE, "cluster.store.chunk", "4").unwrap();
    only_lines(&edited, &[], &["[cluster.store]", "chunk = 4", ""]);
    // A top-level key goes with the other top-level keys.
    let edited = set(FILE, "fidelity", "exact").unwrap();
    only_lines(&edited, &[], &[r#"fidelity = "exact""#]);
    assert!(edited.find("fidelity").unwrap() < edited.find("[api]").unwrap());
}

#[test]
fn an_inline_table_is_edited_on_its_own_line() {
    let edited = set(FILE, "discovery.interfaces.exclude", r#"["veth*"]"#).unwrap();
    only_lines(
        &edited,
        &[r#"interfaces = { include = ["eth*"],   exclude = ["docker*"] }"#],
        &[r#"interfaces = { include = ["eth*"],   exclude = ["veth*"] }"#],
    );
    let edited = set(FILE, "discovery.interfaces.v6", "true").unwrap();
    only_lines(
        &edited,
        &[r#"interfaces = { include = ["eth*"],   exclude = ["docker*"] }"#],
        &[r#"interfaces = { include = ["eth*"],   exclude = ["docker*"], v6 = true }"#],
    );
    let edited = unset(FILE, "discovery.interfaces.exclude")
        .unwrap()
        .unwrap();
    only_lines(
        &edited,
        &[r#"interfaces = { include = ["eth*"],   exclude = ["docker*"] }"#],
        &[r#"interfaces = { include = ["eth*"] }"#],
    );
}

#[test]
fn unset_removes_the_key_s_line_and_keeps_the_comments_above_it() {
    // The comment above `bind` stays, now above `tls`.
    let edited = unset(FILE, "api.bind").unwrap().unwrap();
    only_lines(
        &edited,
        &[r#"  bind   =   ["127.0.0.1:7499"]   # loopback only"#],
        &[],
    );
    assert!(
        edited.contains("  # where to listen\n  tls = \"auto\""),
        "{edited}"
    );
    // The last key of a table: nothing to carry.
    let edited = unset(FILE, "api.tls").unwrap().unwrap();
    only_lines(&edited, &[r#"  tls = "auto""#], &[]);
    let edited = unset(FILE, "swim.period").unwrap().unwrap();
    only_lines(&edited, &[r#"swim.period = "1s"   # gossip"#], &[]);
    // A key the file does not set.
    assert_eq!(unset(FILE, "api.local_socket").unwrap(), None);
    assert_eq!(unset(FILE, "nothing.here").unwrap(), None);
}

#[test]
fn a_last_key_s_comment_moves_to_the_next_header_or_the_end() {
    let file = "[a]\nx = 1\n# about y\ny = 2\n\n[b]\nz = 3\n# about w\nw = 4\n";
    let edited = unset(file, "a.y").unwrap().unwrap();
    assert_eq!(
        edited,
        "[a]\nx = 1\n# about y\n\n[b]\nz = 3\n# about w\nw = 4\n"
    );
    let edited = unset(file, "b.w").unwrap().unwrap();
    assert_eq!(
        edited,
        "[a]\nx = 1\n# about y\ny = 2\n\n[b]\nz = 3\n# about w\n"
    );
}

#[test]
fn edits_that_cannot_be_made_say_why_and_never_quote_a_value() {
    for (path, value, want) in [
        ("api", "1", "is a table"),
        ("api.bind.x", "1", "holds a value"),
        ("api..bind", "1", "is not a key"),
        ("api.b d", "1", "is not a key"),
    ] {
        let error = set(FILE, path, value).unwrap_err();
        assert!(error.to_string().contains(want), "{path}: {error}");
    }
    let error = unset(FILE, "api").unwrap_err();
    assert!(error.to_string().contains("is a table"), "{error}");
    // A broken file is not quoted: the line could hold a secret.
    let broken = "api_key = sk-canary-77\n";
    for error in [
        set(broken, "x", "1").unwrap_err(),
        unset(broken, "x").unwrap_err(),
    ] {
        assert!(!error.to_string().contains("canary"), "{error}");
        assert!(error.to_string().contains("not valid TOML"), "{error}");
    }
}

#[test]
fn quoted_segments_address_selector_tables() {
    assert_eq!(
        key_segments(r#"policy."qwen3*".ram_headroom"#).unwrap(),
        ["policy", "qwen3*", "ram_headroom"]
    );
    let file = "[policy.\"qwen3*\"]\nram_headroom = \"2GiB\"\n";
    let edited = set(file, r#"policy."qwen3*".ram_headroom"#, "4GiB").unwrap();
    assert_eq!(edited, "[policy.\"qwen3*\"]\nram_headroom = \"4GiB\"\n");
    assert!(key_segments(r#"policy."a\"b""#).is_err());
    assert!(key_segments("a.").is_err());
}

#[test]
fn an_empty_file_gets_its_first_key() {
    assert_eq!(set("", "api.tls", "off").unwrap(), "[api]\ntls = \"off\"\n");
    assert_eq!(
        set("", "fidelity", "exact").unwrap(),
        "fidelity = \"exact\"\n"
    );
}

/// `FILE` with Windows line endings.
fn crlf() -> String {
    FILE.replace('\n', "\r\n")
}

/// A CRLF file stays CRLF: each edit changes only its key's lines, each line still ending in
/// `\r\n`, and every other byte stays (review of #1927, B1).
#[test]
fn a_crlf_file_keeps_its_line_endings() {
    let file = crlf();
    let crlf_only = |text: &str| {
        assert_eq!(
            text.matches('\n').count(),
            text.matches("\r\n").count(),
            "{text:?}"
        );
    };
    // Replace.
    let edited = set(&file, "discovery.port", "8000").unwrap();
    assert_eq!(edited, file.replace("port=7999\r\n", "port=8000\r\n"));
    // Add, indented like its neighbour.
    let edited = set(&file, "api.local_socket", "false").unwrap();
    assert_eq!(
        edited,
        file.replace(
            "  tls = \"auto\"\r\n",
            "  tls = \"auto\"\r\n  local_socket = false\r\n"
        )
    );
    // A new table at the end.
    let edited = set(&file, "link.failover_timeout", "10s").unwrap();
    crlf_only(&edited);
    assert!(
        edited.contains("\r\n[link]\r\nfailover_timeout = \"10s\"\r\n"),
        "{edited:?}"
    );
    // Unset, the comment above kept.
    let edited = unset(&file, "api.bind").unwrap().unwrap();
    assert_eq!(
        edited,
        file.replace(
            "  bind   =   [\"127.0.0.1:7499\"]   # loopback only\r\n",
            ""
        )
    );
    // An inline table.
    let edited = set(&file, "discovery.interfaces.v6", "true").unwrap();
    crlf_only(&edited);
    assert_eq!(
        edited,
        file.replace(
            r#"exclude = ["docker*"] }"#,
            r#"exclude = ["docker*"], v6 = true }"#
        )
    );
    // An LF file stays LF.
    let edited = set(FILE, "discovery.port", "8000").unwrap();
    assert!(!edited.contains('\r'));
}

#[test]
fn a_file_that_mixes_line_endings_is_refused() {
    let mixed = "[api]\r\ntls = \"auto\"\nbind = []\r\n";
    for error in [
        set(mixed, "api.tls", "none").unwrap_err(),
        unset(mixed, "api.tls").unwrap_err(),
    ] {
        assert!(error.to_string().contains("mixes"), "{error}");
    }
}

/// The line check refuses an edit that touches any line but the key's, a changed line ending
/// included, even when the parsed tables agree.
#[test]
fn the_line_check_refuses_a_change_outside_the_key() {
    let segments = key_segments("discovery.port").unwrap();
    let expected = |text: &str| -> toml::Table { text.parse().unwrap() };
    let good = FILE.replace("port=7999", "port=8000");
    assert_eq!(
        confirm(FILE, &good, &segments, &expected(&good), "discovery.port"),
        Ok(())
    );
    // Another line's comment changed as well.
    let drift = good.replace("# discovery, tuned", "# discovery");
    assert!(confirm(FILE, &drift, &segments, &expected(&good), "discovery.port").is_err());
    // Another line's ending changed as well.
    let ending = good.replacen("config_version = 1\n", "config_version = 1\r\n", 1);
    assert!(confirm(FILE, &ending, &segments, &expected(&good), "discovery.port").is_err());
    // A new key may only add lines.
    let segments = key_segments("api.local_socket").unwrap();
    let added = FILE.replace(
        "  tls = \"auto\"\n",
        "  tls = \"auto\"\n  local_socket = false\n",
    );
    assert_eq!(
        confirm(
            FILE,
            &added,
            &segments,
            &expected(&added),
            "api.local_socket"
        ),
        Ok(())
    );
    let rewritten = added.replace("  tls = \"auto\"", "  tls = \"auto\" ");
    assert!(
        confirm(
            FILE,
            &rewritten,
            &segments,
            &expected(&added),
            "api.local_socket"
        )
        .is_err()
    );
}

// ---- write_atomic -------------------------------------------------------------------------

/// A fresh directory for one test.
fn dir(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("syntropy-write-{test}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// The directory's entries other than `keep`: a temporary file left behind would show here.
fn others(dir: &Path, keep: &str) -> Vec<String> {
    fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|name| name != keep)
        .collect()
}

#[test]
fn a_write_replaces_the_file_and_leaves_nothing_behind() {
    let dir = dir("replace");
    let path = dir.join("config.toml");
    fs::write(&path, "old = 1\n").unwrap();
    write_atomic(&path, "new = 2\n").unwrap();
    assert_eq!(fs::read_to_string(&path).unwrap(), "new = 2\n");
    assert_eq!(others(&dir, "config.toml"), Vec::<String>::new());
    // A missing directory is created.
    let nested = dir.join("a/b/config.toml");
    write_atomic(&nested, "x = 1\n").unwrap();
    assert_eq!(fs::read_to_string(&nested).unwrap(), "x = 1\n");
    fs::remove_dir_all(&dir).unwrap();
}

#[cfg(unix)]
#[test]
fn a_write_keeps_the_file_s_mode_and_a_new_file_is_private() {
    use std::os::unix::fs::PermissionsExt as _;
    let dir = dir("mode");
    let path = dir.join("config.toml");
    fs::write(&path, "old = 1\n").unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o640)).unwrap();
    write_atomic(&path, "new = 2\n").unwrap();
    let mode = fs::metadata(&path).unwrap().permissions().mode() & 0o7777;
    assert_eq!(mode, 0o640);
    let fresh = dir.join("fresh.toml");
    write_atomic(&fresh, "x = 1\n").unwrap();
    let mode = fs::metadata(&fresh).unwrap().permissions().mode() & 0o7777;
    assert_eq!(mode, 0o600);
    fs::remove_dir_all(&dir).unwrap();
}

#[cfg(unix)]
#[test]
fn a_symbolic_link_at_the_target_is_refused_and_its_file_untouched() {
    let dir = dir("symlink");
    let real = dir.join("elsewhere.toml");
    fs::write(&real, "kept = true\n").unwrap();
    let link = dir.join("config.toml");
    std::os::unix::fs::symlink(&real, &link).unwrap();
    let error = write_atomic(&link, "new = 2\n").unwrap_err();
    assert!(error.to_string().contains("symbolic link"), "{error}");
    assert_eq!(fs::read_to_string(&real).unwrap(), "kept = true\n");
    assert!(
        fs::symlink_metadata(&link)
            .unwrap()
            .file_type()
            .is_symlink()
    );
    assert_eq!(others(&dir, "config.toml"), ["elsewhere.toml"]);
    fs::remove_dir_all(&dir).unwrap();
}

/// A read-only file is refused before anything is written, its text unchanged and no temporary
/// file left, on every OS (review of #1927, N2). Run on Windows by its CI.
#[test]
fn a_read_only_target_is_refused_and_nothing_is_left() {
    let dir = dir("readonly");
    let path = dir.join("config.toml");
    fs::write(&path, "kept = true\n").unwrap();
    let mut permissions = fs::metadata(&path).unwrap().permissions();
    permissions.set_readonly(true);
    fs::set_permissions(&path, permissions).unwrap();
    let error = write_atomic(&path, "new = 2\n").unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
    assert!(error.to_string().contains("read-only"), "{error}");
    assert_eq!(fs::read_to_string(&path).unwrap(), "kept = true\n");
    assert_eq!(others(&dir, "config.toml"), Vec::<String>::new());
    assert!(replaceable(&path).is_err());
    // Writable again, so the directory can go.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
    }
    #[cfg(windows)]
    {
        let mut permissions = fs::metadata(&path).unwrap().permissions();
        #[allow(clippy::permissions_set_readonly_false, reason = "Windows only")]
        permissions.set_readonly(false);
        fs::set_permissions(&path, permissions).unwrap();
    }
    fs::remove_dir_all(&dir).unwrap();
}

/// A temporary file that could not be renamed is removed even when it is read-only, which on
/// Windows needs the attribute cleared first.
#[test]
fn a_read_only_temporary_file_is_still_discarded() {
    let dir = dir("discard");
    let temp = dir.join(".config.toml.syntropy-write-test");
    fs::write(&temp, "secret = 1\n").unwrap();
    let mut permissions = fs::metadata(&temp).unwrap().permissions();
    permissions.set_readonly(true);
    fs::set_permissions(&temp, permissions).unwrap();
    discard(&temp);
    assert!(!temp.exists());
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_directory_at_the_target_is_refused() {
    let dir = dir("isdir");
    let error = write_atomic(&dir, "x = 1\n").unwrap_err();
    assert!(error.to_string().contains("not a regular file"), "{error}");
    fs::remove_dir_all(&dir).unwrap();
}
