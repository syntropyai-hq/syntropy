// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Hub range source against a local HTTP server (RFC-0041, Testing and validation), and,
//! with `SYNTROPY_HUB_LIVE=1`, against a public Hub file.

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;

use syntropy_core::ErrorClass;
use syntropy_manifest::range::replay::{Recording, Replay};
use syntropy_manifest::{HeaderReader, RangeError, RangeSource};

use super::*;

/// One request the server received: its path and its headers, lower-cased names.
#[derive(Clone, Debug)]
struct Seen {
    path: String,
    headers: Vec<(String, String)>,
}

impl Seen {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_str())
    }
}

/// A local HTTP/1.1 server. Every path serves `body` with range support, except:
/// - `/redirect` and `/redirect-twice` redirect to `/file` on the same host, once or twice;
/// - `/cross` redirects to `/file` on `localhost`, another host than `127.0.0.1`;
/// - `/loop` redirects to itself;
/// - `/short` and `/long` send one byte fewer or more than the range;
/// - `/whole` ignores `Range` and sends the whole file with a 200;
/// - `/shifted` answers with the bytes one past those asked for, and says so;
/// - `/gated` is a 401, `/forbidden` a 403, `/missing` a 404 and `/busy` a 503;
/// - `/empty` is an empty file.
struct Server {
    port: u16,
    seen: Arc<Mutex<Vec<Seen>>>,
}

impl Server {
    fn start(body: Vec<u8>) -> Server {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let log = seen.clone();
        let body = Arc::new(body);
        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { continue };
                let (log, body) = (log.clone(), body.clone());
                thread::spawn(move || serve(stream, &body, port, &log));
            }
        });
        Server { port, seen }
    }

    fn url(&self, path: &str) -> String {
        format!("http://127.0.0.1:{}{path}", self.port)
    }

    fn seen(&self) -> Vec<Seen> {
        self.seen.lock().unwrap().clone()
    }
}

fn serve(stream: TcpStream, body: &[u8], port: u16, log: &Mutex<Vec<Seen>>) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut out = stream;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 {
            return;
        }
        let path = line.split_whitespace().nth(1).unwrap_or("/").to_owned();
        let mut headers = Vec::new();
        loop {
            let mut h = String::new();
            reader.read_line(&mut h).unwrap();
            let h = h.trim_end();
            if h.is_empty() {
                break;
            }
            if let Some((n, v)) = h.split_once(':') {
                headers.push((n.trim().to_ascii_lowercase(), v.trim().to_owned()));
            }
        }
        let seen = Seen { path, headers };
        log.lock().unwrap().push(seen.clone());
        let reply = respond(&seen, body, port);
        if out.write_all(&reply).is_err() {
            return;
        }
    }
}

fn response(status: &str, headers: &[(&str, String)], body: &[u8]) -> Vec<u8> {
    let mut out = format!("HTTP/1.1 {status}\r\nContent-Length: {}\r\n", body.len());
    for (n, v) in headers {
        out.push_str(&format!("{n}: {v}\r\n"));
    }
    out.push_str("\r\n");
    let mut out = out.into_bytes();
    out.extend_from_slice(body);
    out
}

fn redirect_to(location: String) -> Vec<u8> {
    response("302 Found", &[("Location", location)], b"")
}

fn respond(seen: &Seen, body: &[u8], port: u16) -> Vec<u8> {
    let file = if seen.path == "/empty" { &[][..] } else { body };
    let total = file.len() as u64;
    match seen.path.as_str() {
        "/redirect" => return redirect_to("/file".into()),
        "/redirect-twice" => return redirect_to("/redirect".into()),
        "/cross" => return redirect_to(format!("http://localhost:{port}/file")),
        "/loop" => return redirect_to("/loop".into()),
        "/gated" => return response("401 Unauthorized", &[], b""),
        "/forbidden" => return response("403 Forbidden", &[], b""),
        "/missing" => return response("404 Not Found", &[], b""),
        "/busy" => return response("503 Service Unavailable", &[], b""),
        "/whole" => return response("200 OK", &[], file),
        _ => {}
    }
    let range = seen
        .header("range")
        .and_then(|r| r.strip_prefix("bytes="))
        .and_then(|r| r.split_once('-'))
        .and_then(|(a, b)| Some((a.parse::<u64>().ok()?, b.parse::<u64>().ok()?)));
    let Some((first, last)) = range else {
        return response("200 OK", &[], file);
    };
    if first >= total || last < first {
        return response(
            "416 Range Not Satisfiable",
            &[("Content-Range", format!("bytes */{total}"))],
            b"",
        );
    }
    let (first, last) = if seen.path == "/shifted" && last + 1 < total {
        (first + 1, last + 1)
    } else {
        (first, last.min(total - 1))
    };
    let mut part = file[first as usize..=last as usize].to_vec();
    match seen.path.as_str() {
        "/short" => {
            part.pop();
        }
        "/long" => part.push(0),
        _ => {}
    }
    response(
        "206 Partial Content",
        &[("Content-Range", format!("bytes {first}-{last}/{total}"))],
        &part,
    )
}

fn client(token: Option<&str>) -> HubClient {
    HubClient::new(&HubConfig {
        token: token.map(str::to_owned),
        allow_loopback_http: true,
        ..HubConfig::default()
    })
    .unwrap()
}

fn sample() -> Vec<u8> {
    (0..10_000u32).map(|i| (i * 31 % 251) as u8).collect()
}

fn read(source: &HubSource, offset: u64, len: usize) -> Result<Vec<u8>, RangeError> {
    let mut buf = vec![0; len];
    source.read_at(offset, &mut buf).map(|()| buf)
}

#[test]
fn reads_exact_ranges_and_learns_the_length() {
    let body = sample();
    let server = Server::start(body.clone());
    let source = client(None).open(&server.url("/file")).unwrap();
    assert_eq!(source.len(), 10_000);
    assert_eq!(read(&source, 0, 16).unwrap(), &body[..16]);
    assert_eq!(read(&source, 9_990, 10).unwrap(), &body[9_990..]);
    assert_eq!(read(&source, 5, 0).unwrap(), Vec::<u8>::new());
    let ranges: Vec<String> = server
        .seen()
        .iter()
        .filter_map(|s| s.header("range").map(str::to_owned))
        .collect();
    assert_eq!(ranges, ["bytes=0-0", "bytes=0-15", "bytes=9990-9999"]);
}

#[test]
fn a_read_past_the_end_is_refused_without_a_request() {
    let server = Server::start(sample());
    let source = client(None).open(&server.url("/file")).unwrap();
    let before = server.seen().len();
    let err = read(&source, 9_995, 10).unwrap_err();
    assert!(matches!(err, RangeError::OutOfRange { .. }), "{err}");
    assert_eq!(server.seen().len(), before);
}

/// Redirects keep `Range`; a redirect to another host drops `Authorization`, so the CDN never
/// sees the token, while the first host does (RFC-0041 §7.2, §7.5).
#[test]
fn redirects_keep_range_and_drop_the_token_across_hosts() {
    let body = sample();
    let server = Server::start(body.clone());
    let hub = client(Some("hf_secret"));
    let same = hub.open(&server.url("/redirect-twice")).unwrap();
    assert_eq!(read(&same, 100, 50).unwrap(), &body[100..150]);
    let cross = hub.open(&server.url("/cross")).unwrap();
    assert_eq!(read(&cross, 200, 50).unwrap(), &body[200..250]);
    let seen = server.seen();
    for s in seen.iter().filter(|s| s.path == "/file") {
        assert!(s.header("range").is_some(), "{s:?}");
    }
    let first = seen.iter().find(|s| s.path == "/cross").unwrap();
    assert_eq!(first.header("authorization"), Some("Bearer hf_secret"));
    let after_cross: Vec<&Seen> = seen
        .iter()
        .skip_while(|s| s.path != "/cross")
        .filter(|s| {
            s.path == "/file" && s.header("host").is_some_and(|h| h.starts_with("localhost"))
        })
        .collect();
    assert!(!after_cross.is_empty());
    for s in after_cross {
        assert_eq!(s.header("authorization"), None, "{s:?}");
    }
}

#[test]
fn a_redirect_loop_is_cut_off() {
    let server = Server::start(sample());
    let err = client(None).open(&server.url("/loop")).unwrap_err();
    assert!(matches!(err, RangeError::Origin { .. }), "{err}");
    let hops = server.seen().iter().filter(|s| s.path == "/loop").count();
    assert_eq!(hops, HubConfig::default().max_redirects + 1);
}

/// A short or long body, a 416, a `Content-Range` for other bytes and a whole-file reply are
/// `HeaderInvalid`, class `Integrity`.
#[test]
fn a_mismatched_reply_is_integrity() {
    let server = Server::start(sample());
    let hub = client(None);
    for path in ["/short", "/long", "/shifted"] {
        let source = hub.open(&server.url(path));
        let err = match source {
            Ok(source) => read(&source, 10, 100).unwrap_err(),
            Err(e) => e,
        };
        assert!(
            matches!(err, RangeError::BadResponse { .. }),
            "{path}: {err}"
        );
        assert_eq!((err.code(), err.class()), (Some(6), ErrorClass::Integrity));
    }
    let err = hub.open(&server.url("/whole")).unwrap_err();
    assert!(matches!(err, RangeError::BadResponse { .. }), "{err}");
    // A 416 for a read inside the length the source learned: the file shrank.
    let small = Server::start(vec![7; 10]);
    let source = hub.open(&small.url("/file")).unwrap();
    let shrunk = HubSource { len: 100, ..source };
    let err = read(&shrunk, 50, 10).unwrap_err();
    assert!(err.to_string().contains("416"), "{err}");
    assert_eq!(err.class(), ErrorClass::Integrity);
}

#[test]
fn statuses_map_to_their_codes() {
    let server = Server::start(sample());
    let hub = client(None);
    for (path, code, class) in [
        ("/gated", 13, ErrorClass::Auth),
        ("/forbidden", 13, ErrorClass::Auth),
        ("/missing", 11, ErrorClass::Link),
        ("/busy", 11, ErrorClass::Link),
    ] {
        let err = hub.open(&server.url(path)).unwrap_err();
        assert_eq!(
            (err.code(), err.class()),
            (Some(code), class),
            "{path}: {err}"
        );
    }
}

#[test]
fn an_unreachable_origin_is_link() {
    // Bind and drop, so nothing listens on the port.
    let port = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let err = client(None)
        .open(&format!("http://127.0.0.1:{port}/file"))
        .unwrap_err();
    assert_eq!(
        (err.code(), err.class()),
        (Some(11), ErrorClass::Link),
        "{err}"
    );
}

#[test]
fn an_empty_file_opens_with_length_zero() {
    let server = Server::start(sample());
    let source = client(None).open(&server.url("/empty")).unwrap();
    assert_eq!(source.len(), 0);
}

/// Plain HTTP is refused unless the config allows it for a loopback address, and never
/// reached through a redirect from HTTPS (the redirect rule is the same check).
#[test]
fn plain_http_is_refused_by_default() {
    let server = Server::start(sample());
    let strict = HubClient::new(&HubConfig::default()).unwrap();
    let err = strict.open(&server.url("/file")).unwrap_err();
    assert!(err.to_string().contains("not HTTPS"), "{err}");
    assert!(server.seen().is_empty());
    let lax = client(None);
    let err = lax.open("http://192.0.2.1/file").unwrap_err();
    assert!(err.to_string().contains("not HTTPS"), "{err}");
    let ok = |u: &str| scheme_allowed(&Url::parse(u).unwrap(), true);
    assert!(ok("http://127.0.0.1/a") && ok("http://[::1]/a") && ok("http://localhost/a"));
    assert!(!ok("http://example.com/a") && !ok("ftp://127.0.0.1/a"));
    assert!(!scheme_allowed(
        &Url::parse("http://127.0.0.1/a").unwrap(),
        false
    ));
}

/// The one header-read routine over a Hub source returns the same bytes as over the recorded
/// fixture of the same file (COMP-E10-T01), with the cap and coalescing unchanged.
#[test]
fn the_header_reader_reads_the_same_bytes_as_the_fixture() {
    let text = include_str!("../../../manifest/tests/data/tiny-header.hubrec");
    let recording = Recording::parse(text).unwrap();
    let mut bytes = vec![0u8; recording.length as usize];
    for (first, body) in &recording.responses {
        bytes[*first as usize..*first as usize + body.len()].copy_from_slice(body);
    }
    let server = Server::start(bytes);
    let hub = client(None).open(&server.url("/file")).unwrap();
    let mut over_hub = HeaderReader::new(hub, 1 << 20).with_initial_read(1024);
    let mut over_fixture =
        HeaderReader::new(Replay::new(recording.clone()), 1 << 20).with_initial_read(1024);
    for (offset, len) in [(0, 24), (24, 1000), (1000, 3000), (0, recording.length)] {
        assert_eq!(
            over_hub.read(offset, len).unwrap(),
            over_fixture.read(offset, len).unwrap(),
            "{len} bytes at {offset}"
        );
    }
    assert_eq!(over_hub.requests(), over_fixture.requests());
    let mut capped = HeaderReader::new(over_hub.source(), 100);
    let err = capped.read(0, 200).unwrap_err();
    assert_eq!(err.code(), Some(5));
}

#[test]
fn resolve_urls_pin_a_commit() {
    let commit = "0123456789abcdef0123456789abcdef01234567";
    let url = resolve_url(HUB_ENDPOINT, "org/model", commit, "sub dir/model.gguf").unwrap();
    assert_eq!(
        url.as_str(),
        format!("https://huggingface.co/org/model/resolve/{commit}/sub%20dir/model.gguf")
    );
    assert!(resolve_url(HUB_ENDPOINT, "org/model", "main", "a").is_err());
    assert!(resolve_url(HUB_ENDPOINT, "org/model", &commit.to_uppercase(), "a").is_err());
    assert!(resolve_url(HUB_ENDPOINT, "model", commit, "a").is_err());
    assert!(resolve_url(HUB_ENDPOINT, "org/model", commit, "../a").is_err());
    assert!(resolve_url(HUB_ENDPOINT, "org/model", commit, "a//b").is_err());
}

#[test]
fn content_range_parses_strictly() {
    assert_eq!(
        parse_content_range("bytes 0-9/10"),
        Some(ContentRange::Bytes {
            first: 0,
            last: 9,
            total: 10
        })
    );
    assert_eq!(
        parse_content_range("bytes */0"),
        Some(ContentRange::Unsatisfied { total: 0 })
    );
    for bad in [
        "bytes 5-4/10",
        "bytes 0-10/10",
        "bytes 0-9/*",
        "items 0-9/10",
        "bytes 0-9",
    ] {
        assert_eq!(parse_content_range(bad), None, "{bad}");
    }
}

#[test]
fn the_token_is_never_printed() {
    let config = HubConfig {
        token: Some("hf_secret".into()),
        ..HubConfig::default()
    };
    assert!(!format!("{config:?}").contains("hf_secret"));
    assert!(!format!("{:?}", HubClient::new(&config).unwrap()).contains("hf_secret"));
    let err = HubClient::new(&HubConfig {
        token: Some("bad\ntoken".into()),
        ..HubConfig::default()
    })
    .unwrap_err();
    assert!(matches!(err, HubError::Token));
}

/// A public Hub file, pinned to a commit, and what `curl` received for its first 4096 bytes on
/// 2026-10-29 (`curl -L -H 'Range: bytes=0-4095' <URL>`: `206`, `Content-Range: bytes
/// 0-4095/453864`), as a length and a BLAKE3 digest computed apart from this crate. Only the
/// digest is kept, not the bytes, since the file is not ours to redistribute.
const LIVE_URL: &str = "https://huggingface.co/hf-internal-testing/tiny-random-gpt2/resolve/71034c5d8bde858ff824298bdedc65515b97d2b9/model.safetensors";
const LIVE_LEN: u64 = 453_864;
const LIVE_HEAD_BLAKE3: &str = "5d96b8d69a2b230e678a473d58b9edc3ac6cf94a753761d25420f9198421f062";

/// Against the live Hub: the first 4096 bytes of a public file, read through the one
/// header-read routine, match what `curl` recorded, and the length matches its
/// `Content-Range`. Needs the network, so it runs only with `SYNTROPY_HUB_LIVE=1`; it goes
/// through `HTTPS_PROXY` when that is set.
#[test]
fn a_public_hub_file_reads_as_recorded() {
    if std::env::var_os("SYNTROPY_HUB_LIVE").is_none() {
        eprintln!("skipped: set SYNTROPY_HUB_LIVE=1 to read from the Hub");
        return;
    }
    let hub = HubClient::new(&HubConfig {
        proxy: std::env::var("HTTPS_PROXY").ok(),
        ..HubConfig::default()
    })
    .unwrap();
    let source = hub.open(LIVE_URL).unwrap();
    assert_eq!(source.len(), LIVE_LEN);
    let mut reader = HeaderReader::new(source, 1 << 20).with_initial_read(4096);
    let head = reader.read(0, 4096).unwrap();
    assert_eq!(blake3::hash(head).to_hex().as_str(), LIVE_HEAD_BLAKE3);
    assert_eq!(reader.requests(), 1);
}
