// SPDX-License-Identifier: MIT OR Apache-2.0

//! An in-process S3 server for the tests: plain HTTP/1.1 on a loopback port, path-style
//! addressing of one bucket, and the bucket's access rule as S3 applies it.
//!
//! - A request with an `Authorization` header must carry a valid SigV4 signature for the
//!   server's credential, recomputed over the headers it names; otherwise `403`.
//! - An anonymous `GET` or `HEAD` reads only an object written `public-read`. Anything else
//!   anonymous gets `403`, a missing object included, as S3 answers a client that may not list
//!   the bucket.
//! - A `PUT` needs a signature and a `content-length`, and stores the `x-amz-acl` it names.
//! - [`FakeS3::fail_with`] makes every later request fail with that status.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::{Arc, Mutex};

use super::sigv4::{self, Credential};

pub const BUCKET: &str = "syntropy-fixtures";
pub const REGION: &str = "us-east-1";

/// The lab credential the server accepts.
pub fn lab() -> Credential {
    Credential {
        access_key: "AKIDTESTLAB".into(),
        secret: "test-secret-not-real".into(),
        session_token: None,
    }
}

#[derive(Default)]
struct State {
    /// Path → (bytes, public).
    objects: BTreeMap<String, (Vec<u8>, bool)>,
}

pub struct FakeS3 {
    pub port: u16,
    state: Arc<Mutex<State>>,
    fail: Arc<AtomicU16>,
}

struct Req {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Req {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }
}

fn read_request(r: &mut BufReader<TcpStream>) -> Option<Req> {
    let mut line = String::new();
    if r.read_line(&mut line).ok()? == 0 {
        return None;
    }
    let mut parts = line.split_whitespace();
    let method = parts.next()?.to_owned();
    let path = parts.next()?.to_owned();
    let mut headers = Vec::new();
    loop {
        let mut h = String::new();
        r.read_line(&mut h).ok()?;
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        let (k, v) = h.split_once(':')?;
        headers.push((k.trim().to_ascii_lowercase(), v.trim().to_owned()));
    }
    let len: usize = headers
        .iter()
        .find(|(k, _)| k == "content-length")
        .and_then(|(_, v)| v.parse().ok())
        .unwrap_or(0);
    let mut body = vec![0; len];
    r.read_exact(&mut body).ok()?;
    Some(Req {
        method,
        path,
        headers,
        body,
    })
}

fn respond(w: &mut TcpStream, status: u16, body: &[u8], head: bool) {
    let reason = match status {
        200 => "OK",
        403 => "Forbidden",
        404 => "Not Found",
        411 => "Length Required",
        _ => "Error",
    };
    let _ = write!(
        w,
        "HTTP/1.1 {status} {reason}\r\ncontent-length: {}\r\n\r\n",
        body.len()
    );
    if !head {
        let _ = w.write_all(body);
    }
    let _ = w.flush();
}

/// Whether `req` carries a valid signature for the lab credential.
fn signature_valid(req: &Req) -> bool {
    let Some(auth) = req.header("authorization") else {
        return false;
    };
    let Some(signed) = auth
        .split(", ")
        .find_map(|p| p.strip_prefix("SignedHeaders="))
    else {
        return false;
    };
    let headers: Vec<(&str, String)> = signed
        .split(';')
        .map(|name| (name, req.header(name).unwrap_or_default().to_owned()))
        .collect();
    let Some(date) = req.header("x-amz-date") else {
        return false;
    };
    let payload = req.header("x-amz-content-sha256").unwrap_or_default();
    let path = percent_decode(&req.path);
    let want = sigv4::authorization(
        &sigv4::Request {
            method: &req.method,
            path: &path,
            headers: &headers,
        },
        payload,
        date,
        REGION,
        &lab(),
    );
    want == auth
}

fn percent_decode(p: &str) -> String {
    let b = p.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%'
            && i + 2 < b.len()
            && let Ok(v) = u8::from_str_radix(&p[i + 1..i + 3], 16)
        {
            out.push(v);
            i += 3;
            continue;
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn handle(state: &Mutex<State>, fail: &AtomicU16, mut stream: TcpStream) {
    let mut reader = BufReader::new(stream.try_clone().expect("clone the stream"));
    while let Some(req) = read_request(&mut reader) {
        let head = req.method == "HEAD";
        let failing = fail.load(Ordering::SeqCst);
        if failing != 0 {
            respond(&mut stream, failing, b"injected failure", head);
            continue;
        }
        let path = percent_decode(&req.path);
        if !path.starts_with(&format!("/{BUCKET}/")) {
            respond(&mut stream, 404, b"NoSuchBucket", head);
            continue;
        }
        let signed = req.header("authorization").is_some();
        if signed && !signature_valid(&req) {
            respond(&mut stream, 403, b"SignatureDoesNotMatch", head);
            continue;
        }
        let mut st = state.lock().unwrap();
        match req.method.as_str() {
            "GET" | "HEAD" => match st.objects.get(&path) {
                Some((bytes, public)) if signed || *public => {
                    let bytes = bytes.clone();
                    drop(st);
                    respond(&mut stream, 200, &bytes, head);
                }
                None if signed => respond(&mut stream, 404, b"NoSuchKey", head),
                _ => respond(&mut stream, 403, b"AccessDenied", head),
            },
            "PUT" if signed => {
                if req.header("content-length").is_none() {
                    respond(&mut stream, 411, b"MissingContentLength", false);
                    continue;
                }
                let public = req.header("x-amz-acl") == Some("public-read");
                st.objects.insert(path, (req.body, public));
                drop(st);
                respond(&mut stream, 200, b"", false);
            }
            _ => respond(&mut stream, 403, b"AccessDenied", head),
        }
    }
}

impl FakeS3 {
    pub fn start() -> FakeS3 {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let state = Arc::new(Mutex::new(State::default()));
        let fail = Arc::new(AtomicU16::new(0));
        let (s, f) = (state.clone(), fail.clone());
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let (s, f) = (s.clone(), f.clone());
                std::thread::spawn(move || handle(&s, &f, stream));
            }
        });
        FakeS3 { port, state, fail }
    }

    /// `--store` for this server.
    pub fn spec(&self) -> String {
        format!("s3+http://127.0.0.1:{}/{BUCKET}", self.port)
    }

    /// The environment of a lab runner (`signed`) or of anyone else.
    pub fn env(signed: bool) -> impl Fn(&str) -> Option<String> {
        move |k: &str| {
            let c = lab();
            match k {
                "SYNTROPY_FIXTURES_S3_ACCESS_KEY_ID" if signed => Some(c.access_key),
                "SYNTROPY_FIXTURES_S3_SECRET_ACCESS_KEY" if signed => Some(c.secret),
                "SYNTROPY_FIXTURES_S3_REGION" => Some(REGION.into()),
                _ => None,
            }
        }
    }

    /// Every later request fails with `status`; 0 stops it.
    pub fn fail_with(&self, status: u16) {
        self.fail.store(status, Ordering::SeqCst);
    }

    /// Whether the object at `<prefix>/<name>` is stored, and public.
    pub fn stored(&self, key: &str) -> Option<bool> {
        let st = self.state.lock().unwrap();
        st.objects.get(&format!("/{BUCKET}/{key}")).map(|o| o.1)
    }

    pub fn len(&self) -> usize {
        self.state.lock().unwrap().objects.len()
    }
}
