// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Hub's HTTP range source (RFC-0041 §2, §7.2; COMP-E10-T14): a [`RangeSource`] over a
//! file on the Hub, read with HTTP range requests against its commit-pinned resolve URL.
//!
//! `store` is the one crate that links the HTTPS client, `reqwest` (ADR-0432, proposed, which
//! answers OQ-MF-18; RFC-0001 §4.2 rule 6). `reqwest`'s blocking client matches the blocking
//! [`RangeSource::read_at`]. It runs its own runtime on a background thread, so a [`HubClient`]
//! must be built, used and dropped on a blocking thread, never inside an async task, where it
//! panics: on the daemon's `bulk` runtime every call goes through `spawn_blocking`.
//!
//! TLS is `rustls` on `ring`'s provider, with the platform's verifier, so a Hub read trusts what
//! the OS trusts and no OpenSSL or system TLS library is linked (ADR-0018 §3).
//!
//! What a [`HubSource`] does on every read:
//! - asks for exactly the read's bytes with one `Range` request, and accepts only a
//!   `206 Partial Content` whose `Content-Range` names those bytes and whose body is exactly
//!   that long. Anything else is [`RangeError::BadResponse`] (`HeaderInvalid`, class
//!   `Integrity`): a 416, a short, cut-off or long body, a `Content-Range` for other bytes, and a
//!   `200` with the whole file, since every origin RFC-0041 §7.1 reads by range must honour
//!   `Range`;
//! - starts from the resolve URL and follows its redirects itself, up to
//!   [`HubConfig::max_redirects`], with `Range` on every hop and never from HTTPS to plain HTTP.
//!   Each read re-resolves, so a signed CDN URL that expires between reads is never reused;
//! - sends the token, in `Authorization` only, on a hop whose origin (scheme, host and port) is
//!   the Hub endpoint's, and on no other, so neither the CDN nor any other origin sees it
//!   (RFC-0041 §7.5). No `Referer` is sent, so a signed CDN URL does not reach the next hop;
//! - maps a 401 or 403 to [`RangeError::NoAccess`] (`GatedNoAccess`, class `Auth`), and a
//!   transport failure, a timeout or any other status to [`RangeError::Origin`]
//!   (`OriginUnavailable`, class `Link`), which carries a `Retry-After` given in seconds.
//!
//! The one header-read routine, [`syntropy_manifest::HeaderReader`], wraps a `HubSource` like
//! any other source, so the header-read cap and the coalescing of COMP-E10-T01 apply to Hub
//! reads unchanged. Retries, `hub.max_streams` and its adaptation on 429 and 503, and a total
//! deadline per fetch on the daemon's `Clock` belong to the chunk fetch (RFC-0041 §7.3,
//! COMP-E12-T02), which builds on this source.

use std::fmt;
use std::io::Read;
use std::net::IpAddr;
use std::sync::Arc;
use std::time::Duration;

use reqwest::StatusCode;
use reqwest::Url;
use reqwest::blocking::{Client, Response};
use reqwest::header::{
    ACCEPT_ENCODING, AUTHORIZATION, CONTENT_RANGE, HeaderValue, LOCATION, RANGE, RETRY_AFTER,
};
use reqwest::redirect;
use syntropy_manifest::range::{RangeError, RangeSource, check_bounds};

/// The Hub's public endpoint.
pub const HUB_ENDPOINT: &str = "https://huggingface.co";

/// How a [`HubClient`] reaches the Hub.
#[derive(Clone)]
pub struct HubConfig {
    /// The Hub endpoint: the one origin the token is sent to. Default [`HUB_ENDPOINT`].
    pub endpoint: String,
    /// The Hub token, sent as `Authorization: Bearer <token>` to [`endpoint`](Self::endpoint)'s
    /// origin only. `None` reads public files only.
    pub token: Option<String>,
    /// The longest wait for a hop's response headers, and then for each read of its body. A
    /// body that keeps arriving, however slowly, is not cut off; the chunk fetch sets a total
    /// deadline (module docs). The default is 60 s \[target\].
    pub timeout: Duration,
    /// The longest a connection may take to open. The default is 30 s \[target\].
    pub connect_timeout: Duration,
    /// The most redirects one read follows. The default is 5 \[target\].
    pub max_redirects: usize,
    /// Whether plain HTTP to a loopback IP address is allowed, for tests against a local
    /// server. Every other URL must be HTTPS. Default `false`.
    pub allow_loopback_http: bool,
    /// A proxy every request goes through, as a URL. `None`, the default, connects directly:
    /// the system's proxy settings, such as `HTTPS_PROXY`, are not read. Whether the Hub
    /// source should read them, or a `hub.proxy` key, is ADR-0432's open point 4; this field
    /// is the mechanism either answer would use, and nothing sets it outside tests yet.
    pub proxy: Option<String>,
}

impl Default for HubConfig {
    fn default() -> Self {
        Self {
            endpoint: HUB_ENDPOINT.to_owned(),
            token: None,
            timeout: Duration::from_secs(60),
            connect_timeout: Duration::from_secs(30),
            max_redirects: 5,
            allow_loopback_http: false,
            proxy: None,
        }
    }
}

impl fmt::Debug for HubConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HubConfig")
            .field(
                "endpoint",
                &Url::parse(&self.endpoint).map_or_else(|_| "<unparsed>".into(), |u| redact(&u)),
            )
            .field("token", &self.token.as_ref().map(|_| "<redacted>"))
            .field("timeout", &self.timeout)
            .field("connect_timeout", &self.connect_timeout)
            .field("max_redirects", &self.max_redirects)
            .field("allow_loopback_http", &self.allow_loopback_http)
            .field("proxy", &self.proxy.as_ref().map(|_| "<set>"))
            .finish()
    }
}

/// Why a [`HubClient`] could not be built, or a URL not used.
#[non_exhaustive]
#[derive(Debug)]
pub enum HubError {
    /// TLS could not be set up: the platform's verifier failed to load its roots, say.
    Tls(String),
    /// The HTTP client could not be built.
    Client(String),
    /// The URL is malformed, not HTTPS, or names no file.
    Url(String),
    /// The token is not a valid header value.
    Token,
}

impl fmt::Display for HubError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HubError::Tls(m) => write!(f, "TLS setup failed: {m}"),
            HubError::Client(m) => write!(f, "the HTTP client could not be built: {m}"),
            HubError::Url(m) => write!(f, "bad Hub URL: {m}"),
            HubError::Token => write!(f, "the Hub token is not a valid header value"),
        }
    }
}

impl std::error::Error for HubError {}

/// The token and the one origin it may go to.
#[derive(Clone)]
struct Credential {
    origin: reqwest::Url,
    header: Arc<HeaderValue>,
}

/// The HTTP client every Hub read in a process shares. Cheap to clone. Build it, use it and
/// drop it only on blocking threads (module docs).
#[derive(Clone)]
pub struct HubClient {
    http: Client,
    credential: Option<Credential>,
    max_redirects: usize,
    allow_loopback_http: bool,
}

impl fmt::Debug for HubClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HubClient")
            .field(
                "token_for",
                &self.credential.as_ref().map(|c| c.origin.as_str()),
            )
            .field("max_redirects", &self.max_redirects)
            .field("allow_loopback_http", &self.allow_loopback_http)
            .finish_non_exhaustive()
    }
}

impl HubClient {
    /// A client with `config`.
    ///
    /// # Errors
    ///
    /// [`HubError::Url`] for an endpoint that does not parse or is not HTTPS,
    /// [`HubError::Token`] for a token that is not a header value, [`HubError::Tls`] when the
    /// platform's verifier cannot be set up, and [`HubError::Client`] when the client cannot be
    /// built.
    pub fn new(config: &HubConfig) -> Result<Self, HubError> {
        let endpoint = Url::parse(&config.endpoint)
            .map_err(|e| HubError::Url(format!("an endpoint that does not parse: {e}")))?;
        if !endpoint.username().is_empty() || endpoint.password().is_some() {
            return Err(HubError::Url(
                "an endpoint with a user name or password".into(),
            ));
        }
        if !scheme_allowed(&endpoint, config.allow_loopback_http) {
            return Err(HubError::Url(format!(
                "endpoint {} is not HTTPS",
                redact(&endpoint)
            )));
        }
        let credential = config
            .token
            .as_ref()
            .map(|t| {
                let mut header =
                    HeaderValue::try_from(format!("Bearer {t}")).map_err(|_| HubError::Token)?;
                header.set_sensitive(true);
                Ok(Credential {
                    origin: origin_of(&endpoint),
                    header: Arc::new(header),
                })
            })
            .transpose()?;
        let builder = match &config.proxy {
            Some(url) => Client::builder().proxy(
                reqwest::Proxy::all(url.as_str())
                    .map_err(|e| HubError::Client(format!("proxy: {e}")))?,
            ),
            None => Client::builder().no_proxy(),
        };
        let http = builder
            .tls_backend_preconfigured(tls_config()?)
            .user_agent(concat!("syntropy/", env!("CARGO_PKG_VERSION")))
            .timeout(config.timeout)
            .connect_timeout(config.connect_timeout)
            // Redirects are followed in `get`, so the token's origin is checked on every hop.
            .redirect(redirect::Policy::none())
            .referer(false)
            .build()
            .map_err(|e| HubError::Client(e.to_string()))?;
        Ok(Self {
            http,
            credential,
            max_redirects: config.max_redirects,
            allow_loopback_http: config.allow_loopback_http,
        })
    }

    /// Opens the file at `url` as a [`RangeSource`], learning its length from a one-byte range
    /// request.
    ///
    /// # Errors
    ///
    /// [`RangeError::Origin`] for a malformed or non-HTTPS URL, or one that carries a user name
    /// or password, and the errors of a read (module docs).
    pub fn open(&self, url: &str) -> Result<HubSource, RangeError> {
        let url =
            Url::parse(url).map_err(|e| origin(None, format!("a URL that does not parse: {e}")))?;
        if !url.username().is_empty() || url.password().is_some() {
            return Err(origin(None, "a URL with a user name or password"));
        }
        if !scheme_allowed(&url, self.allow_loopback_http) {
            return Err(origin(None, format!("{} is not HTTPS", redact(&url))));
        }
        let response = self.get(&url, "bytes=0-0")?;
        let len = match response.status() {
            StatusCode::PARTIAL_CONTENT => match content_range(&response) {
                Some(ContentRange::Bytes {
                    first: 0,
                    last: 0,
                    total,
                }) => total,
                other => return Err(bad(0, 1, format!("Content-Range {other:?} for bytes 0-0"))),
            },
            // An empty file has no byte 0 to send.
            StatusCode::RANGE_NOT_SATISFIABLE => match content_range(&response) {
                Some(ContentRange::Unsatisfied { total: 0 }) => 0,
                other => {
                    return Err(bad(
                        0,
                        1,
                        format!("416 for byte 0 with Content-Range {other:?}, not bytes */0"),
                    ));
                }
            },
            StatusCode::OK => {
                return Err(bad(
                    0,
                    1,
                    "the origin ignored Range and sent the whole file",
                ));
            }
            status => return Err(status_error(status, &response)),
        };
        Ok(HubSource {
            client: self.clone(),
            url,
            len,
        })
    }

    /// GETs `range` of `url`, following redirects (module docs).
    fn get(&self, url: &Url, range: &str) -> Result<Response, RangeError> {
        let mut at = url.clone();
        let mut hops = 0usize;
        loop {
            let mut request = self
                .http
                .get(at.clone())
                .header(RANGE, range)
                .header(ACCEPT_ENCODING, "identity");
            if let Some(credential) = &self.credential
                && origin_of(&at) == credential.origin
            {
                request = request.header(AUTHORIZATION, (*credential.header).clone());
            }
            let response = request.send().map_err(transport)?;
            if !response.status().is_redirection() {
                return Ok(response);
            }
            let Some(next) = response
                .headers()
                .get(LOCATION)
                .and_then(|l| l.to_str().ok())
                .and_then(|l| at.join(l).ok())
            else {
                return Err(origin(
                    Some(response.status().as_u16()),
                    format!("a redirect from {} with no usable Location", redact(&at)),
                ));
            };
            hops += 1;
            if hops > self.max_redirects {
                return Err(origin(
                    None,
                    format!(
                        "more than {} redirects from {}",
                        self.max_redirects,
                        redact(url)
                    ),
                ));
            }
            if !scheme_allowed(&next, self.allow_loopback_http) {
                return Err(origin(
                    None,
                    format!(
                        "a redirect from {} to {}, not HTTPS",
                        redact(&at),
                        redact(&next)
                    ),
                ));
            }
            at = next;
        }
    }
}

/// The resolve URL of `path` in model repository `repo` at commit `commit` on the Hub at
/// `endpoint`: `<endpoint>/<repo>/resolve/<commit>/<path>`. Pinning the commit makes every
/// read of the file see the same bytes (RFC-0041 §7.1).
///
/// # Errors
///
/// [`HubError::Url`] when `commit` is not a 40-digit lower-case hex commit ID, `repo` is not
/// `<owner>/<name>`, `path` is empty or holds an empty, `.` or `..` segment, or `endpoint` does
/// not parse.
pub fn resolve_url(endpoint: &str, repo: &str, commit: &str, path: &str) -> Result<Url, HubError> {
    if commit.len() != 40
        || !commit
            .bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
    {
        return Err(HubError::Url(format!(
            "{commit:?} is not a 40-digit lower-case hex commit ID"
        )));
    }
    let repo_parts: Vec<&str> = repo.split('/').collect();
    if repo_parts.len() != 2 || repo_parts.iter().any(|p| !plain_segment(p)) {
        return Err(HubError::Url(format!("{repo:?} is not <owner>/<name>")));
    }
    let path_parts: Vec<&str> = path.split('/').collect();
    if path_parts.iter().any(|p| !plain_segment(p)) {
        return Err(HubError::Url(format!("{path:?} is not a file path")));
    }
    let mut url =
        Url::parse(endpoint).map_err(|e| HubError::Url(format!("endpoint {endpoint}: {e}")))?;
    url.path_segments_mut()
        .map_err(|()| HubError::Url(format!("endpoint {endpoint} cannot have a path")))?
        .pop_if_empty()
        .extend(repo_parts)
        .extend(["resolve", commit])
        .extend(path_parts);
    Ok(url)
}

fn plain_segment(s: &str) -> bool {
    !s.is_empty() && s != "." && s != ".."
}

/// A file on the Hub, read with range requests (module docs).
pub struct HubSource {
    client: HubClient,
    url: Url,
    len: u64,
}

impl fmt::Debug for HubSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HubSource")
            .field("url", &redact(&self.url))
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

impl HubSource {
    /// The URL it reads.
    #[must_use]
    pub fn url(&self) -> &Url {
        &self.url
    }
}

impl RangeSource for HubSource {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), RangeError> {
        let len = u64::try_from(buf.len()).unwrap_or(u64::MAX);
        check_bounds(offset, len, self.len)?;
        if len == 0 {
            return Ok(());
        }
        let last = offset + len - 1;
        let response = self
            .client
            .get(&self.url, &format!("bytes={offset}-{last}"))?;
        match response.status() {
            StatusCode::PARTIAL_CONTENT => {}
            StatusCode::RANGE_NOT_SATISFIABLE => {
                return Err(bad(offset, len, "416 Range Not Satisfiable"));
            }
            StatusCode::OK => {
                return Err(bad(offset, len, "the origin ignored Range"));
            }
            status => return Err(status_error(status, &response)),
        }
        let expected = ContentRange::Bytes {
            first: offset,
            last,
            total: self.len,
        };
        let got = content_range(&response);
        if got != Some(expected) {
            return Err(bad(offset, len, format!("Content-Range {got:?}")));
        }
        // Read one byte past the range, so a long body is caught as well as a short one.
        let mut body = response.take(len + 1);
        let mut filled = 0usize;
        loop {
            let into = if filled < buf.len() {
                &mut buf[filled..]
            } else {
                &mut [0u8; 1][..]
            };
            match body.read(into) {
                Ok(0) => break,
                Ok(n) if filled < buf.len() => filled += n,
                Ok(_) => return Err(bad(offset, len, "the body is longer than the range")),
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                // A stalled body is the link's failure, which the fetch may retry.
                Err(e) if timed_out(&e) => {
                    return Err(origin(
                        None,
                        format!("the body timed out after {filled} of {len} bytes"),
                    ));
                }
                // A body cut off before its `Content-Length` is a short read. A reset link
                // looks the same at this layer.
                Err(e) => {
                    return Err(bad(
                        offset,
                        len,
                        format!(
                            "the body failed after {filled} of {len} bytes: {}",
                            chain(&e)
                        ),
                    ));
                }
            }
        }
        if filled < buf.len() {
            return Err(bad(
                offset,
                len,
                format!("the body ended after {filled} of {len} bytes"),
            ));
        }
        Ok(())
    }
}

/// A `Content-Range` header's value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ContentRange {
    /// `bytes <first>-<last>/<total>`.
    Bytes { first: u64, last: u64, total: u64 },
    /// `bytes */<total>`, sent with a 416.
    Unsatisfied { total: u64 },
}

fn content_range(response: &Response) -> Option<ContentRange> {
    parse_content_range(response.headers().get(CONTENT_RANGE)?.to_str().ok()?)
}

fn parse_content_range(value: &str) -> Option<ContentRange> {
    let rest = value.strip_prefix("bytes ")?;
    let (range, total) = rest.split_once('/')?;
    let total: u64 = total.parse().ok()?;
    if range == "*" {
        return Some(ContentRange::Unsatisfied { total });
    }
    let (first, last) = range.split_once('-')?;
    let (first, last): (u64, u64) = (first.parse().ok()?, last.parse().ok()?);
    (first <= last && last < total).then_some(ContentRange::Bytes { first, last, total })
}

/// HTTPS, or, when allowed, plain HTTP to a loopback IP literal. A host name is never taken as
/// loopback, since it resolves through whatever resolver the machine has.
fn scheme_allowed(url: &Url, allow_loopback_http: bool) -> bool {
    match url.scheme() {
        "https" => true,
        "http" if allow_loopback_http => url.host_str().is_some_and(|host| {
            host.trim_start_matches('[')
                .trim_end_matches(']')
                .parse::<IpAddr>()
                .is_ok_and(|ip| ip.is_loopback())
        }),
        _ => false,
    }
}

/// The URL's origin: its scheme, host and port, with no path, query or user.
fn origin_of(url: &Url) -> Url {
    let mut origin = url.clone();
    origin.set_path("/");
    origin.set_query(None);
    origin.set_fragment(None);
    let _ = origin.set_username("");
    let _ = origin.set_password(None);
    let _ = origin.set_port(url.port_or_known_default());
    origin
}

/// `rustls` on `ring`'s provider with the platform's verifier, offering HTTP/2 and HTTP/1.1.
fn tls_config() -> Result<rustls::ClientConfig, HubError> {
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let verifier = rustls_platform_verifier::Verifier::new(provider.clone())
        .map_err(|e| HubError::Tls(e.to_string()))?;
    let mut config = rustls::ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .map_err(|e| HubError::Tls(e.to_string()))?
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(verifier))
        .with_no_client_auth();
    config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
    Ok(config)
}

fn bad(offset: u64, len: u64, message: impl Into<String>) -> RangeError {
    RangeError::BadResponse {
        offset,
        len,
        message: message.into(),
    }
}

fn origin(status: Option<u16>, message: impl Into<String>) -> RangeError {
    RangeError::Origin {
        status,
        retry_after_secs: None,
        message: message.into(),
    }
}

fn status_error(status: StatusCode, response: &Response) -> RangeError {
    let message = format!("{status} from {}", redact(response.url()));
    if status == StatusCode::UNAUTHORIZED || status == StatusCode::FORBIDDEN {
        RangeError::NoAccess {
            status: status.as_u16(),
            message,
        }
    } else {
        // `Retry-After` in seconds. An HTTP date would need the wall clock, which the chunk
        // fetch reads through its `Clock`; it is left out.
        let retry_after_secs = response
            .headers()
            .get(RETRY_AFTER)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.trim().parse().ok());
        RangeError::Origin {
            status: Some(status.as_u16()),
            retry_after_secs,
            message,
        }
    }
}

fn transport(e: reqwest::Error) -> RangeError {
    let status = e.status().map(|s| s.as_u16());
    let what = if e.is_timeout() {
        "timed out"
    } else if e.is_connect() {
        "could not connect"
    } else {
        "request failed"
    };
    let at = e.url().map(redact);
    let message = match at {
        Some(url) => format!("{what} ({url}): {}", chain(&e.without_url())),
        None => format!("{what}: {}", chain(&e)),
    };
    origin(status, message)
}

/// Whether a body read failed because the read timed out.
fn timed_out(e: &std::io::Error) -> bool {
    e.kind() == std::io::ErrorKind::TimedOut
        || e.get_ref()
            .and_then(|inner| inner.downcast_ref::<reqwest::Error>())
            .is_some_and(reqwest::Error::is_timeout)
}

/// An error and its causes, joined with ": ".
fn chain(e: &dyn std::error::Error) -> String {
    let mut out = e.to_string();
    let mut cause = e.source();
    while let Some(c) = cause {
        out.push_str(": ");
        out.push_str(&c.to_string());
        cause = c.source();
    }
    out
}

/// The URL without its user, password and query: a CDN URL's query carries a signature.
fn redact(url: &Url) -> String {
    let mut url = url.clone();
    url.set_query(None);
    let _ = url.set_username("");
    let _ = url.set_password(None);
    url.to_string()
}

#[cfg(test)]
mod tests;
