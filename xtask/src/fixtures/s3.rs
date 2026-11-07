// SPDX-License-Identifier: MIT OR Apache-2.0

//! The S3 backend of the fixture store (QUAL-E7-T15; RFC-0060 §12.3; ADR-0603 decisions 5, 6
//! and 8): a [`Store`] over the project bucket.
//!
//! - **Addressing.** `--store s3+https://<host>[:port]/<bucket>` addresses the bucket by path;
//!   `--store s3+https://<bucket host>/`, with no path, addresses a virtual-hosted bucket.
//!   `s3+http://` is accepted only for a loopback host, which is how the tests reach their
//!   in-process server.
//! - **Anonymous or the lab credential.** With no credential in the environment every request
//!   is anonymous, so only public objects can be read. With the lab credential
//!   (`SYNTROPY_FIXTURES_S3_ACCESS_KEY_ID`, `SYNTROPY_FIXTURES_S3_SECRET_ACCESS_KEY`, and
//!   `SYNTROPY_FIXTURES_S3_SESSION_TOKEN` for a temporary one) every request is signed with
//!   SigV4 ([`super::sigv4`]) for `SYNTROPY_FIXTURES_S3_REGION` (default `us-east-1`). The
//!   credential exists only on lab runners (ADR-0603 decision 6; ADR-0624).
//! - **Statuses.** `200` reads; `404` is [`Error::NotFound`] (and `false` from `exists`); `401`
//!   and `403` are [`Error::Denied`]; anything else, and a request that does not complete, is
//!   [`Error::Remote`]. `exists` never reads a failed request as absent.
//! - **Writes.** `put` spools the data to a temporary file while hashing it, refuses bytes that
//!   do not hash to the key, then sends one `PUT` with its length and the object's ACL
//!   (`x-amz-acl: public-read` or `private`, from [`Access`]). One `PUT` holds at most 5 GiB, S3's
//!   single-request limit; every fixture recorded so far is below it.
//! - **TLS.** `rustls` on `ring` with the platform's verifier, TLS 1.3 only, as ADR-0432 sets up
//!   the Hub client; no proxy.

use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use reqwest::StatusCode;
use reqwest::blocking::{Body, Client, RequestBuilder, Response};

use super::sigv4::{self, Credential, UNSIGNED_PAYLOAD};
use super::store::{Access, Error, Key, Store, check_put, copy_hashing, temp_beside};

/// S3's largest single `PUT`: 5 GiB.
pub const MAX_PUT: u64 = 5 << 30;

/// Where the bucket is and how to reach it.
#[derive(Clone, Debug)]
pub struct S3Config {
    /// `https://host[:port]` or, for a loopback test server, `http://…`.
    pub endpoint: String,
    /// The bucket for path-style addressing; `None` for a virtual-hosted endpoint.
    pub bucket: Option<String>,
    pub region: String,
    /// `None`: every request is anonymous.
    pub credential: Option<Credential>,
    /// The request timeout.
    pub timeout: Duration,
}

impl S3Config {
    /// Parses `--store s3+https://…`, taking the region and the lab credential from `env`.
    /// `None` when `store` is not an `s3+` URL, so the caller treats it as a directory.
    pub fn parse(
        store: &str,
        env: &dyn Fn(&str) -> Option<String>,
    ) -> Option<Result<S3Config, String>> {
        let rest = store.strip_prefix("s3+")?;
        Some(Self::parse_url(rest, env))
    }

    fn parse_url(url: &str, env: &dyn Fn(&str) -> Option<String>) -> Result<S3Config, String> {
        let (scheme, after) = url
            .split_once("://")
            .ok_or_else(|| format!("`s3+{url}` is not a URL"))?;
        let (authority, path) = after.split_once('/').unwrap_or((after, ""));
        if authority.is_empty() {
            return Err(format!("`s3+{url}` has no host"));
        }
        let host = authority
            .rsplit_once(':')
            .filter(|(_, port)| port.bytes().all(|b| b.is_ascii_digit()))
            .map_or(authority, |(h, _)| h);
        match scheme {
            "https" => {}
            "http" if matches!(host, "127.0.0.1" | "localhost" | "[::1]") => {}
            "http" => return Err("`s3+http://` is allowed only for a loopback host".into()),
            other => return Err(format!("unknown scheme `{other}`")),
        }
        let bucket = path.trim_matches('/');
        if bucket.contains('/') {
            return Err(format!("`{bucket}` is not a bucket name"));
        }
        let credential = match (
            env("SYNTROPY_FIXTURES_S3_ACCESS_KEY_ID"),
            env("SYNTROPY_FIXTURES_S3_SECRET_ACCESS_KEY"),
        ) {
            (Some(access_key), Some(secret)) => Some(Credential {
                access_key,
                secret,
                session_token: env("SYNTROPY_FIXTURES_S3_SESSION_TOKEN"),
            }),
            (None, None) => None,
            _ => {
                return Err(
                    "the lab credential needs both SYNTROPY_FIXTURES_S3_ACCESS_KEY_ID and \
                            SYNTROPY_FIXTURES_S3_SECRET_ACCESS_KEY"
                        .into(),
                );
            }
        };
        Ok(S3Config {
            endpoint: format!("{scheme}://{authority}"),
            bucket: (!bucket.is_empty()).then(|| bucket.to_owned()),
            region: env("SYNTROPY_FIXTURES_S3_REGION").unwrap_or_else(|| "us-east-1".into()),
            credential,
            timeout: Duration::from_secs(600),
        })
    }
}

/// The store over the project bucket.
pub struct S3Store {
    config: S3Config,
    host: String,
    http: Client,
}

/// `rustls` on `ring`'s provider with the platform's verifier, TLS 1.3 only (ADR-0432).
fn tls_config() -> Result<rustls::ClientConfig, String> {
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let verifier = rustls_platform_verifier::Verifier::new(provider.clone())
        .map_err(|e| format!("TLS verifier: {e}"))?;
    let mut config = rustls::ClientConfig::builder_with_provider(provider)
        .with_protocol_versions(&[&rustls::version::TLS13])
        .map_err(|e| format!("TLS: {e}"))?
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(verifier))
        .with_no_client_auth();
    config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
    Ok(config)
}

fn remote(e: impl std::fmt::Display) -> Error {
    Error::Remote(e.to_string())
}

impl S3Store {
    pub fn new(config: S3Config) -> Result<S3Store, String> {
        let host = config
            .endpoint
            .split_once("://")
            .map(|(_, a)| a.to_owned())
            .ok_or("endpoint without a scheme")?;
        let http = Client::builder()
            .no_proxy()
            .tls_backend_preconfigured(tls_config()?)
            .user_agent(concat!("syntropy-xtask/", env!("CARGO_PKG_VERSION")))
            .timeout(config.timeout)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| format!("HTTP client: {e}"))?;
        Ok(S3Store { config, host, http })
    }

    /// The request path of `key`: `/<bucket>/<prefix>/<name>`, or `/<prefix>/<name>`.
    fn path(&self, key: &Key) -> String {
        match &self.config.bucket {
            Some(b) => format!("/{b}/{key}"),
            None => format!("/{key}"),
        }
    }

    /// A request for `key`, signed when the client holds the lab credential. `extra` headers
    /// are signed too.
    fn request(&self, method: &str, key: &Key, extra: &[(&str, String)]) -> RequestBuilder {
        let path = self.path(key);
        let url = format!("{}{}", self.config.endpoint, sigv4::encode_path(&path));
        let m = reqwest::Method::from_bytes(method.as_bytes()).expect("a known method");
        let mut rb = self.http.request(m, url);
        let Some(cred) = &self.config.credential else {
            for (k, v) in extra {
                rb = rb.header(*k, v);
            }
            return rb;
        };
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let date = sigv4::amz_date(secs);
        let mut headers: Vec<(&str, String)> = vec![
            ("host", self.host.clone()),
            ("x-amz-content-sha256", UNSIGNED_PAYLOAD.to_owned()),
            ("x-amz-date", date.clone()),
        ];
        if let Some(token) = &cred.session_token {
            headers.push(("x-amz-security-token", token.clone()));
        }
        headers.extend(extra.iter().cloned());
        let auth = sigv4::authorization(
            &sigv4::Request {
                method,
                path: &path,
                headers: &headers,
            },
            UNSIGNED_PAYLOAD,
            &date,
            &self.config.region,
            cred,
        );
        for (k, v) in &headers {
            // `host` is the URL's; reqwest sets it.
            if *k != "host" {
                rb = rb.header(*k, v);
            }
        }
        rb.header("authorization", auth)
    }

    /// The error for a reply that is not a success.
    fn status_error(key: &Key, resp: &Response) -> Error {
        match resp.status() {
            StatusCode::NOT_FOUND => Error::NotFound(key.to_string()),
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
                Error::Denied(format!("{key}: HTTP {}", resp.status()))
            }
            s => Error::Remote(format!("{key}: HTTP {s}")),
        }
    }
}

impl Store for S3Store {
    fn open(&self, key: &Key) -> Result<Box<dyn Read>, Error> {
        let resp = self.request("GET", key, &[]).send().map_err(remote)?;
        if resp.status() == StatusCode::OK {
            Ok(Box::new(resp))
        } else {
            Err(Self::status_error(key, &resp))
        }
    }

    fn put(&self, key: &Key, data: &mut dyn Read, access: Access) -> Result<(), Error> {
        let spool = temp_beside(&std::env::temp_dir().join(format!("syntropy-s3-{}", key.hex())));
        let result = (|| {
            let mut file = File::options()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&spool)?;
            let (actual, len) = copy_hashing(data, &mut file)?;
            check_put(key, &actual)?;
            if len > MAX_PUT {
                return Err(Error::Remote(format!(
                    "{key}: {len} bytes is over S3's single-PUT limit of {MAX_PUT}"
                )));
            }
            file.seek(SeekFrom::Start(0))?;
            let acl = match access {
                Access::Public => "public-read",
                Access::Private => "private",
            };
            let resp = self
                .request("PUT", key, &[("x-amz-acl", acl.to_owned())])
                .header("content-length", len)
                .body(Body::sized(file, len))
                .send()
                .map_err(remote)?;
            if resp.status().is_success() {
                Ok(())
            } else {
                Err(Self::status_error(key, &resp))
            }
        })();
        let _ = fs::remove_file(&spool);
        result
    }

    fn exists(&self, key: &Key) -> Result<bool, Error> {
        let resp = self.request("HEAD", key, &[]).send().map_err(remote)?;
        match resp.status() {
            StatusCode::OK => Ok(true),
            StatusCode::NOT_FOUND => Ok(false),
            _ => Err(Self::status_error(key, &resp)),
        }
    }
}
