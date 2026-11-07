// SPDX-License-Identifier: MIT OR Apache-2.0

//! AWS Signature Version 4 for the S3 backend (QUAL-E7-T15): the lab credential signs its
//! requests; anonymous reads are not signed. HMAC-SHA256 is written here over `sha2` (RFC 2104),
//! so signing needs no crate of its own.
//!
//! Every request is signed with the payload hash `UNSIGNED-PAYLOAD`, which S3 accepts over TLS:
//! an object's integrity is its BLAKE3 key, which `verify` and the cache check on every read.

use sha2::{Digest, Sha256};

/// The lab credential (ADR-0603 decision 6): an access key and its secret, and a session token
/// for a temporary credential.
#[derive(Clone)]
pub struct Credential {
    pub access_key: String,
    pub secret: String,
    pub session_token: Option<String>,
}

impl std::fmt::Debug for Credential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never the secret or the token.
        f.debug_struct("Credential")
            .field("access_key", &self.access_key)
            .finish_non_exhaustive()
    }
}

/// The payload hash every signed request carries.
pub const UNSIGNED_PAYLOAD: &str = "UNSIGNED-PAYLOAD";

/// HMAC-SHA256 of `data` under `key` (RFC 2104, block size 64).
pub fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
    let mut block = [0u8; 64];
    if key.len() > 64 {
        block[..32].copy_from_slice(&Sha256::digest(key));
    } else {
        block[..key.len()].copy_from_slice(key);
    }
    let mut inner = Sha256::new();
    inner.update(block.map(|b| b ^ 0x36));
    inner.update(data);
    let mut outer = Sha256::new();
    outer.update(block.map(|b| b ^ 0x5c));
    outer.update(inner.finalize());
    outer.finalize().into()
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn sha256_hex(data: &[u8]) -> String {
    hex(&Sha256::digest(data))
}

/// `YYYYMMDDTHHMMSSZ` for `secs` since the Unix epoch, in UTC.
pub fn amz_date(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let (y, m, d) = civil(days);
    format!(
        "{y:04}{m:02}{d:02}T{:02}{:02}{:02}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

/// The proleptic Gregorian date of a day count since 1970-01-01.
fn civil(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = yoe + era * 400 + i64::from(m <= 2);
    (y, m, d)
}

/// S3's URI encoding of one path: every byte but unreserved characters and `/` is
/// percent-encoded.
pub fn encode_path(path: &str) -> String {
    let mut out = String::with_capacity(path.len());
    for b in path.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~' | b'/') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// One request to sign: its method, its path (unencoded, starting with `/`) and its headers,
/// which must include `host`, `x-amz-date` and `x-amz-content-sha256`. The query is empty for
/// every request the backend makes.
pub struct Request<'a> {
    pub method: &'a str,
    pub path: &'a str,
    pub headers: &'a [(&'a str, String)],
}

/// The canonical request (SigV4 task 1) and its signed-header list.
pub fn canonical(req: &Request<'_>, payload_hash: &str) -> (String, String) {
    let mut headers: Vec<(String, String)> = req
        .headers
        .iter()
        .map(|(k, v)| {
            (
                k.to_ascii_lowercase(),
                v.split_whitespace().collect::<Vec<_>>().join(" "),
            )
        })
        .collect();
    headers.sort();
    let signed = headers
        .iter()
        .map(|(k, _)| k.as_str())
        .collect::<Vec<_>>()
        .join(";");
    let mut text = format!("{}\n{}\n\n", req.method, encode_path(req.path));
    for (k, v) in &headers {
        text.push_str(&format!("{k}:{v}\n"));
    }
    text.push_str(&format!("\n{signed}\n{payload_hash}"));
    (text, signed)
}

/// The `Authorization` header for `req`, signed at `amz_date` (which `req`'s `x-amz-date`
/// header must equal) for `region` and service `s3`.
pub fn authorization(
    req: &Request<'_>,
    payload_hash: &str,
    amz_date: &str,
    region: &str,
    cred: &Credential,
) -> String {
    let (canonical, signed) = canonical(req, payload_hash);
    let date = &amz_date[..8];
    let scope = format!("{date}/{region}/s3/aws4_request");
    let to_sign = format!(
        "AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{}",
        sha256_hex(canonical.as_bytes())
    );
    let k_date = hmac_sha256(format!("AWS4{}", cred.secret).as_bytes(), date.as_bytes());
    let k_region = hmac_sha256(&k_date, region.as_bytes());
    let k_service = hmac_sha256(&k_region, b"s3");
    let k_signing = hmac_sha256(&k_service, b"aws4_request");
    let signature = hex(&hmac_sha256(&k_signing, to_sign.as_bytes()));
    format!(
        "AWS4-HMAC-SHA256 Credential={}/{scope}, SignedHeaders={signed}, Signature={signature}",
        cred.access_key
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RFC 4231 test cases 1, 2 and 6 (a key longer than the block).
    #[test]
    fn hmac_matches_rfc_4231() {
        assert_eq!(
            hex(&hmac_sha256(&[0x0b; 20], b"Hi There")),
            "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
        );
        assert_eq!(
            hex(&hmac_sha256(b"Jefe", b"what do ya want for nothing?")),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
        assert_eq!(
            hex(&hmac_sha256(
                &[0xaa; 131],
                b"Test Using Larger Than Block-Size Key - Hash Key First"
            )),
            "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"
        );
    }

    #[test]
    fn amz_dates_are_utc_civil_dates() {
        assert_eq!(amz_date(0), "19700101T000000Z");
        // 2013-05-24T00:00:00Z, the date of the S3 signing example below.
        assert_eq!(amz_date(1_369_353_600), "20130524T000000Z");
        // A leap day and the end of a day.
        assert_eq!(amz_date(951_868_799), "20000229T235959Z");
        assert_eq!(amz_date(1_709_164_800), "20240229T000000Z");
    }

    #[test]
    fn paths_keep_slashes_and_unreserved_bytes() {
        assert_eq!(encode_path("/fixtures/ab.licence"), "/fixtures/ab.licence");
        assert_eq!(encode_path("/a b/c+d"), "/a%20b/c%2Bd");
    }

    /// The "GET Object" example of the S3 SigV4 documentation ("Signature Calculations for the
    /// Authorization Header: Transferring Payload in a Single Chunk"): its documented signature.
    #[test]
    fn the_s3_get_object_example_signs_as_documented() {
        let empty = sha256_hex(b"");
        assert_eq!(
            empty,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        let headers = [
            ("Host", "examplebucket.s3.amazonaws.com".to_owned()),
            ("Range", "bytes=0-9".to_owned()),
            ("x-amz-content-sha256", empty.clone()),
            ("x-amz-date", "20130524T000000Z".to_owned()),
        ];
        let req = Request {
            method: "GET",
            path: "/test.txt",
            headers: &headers,
        };
        let cred = Credential {
            access_key: "AKIAIOSFODNN7EXAMPLE".into(),
            secret: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY".into(),
            session_token: None,
        };
        let auth = authorization(&req, &empty, "20130524T000000Z", "us-east-1", &cred);
        assert_eq!(
            auth,
            "AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request, \
             SignedHeaders=host;range;x-amz-content-sha256;x-amz-date, \
             Signature=f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41"
        );
    }

    #[test]
    fn debug_never_shows_the_secret() {
        let c = Credential {
            access_key: "AK".into(),
            secret: "s3cr3t".into(),
            session_token: Some("t0k3n".into()),
        };
        let shown = format!("{c:?}");
        assert!(
            !shown.contains("s3cr3t") && !shown.contains("t0k3n"),
            "{shown}"
        );
    }
}
