// SPDX-License-Identifier: MIT OR Apache-2.0

//! The activation quantisation references against ggml's own quantizers, bit for bit
//! (COMP-E13-T03 criterion 1; RFC-0042 §4.3). `data/act_quant/README.md` gives the vectors'
//! provenance: the pinned ggml commit, the generating program and the command.

use syntropy_kernels::reference::act_quant::{
    BlockQ8_0, BlockQ8_1, BlockQ8K, Q8Path, f16_bits, quantize_q8_0, quantize_q8_1, quantize_q8_k,
    round_bf16,
};

const VECTORS: &str = include_str!("data/act_quant/ggml-q8.txt");

/// One recorded case: its name, its input row, and each recorded output by (type, path).
struct Case {
    name: String,
    input: Vec<f32>,
    outputs: Vec<(String, String, Vec<u8>)>,
}

fn hex(s: &str) -> Vec<u8> {
    assert!(s.len().is_multiple_of(2), "odd-length hex");
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex digit"))
        .collect()
}

/// The vectors' header (`ggml`, `host`) and cases.
fn parse() -> (String, String, Vec<Case>) {
    let mut lines = VECTORS.lines().filter(|l| !l.starts_with('#'));
    assert_eq!(lines.next(), Some("syntropy-ggml-q8-vectors 1"));
    let field = |line: Option<&str>, key: &str| {
        line.and_then(|l| l.strip_prefix(key))
            .and_then(|l| l.strip_prefix(' '))
            .unwrap_or_else(|| panic!("missing `{key}` line"))
            .to_owned()
    };
    let ggml = field(lines.next(), "ggml");
    let host = field(lines.next(), "host");
    let mut cases = Vec::new();
    let mut current: Option<Case> = None;
    for line in lines {
        let words: Vec<&str> = line.split(' ').collect();
        match words.as_slice() {
            ["case", name, len] => {
                assert!(current.is_none(), "case {name} opened inside another");
                current = Some(Case {
                    name: (*name).to_owned(),
                    input: Vec::with_capacity(len.parse().expect("length")),
                    outputs: Vec::new(),
                });
            }
            ["in", data] => {
                let case = current.as_mut().expect("`in` outside a case");
                // The generator prints each F32's bits as `%08x`, most significant digit first.
                let bytes = hex(data);
                let (words, rest) = bytes.as_chunks::<4>();
                assert!(rest.is_empty(), "input is not whole F32 words");
                case.input = words
                    .iter()
                    .map(|&b| f32::from_bits(u32::from_be_bytes(b)))
                    .collect();
            }
            [ty, path, data] => {
                let case = current.as_mut().expect("output outside a case");
                case.outputs
                    .push(((*ty).to_owned(), (*path).to_owned(), hex(data)));
            }
            ["end"] => cases.push(current.take().expect("`end` outside a case")),
            _ => panic!("unexpected line: {line}"),
        }
    }
    assert!(current.is_none(), "unterminated case");
    (ggml, host, cases)
}

fn host_path(host: &str) -> Q8Path {
    match host {
        "arm64-neon" => Q8Path::Arm64Neon,
        "x86-avx" => Q8Path::X86Avx,
        "reference" => Q8Path::Reference,
        other => panic!("unknown host path {other}"),
    }
}

fn q8_0_bytes(x: &[f32], path: Q8Path) -> Vec<u8> {
    quantize_q8_0(x, path)
        .expect("whole blocks")
        .iter()
        .flat_map(BlockQ8_0::to_bytes)
        .collect()
}

fn q8_1_bytes(x: &[f32], path: Q8Path) -> Vec<u8> {
    quantize_q8_1(x, path)
        .expect("whole blocks")
        .iter()
        .flat_map(BlockQ8_1::to_bytes)
        .collect()
}

fn q8_k_bytes(x: &[f32]) -> Vec<u8> {
    quantize_q8_k(x)
        .expect("whole blocks")
        .iter()
        .flat_map(BlockQ8K::to_bytes)
        .collect()
}

/// The index of the first byte where `got` and `want` differ, for the failure message.
fn first_difference(got: &[u8], want: &[u8]) -> Option<usize> {
    if got.len() != want.len() {
        return Some(got.len().min(want.len()));
    }
    got.iter().zip(want).position(|(a, b)| a != b)
}

#[test]
fn the_vectors_are_pinned_and_cover_every_quantizer() {
    let (ggml, host, cases) = parse();
    assert_eq!(ggml, "da8e2a54b919557b816b1c2beb432ccb766f7ce0");
    host_path(&host);
    assert!(cases.len() >= 20, "{} cases", cases.len());
    for (ty, path) in [
        ("q8_0", "ref"),
        ("q8_0", "cpu"),
        ("q8_1", "ref"),
        ("q8_1", "cpu"),
        ("q8_K", "ref"),
        ("q8_K", "cpu"),
        ("f16", "ref"),
        ("bf16", "ref"),
        ("bf16", "cpu"),
    ] {
        let n = cases
            .iter()
            .filter(|c| c.outputs.iter().any(|(t, p, _)| t == ty && p == path))
            .count();
        assert!(n >= 19, "{ty} {path}: {n} cases");
    }
}

#[test]
fn the_references_are_bit_identical_to_ggml() {
    let (_, host, cases) = parse();
    let cpu = host_path(&host);
    let mut checked = 0;
    for case in &cases {
        let x = &case.input;
        for (ty, path, want) in &case.outputs {
            let q8 = if path == "ref" {
                Q8Path::Reference
            } else {
                cpu
            };
            let got = match ty.as_str() {
                "q8_0" => q8_0_bytes(x, q8),
                "q8_1" => q8_1_bytes(x, q8),
                "q8_K" => q8_k_bytes(x),
                "f16" => x.iter().flat_map(|&v| f16_bits(v).to_le_bytes()).collect(),
                "bf16" => x
                    .iter()
                    .flat_map(|&v| round_bf16(v).to_le_bytes())
                    .collect(),
                other => panic!("unknown type {other}"),
            };
            if let Some(i) = first_difference(&got, want) {
                panic!(
                    "case {} {ty} {path}: first difference at byte {i} of {}",
                    case.name,
                    want.len()
                );
            }
            checked += 1;
        }
    }
    assert!(checked > 150, "{checked} outputs checked");
}

/// ggml's portable and NEON quantizers disagree on rounding ties, so the vectors can tell the
/// two paths apart: each path matches its own lines and not the other's on the tie cases.
#[test]
fn the_tie_cases_tell_the_paths_apart() {
    let (_, host, cases) = parse();
    let cpu = host_path(&host);
    if cpu == Q8Path::Reference {
        return;
    }
    let ties = cases
        .iter()
        .find(|c| c.name == "ties_scale_0")
        .expect("the tie case");
    let x = &ties.input;
    let recorded = |path: &str| {
        ties.outputs
            .iter()
            .find(|(t, p, _)| t == "q8_0" && p == path)
            .map(|(_, _, b)| b.clone())
            .expect("recorded")
    };
    assert_ne!(recorded("ref"), recorded("cpu"));
    assert_ne!(q8_0_bytes(x, cpu), recorded("ref"));
    assert_ne!(q8_0_bytes(x, Q8Path::Reference), recorded("cpu"));
}

/// On the tie cases every block's maximum is `127 · 2^k`, so `127/amax` and `1/d` are the same
/// power of two and ggml's AVX and NEON quantizers do the same arithmetic: both round ties to
/// even. The AVX path must therefore give the recorded NEON bytes there. No x86 vectors are
/// committed yet (`data/act_quant/README.md`), so this is the AVX path's only check against
/// ggml's output.
#[test]
fn the_avx_path_gives_the_neon_bytes_on_the_tie_cases() {
    let (_, host, cases) = parse();
    if host_path(&host) != Q8Path::Arm64Neon {
        return;
    }
    let mut checked = 0;
    for name in ["ties_scale_0", "ties_scale_1", "ties_scale_2"] {
        let case = cases
            .iter()
            .find(|c| c.name == name)
            .unwrap_or_else(|| panic!("the {name} case"));
        let x = &case.input;
        for block in x.chunks(32) {
            let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            assert_eq!(127.0 / amax, 1.0 / (amax / 127.0), "{name}: inexact scale");
        }
        for (ty, path, want) in &case.outputs {
            let got = match (ty.as_str(), path.as_str()) {
                ("q8_0", "cpu") => q8_0_bytes(x, Q8Path::X86Avx),
                ("q8_1", "cpu") => q8_1_bytes(x, Q8Path::X86Avx),
                _ => continue,
            };
            if let Some(i) = first_difference(&got, want) {
                panic!(
                    "case {name} {ty} on the AVX path: first difference at byte {i} of {}",
                    want.len()
                );
            }
            checked += 1;
        }
    }
    assert_eq!(checked, 6);
}
