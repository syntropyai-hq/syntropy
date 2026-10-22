// SPDX-License-Identifier: MIT OR Apache-2.0

//! `SimEntropy`'s stream is fixed by its seed: the same bytes in every process and on every
//! platform (QUAL-E3-T02; RFC-0060 §2.1 rule T2, §8.2).
//!
//! The golden vectors are the ChaCha20 keystream for each seed's key (the seed's eight
//! little-endian bytes, then 24 zero bytes), with a zero nonce and the block counter from zero.
//! They were computed outside Rust with two independent implementations, which agreed:
//!
//! ```sh
//! key=<64 hex digits>
//! head -c 320 /dev/zero | openssl enc -chacha20 -K "$key" -iv 00000000000000000000000000000000 -nosalt | xxd -p -c 64
//! python3 -c 'import sys; from cryptography.hazmat.primitives.ciphers import Cipher, algorithms; print(Cipher(algorithms.ChaCha20(bytes.fromhex(sys.argv[1]), bytes(16)), mode=None).encryptor().update(bytes(320)).hex())' "$key"
//! ```
//!
//! OpenSSL's 16-byte IV is the 32-bit block counter followed by the nonce, so an all-zero IV is
//! this stream's counter and nonce for the first 2^32 blocks. Three published vectors, from
//! RFC 8439 Appendix A.1, are checked as well.

use std::process::Command;

use syntropy_core::Entropy;
use syntropy_testkit::sim::SimEntropy;

/// One seed's first 320 bytes: five ChaCha20 blocks, so the vector crosses the generator's
/// 256-byte buffer as well as block boundaries.
struct Golden {
    seed: u64,
    /// The stream in hex, 32 bytes a line.
    stream: &'static [&'static str],
}

const GOLDEN: &[Golden] = &[
    Golden {
        seed: 0,
        stream: &[
            "76b8e0ada0f13d90405d6ae55386bd28bdd219b8a08ded1aa836efcc8b770dc7",
            "da41597c5157488d7724e03fb8d84a376a43b8f41518a11cc387b669b2ee6586",
            "9f07e7be5551387a98ba977c732d080dcb0f29a048e3656912c6533e32ee7aed",
            "29b721769ce64e43d57133b074d839d531ed1f28510afb45ace10a1f4b794d6f",
            "2d09a0e663266ce1ae7ed1081968a0758e718e997bd362c6b0c34634a9a0b35d",
            "012737681f7b5d0f281e3afde458bc1e73d2d313c9cf94c05ff3716240a248f2",
            "1320a058d7b3566bd520daaa3ed2bf0ac5b8b120fb852773c3639734b45c91a4",
            "2dd4cb83f8840d2eedb158131062ac3f1f2cf8ff6dcd1856e86a1e6c3167167e",
            "e5a688742b47c5adfb59d4df76fd1db1e51ee03b1ca9f82aca173edb8b729347",
            "4ebe980f904d10c916442b4783a0e984860cb6c957b39c38ed8f51cffaa68a4d",
        ],
    },
    Golden {
        seed: 1,
        stream: &[
            "c5d30a7ce1ec119378c84f487d775a8542f13ece238a9455e8229e888de85bbd",
            "29eb63d0a17a5b999b52da22be4023eb07620a54f6fa6ad8737b71eb0464dac0",
            "10f656e6d1fd55053e50c4875c9930a33f6d0263bd14dfd6ab8c70521c19338b",
            "2308b95cf8d0bb7d202d2102780ea3528f1cb48560f76b20f382b942500fceac",
            "26c02c80147c34a4282b3dee6a1886ad8f4364c1fca82134d318224ecc6c45ed",
            "c86654e8bb4d40bc7bfbf0fbd9d91f943e6225a8f8968aa7242402170de92c28",
            "a3e2edc43e1057b25345acd3350a0029dd74f39cd4f2f8a9317b4aef38af5897",
            "6e3a4197b16c89feec107a8887b26f1629a50bd7658894fee026b65c5da3a3b0",
            "2cb334487f6949e8d2edfbc555c0324fd2619ab82d16c4f8a08dd39815da5f16",
            "e468e5adfc24b24aef0fc5e72ffef3b6bc87f5d24d59197846874c66aed8b07e",
        ],
    },
    Golden {
        seed: 0x0123_4567_89ab_cdef,
        stream: &[
            "81ff174f0ce9b04ffb10a32b7749b6fcc78840ad67a0d5f816075871af4fc883",
            "c0dd9c13a8da15d23264aca12b5881d3a574feab858c439d7dd549a01cee528f",
            "ee3305ac945e474a1b0143d6658c131e8440ac6d876e43a741fd25d87d67f0fb",
            "f6672c18c5464fa0980cced07410e9c54fbc529a19ad8e5fd6569f6393b5440e",
            "8c9146ca3b31fa041a4d91e165db6ff73f0a2cbbe54aa5129463d430e53c9862",
            "e9502824a5629e698c133c5f9870278554562582b44f0626663cf45cb4ac8c04",
            "12a2c3353861e205cfa380ae3ea21d9f1c78968b6be83aa2f697a29050da0ed6",
            "c685c5741b9fe9ea2b4d29be35da771defe27eb671c42c1baff35be8b643ac1f",
            "bb8ac5eb2c22a58743e86c952ebba27e510353fa29917da2a71e0ee9aed3b0e1",
            "11f8a33f6518daea11e35d728428b4cfdc1905c3cebe414a8cea92e386f7b2df",
        ],
    },
    Golden {
        seed: u64::MAX,
        stream: &[
            "3fa2ee6bda5341eb24428afc2ae53638099223f33bb44e43f3d5fe9c4a4d4016",
            "fb7d07d2f95f343aae5156967dd81bd435fd58263e8fa43c345cac05a82e658c",
            "a6ce0ebba9b5fa8be2a01bd56bf6023ae7549e74e3114c9705ba34de41ee306a",
            "f275ea7138fe2506c8a297a4f0b3e10e12d430351311d22e79b216683b6d2973",
            "84064a2ad4a3779ed595724020feb9750ec595467aa4cff468c0ab0d531e1a49",
            "e0aa305d1eaa6ed33ff1e93453eb346e87398c5566a726fc88e5afcc7588caa9",
            "51f5d7f04a7cd5e2309607680cd9d62c963ceead1aa9d545dfd694a984dd27c8",
            "8bb9ea30052c1a739ba820ad2d3ba25bd19172a65ad8d1f9c85f3bad52c956e6",
            "63ca115fb7e7e4d5f3633d2c475226f6d5bef76b7b416fd699c273685b3d0920",
            "0014eeb2f50291fcedb76308e620be91c02885841ef6ad35a1823ad9e1f14c5e",
        ],
    },
];

/// Seeds for the cross-process check: the golden ones and a few more.
const SEEDS: &[u64] = &[
    0,
    1,
    42,
    0x0123_4567_89ab_cdef,
    0x8000_0000_0000_0000,
    u64::MAX,
];

/// Bytes of each seed's stream the cross-process check compares.
const CROSS_PROCESS_LEN: usize = 4096;

/// The environment variable that makes [`child_prints_its_streams`] print.
const CHILD: &str = "SYNTROPY_TESTKIT_SIM_ENTROPY_CHILD";

/// The marker before each stream the child prints.
const PREFIX: &str = "sim-entropy-stream ";

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn unhex(text: &str) -> Vec<u8> {
    assert_eq!(text.len() % 2, 0, "odd hex length");
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).expect("hex digit"))
        .collect()
}

fn read(seed: u64, len: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; len];
    SimEntropy::new(seed).fill(&mut bytes);
    bytes
}

#[test]
fn every_seed_gives_its_golden_stream() {
    for golden in GOLDEN {
        let expected = unhex(&golden.stream.concat());
        assert_eq!(expected.len(), 320);
        assert_eq!(
            hex(&read(golden.seed, expected.len())),
            hex(&expected),
            "seed {:#x}",
            golden.seed
        );
    }
}

#[test]
fn the_rfc_8439_appendix_a1_vectors_hold() {
    // Test vectors #1 and #2: the all-zero key and nonce, block counter 0 and then 1, which is
    // seed 0.
    let vector_1 = "76b8e0ada0f13d90405d6ae55386bd28bdd219b8a08ded1aa836efcc8b770dc7\
                    da41597c5157488d7724e03fb8d84a376a43b8f41518a11cc387b669b2ee6586";
    let vector_2 = "9f07e7be5551387a98ba977c732d080dcb0f29a048e3656912c6533e32ee7aed\
                    29b721769ce64e43d57133b074d839d531ed1f28510afb45ace10a1f4b794d6f";
    let stream = read(0, 128);
    assert_eq!(hex(&stream[..64]), vector_1);
    assert_eq!(hex(&stream[64..]), vector_2);

    // Test vector #4: key bytes `00 ff 00 ... 00` and block counter 2. That key is seed 0xff00
    // (little-endian), and block 2 is bytes 128 to 191 of its stream.
    let vector_4 = "72d54dfbf12ec44b362692df94137f328fea8da73990265ec1bbbea1ae9af0ca\
                    13b25aa26cb4a648cb9b9d1be65b2c0924a66c54d545ec1b7374f4872e99f096";
    assert_eq!(hex(&read(0xff00, 192)[128..]), vector_4);
}

/// Runs in a child process only: prints each seed's stream for
/// [`the_same_seed_gives_the_same_stream_in_other_processes`].
#[test]
fn child_prints_its_streams() {
    if std::env::var_os(CHILD).is_none() {
        return;
    }
    for &seed in SEEDS {
        println!("{PREFIX}{seed} {}", hex(&read(seed, CROSS_PROCESS_LEN)));
    }
}

/// The streams a fresh process of this test binary prints, by seed.
fn streams_from_a_child_process() -> Vec<(u64, String)> {
    let output = Command::new(std::env::current_exe().expect("the test binary's path"))
        .args([
            "child_prints_its_streams",
            "--exact",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD, "1")
        .output()
        .expect("the child process runs");
    let stdout = String::from_utf8(output.stdout).expect("UTF-8 output");
    assert!(output.status.success(), "child failed: {stdout}");
    // libtest prints `test <name> ... ` before the test runs, so the first stream shares its
    // line: look for the prefix anywhere in a line.
    stdout
        .lines()
        .filter_map(|line| line.split_once(PREFIX).map(|(_, rest)| rest))
        .map(|line| {
            let (seed, stream) = line.split_once(' ').expect("a seed and a stream");
            (seed.parse().expect("a seed"), stream.to_owned())
        })
        .collect()
}

#[test]
fn the_same_seed_gives_the_same_stream_in_other_processes() {
    let here: Vec<(u64, String)> = SEEDS
        .iter()
        .map(|&seed| (seed, hex(&read(seed, CROSS_PROCESS_LEN))))
        .collect();
    let first = streams_from_a_child_process();
    let second = streams_from_a_child_process();
    assert_eq!(first.len(), SEEDS.len(), "the child printed every seed");
    assert_eq!(first, here, "a child process gave other bytes");
    assert_eq!(second, here, "a second child process gave other bytes");
}
