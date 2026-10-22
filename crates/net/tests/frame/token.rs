// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `Token` codec against the golden frames in `tests/golden/token/` (DIST-E8-T03; RFC-0031
//! §6.5).
//!
//! Each golden frame is written twice: as wire values, which [`spec`] lays out from RFC-0031
//! §6.5's tables, and as the codec's typed values. The tests check that the files match the
//! tables, that each decodes to its typed values and encodes to the same bytes, that the tail's
//! whole step encodes to the ring-return golden without its taps and to the ingress goldens
//! without its drafts, and that a frame breaking a rule is refused with a classified error,
//! never `Internal` (RFC-0060 §10.2).
//!
//! `SYNTROPY_BLESS_GOLDEN=1` rewrites the files, as it does the other goldens.

use std::collections::BTreeSet;
use std::path::PathBuf;

use proptest::prelude::*;
use syntropy_core::{ApproxFlags, InstanceId, SequenceId};
use syntropy_net::frame::{
    ENTRY_HEADER_LEN, Entries, Finish, Flags, Frame, FrameError, FrameType, Route, Side, SideItem,
    SideItems, StepTokens, Token, TokenCopy, TokenEntry, TokenSpec, TraceContext, TreeArea,
    TreeNode, WIRE_MINOR, tlv, token_flags,
};

use super::{INSTANCE, REFUSAL_CLASSES, base, epoch, golden_dir, leak, reseal, spec};

/// Two ingress nodes' short IDs.
const X: u64 = 0x0102_0304_0506_0708;
const Y: u64 = 0x1111_2222_3333_4444;

fn seq(ingress: u64, n: u32) -> SequenceId {
    SequenceId {
        ingress,
        boot: 2,
        n,
    }
}

fn le32s(values: &[u32]) -> &'static [u8] {
    leak(values.iter().flat_map(|v| v.to_le_bytes()).collect())
}

fn spec_for(step: u64, chan_seq: u64) -> TokenSpec {
    TokenSpec {
        wire_minor: WIRE_MINOR,
        flags: Flags::default(),
        instance: InstanceId(INSTANCE),
        epoch: epoch(),
        step,
        chan_seq,
    }
}

fn plain(
    seq: SequenceId,
    first_position: u32,
    tokens: &[u32],
    rng_counter: u64,
) -> TokenEntry<'static> {
    TokenEntry {
        seq,
        first_position,
        tokens: le32s(tokens),
        finish: None,
        logprob_k: 0,
        logprobs: &[],
        approx: ApproxFlags::empty(),
        draft: &[],
        draft_tree: None,
        tap: &[],
        rng_counter,
    }
}

fn tree(nodes: &[(Option<u16>, u8, u8)]) -> TreeArea<'static> {
    let nodes: Vec<TreeNode> = nodes
        .iter()
        .map(|&(parent, depth, flags)| TreeNode {
            parent,
            depth,
            flags,
        })
        .collect();
    TreeArea::new(Vec::leak(nodes))
}

/// Per token: the chosen logprob, then `k` alternatives `(id, logprob)`.
fn logprobs(per_token: &[(f32, &[(u32, f32)])]) -> &'static [u8] {
    let mut out = Vec::new();
    for (chosen, alternatives) in per_token {
        out.extend(chosen.to_le_bytes());
        for (id, logprob) in *alternatives {
            out.extend(id.to_le_bytes());
            out.extend(logprob.to_le_bytes());
        }
    }
    leak(out)
}

/// Step 21's entries as the tail holds them: a plain decode for X, a verify result for X with
/// logprobs, a chain draft and a tap block, and an error for Y. Two `rng_counter`s are above
/// 2³².
fn step21() -> Vec<TokenEntry<'static>> {
    vec![
        plain(seq(X, 77), 1600, &[1001], 7),
        TokenEntry {
            logprob_k: 2,
            logprobs: logprobs(&[
                (-0.25, &[(6, -1.5), (9, -2.0)]),
                (-0.5, &[(5, -1.25), (8, -3.0)]),
                (-0.125, &[(1, -2.5), (2, -4.0)]),
            ]),
            approx: ApproxFlags::SPEC_LENIENT,
            draft: le32s(&[8, 9, 10]),
            tap: leak(spec::fill(0x90, 40)),
            ..plain(seq(X, 78), 41, &[5, 6, 7], 0x1_0000_0005)
        },
        TokenEntry {
            finish: Some(Finish::Error {
                domain: 3,
                code: 0x030C,
            }),
            ..plain(seq(Y, 3), 7, &[], 0x2_0000_0000)
        },
    ]
}

/// A sampled step's trace context for entry `n`.
fn trace(n: u8) -> SideItem<'static> {
    SideItem::Trace(TraceContext {
        version: 0,
        trace_id: spec::fill(0x70 + n, 16).try_into().unwrap(),
        span_id: spec::fill(0x80 + n, 8).try_into().unwrap(),
        flags: 1,
    })
}

/// Step 21's items: `KV_HEADROOM` for two stages and a `RESTART` for entry 0, both carried as
/// bytes until DIST-E8-T04 types them, and `TRACE` for the sampled entries 1 and 2.
fn step21_side() -> Vec<Side<'static>> {
    let mut headroom = vec![2u8, 0, 0, 0, 0, 0, 0, 0];
    for stage in 0..2u16 {
        headroom.extend(stage.to_le_bytes());
        headroom.extend(0u16.to_le_bytes());
        for tokens in [4096u32, 65536, 1 << 20] {
            headroom.extend((tokens >> stage).to_le_bytes());
        }
    }
    vec![
        Side {
            span: None,
            item: SideItem::Other {
                ty: tlv::KV_HEADROOM,
                value: leak(headroom),
            },
        },
        Side {
            span: Some(0),
            item: SideItem::Other {
                ty: tlv::RESTART,
                value: leak([1536u32.to_le_bytes(), [0; 4]].concat()),
            },
        },
        Side {
            span: Some(1),
            item: trace(1),
        },
        Side {
            span: Some(2),
            item: trace(2),
        },
    ]
}

/// Step 22's ring-return entries: a 4-node tree draft with a prompt-lookup node, a chain draft
/// written as a tree, and an entry without a draft.
fn step22() -> Vec<TokenEntry<'static>> {
    vec![
        TokenEntry {
            draft: le32s(&[11, 12, 13, 14]),
            draft_tree: Some(tree(&[
                (None, 1, 0),
                (Some(0), 2, 0),
                (Some(0), 2, TreeNode::PROMPT_LOOKUP),
                (Some(1), 3, 0),
            ])),
            ..plain(seq(X, 77), 1601, &[1002], 8)
        },
        TokenEntry {
            draft: le32s(&[16, 17]),
            draft_tree: Some(tree(&[(None, 1, 0), (Some(0), 2, 0)])),
            ..plain(seq(X, 78), 44, &[15], 0x1_0000_0006)
        },
        plain(seq(Y, 4), 8, &[20], 1),
    ]
}

/// The entry `copy` carries: no tap on the ring-return copy, no draft on an ingress copy.
fn as_carried<'a>(entry: &TokenEntry<'a>, route: Route) -> TokenEntry<'a> {
    match route {
        Route::RingReturn => TokenEntry { tap: &[], ..*entry },
        Route::ToIngress => TokenEntry {
            draft: &[],
            draft_tree: None,
            ..*entry
        },
    }
}

/// The RFC-0031 §6.5 wire values of `entry`, as `route` carries it.
fn entry_fields(entry: &TokenEntry<'_>, route: Route) -> spec::EntryFields {
    let entry = as_carried(entry, route);
    let (finish, error_domain, error_code) = match entry.finish {
        None => (0, 0, 0),
        Some(Finish::Stop) => (1, 0, 0),
        Some(Finish::Length) => (2, 0, 0),
        Some(Finish::TokenStop) => (3, 0, 0),
        Some(Finish::Cancelled) => (4, 0, 0),
        Some(Finish::Error { domain, code }) => (5, domain, code),
    };
    let words = |bytes: &[u8]| -> Vec<u32> {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes(*w))
            .collect()
    };
    let draft = words(entry.draft);
    spec::EntryFields {
        ingress: entry.seq.ingress,
        boot: entry.seq.boot,
        n: entry.seq.n,
        first_position: entry.first_position,
        tokens: words(entry.tokens),
        finish,
        logprob_k: entry.logprob_k,
        approx: entry.approx.bits(),
        error_domain,
        error_code,
        draft_nodes: draft.len() as u16,
        draft,
        tree: entry
            .draft_tree
            .iter()
            .flat_map(|t| t.iter())
            .map(|n| (n.parent.unwrap_or(0xFFFF), n.depth, n.flags))
            .collect(),
        logprobs: entry.logprobs.to_vec(),
        tap: entry.tap.to_vec(),
        rng_counter: entry.rng_counter,
    }
}

/// A golden frame: its bytes, and the `Token` they decode to.
struct Golden {
    name: &'static str,
    about: &'static str,
    bytes: Vec<u8>,
    spec: TokenSpec,
    route: Route,
    part: (u16, u16),
    side: Vec<Side<'static>>,
    entries: Vec<TokenEntry<'static>>,
}

impl Golden {
    fn token(&self) -> Token<'_> {
        Token {
            route: self.route,
            part_index: self.part.0,
            part_count: self.part.1,
            side: SideItems::new(&self.side),
            entries: Entries::new(&self.entries),
        }
    }
}

/// The golden's bytes from wire values: RFC-0031 §6.5's fields for `entries` as `route`
/// carries them, laid out by [`spec`].
#[allow(clippy::too_many_arguments)]
fn golden(
    name: &'static str,
    about: &'static str,
    spec: TokenSpec,
    route: Route,
    part: (u16, u16),
    side: Vec<Side<'static>>,
    entries: Vec<TokenEntry<'static>>,
) -> Golden {
    let entries: Vec<TokenEntry<'static>> = entries.iter().map(|e| as_carried(e, route)).collect();
    let mut flags = match route {
        Route::RingReturn => 1,
        Route::ToIngress => 2,
    };
    if entries.iter().any(|e| e.draft_tree.is_some()) {
        flags |= 4; // DRAFT_TREE
    }
    let tlvs: Vec<(u16, u16, Vec<u8>)> = side
        .iter()
        .map(|s| {
            let span = s.span.unwrap_or(0xFFFF);
            match s.item {
                SideItem::Other { ty, value } => (ty, span, value.to_vec()),
                // §6.3: version, trace-id, span-id, flags.
                SideItem::Trace(t) => {
                    let mut v = vec![t.version];
                    v.extend(t.trace_id);
                    v.extend(t.span_id);
                    v.push(t.flags);
                    (5, span, v)
                }
                other => panic!("not a Token item: {other:?}"),
            }
        })
        .collect();
    let has_trace = side.iter().any(|s| matches!(s.item, SideItem::Trace(_)));
    let ext = spec::token_ext(entries.len() as u16, flags, part.0, part.1, &tlvs);
    let payload: Vec<u8> = entries
        .iter()
        .flat_map(|e| spec::token_entry(&entry_fields(e, route)))
        .collect();
    let bytes = spec::frame(
        &base(0x02, u16::from(has_trace), spec.step, spec.chan_seq),
        &ext,
        &payload,
    );
    Golden {
        name,
        about,
        bytes,
        spec,
        route,
        part,
        side,
        entries,
    }
}

fn goldens() -> Vec<Golden> {
    let ingress = |node: u64| -> Vec<TokenEntry<'static>> {
        step21()
            .into_iter()
            .filter(|e| e.seq.ingress == node)
            .collect()
    };
    vec![
        golden(
            "token-ring-return",
            "step 21's ring-return copy: 3 entries, the verify result with logprobs (k = 2) and a \
             chain draft and no tap, an Error finish, rng_counters above 2^32, KV_HEADROOM, \
             RESTART, and TRACE for entries 1 and 2 with HAS_TRACE",
            spec_for(21, 1),
            Route::RingReturn,
            (0, 1),
            step21_side(),
            step21(),
        ),
        golden(
            "token-ingress-x",
            "step 21's copy to ingress X: its 2 entries, the tap block and no draft, and entry 1's \
             TRACE",
            spec_for(21, 1),
            Route::ToIngress,
            (0, 1),
            vec![Side {
                span: Some(1),
                item: trace(1),
            }],
            ingress(X),
        ),
        golden(
            "token-ingress-y",
            "step 21's copy to ingress Y: the Error entry alone, its TRACE moved to span 0",
            spec_for(21, 1),
            Route::ToIngress,
            (0, 1),
            vec![Side {
                span: Some(0),
                item: trace(2),
            }],
            ingress(Y),
        ),
        golden(
            "token-ring-return-tree",
            "step 22's ring-return copy with DRAFT_TREE: a 4-node tree draft with a prompt-lookup \
             node, a chain draft written as a tree, and an entry without a draft",
            spec_for(22, 2),
            Route::RingReturn,
            (0, 1),
            vec![],
            step22(),
        ),
        golden(
            "token-ingress-part1",
            "part 1 of 2 of step 23's copy to ingress Y: a prefill chunk's tap block, no tokens",
            spec_for(23, 4),
            Route::ToIngress,
            (1, 2),
            vec![],
            vec![TokenEntry {
                tap: leak(spec::fill(0xA0, 48)),
                ..plain(seq(Y, 5), 0, &[], 0)
            }],
        ),
        golden(
            "token-empty-ring-return",
            "an empty ring-return frame answering an idle relay: no entries, one SESSION_SAVED",
            spec_for(24, 3),
            Route::RingReturn,
            (0, 1),
            vec![Side {
                span: None,
                item: SideItem::Other {
                    ty: tlv::SESSION_SAVED,
                    value: leak(
                        [
                            spec::fill(0xC0, 16),
                            vec![1, 0, 0, 0, 0, 0, 0, 0],
                            spec::fill(0xD0, 32),
                        ]
                        .concat(),
                    ),
                },
            }],
            vec![],
        ),
    ]
}

fn token_dir() -> PathBuf {
    golden_dir().join("token")
}

fn read_golden(name: &str) -> Vec<u8> {
    let path = token_dir().join(format!("{name}.hex"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    spec::from_hex(&text)
}

#[test]
fn token_golden_files_match_the_spec_tables() {
    let goldens = goldens();
    if std::env::var_os("SYNTROPY_BLESS_GOLDEN").is_some() {
        std::fs::create_dir_all(token_dir()).unwrap();
        for golden in &goldens {
            let text = spec::to_hex(golden.name, golden.about, &golden.bytes);
            std::fs::write(token_dir().join(format!("{}.hex", golden.name)), text).unwrap();
        }
    }
    for golden in &goldens {
        assert_eq!(
            read_golden(golden.name),
            golden.bytes,
            "{} differs from the tables",
            golden.name
        );
    }
    let listed: BTreeSet<String> = goldens.iter().map(|g| format!("{}.hex", g.name)).collect();
    let on_disk: BTreeSet<String> = std::fs::read_dir(token_dir())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .filter(|name| name.ends_with(".hex"))
        .collect();
    assert_eq!(on_disk, listed);
}

/// The criterion "golden frames round-trip, including an `rng_counter` above 2³²".
#[test]
fn token_goldens_round_trip() {
    for golden in goldens() {
        let name = golden.name;
        let bytes = read_golden(name);
        let frame = Frame::decode(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(frame.header.spec.ftype, FrameType::Token, "{name}");
        let token = frame.token().unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(token, golden.token(), "{name}");
        assert_eq!(
            golden.token().encode_to_vec(&golden.spec),
            Ok(bytes.clone()),
            "{name}"
        );
        let mut slot = vec![0xAA; bytes.len() + 64];
        assert_eq!(
            golden.token().encode(&golden.spec, &mut slot),
            Ok(bytes.len()),
            "{name}"
        );
        assert_eq!(&slot[..bytes.len()], &bytes[..], "{name}");
    }
    // The ring-return golden's second and third entries carry rng_counters above 2³², at entry
    // offset 40 as a u64.
    let bytes = read_golden("token-ring-return");
    let frame = Frame::decode(&bytes).unwrap();
    let counters: Vec<u64> = frame
        .token()
        .unwrap()
        .entries
        .iter()
        .map(|e| e.rng_counter)
        .collect();
    assert_eq!(counters, [7, 0x1_0000_0005, 0x2_0000_0000]);
    assert!(counters[1] > u64::from(u32::MAX));
    // Entry 0 is its header and one token, 52 bytes padded to 56.
    let entry1 = (ENTRY_HEADER_LEN + 4).next_multiple_of(8);
    assert_eq!(
        &frame.payload[entry1 + 40..entry1 + 48],
        &0x1_0000_0005u64.to_le_bytes()
    );
}

/// The criterion "the ingress copy carries no draft, and the ring-return copy no tap (encoder
/// test)": the tail's whole step, every entry with its draft and tap, encodes to the
/// ring-return golden and to each ingress golden.
#[test]
fn each_copy_leaves_out_what_it_does_not_carry() {
    let entries = step21();
    let side = step21_side();
    let step = StepTokens {
        entries: Entries::new(&entries),
        side: SideItems::new(&side),
        part_index: 0,
        part_count: 1,
    };
    assert!(
        entries
            .iter()
            .any(|e| !e.draft.is_empty() && !e.tap.is_empty())
    );
    let spec = spec_for(21, 1);
    let ring = step.encode_to_vec(TokenCopy::RingReturn, &spec).unwrap();
    assert_eq!(ring, read_golden("token-ring-return"));
    let frame = Frame::decode(&ring).unwrap();
    let token = frame.token().unwrap();
    assert_eq!(token.route, Route::RingReturn);
    assert_eq!(token.entries.len(), 3);
    assert!(token.entries.iter().all(|e| e.tap.is_empty()));
    assert_eq!(
        token.entries.iter().nth(1).unwrap().draft,
        le32s(&[8, 9, 10])
    );
    assert_eq!(token.side.iter().count(), 4);
    assert!(frame.header.spec.flags.has_trace());
    // Each ingress copy keeps its own entries' TRACE, with the span counted among them, and no
    // other item.
    for (node, name, count, traced) in [
        (X, "token-ingress-x", 2, (1, trace(1))),
        (Y, "token-ingress-y", 1, (0, trace(2))),
    ] {
        let bytes = step
            .encode_to_vec(TokenCopy::Ingress { ingress: node }, &spec)
            .unwrap();
        assert_eq!(bytes, read_golden(name), "{name}");
        let frame = Frame::decode(&bytes).unwrap();
        let token = frame.token().unwrap();
        assert_eq!(token.route, Route::ToIngress);
        assert_eq!(token.entries.len(), count);
        assert!(token.entries.iter().all(|e| e.seq.ingress == node));
        assert!(
            token
                .entries
                .iter()
                .all(|e| e.draft.is_empty() && e.draft_tree.is_none())
        );
        let side: Vec<Side<'_>> = token.side.iter().collect();
        assert_eq!(
            side,
            [Side {
                span: Some(traced.0),
                item: traced.1
            }],
            "{name}"
        );
        assert!(frame.header.spec.flags.has_trace(), "{name}");
    }
    let bytes = read_golden("token-ingress-x");
    let frame = Frame::decode(&bytes).unwrap();
    let tap: Vec<&[u8]> = frame
        .token()
        .unwrap()
        .entries
        .iter()
        .map(|e| e.tap)
        .collect();
    assert_eq!(tap, [&[][..], &spec::fill(0x90, 40)[..]]);
}

fn field(field: &'static str, value: u64) -> FrameError {
    FrameError::BadField {
        ftype: FrameType::Token,
        field,
        value,
    }
}

fn inconsistent(field: &'static str, value: u64, expected: u64) -> FrameError {
    FrameError::Inconsistent {
        ftype: FrameType::Token,
        field,
        value,
        expected,
    }
}

/// A `Token` frame as wire values, to break one rule at a time.
struct Raw {
    ftype: u8,
    base_flags: u16,
    flags: u16,
    part: (u16, u16),
    tlvs: Vec<(u16, u16, Vec<u8>)>,
    entries: Vec<spec::EntryFields>,
    /// Bytes after the last entry.
    trailer: Vec<u8>,
}

impl Raw {
    fn ext(&self) -> Vec<u8> {
        spec::token_ext(
            self.entries.len() as u16,
            self.flags,
            self.part.0,
            self.part.1,
            &self.tlvs,
        )
    }

    fn frame(&self, ext: &[u8]) -> Vec<u8> {
        let mut payload: Vec<u8> = self.entries.iter().flat_map(spec::token_entry).collect();
        payload.extend(&self.trailer);
        spec::frame(&base(self.ftype, self.base_flags, 1, 1), ext, &payload)
    }

    fn bytes(&self) -> Vec<u8> {
        self.frame(&self.ext())
    }
}

/// Ring-return, two entries: a plain decode for X and a verify result for X with a chain draft.
fn raw_ring() -> Raw {
    let entries = step21();
    Raw {
        ftype: 0x02,
        base_flags: 0,
        flags: token_flags::RING_RETURN,
        part: (0, 1),
        tlvs: vec![],
        entries: entries[..2]
            .iter()
            .map(|e| entry_fields(e, Route::RingReturn))
            .collect(),
        trailer: vec![],
    }
}

/// Ring-return with `DRAFT_TREE`: one entry with a 3-node tree draft.
fn raw_tree() -> Raw {
    let mut raw = raw_ring();
    raw.flags |= token_flags::DRAFT_TREE;
    raw.entries.truncate(1);
    raw.entries[0].draft = vec![11, 12, 13];
    raw.entries[0].draft_nodes = 3;
    raw.entries[0].tree = vec![(0xFFFF, 1, 0), (0, 2, 0), (1, 3, 0)];
    raw
}

/// The ingress copy for X: both its entries, the second with a tap block.
fn raw_ingress() -> Raw {
    let entries = step21();
    Raw {
        flags: token_flags::TO_INGRESS,
        entries: entries[..2]
            .iter()
            .map(|e| entry_fields(e, Route::ToIngress))
            .collect(),
        ..raw_ring()
    }
}

/// Ring-return, as [`raw_ring`], with one 8-byte `RESTART` item for entry 0.
fn raw_with_tlv() -> Raw {
    let mut raw = raw_ring();
    raw.tlvs = vec![(tlv::RESTART, 0, vec![0; 8])];
    raw
}

fn case(
    name: &'static str,
    mut raw: Raw,
    edit: fn(&mut Raw),
    error: FrameError,
) -> (&'static str, Vec<u8>, FrameError) {
    edit(&mut raw);
    (name, raw.bytes(), error)
}

/// Every rule the codec checks, broken one at a time, is refused with the error that names it,
/// in a class RFC-0060 §10.2 allows.
#[test]
fn a_token_frame_breaking_a_rule_is_refused_with_a_classified_error() {
    let (ring, tree, ingress) = (raw_ring, raw_tree, raw_ingress);
    let mut frames = vec![
        case(
            "a tap on the ring-return copy",
            ring(),
            |r| r.entries[1].tap = vec![1; 8],
            field("tap_len", 8),
        ),
        case(
            "a draft on an ingress copy",
            ingress(),
            |r| {
                r.entries[1].draft = vec![8, 9];
                r.entries[1].draft_nodes = 2;
            },
            field("n_draft", 2),
        ),
        case(
            "an item other than TRACE on an ingress copy",
            ingress(),
            |r| {
                r.tlvs = vec![(tlv::KV_HEADROOM, 0xFFFF, vec![0; 8])];
            },
            field("ingress TLV type", 8),
        ),
        case(
            "both copies at once",
            ring(),
            |r| r.flags = 3,
            field("flags", 3),
        ),
        case("neither copy", ring(), |r| r.flags = 0, field("flags", 0)),
        case(
            "finish 6",
            ring(),
            |r| r.entries[0].finish = 6,
            field("finish", 6),
        ),
        case(
            "draft_nodes not n_draft",
            ring(),
            |r| r.entries[1].draft_nodes = 2,
            inconsistent("draft_nodes", 2, 3),
        ),
        case(
            "a node that is its own parent",
            tree(),
            |r| r.entries[0].tree[1].0 = 1,
            field("parent", 1),
        ),
        case(
            "a depth that is not its parent's plus one",
            tree(),
            |r| r.entries[0].tree[2].1 = 5,
            field("depth", 5),
        ),
        case(
            "a node under the last accepted token at depth 0",
            tree(),
            |r| r.entries[0].tree[0].1 = 0,
            field("depth", 0),
        ),
        case(
            "a parent after its child",
            tree(),
            |r| r.entries[0].tree[1].0 = 2,
            field("parent", 2),
        ),
        case(
            "a TRUNCATE item",
            ring(),
            |r| r.tlvs = vec![(tlv::TRUNCATE, 0, vec![0; 4])],
            field("TLV type", 6),
        ),
        case(
            "a TOKEN_IDS item",
            ring(),
            |r| r.tlvs = vec![(tlv::TOKEN_IDS, 0xFFFF, vec![])],
            field("TLV type", 1),
        ),
        case(
            "an item naming an entry the frame lacks",
            ring(),
            |r| {
                r.tlvs = vec![(tlv::RESTART, 2, vec![0; 8])];
            },
            field("TLV span", 2),
        ),
        case(
            "part 1 of 1",
            ring(),
            |r| r.part = (1, 1),
            field("part_index", 1),
        ),
        case(
            "no parts",
            ring(),
            |r| r.part = (0, 0),
            field("part_count", 0),
        ),
        case(
            "bytes after the last entry",
            ring(),
            |r| r.trailer = vec![0; 8],
            inconsistent("payload_len", 200, 192),
        ),
        case(
            "HAS_TRACE without a TRACE",
            ring(),
            |r| r.base_flags = 1,
            inconsistent("HAS_TRACE", 1, 0),
        ),
        case(
            "a TRACE for no entry",
            ring(),
            |r| {
                r.base_flags = 1;
                r.tlvs = vec![(tlv::TRACE, 0xFFFF, vec![0; 26])];
            },
            field("TRACE span", 0xFFFF),
        ),
        case(
            "a second TRACE for one entry",
            ring(),
            |r| {
                r.base_flags = 1;
                r.tlvs = vec![(tlv::TRACE, 1, vec![0; 26]), (tlv::TRACE, 1, vec![1; 26])];
            },
            field("second TRACE for entry", 1),
        ),
        case(
            "a TRACE without HAS_TRACE",
            ring(),
            |r| r.tlvs = vec![(tlv::TRACE, 1, vec![0; 26])],
            inconsistent("HAS_TRACE", 0, 1),
        ),
        case(
            "an Activation frame",
            ring(),
            |r| r.ftype = 0x01,
            FrameError::WrongType {
                expected: FrameType::Token,
                found: FrameType::Activation,
            },
        ),
    ];
    let raw = raw_ring();
    let edit = |raw: &Raw, edit: fn(&mut Vec<u8>)| {
        let mut ext = raw.ext();
        edit(&mut ext);
        raw.frame(&ext)
    };
    frames.push((
        "a TLV running past the side area",
        // The only TLV claims 64 value bytes of its 16-byte area: the error names the Token frame.
        edit(&raw_with_tlv(), |e| e[16 + 4] = 64),
        inconsistent("side_len", 16, 72),
    ));
    frames.push((
        "a side area of 12 bytes",
        edit(&raw, |e| e[4] = 12),
        field("side_len", 12),
    ));
    frames.push((
        "a side area past the extension",
        edit(&raw, |e| e[4] = 64),
        FrameError::ExtTooShort {
            ftype: FrameType::Token,
            len: 64,
            needed: 80,
        },
    ));
    frames.push((
        "one entry more than the payload holds",
        edit(&raw, |e| e[0] = 3),
        inconsistent("payload_len", 192, 240),
    ));
    frames.push((
        "no extension",
        raw.frame(&[]),
        FrameError::ExtTooShort {
            ftype: FrameType::Token,
            len: 0,
            needed: 16,
        },
    ));
    for (name, bytes, expected) in frames {
        let frame = Frame::decode(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        let error = frame.token().expect_err(name);
        assert_eq!(error, expected, "{name}");
        assert!(
            REFUSAL_CLASSES.contains(&error.class()),
            "{name}: {error} is {:?}",
            error.class()
        );
    }
    for raw in [raw_ring(), raw_tree(), raw_ingress()] {
        assert!(Frame::decode(&raw.bytes()).unwrap().token().is_ok());
    }
}

/// Error fields outside an `Error` finish, reserved extension bytes and flag bits, `DRAFT_TREE`
/// on an ingress copy, bytes past the side area and knob bits this build does not name are all
/// accepted (RFC-0031 §6, §10); the unknown knob bits are kept.
#[test]
fn reserved_fields_and_later_additions_are_ignored() {
    let raw = raw_ring();
    let plain_bytes = raw.bytes();
    let plain_frame = Frame::decode(&plain_bytes).unwrap();
    let plain = plain_frame.token().unwrap();
    let mut edited = raw_ring();
    edited.entries[0].error_domain = 9;
    edited.entries[0].error_code = 9;
    let bytes = edited.bytes();
    assert_eq!(Frame::decode(&bytes).unwrap().token(), Ok(plain));
    type ExtEdit = fn(&mut Vec<u8>);
    let ext_edits: [(&str, ExtEdit); 3] = [
        ("reserved extension bytes", |e| e[12..16].fill(0xEE)),
        ("reserved flag bits", |e| e[2] |= 0xF8),
        ("a later minor's trailing extension fields", |e| {
            e.extend([0xEE; 64])
        }),
    ];
    for (name, edit) in ext_edits {
        let mut ext = raw.ext();
        edit(&mut ext);
        let bytes = raw.frame(&ext);
        let frame = Frame::decode(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(frame.token(), Ok(plain), "{name}");
    }
    let mut ingress = raw_ingress();
    ingress.flags |= token_flags::DRAFT_TREE;
    let bytes = ingress.bytes();
    let tree_flag = Frame::decode(&bytes).unwrap().token().unwrap();
    let bytes = raw_ingress().bytes();
    assert_eq!(Frame::decode(&bytes).unwrap().token(), Ok(tree_flag));
    let mut knobs = raw_ring();
    knobs.entries[0].approx = 0x8000_0001;
    let bytes = knobs.bytes();
    let frame = Frame::decode(&bytes).unwrap();
    let token = frame.token().unwrap();
    assert_eq!(
        token.entries.iter().next().unwrap().approx.bits(),
        0x8000_0001
    );
    assert_eq!(token.encode_to_vec(&spec_for(1, 1)), Ok(bytes));
}

/// A sender cannot write what a receiver would refuse.
#[test]
fn the_encoder_refuses_what_the_decoder_would() {
    let entries = step21();
    let mut wrong = entries.clone();
    let logprobs = wrong[1].logprobs;
    wrong[1].logprobs = &logprobs[..56];
    let step =
        |entries: &'static [TokenEntry<'static>], side: &'static [Side<'static>]| StepTokens {
            entries: Entries::new(entries),
            side: SideItems::new(side),
            part_index: 0,
            part_count: 1,
        };
    let spec = spec_for(21, 1);
    let wrong = Vec::leak(wrong);
    assert_eq!(
        step(wrong, &[]).encode_to_vec(TokenCopy::RingReturn, &spec),
        Err(inconsistent("logprobs len", 56, 60))
    );
    let mut wrong = entries.clone();
    wrong[1].draft_tree = Some(tree(&[(None, 1, 0)]));
    assert_eq!(
        step(Vec::leak(wrong), &[]).encode_to_vec(TokenCopy::RingReturn, &spec),
        Err(inconsistent("draft_nodes", 1, 3))
    );
    // A tree for one draft and a chain for another cannot share a frame.
    let mut mixed = step22();
    mixed[1].draft_tree = None;
    let mixed = Vec::leak(mixed);
    assert_eq!(
        step(mixed, &[]).encode_to_vec(TokenCopy::RingReturn, &spec),
        Err(field("draft_tree", 0))
    );
    // The ingress copy has no drafts, so it is written either way.
    assert!(
        step(mixed, &[])
            .encode_to_vec(TokenCopy::Ingress { ingress: X }, &spec)
            .is_ok()
    );
    let disguised = Vec::leak(vec![Side {
        span: None,
        item: SideItem::Other {
            ty: tlv::TRACE,
            value: &[0; 26],
        },
    }]);
    let entries = Vec::leak(entries);
    assert_eq!(
        step(entries, disguised).encode_to_vec(TokenCopy::RingReturn, &spec),
        Err(field("TLV type", 5))
    );
    let sealed = TokenSpec {
        flags: Flags(Flags::ENCRYPTED),
        ..spec
    };
    assert_eq!(
        step(entries, &[]).encode_to_vec(TokenCopy::RingReturn, &sealed),
        Err(field("flags", u64::from(Flags::ENCRYPTED)))
    );
    // `HAS_TRACE` follows the items the copy carries, whatever the spec says.
    let traced = TokenSpec {
        flags: Flags(Flags::HAS_TRACE),
        ..spec
    };
    let bytes = step(entries, &[])
        .encode_to_vec(TokenCopy::Ingress { ingress: X }, &traced)
        .unwrap();
    assert!(!Frame::decode(&bytes).unwrap().header.spec.flags.has_trace());
    // A step's `TRACE` names one entry, at most once: an unspanned one would reach every ingress
    // copy (review-net N1).
    let unspanned = Vec::leak(vec![Side {
        span: None,
        item: trace(1),
    }]);
    for copy in [
        TokenCopy::RingReturn,
        TokenCopy::Ingress { ingress: X },
        TokenCopy::Ingress { ingress: Y },
    ] {
        assert_eq!(
            step(entries, unspanned).encode_to_vec(copy, &spec),
            Err(field("TRACE span", 0xFFFF))
        );
    }
    let twice = Vec::leak(vec![
        Side {
            span: Some(1),
            item: trace(1),
        },
        Side {
            span: Some(1),
            item: trace(2),
        },
    ]);
    assert_eq!(
        step(entries, twice).encode_to_vec(TokenCopy::RingReturn, &spec),
        Err(field("second TRACE for entry", 1))
    );
    // A decoded frame re-encodes only as it came.
    let token = Token {
        route: Route::RingReturn,
        part_index: 0,
        part_count: 1,
        side: SideItems::new(&[]),
        entries: Entries::new(entries),
    };
    assert_eq!(token.encode_to_vec(&spec), Err(field("tap_len", 40)));
}

/// The criterion "a corrupted byte anywhere is detected", for the `Token` goldens.
#[test]
fn a_corrupted_token_byte_is_detected() {
    for golden in goldens() {
        let bytes = read_golden(golden.name);
        for at in 0..bytes.len() {
            for mask in [1, 2, 4, 8, 16, 32, 64, 128, 0xFF] {
                let mut corrupt = bytes.clone();
                corrupt[at] ^= mask;
                assert!(
                    Frame::decode(&corrupt).is_err(),
                    "{}: byte {at} ^ {mask:#04x} was accepted",
                    golden.name
                );
            }
        }
    }
}

/// An entry a generated step holds, with its own bytes.
#[derive(Clone, Debug)]
struct Owned {
    seq: SequenceId,
    tokens: Vec<u8>,
    finish: Option<Finish>,
    logprob_k: u8,
    logprobs: Vec<u8>,
    draft: Vec<u8>,
    tree: Vec<TreeNode>,
    tap: Vec<u8>,
    rng_counter: u64,
}

impl Owned {
    fn entry(&self) -> TokenEntry<'_> {
        TokenEntry {
            seq: self.seq,
            first_position: self.seq.n * 10,
            tokens: &self.tokens,
            finish: self.finish,
            logprob_k: self.logprob_k,
            logprobs: &self.logprobs,
            approx: ApproxFlags::from_bits_retain(self.seq.n),
            draft: &self.draft,
            draft_tree: (!self.tree.is_empty()).then(|| TreeArea::new(&self.tree)),
            tap: &self.tap,
            rng_counter: self.rng_counter,
        }
    }
}

/// A valid entry from a few numbers: tokens, logprob k, draft length, tap length, the ingress
/// node (X or Y), and whether its draft is a tree.
fn generate(
    index: usize,
    (n_tokens, k, n_draft, tap, on_y, rng): (u8, u8, u8, u8, bool, u64),
    trees: bool,
) -> Owned {
    let n_tokens = usize::from(n_tokens);
    let n_draft = usize::from(n_draft);
    let finish = match rng % 7 {
        0 => Some(Finish::Stop),
        1 => Some(Finish::Error {
            domain: 3,
            code: rng as u16,
        }),
        2 => Some(Finish::Cancelled),
        _ => None,
    };
    let mut tree: Vec<TreeNode> = Vec::new();
    if trees && n_draft > 0 {
        for i in 0..n_draft {
            let parent = (i > 0).then(|| (rng % i as u64) as u16);
            // 1 under the last accepted token, else one below the parent.
            let depth = parent.map_or(1, |p| tree[usize::from(p)].depth + 1);
            tree.push(TreeNode {
                parent,
                depth,
                flags: (i % 2) as u8,
            });
        }
    }
    Owned {
        seq: SequenceId {
            ingress: if on_y { Y } else { X },
            boot: 1,
            n: index as u32,
        },
        tokens: spec::fill(n_tokens as u8, 4 * n_tokens),
        finish,
        logprob_k: k,
        logprobs: if k == 0 {
            Vec::new()
        } else {
            spec::fill(k, n_tokens * (4 + 8 * usize::from(k)))
        },
        draft: spec::fill(n_draft as u8, 4 * n_draft),
        tree,
        tap: spec::fill(tap, usize::from(tap)),
        rng_counter: rng,
    }
}

proptest! {
    /// The `Token` parser on arbitrary extensions and payloads: no panic, every refusal is
    /// classified, and an accepted frame encodes to one that decodes to it again.
    #[test]
    fn arbitrary_token_bodies_parse_safely(
        ext in proptest::collection::vec(any::<u8>(), 0..200),
        payload in proptest::collection::vec(any::<u8>(), 0..400),
    ) {
        match Token::parse(&ext, &payload) {
            Err(error) => {
                prop_assert!(REFUSAL_CLASSES.contains(&error.class()), "{error}: {:?}", error.class());
            }
            Ok(token) => {
                let bytes = token.encode_to_vec(&spec_for(1, 1)).unwrap();
                let frame = Frame::decode(&bytes).unwrap();
                prop_assert_eq!(frame.token(), Ok(token));
            }
        }
    }

    /// Golden `Token` frames with header, extension or payload bytes changed and both CRCs
    /// recomputed: no panic, every refusal is classified, and an accepted frame re-encodes to
    /// one that decodes to it again.
    #[test]
    fn resealed_token_mutations_parse_safely(
        which in any::<usize>(),
        edits in proptest::collection::vec((any::<usize>(), any::<u8>()), 1..6),
    ) {
        let goldens = goldens();
        let golden = &goldens[which % goldens.len()];
        let mut bytes = read_golden(golden.name);
        for (at, value) in edits {
            let at = at % bytes.len();
            if (4..48).contains(&at) || at >= 64 {
                bytes[at] = value;
            }
        }
        reseal(&mut bytes);
        let Ok(frame) = Frame::decode(&bytes) else { return Ok(()) };
        match frame.token() {
            Err(error) => {
                prop_assert!(REFUSAL_CLASSES.contains(&error.class()), "{error}: {:?}", error.class());
            }
            Ok(token) => {
                let header = frame.header.spec;
                let spec = TokenSpec {
                    wire_minor: header.wire_minor,
                    flags: header.flags,
                    instance: header.instance,
                    epoch: header.epoch,
                    step: header.step,
                    chan_seq: header.chan_seq,
                };
                let again = token.encode_to_vec(&spec).unwrap();
                let frame = Frame::decode(&again).unwrap();
                prop_assert_eq!(frame.token(), Ok(token));
            }
        }
    }

    /// A generated step's ring-return copy holds every entry, in order, with its draft and no
    /// tap, and every `TRACE`; its two ingress copies hold each entry once between them, with
    /// its tap and no draft, and each copy the `TRACE` of its own entries.
    #[test]
    fn generated_steps_split_into_their_copies(
        entries in proptest::collection::vec(
            (0u8..6, 0u8..4, 0u8..6, 0u8..80, any::<bool>(), any::<u64>()),
            0..6,
        ),
        trees in any::<bool>(),
    ) {
        let owned: Vec<Owned> = entries
            .iter()
            .enumerate()
            .map(|(index, numbers)| generate(index, *numbers, trees))
            .collect();
        let typed: Vec<TokenEntry<'_>> = owned.iter().map(Owned::entry).collect();
        // Entries whose `rng_counter` has bit 3 set are on a sampled step.
        let side: Vec<Side<'_>> = typed
            .iter()
            .enumerate()
            .filter(|(_, e)| e.rng_counter & 8 != 0)
            .map(|(index, _)| Side { span: Some(index as u16), item: trace(index as u8) })
            .collect();
        // Each copy's TRACE items, by the sequence of the entry they name.
        let traced = |token: &Token<'_>| -> Vec<(SequenceId, SideItem<'static>)> {
            let entries: Vec<TokenEntry<'_>> = token.entries.iter().collect();
            token
                .side
                .iter()
                .map(|s| {
                    let SideItem::Trace(t) = s.item else { panic!("{s:?} is not a TRACE") };
                    (entries[usize::from(s.span.unwrap())].seq, SideItem::Trace(t))
                })
                .collect()
        };
        let expected = |node: Option<u64>| -> Vec<(SequenceId, SideItem<'static>)> {
            side.iter()
                .filter_map(|s| {
                    let SideItem::Trace(t) = s.item else { return None };
                    let seq = typed[usize::from(s.span.unwrap())].seq;
                    node.is_none_or(|node| seq.ingress == node)
                        .then_some((seq, SideItem::Trace(t)))
                })
                .collect()
        };
        let step = StepTokens {
            entries: Entries::new(&typed),
            side: SideItems::new(&side),
            part_index: 0,
            part_count: 1,
        };
        let spec = spec_for(5, 1);
        let ring = step.encode_to_vec(TokenCopy::RingReturn, &spec).unwrap();
        let frame = Frame::decode(&ring).unwrap();
        let token = frame.token().unwrap();
        let carried: Vec<TokenEntry<'_>> = typed.iter().map(|e| as_carried(e, Route::RingReturn)).collect();
        prop_assert_eq!(token.entries, Entries::new(&carried));
        prop_assert_eq!(traced(&token), expected(None));
        prop_assert_eq!(frame.header.spec.flags.has_trace(), !side.is_empty());
        let mut seen = Vec::new();
        for node in [X, Y] {
            let bytes = step.encode_to_vec(TokenCopy::Ingress { ingress: node }, &spec).unwrap();
            let frame = Frame::decode(&bytes).unwrap();
            let token = frame.token().unwrap();
            let carried: Vec<TokenEntry<'_>> = typed
                .iter()
                .filter(|e| e.seq.ingress == node)
                .map(|e| as_carried(e, Route::ToIngress))
                .collect();
            prop_assert_eq!(token.entries, Entries::new(&carried));
            prop_assert_eq!(traced(&token), expected(Some(node)));
            seen.extend(token.entries.iter().map(|e| e.seq.n));
        }
        seen.sort();
        prop_assert_eq!(seen, (0..typed.len() as u32).collect::<Vec<_>>());
    }
}
