// SPDX-License-Identifier: MIT OR Apache-2.0

//! The stop matcher against a naive search of the whole text (PROD-E2-T03).

use proptest::prelude::*;

use super::*;

/// The reference: for each stop string, its first occurrence in the whole text; the one that
/// ends first wins, and on a tie the one that starts first. Written apart from the matcher's own
/// search.
fn naive(stops: &[String], text: &str) -> (String, Option<usize>) {
    let best = stops
        .iter()
        .enumerate()
        .filter(|(_, s)| !s.is_empty())
        .filter_map(|(i, s)| text.find(s.as_str()).map(|at| (at + s.len(), at, i)))
        .min();
    match best {
        Some((_, at, i)) => (text[..at].to_owned(), Some(i)),
        None => (text.to_owned(), None),
    }
}

/// Runs the matcher over `chunks`, checking the hold-back bound after every push.
fn stream(stops: &[String], chunks: &[String]) -> (String, Option<usize>) {
    let mut m = StopMatcher::new(stops);
    let bound = m.max_stop_len().saturating_sub(1);
    let mut out = String::new();
    for c in chunks {
        let r = m.push(c);
        out.push_str(&r.text);
        assert!(
            m.held().len() <= bound,
            "held {:?} over {bound} bytes",
            m.held()
        );
        if let Some(s) = r.stop {
            assert!(m.stopped());
            assert_eq!(m.push("more"), Released::default());
            return (out, Some(s.index));
        }
    }
    out.push_str(&m.finish());
    (out, None)
}

#[test]
fn a_stop_string_spanning_pushes_matches_and_nothing_after_it_is_released() {
    let stops = ["</answer>".to_owned()];
    let mut m = StopMatcher::new(&stops);
    assert_eq!(m.push("The result is 4</an").text, "The result is 4");
    assert_eq!(m.held(), "</an");
    let r = m.push("swer> and more");
    assert_eq!(r.text, "");
    assert_eq!(r.stop, Some(StopMatch { index: 0 }));
    assert_eq!(m.finish(), "");
}

#[test]
fn a_held_prefix_that_does_not_complete_is_released() {
    let stops = ["STOP".to_owned()];
    let mut m = StopMatcher::new(&stops);
    assert_eq!(m.push("ST").text, "");
    assert_eq!(m.push("ART").text, "START");
    assert_eq!(m.push("S").text, "");
    assert_eq!(m.finish(), "S");
}

#[test]
fn the_first_stop_to_complete_wins_and_ties_go_to_the_earlier_start() {
    // "bc" completes before "abcd" can.
    let stops = ["abcd".to_owned(), "bc".to_owned()];
    assert_eq!(
        stream(&stops, &["xabcd".to_owned()]),
        ("xa".to_owned(), Some(1))
    );
    // Both end at the same byte: the longer, earlier one wins.
    let stops = ["c".to_owned(), "abc".to_owned()];
    assert_eq!(
        stream(&stops, &["xab".to_owned(), "c".to_owned()]),
        ("x".to_owned(), Some(1))
    );
}

#[test]
fn multi_byte_stop_strings_hold_back_whole_characters() {
    let stops = ["€uro".to_owned()];
    let mut m = StopMatcher::new(&stops);
    assert_eq!(m.push("price: €").text, "price: ");
    assert_eq!(m.held(), "€");
    assert_eq!(m.push("u").text, "");
    assert_eq!(m.push("p").text, "€up");
}

#[test]
fn empty_stop_strings_never_match_and_no_stops_hold_nothing() {
    let stops = [String::new(), "z".to_owned()];
    assert_eq!(
        stream(&stops, &["abc".to_owned(), "z".to_owned()]),
        ("abc".to_owned(), Some(1))
    );
    let mut m = StopMatcher::new::<String>(&[]);
    assert_eq!(m.max_stop_len(), 0);
    assert_eq!(m.push("anything").text, "anything");
    assert_eq!(m.held(), "");
}

/// Text over a small alphabet, multi-byte characters included, so stop strings occur often.
fn alphabet_text(max: usize) -> impl Strategy<Value = String> {
    prop::collection::vec(
        prop::sample::select(vec!["a", "b", "c", "é", "€", " ", "\n"]),
        0..max,
    )
    .prop_map(|v| v.concat())
}

/// `text` cut into pushes at character boundaries.
fn chunked(text: &str, cuts: &[usize]) -> Vec<String> {
    let bounds: Vec<usize> = text
        .char_indices()
        .map(|(i, _)| i)
        .chain([text.len()])
        .collect();
    let mut at: Vec<usize> = cuts.iter().map(|c| bounds[c % bounds.len()]).collect();
    at.push(0);
    at.push(text.len());
    at.sort_unstable();
    at.dedup();
    at.windows(2).map(|w| text[w[0]..w[1]].to_owned()).collect()
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 2048, ..ProptestConfig::default() })]

    /// RFC-0050 §5.2: the streaming matcher agrees with the naive search for any split of the
    /// text into pushes, and never holds more than `max_stop_len − 1` bytes.
    #[test]
    fn the_matcher_agrees_with_a_naive_search(
        stops in prop::collection::vec(alphabet_text(5), 0..4),
        text in alphabet_text(40),
        cuts in prop::collection::vec(any::<usize>(), 0..12),
    ) {
        let chunks = chunked(&text, &cuts);
        prop_assert_eq!(stream(&stops, &chunks), naive(&stops, &text));
    }
}
