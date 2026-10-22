// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tests of the fault points against RFC-0060 §9.2: the macro in both builds, each trigger and
//! effect, filters, loading and the counters.
//!
//! The plan and counters are process-wide, so the tests with the `fault-injection` feature hold
//! a lock (`injected::serial`) while they run, and each uses point names of its own.

use syntropy_core::{ErrorClass, Nanos};

#[cfg(not(feature = "fault-injection"))]
use super::Ctx;
use super::FaultEffect;

#[test]
fn effects_have_the_plan_files_names() {
    let cases = [
        (FaultEffect::None, r#""none""#),
        (
            FaultEffect::Error(ErrorClass::Device),
            r#"{"error":"Device"}"#,
        ),
        (FaultEffect::Delay(Nanos(5_000)), r#"{"delay":5000}"#),
        (
            FaultEffect::Corrupt { bytes: 4 },
            r#"{"corrupt":{"bytes":4}}"#,
        ),
        (FaultEffect::Panic, r#""panic""#),
        (FaultEffect::Hang, r#""hang""#),
    ];
    for (effect, json) in cases {
        assert_eq!(serde_json::to_string(&effect).unwrap(), json);
        assert_eq!(serde_json::from_str::<FaultEffect>(json).unwrap(), effect);
    }
}

/// Without the feature the macro is `FaultEffect::None` in both forms, and the context
/// expression is never evaluated.
#[cfg(not(feature = "fault-injection"))]
#[test]
fn without_the_feature_a_point_is_none_and_costs_nothing() {
    let evaluated = std::cell::Cell::new(false);
    assert_eq!(crate::fault_point!("obs.test.off"), FaultEffect::None);
    let effect = crate::fault_point!("obs.test.off", {
        evaluated.set(true);
        Ctx::default()
    });
    assert_eq!(effect, FaultEffect::None);
    assert!(!evaluated.get());
}

#[cfg(feature = "fault-injection")]
mod injected {
    use std::collections::BTreeMap;
    use std::panic;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::mpsc;
    use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
    use std::time::Duration;

    use syntropy_core::{
        Clock, DeviceId, Endpoint, Entropy, LinkId, LinkKind, MonoTime, NetMedium, NodeId,
        SequenceId,
    };

    use super::super::inject::{draw, install_with_sleep, load_from_var};
    use super::super::*;

    /// Holds the process-wide plan for one test, starting with no plan and zero counters.
    pub(super) fn serial() -> MutexGuard<'static, ()> {
        static LOCK: Mutex<()> = Mutex::new(());
        let guard = LOCK.lock().unwrap_or_else(PoisonError::into_inner);
        clear();
        reset_counters();
        guard
    }

    /// A clock a test sets by hand.
    struct ManualClock(AtomicU64);

    impl ManualClock {
        fn at(nanos: u64) -> Arc<ManualClock> {
            Arc::new(ManualClock(AtomicU64::new(nanos)))
        }

        fn set(&self, nanos: u64) {
            self.0.store(nanos, Ordering::SeqCst);
        }
    }

    impl Clock for ManualClock {
        fn now(&self) -> MonoTime {
            MonoTime(self.0.load(Ordering::SeqCst))
        }

        fn unix_ms(&self) -> u64 {
            0
        }
    }

    fn rule(point: &str, when: Trigger, effect: FaultEffect) -> FaultRule {
        FaultRule {
            point: point.to_owned(),
            when,
            filter: None,
            effect,
        }
    }

    fn run(rules: Vec<FaultRule>) -> Arc<ManualClock> {
        let clock = ManualClock::at(0);
        install(FaultPlan { rules }, clock.clone()).unwrap();
        clock
    }

    const DEVICE: FaultEffect = FaultEffect::Error(ErrorClass::Device);

    fn node(last: u8) -> NodeId {
        let mut id = [0; 32];
        id[31] = last;
        NodeId(id)
    }

    fn device(last: u8, ordinal: u16) -> DeviceId {
        DeviceId {
            node: node(last),
            ordinal,
        }
    }

    /// The hits, from 1, among the first `hits` that fire.
    fn firing(hits: u64, mut point: impl FnMut() -> FaultEffect) -> Vec<u64> {
        (1..=hits)
            .filter(|_| point() != FaultEffect::None)
            .collect()
    }

    #[test]
    fn no_plan_fires_nothing_and_counts_nothing() {
        let _serial = serial();
        assert_eq!(crate::fault_point!("obs.test.none"), FaultEffect::None);
        assert!(counters().is_empty());
    }

    #[test]
    fn always_fires_every_hit() {
        let _serial = serial();
        run(vec![rule("obs.test.always", Trigger::Always, DEVICE)]);
        assert_eq!(
            firing(5, || crate::fault_point!("obs.test.always")),
            [1, 2, 3, 4, 5]
        );
    }

    #[test]
    fn nth_fires_once_on_that_hit() {
        let _serial = serial();
        run(vec![rule("obs.test.nth", Trigger::Nth(3), DEVICE)]);
        assert_eq!(firing(8, || crate::fault_point!("obs.test.nth")), [3]);
    }

    #[test]
    fn probability_zero_and_one_are_never_and_always() {
        let _serial = serial();
        run(vec![
            rule(
                "obs.test.p0",
                Trigger::Probability { p: 0.0, seed: 1 },
                DEVICE,
            ),
            rule(
                "obs.test.p1",
                Trigger::Probability { p: 1.0, seed: 1 },
                DEVICE,
            ),
        ]);
        assert!(firing(1_000, || crate::fault_point!("obs.test.p0")).is_empty());
        assert_eq!(
            firing(1_000, || crate::fault_point!("obs.test.p1")).len(),
            1_000
        );
    }

    /// The fires are the draws below p, a seed replays them, and another seed gives others.
    #[test]
    fn probability_is_a_replayable_function_of_its_seed() {
        let _serial = serial();
        let point = "obs.test.probability";
        let plan = |seed| vec![rule(point, Trigger::Probability { p: 0.25, seed }, DEVICE)];
        run(plan(42));
        let first = firing(10_000, || crate::fault_point!("obs.test.probability"));
        let expected: Vec<u64> = (1..=10_000)
            .filter(|&k| draw(42, point, k) < 0.25)
            .collect();
        assert_eq!(first, expected);
        // Installing the plan again starts its hit count again, so the same seed replays.
        run(plan(42));
        assert_eq!(
            firing(10_000, || crate::fault_point!("obs.test.probability")),
            first
        );
        run(plan(43));
        assert_ne!(
            firing(10_000, || crate::fault_point!("obs.test.probability")),
            first
        );
        // 10,000 draws at p = 0.25 have a standard deviation of about 43 fires; this seed's
        // count is a fixed number, checked against a band of 7 of them either side.
        assert!((2_200..=2_800).contains(&first.len()), "{}", first.len());
    }

    /// A seed drawn from an `Entropy` is its first eight bytes, read little-endian (RFC-0060
    /// §2.1): the run's seeded source decides the fires.
    #[test]
    fn a_probability_seed_comes_from_the_entropy_given() {
        struct Fixed;
        impl Entropy for Fixed {
            fn fill(&self, buf: &mut [u8]) {
                for (i, byte) in buf.iter_mut().enumerate() {
                    *byte = i as u8 + 1;
                }
            }
        }
        assert_eq!(
            Trigger::probability(0.5, &Fixed),
            Trigger::Probability {
                p: 0.5,
                seed: u64::from_le_bytes([1, 2, 3, 4, 5, 6, 7, 8]),
            }
        );
    }

    /// Draws lie in [0, 1) and depend on the seed, the point and the hit.
    #[test]
    fn draws_are_in_the_unit_interval_and_keyed() {
        for k in 1..=1_000 {
            let value = draw(9, "obs.test.draw", k);
            assert!((0.0..1.0).contains(&value), "{value}");
        }
        assert_ne!(draw(9, "obs.test.draw", 1), draw(10, "obs.test.draw", 1));
        assert_ne!(draw(9, "obs.test.draw", 1), draw(9, "obs.test.drew", 1));
        assert_ne!(draw(9, "obs.test.draw", 1), draw(9, "obs.test.draw", 2));
    }

    #[test]
    fn after_fires_from_its_instant_on() {
        let _serial = serial();
        let clock = run(vec![rule(
            "obs.test.after",
            Trigger::After(MonoTime(200)),
            DEVICE,
        )]);
        for (now, effect) in [
            (0, FaultEffect::None),
            (199, FaultEffect::None),
            (200, DEVICE),
            (5_000, DEVICE),
        ] {
            clock.set(now);
            assert_eq!(crate::fault_point!("obs.test.after"), effect, "at {now} ns");
        }
    }

    #[test]
    fn window_fires_from_its_start_up_to_its_end() {
        let _serial = serial();
        let clock = run(vec![rule(
            "obs.test.window",
            Trigger::Window {
                from: MonoTime(200),
                to: MonoTime(300),
            },
            DEVICE,
        )]);
        for (now, effect) in [
            (199, FaultEffect::None),
            (200, DEVICE),
            (299, DEVICE),
            (300, FaultEffect::None),
        ] {
            clock.set(now);
            assert_eq!(
                crate::fault_point!("obs.test.window"),
                effect,
                "at {now} ns"
            );
        }
    }

    #[test]
    fn error_and_corrupt_return_to_the_call_site() {
        let _serial = serial();
        run(vec![
            rule(
                "obs.test.error",
                Trigger::Always,
                FaultEffect::Error(ErrorClass::Integrity),
            ),
            rule(
                "obs.test.corrupt",
                Trigger::Always,
                FaultEffect::Corrupt { bytes: 4 },
            ),
        ]);
        assert_eq!(
            crate::fault_point!("obs.test.error"),
            FaultEffect::Error(ErrorClass::Integrity)
        );
        assert_eq!(
            crate::fault_point!("obs.test.corrupt"),
            FaultEffect::Corrupt { bytes: 4 }
        );
    }

    /// Waits up to 10 s, in 1 ms sleeps, for some point to fire. The tests read no clock (RFC-0060
    /// §2.1 rule T1), so the bound is a count of sleeps.
    fn wait_for_a_fire() {
        for _ in 0..10_000 {
            if !fired().is_empty() {
                return;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        panic!("the point never fired");
    }

    /// The delays a `Delay` effect asked the recording sleep for, in order.
    static SLEPT: Mutex<Vec<Duration>> = Mutex::new(Vec::new());

    fn record_sleep(delay: Duration) {
        SLEPT
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(delay);
    }

    /// The point sleeps for exactly the rule's delay, through the sleep it was installed with,
    /// then returns the effect. Recording the request keeps the test free of timing.
    #[test]
    fn delay_sleeps_for_the_rules_duration_then_returns() {
        let _serial = serial();
        SLEPT.lock().unwrap_or_else(PoisonError::into_inner).clear();
        let delay = Nanos(200_000_000);
        install_with_sleep(
            FaultPlan {
                rules: vec![rule(
                    "obs.test.delay",
                    Trigger::Always,
                    FaultEffect::Delay(delay),
                )],
            },
            ManualClock::at(0),
            record_sleep,
        )
        .unwrap();
        assert_eq!(
            crate::fault_point!("obs.test.delay"),
            FaultEffect::Delay(delay)
        );
        crate::fault_point!("obs.test.delay");
        assert_eq!(
            *SLEPT.lock().unwrap_or_else(PoisonError::into_inner),
            [Duration::from_nanos(delay.0); 2]
        );
    }

    /// With the real sleep, a delayed point returns its effect. Only an upper bound is checked:
    /// how long the OS sleeps past the delay varies, so there is no "not yet" check.
    #[test]
    fn a_real_delay_returns_within_a_generous_bound() {
        let _serial = serial();
        let delay = Nanos(10_000_000);
        run(vec![rule(
            "obs.test.delay-real",
            Trigger::Always,
            FaultEffect::Delay(delay),
        )]);
        let (sent, received) = mpsc::channel();
        let thread = std::thread::spawn(move || {
            sent.send(crate::fault_point!("obs.test.delay-real"))
                .unwrap();
        });
        assert_eq!(
            received.recv_timeout(Duration::from_secs(10)),
            Ok(FaultEffect::Delay(delay))
        );
        thread.join().unwrap();
    }

    #[test]
    fn panic_names_the_point_and_leaves_the_plan_usable() {
        let _serial = serial();
        run(vec![rule(
            "obs.test.panic",
            Trigger::Nth(1),
            FaultEffect::Panic,
        )]);
        let payload = panic::catch_unwind(|| crate::fault_point!("obs.test.panic")).unwrap_err();
        let message = payload.downcast_ref::<String>().unwrap();
        assert_eq!(
            message,
            "fault point `obs.test.panic` fired: an injected panic"
        );
        // The lock was not held when the point panicked, so the next hit goes through.
        assert_eq!(crate::fault_point!("obs.test.panic"), FaultEffect::None);
        assert_eq!(
            counters()["obs.test.panic"],
            PointCounts { hits: 2, fires: 1 }
        );
    }

    /// A hang holds its thread until the plan is cleared, then returns.
    #[test]
    fn hang_blocks_until_the_plan_is_cleared() {
        let _serial = serial();
        run(vec![rule(
            "obs.test.hang",
            Trigger::Always,
            FaultEffect::Hang,
        )]);
        let (sent, received) = mpsc::channel();
        let thread = std::thread::spawn(move || {
            sent.send(crate::fault_point!("obs.test.hang")).unwrap();
        });
        // The fire is counted before the thread blocks.
        wait_for_a_fire();
        assert_eq!(
            received.recv_timeout(Duration::from_millis(200)),
            Err(mpsc::RecvTimeoutError::Timeout)
        );
        clear();
        assert_eq!(
            received.recv_timeout(Duration::from_secs(10)),
            Ok(FaultEffect::Hang)
        );
        thread.join().unwrap();
    }

    /// Installing a new plan releases a hang too.
    #[test]
    fn hang_ends_when_the_plan_is_replaced() {
        let _serial = serial();
        run(vec![rule(
            "obs.test.hang-replace",
            Trigger::Always,
            FaultEffect::Hang,
        )]);
        let thread = std::thread::spawn(|| crate::fault_point!("obs.test.hang-replace"));
        wait_for_a_fire();
        run(Vec::new());
        assert_eq!(thread.join().unwrap(), FaultEffect::Hang);
    }

    #[test]
    fn filters_match_only_the_context_they_name() {
        let _serial = serial();
        let link = LinkId {
            src: Endpoint::Device(device(1, 0)),
            dst: Endpoint::Device(device(2, 0)),
            kind: LinkKind::Network(NetMedium::Ethernet),
            path: 0,
        };
        let mut by_device = rule("obs.test.filter", Trigger::Always, DEVICE);
        by_device.filter = Some(CtxFilter {
            device: Some(device(1, 0)),
            ..CtxFilter::default()
        });
        let mut by_link = rule("obs.test.filter", Trigger::Always, FaultEffect::Panic);
        by_link.filter = Some(CtxFilter {
            node: Some(node(1)),
            link: Some(link),
            ..CtxFilter::default()
        });
        run(vec![by_device, by_link]);
        let at = |device, node, link| Ctx {
            node,
            device,
            link,
            sequence: None,
        };
        let none = FaultEffect::None;
        // The device filter: only that device matches, and a context without one does not.
        assert_eq!(
            crate::fault_point!("obs.test.filter", at(Some(device(1, 0)), None, None)),
            DEVICE
        );
        assert_eq!(
            crate::fault_point!("obs.test.filter", at(Some(device(1, 1)), None, None)),
            none
        );
        assert_eq!(crate::fault_point!("obs.test.filter", Ctx::default()), none);
        // The link filter needs both fields it names; the panic shows it matched.
        assert_eq!(
            crate::fault_point!("obs.test.filter", at(None, None, Some(link))),
            none
        );
        let both = at(None, Some(node(1)), Some(link));
        assert!(panic::catch_unwind(|| crate::fault_point!("obs.test.filter", both)).is_err());
    }

    /// A sequence filter matches only the sequence it names, so a plan can fail one request.
    #[test]
    fn a_sequence_filter_fails_only_that_sequence() {
        let _serial = serial();
        let target = SequenceId::new(&node(1), 3, 7);
        let mut on_sequence = rule("obs.test.sequence", Trigger::Always, DEVICE);
        on_sequence.filter = Some(CtxFilter {
            sequence: Some(target),
            ..CtxFilter::default()
        });
        run(vec![on_sequence]);
        let serving = |sequence| Ctx {
            sequence: Some(sequence),
            ..Ctx::default()
        };
        assert_eq!(
            crate::fault_point!("obs.test.sequence", serving(target)),
            DEVICE
        );
        assert_eq!(
            crate::fault_point!(
                "obs.test.sequence",
                serving(SequenceId::new(&node(1), 3, 8))
            ),
            FaultEffect::None
        );
        assert_eq!(
            crate::fault_point!("obs.test.sequence", Ctx::default()),
            FaultEffect::None
        );
        let plan = FaultPlan::from_toml(
            "[[rules]]\npoint = \"obs.test.sequence\"\nwhen = \"always\"\neffect = \"panic\"\n\
             [rules.filter.sequence]\ningress = 9\nboot = 3\nn = 7\n",
        )
        .unwrap();
        assert_eq!(
            plan.rules[0].filter.unwrap().sequence,
            Some(SequenceId {
                ingress: 9,
                boot: 3,
                n: 7
            })
        );
    }

    /// The one-argument form reads the context the innermost `Ctx::enter` set.
    #[test]
    fn the_thread_context_is_the_innermost_entered() {
        let _serial = serial();
        let mut on_device = rule("obs.test.here", Trigger::Always, DEVICE);
        on_device.filter = Some(CtxFilter {
            device: Some(device(1, 0)),
            ..CtxFilter::default()
        });
        run(vec![on_device]);
        let on = |d| Ctx {
            device: Some(d),
            ..Ctx::default()
        };
        assert_eq!(Ctx::here(), Ctx::default());
        assert_eq!(crate::fault_point!("obs.test.here"), FaultEffect::None);
        {
            let _outer = on(device(1, 0)).enter();
            assert_eq!(crate::fault_point!("obs.test.here"), DEVICE);
            {
                let _inner = on(device(1, 1)).enter();
                assert_eq!(Ctx::here(), on(device(1, 1)));
                assert_eq!(crate::fault_point!("obs.test.here"), FaultEffect::None);
            }
            assert_eq!(Ctx::here(), on(device(1, 0)));
            assert_eq!(crate::fault_point!("obs.test.here"), DEVICE);
            // Another thread starts with the empty context.
            let other = std::thread::spawn(Ctx::here).join().unwrap();
            assert_eq!(other, Ctx::default());
        }
        assert_eq!(Ctx::here(), Ctx::default());
    }

    /// The first matching rule that fires gives the effect, and every matching rule counts the
    /// hit, so a later rule's `nth` does not depend on the rules before it.
    #[test]
    fn the_first_rule_that_fires_wins_and_every_match_counts() {
        let _serial = serial();
        run(vec![
            rule("obs.test.order", Trigger::Nth(2), DEVICE),
            rule(
                "obs.test.order",
                Trigger::Always,
                FaultEffect::Corrupt { bytes: 1 },
            ),
            rule("obs.test.order", Trigger::Nth(3), FaultEffect::Panic),
            rule("obs.test.other", Trigger::Always, FaultEffect::Panic),
        ]);
        let corrupt = FaultEffect::Corrupt { bytes: 1 };
        assert_eq!(crate::fault_point!("obs.test.order"), corrupt);
        assert_eq!(crate::fault_point!("obs.test.order"), DEVICE);
        // The third rule's third match comes after the `Always` rule, which fires first.
        assert_eq!(crate::fault_point!("obs.test.order"), corrupt);
    }

    #[test]
    fn counters_report_hits_and_every_point_that_fired() {
        let _serial = serial();
        run(vec![
            rule("obs.test.count-a", Trigger::Nth(2), DEVICE),
            rule("obs.test.count-b", Trigger::Always, DEVICE),
            rule("obs.test.count-never", Trigger::Nth(100), DEVICE),
        ]);
        for _ in 0..3 {
            crate::fault_point!("obs.test.count-a");
        }
        crate::fault_point!("obs.test.count-b");
        crate::fault_point!("obs.test.count-never");
        crate::fault_point!("obs.test.count-unplanned");
        let expected = BTreeMap::from([
            ("obs.test.count-a", PointCounts { hits: 3, fires: 1 }),
            ("obs.test.count-b", PointCounts { hits: 1, fires: 1 }),
            ("obs.test.count-never", PointCounts { hits: 1, fires: 0 }),
            (
                "obs.test.count-unplanned",
                PointCounts { hits: 1, fires: 0 },
            ),
        ]);
        assert_eq!(counters(), expected);
        assert_eq!(fired(), ["obs.test.count-a", "obs.test.count-b"]);
        // Counters carry on across plans until they are reset.
        run(vec![rule("obs.test.count-b", Trigger::Always, DEVICE)]);
        crate::fault_point!("obs.test.count-b");
        assert_eq!(
            counters()["obs.test.count-b"],
            PointCounts { hits: 2, fires: 2 }
        );
        reset_counters();
        assert!(counters().is_empty());
        assert!(fired().is_empty());
    }

    /// Every trigger and effect in its TOML form, including the example in `FaultPlan`'s docs.
    #[test]
    fn plans_parse_from_toml() {
        let text = r#"
            [[rules]]
            point = "io.ring.submit"
            when = { nth = 3 }
            effect = { error = "Device" }

            [[rules]]
            point = "net.channel.send"
            when = { probability = { p = 0.25, seed = 7 } }
            effect = { delay = 5000000 }

            [rules.filter.device]
            node = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
            ordinal = 0

            [[rules]]
            point = "store.chunk.receive"
            when = "always"
            effect = { corrupt = { bytes = 4 } }

            [[rules]]
            point = "kernels.cuda.launch"
            when = { after = 1000 }
            effect = "panic"

            [[rules]]
            point = "ipc.channel.send"
            when = { window = { from = 10, to = 20 } }
            effect = "hang"
        "#;
        let mut probability = rule(
            "net.channel.send",
            Trigger::Probability { p: 0.25, seed: 7 },
            FaultEffect::Delay(Nanos(5_000_000)),
        );
        probability.filter = Some(CtxFilter {
            device: Some(device(1, 0)),
            ..CtxFilter::default()
        });
        let expected = FaultPlan {
            rules: vec![
                rule("io.ring.submit", Trigger::Nth(3), DEVICE),
                probability,
                rule(
                    "store.chunk.receive",
                    Trigger::Always,
                    FaultEffect::Corrupt { bytes: 4 },
                ),
                rule(
                    "kernels.cuda.launch",
                    Trigger::After(MonoTime(1_000)),
                    FaultEffect::Panic,
                ),
                rule(
                    "ipc.channel.send",
                    Trigger::Window {
                        from: MonoTime(10),
                        to: MonoTime(20),
                    },
                    FaultEffect::Hang,
                ),
            ],
        };
        assert_eq!(FaultPlan::from_toml(text).unwrap(), expected);
        assert_eq!(FaultPlan::from_toml("").unwrap(), FaultPlan::default());
    }

    #[test]
    fn malformed_plans_are_refused() {
        for text in [
            "rules = 3",
            "[[rules]]\npoint = \"a.b.c\"\nwhen = \"always\"",
            "[[rules]]\npoint = \"a.b.c\"\nwhen = \"sometimes\"\neffect = \"panic\"",
            "[[rules]]\npoint = \"a.b.c\"\nwhen = \"always\"\neffect = \"panic\"\nextra = 1",
            "[[rules]]\npoint = \"a.b.c\"\nwhen = { nth = -1 }\neffect = \"panic\"",
        ] {
            assert!(
                matches!(FaultPlan::from_toml(text), Err(PlanError::Parse(_))),
                "{text}"
            );
        }
    }

    #[test]
    fn invalid_rules_are_refused_and_the_installed_plan_stays() {
        let _serial = serial();
        let cases = [
            (rule("", Trigger::Always, DEVICE), "the point name is empty"),
            (
                rule("a.b.c", Trigger::Nth(0), DEVICE),
                "`nth` counts hits from 1, not 0",
            ),
            (
                rule("a.b.c", Trigger::Probability { p: 1.5, seed: 0 }, DEVICE),
                "probability 1.5 is not in [0, 1]",
            ),
            (
                rule("a.b.c", Trigger::Probability { p: -0.1, seed: 0 }, DEVICE),
                "probability -0.1 is not in [0, 1]",
            ),
            (
                rule(
                    "a.b.c",
                    Trigger::Probability {
                        p: f64::NAN,
                        seed: 0,
                    },
                    DEVICE,
                ),
                "probability NaN is not in [0, 1]",
            ),
            (
                rule(
                    "a.b.c",
                    Trigger::Window {
                        from: MonoTime(5),
                        to: MonoTime(5),
                    },
                    DEVICE,
                ),
                "the window from 5 ns to 5 ns is empty",
            ),
        ];
        run(vec![rule("obs.test.kept", Trigger::Always, DEVICE)]);
        for (bad, reason) in cases {
            let plan = FaultPlan {
                rules: vec![rule("a.b.ok", Trigger::Always, DEVICE), bad],
            };
            let error = install(plan, ManualClock::at(0)).unwrap_err();
            assert_eq!(error.to_string(), format!("fault plan rule 1: {reason}"));
        }
        assert_eq!(crate::fault_point!("obs.test.kept"), DEVICE);
    }

    fn plan_file(name: &str, text: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "syntropy-obs-faults-{}-{name}.toml",
            std::process::id()
        ));
        std::fs::write(&path, text).unwrap();
        path
    }

    #[test]
    fn plans_load_from_the_variables_file() {
        let _serial = serial();
        assert_eq!(PLAN_ENV, "SYNTROPY_FAULT_PLAN");
        assert!(!load_from_var(None, ManualClock::at(0)).unwrap());
        assert!(!load_from_var(Some("".into()), ManualClock::at(0)).unwrap());
        assert_eq!(crate::fault_point!("obs.test.file"), FaultEffect::None);
        let path = plan_file(
            "load",
            "[[rules]]\npoint = \"obs.test.file\"\nwhen = \"always\"\neffect = { error = \"Link\" }\n",
        );
        let loaded = load_from_var(Some(path.clone().into_os_string()), ManualClock::at(0));
        std::fs::remove_file(&path).unwrap();
        assert!(loaded.unwrap());
        assert_eq!(
            crate::fault_point!("obs.test.file"),
            FaultEffect::Error(ErrorClass::Link)
        );
        let missing = path.with_extension("missing");
        let error =
            load_from_var(Some(missing.clone().into_os_string()), ManualClock::at(0)).unwrap_err();
        assert!(matches!(&error, PlanError::Read { path, .. } if *path == missing));
        let bad = plan_file("bad", "[[rules]]\npoint = 1\n");
        let error = load_from_path(&bad, ManualClock::at(0)).unwrap_err();
        std::fs::remove_file(&bad).unwrap();
        assert!(matches!(error, PlanError::Parse(_)), "{error}");
    }

    const CHILD: &str = "SYNTROPY_OBS_FAULTS_TEST_CHILD";

    /// End to end through the real environment: a child process of this test binary starts with
    /// `SYNTROPY_FAULT_PLAN` set and loads the plan it names. Setting the variable in this process
    /// would need `unsafe`, which `obs` forbids.
    #[test]
    fn a_process_loads_the_plan_its_environment_names() {
        if std::env::var_os(CHILD).is_some() {
            // The child: load the plan, then fire its point.
            assert!(load_from_env(ManualClock::at(0)).unwrap());
            assert_eq!(
                crate::fault_point!("obs.test.env"),
                FaultEffect::Error(ErrorClass::Timeout)
            );
            assert_eq!(fired(), ["obs.test.env"]);
            return;
        }
        let path = plan_file(
            "env",
            "[[rules]]\npoint = \"obs.test.env\"\nwhen = \"always\"\neffect = { error = \"Timeout\" }\n",
        );
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "faults::tests::injected::a_process_loads_the_plan_its_environment_names",
            ])
            .args(["--test-threads", "1"])
            .env(CHILD, "1")
            .env(PLAN_ENV, &path)
            .output()
            .unwrap();
        std::fs::remove_file(&path).unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success(),
            "{stdout}\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(stdout.contains("1 passed"), "{stdout}");
    }
}
