// SPDX-License-Identifier: MIT OR Apache-2.0

//! The probe end to end over real endpoints on this host: a direct run, a forced relay through a
//! relay server started here, and the binary from `listen` to `check`. None needs a network beyond
//! loopback.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use syntropy_net::endpoint::test_relay::{TestRelay, bind_trusting_any_relay};
use syntropy_net::endpoint::{Config, Endpoint, NodeAddr, Relays};
use syntropy_netprobe::platform::Fixed;
use syntropy_netprobe::probe::{self, RunOptions};
use syntropy_netprobe::report::{EndInfo, Medium, Os, Outcome, Path, RelayChoice, Report, Setting};
use syntropy_netprobe::scan::scan;
use syntropy_netprobe::wire::ALPN;

const LISTENER: EndInfo = EndInfo {
    os: Os::Linux,
    medium: Medium::Loopback,
    wifi: None,
};
const RUNNER: EndInfo = EndInfo {
    os: Os::Macos,
    medium: Medium::Loopback,
    wifi: None,
};

/// A short run: 3 s at 100 ms, a 1 s timeout, the local end looked at again every second.
fn short(relays: RelayChoice, direct_paths: bool) -> RunOptions {
    let mut options = RunOptions::new(Setting::Lan, relays, direct_paths);
    options.duration = Duration::from_secs(3);
    options.period = Duration::from_millis(100);
    options.timeout = Duration::from_secs(1);
    options.detect_every = Duration::from_secs(1);
    options.checkpoint_every = Duration::from_secs(1);
    options.connect_timeout = Duration::from_secs(30);
    options
}

/// A bound socket as a peer on this host dials it.
fn local(addr: SocketAddr) -> SocketAddr {
    match addr.ip() {
        IpAddr::V4(ip) if ip.is_unspecified() => {
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), addr.port())
        }
        IpAddr::V6(ip) if ip.is_unspecified() => {
            SocketAddr::new(Ipv6Addr::LOCALHOST.into(), addr.port())
        }
        _ => addr,
    }
}

/// The checks every report must pass: the schema test, and no identifier in its text.
fn assert_clean(report: &Report) {
    assert_eq!(report.check(), Vec::<String>::new());
    let text = serde_json::to_string_pretty(report).unwrap();
    assert_eq!(scan(&text), []);
    let back: Report = serde_json::from_str(&text).unwrap();
    assert_eq!(&back, report);
}

#[tokio::test]
async fn a_direct_run_reports_a_direct_hop_with_both_ends() {
    let listener = Endpoint::bind([21; 32], Config::default().accept_alpn(ALPN))
        .await
        .unwrap();
    let mut peer = NodeAddr::new(listener.node_id()).unwrap();
    for addr in listener.bound_addrs() {
        peer = peer.with_direct_addr(local(addr));
    }
    let serving = tokio::spawn(probe::serve(
        listener.clone(),
        Arc::new(Fixed(LISTENER)),
        Duration::from_secs(1),
    ));

    let runner = Endpoint::bind([22; 32], Config::default()).await.unwrap();
    let mut checkpoints = Vec::new();
    let report = probe::run(
        &runner,
        peer,
        short(RelayChoice::Disabled, true),
        Arc::new(Fixed(RUNNER)),
        |report| checkpoints.push(report.clone()),
    )
    .await
    .unwrap();
    runner.close().await;
    listener.close().await;
    serving.await.unwrap().unwrap();

    assert_clean(&report);
    assert_eq!(report.outcome, Outcome::Completed);
    assert_eq!(report.join.path_at_connect, Path::Direct);
    assert!(report.join.first_direct_ms.is_some());
    assert_eq!(report.join.relayed_rtt_us, None);
    assert_eq!(report.ends.local, RUNNER);
    assert_eq!(report.ends.remote, Some(LISTENER));
    // 3 s at 100 ms: about 30 requests, all on the direct path, nearly all echoed.
    let sent = report.trace.rtt_us.len();
    assert!((25..=32).contains(&sent), "{sent} requests");
    assert!(report.hops.iter().all(|hop| hop.path == Path::Direct));
    let echoed = report.trace.rtt_us.iter().flatten().count();
    assert!(echoed * 10 >= sent * 9, "{echoed} of {sent} echoed");
    assert!(report.hops[0].rtt_us.unwrap().p50 > 0);
    // Checkpoints came during the run, each a valid report of its own.
    assert!(!checkpoints.is_empty());
    for checkpoint in &checkpoints {
        assert_eq!(checkpoint.outcome, Outcome::InProgress);
        assert_clean(checkpoint);
    }
}

#[tokio::test]
async fn a_forced_relay_is_reported_as_relayed() {
    // A relay server on this host, and neither endpoint with a direct path: every packet goes
    // through the relay.
    let relay = TestRelay::start().await.unwrap();
    let relayed = Config::default()
        .relays(Relays::Url(relay.url()))
        .direct_paths(false);
    let listener = bind_trusting_any_relay([23; 32], relayed.clone().accept_alpn(ALPN))
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(30), listener.online())
        .await
        .expect("the listener reaches the relay within 30 s");
    let peer = NodeAddr::new(listener.node_id())
        .unwrap()
        .with_relay_url(relay.url());
    let serving = tokio::spawn(probe::serve(
        listener.clone(),
        Arc::new(Fixed(LISTENER)),
        Duration::from_secs(1),
    ));

    let runner = bind_trusting_any_relay([24; 32], relayed).await.unwrap();
    let report = probe::run(
        &runner,
        peer,
        short(RelayChoice::Custom, false),
        Arc::new(Fixed(RUNNER)),
        |_| {},
    )
    .await
    .unwrap();
    runner.close().await;
    listener.close().await;
    serving.await.unwrap().unwrap();

    assert_clean(&report);
    assert_eq!(report.join.path_at_connect, Path::Relay);
    assert_eq!(report.join.first_direct_ms, None);
    let relayed_rtt = report
        .join
        .relayed_rtt_us
        .expect("relayed echoes have RTTs");
    assert!(relayed_rtt.samples > 0 && relayed_rtt.p50 > 0);
    assert!(!report.hops.is_empty());
    assert!(report.hops.iter().all(|hop| hop.path == Path::Relay));
    assert_eq!(report.settings.relays, RelayChoice::Custom);
    assert!(!report.settings.direct_paths);
    // The relay's URL is in no field: the report says only that a custom relay was used.
    let text = serde_json::to_string(&report).unwrap();
    assert!(!text.contains("http"), "{text}");
}

/// Kills the listener when the test ends, whatever happens.
struct Child(std::process::Child);

impl Drop for Child {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn the_binary_listens_runs_and_checks_its_report() {
    use std::io::{BufRead, BufReader};
    use std::process::{Command, Stdio};

    let binary = env!("CARGO_BIN_EXE_syntropy-netprobe");
    let mut listener = Child(
        Command::new(binary)
            .args(["listen", "--relay", "none"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap(),
    );
    let stdout = listener.0.stdout.take().unwrap();
    let ticket = BufReader::new(stdout)
        .lines()
        .map(Result::unwrap)
        .find_map(|line| {
            line.trim()
                .strip_prefix("syntropy-netprobe run ")
                .map(str::to_owned)
        })
        .expect("the listener prints a ticket");
    assert!(
        ticket.contains(",addr="),
        "a ticket with no relay names addresses"
    );

    let dir = std::env::temp_dir().join(format!("netprobe-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let out = dir.join("report.json");
    let run = Command::new(binary)
        .args(["run", &ticket, "--relay", "none", "--duration", "3s"])
        .args(["--period-ms", "100", "--out"])
        .arg(&out)
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let shown = String::from_utf8_lossy(&run.stdout);
    assert!(shown.contains("Joined in"), "{shown}");
    assert_eq!(scan(&shown), [], "the summary shows no identifier");

    let check = Command::new(binary)
        .arg("check")
        .arg(&out)
        .output()
        .unwrap();
    assert!(check.status.success());
    assert!(String::from_utf8_lossy(&check.stdout).contains("passes the schema test"));
    let report: Report = serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
    assert_eq!(report.outcome, Outcome::Completed);
    assert_eq!(report.join.path_at_connect, Path::Direct);

    // A report with an address added fails the check, and the output names no address.
    let tampered = dir.join("tampered.json");
    let text = std::fs::read_to_string(&out).unwrap().replacen(
        "\"probe_version\"",
        "\"x\": \"192.0.2.9\", \"probe_version\"",
        1,
    );
    std::fs::write(&tampered, text).unwrap();
    let check = Command::new(binary)
        .arg("check")
        .arg(&tampered)
        .output()
        .unwrap();
    assert!(!check.status.success());
    assert!(!String::from_utf8_lossy(&check.stdout).contains("192.0.2.9"));

    // An address read as an integer, 0xC0A80001, in a time field: refused, and not repeated.
    let text = std::fs::read_to_string(&out).unwrap();
    let at = text.find("\"elapsed_ms\": ").unwrap() + "\"elapsed_ms\": ".len();
    let end = at + text[at..].find(|c: char| !c.is_ascii_digit()).unwrap();
    std::fs::write(
        &tampered,
        format!("{}3232235521{}", &text[..at], &text[end..]),
    )
    .unwrap();
    let check = Command::new(binary)
        .arg("check")
        .arg(&tampered)
        .output()
        .unwrap();
    assert!(!check.status.success());
    let shown = String::from_utf8_lossy(&check.stdout);
    assert!(shown.contains("settings.elapsed_ms"), "{shown}");
    assert!(!shown.contains("3232235521"), "{shown}");
    std::fs::remove_dir_all(&dir).unwrap();
}
