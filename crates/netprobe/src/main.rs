// SPDX-License-Identifier: MIT OR Apache-2.0

//! `syntropy-netprobe`: the field network probe (DIST-E4-T15). See the library's documentation,
//! and `syntropy-netprobe help`.

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use syntropy_core::Entropy;
use syntropy_io::os::OsEntropy;
use syntropy_net::endpoint::{Config, Endpoint, Relays};
use syntropy_netprobe::platform::{Detect, Platform};
use syntropy_netprobe::probe::{self, RunOptions};
use syntropy_netprobe::report::{
    MAX_DURATION_S, Outcome, RelayChoice, Report, Setting, check_text,
};
use syntropy_netprobe::wire::ALPN;
use syntropy_netprobe::{scan, ticket};

const USAGE: &str = "\
syntropy-netprobe: measures the network between two of your machines.

On the first machine:
    syntropy-netprobe listen [--relay public|none|<url>] [--no-direct]
It prints a ticket. On the second machine:
    syntropy-netprobe run <ticket> [--duration 24h] [--out <file>] [--off-lan]
                                   [--relay public|none|<url>] [--no-direct] [--period-ms 1000]
It sends a small echo request every period for the duration, then writes the report: by default
syntropy-netprobe-<time>.json in the current directory. It also rewrites the file every minute,
so an interrupted run leaves a report.

To check a report before you submit it:
    syntropy-netprobe check <file>

    --relay      public (the default): iroh's public relays and n0's address lookup service,
                 for alpha testing only. Both see this machine's node ID and IP address. none: no
                 relay and no lookup, so the machines must reach each other directly. Or a relay
                 URL, with no lookup.
    --no-direct  no direct paths: every packet goes through the relay.
    --off-lan    say that this run is a test join, with one machine moved to another network.
    --duration   how long to measure: a number with s, m or h, such as 90s, 30m or 24h; at
                 most 7 days.

The ticket holds this machine's addresses, so share it only between your own machines. The report
holds no IP address, MAC address or network name: `check` scans for them.
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("listen") => runtime().block_on(listen(&args[1..])),
        Some("run") => runtime().block_on(run(&args[1..])),
        Some("check") => check(&args[1..]),
        Some("help" | "--help" | "-h") | None => {
            print!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Some(other) => Err(Failure::Usage(format!("unknown command `{other}`"))),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(Failure::Usage(message)) => {
            eprintln!("syntropy-netprobe: {message}\n\n{USAGE}");
            ExitCode::from(2)
        }
        Err(Failure::Run(message)) => {
            eprintln!("syntropy-netprobe: {message}");
            ExitCode::FAILURE
        }
    }
}

enum Failure {
    Usage(String),
    Run(String),
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("the async runtime starts")
}

/// The options shared by `listen` and `run`, and the arguments left over.
struct Common {
    relays: Relays,
    choice: RelayChoice,
    direct_paths: bool,
    rest: Vec<String>,
}

fn common(args: &[String]) -> Result<Common, Failure> {
    let mut parsed = Common {
        relays: Relays::Public,
        choice: RelayChoice::Public,
        direct_paths: true,
        rest: Vec::new(),
    };
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--relay" => {
                let value = args
                    .next()
                    .ok_or_else(|| Failure::Usage("--relay needs a value".to_owned()))?;
                (parsed.relays, parsed.choice) = match value.as_str() {
                    "public" => (Relays::Public, RelayChoice::Public),
                    "none" => (Relays::Disabled, RelayChoice::Disabled),
                    url => (
                        Relays::Url(
                            url.parse()
                                .map_err(|error| Failure::Usage(format!("--relay: {error}")))?,
                        ),
                        RelayChoice::Custom,
                    ),
                };
            }
            "--no-direct" => parsed.direct_paths = false,
            _ => parsed.rest.push(arg.clone()),
        }
    }
    Ok(parsed)
}

async fn bind(common: &Common, accept: bool) -> Result<Endpoint, Failure> {
    let mut key = [0u8; 32];
    OsEntropy.fill(&mut key);
    let mut config = Config::default()
        .relays(common.relays.clone())
        .direct_paths(common.direct_paths);
    if accept {
        config = config.accept_alpn(ALPN);
    }
    Endpoint::bind(key, config)
        .await
        .map_err(|error| Failure::Run(error.to_string()))
}

async fn listen(args: &[String]) -> Result<(), Failure> {
    let common = common(args)?;
    if let Some(extra) = common.rest.first() {
        return Err(Failure::Usage(format!("unexpected `{extra}`")));
    }
    let endpoint = bind(&common, true).await?;
    if common.choice != RelayChoice::Disabled
        && tokio::time::timeout(Duration::from_secs(15), endpoint.online())
            .await
            .is_err()
    {
        eprintln!("syntropy-netprobe: the relay did not answer within 15 s; listening anyway");
    }
    // Give the endpoint a moment to find its direct addresses.
    tokio::time::sleep(Duration::from_secs(1)).await;
    println!(
        "Listening. On the other machine, run:\n\n    syntropy-netprobe run {}\n\n\
         Leave this running until that run ends. Press Ctrl-C to stop.",
        ticket::encode(&endpoint.addr())
    );
    probe::serve(endpoint, Arc::new(Platform), Duration::from_secs(60))
        .await
        .map_err(|error| Failure::Run(error.to_string()))
}

fn parse_duration(text: &str) -> Option<Duration> {
    let (number, unit) = match text.char_indices().last()? {
        (at, 's') => (&text[..at], 1),
        (at, 'm') => (&text[..at], 60),
        (at, 'h') => (&text[..at], 3_600),
        _ => (text, 1),
    };
    let seconds = number.parse::<u64>().ok()?.checked_mul(unit)?;
    (1..=MAX_DURATION_S)
        .contains(&seconds)
        .then(|| Duration::from_secs(seconds))
}

async fn run(args: &[String]) -> Result<(), Failure> {
    let common = common(args)?;
    let mut ticket_text = None;
    let mut out = None;
    let mut setting = Setting::Lan;
    let mut options = RunOptions::new(Setting::Lan, common.choice, common.direct_paths);
    let mut rest = common.rest.iter();
    while let Some(arg) = rest.next() {
        let mut value = |name: &str| {
            rest.next()
                .cloned()
                .ok_or_else(|| Failure::Usage(format!("{name} needs a value")))
        };
        match arg.as_str() {
            "--duration" => {
                let text = value("--duration")?;
                options.duration = parse_duration(&text)
                    .ok_or_else(|| Failure::Usage(format!("--duration: cannot read `{text}`")))?;
            }
            "--out" => out = Some(PathBuf::from(value("--out")?)),
            "--off-lan" => setting = Setting::OffLan,
            "--period-ms" => {
                let text = value("--period-ms")?;
                let ms: u64 = text
                    .parse()
                    .ok()
                    .filter(|ms| (10..=1_000).contains(ms))
                    .ok_or_else(|| Failure::Usage("--period-ms takes 10 to 1000".to_owned()))?;
                options.period = Duration::from_millis(ms);
            }
            flag if flag.starts_with("--") => {
                return Err(Failure::Usage(format!("unknown option `{flag}`")));
            }
            text if ticket_text.is_none() => ticket_text = Some(text.to_owned()),
            extra => return Err(Failure::Usage(format!("unexpected `{extra}`"))),
        }
    }
    options.setting = setting;
    let ticket_text = ticket_text.ok_or_else(|| Failure::Usage("run needs a ticket".to_owned()))?;
    let peer = ticket::decode(&ticket_text).map_err(|error| Failure::Usage(error.to_string()))?;
    let out = out.unwrap_or_else(default_out);

    let endpoint = bind(&common, false).await?;
    println!(
        "Measuring for {} s; the report goes to {}.",
        options.duration.as_secs(),
        out.display()
    );
    let detect: Arc<dyn Detect> = Arc::new(Platform);
    let mut write_error = None;
    let result = probe::run(&endpoint, peer, options, detect, |report| {
        if let Err(error) = write_report(&out, report) {
            write_error.get_or_insert(error);
        }
    })
    .await;
    endpoint.close().await;
    if let Some(error) = write_error {
        return Err(Failure::Run(error));
    }
    let report = result.map_err(|error| Failure::Run(error.to_string()))?;
    write_report(&out, &report).map_err(Failure::Run)?;
    print_summary(&report);
    println!(
        "\nThe report is {}. Read it, then submit it.",
        out.display()
    );
    Ok(())
}

fn default_out() -> PathBuf {
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |since| since.as_secs());
    PathBuf::from(format!("syntropy-netprobe-{seconds}.json"))
}

/// Checks `report` and writes it to `path` through a temporary file, so a reader never sees half
/// a report. A report the schema test or the identifier scan refuses is not written.
fn write_report(path: &Path, report: &Report) -> Result<(), String> {
    let text = serde_json::to_string_pretty(report).map_err(|error| error.to_string())?;
    let problems = report.check();
    if !problems.is_empty() {
        return Err(format!(
            "the report failed its schema test: {}",
            problems.join("; ")
        ));
    }
    if !scan::scan(&text).is_empty() {
        return Err("the report would hold an identifier, so it was not written".to_owned());
    }
    let temporary = path.with_extension("json.tmp");
    std::fs::write(&temporary, text + "\n")
        .and_then(|()| std::fs::rename(&temporary, path))
        .map_err(|error| format!("cannot write {}: {error}", path.display()))
}

fn print_summary(report: &Report) {
    println!(
        "\nJoined in {} ms on a {:?} path.",
        report.join.connect_ms, report.join.path_at_connect
    );
    for hop in &report.hops {
        let rtt = hop.rtt_us.map_or("no echoes".to_owned(), |rtt| {
            format!(
                "RTT p50 {} µs, p99 {} µs, p99.9 {} µs",
                rtt.p50, rtt.p99, rtt.p999
            )
        });
        let remote = hop
            .remote_medium
            .map_or("unknown".to_owned(), |medium| format!("{medium:?}"));
        println!(
            "{:?} to {remote}, {:?}: {} requests, {} lost; {rtt}",
            hop.local_medium, hop.path, hop.beats, hop.lost
        );
    }
    println!(
        "{} stalls. LAN timings would have raised {} suspicions and {} failures; WAN timings {} \
         and {}.",
        report.stalls.log.len(),
        report.timings.lan.would_suspect,
        report.timings.lan.would_fail,
        report.timings.wan.would_suspect,
        report.timings.wan.would_fail
    );
    if report.outcome != Outcome::Completed {
        println!("The run did not complete.");
    }
}

fn check(args: &[String]) -> Result<(), Failure> {
    let [path] = args else {
        return Err(Failure::Usage("check takes one file".to_owned()));
    };
    let text = std::fs::read_to_string(path)
        .map_err(|error| Failure::Run(format!("cannot read {path}: {error}")))?;
    // The problems name places and rules only, never what the report holds.
    if let Err(problems) = check_text(&text) {
        for problem in problems {
            println!("{problem}");
        }
        return Err(Failure::Run(format!("{path} failed its checks")));
    }
    println!("{path}: passes the schema test and the identifier scan");
    Ok(())
}
