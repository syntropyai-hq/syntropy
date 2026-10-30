// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo xtask ci toolchains-windows` and `cargo xtask ci toolchains-macos`: the hosted Windows
//! and macOS runners carry the pinned toolchains, and print them (RFC-0062 §4.1 and §4.2;
//! QUAL-E16-T30).
//!
//! Neither runner image carries every pinned tool, so each job fetches what the image lacks from
//! its publisher and checks the download's SHA-256 against `ci/toolchains.toml` before it runs it:
//! - Windows: the CUDA compiler (`cuda_nvcc`, one of NVIDIA's redistributable archives) and
//!   CMake (Kitware's release archive). The MSVC toolset comes with the image, and the job checks
//!   the Visual Studio version, the default toolset and the `cl` banner against the pins.
//! - macOS: CMake (Kitware's release archive). Xcode comes with the image, and the job checks its
//!   version and build, Apple clang's version and the Metal compiler's version against the pins.
//!   From Xcode 26 the Metal toolchain is a component of its own, which the image does not always
//!   hold; when `metal` reports it missing, the job fetches it with `xcodebuild
//!   -downloadComponent MetalToolchain` and asks again.
//!
//! Each job prints every version it read, then fails if any differs from its pin. The checks are
//! pure functions of what the tools printed, so the tests run them on recorded output.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use sha2::{Digest as _, Sha256};

use super::{Component, TOOLCHAINS, Toolchains, Unpinned};
use crate::run::Context;

/// NVIDIA's redistributable archives, by component, platform and version.
const NVIDIA_REDIST: &str = "https://developer.download.nvidia.com/compute/cuda/redist";

/// Kitware's CMake release archives.
const KITWARE_RELEASES: &str = "https://github.com/Kitware/CMake/releases/download";

/// The User-Agent of every download: the project, never a person.
const USER_AGENT: &str = "https://github.com/syntropyai-hq/syntropy";

/// The default install of the Visual Studio locator, on every hosted Windows image.
const VSWHERE: &str = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe";

/// What the Windows runner's tools printed.
#[derive(Debug, Default)]
pub struct WindowsPrinted {
    /// `nvcc --version`.
    pub nvcc: String,
    /// `cmake --version`.
    pub cmake: String,
    /// `vswhere -latest … -property installationVersion`.
    pub visual_studio: String,
    /// `VC\Auxiliary\Build\Microsoft.VCToolsVersion.default.txt`: the default MSVC toolset.
    pub toolset: String,
    /// `cl`'s banner, which it writes to stderr when run with no arguments.
    pub cl: String,
}

/// What the macOS runner's tools printed.
#[derive(Debug, Default)]
pub struct MacosPrinted {
    /// `cmake --version`.
    pub cmake: String,
    /// `xcodebuild -version`.
    pub xcode: String,
    /// `xcrun clang --version`.
    pub clang: String,
    /// `xcrun -sdk macosx metal --version`.
    pub metal: String,
}

/// The first line of `text` that contains `marker`.
fn line_with<'a>(text: &'a str, marker: &str) -> Option<&'a str> {
    text.lines().map(str::trim).find(|l| l.contains(marker))
}

/// A problem when `tool`'s line holding `marker` in `printed` lacks `want`.
fn expect_line(problems: &mut Vec<String>, tool: &str, printed: &str, marker: &str, want: &str) {
    match line_with(printed, marker) {
        None => problems.push(format!("`{tool}` printed no version line")),
        Some(line) if !line.contains(want) => {
            problems.push(format!(
                "`{tool}` prints `{line}`, but the pin needs `{want}`"
            ));
        }
        Some(_) => {}
    }
}

/// As [`expect_line`] for a pin that may still be `verified = false`: an unverified pin is a
/// problem that quotes what the runner printed, so the job's log is the report a later change
/// pins it from.
fn expect_pinned_line(
    problems: &mut Vec<String>,
    tool: &str,
    printed: &str,
    marker: &str,
    pin_name: &str,
    pin: &Unpinned,
    want: impl Fn(&str) -> String,
) {
    match &pin.version {
        Some(version) => expect_line(problems, tool, printed, marker, &want(version)),
        None => problems.push(format!(
            "{TOOLCHAINS}: `{pin_name}` is not pinned yet (`verified = false`); `{tool}` prints `{}`",
            line_with(printed, marker).unwrap_or("nothing")
        )),
    }
}

/// The `cuda_nvcc` component of `components`.
fn nvcc_component(components: &[Component]) -> Option<&Component> {
    components.iter().find(|c| c.name == "cuda_nvcc")
}

/// The problems in what the Windows runner printed, against `pins`.
pub fn windows_problems(printed: &WindowsPrinted, pins: &Toolchains) -> Vec<String> {
    let w = &pins.windows;
    let mut problems = Vec::new();
    match nvcc_component(&w.cuda.component) {
        None => problems.push(format!(
            "{TOOLCHAINS}: no `windows.cuda` `cuda_nvcc` component to check `nvcc` against"
        )),
        Some(nvcc) => expect_line(
            &mut problems,
            "nvcc",
            &printed.nvcc,
            "Cuda compilation tools",
            &format!(", V{}", nvcc.version),
        ),
    }
    expect_line(
        &mut problems,
        "cmake",
        &printed.cmake,
        "cmake version",
        &format!("cmake version {}", w.cmake.version),
    );
    let exact = |problems: &mut Vec<String>, what: &str, got: &str, want: &str| {
        let got = got.trim();
        if got.is_empty() {
            problems.push(format!("`{what}` printed nothing"));
        } else if got != want {
            problems.push(format!("`{what}` is `{got}`, but the pin is `{want}`"));
        }
    };
    exact(
        &mut problems,
        "Visual Studio",
        &printed.visual_studio,
        &w.msvc.visual_studio,
    );
    match &w.msvc.toolset.version {
        Some(toolset) => exact(&mut problems, "the MSVC toolset", &printed.toolset, toolset),
        None => problems.push(format!(
            "{TOOLCHAINS}: `windows.msvc.toolset` is not pinned yet (`verified = false`); the \
             default toolset is `{}`",
            printed.toolset.trim()
        )),
    }
    expect_pinned_line(
        &mut problems,
        "cl",
        &printed.cl,
        "Compiler Version",
        "windows.msvc.cl",
        &w.msvc.cl,
        |v| format!("Compiler Version {v} for x64"),
    );
    problems
}

/// The problems in what the macOS runner printed, against `pins`.
pub fn macos_problems(printed: &MacosPrinted, pins: &Toolchains) -> Vec<String> {
    let m = &pins.macos;
    let mut problems = Vec::new();
    expect_line(
        &mut problems,
        "cmake",
        &printed.cmake,
        "cmake version",
        &format!("cmake version {}", m.cmake.version),
    );
    // `xcodebuild -version` prints `Xcode <version>`, then `Build version <build>`.
    match line_with(&printed.xcode, "Xcode ") {
        None => problems.push("`xcodebuild` printed no version line".to_owned()),
        Some(line) if line != format!("Xcode {}", m.xcode.version) => problems.push(format!(
            "`xcodebuild` prints `{line}`, but the pin needs `Xcode {}`",
            m.xcode.version
        )),
        Some(_) => {}
    }
    match line_with(&printed.xcode, "Build version") {
        None => problems.push("`xcodebuild` printed no build line".to_owned()),
        Some(line) if line != format!("Build version {}", m.xcode.build) => {
            problems.push(format!(
                "`xcodebuild` prints `{line}`, but the pin needs `Build version {}`",
                m.xcode.build
            ));
        }
        Some(_) => {}
    }
    expect_pinned_line(
        &mut problems,
        "clang",
        &printed.clang,
        "clang version",
        "macos.xcode.clang",
        &m.xcode.clang,
        |v| format!("Apple clang version {v}"),
    );
    expect_pinned_line(
        &mut problems,
        "metal",
        &printed.metal,
        "metal version",
        "macos.metal",
        &m.metal,
        |v| format!("Apple metal version {v}"),
    );
    problems
}

/// The lowercase hex SHA-256 of `bytes`.
fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Runs `program` with `args` and returns its stdout and stderr, or why it could not start.
fn output(program: &Path, args: &[&str]) -> Result<(bool, String, String), String> {
    let out = Command::new(program)
        .args(args)
        .output()
        .map_err(|e| format!("could not start `{}`: {e}", program.display()))?;
    Ok((
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    ))
}

/// Runs `program` with `args` and returns its stdout, failing on a non-zero exit.
fn stdout_of(program: &Path, args: &[&str]) -> Result<String, String> {
    let (ok, stdout, stderr) = output(program, args)?;
    if ok {
        Ok(stdout)
    } else {
        Err(format!(
            "`{} {}` failed: {stderr}",
            program.display(),
            args.join(" ")
        ))
    }
}

/// Downloads `url` into `work` and unpacks it there, after checking its SHA-256 against
/// `sha256`. Returns the directory it unpacked into.
fn fetch_and_unpack(work: &Path, url: &str, sha256: &str) -> Result<PathBuf, String> {
    let name = url.rsplit('/').next().unwrap_or("download");
    let archive = work.join(name);
    let status = Command::new("curl")
        .args(["-sSfL", "--retry", "3", "-A", USER_AGENT, "-o"])
        .arg(&archive)
        .arg(url)
        .status()
        .map_err(|e| format!("could not start `curl`: {e}"))?;
    if !status.success() {
        return Err(format!("downloading {url} failed: {status}"));
    }
    let bytes = fs::read(&archive).map_err(|e| format!("{}: {e}", archive.display()))?;
    let got = sha256_hex(&bytes);
    if got != sha256 {
        return Err(format!(
            "{url} has SHA-256 {got}, but {TOOLCHAINS} pins {sha256}"
        ));
    }
    eprintln!("toolchains: {name}: SHA-256 matches the pin");
    let dir = work.join(format!("{name}.d"));
    fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    // `tar` is bsdtar on both runners, which unpacks zip archives too.
    let status = Command::new("tar")
        .arg("-xf")
        .arg(&archive)
        .arg("-C")
        .arg(&dir)
        .status()
        .map_err(|e| format!("could not start `tar`: {e}"))?;
    if !status.success() {
        return Err(format!("unpacking {name} failed: {status}"));
    }
    Ok(dir)
}

/// A scratch directory for one job's downloads, removed by the caller.
fn work_dir(job: &str) -> Result<PathBuf, String> {
    let work = std::env::temp_dir().join(format!("syntropy-{job}-{}", std::process::id()));
    fs::create_dir_all(&work).map_err(|e| format!("{}: {e}", work.display()))?;
    Ok(work)
}

/// Reads the pins.
fn pins(context: &Context) -> Result<Toolchains, String> {
    let text = fs::read_to_string(context.root.join(TOOLCHAINS))
        .map_err(|e| format!("{TOOLCHAINS}: {e}"))?;
    toml::from_str(&text).map_err(|e| format!("{TOOLCHAINS}: {e}"))
}

/// Prints `printed`'s fields, then passes or lists `problems`.
fn verdict(job: &str, printed: &[(&str, &str)], problems: Vec<String>) -> Result<(), String> {
    for (tool, text) in printed {
        eprintln!("--- {tool}\n{}", text.trim_end());
    }
    if problems.is_empty() {
        eprintln!("{job}: every printed version matches {TOOLCHAINS}");
        Ok(())
    } else {
        Err(problems.join("\n"))
    }
}

/// `cargo xtask ci toolchains-windows`.
pub fn windows_step(context: &Context) -> Result<(), String> {
    let pins = pins(context)?;
    let work = work_dir("toolchains-windows")?;
    let printed = windows_print(&pins, &work);
    fs::remove_dir_all(&work).ok();
    let printed = printed?;
    let problems = windows_problems(&printed, &pins);
    verdict(
        "toolchains-windows",
        &[
            ("nvcc --version", &printed.nvcc),
            ("cmake --version", &printed.cmake),
            (
                "Visual Studio (vswhere installationVersion)",
                &printed.visual_studio,
            ),
            (
                "MSVC toolset (Microsoft.VCToolsVersion.default.txt)",
                &printed.toolset,
            ),
            ("cl", &printed.cl),
        ],
        problems,
    )
}

fn windows_print(pins: &Toolchains, work: &Path) -> Result<WindowsPrinted, String> {
    let w = &pins.windows;
    let nvcc = nvcc_component(&w.cuda.component)
        .ok_or_else(|| format!("{TOOLCHAINS}: no `windows.cuda` `cuda_nvcc` component"))?;
    let stem = format!("cuda_nvcc-windows-x86_64-{}-archive", nvcc.version);
    let dir = fetch_and_unpack(
        work,
        &format!("{NVIDIA_REDIST}/cuda_nvcc/windows-x86_64/{stem}.zip"),
        &nvcc.sha256,
    )?;
    let nvcc_out = stdout_of(
        &dir.join(&stem).join("bin").join("nvcc.exe"),
        &["--version"],
    )?;

    let stem = format!("cmake-{}-windows-x86_64", w.cmake.version);
    let dir = fetch_and_unpack(
        work,
        &format!("{KITWARE_RELEASES}/v{}/{stem}.zip", w.cmake.version),
        &w.cmake.sha256,
    )?;
    let cmake = stdout_of(
        &dir.join(&stem).join("bin").join("cmake.exe"),
        &["--version"],
    )?;

    let vswhere = |property: &str| {
        stdout_of(
            Path::new(VSWHERE),
            &[
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                property,
            ],
        )
        .map(|s| s.trim().to_owned())
    };
    let visual_studio = vswhere("installationVersion")?;
    let install = PathBuf::from(vswhere("installationPath")?);
    let default = install
        .join(r"VC\Auxiliary\Build")
        .join("Microsoft.VCToolsVersion.default.txt");
    let toolset = fs::read_to_string(&default)
        .map_err(|e| format!("{}: {e}", default.display()))?
        .trim()
        .to_owned();
    let cl = install
        .join(r"VC\Tools\MSVC")
        .join(&toolset)
        .join(r"bin\Hostx64\x64\cl.exe");
    // `cl` with no arguments prints its banner to stderr and a usage line.
    let (_, cl_out, cl_err) = output(&cl, &[])?;
    Ok(WindowsPrinted {
        nvcc: nvcc_out,
        cmake,
        visual_studio,
        toolset,
        cl: format!("{cl_err}{cl_out}"),
    })
}

/// `cargo xtask ci toolchains-macos`.
pub fn macos_step(context: &Context) -> Result<(), String> {
    let pins = pins(context)?;
    let work = work_dir("toolchains-macos")?;
    let printed = macos_print(&pins, &work);
    fs::remove_dir_all(&work).ok();
    let printed = printed?;
    let problems = macos_problems(&printed, &pins);
    verdict(
        "toolchains-macos",
        &[
            ("cmake --version", &printed.cmake),
            ("xcodebuild -version", &printed.xcode),
            ("xcrun clang --version", &printed.clang),
            ("xcrun -sdk macosx metal --version", &printed.metal),
        ],
        problems,
    )
}

fn macos_print(pins: &Toolchains, work: &Path) -> Result<MacosPrinted, String> {
    let m = &pins.macos;
    let stem = format!("cmake-{}-macos-universal", m.cmake.version);
    let dir = fetch_and_unpack(
        work,
        &format!("{KITWARE_RELEASES}/v{}/{stem}.tar.gz", m.cmake.version),
        &m.cmake.sha256,
    )?;
    let cmake = stdout_of(
        &dir.join(&stem).join("CMake.app/Contents/bin/cmake"),
        &["--version"],
    )?;
    let xcode = stdout_of(Path::new("xcodebuild"), &["-version"])?;
    let clang = stdout_of(Path::new("xcrun"), &["clang", "--version"])?;
    let metal_args = ["-sdk", "macosx", "metal", "--version"];
    let (ok, out, err) = output(Path::new("xcrun"), &metal_args)?;
    let metal = if ok {
        out
    } else if err.contains("missing Metal Toolchain") {
        eprintln!("toolchains-macos: the image lacks the Metal toolchain; fetching it");
        stdout_of(
            Path::new("xcodebuild"),
            &["-downloadComponent", "MetalToolchain"],
        )?;
        stdout_of(Path::new("xcrun"), &metal_args)?
    } else {
        return Err(format!("`xcrun {}` failed: {err}", metal_args.join(" ")));
    };
    Ok(MacosPrinted {
        cmake,
        xcode,
        clang,
        metal,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pins() -> Toolchains {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let text = fs::read_to_string(root.join(TOOLCHAINS)).expect("pins");
        toml::from_str(&text).expect("pins parse")
    }

    /// Output in each tool's format, built from the pins (`pin` stands in for an unverified
    /// one): the checks, not the versions, are under test.
    fn windows(p: &Toolchains, pin: &str) -> WindowsPrinted {
        let w = &p.windows;
        let nvcc = nvcc_component(&w.cuda.component).expect("nvcc");
        WindowsPrinted {
            nvcc: format!(
                "nvcc: NVIDIA (R) Cuda compiler driver\nCuda compilation tools, release 13.4, V{}\n",
                nvcc.version
            ),
            cmake: format!(
                "cmake version {}\n\nCMake suite maintained and supported by Kitware.\n",
                w.cmake.version
            ),
            visual_studio: format!("{}\r\n", w.msvc.visual_studio),
            toolset: format!("{}\r\n", w.msvc.toolset.version.as_deref().unwrap_or(pin)),
            cl: format!(
                "Microsoft (R) C/C++ Optimizing Compiler Version {} for x64\nusage: cl [ option... ] filename... [ /link linkoption... ]\n",
                w.msvc.cl.version.as_deref().unwrap_or(pin)
            ),
        }
    }

    fn macos(p: &Toolchains, pin: &str) -> MacosPrinted {
        let m = &p.macos;
        MacosPrinted {
            cmake: format!("cmake version {}\n", m.cmake.version),
            xcode: format!(
                "Xcode {}\nBuild version {}\n",
                m.xcode.version, m.xcode.build
            ),
            clang: format!(
                "Apple clang version {}\nTarget: arm64-apple-darwin25.6.0\n",
                m.xcode.clang.version.as_deref().unwrap_or(pin)
            ),
            metal: format!(
                "Apple metal version {}\nTarget: air64-apple-darwin25.6.0\n",
                m.metal.version.as_deref().unwrap_or(pin)
            ),
        }
    }

    /// The problems that are not about a pin still `verified = false`.
    fn real(problems: Vec<String>) -> Vec<String> {
        problems
            .into_iter()
            .filter(|p| !p.contains("not pinned yet"))
            .collect()
    }

    #[test]
    fn printed_versions_that_match_the_pins_pass() {
        let p = pins();
        assert_eq!(
            real(windows_problems(&windows(&p, "X"), &p)),
            Vec::<String>::new()
        );
        assert_eq!(
            real(macos_problems(&macos(&p, "X"), &p)),
            Vec::<String>::new()
        );
    }

    #[test]
    fn an_unverified_pin_fails_and_quotes_what_the_runner_printed() {
        let p = pins();
        let problems = macos_problems(&macos(&p, "9.9.9 (stand-in)"), &p);
        if p.macos.metal.version.is_none() {
            assert!(
                problems
                    .iter()
                    .any(|x| x.contains("macos.metal") && x.contains("Apple metal version 9.9.9")),
                "{problems:?}"
            );
        } else {
            assert!(
                problems.iter().all(|x| !x.contains("macos.metal")),
                "{problems:?}"
            );
        }
    }

    #[test]
    fn a_windows_version_that_differs_from_its_pin_is_named() {
        let p = pins();
        let nvcc = nvcc_component(&p.windows.cuda.component).unwrap();
        let mut printed = windows(&p, "X");
        printed.nvcc = printed.nvcc.replace(&nvcc.version, "0.0.0");
        printed.cmake = printed.cmake.replace(&p.windows.cmake.version, "3.31.6");
        printed.visual_studio = "17.14.1.1".into();
        let problems = real(windows_problems(&printed, &p));
        for tool in ["nvcc", "cmake", "Visual Studio"] {
            assert!(
                problems.iter().any(|x| x.contains(tool)),
                "{tool}: {problems:?}"
            );
        }
        assert_eq!(problems.len(), 3, "{problems:?}");
    }

    #[test]
    fn a_macos_version_that_differs_from_its_pin_is_named() {
        let p = pins();
        let mut printed = macos(&p, "X");
        printed.xcode = printed.xcode.replace(&p.macos.xcode.build, "17F42");
        printed.cmake = String::new();
        let problems = real(macos_problems(&printed, &p));
        assert!(
            problems.iter().any(|x| x.contains("Build version")),
            "{problems:?}"
        );
        assert!(
            problems.contains(&"`cmake` printed no version line".to_owned()),
            "{problems:?}"
        );
        assert_eq!(problems.len(), 2, "{problems:?}");
        // A newer Xcode whose version starts with the pin is still a different version.
        let mut printed = macos(&p, "X");
        printed.xcode = printed.xcode.replace(
            &format!("Xcode {}", p.macos.xcode.version),
            &format!("Xcode {}.1", p.macos.xcode.version),
        );
        assert!(
            real(macos_problems(&printed, &p))
                .iter()
                .any(|x| x.contains("`Xcode "))
        );
    }

    #[test]
    fn the_sha256_is_lowercase_hex() {
        // FIPS 180-2's "abc" vector.
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
