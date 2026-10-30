// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `toolchains` step of `lint`: the build toolchain pins are well formed and agree
//! (QUAL-E16-T24; RFC-0062 §4.2 and §4.3).
//!
//! `ci/toolchains.toml` repeats the pins of the Linux build image, `ci/images/cuda-13/Dockerfile`.
//! This step fails unless:
//! - `ci/toolchains.toml` parses with no unknown key, every checksum is 64 lowercase hex digits,
//!   every pin is either set or marked `verified = false` with no value, and the BuildKit image
//!   is pinned by digest;
//! - each Linux pin appears in the Dockerfile: the base image, the syntax frontend, the archive
//!   snapshot, the epoch, the compiler version, the CMake version and checksum, and the CUDA
//!   release and each component's checksum;
//! - `ci/arch-sets.toml` parses, and its `pr` profile lists `sm_89` (R-ADA, the M0 rig) and no
//!   PTX.
//!
//! `cargo xtask ci toolchains-linux` builds the image and prints its tools here; the Windows and
//! macOS jobs, which check the hosted runners' tools, are in [`hosted`] (QUAL-E16-T30).

use std::fs;
use std::path::Path;
use std::process::Command;

use serde::Deserialize;

use crate::run::Context;

mod hosted;

pub use hosted::{macos_step, windows_step};

/// The pins file, relative to the workspace root.
pub const TOOLCHAINS: &str = "ci/toolchains.toml";

/// The architecture sets file, relative to the workspace root.
pub const ARCH_SETS: &str = "ci/arch-sets.toml";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Toolchains {
    format: u32,
    linux: Linux,
    windows: Windows,
    macos: Macos,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Linux {
    image: Image,
    compiler: Compiler,
    cmake: Cmake,
    cuda: Cuda,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Image {
    dockerfile: String,
    base: String,
    syntax: String,
    apt_snapshot: String,
    source_date_epoch: u64,
    glibc: String,
    buildkit: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Compiler {
    gxx: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Cmake {
    version: String,
    sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Cuda {
    release: String,
    component: Vec<Component>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Component {
    name: String,
    version: String,
    sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Windows {
    cmake: Cmake,
    cuda: Cuda,
    msvc: Msvc,
}

/// The MSVC toolchain of the hosted Windows image.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Msvc {
    /// Visual Studio's `installationVersion`, as the image report lists it.
    visual_studio: String,
    /// The default MSVC toolset, `Microsoft.VCToolsVersion.default.txt`.
    toolset: Unpinned,
    /// The version in `cl`'s banner.
    cl: Unpinned,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Macos {
    cmake: Cmake,
    xcode: Xcode,
    metal: Unpinned,
}

/// The default Xcode of the hosted macOS image.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Xcode {
    /// As `xcodebuild -version` and the image report give it.
    version: String,
    build: String,
    /// What follows `Apple clang version` in `xcrun clang --version`.
    clang: Unpinned,
}

/// A pin that only the hosted runner's report can fill in.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Unpinned {
    verified: bool,
    version: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArchSets {
    format: u32,
    profile: Profiles,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Profiles {
    pr: Profile,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Profile {
    cuda_sass: Vec<String>,
    cuda_ptx: Vec<String>,
}

fn is_sha256(text: &str) -> bool {
    text.len() == 64 && text.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

fn is_arch(text: &str, prefix: &str) -> bool {
    text.strip_prefix(prefix).is_some_and(|rest| {
        let digits = rest.trim_end_matches(['a', 'f']);
        !digits.is_empty()
            && digits.bytes().all(|b| b.is_ascii_digit())
            && rest.len() - digits.len() <= 1
    })
}

/// Every problem in `toolchains` against `dockerfile`, and in `arch_sets`.
pub fn check(toolchains: &str, dockerfile: &str, arch_sets: &str) -> Vec<String> {
    let mut problems = Vec::new();
    match toml::from_str::<Toolchains>(toolchains) {
        Err(error) => problems.push(format!("{TOOLCHAINS}: {error}")),
        Ok(t) => {
            if t.format != 1 {
                problems.push(format!("{TOOLCHAINS}: `format` is {}, not 1", t.format));
            }
            let mut checksums: Vec<(&str, &str)> = vec![
                ("linux.cmake", &t.linux.cmake.sha256),
                ("windows.cmake", &t.windows.cmake.sha256),
                ("macos.cmake", &t.macos.cmake.sha256),
            ];
            for (os, cuda) in [("linux", &t.linux.cuda), ("windows", &t.windows.cuda)] {
                if cuda.release != t.linux.cuda.release {
                    problems.push(format!(
                        "{TOOLCHAINS}: {os}.cuda.release differs from linux's"
                    ));
                }
                checksums.extend(
                    cuda.component
                        .iter()
                        .map(|c| (c.name.as_str(), c.sha256.as_str())),
                );
            }
            for (name, sum) in checksums {
                if !is_sha256(sum) {
                    problems.push(format!(
                        "{TOOLCHAINS}: `{name}` checksum is not 64 lowercase hex digits"
                    ));
                }
            }
            for (name, pin) in [
                ("windows.msvc.toolset", &t.windows.msvc.toolset),
                ("windows.msvc.cl", &t.windows.msvc.cl),
                ("macos.xcode.clang", &t.macos.xcode.clang),
                ("macos.metal", &t.macos.metal),
            ] {
                if pin.verified == pin.version.is_none() {
                    problems.push(format!(
                        "{TOOLCHAINS}: `{name}` needs a `version` when verified, and none when not"
                    ));
                }
            }
            let i = &t.linux.image;
            if !i
                .buildkit
                .split_once("@sha256:")
                .is_some_and(|(_, digest)| is_sha256(digest))
            {
                problems.push(format!(
                    "{TOOLCHAINS}: `linux.image.buildkit` is not pinned by a sha256 digest"
                ));
            }
            let mut wants: Vec<(String, String)> = vec![
                ("base".into(), i.base.clone()),
                ("syntax".into(), i.syntax.clone()),
                (
                    "apt_snapshot".into(),
                    format!("APT_SNAPSHOT={}", i.apt_snapshot),
                ),
                (
                    "source_date_epoch".into(),
                    format!("SOURCE_DATE_EPOCH={}", i.source_date_epoch),
                ),
                (
                    "compiler.gxx".into(),
                    format!("g++-11={}", t.linux.compiler.gxx),
                ),
                (
                    "cmake.version".into(),
                    format!("cmake-{}-linux", t.linux.cmake.version),
                ),
                ("cmake.sha256".into(), t.linux.cmake.sha256.clone()),
                (
                    "cuda.release".into(),
                    format!(
                        "CUDA_PATH=/usr/local/cuda-{}",
                        t.linux
                            .cuda
                            .release
                            .rsplit_once('.')
                            .map_or("", |(minor, _)| minor)
                    ),
                ),
            ];
            for c in &t.linux.cuda.component {
                wants.push((format!("cuda {}", c.name), format!("sha256:{}", c.sha256)));
                wants.push((
                    format!("cuda {} version", c.name),
                    format!("-{}-archive", c.version),
                ));
            }
            // Only the Dockerfile's instructions and its syntax line count: a pin named in a comment is
            // not applied.
            let instructions: String = dockerfile
                .lines()
                .filter(|l| !l.trim_start().starts_with('#') || l.starts_with("# syntax="))
                .collect::<Vec<_>>()
                .join("\n");
            for (what, text) in wants {
                if !instructions.contains(&text) {
                    problems.push(format!(
                        "{}: does not carry the pin `{what}` ({text})",
                        i.dockerfile
                    ));
                }
            }
            // The base's Ubuntu release fixes the glibc; the header comment names the floor.
            if !dockerfile.contains(&format!("glibc {}", i.glibc)) {
                problems.push(format!("{}: does not name glibc {}", i.dockerfile, i.glibc));
            }
        }
    }
    match toml::from_str::<ArchSets>(arch_sets) {
        Err(error) => problems.push(format!("{ARCH_SETS}: {error}")),
        Ok(a) => {
            if a.format != 1 {
                problems.push(format!("{ARCH_SETS}: `format` is {}, not 1", a.format));
            }
            let pr = &a.profile.pr;
            if !pr.cuda_sass.iter().any(|s| s == "sm_89") {
                problems.push(format!("{ARCH_SETS}: profile `pr` does not list `sm_89`"));
            }
            if !pr.cuda_ptx.is_empty() {
                problems.push(format!(
                    "{ARCH_SETS}: profile `pr` lists PTX (RFC-0062 §4.3: none)"
                ));
            }
            for s in &pr.cuda_sass {
                if !is_arch(s, "sm_") {
                    problems.push(format!("{ARCH_SETS}: `{s}` is not an `sm_<n>` name"));
                }
            }
        }
    }
    problems
}

/// The `toolchains` lint step.
pub fn step(context: &Context) -> Result<(), String> {
    let read = |path: &str| -> Result<String, String> {
        fs::read_to_string(Path::new(&context.root).join(path)).map_err(|e| format!("{path}: {e}"))
    };
    let toolchains = read(TOOLCHAINS)?;
    let dockerfile_path = toml::from_str::<toml::Table>(&toolchains)
        .ok()
        .and_then(|t| {
            t.get("linux")?
                .get("image")?
                .get("dockerfile")?
                .as_str()
                .map(str::to_owned)
        })
        .unwrap_or_else(|| "ci/images/cuda-13/Dockerfile".to_owned());
    let problems = check(&toolchains, &read(&dockerfile_path)?, &read(ARCH_SETS)?);
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("\n"))
    }
}

/// The problems in `printed`, the output of `nvcc --version`, then the first lines of `cmake
/// --version`, `g++ --version` and `ldd --version`, against `pins`: each pinned version must be
/// in the line of its tool.
fn version_problems(printed: &str, pins: &Toolchains) -> Vec<String> {
    let nvcc = pins
        .linux
        .cuda
        .component
        .iter()
        .find(|c| c.name == "cuda_nvcc")
        .map(|c| c.version.as_str());
    let wants = [
        (
            "nvcc",
            "Cuda compilation tools",
            nvcc.map(|v| format!(", V{v}")),
        ),
        (
            "cmake",
            "cmake version",
            Some(format!("cmake version {}", pins.linux.cmake.version)),
        ),
        (
            "g++",
            "g++ (",
            Some(format!(
                ") {}",
                pins.linux.compiler.gxx.split('-').next().unwrap_or("")
            )),
        ),
        (
            "glibc",
            "ldd (",
            Some(format!(") {}", pins.linux.image.glibc)),
        ),
    ];
    let mut problems = Vec::new();
    for (tool, marker, want) in wants {
        let Some(want) = want else {
            problems.push(format!(
                "{TOOLCHAINS}: no `cuda_nvcc` component to check `{tool}` against"
            ));
            continue;
        };
        match printed.lines().find(|l| l.contains(marker)) {
            None => problems.push(format!("`{tool}` printed no version line")),
            Some(line) if !line.contains(&want) => {
                problems.push(format!(
                    "`{tool}` prints `{line}`, but the pin needs `{want}`"
                ));
            }
            Some(_) => {}
        }
    }
    if let Some(line) = printed.lines().find(|l| l.contains("g++ (")) {
        let full = &pins.linux.compiler.gxx;
        if !line.contains(full) {
            problems.push(format!("`g++` prints `{line}`, but the pin needs `{full}`"));
        }
    }
    problems
}

/// The build, twice, and the tools it must carry: `cargo xtask ci toolchains-linux`.
///
/// Builds the Linux image twice with no cache, as the Dockerfile's header says, and fails unless
/// the two image tarballs are byte-identical. Then it loads the first and prints `nvcc`, CMake and
/// the C++ compiler from inside it (RFC-0062 §4.1; QUAL-E16-T24). Needs `docker buildx`.
///
/// The builds run on a `docker-container` builder of its own, on the pinned BuildKit image
/// (`linux.image.buildkit`), because buildx's default `docker` driver cannot export an image
/// tarball. The step creates it, leaves the default builder as it was, and removes it at the
/// end, so a local run and the hosted job build the same way.
pub fn linux_step(context: &Context) -> Result<(), String> {
    let toolchains = fs::read_to_string(context.root.join(TOOLCHAINS))
        .map_err(|e| format!("{TOOLCHAINS}: {e}"))?;
    let pins: Toolchains = toml::from_str(&toolchains).map_err(|e| format!("{TOOLCHAINS}: {e}"))?;
    let dockerfile = Path::new(&pins.linux.image.dockerfile);
    let dir = dockerfile
        .parent()
        .ok_or("the Dockerfile has no directory")?;
    let work = std::env::temp_dir().join(format!("syntropy-toolchains-{}", std::process::id()));
    fs::create_dir_all(&work).map_err(|e| format!("{}: {e}", work.display()))?;
    let docker = |args: &[&str]| -> Result<(), String> {
        let status = Command::new("docker")
            .current_dir(&context.root)
            .args(args)
            .status()
            .map_err(|e| format!("could not start `docker`: {e}"))?;
        if status.success() {
            Ok(())
        } else {
            Err(format!("`docker {}`: {status}", args.join(" ")))
        }
    };
    let builder = format!("syntropy-image-{}", std::process::id());
    let driver_opt = format!("image={}", pins.linux.image.buildkit);
    docker(&[
        "buildx",
        "create",
        "--name",
        &builder,
        "--driver",
        "docker-container",
        "--driver-opt",
        &driver_opt,
        "--bootstrap",
    ])?;
    let built = build_and_print(&pins, dir, &work, &builder, &docker);
    fs::remove_dir_all(&work).ok();
    docker(&["buildx", "rm", &builder]).ok();
    let printed = built?;
    let problems = version_problems(&printed, &pins);
    if problems.is_empty() {
        eprintln!("toolchains-linux: nvcc, cmake, g++ and glibc match ci/toolchains.toml");
        Ok(())
    } else {
        Err(problems.join("\n"))
    }
}

/// The two no-cache builds on `builder`, their comparison, and the version print from inside the
/// image: what the image printed. Both builds export Docker image tarballs. The print runs in the
/// first of the two compared tarballs,
/// loaded with `docker load`, so the versions shown are those of the bytes shown to rebuild
/// identically, not of a third build. `work` holds the archives; the caller removes it and the
/// builder.
fn build_and_print(
    pins: &Toolchains,
    dir: &Path,
    work: &Path,
    builder: &str,
    docker: &dyn Fn(&[&str]) -> Result<(), String>,
) -> Result<String, String> {
    let epoch = format!("SOURCE_DATE_EPOCH={}", pins.linux.image.source_date_epoch);
    let context = dir.display().to_string();
    let mut archives = Vec::new();
    for n in 1..=2 {
        let archive = work.join(format!("cuda-13-{n}.tar"));
        // A Docker image tarball, which carries the `manifest.json` the daemon's classic image
        // store needs to load it (an OCI-layout tarball has none). `name` is the same in both
        // builds, so the comparison is unaffected and `docker load` tags the image it loads.
        let output = format!(
            "type=docker,dest={},rewrite-timestamp=true,name={CHECK_TAG}",
            archive.display()
        );
        docker(&[
            "buildx",
            "build",
            "--builder",
            builder,
            "--platform",
            "linux/amd64",
            "--no-cache",
            "--build-arg",
            &epoch,
            "--output",
            &output,
            &context,
        ])?;
        archives.push(archive);
    }
    let read = |p: &Path| fs::read(p).map_err(|e| format!("{}: {e}", p.display()));
    if read(&archives[0])? != read(&archives[1])? {
        return Err(
            "the two builds of the Linux image differ: it does not rebuild byte for byte".into(),
        );
    }
    eprintln!("toolchains-linux: two no-cache builds are byte-identical");
    let load = Command::new("docker")
        .args(["load", "--input"])
        .arg(&archives[0])
        .output()
        .map_err(|e| format!("could not start `docker load`: {e}"))?;
    if !load.status.success() {
        return Err(format!(
            "`docker load` of the compared archive failed ({}): {}",
            load.status,
            String::from_utf8_lossy(&load.stderr)
        ));
    }
    let loaded = String::from_utf8_lossy(&load.stdout);
    let Some(tag) = loaded_image(&loaded) else {
        // The load may still have stored the image under the archive's name.
        docker(&["image", "rm", CHECK_TAG]).ok();
        return Err(format!("`docker load` named no image it loaded: {loaded}"));
    };
    eprintln!("toolchains-linux: the version print runs in the compared archive, loaded as {tag}");
    let run = Command::new("docker")
        .args([
            "run",
            "--rm",
            "--platform",
            "linux/amd64",
            &tag,
            "sh",
            "-c",
            "nvcc --version && cmake --version | head -1 && g++ --version | head -1 && ldd --version | head -1",
        ])
        .output();
    // Removed whether or not the print ran, so no image is left behind.
    docker(&["image", "rm", &tag]).ok();
    let run = run.map_err(|e| format!("could not start `docker run`: {e}"))?;
    if !run.status.success() {
        return Err(format!("the version print failed: {}", run.status));
    }
    let printed = String::from_utf8_lossy(&run.stdout).into_owned();
    eprintln!("{printed}");
    Ok(printed)
}

/// The name `docker load` gives the image the compared tarball holds (the export's `name`).
const CHECK_TAG: &str = "syntropy-build-check:local";

/// The image `docker load` reports it loaded: the last `Loaded image: <name>` line, or else the
/// last `Loaded image ID: <id>` line for an image that came without a name.
fn loaded_image(stdout: &str) -> Option<String> {
    let last = |prefix: &str| {
        stdout
            .lines()
            .filter_map(|line| line.trim().strip_prefix(prefix))
            .map(|rest| rest.trim().to_owned())
            .rfind(|image| !image.is_empty())
    };
    last("Loaded image: ").or_else(|| last("Loaded image ID: "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_loaded_image_is_read_from_docker_loads_report() {
        assert_eq!(
            loaded_image("Loaded image: syntropy-build-check:local\n").as_deref(),
            Some("syntropy-build-check:local")
        );
        assert_eq!(
            loaded_image("Loaded image ID: sha256:0123abcd\n").as_deref(),
            Some("sha256:0123abcd")
        );
        // A name wins over an ID, and the last of each counts.
        assert_eq!(
            loaded_image("Loaded image ID: sha256:aa\nLoaded image: a:1\nLoaded image: b:2\n")
                .as_deref(),
            Some("b:2")
        );
        assert_eq!(loaded_image("nothing loaded\n"), None);
        assert_eq!(loaded_image("Loaded image: \n"), None);
    }

    fn real() -> (String, String, String) {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let read = |p: &str| fs::read_to_string(root.join(p)).expect(p);
        (
            read(TOOLCHAINS),
            read("ci/images/cuda-13/Dockerfile"),
            read(ARCH_SETS),
        )
    }

    /// What the image printed on 2026-10-26 (the `toolchains-linux` run in the PR description).
    const PRINTED: &str = "nvcc: NVIDIA (R) Cuda compiler driver
Cuda compilation tools, release 13.4, V13.4.92
Build cuda_13.4.r13.4/compiler.38855100_0
cmake version 4.4.4
g++ (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0
ldd (Ubuntu GLIBC 2.35-0ubuntu3.15) 2.35
";

    fn pins() -> Toolchains {
        toml::from_str(&real().0).expect("pins parse")
    }

    #[test]
    fn the_printed_versions_match_the_pins() {
        assert_eq!(version_problems(PRINTED, &pins()), Vec::<String>::new());
    }

    #[test]
    fn a_version_that_differs_from_its_pin_is_named() {
        let cmake = PRINTED.replace("cmake version 4.4.4", "cmake version 4.4.3");
        let problems = version_problems(&cmake, &pins());
        assert_eq!(problems.len(), 1, "{problems:?}");
        assert!(problems[0].contains("cmake"), "{problems:?}");
        let nvcc = PRINTED.replace("V13.4.92", "V13.4.91");
        assert!(version_problems(&nvcc, &pins())[0].contains("nvcc"));
        let gxx = PRINTED.replace("22.04.3", "22.04.2");
        assert!(
            version_problems(&gxx, &pins())
                .iter()
                .any(|p| p.contains("g++"))
        );
        let glibc = PRINTED.replace(" 2.35\n", " 2.39\n");
        assert!(version_problems(&glibc, &pins())[0].contains("glibc"));
    }

    #[test]
    fn a_tool_that_printed_nothing_is_named() {
        let none = PRINTED.replace("cmake version 4.4.4\n", "");
        assert_eq!(
            version_problems(&none, &pins()),
            ["`cmake` printed no version line"]
        );
    }

    #[test]
    fn the_checked_in_files_agree() {
        let (t, d, a) = real();
        assert_eq!(check(&t, &d, &a), Vec::<String>::new());
    }

    #[test]
    fn a_pin_the_dockerfile_lacks_is_named() {
        let (t, d, a) = real();
        let problems = check(&t, &d.replace("20261020T000000Z", "20261021T000000Z"), &a);
        assert_eq!(problems.len(), 1, "{problems:?}");
        assert!(problems[0].contains("apt_snapshot"), "{problems:?}");
    }

    #[test]
    fn a_pin_named_only_in_a_comment_is_not_applied() {
        let (t, d, a) = real();
        let line = d
            .lines()
            .find(|l| l.contains("ARG APT_SNAPSHOT="))
            .expect("snapshot line");
        let commented = d.replace(line, &format!("# {line}"));
        let problems = check(&t, &commented, &a);
        assert!(
            problems.iter().any(|p| p.contains("apt_snapshot")),
            "{problems:?}"
        );
    }

    #[test]
    fn a_bad_checksum_and_an_unknown_key_are_refused() {
        let (t, d, a) = real();
        let bad = t.replacen(&t[t.find("sha256 = \"").unwrap() + 10..][..64], "XYZ", 1);
        assert!(check(&bad, &d, &a).iter().any(|p| p.contains("checksum")));
        assert!(
            check(&format!("{t}\nextra = 1\n"), &d, &a)
                .iter()
                .any(|p| p.contains("extra"))
        );
    }

    #[test]
    fn a_buildkit_image_without_a_digest_is_refused() {
        let (t, d, a) = real();
        let tag_only = t.replace(
            "moby/buildkit:v0.34.1@sha256:99368f3e46177c6405b9740e4461d91cee004128daeecca78d4ceac50ec1da5c",
            "moby/buildkit:v0.34.1",
        );
        assert_ne!(tag_only, t);
        assert!(
            check(&tag_only, &d, &a)
                .iter()
                .any(|p| p.contains("linux.image.buildkit"))
        );
    }

    #[test]
    fn an_unverified_pin_with_a_value_is_refused() {
        let (t, d, a) = real();
        let bad = t.replace(
            "[macos.metal]\nverified = false",
            "[macos.metal]\nverified = false\nversion = \"1\"",
        );
        assert_ne!(bad, t);
        assert!(
            check(&bad, &d, &a)
                .iter()
                .any(|p| p.contains("macos.metal"))
        );
    }

    #[test]
    fn the_pr_profile_needs_sm_89_and_no_ptx() {
        let (t, d, a) = real();
        let no_sm = a.replace("sm_89", "sm_80");
        assert!(check(&t, &d, &no_sm).iter().any(|p| p.contains("sm_89")));
        let ptx = a.replace("cuda_ptx = []", "cuda_ptx = [\"compute_89\"]");
        assert!(check(&t, &d, &ptx).iter().any(|p| p.contains("PTX")));
    }
}
