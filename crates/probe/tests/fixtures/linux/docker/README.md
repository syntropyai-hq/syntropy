# Fixture: Docker container

The values are hand-chosen, not captured from a real machine, and `../generate.py` writes the
files from them. They follow the formats the
Linux kernel uses for each file, with values chosen to be realistic for the hardware described,
from public specifications of the parts named. Only the files E1 reads are present, plus a few
neighbouring fields so the parsers see real-looking files. The tree mirrors the real paths:
`proc/...` is `/proc/...`, `sys/...` is `/sys/...`, `.dockerenv` is `/.dockerenv`.

`crates/probe/tests/linux_fixtures.rs` asserts the `NodeFacts`, CPU `DeviceInfo` and
`HostMemInfo` this tree must produce.

## What it describes

- CPU: the host's AMD Ryzen 9 5950X, which a container sees unchanged in sysfs: 16 cores,
  32 threads, core k is CPUs k and k+16, two 32 MiB L3s (one per CCD). x86-64-v3, no AVX-512.
- Memory: the host's 64 GB in `/proc/meminfo`, and a cgroup v2 limit at the container's
  namespace root: `memory.max` 8 GiB with `memory.current` 1 GiB.
- `/.dockerenv` exists. Seccomp mode 2 (a filter), as Docker's default profile installs.
- The process runs as root inside the container (uid 0), which is Docker's default. Its
  effective and bounding capability sets are Docker's default set, `a80425fb`, which has no
  `CAP_IPC_LOCK`. `RLIMIT_MEMLOCK` 8 MiB. Kernel 6.8, io_uring sysctl 0, THP `madvise`. No
  power supplies.
