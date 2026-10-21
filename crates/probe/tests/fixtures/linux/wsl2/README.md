# Fixture: WSL2

These files are hand-written, not captured from a real machine. They follow the formats the
Linux kernel uses for each file, with values chosen to be realistic for the hardware described,
from public specifications of the parts named. Only the files E1 reads are present, plus a few
neighbouring fields so the parsers see real-looking files. The tree mirrors the real paths:
`proc/...` is `/proc/...`, `sys/...` is `/sys/...`, `dev/dxg` is `/dev/dxg` (an empty file
standing in for the device node; enumeration only checks that it exists).

`crates/probe/tests/linux_fixtures.rs` asserts the `NodeFacts`, CPU `DeviceInfo` and
`HostMemInfo` this tree must produce.

## What it describes

- CPU: an AMD Ryzen 7 7800X3D as a WSL2 guest sees it: 8 cores, 16 threads, siblings
  adjacent, the `hypervisor` flag. x86-64-v4 with AVX-512 VNNI and BF16, no AVX-512 FP16 and
  no AVX-VNNI. 1 MiB L2 per core, 96 MiB L3.
- Memory: 32 GB, half the host's, which is what WSL2 assigns by default.
- WSL2 kernel 5.15: `/dev/dxg` exists, and `kernel.io_uring_disabled` does not (the sysctl
  arrived in 6.6). cgroup v2 at the root, with no memory limit. `RLIMIT_MEMLOCK` 64 KiB, the
  kernel default before 5.16. THP `madvise`. No power supplies.
