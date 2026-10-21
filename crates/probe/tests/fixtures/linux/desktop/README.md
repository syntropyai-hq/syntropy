# Fixture: desktop with hybrid cores

These files are hand-written, not captured from a real machine. They follow the formats the
Linux kernel uses for each file, with values chosen to be realistic for the hardware described,
from public specifications of the parts named. Only the files E1 reads are present, plus a few
neighbouring fields so the parsers see real-looking files. The tree mirrors the real paths:
`proc/...` is `/proc/...`, `sys/...` is `/sys/...`.

`crates/probe/tests/linux_fixtures.rs` asserts the `NodeFacts`, CPU `DeviceInfo` and
`HostMemInfo` this tree must produce.

## What it describes

- CPU: the layout of a 12th-generation Intel Core i5-12600K. 6 P-cores with two threads each
  (CPUs 0-11) and 4 E-cores with one thread each (CPUs 12-15). `sys/devices/cpu_core/cpus` and
  `sys/devices/cpu_atom/cpus` list the two core types, as the kernel's hybrid PMU directories
  do. x86-64-v3 with AVX-VNNI, no AVX-512. Caches: 48 KiB L1d and 1.25 MiB L2 per P-core,
  2 MiB L2 per 4-E-core cluster, one 20 MiB L3.
- Memory: 32 GB, one NUMA node.
- Kernel 6.8, cgroup v2, the process in a user-session scope with no memory limit at any level.
  `RLIMIT_MEMLOCK` 8 MiB, no capabilities. THP `madvise`. io_uring sysctl 0.
- Power: no mains supply entry, and one wireless mouse battery with `scope` `Device`, which
  enumeration must ignore.
