# Fixture: Raspberry Pi 5

The values are hand-chosen, not captured from a real machine, and `../generate.py` writes the
files from them. They follow the formats the
Linux kernel uses for each file, with values chosen to be realistic for the hardware described,
from public specifications of the parts named. Only the files E1 reads are present, plus a few
neighbouring fields so the parsers see real-looking files. The tree mirrors the real paths:
`proc/...` is `/proc/...`, `sys/...` is `/sys/...`.

`crates/probe/tests/linux_fixtures.rs` asserts the `NodeFacts`, CPU `DeviceInfo` and
`HostMemInfo` this tree must produce.

## What it describes

- CPU: 4 Arm Cortex-A76 cores (implementer 0x41, part 0xd0b), one thread each. AArch64 v8.2
  with dot product and FP16 (`asimddp`, `asimdhp`); no I8MM, BF16 or SVE2. Every
  `cpu_capacity` is 1024, so the CPU is homogeneous. 512 KiB L2 per core, 2 MiB L3.
  `/proc/cpuinfo` has no `model name`, as on AArch64 kernels.
- Memory: 8 GB. No `sys/devices/system/node` directory, as with a kernel built without NUMA,
  so the host is one domain and its free memory comes from `/proc/meminfo`.
- A 6.6 Raspberry Pi kernel with 16 KiB pages, so the THP size is 32 MiB. cgroup v2 with the
  memory controller disabled (no `memory.*` files). `RLIMIT_MEMLOCK` unlimited, as an
  administrator would set it in `limits.conf`. io_uring sysctl 0. No power supplies.
