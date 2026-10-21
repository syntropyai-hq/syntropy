# Fixture: dual-socket server

The values are hand-chosen, not captured from a real machine, and `../generate.py` writes the
files from them. They follow the formats the
Linux kernel uses for each file, with values chosen to be realistic for the hardware described,
from public specifications of the parts named. Only the files E1 reads are present, plus a few
neighbouring fields so the parsers see real-looking files. The tree mirrors the real paths:
`proc/...` is `/proc/...`, `sys/...` is `/sys/...`.

`crates/probe/tests/linux_fixtures.rs` asserts the `NodeFacts`, CPU `DeviceInfo` and
`HostMemInfo` this tree must produce.

## What it describes

- CPU: two Intel Xeon Gold 5415+ class sockets, 8 cores and 16 threads each, numbered the way
  the kernel numbers two-socket hosts: socket 0's cores are CPUs 0-7 with siblings 16-23,
  socket 1's are CPUs 8-15 with siblings 24-31. x86-64-v4 with AVX-512 VNNI, BF16 and FP16,
  AVX-VNNI and AMX. 2 MiB L2 per core, 22.5 MiB L3 per socket.
- Memory: 256 GB in two NUMA nodes of about 128 GB, one per socket.
- Kernel 6.8, cgroup v2, a systemd service with no memory limit, running as uid 1000.
  `RLIMIT_MEMLOCK` 8 MiB, but `CAP_IPC_LOCK` is in the permitted and effective sets, so locking
  is not limited. The capability comes from a file capability on the binary
  (`setcap cap_ipc_lock+ep`), which is why the inheritable and ambient sets are empty; systemd's
  `AmbientCapabilities=` would also have set `CapAmb`. THP `always`. io_uring sysctl 0. No power
  supplies.
