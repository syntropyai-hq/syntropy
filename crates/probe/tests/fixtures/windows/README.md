# Windows E1 fixtures

Recorded responses of the Windows calls E1 makes, in the fixture format of
`src/windows/record.rs`. `crates/probe/tests/windows_fixtures.rs` asserts the `NodeFacts`, CPU
`DeviceInfo` and `HostMemInfo` each must produce.

`desktop.txt` and `dual-socket.txt` are hand-chosen, not captured from a real machine, like the
Linux trees in `../linux`. `generate.py` writes them, laying the values out in the binary
layouts the calls return: the `GetLogicalProcessorInformationEx(RelationAll)` buffer in the
64-bit `<winnt.h>` layout and the ACPI SRAT (ACPI 6.5 §5.2.16). Run
`python3 generate.py <dir>` and diff to check them.

- `desktop.txt`: the layout of a 12th-generation Intel Core i5-12600K, as `../linux/desktop`:
  6 P-cores with two threads each (`EfficiencyClass` 1) and 4 E-cores (`EfficiencyClass` 0),
  1.25 MiB L2 per P-core, 2 MiB L2 per E-core cluster, one 20 MiB L3; one NUMA node; 32 GB;
  on mains with no battery; Windows 11 build 26100.
- `dual-socket.txt`: two 16-core sockets with SMT in one processor group, each socket one NUMA
  node with two 32 MiB L3 caches; the SRAT gives each node 128 GiB, in two ranges for node 0.

Responses captured on a live host have a header naming the host and the run; see
`windows-2025-runner.txt` once it is recorded.
