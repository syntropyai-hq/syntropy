#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Writes the Linux sysfs/procfs fixture trees the `syntropy-probe` tests read.

The values are hand-chosen (each scenario's README says where they come from); this script
only lays them out as files, including the per-CPU topology and cache trees. It is the
source of every file under the scenario directories except the READMEs.

    python3 generate.py <output directory>

Run it into an empty directory and diff against the committed trees, or run it on this
directory to rewrite them in place. Standard library only.
"""
import os
import sys

OUT = sys.argv[1] if len(sys.argv) == 2 else sys.exit(__doc__)


def w(root, path, text):
    p = os.path.join(root, path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(text if text.endswith("\n") or text == "" else text + "\n")


def cpulist(cpus):
    cpus = sorted(cpus)
    out, i = [], 0
    while i < len(cpus):
        j = i
        while j + 1 < len(cpus) and cpus[j + 1] == cpus[j] + 1:
            j += 1
        out.append(str(cpus[i]) if i == j else f"{cpus[i]}-{cpus[j]}")
        i = j + 1
    return ",".join(out)


def limits(memlock):
    rows = [
        ("Max cpu time", "unlimited", "unlimited", "seconds"),
        ("Max file size", "unlimited", "unlimited", "bytes"),
        ("Max data size", "unlimited", "unlimited", "bytes"),
        ("Max stack size", "8388608", "unlimited", "bytes"),
        ("Max core file size", "0", "unlimited", "bytes"),
        ("Max resident set", "unlimited", "unlimited", "bytes"),
        ("Max processes", "127573", "127573", "processes"),
        ("Max open files", "1024", "1048576", "files"),
        ("Max locked memory", memlock, memlock, "bytes"),
        ("Max address space", "unlimited", "unlimited", "bytes"),
        ("Max file locks", "unlimited", "unlimited", "locks"),
        ("Max pending signals", "127573", "127573", "signals"),
        ("Max msgqueue size", "819200", "819200", "bytes"),
        ("Max nice priority", "0", "0", ""),
        ("Max realtime priority", "0", "0", ""),
        ("Max realtime timeout", "unlimited", "unlimited", "us"),
    ]
    s = "Limit                     Soft Limit           Hard Limit           Units     \n"
    for n, a, b, u in rows:
        s += f"{n:<26}{a:<21}{b:<21}{u:<10}\n"
    return s


def status(name, uid, gid, groups, capeff, capbnd, seccomp):
    return (
        f"Name:\t{name}\nUmask:\t0022\nState:\tR (running)\nTgid:\t4242\nPid:\t4242\nPPid:\t4200\n"
        f"Uid:\t{uid}\t{uid}\t{uid}\t{uid}\nGid:\t{gid}\t{gid}\t{gid}\t{gid}\n"
        f"Groups:\t{' '.join(map(str, groups))} \n"
        "VmLck:\t       0 kB\nVmPin:\t       0 kB\nThreads:\t1\n"
        "CapInh:\t0000000000000000\nCapPrm:\t" + capeff + "\nCapEff:\t" + capeff + "\n"
        "CapBnd:\t" + capbnd + "\nCapAmb:\t0000000000000000\nNoNewPrivs:\t0\n"
        f"Seccomp:\t{seccomp}\nSeccomp_filters:\t{1 if seccomp == 2 else 0}\n"
    )


# Active(file), Inactive(file) and SReclaimable in kB, per node and host-wide: the clean cache
# RFC-0020 §5's per-NUMA estimate counts as reclaimable. A one-node host's /proc/meminfo
# matches its node; the dual-socket host's is the sum of its two nodes.
DESKTOP_RECLAIM = (2310456, 3402108, 389204)
DUAL_NODE0_RECLAIM = (1602340, 2104772, 512388)
DUAL_NODE1_RECLAIM = (1498116, 1987204, 498772)
DUAL_RECLAIM = tuple(a + b for a, b in zip(DUAL_NODE0_RECLAIM, DUAL_NODE1_RECLAIM))
WSL2_RECLAIM = (312444, 401256, 98312)
DOCKER_RECLAIM = (2198004, 3011552, 402876)
RPI5_RECLAIM = (301224, 289908, 61004)


def meminfo(total, free, avail, reclaim):
    act, inact, srecl = reclaim
    return (
        f"MemTotal:       {total:>8} kB\nMemFree:        {free:>8} kB\nMemAvailable:   {avail:>8} kB\n"
        "Buffers:          412312 kB\nCached:          6123044 kB\nSwapCached:            0 kB\n"
        f"Active(file):   {act:>8} kB\nInactive(file): {inact:>8} kB\n"
        "SwapTotal:       2097148 kB\nSwapFree:        2097148 kB\n"
        f"SReclaimable:   {srecl:>8} kB\nHugePages_Total:       0\n"
        "HugePages_Free:        0\nHugepagesize:       2048 kB\n"
    )


def node_meminfo(n, total, free, reclaim):
    act, inact, srecl = reclaim
    return (
        f"Node {n} MemTotal:       {total:>8} kB\nNode {n} MemFree:        {free:>8} kB\n"
        f"Node {n} MemUsed:        {total - free:>8} kB\n"
        f"Node {n} Active(file):   {act:>8} kB\nNode {n} Inactive(file): {inact:>8} kB\n"
        f"Node {n} SReclaimable:   {srecl:>8} kB\nNode {n} HugePages_Total:     0\n"
    )


def cpu_topo(root, cpu, core_cpus, package, core_id):
    b = f"sys/devices/system/cpu/cpu{cpu}/topology"
    w(root, f"{b}/core_cpus_list", cpulist(core_cpus))


def cache(root, cpu, idx, level, typ, size, shared):
    b = f"sys/devices/system/cpu/cpu{cpu}/cache/index{idx}"
    w(root, f"{b}/level", str(level))
    w(root, f"{b}/type", typ)
    w(root, f"{b}/size", size)
    w(root, f"{b}/shared_cpu_list", cpulist(shared))


def x86_cpuinfo(cpus, model, family, modelno, flags, cores_of, mhz):
    s = ""
    for c in cpus:
        pkg, core, siblings, ncores = cores_of(c)
        s += (
            f"processor\t: {c}\nvendor_id\t: {'GenuineIntel' if 'Intel' in model else 'AuthenticAMD'}\n"
            f"cpu family\t: {family}\nmodel\t\t: {modelno}\nmodel name\t: {model}\nstepping\t: 2\n"
            f"cpu MHz\t\t: {mhz}\nphysical id\t: {pkg}\nsiblings\t: {siblings}\ncore id\t\t: {core}\n"
            f"cpu cores\t: {ncores}\napicid\t\t: {c * 2}\nfpu\t\t: yes\nfpu_exception\t: yes\n"
            f"cpuid level\t: 32\nwp\t\t: yes\nflags\t\t: {flags}\nbogomips\t: 7372.80\n"
            "clflush size\t: 64\ncache_alignment\t: 64\naddress sizes\t: 46 bits physical, 48 bits virtual\n"
            "power management:\n\n"
        )
    return s


BASE = ("fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr "
        "sse sse2 ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology nonstop_tsc "
        "cpuid pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx "
        "f16c rdrand lahf_lm abm 3dnowprefetch bmi1 avx2 smep bmi2 erms invpcid rdseed adx smap "
        "clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 xsaves")


# A desktop or service user: uid 1000 with ordinary supplementary groups.
USER = (1000, 1000, [4, 24, 27, 1000])
# Root inside a container, as Docker runs a process by default.
CONTAINER_ROOT = (0, 0, [0, 1, 2, 3, 4, 6, 10, 11, 20, 26, 27])


def proc_common(root, osrelease, memlock, capeff, seccomp, cgroup_self, io_uring, thp, thp_size,
                user=USER, capbnd="000001ffffffffff"):
    w(root, "proc/sys/kernel/osrelease", osrelease)
    w(root, "proc/self/limits", limits(memlock))
    w(root, "proc/self/status", status("syntropy", *user, capeff, capbnd, seccomp))
    w(root, "proc/self/cgroup", cgroup_self)
    if io_uring is not None:
        w(root, "proc/sys/kernel/io_uring_disabled", str(io_uring))
        w(root, "proc/sys/kernel/io_uring_group", "-1")
    w(root, "sys/kernel/mm/transparent_hugepage/enabled", thp)
    w(root, "sys/kernel/mm/transparent_hugepage/hpage_pmd_size", str(thp_size))


def desktop(root):
    # 6 hybrid P-cores with two threads each (CPUs 0-11), 4 E-cores (12-15).
    cpus = list(range(16))
    flags = BASE + " avx_vnni serialize arch_lbr"

    def cores_of(c):
        return (0, c // 2 * 4 if c < 12 else 32 + (c - 12) * 2, 16, 10)
    w(root, "proc/cpuinfo", x86_cpuinfo(cpus, "12th Gen Intel(R) Core(TM) i5-12600K", 6, 151, flags, cores_of, "3700.000"))
    w(root, "proc/meminfo", meminfo(32562044, 18304412, 26120588, DESKTOP_RECLAIM))
    w(root, "sys/devices/system/cpu/online", "0-15")
    w(root, "sys/devices/cpu_core/cpus", "0-11")
    w(root, "sys/devices/cpu_atom/cpus", "12-15")
    for c in cpus:
        if c < 12:
            core = [c - c % 2, c - c % 2 + 1]
            cpu_topo(root, c, core, 0, c // 2 * 4)
            cache(root, c, 0, 1, "Data", "48K", core)
            cache(root, c, 1, 1, "Instruction", "32K", core)
            cache(root, c, 2, 2, "Unified", "1280K", core)
        else:
            cpu_topo(root, c, [c], 0, 32 + (c - 12) * 2)
            cache(root, c, 0, 1, "Data", "32K", [c])
            cache(root, c, 1, 1, "Instruction", "64K", [c])
            cache(root, c, 2, 2, "Unified", "2048K", [12, 13, 14, 15])
        cache(root, c, 3, 3, "Unified", "20480K", cpus)
    w(root, "sys/devices/system/node/node0/cpulist", "0-15")
    w(root, "sys/devices/system/node/node0/meminfo", node_meminfo(0, 32562044, 18304412, DESKTOP_RECLAIM))
    proc_common(root, "6.8.0-45-generic", "8388608", "0000000000000000", 0,
                "0::/user.slice/user-1000.slice/user@1000.service/app.slice/syntropy.scope",
                0, "always [madvise] never", 2097152)
    cg = "sys/fs/cgroup"
    w(root, f"{cg}/cgroup.controllers", "cpuset cpu io memory hugetlb pids rdma misc")
    leaf = f"{cg}/user.slice/user-1000.slice/user@1000.service/app.slice/syntropy.scope"
    for d, cur in [(f"{cg}/user.slice", 9123393536), (f"{cg}/user.slice/user-1000.slice", 8023187456),
                   (f"{cg}/user.slice/user-1000.slice/user@1000.service", 7983247360),
                   (f"{cg}/user.slice/user-1000.slice/user@1000.service/app.slice", 6012510208),
                   (leaf, 41943040)]:
        w(root, f"{d}/memory.max", "max")
        w(root, f"{d}/memory.high", "max")
        w(root, f"{d}/memory.current", str(cur))
    # A wireless mouse's battery: a peripheral, scope Device.
    ps = "sys/class/power_supply/hidpp_battery_0"
    w(root, f"{ps}/type", "Battery")
    w(root, f"{ps}/scope", "Device")
    w(root, f"{ps}/status", "Discharging")
    w(root, f"{ps}/capacity", "70")
    w(root, "proc/1/cgroup", "0::/init.scope")


def dual_socket(root):
    # 2 sockets x 8 cores x 2 threads. Socket s core k: CPUs 8s+k and 16+8s+k.
    cpus = list(range(32))
    flags = (BASE + " avx512f avx512dq avx512ifma avx512cd avx512bw avx512vl avx512_bf16 avx512vbmi "
             "umip pku ospke waitpkg avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg "
             "tme avx512_vpopcntdq la57 rdpid bus_lock_detect cldemote movdiri movdir64b enqcmd "
             "fsrm md_clear serialize tsxldtrk pconfig arch_lbr ibt amx_bf16 avx512_fp16 amx_tile "
             "amx_int8 flush_l1d arch_capabilities avx_vnni")

    def where(c):
        t = c % 16
        return t // 8, t % 8

    def cores_of(c):
        s, k = where(c)
        return (s, k, 16, 8)
    w(root, "proc/cpuinfo", x86_cpuinfo(cpus, "Intel(R) Xeon(R) Gold 5415+", 6, 143, flags, cores_of, "2900.000"))
    w(root, "proc/meminfo", meminfo(263856048, 241002316, 252874220, DUAL_RECLAIM))
    w(root, "sys/devices/system/cpu/online", "0-31")
    for c in cpus:
        s, k = where(c)
        core = [8 * s + k, 16 + 8 * s + k]
        sock = [x for x in cpus if where(x)[0] == s]
        cpu_topo(root, c, core, s, k)
        cache(root, c, 0, 1, "Data", "48K", core)
        cache(root, c, 1, 1, "Instruction", "32K", core)
        cache(root, c, 2, 2, "Unified", "2048K", core)
        cache(root, c, 3, 3, "Unified", "23040K", sock)
    w(root, "sys/devices/system/node/node0/cpulist", "0-7,16-23")
    w(root, "sys/devices/system/node/node0/meminfo", node_meminfo(0, 131854356, 120112508, DUAL_NODE0_RECLAIM))
    w(root, "sys/devices/system/node/node1/cpulist", "8-15,24-31")
    w(root, "sys/devices/system/node/node1/meminfo", node_meminfo(1, 132001692, 120889808, DUAL_NODE1_RECLAIM))
    w(root, "sys/devices/system/node/online", "0-1")
    proc_common(root, "6.8.0-45-generic", "8388608", "0000000000004000", 0,
                "0::/system.slice/syntropy.service", 0, "[always] madvise never", 2097152)
    cg = "sys/fs/cgroup"
    w(root, f"{cg}/cgroup.controllers", "cpuset cpu io memory hugetlb pids rdma misc")
    for d, cur in [(f"{cg}/system.slice", 2147483648), (f"{cg}/system.slice/syntropy.service", 524288000)]:
        w(root, f"{d}/memory.max", "max")
        w(root, f"{d}/memory.current", str(cur))
    w(root, "proc/1/cgroup", "0::/init.scope")


def wsl2(root):
    cpus = list(range(16))
    flags = (BASE + " hypervisor avx512f avx512dq avx512ifma avx512cd avx512bw avx512vl avx512_bf16 "
             "avx512vbmi umip avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg "
             "avx512_vpopcntdq rdpid fsrm")

    def cores_of(c):
        return (0, c // 2, 16, 8)
    w(root, "proc/cpuinfo", x86_cpuinfo(cpus, "AMD Ryzen 7 7800X3D 8-Core Processor", 25, 97, flags, cores_of, "4199.998"))
    w(root, "proc/meminfo", meminfo(32803456, 30112708, 31207884, WSL2_RECLAIM))
    w(root, "sys/devices/system/cpu/online", "0-15")
    for c in cpus:
        core = [c - c % 2, c - c % 2 + 1]
        cpu_topo(root, c, core, 0, c // 2)
        cache(root, c, 0, 1, "Data", "32K", core)
        cache(root, c, 1, 1, "Instruction", "32K", core)
        cache(root, c, 2, 2, "Unified", "1024K", core)
        cache(root, c, 3, 3, "Unified", "98304K", cpus)
    w(root, "sys/devices/system/node/node0/cpulist", "0-15")
    w(root, "sys/devices/system/node/node0/meminfo", node_meminfo(0, 32803456, 30112708, WSL2_RECLAIM))
    proc_common(root, "5.15.153.1-microsoft-standard-WSL2", "65536", "0000000000000000", 0,
                "0::/", None, "always [madvise] never", 2097152)
    w(root, "sys/fs/cgroup/cgroup.controllers", "cpuset cpu io memory pids")
    w(root, "dev/dxg", "")
    w(root, "proc/1/cgroup", "0::/")


def docker(root):
    # Ryzen 9 5950X host: core k is CPUs k and k+16; two L3s of 32 MiB, one per CCD.
    cpus = list(range(32))
    flags = BASE.replace(" invpcid", "") + " clzero irperf xsaveerptr rdpru wbnoinvd arat npt lbrv vaes vpclmulqdq rdpid overflow_recov succor smca fsrm"

    def cores_of(c):
        return (0, c % 16, 32, 16)
    w(root, "proc/cpuinfo", x86_cpuinfo(cpus, "AMD Ryzen 9 5950X 16-Core Processor", 25, 33, flags, cores_of, "3400.000"))
    w(root, "proc/meminfo", meminfo(65777452, 40221396, 58993172, DOCKER_RECLAIM))
    w(root, "sys/devices/system/cpu/online", "0-31")
    for c in cpus:
        k = c % 16
        core = [k, k + 16]
        ccd = [x for x in cpus if (x % 16) // 8 == k // 8]
        cpu_topo(root, c, core, 0, k)
        cache(root, c, 0, 1, "Data", "32K", core)
        cache(root, c, 1, 1, "Instruction", "32K", core)
        cache(root, c, 2, 2, "Unified", "512K", core)
        cache(root, c, 3, 3, "Unified", "32768K", ccd)
    w(root, "sys/devices/system/node/node0/cpulist", "0-31")
    w(root, "sys/devices/system/node/node0/meminfo", node_meminfo(0, 65777452, 40221396, DOCKER_RECLAIM))
    proc_common(root, "6.8.0-45-generic", "8388608", "00000000a80425fb", 2,
                "0::/", 0, "always [madvise] never", 2097152,
                user=CONTAINER_ROOT, capbnd="00000000a80425fb")
    cg = "sys/fs/cgroup"
    w(root, f"{cg}/cgroup.controllers", "cpuset cpu io memory hugetlb pids rdma misc")
    w(root, f"{cg}/memory.max", "8589934592")
    w(root, f"{cg}/memory.high", "max")
    w(root, f"{cg}/memory.current", "1073741824")
    w(root, ".dockerenv", "")
    w(root, "proc/1/cgroup", "0::/")


def rpi5(root):
    cpus = list(range(4))
    feats = "fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm lrcpc dcpop asimddp"
    s = ""
    for c in cpus:
        s += (f"processor\t: {c}\nBogoMIPS\t: 108.00\nFeatures\t: {feats}\nCPU implementer\t: 0x41\n"
              "CPU architecture: 8\nCPU variant\t: 0x4\nCPU part\t: 0xd0b\nCPU revision\t: 1\n\n")
    s += "Revision\t: d04170\nSerial\t\t: 0000000000000000\nModel\t\t: Raspberry Pi 5 Model B Rev 1.0\n"
    w(root, "proc/cpuinfo", s)
    w(root, "proc/meminfo", meminfo(8245168, 6922480, 7570784, RPI5_RECLAIM))
    w(root, "sys/devices/system/cpu/online", "0-3")
    for c in cpus:
        cpu_topo(root, c, [c], 0, c)
        w(root, f"sys/devices/system/cpu/cpu{c}/cpu_capacity", "1024")
        cache(root, c, 0, 1, "Data", "64K", [c])
        cache(root, c, 1, 1, "Instruction", "64K", [c])
        cache(root, c, 2, 2, "Unified", "512K", [c])
        cache(root, c, 3, 3, "Unified", "2048K", cpus)
    proc_common(root, "6.6.51+rpt-rpi-2712", "unlimited", "0000000000000000", 0,
                "0::/user.slice/user-1000.slice/session-3.scope", 0, "always [madvise] never", 33554432)
    cg = "sys/fs/cgroup"
    # The memory controller is off (cgroup_disable=memory), so no memory.* files exist.
    w(root, f"{cg}/cgroup.controllers", "cpuset cpu io pids")
    w(root, f"{cg}/user.slice/user-1000.slice/session-3.scope/cgroup.procs", "4242")
    w(root, "proc/1/cgroup", "0::/init.scope")


for name, fn in [("desktop", desktop), ("dual-socket", dual_socket), ("wsl2", wsl2),
                 ("docker", docker), ("rpi5", rpi5)]:
    fn(os.path.join(OUT, name))
