#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Writes the Windows API response fixtures the `syntropy-probe` tests read.

The values are hand-chosen, not captured, as for the Linux trees in `../linux`. This script
lays them out in the formats the calls return: the `GetLogicalProcessorInformationEx`
buffer in the 64-bit `<winnt.h>` layout, and the ACPI SRAT (ACPI 6.5 §5.2.16), as hex in
the fixture format of `src/windows/record.rs`. README.md says what each scenario describes.

    python3 generate.py <output directory>

Standard library only.
"""
import os
import struct
import sys

OUT = sys.argv[1] if len(sys.argv) == 2 else sys.exit(__doc__)
GIB = 1 << 30
MIB = 1 << 20


def masks(lps):
    groups = {}
    for g, b in lps:
        groups[g] = groups.get(g, 0) | (1 << b)
    out = b""
    for g in sorted(groups):
        out += struct.pack("<QH6x", groups[g], g)
    return len(groups), out


def record(rel, body):
    return struct.pack("<II", rel, len(body) + 8) + body


def core(smt, eff, lps):
    n, m = masks(lps)
    return record(0, struct.pack("<BB20xH", 1 if smt else 0, eff, n) + m)


def package(lps):
    n, m = masks(lps)
    return record(3, struct.pack("<BB20xH", 0, 0, n) + m)


def node(number, lps):
    n, m = masks(lps)
    return record(1, struct.pack("<I18xH", number, n) + m)


def cache(level, ty, size, lps):
    # PROCESSOR_CACHE_TYPE: 0 unified, 1 instruction, 2 data.
    n, m = masks(lps)
    return record(2, struct.pack("<BBHII18xH", level, 8, 64, size, ty, n) + m)


def group(active):
    # GROUP_RELATIONSHIP with one group: E1 skips it.
    return record(4, struct.pack("<HH20xBB38xQ", 1, 1, 64, active, (1 << active) - 1))


def srat(ranges):
    body = b""
    base = 0
    for pd, length in ranges:
        body += struct.pack("<BBI2xQQ4xI8x", 1, 40, pd, base, length, 1)
        base += length
    t = b"SRAT" + struct.pack("<I", 48 + len(body)) + bytes(40) + body
    return t


def hexlines(key, data):
    return "".join(
        f"{key} {data[i:i + 48].hex()}\n" for i in range(0, len(data), 48)
    )


def write(name, header, cpu, glpi, memory, numa, srat_table, proximity, power, ceiling, version):
    lines = "".join(f"# {h}\n" for h in header)
    lines += f"cpu arch {cpu[0]}\ncpu name {cpu[1]}\ncpu flags {' '.join(sorted(cpu[2]))}\n"
    lines += hexlines("glpi", glpi)
    lines += f"memory total={memory[0]} available={memory[1]}\n"
    for n, b in numa:
        lines += f"numa-available {n} {b}\n"
    lines += hexlines("srat", srat_table) if srat_table is not None else "srat absent\n"
    for pd, n in proximity:
        lines += f"proximity {pd} {n}\n"
    lines += f"power ac={power[0]} flag={power[1]} percent={power[2]}\n"
    lines += f"working-set-ceiling {ceiling}\nversion {version[0]} {version[1]} {version[2]}\n"
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as f:
        f.write(lines)


V2 = ["cx16", "lahf_lm", "popcnt", "sse4_1", "sse4_2", "ssse3"]
V3 = ["avx2", "fma", "f16c"]
V4 = ["avx512f", "avx512bw", "avx512dq", "avx512vl"]

# Desktop: the i5-12600K layout of ../linux/desktop. 6 P-cores with SMT (LPs 0-11,
# EfficiencyClass 1) and 4 E-cores (LPs 12-15, class 0); 48 KiB L1d and 1.25 MiB L2 per
# P-core, 2 MiB L2 per E-core cluster, one 20 MiB L3; one NUMA node; 32 GB.
lps = [(0, b) for b in range(16)]
buf = b""
for c in range(6):
    t = [(0, 2 * c), (0, 2 * c + 1)]
    buf += core(True, 1, t)
for c in range(4):
    buf += core(False, 0, [(0, 12 + c)])
for c in range(6):
    t = [(0, 2 * c), (0, 2 * c + 1)]
    buf += cache(1, 2, 48 * 1024, t) + cache(1, 1, 32 * 1024, t) + cache(2, 0, 1280 * 1024, t)
for c in range(4):
    buf += cache(1, 2, 32 * 1024, [(0, 12 + c)]) + cache(1, 1, 64 * 1024, [(0, 12 + c)])
buf += cache(2, 0, 2 * MIB, [(0, b) for b in range(12, 16)])
buf += cache(3, 0, 20 * MIB, lps)
buf += package(lps) + node(0, lps) + group(16)
write(
    "desktop.txt",
    ["Hand-chosen, not captured: see README.md. Written by generate.py."],
    ("x86_64", "12th Gen Intel(R) Core(TM) i5-12600K", V2 + V3 + ["avx_vnni"]),
    buf,
    (34_190_000_128, 25_769_803_776),
    [(0, 25_769_803_776)],
    None,
    [],
    (1, 128, 255),
    30 * GIB,
    (10, 0, 26100),
)

# Dual socket: two 16-core sockets with SMT (64 LPs, one processor group), each socket one
# NUMA node with two 32 MiB L3 caches (two core complexes); 1 MiB L2 per core; 128 GiB per
# node in the SRAT, whose proximity domains 0 and 1 map to nodes 0 and 1.
buf = b""
for s in range(2):
    for c in range(16):
        base = s * 32 + 2 * c
        t = [(0, base), (0, base + 1)]
        buf += core(True, 0, t)
        buf += cache(1, 2, 32 * 1024, t) + cache(2, 0, MIB, t)
    for x in range(2):
        buf += cache(3, 0, 32 * MIB, [(0, s * 32 + x * 16 + b) for b in range(16)])
    buf += package([(0, s * 32 + b) for b in range(32)])
    buf += node(s, [(0, s * 32 + b) for b in range(32)])
buf += group(64)
write(
    "dual-socket.txt",
    ["Hand-chosen, not captured: see README.md. Written by generate.py."],
    ("x86_64", "AMD EPYC 9124 16-Core Processor", V2 + V3 + V4 + ["avx512_vnni", "avx512_bf16"]),
    buf,
    (274_877_906_944 - 3 * GIB, 250 * GIB),
    [(0, 120 * GIB), (1, 126 * GIB)],
    srat([(0, 2 * GIB), (0, 126 * GIB), (1, 128 * GIB)]),
    [(0, 0), (1, 1)],
    (1, 128, 255),
    200 * GIB,
    (10, 0, 26100),
)
