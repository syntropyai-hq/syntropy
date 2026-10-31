#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Capture a Linux rig's inventory record for bench/lab/rigs.toml (RFC-0061 §6.3, QUAL-E15-T02).
#
# Read-only. It prints one `[[rig]]` TOML record on stdout and changes nothing on the rig. It
# needs root, for `dmidecode`. A value it cannot read stops the script with a message and a
# non-zero status; it never guesses or fills a default. Run it on the rig, then review the
# output and append it to bench/lab/rigs.toml in a PR; `syntropy-bench rigs check` validates it.
#
# Usage:
#   capture-inventory.sh --id R-ADA --os-image <image ID> --room-sensor <sensor ID> \
#       --memory-profile <profile> --power-limits <limits> --resizable-bar <true|false> \
#       --attachment <nvme name>=<cpu|chipset> [--attachment ...] \
#       [--expect-compute-cap 8.9]
#
# The values the operating system cannot report come from flags, read off the firmware setup
# screen or the board manual: the BIOS memory profile, power limits and resizable-BAR state, the
# image ID, the room sensor, and where each NVMe slot attaches (`cpu` or `chipset`).
#
# What is read, and from where:
#   kernel, power plan   uname -r; the cpu0 cpufreq governor
#   CPU                  lscpu; the first `microcode` line of /proc/cpuinfo (omitted if none)
#   memory               dmidecode -t memory: one entry per populated module, at its
#                        configured speed
#   GPU                  nvidia-smi: PCI bus ID as the slot, model, VBIOS, driver, and the
#                        highest link the GPU and host support (the current link drops at idle)
#   NVMe                 /sys/class/nvme: model, firmware, the negotiated link
#   NIC                  /sys/class/net devices that sit on PCI: lspci name, ethtool firmware,
#                        the negotiated link
#   BIOS                 dmidecode -s bios-version; SMT from /sys/devices/system/cpu/smt
#
# A GPU's current link falls to a low power state at idle, so the GPU's link is the highest that
# the GPU and the slot support together (nvidia-smi's gpumax, hostmax and width.max); a GPU in a
# slower slot still shows the slot's limit. NVMe and NIC links are the negotiated ones.
#
# The spare GPU of ADR-0610 decision 1 has no format of its own: RFC-0061 §6.3 defines one record
# per rig. Install the spare in the rig's host, run this script with `--id R-ADA-SPARE`, and
# commit that record next to R-ADA's. Its fingerprint then differs from R-ADA's by the GPU's
# slot, VBIOS or link, which is the point: a swap is a visible epoch change.
#
# SYNTROPY_SYSROOT prefixes /sys and /proc, for the tests; leave it unset on a rig.

set -euo pipefail

root="${SYNTROPY_SYSROOT:-}"

# Stop the whole script. Most callers run inside `$(...)`, where a plain `exit` would end only the
# subshell and let the record go on with an empty value, so signal the main shell instead.
trap 'exit 1' TERM
die() {
    echo "capture-inventory: $*" >&2
    kill -s TERM "$$"
    exit 1
}

# The record is built in a file and printed only when complete, so a refusal prints no record.
buf="$(mktemp)"
trap 'rm -f "$buf"' EXIT
exec 3>&1 1> "$buf"

# Print $1 as a TOML basic string, or stop: a value with a quote, backslash or control character
# is not one line of plain text (rigs.toml, header comment).
quote() {
    local value="$1"
    [ -n "$value" ] || die "empty value for ${2:-a field}"
    case "$value" in
        *\"* | *\\*) die "value for ${2:-a field} contains a quote or backslash: $value" ;;
    esac
    if printf '%s' "$value" | LC_ALL=C grep -q '[[:cntrl:]]'; then
        die "value for ${2:-a field} contains a control character"
    fi
    [ "$value" = "$(printf '%s' "$value" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')" ] \
        || die "value for ${2:-a field} has surrounding whitespace: '$value'"
    printf '"%s"' "$value"
}

trim() {
    printf '%s' "$1" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}

need() {
    command -v "$1" > /dev/null 2>&1 || die "$1 is not installed"
}

read_file() {
    [ -r "$1" ] || die "cannot read $1"
    trim "$(cat "$1")"
}

# GT/s as sysfs prints it ("16.0 GT/s PCIe") -> PCIe generation.
gen_of_speed() {
    case "$1" in
        2.5*) echo 1 ;;
        5.0* | 5\ *) echo 2 ;;
        8.0* | 8\ *) echo 3 ;;
        16.0* | 16\ *) echo 4 ;;
        32.0* | 32\ *) echo 5 ;;
        64.0* | 64\ *) echo 6 ;;
        128.0* | 128\ *) echo 7 ;;
        *) die "unknown PCIe link speed '$1'" ;;
    esac
}

# The link a PCI device negotiated in its slot (RFC-0061 §6.3), from its sysfs directory:
# `gen4x4`. A Gen4 drive in a Gen3 slot reads `gen3x4`. Not read from `max_link_*`, which is the
# device's rating and would hide the slot.
negotiated_link() {
    local dir="$1"
    echo "gen$(gen_of_speed "$(read_file "$dir/current_link_speed")")x$(read_file "$dir/current_link_width")"
}

id=""
os_image=""
room_sensor=""
memory_profile=""
power_limits=""
resizable_bar=""
expect_cc=""
attachments=()
while [ $# -gt 0 ]; do
    [ $# -ge 2 ] || die "$1 needs a value"
    case "$1" in
        --id) id="$2" ;;
        --os-image) os_image="$2" ;;
        --room-sensor) room_sensor="$2" ;;
        --memory-profile) memory_profile="$2" ;;
        --power-limits) power_limits="$2" ;;
        --resizable-bar) resizable_bar="$2" ;;
        --attachment) attachments+=("$2") ;;
        --expect-compute-cap) expect_cc="$2" ;;
        *) die "unknown option $1" ;;
    esac
    shift 2
done
for required in id os_image room_sensor memory_profile power_limits resizable_bar; do
    [ -n "${!required}" ] || die "--${required//_/-} is required"
done
case "$resizable_bar" in true | false) ;; *) die "--resizable-bar is true or false" ;; esac

for tool in uname lscpu dmidecode nvidia-smi lspci ethtool awk sed; do need "$tool"; done

kernel="$(uname -r)"
power_plan="$(read_file "$root/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")"
# Every "key: value" read below drops only the text up to the first colon: a value may hold more
# colons (a NIC firmware "2017:888d").
cpu_model="$(lscpu | awk '/^Model name:/ { sub(/^[^:]*:[ \t]*/, ""); print; exit }')"
cpu_model="$(trim "$cpu_model")"
microcode="$(awk '/^microcode/ { sub(/^[^:]*:[ \t]*/, ""); gsub(/[ \t]+$/, ""); print; exit }' \
    "$root/proc/cpuinfo")"
bios_version="$(trim "$(dmidecode -s bios-version)")"
smt_active="$(read_file "$root/sys/devices/system/cpu/smt/active")"
case "$smt_active" in 1) smt=true ;; 0) smt=false ;; *) die "unexpected smt/active '$smt_active'" ;; esac

printf '[[rig]]\n'
printf 'id = %s\n' "$(quote "$id" id)"
printf 'os_image = %s\n' "$(quote "$os_image" os_image)"
printf 'kernel = %s\n' "$(quote "$kernel" kernel)"
printf 'power_plan = %s\n' "$(quote "$power_plan" power_plan)"
printf 'room_sensor = %s\n' "$(quote "$room_sensor" room_sensor)"
printf '\n[rig.cpu]\n'
printf 'model = %s\n' "$(quote "$cpu_model" cpu.model)"
if [ -n "$microcode" ]; then
    printf 'microcode = %s\n' "$(quote "$microcode" cpu.microcode)"
fi

# Memory: one record per populated module.
modules=0
while IFS='|' read -r slot size speed clock part; do
    [ -n "$slot" ] || continue
    case "$size" in
        *GB) size_gib="${size% GB}" ;;
        *) die "memory module $slot has size '$size'; expected whole GB" ;;
    esac
    # "Configured Memory Speed" is the label of current dmidecode. The older "Configured Clock
    # Speed" is taken only when the newer label is absent, or when both name the same speed.
    if [ -z "$speed" ]; then
        speed="$clock"
    elif [ -n "$clock" ] && [ "$clock" != "$speed" ]; then
        die "memory module $slot: configured speeds disagree ('$speed' and '$clock')"
    fi
    [ -n "$speed" ] || die "memory module $slot reports no configured speed"
    speed_mts="${speed% MT/s}"
    case "$size_gib" in "" | *[!0-9]*) die "memory module $slot: size '$size' is not a whole number of GB" ;; esac
    case "$speed_mts" in "" | *[!0-9]*) die "memory module $slot: speed '$speed' is not a whole number of MT/s" ;; esac
    printf '\n[[rig.memory]]\nslot = %s\nsize_gib = %s\nspeed_mts = %s\npart = %s\n' \
        "$(quote "$slot" memory.slot)" "$size_gib" "$speed_mts" "$(quote "$part" memory.part)"
    modules=$((modules + 1))
done < <(dmidecode -t memory | awk '
    function v(s) { sub(/^[^:]*:/, "", s); gsub(/^[ \t]+|[ \t]+$/, "", s); return s }
    function flush() { if (size != "" && size != "No Module Installed") print loc "|" size "|" spd "|" clk "|" pn }
    /^Memory Device/ { flush(); loc = ""; size = ""; spd = ""; clk = ""; pn = "" }
    /^\tLocator:/ { loc = v($0) }
    /^\tSize:/ { size = v($0) }
    /^\tConfigured Memory Speed:/ { spd = v($0) }
    /^\tConfigured Clock Speed:/ { clk = v($0) }
    /^\tPart Number:/ { pn = v($0) }
    END { flush() }')
[ "$modules" -gt 0 ] || die "dmidecode lists no populated memory module"

# GPUs.
gpus=0
while IFS=',' read -r bus name vbios driver cc gpumax hostmax width; do
    [ -n "$bus" ] || continue
    bus="$(trim "$bus")"; name="$(trim "$name")"; vbios="$(trim "$vbios")"
    driver="$(trim "$driver")"; cc="$(trim "$cc")"; gpumax="$(trim "$gpumax")"
    hostmax="$(trim "$hostmax")"; width="$(trim "$width")"
    if [ -n "$expect_cc" ] && [ "$cc" != "$expect_cc" ]; then
        die "GPU $bus has compute capability $cc, expected $expect_cc"
    fi
    case "$gpumax$hostmax$width" in
        *[!0-9]* | "") die "GPU $bus: link query returned '$gpumax' '$hostmax' '$width'" ;;
    esac
    gen="$gpumax"
    [ "$hostmax" -ge "$gpumax" ] 2> /dev/null || gen="$hostmax"
    printf '\n[[rig.gpu]]\nslot = %s\nmodel = %s\nvbios = %s\ndriver = %s\npcie = "gen%sx%s"\n' \
        "$(quote "$bus" gpu.slot)" "$(quote "$name" gpu.model)" "$(quote "$vbios" gpu.vbios)" \
        "$(quote "$driver" gpu.driver)" "$gen" "$width"
    gpus=$((gpus + 1))
done < <(nvidia-smi --format=csv,noheader \
    --query-gpu=pci.bus_id,name,vbios_version,driver_version,compute_cap,pcie.link.gen.gpumax,pcie.link.gen.hostmax,pcie.link.width.max)
[ "$gpus" -gt 0 ] || die "nvidia-smi lists no GPU"

# NVMe drives.
for ctrl in "$root"/sys/class/nvme/nvme*; do
    [ -e "$ctrl" ] || continue
    name="$(basename "$ctrl")"
    attach=""
    for entry in ${attachments[@]+"${attachments[@]}"}; do
        [ "${entry%%=*}" = "$name" ] && attach="${entry#*=}"
    done
    case "$attach" in
        cpu | chipset) ;;
        *) die "$name needs --attachment $name=cpu or $name=chipset (read it from the board manual)" ;;
    esac
    printf '\n[[rig.nvme]]\nslot = %s\nmodel = %s\nfirmware = %s\nattachment = %s\npcie = %s\n' \
        "$(quote "$name" nvme.slot)" "$(quote "$(read_file "$ctrl/model")" nvme.model)" \
        "$(quote "$(read_file "$ctrl/firmware_rev")" nvme.firmware)" "$(quote "$attach")" \
        "$(quote "$(negotiated_link "$ctrl/device")" nvme.pcie)"
done

# NICs on PCI.
nic_slots=""
for dev in "$root"/sys/class/net/*; do
    [ -e "$dev/device" ] || continue
    ifname="$(basename "$dev")"
    target="$(readlink "$dev/device")" || die "$ifname: cannot read its device link"
    addr="$(basename "$target")"
    case "$addr" in
        [0-9a-f][0-9a-f][0-9a-f][0-9a-f]:[0-9a-f][0-9a-f]:[0-9a-f][0-9a-f].[0-7]) ;;
        *) die "$ifname: its device link points at '$addr', not a PCI function; remove or exclude this network device" ;;
    esac
    case " $nic_slots " in
        *" $addr "*) die "$ifname: PCI function $addr is already recorded for another network device" ;;
    esac
    nic_slots="$nic_slots $addr"
    model="$(lspci -s "$addr" -mm | awk -F'"' '{ print $6 }')" || die "lspci failed for $addr"
    fw="$(ethtool -i "$ifname" | awk '/^firmware-version:/ { sub(/^[^:]*:[ \t]*/, ""); print; exit }')" \
        || die "ethtool -i $ifname failed"
    printf '\n[[rig.nic]]\nslot = %s\nmodel = %s\nfirmware = %s\npcie = %s\n' \
        "$(quote "$addr" nic.slot)" "$(quote "$(trim "$model")" nic.model)" \
        "$(quote "$(trim "$fw")" "nic.firmware of $ifname")" \
        "$(quote "$(negotiated_link "$dev/device")" nic.pcie)"
done

printf '\n[rig.bios]\nversion = %s\nmemory_profile = %s\npower_limits = %s\nsmt = %s\nresizable_bar = %s\n' \
    "$(quote "$bios_version" bios.version)" "$(quote "$memory_profile" memory_profile)" \
    "$(quote "$power_limits" power_limits)" "$smt" "$resizable_bar"
cat "$buf" >&3
