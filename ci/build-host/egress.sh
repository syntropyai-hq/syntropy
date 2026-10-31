#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0

# The job network of the Linux build host (ADR-0625): a Docker bridge whose containers reach the
# internet and nothing else. They reach no private, link-local or tailnet address, so no LAN host,
# and nothing on the build host itself. Idempotent; the runner units run it before each start.

set -euo pipefail

network=syntropy-build
bridge=br-syn-build
subnet=172.30.0.0/24

if ! docker network inspect "$network" >/dev/null 2>&1; then
  docker network create --driver bridge --subnet "$subnet" \
    --opt com.docker.network.bridge.name="$bridge" \
    --opt com.docker.network.bridge.enable_icc=false \
    "$network" >/dev/null
fi

# Insert a rule at the top of a chain unless it is already there.
rule() {
  local chain="$1"; shift
  iptables -C "$chain" "$@" 2>/dev/null || iptables -I "$chain" 1 "$@"
}

# Forwarded traffic, before Docker's own rules.
for cidr in 10.0.0.0/8 172.16.0.0/12 192.168.0.0/16 100.64.0.0/10 169.254.0.0/16; do
  rule DOCKER-USER -i "$bridge" -d "$cidr" -j DROP
done
# The build host itself, on any of its addresses.
rule INPUT -i "$bridge" -j DROP
