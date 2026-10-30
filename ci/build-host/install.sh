#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0

# Installs or updates the Linux build host (ADR-0625) on an x86-64 Linux machine with Docker and
# systemd. Run as root from the root of a `main` checkout:
#   sudo ci/build-host/install.sh [slots]
# It builds the job image, installs the scripts and the unit, and starts `slots` job slots
# (default 3). The token must already be in /etc/syntropy-build/token (README.md).

set -euo pipefail

slots="${1:-3}"
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"

[ "$(id -u)" -eq 0 ] || { echo "run as root" >&2; exit 1; }
[ -s /etc/syntropy-build/token ] || { echo "put the token in /etc/syntropy-build/token first" >&2; exit 1; }
chmod 0600 /etc/syntropy-build/token

docker build --pull -f "$here/Dockerfile" -t syntropy-build:latest "$root"

install -d -m 0755 /opt/syntropy-build
install -m 0755 "$here/runner-slot.sh" "$here/egress.sh" /opt/syntropy-build/
install -m 0644 "$here/syntropy-build-runner@.service" /etc/systemd/system/
systemctl daemon-reload

for slot in $(seq 1 "$slots"); do
  systemctl enable "syntropy-build-runner@$slot" >/dev/null
  systemctl restart "syntropy-build-runner@$slot"
done
# Slots above the count, from an earlier install with more.
for unit in $(systemctl list-units --plain --no-legend 'syntropy-build-runner@*' | awk '{print $1}'); do
  n="${unit#syntropy-build-runner@}"; n="${n%.service}"
  if [ "$n" -gt "$slots" ]; then systemctl disable --now "$unit"; fi
done
systemctl --no-pager --lines=0 status 'syntropy-build-runner@*'
