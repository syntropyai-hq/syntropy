#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0

# Installs or updates the Linux build host (ADR-0625) on an x86-64 Linux machine with Docker and
# systemd. Run as root from the root of a `main` checkout:
#   sudo ci/build-host/install.sh [slots]
# It builds the job image, installs the scripts and the units, starts `slots` job slots (default
# 3) and enables the daily update from `main` (update.sh). The credential must already be in
# /etc/syntropy-build: a GitHub App's `app-key.pem` and `app-id`, or a `token` (README.md).

set -euo pipefail

slots="${1:-3}"
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"

[ "$(id -u)" -eq 0 ] || { echo "run as root" >&2; exit 1; }
cred=/etc/syntropy-build
if [ -s "$cred/app-key.pem" ] && [ -s "$cred/app-id" ]; then :
elif [ -s "$cred/token" ]; then :
else echo "put app-key.pem and app-id, or a token, in $cred first" >&2; exit 1
fi
chmod 0700 "$cred"; chmod 0600 "$cred"/*

docker build --pull -f "$here/Dockerfile" -t syntropy-build:latest "$root"

install -d -m 0755 /opt/syntropy-build
install -m 0755 "$here/runner-slot.sh" "$here/egress.sh" "$here/update.sh" /opt/syntropy-build/
install -m 0644 "$here/syntropy-build-runner@.service" "$here/syntropy-build-update.service" \
  "$here/syntropy-build-update.timer" /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now syntropy-build-update.timer >/dev/null

for slot in $(seq 1 "$slots"); do
  systemctl enable "syntropy-build-runner@$slot" >/dev/null
  systemctl restart "syntropy-build-runner@$slot"
done
# Slots above the count, from an earlier install with more.
for unit in $(systemctl list-units --all --plain --no-legend 'syntropy-build-runner@*' | awk '{print $1}'); do
  n="${unit#syntropy-build-runner@}"; n="${n%.service}"
  if [ "$n" -gt "$slots" ]; then systemctl disable --now "$unit"; fi
done
# Informational: a slot between jobs, or still starting, makes `status` exit non-zero.
systemctl --no-pager --lines=0 status 'syntropy-build-runner@*' || true
