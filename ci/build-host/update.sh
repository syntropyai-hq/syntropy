#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0

# Keeps the Linux build host (ADR-0625 decision 9) on what `main` says it is. Run daily by
# `syntropy-build-update.timer`, as root. It fetches `main` into its own clone and, when
# `ci/build-host/` or `rust-toolchain.toml` changed since the last install, rebuilds the job image
# and installs the scripts and units. It restarts only the slots with no job running, so a new
# image or script takes over without stopping a job; a busy slot takes it at its next run.

set -euo pipefail

repo_url="${SYNTROPY_BUILD_REPO_URL:-https://github.com/syntropyai-hq/syntropy.git}"
branch="${SYNTROPY_BUILD_BRANCH:-main}"
state=/var/lib/syntropy-build
src="$state/src"

[ "$(id -u)" -eq 0 ] || { echo "run as root" >&2; exit 1; }
install -d -m 0755 "$state"
[ -d "$src/.git" ] || git clone --quiet --depth 1 --branch "$branch" "$repo_url" "$src"
git -C "$src" fetch --quiet --depth 1 origin "$branch"
git -C "$src" checkout --quiet --force --detach FETCH_HEAD

# The tree hashes of what the host is built from: equal hashes, equal host.
stamp="$(git -C "$src" rev-parse HEAD:ci/build-host HEAD:rust-toolchain.toml | tr '\n' ' ')"
if [ "$stamp" = "$(cat "$state/installed" 2>/dev/null || true)" ]; then
  echo "up to date with $branch $(git -C "$src" rev-parse --short HEAD)"
  exit 0
fi

here="$src/ci/build-host"
docker build --pull --quiet -f "$here/Dockerfile" -t syntropy-build:latest "$src"
install -d -m 0755 /opt/syntropy-build
install -m 0755 "$here/runner-slot.sh" "$here/egress.sh" "$here/update.sh" /opt/syntropy-build/
install -m 0644 "$here/syntropy-build-runner@.service" "$here/syntropy-build-update.service" \
  "$here/syntropy-build-update.timer" /etc/systemd/system/
systemctl daemon-reload

# A slot's container runs `Runner.Worker` only while it has a job.
for unit in $(systemctl list-units --plain --no-legend --state=active 'syntropy-build-runner@*' | awk '{print $1}'); do
  slot="${unit#syntropy-build-runner@}"; slot="${slot%.service}"
  if ! docker top "syntropy-build-$slot" 2>/dev/null | grep -q Runner.Worker; then
    systemctl restart "$unit"
  fi
done
echo "$stamp" >"$state/installed"
echo "installed $branch $(git -C "$src" rev-parse --short HEAD)"
