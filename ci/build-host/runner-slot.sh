#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0

# One job slot of the Linux build host (ADR-0625), run by `syntropy-build-runner@<slot>.service`.
# Each run mints a single-use just-in-time runner configuration, runs one job in a fresh container
# of the `syntropy-build` image, and exits when the container does; systemd starts the next run.
#
# The token that mints configurations stays on the host, in the unit's credential directory. The
# container gets only the configuration, which registers one runner for one job.

set -euo pipefail

slot="${1:?usage: runner-slot.sh <slot>}"
repo="${SYNTROPY_BUILD_REPO:-syntropyai-hq/syntropy}"
label="${SYNTROPY_BUILD_LABEL:-syntropy-build}"
image="${SYNTROPY_BUILD_IMAGE:-syntropy-build:latest}"
memory="${SYNTROPY_BUILD_MEMORY:-14g}"
token="$(cat "${CREDENTIALS_DIRECTORY:?run under systemd with LoadCredential=token}/token")"

name="$(hostname -s)-${slot}-$(date +%s)"
request="$(jq -n --arg name "$name" --arg label "$label" \
  '{name: $name, runner_group_id: 1, labels: [$label], work_folder: "_work"}')"
jit="$(curl --fail --silent --show-error --proto '=https' --tlsv1.2 --retry 3 \
  -X POST \
  -H "Authorization: Bearer $token" \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  --data "$request" \
  "https://api.github.com/repos/$repo/actions/runners/generate-jitconfig" |
  jq -r '.encoded_jit_config')"
unset token
[ -n "$jit" ] && [ "$jit" != null ] || { echo "no JIT configuration returned" >&2; exit 1; }

# A container left by a killed run would hold the name.
docker rm --force "syntropy-build-$slot" >/dev/null 2>&1 || true

# Public DNS: the network cannot reach the LAN resolver (egress.sh).
exec docker run --rm --init \
  --name "syntropy-build-$slot" \
  --network syntropy-build \
  --dns 1.1.1.1 --dns 9.9.9.9 \
  --memory "$memory" --pids-limit 16384 \
  --env JITCONFIG="$jit" \
  "$image"
