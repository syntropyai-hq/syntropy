#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0

# One job slot of the Linux build host (ADR-0625), run by `syntropy-build-runner@<slot>.service`.
# Each run mints a single-use just-in-time runner configuration, runs one job in a fresh container
# of the `syntropy-build` image, and exits when the container does; systemd starts the next run.
#
# The credential that mints configurations stays on the host, in the unit's credential directory:
# a GitHub App's private key (`app-key.pem` with its ID in `app-id`), from which each run mints an
# installation token that lasts an hour, or else a fine-grained token (`token`). The container gets
# only the configuration, which registers one runner for one job.

set -euo pipefail

slot="${1:?usage: runner-slot.sh <slot>}"
repo="${SYNTROPY_BUILD_REPO:-syntropyai-hq/syntropy}"
label="${SYNTROPY_BUILD_LABEL:-syntropy-build}"
image="${SYNTROPY_BUILD_IMAGE:-syntropy-build:latest}"
memory="${SYNTROPY_BUILD_MEMORY:-14g}"
creds="${CREDENTIALS_DIRECTORY:?run under systemd with LoadCredential=syntropy-build:/etc/syntropy-build}"
api=https://api.github.com

# GET or POST to the API with the bearer on curl's stdin, never in its arguments, which any local
# user can read. Usage: gh_api <bearer> <method> <path> [body].
gh_api() {
  local body=()
  [ $# -lt 4 ] || body=(--data "$4")
  printf 'Authorization: Bearer %s\n' "$1" |
    curl --fail --silent --show-error --proto '=https' --tlsv1.2 --retry 3 \
      -X "$2" -H @- \
      -H "Accept: application/vnd.github+json" \
      -H "X-GitHub-Api-Version: 2022-11-28" \
      "${body[@]}" \
      "$api$3"
}

b64url() { openssl base64 -A | tr '+/' '-_' | tr -d '='; }

if [ -s "$creds/syntropy-build_app-key.pem" ]; then
  # A JWT signed with the app's key, valid for 9 minutes (GitHub allows 10), backdated a minute
  # for clock skew, and exchanged for an installation token limited to this repository and to
  # Administration: write, the permission that registers repository runners.
  app_id="$(tr -d '[:space:]' <"$creds/syntropy-build_app-id")"
  now="$(date +%s)"
  claims="$(jq -cn --arg iss "$app_id" --argjson now "$now" '{iat: ($now - 60), exp: ($now + 540), iss: $iss}' | b64url)"
  unsigned="$(printf '{"alg":"RS256","typ":"JWT"}' | b64url).$claims"
  signature="$(printf '%s' "$unsigned" | openssl dgst -sha256 -sign "$creds/syntropy-build_app-key.pem" | b64url)"
  jwt="$unsigned.$signature"
  installation="$(gh_api "$jwt" GET "/repos/$repo/installation" | jq -r '.id')"
  token="$(gh_api "$jwt" POST "/app/installations/$installation/access_tokens" \
    "$(jq -cn --arg r "${repo#*/}" '{repositories: [$r], permissions: {administration: "write"}}')" |
    jq -r '.token')"
  unset jwt signature
else
  token="$(cat "$creds/syntropy-build_token")"
fi
[ -n "$token" ] && [ "$token" != null ] || { echo "no token to mint a runner configuration with" >&2; exit 1; }

# A run that ends without taking a job, because a restart or an update stopped it, leaves its
# runner registered and offline. Remove this slot's; a failure here never holds up the slot.
prefix="$(hostname -s)-${slot}-"
for id in $(gh_api "$token" GET "/repos/$repo/actions/runners?per_page=100" |
  jq -r --arg p "$prefix" '.runners[] | select(.status == "offline" and (.name | startswith($p))) | .id' ||
  true); do
  gh_api "$token" DELETE "/repos/$repo/actions/runners/$id" >/dev/null || true
done

name="${prefix}$(date +%s)"
request="$(jq -n --arg name "$name" --arg label "$label" \
  '{name: $name, runner_group_id: 1, labels: [$label], work_folder: "_work"}')"
jit="$(gh_api "$token" POST "/repos/$repo/actions/runners/generate-jitconfig" "$request" |
  jq -r '.encoded_jit_config')"
unset token
[ -n "$jit" ] && [ "$jit" != null ] || { echo "no JIT configuration returned" >&2; exit 1; }

# A container left by a killed run would hold the name.
docker rm --force "syntropy-build-$slot" >/dev/null 2>&1 || true

# Public DNS: the network cannot reach the LAN resolver (egress.sh). Capabilities: every one is
# dropped, then the job gets back only what sudo, `useradd` and the IPC peer test need, and
# `SYS_NICE` for `mbind` in `syntropy-tier`'s NUMA tests; each reaches only the container's own
# namespaces. No `NET_RAW`, so a job cannot forge packets on the bridge. No seccomp filter, as on
# a hosted runner: `syntropy-io`'s io_uring seccomp test installs its own filter and needs an
# unfiltered process for its control, and Docker's default profile also blocks io_uring. AppArmor
# stays on. The configuration reaches the container through the environment, by name, not
# through docker's arguments.
export JITCONFIG="$jit"
exec docker run --rm --init \
  --name "syntropy-build-$slot" \
  --network syntropy-build \
  --dns 1.1.1.1 --dns 9.9.9.9 \
  --cap-drop ALL \
  --cap-add CHOWN --cap-add DAC_OVERRIDE --cap-add FOWNER --cap-add SETUID --cap-add SETGID \
  --cap-add KILL --cap-add AUDIT_WRITE --cap-add SYS_NICE \
  --security-opt apparmor=docker-default --security-opt seccomp=unconfined \
  --memory "$memory" --memory-swap "$memory" --pids-limit 16384 \
  --env JITCONFIG \
  "$image"
