#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0

# The build host's watchdog (ADR-0625 decision 8), run by `build-host-watchdog.yml` on a hosted
# runner every 10 minutes. It moves the gate variable `SYNTROPY_BUILD_HOST` between `on` and
# `auto-off` from whether any runner labelled `syntropy-build` is online, so a host that is down
# sends jobs back to hosted runners instead of leaving them queued. It never touches any other
# value, so a maintainer's `off` stays off.
#
# Environment:
#   REPO          owner/name
#   APP_TOKEN     reads the runners and writes the variable (Administration: read, Variables: write)
#   RUNS_TOKEN    cancels and re-runs workflow runs (the job's token, `actions: write`)
#   DRY_RUN=1     report what it would do, change nothing

set -euo pipefail

: "${REPO:?}" "${APP_TOKEN:?}" "${RUNS_TOKEN:?}"
label=syntropy-build
var=SYNTROPY_BUILD_HOST
dry="${DRY_RUN:-0}"

# gh with one of the two tokens: as_app or as_runs, then gh's arguments.
as_app() { GH_TOKEN="$APP_TOKEN" gh "$@"; }
as_runs() { GH_TOKEN="$RUNS_TOKEN" gh "$@"; }

online() {
  as_app api "repos/$REPO/actions/runners?per_page=100" \
    --jq "[.runners[] | select(.status == \"online\") | select(any(.labels[]; .name == \"$label\"))] | length"
}

set_gate() {
  echo "setting $var to $1"
  [ "$dry" = 1 ] || as_app variable set "$var" --repo "$REPO" --body "$1"
}

# A run with a job still queued for the build host waits for it even after the gate moves, since
# a job's runner is chosen when the run starts. Cancel each such run and run it again, so it reads
# the gate afresh and goes to hosted runners.
requeue_stuck_runs() {
  local id status
  for id in $(as_runs run list --repo "$REPO" --limit 100 --json databaseId,status \
    --jq '.[] | select(.status == "queued" or .status == "in_progress" or .status == "waiting") | .databaseId'); do
    as_runs api "repos/$REPO/actions/runs/$id/jobs?per_page=100" \
      --jq "any(.jobs[]; .status == \"queued\" and any(.labels[]; . == \"$label\"))" | grep -q true || continue
    echo "run $id waits for the build host: cancelling and re-running it"
    [ "$dry" = 1 ] && continue
    as_runs run cancel "$id" --repo "$REPO" || true
    status=""
    for _ in $(seq 1 30); do
      status="$(as_runs api "repos/$REPO/actions/runs/$id" --jq .status)"
      [ "$status" = completed ] && break
      sleep 5
    done
    if [ "$status" = completed ]; then
      as_runs run rerun "$id" --repo "$REPO" || echo "::warning::could not re-run run $id"
    else
      echo "::warning::run $id did not finish cancelling; re-run it by hand"
    fi
  done
}

gate="$(as_app variable get "$var" --repo "$REPO" 2>/dev/null || true)"
n="$(online)"
echo "$var=$gate; $n runner(s) labelled $label online"

case "$gate" in
  on)
    [ "$n" -gt 0 ] && exit 0
    # Every slot re-registers between jobs, for a few seconds, so look again before acting.
    sleep "${RECHECK_SECONDS:-90}"
    n="$(online)"
    [ "$n" -gt 0 ] && { echo "$n online on the second look"; exit 0; }
    set_gate auto-off
    echo "::warning::no build-host runner online: jobs go to hosted runners until one is back"
    requeue_stuck_runs
    ;;
  auto-off)
    if [ "$n" -gt 0 ]; then set_gate on; fi
    ;;
  *)
    echo "the gate is '$gate', which the watchdog leaves alone"
    ;;
esac
