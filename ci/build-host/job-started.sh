#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0

# The build host's job-started hook (ADR-0625 decision 6). The runner runs it before the first
# step of every job, with the job's context, and fails the job without running a step when it
# exits non-zero. It accepts only `pr.yml` on a `pull_request` from a branch of this repository,
# so a fork's PR that names the label, approved by mistake, still runs nothing here. The event
# payload is the runner's copy of GitHub's, written before any step.

set -euo pipefail

repo=syntropyai-hq/syntropy

refuse() {
  echo "::error::the build host refuses this job: $*; it runs only $repo's pr.yml for a PR from a branch of $repo (ADR-0625)"
  exit 1
}

[ "${GITHUB_REPOSITORY:-}" = "$repo" ] || refuse "repository '${GITHUB_REPOSITORY:-}'"
[ "${GITHUB_EVENT_NAME:-}" = pull_request ] || refuse "event '${GITHUB_EVENT_NAME:-}'"
case "${GITHUB_WORKFLOW_REF:-}" in
  "$repo/.github/workflows/pr.yml@"*) ;;
  *) refuse "workflow '${GITHUB_WORKFLOW_REF:-}'" ;;
esac
[ -r "${GITHUB_EVENT_PATH:-}" ] || refuse "no event payload"
head="$(jq -r '.pull_request.head.repo.full_name // empty' "$GITHUB_EVENT_PATH")"
base="$(jq -r '.pull_request.base.repo.full_name // empty' "$GITHUB_EVENT_PATH")"
[ "$head" = "$repo" ] && [ "$base" = "$repo" ] || refuse "PR from '${head:-unknown}' into '${base:-unknown}'"
echo "build host: $repo pr.yml, PR #$(jq -r '.pull_request.number' "$GITHUB_EVENT_PATH") from a branch of $repo"
