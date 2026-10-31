#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0

# The build host's job-started hook (ADR-0625 decision 6). The runner runs it before the first
# step of every job, with the job's context, and fails the job without running a step when it
# exits non-zero. It accepts a workflow of this repository on one of the events its build-host
# jobs use, and refuses any event whose payload carries a pull request from another repository,
# so a fork's PR that names the label, approved by mistake, still runs nothing here.
# `issue_comment` is refused outright: its payload names a PR without its head repository, so a
# comment on a fork's PR could not be told apart. The event payload
# is the runner's copy of GitHub's, written before any step.

set -euo pipefail

repo=syntropyai-hq/syntropy

refuse() {
  echo "::error::the build host refuses this job: $*; it runs only $repo's workflows, and no fork's PR (ADR-0625)"
  exit 1
}

[ "${GITHUB_REPOSITORY:-}" = "$repo" ] || refuse "repository '${GITHUB_REPOSITORY:-}'"
case "${GITHUB_EVENT_NAME:-}" in
  pull_request|merge_group|push|schedule|workflow_dispatch) ;;
  *) refuse "event '${GITHUB_EVENT_NAME:-}'" ;;
esac
case "${GITHUB_WORKFLOW_REF:-}" in
  "$repo/.github/workflows/"*.yml@*) ;;
  *) refuse "workflow '${GITHUB_WORKFLOW_REF:-}'" ;;
esac
[ -r "${GITHUB_EVENT_PATH:-}" ] || refuse "no event payload"
if [ "$(jq -r 'has("pull_request")' "$GITHUB_EVENT_PATH")" = true ]; then
  head="$(jq -r '.pull_request.head.repo.full_name // empty' "$GITHUB_EVENT_PATH")"
  base="$(jq -r '.pull_request.base.repo.full_name // empty' "$GITHUB_EVENT_PATH")"
  [ "$head" = "$repo" ] && [ "$base" = "$repo" ] || refuse "PR from '${head:-unknown}' into '${base:-unknown}'"
  echo "build host: ${GITHUB_WORKFLOW_REF%%@*}, PR #$(jq -r '.pull_request.number' "$GITHUB_EVENT_PATH") from a branch of $repo"
else
  echo "build host: ${GITHUB_WORKFLOW_REF%%@*}, $GITHUB_EVENT_NAME"
fi
