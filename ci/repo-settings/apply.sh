#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Applies the repository settings and branch rulesets declared in this directory (QUAL-E1-T06;
# RFC-0062 §3.3 item 1, §8.3, §12.3, §12.6; ADR-0620 decision 4; ADR-0621 decision 2). Each
# JSON file is the body of one REST request, and `cargo xtask ci repo-settings` reads the same
# files to check the result.
#
# Maintainer action: only an organisation owner can change these settings. Agents run this
# script without `--apply` only.
#
# Usage: ci/repo-settings/apply.sh [--apply] [--repo <owner/name>]
#
#   Without --apply, it prints each request and changes nothing; it reads the existing rulesets
#   to say whether each would be created or updated. With --apply, it sends the requests with
#   `gh api` as the logged-in account, then runs `cargo xtask ci repo-settings`, which must pass.
#   The repository defaults to this checkout's. It needs `gh` and `jq`.
#
# Apply after `pr.yml` and `pr-meta.yml` are on `main`: the `main` ruleset requires `ci-gate`,
# `dco` and `pr-title`, which those workflows post. An open PR gets them on its next push.
#
# After applying, the remaining acceptance checks need a test account: a GitHub account that is
# not a maintainer and not an organisation owner. Delete what each step creates when it is done.
#
# 1. Force-push to `main` is rejected. Add the test account as a collaborator with the Write role.
#    As the test account, in a clone of the repository:
#        git fetch origin main
#        git push --force origin origin/main~1:refs/heads/main
#    Expect `GH013: Repository rule violations found` naming the non-fast-forward rule, and
#    `main` unchanged. If the push succeeds, the ruleset is not in force: restore `main` at once
#    from an owner account with `git push --force origin <the SHA main had>:refs/heads/main`,
#    then find out why `cargo xtask ci repo-settings` passed.
# 2. A fork's workflows wait for approval on every PR. Remove the test account as a collaborator.
#    As the test account, fork the repository and open a PR with a one-line change. Its `pr` and
#    `pr-meta` runs show "approval required" and do not start. Approve them as a maintainer, close
#    the PR, then open a second PR from the same fork: its runs wait for approval again.
# 3. `pr-title` trusts the base branch's roster. Give the test account the Triage role, so it can
#    apply labels. As the test account, push a branch that adds its own login to `maintainers` in
#    `governance/maintainers.toml`, open a PR from it titled `feat!: roster fixture`, and apply
#    `semver-major-approved`. `pr-title` fails, naming the test account as not a maintainer.
#    Close the PR, delete the branch and remove the Triage role.
# 4. Code-owner review, if `ruleset-main-code-owners.json` is kept: as the test account with the
#    Write role, open a PR that changes a file under `.github/`, and one that changes a file
#    `ci/agent-instructions.toml` lists, such as `AGENTS.md`. Neither can merge without a
#    maintainer's approval (QUAL-E1-T19). Close them and remove the role.
# 5. A new dependency needs a maintainer: as the test account with the Write role, open a PR that
#    adds a dev-dependency no manifest uses yet to one crate. `dep-diff` fails and the PR cannot
#    merge. A maintainer's approving review on its head commit turns `dep-diff` green, and a push
#    after it turns it red again. Close it and remove the role.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
apply=false
repo=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --apply) apply=true ;;
    --repo)
      [ "$#" -ge 2 ] || { echo "error: --repo needs a value" >&2; exit 2; }
      repo="$2"
      shift
      ;;
    --repo=*) repo="${1#--repo=}" ;;
    -h|--help) sed -n '4,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "error: unknown argument '$1'" >&2; exit 2 ;;
  esac
  shift
done

for tool in gh jq; do
  command -v "$tool" >/dev/null || { echo "error: $tool is not installed" >&2; exit 2; }
done
if [ -z "$repo" ]; then
  repo="$(cd "$root" && gh repo view --json nameWithOwner --jq .nameWithOwner)"
fi
if ! [[ "$repo" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]]; then
  echo "error: '$repo' is not a repository name of the form owner/name" >&2
  exit 2
fi

# Sends one request, or prints it on a dry run. `$3` is a body file, or empty for none.
send() {
  local method="$1" endpoint="$2" body="${3:-}"
  if [ -n "$body" ]; then
    echo "$method /$endpoint  <- ${body#"$here"/}"
  else
    echo "$method /$endpoint"
  fi
  if [ "$apply" = true ]; then
    local args=(api --method "$method" -H "Accept: application/vnd.github+json"
      -H "X-GitHub-Api-Version: 2022-11-28" "$endpoint")
    if [ -n "$body" ]; then
      args+=(--input "$body")
    fi
    gh "${args[@]}" >/dev/null
  fi
}

if [ "$apply" = true ]; then
  echo "Applying the settings in ${here#"$PWD"/} to $repo as $(gh api user --jq .login)."
else
  echo "Dry run for $repo: nothing is changed. Pass --apply to send these requests."
fi

# Merge settings and secret scanning with push protection.
send PATCH "repos/$repo" "$here/repository.json"

# Private vulnerability reporting has no body: PUT turns it on, DELETE turns it off.
if [ "$(jq -r '.enabled' "$here/private-vulnerability-reporting.json")" = true ]; then
  send PUT "repos/$repo/private-vulnerability-reporting"
else
  send DELETE "repos/$repo/private-vulnerability-reporting"
fi

# The Actions policy first: the allow-list can be set only once the policy is `selected`.
send PUT "repos/$repo/actions/permissions" "$here/actions-permissions.json"
if [ "$(jq -r '.allowed_actions' "$here/actions-permissions.json")" = selected ]; then
  send PUT "repos/$repo/actions/permissions/selected-actions" "$here/actions-selected-actions.json"
fi
# The default token for a workflow with no `permissions:` block: read-only, unable to approve.
send PUT "repos/$repo/actions/permissions/workflow" "$here/actions-workflow-permissions.json"
send PUT "repos/$repo/actions/permissions/fork-pr-contributor-approval" \
  "$here/fork-pr-contributor-approval.json"

# Each ruleset is matched to the repository's ruleset of the same name: updated if there is one,
# created if not.
existing="$(gh api --paginate "repos/$repo/rulesets?includes_parents=false" | jq -s 'add // []')"
for file in "$here"/ruleset-*.json; do
  name="$(jq -r '.name' "$file")"
  id="$(jq -r --arg name "$name" \
    '[.[] | select(.name == $name and .source_type == "Repository") | .id] | first // empty' \
    <<<"$existing")"
  if [ -n "$id" ]; then
    send PUT "repos/$repo/rulesets/$id" "$file"
  else
    send POST "repos/$repo/rulesets" "$file"
  fi
done

if [ "$apply" = true ]; then
  echo "Checking the result."
  (cd "$root" && cargo xtask ci repo-settings --repo "$repo")
fi
