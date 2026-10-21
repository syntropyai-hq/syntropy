# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Shared by the hooks in this directory, which source it. POSIX sh only.

# Prints the first of `python3` and `python` that runs Python 3.8 or later. Fails if neither
# does: a launcher stub that only says Python is not installed fails the version test too.
find_python() {
    for candidate in python3 python; do
        if command -v "$candidate" >/dev/null 2>&1 &&
            "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 8))' >/dev/null 2>&1; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

# True when the repository being committed to shares its git directory with the checkout that
# holds these hooks, the directory above $1. Only then do the hooks run code from that repository
# (cargo fmt, the plan tools, naming_check.py), which is then as trusted as the hooks themselves.
# Hooks that core.hooksPath points at from another repository run no code from it, as git itself
# never runs a repository's own hooks.
same_repository() {
    here=$(git rev-parse --path-format=absolute --git-common-dir) || return 1
    # git exports GIT_DIR and GIT_INDEX_FILE to hooks; clear them to ask about the hooks' checkout.
    theirs=$(
        unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_COMMON_DIR GIT_OBJECT_DIRECTORY
        git -C "$1/.." rev-parse --path-format=absolute --git-common-dir 2>/dev/null
    ) || return 1
    [ "$here" = "$theirs" ]
}
