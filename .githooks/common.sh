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

# True when the checkout being committed to is the one that holds these hooks: its physical top
# level is the directory above $1. Only then do the hooks run code from it (cargo fmt, the plan
# tools, naming_check.py), which comes from the same checkout as the hooks themselves. Hooks that
# core.hooksPath points at from anywhere else run no code from there: not another repository, not
# another worktree of this one, and not a directory whose .git file points at this one.
same_checkout() {
    here=$(top=$(git rev-parse --show-toplevel) && cd "$top" && pwd -P) || return 1
    theirs=$(cd "$1/.." && pwd -P) || return 1
    [ "$here" = "$theirs" ]
}
