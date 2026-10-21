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
