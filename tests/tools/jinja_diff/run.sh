#!/usr/bin/env bash
# Differential test for the Jinja interpreter: renders every context under
# contexts/ with both our interpreter and Python's jinja2 and requires the two
# to be byte-identical. See docs/custom-chat-templates.md.
#
#   run.sh <build-dir> <template.jinja> [more templates...]
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
build="${1:?usage: run.sh <build-dir> <template.jinja> [...]}"
shift

renderer="$build/jinja_diff_render"
g++ -std=c++20 -O1 -I"$repo/src" -o "$renderer" \
    "$here/render.cpp" "$repo/src/targets/qwen3_6/impl/frontend/jinja.cpp" || exit 1

pass=0
fail=0
for template in "$@"; do
    for context in "$here"/contexts/*.json "$here"/context_*.json; do
        [ -e "$context" ] || continue
        py_out=$(python3 "$here/reference_render.py" "$template" "$context" 2>/dev/null)
        py_status=$?
        cc_out=$("$renderer" "$template" "$context" 2>/dev/null)
        cc_status=$?
        label="$(basename "$template" .jinja)/$(basename "$context" .json)"
        # Both sides rejecting the input counts as agreement.
        if [ $py_status -ne 0 ] && [ $cc_status -ne 0 ]; then
            pass=$((pass + 1))
        elif [ $py_status -eq $cc_status ] && [ "$py_out" = "$cc_out" ]; then
            pass=$((pass + 1))
        else
            printf 'FAIL %s (py=%s cc=%s)\n' "$label" "$py_status" "$cc_status"
            diff <(printf '%s' "$py_out") <(printf '%s' "$cc_out") | head -10
            fail=$((fail + 1))
        fi
    done
done

printf 'jinja differential: %d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
