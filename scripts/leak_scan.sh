#!/usr/bin/env bash
# Repository content linter.
# Fails the build if any disallowed pattern matches outside ignored paths.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Patterns disallowed in tracked files.
PATTERNS=(
  "claude code"
  "claude-code"
  "anthropic\\.com"
  "session [0-9]"
  "hardening session"
  "build session"
  "loop session"
  "made by claude"
  "generated with claude"
  "co-authored-by: claude"
  "co-authored by claude"
  "internal decision log"
  "this slice"
  "as agreed in session"
  "per the loop"
  "wake-?up"
  "co-authored-by: claude"
)

SCAN_PATHS=(
  "src"
  "tests"
  "docs"
  "examples"
  "scripts"
  "README.md"
  "CONTRIBUTING.md"
  "CHANGELOG.md"
  "LICENSE"
  "pyproject.toml"
  ".github"
)

# Allow the scan script itself and this file's patterns to mention themselves.
EXCLUDES=(
  "--exclude=leak_scan.sh"
  "--exclude=publish.sh"
)

FAIL=0
for p in "${PATTERNS[@]}"; do
  for path in "${SCAN_PATHS[@]}"; do
    [ -e "$path" ] || continue
    if grep -rniE "${EXCLUDES[@]}" "$p" "$path" >/dev/null 2>&1; then
      echo "DISALLOWED: pattern '$p' found in $path:"
      grep -rniE "${EXCLUDES[@]}" "$p" "$path" || true
      FAIL=1
    fi
  done
done

if [ "$FAIL" -eq 1 ]; then
  echo ""
  echo "Content lint FAILED. See matches above."
  exit 1
fi

echo "Content lint OK."
