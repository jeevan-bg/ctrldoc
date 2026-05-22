#!/usr/bin/env bash
# Public-leak scan: ensure internal/process language does not appear in public files.
# Fails the build if any forbidden pattern matches outside ignored paths.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Patterns that must never appear in public files.
# Case-insensitive whole-word matches where possible.
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

# Patterns that are explicitly ALLOWED (whitelist overrides above).
# Example: `.ctrldoc` is just a folder name, not sensitive content.

# Public paths to scan (everything except gitignored internal dirs).
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
      echo "LEAK: pattern '$p' found in $path:"
      grep -rniE "${EXCLUDES[@]}" "$p" "$path" || true
      FAIL=1
    fi
  done
done

if [ "$FAIL" -eq 1 ]; then
  echo ""
  echo "Public-leak scan FAILED. Remove internal/process language from public files."
  exit 1
fi

echo "Public-leak scan OK."
