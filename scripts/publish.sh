#!/usr/bin/env bash
# Push the public tree to the GitHub remote `public`.
# .ctrldoc/ is gitignored so it never enters history.
# A leak scan runs first.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
  DRY_RUN=1
fi

echo "Step 1/3: leak scan"
bash scripts/leak_scan.sh

echo "Step 2/3: ensure .ctrldoc/ is not tracked"
if git ls-files --error-unmatch .ctrldoc/ >/dev/null 2>&1; then
  echo "FATAL: .ctrldoc/ is tracked in git. Aborting."
  exit 1
fi

echo "Step 3/3: push to public remote"
if [ "$DRY_RUN" -eq 1 ]; then
  git push --dry-run public main
else
  git push public main
fi

echo "Publish complete."
