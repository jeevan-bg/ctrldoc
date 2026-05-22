#!/usr/bin/env bash
# Push tracked files to the GitHub remote `public`, after a content lint.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
  DRY_RUN=1
fi

echo "Step 1/2: content lint"
bash scripts/leak_scan.sh

echo "Step 2/2: push to public remote"
if [ "$DRY_RUN" -eq 1 ]; then
  git push --dry-run public main
else
  git push public main
fi

echo "Publish complete."
