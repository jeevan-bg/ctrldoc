#!/usr/bin/env bash
# check-context-budget.sh — pre-commit / pre-wake budget gate.
#
# Mechanical enforcement of .ctrldoc/CONTEXT_POLICY.md token caps. Fails
# (exit 1) if any Tier 0 file has grown past its cap. The loop's hygiene
# rules in LOOP_PROMPT.md Step 6 are procedural; this script is the
# fall-back hard gate.
#
# Caps:
#   .ctrldoc/STATE.md          — slice-log entries ≤ 20
#   .ctrldoc/ROADMAP.md        — trailing `done` rows above first `pending` ≤ 5
#   docs/SPEC_DIGEST.md        — total lines ≤ 130
#
# Also runs the shard --check scripts to ensure docs/spec/ and
# docs/decisions/ are in sync with their canonical write-targets.
#
# Usage:   scripts/check-context-budget.sh
#          (intended as pre-commit hook and pre-wake gate; exit 1 = fail)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

FAIL=0

# Initialise summary counters so the final echo does not trip `set -u`
# if a Tier 0 file is absent (e.g. running from a fresh git worktree
# before .ctrldoc/ is materialised). Per-file blocks below overwrite.
state_entries=0
trailing_done=0
digest_lines=0

# ----- STATE.md slice-log cap (≤ 20 entries) -----
STATE_FILE=".ctrldoc/STATE.md"
if [[ -f "$STATE_FILE" ]]; then
  state_entries="$(grep -cE '^20[0-9]{2}-' "$STATE_FILE" || true)"
  if [[ "$state_entries" -gt 20 ]]; then
    echo "FAIL: $STATE_FILE has $state_entries slice-log entries (cap is 20)." >&2
    echo "  → roll oldest into .ctrldoc/SESSIONS/history.md (LOOP_PROMPT Step 6.5)." >&2
    FAIL=1
  fi
fi

# ----- ROADMAP.md trailing-done cap (≤ 5) -----
# Count consecutive `done` rows that appear before the first `pending` row.
ROADMAP_FILE=".ctrldoc/ROADMAP.md"
if [[ -f "$ROADMAP_FILE" ]]; then
  trailing_done="$(awk '
    /\| pending \|/ { found_pending = 1; exit }
    /\| done \|/    { count++ }
    END { print count + 0 }
  ' "$ROADMAP_FILE")"
  if [[ "$trailing_done" -gt 5 ]]; then
    echo "FAIL: $ROADMAP_FILE has $trailing_done trailing 'done' rows before first 'pending' (cap is 5)." >&2
    echo "  → archive oldest into .ctrldoc/ROADMAP_DONE.md (LOOP_PROMPT Step 6.2)." >&2
    FAIL=1
  fi
fi

# ----- SPEC_DIGEST.md line cap (≤ 130) -----
DIGEST_FILE="docs/SPEC_DIGEST.md"
if [[ -f "$DIGEST_FILE" ]]; then
  digest_lines="$(wc -l < "$DIGEST_FILE" | tr -d ' ')"
  if [[ "$digest_lines" -gt 130 ]]; then
    echo "FAIL: $DIGEST_FILE has $digest_lines lines (cap is 130 — it is the Tier 0 cacheable proxy)." >&2
    echo "  → tighten the digest; the full spec lives in docs/SPEC.md / docs/spec/*.md." >&2
    FAIL=1
  fi
fi

# ----- Shard freshness -----
if [[ -x scripts/dev/shard-spec.sh ]]; then
  if ! scripts/dev/shard-spec.sh --check > /dev/null 2>&1; then
    echo "FAIL: docs/spec/ shards are stale relative to docs/SPEC.md." >&2
    echo "  → run scripts/dev/shard-spec.sh" >&2
    FAIL=1
  fi
fi
if [[ -x scripts/dev/shard-decisions.sh ]]; then
  if ! scripts/dev/shard-decisions.sh --check > /dev/null 2>&1; then
    echo "FAIL: docs/decisions/ shards are stale relative to docs/DECISIONS.md." >&2
    echo "  → run scripts/dev/shard-decisions.sh" >&2
    FAIL=1
  fi
fi

if [[ "$FAIL" -ne 0 ]]; then
  echo "" >&2
  echo "Context budget violations detected. Fix before committing." >&2
  exit 1
fi

echo "context budget OK (STATE log $state_entries/20 · ROADMAP trailing done $trailing_done/5 · DIGEST $digest_lines/130 lines · shards fresh)"
exit 0
