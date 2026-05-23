#!/usr/bin/env bash
# post-edit-shard.sh — PostToolUse hook that auto-regenerates spec/decisions
# shards when their canonical source files are edited.
#
# Trigger: a successful Edit / Write on `docs/SPEC.md` or `docs/DECISIONS.md`.
# Action: run the matching shard generator. Fails silently — the next
# pre-commit `shard-spec --check` / `shard-decisions --check` is the
# blocking gate. This is an automation convenience, not enforcement.
#
# Hook contract (PostToolUse): reads JSON from stdin; never blocks (exit 0
# always); side effect is the shard regeneration.

set -euo pipefail

INPUT="$(cat)"

tool_name=$(printf '%s' "$INPUT" | sed -n 's/.*"tool_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
case "$tool_name" in
  Edit|Write) ;;
  *) exit 0 ;;
esac

# Extract the file path of the edited file.
if command -v jq >/dev/null 2>&1; then
  file_path=$(printf '%s' "$INPUT" | jq -r '.tool_input.file_path // empty')
else
  file_path=$(printf '%s' "$INPUT" | sed -n 's/.*"file_path"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
fi

if [[ -z "$file_path" ]]; then
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

case "$file_path" in
  */docs/SPEC.md|docs/SPEC.md)
    if [[ -x "$REPO_ROOT/scripts/dev/shard-spec.sh" ]]; then
      "$REPO_ROOT/scripts/dev/shard-spec.sh" >/dev/null 2>&1 || true
    fi
    ;;
  */docs/DECISIONS.md|docs/DECISIONS.md)
    if [[ -x "$REPO_ROOT/scripts/dev/shard-decisions.sh" ]]; then
      "$REPO_ROOT/scripts/dev/shard-decisions.sh" >/dev/null 2>&1 || true
    fi
    ;;
esac

exit 0
