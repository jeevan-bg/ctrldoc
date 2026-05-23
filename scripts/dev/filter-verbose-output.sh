#!/usr/bin/env bash
# filter-verbose-output.sh — PreToolUse Bash hook that trims known verbose
# tool output before the agent sees it. Offloads "scroll through huge output"
# work to a shell pipeline, reducing per-slice context cost.
#
# Pattern from https://code.claude.com/docs/en/costs#offload-processing-to-hooks-and-skills
#
# Filtered commands (only when not already piped by the caller):
#
#   pytest                       → tail -80   (keeps summary + any failure tracebacks)
#   ruff check / ruff format     → tail -50   (errors with filename:line:col; clean output is small)
#   mypy                         → tail -50
#   bash scripts/leak_scan.sh    → tail -10
#
# A command that already contains a pipe (`|`) is passed through unmodified —
# the caller has scoped intent. The hook never blocks; on no-match it exits 0
# silently and the original command runs as-is.
#
# Hook contract (PreToolUse): reads JSON from stdin describing the tool call;
# on rewrite, emits a hookSpecificOutput JSON object on stdout with
# updatedInput.command set; on pass-through, exits 0 with no output.

set -euo pipefail

INPUT="$(cat)"

# Only fire for Bash tool calls.
tool_name=$(printf '%s' "$INPUT" | sed -n 's/.*"tool_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
if [[ "$tool_name" != "Bash" ]]; then
  exit 0
fi

# Extract the command. Prefer jq if available; sed fallback for portability.
if command -v jq >/dev/null 2>&1; then
  cmd=$(printf '%s' "$INPUT" | jq -r '.tool_input.command // empty')
else
  cmd=$(printf '%s' "$INPUT" | sed -n 's/.*"command"[[:space:]]*:[[:space:]]*"\(.*\)"[[:space:]]*}[[:space:]]*}.*/\1/p')
fi

if [[ -z "$cmd" ]]; then
  exit 0
fi

# Skip if the caller already piped — they have explicit intent.
case "$cmd" in
  *"|"*) exit 0 ;;
esac

# Decide what to do based on the command shape.
tail_n=""
case "$cmd" in
  *"pytest"*)                                       tail_n=80  ;;
  *"ruff check"*|*"ruff format"*)                   tail_n=50  ;;
  *"mypy "*|*"mypy src"*|*"mypy tests"*)            tail_n=50  ;;
  *"bash scripts/leak_scan.sh"*)                    tail_n=10  ;;
esac

if [[ -z "$tail_n" ]]; then
  exit 0
fi

# Build the filtered command. `2>&1` so stderr is captured too.
filtered="$cmd 2>&1 | tail -${tail_n}"

# Emit hookSpecificOutput JSON. Use jq if available so escaping is robust.
if command -v jq >/dev/null 2>&1; then
  jq -nc \
    --arg cmd "$filtered" \
    '{hookSpecificOutput: {hookEventName: "PreToolUse", permissionDecision: "allow", updatedInput: {command: $cmd}}}'
else
  # Minimal manual escape: backslash, then quote.
  filtered_esc="${filtered//\\/\\\\}"
  filtered_esc="${filtered_esc//\"/\\\"}"
  printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"allow","updatedInput":{"command":"%s"}}}\n' "$filtered_esc"
fi
