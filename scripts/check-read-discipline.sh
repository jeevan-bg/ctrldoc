#!/usr/bin/env bash
# check-read-discipline.sh — PreToolUse hook enforcing context-bloat discipline.
#
# Refuses linear `Read` of mega-files (existing or future-grown) unless
# offset+limit is supplied. Redirects callers to the appropriate shard
# directory, slash command, or `grep`.
#
# Hook contract (PreToolUse): reads JSON from stdin describing the tool
# call; non-zero exit blocks the call; zero exit allows. stderr is surfaced
# to the operator.
#
# Gated files (linear Read refused; offset+limit always allowed; Tier 3 files
# refused linearly even with offset+limit is not enforced here — those need
# explicit operator intent):
#
#   docs/SPEC.md                       → read docs/spec/<N.M>.md (shard) | grep
#   docs/DECISIONS.md                  → read docs/decisions/ADR-NNNN.md  | grep
#   docs/SPEC_TRACE.md                 → grep | offset+limit
#   CHANGELOG.md                       → grep | offset+limit (per release)
#   .ctrldoc/ROADMAP_DONE.md           → grep (Tier 3 — historical)
#   .ctrldoc/SESSIONS/history.md       → grep (Tier 3 — historical)
#   .ctrldoc/SPEC_v0.3_ARCHIVE.md      → grep (Tier 3 — archived spec)
#   .ctrldoc/SPEC_ORIGINAL.md          → grep (Tier 3 — older archived spec)
#
# Files NOT gated (small, stable, or fully loaded by design — see
# .ctrldoc/CONTEXT_POLICY.md):
#
#   docs/SPEC_DIGEST.md, LICENSE, CONTRIBUTING.md, .ctrldoc/LOOP_PROMPT.md,
#   .ctrldoc/WAYS_OF_WORKING.md, .ctrldoc/CONTEXT_POLICY.md, .ctrldoc/STATE.md
#   (capped via budget), .ctrldoc/ROADMAP.md (capped via budget),
#   .ctrldoc/BUDGET.md, .ctrldoc/PERSONAS.md, .ctrldoc/RESUME.md,
#   .ctrldoc/POST_V0_1_0.md, docs/ARCHITECTURE.md (small), docs/TESTING.md
#   (small).

set -euo pipefail

INPUT="$(cat)"
tool_name=$(printf '%s' "$INPUT" | sed -n 's/.*"tool_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')

if [[ "$tool_name" != "Read" ]]; then
  exit 0
fi

file_path=$(printf '%s' "$INPUT" | sed -n 's/.*"file_path"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
has_offset=$(printf '%s' "$INPUT" | grep -c '"offset"' || true)
has_limit=$(printf '%s' "$INPUT" | grep -c '"limit"' || true)

# Offset+limit always allowed (caller has scoped intent).
if [[ "$has_offset" -gt 0 && "$has_limit" -gt 0 ]]; then
  exit 0
fi

case "$file_path" in
  */docs/SPEC.md|docs/SPEC.md)
    cat <<'EOF' >&2
REFUSED: linear Read of docs/SPEC.md (grows with v1 work) burns context.
Use one of:
  - Read docs/spec/<N.M>.md            # e.g. docs/spec/6.4.md for §6.4
  - cat docs/spec/INDEX.md             # for the section→file map
  - grep -n "<token>" docs/SPEC.md     # narrow lookup
  - Read docs/SPEC.md offset=N limit=M # explicit window if you must
See .ctrldoc/CONTEXT_POLICY.md (Tier 2 — load only the §X.Y a slice references).
EOF
    exit 2
    ;;
  */docs/DECISIONS.md|docs/DECISIONS.md)
    cat <<'EOF' >&2
REFUSED: linear Read of docs/DECISIONS.md (grows per ADR) burns context.
Use one of:
  - Read docs/decisions/ADR-NNNN.md          # e.g. docs/decisions/ADR-0002.md
  - cat docs/decisions/INDEX.md              # ADR list
  - grep -n "<token>" docs/DECISIONS.md      # narrow lookup
  - Read docs/DECISIONS.md offset=N limit=M  # explicit window
EOF
    exit 2
    ;;
  */docs/SPEC_TRACE.md|docs/SPEC_TRACE.md)
    cat <<'EOF' >&2
REFUSED: linear Read of docs/SPEC_TRACE.md (grows per slice) burns context.
Use one of:
  - grep -n "S-<NNN>" docs/SPEC_TRACE.md       # look up a specific slice
  - grep -n "§<X.Y>" docs/SPEC_TRACE.md        # look up a specific spec section
  - Read docs/SPEC_TRACE.md offset=N limit=M   # explicit window
EOF
    exit 2
    ;;
  */CHANGELOG.md|CHANGELOG.md)
    cat <<'EOF' >&2
REFUSED: linear Read of CHANGELOG.md (grows per release) burns context.
Use one of:
  - grep -n "^## " CHANGELOG.md                # release section index
  - grep -n "<token>" CHANGELOG.md             # narrow lookup
  - Read CHANGELOG.md offset=N limit=M         # explicit window
EOF
    exit 2
    ;;
  */.ctrldoc/ROADMAP_DONE.md)
    cat <<'EOF' >&2
REFUSED: linear Read of .ctrldoc/ROADMAP_DONE.md (Tier 3 historical archive).
This file is the v0.3 done-arc archive — the loop should not need to read it
unless a v1 slice explicitly cites a v0.3 predecessor. If so:
  - grep -n "S-<NNN>" .ctrldoc/ROADMAP_DONE.md
  - Read .ctrldoc/ROADMAP_DONE.md offset=N limit=M
Active roadmap lives in .ctrldoc/ROADMAP.md.
EOF
    exit 2
    ;;
  */.ctrldoc/SESSIONS/history.md)
    cat <<'EOF' >&2
REFUSED: linear Read of .ctrldoc/SESSIONS/history.md (Tier 3 archived slice log).
Use:
  - grep -n "S-<NNN>" .ctrldoc/SESSIONS/history.md
  - Read .ctrldoc/SESSIONS/history.md offset=N limit=M
Active slice log (last 20) lives in .ctrldoc/STATE.md.
EOF
    exit 2
    ;;
  */.ctrldoc/SPEC_v0.3_ARCHIVE.md|*/.ctrldoc/SPEC_ORIGINAL.md)
    cat <<'EOF' >&2
REFUSED: linear Read of archived spec (Tier 3 — frozen, large).
Use:
  - grep -n "<token>" <path>
  - Read <path> offset=N limit=M
The live spec is docs/SPEC.md (use its shards in docs/spec/).
EOF
    exit 2
    ;;
esac

exit 0
