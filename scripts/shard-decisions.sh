#!/usr/bin/env bash
# shard-decisions.sh — split docs/DECISIONS.md into docs/decisions/ADR-NNNN.md
# fragments + INDEX.md.
#
# Generates a per-ADR read-view of docs/DECISIONS.md so the build loop's
# Tier-2 step can load only the ADR a slice references via
# `Read("docs/decisions/ADR-0002.md")` instead of the whole DECISIONS.md.
#
# docs/DECISIONS.md remains the canonical append-target. This script only
# WRITES fragments; it never modifies docs/DECISIONS.md (byte-preserved,
# checked via md5).
#
# Usage:   scripts/shard-decisions.sh          — regenerate docs/decisions/*.md
#          scripts/shard-decisions.sh --check  — verify idempotency
#
# Invariants:
# * Every `^## ADR-NNNN` header in docs/DECISIONS.md produces exactly one
#   docs/decisions/ADR-NNNN.md fragment.
# * Each fragment carries frontmatter: id, source-line, header-level.
# * docs/decisions/INDEX.md is regenerated as a flat TOC.
# * Idempotent: running twice produces byte-identical output.
# * Pure bash + awk.
# * docs/DECISIONS.md md5 MUST be unchanged after a run.

set -euo pipefail

CHECK_MODE=0
case "${1:-}" in
  --check) CHECK_MODE=1 ;;
  "")      ;;
  *)       echo "Usage: $0 [--check]" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$REPO_ROOT/docs/DECISIONS.md"
OUT_DIR="$REPO_ROOT/docs/decisions"

if [[ ! -f "$SRC" ]]; then
  echo "ERROR: $SRC not found" >&2
  exit 1
fi

SRC_MD5_BEFORE="$(md5 -q "$SRC" 2>/dev/null || md5sum "$SRC" | awk '{print $1}')"

if [[ "${CHECK_MODE}" -eq 1 ]]; then
  TARGET_DIR="$(mktemp -d)"
else
  TARGET_DIR="$OUT_DIR"
  mkdir -p "$TARGET_DIR"
  find "$TARGET_DIR" -maxdepth 1 -name '*.md' -delete
fi

awk -v out_dir="$TARGET_DIR" '
function slug(s,    t) {
  t = s
  gsub(/[^A-Za-z0-9_-]/, "-", t)
  gsub(/-+/, "-", t)
  sub(/^-/, "", t); sub(/-$/, "", t)
  return t
}
function flush() {
  if (current_id != "" && n > 0) {
    fname = out_dir "/" current_id ".md"
    printf "---\nid: %s\nsource: docs/DECISIONS.md\nsource_line: %d\nheader_level: %d\n---\n\n", \
      current_id, current_start, current_level > fname
    for (i = 1; i <= n; i++) print buf[i] >> fname
    close(fname)
  }
}
/^#+[ \t]+ADR-[0-9]+[a-z]*/ {
  flush()
  delete buf
  n = 0
  match($0, /^#+/)
  current_level = RLENGTH
  id = $2
  current_id = slug(id)
  current_start = NR
  buf[++n] = $0
  next
}
{
  if (current_id != "") buf[++n] = $0
}
END { flush() }
' "$SRC"

INDEX="$TARGET_DIR/INDEX.md"
{
  echo "# docs/decisions/ Index — auto-generated"
  echo ""
  echo "**Generator:** \`scripts/shard-decisions.sh\`."
  echo "**Source:** \`docs/DECISIONS.md\` (canonical append-target — public ADRs)."
  echo "**Cadence:** regenerate whenever \`docs/DECISIONS.md\` adds a new ADR (pre-commit \`shard-decisions --check\` enforces freshness)."
  echo ""
  echo "Each fragment is the random-access read-view of one ADR. The loop's Tier-2 step reads exactly the ADR a slice cites, e.g. \`Read(\"docs/decisions/ADR-0002.md\")\`."
  echo ""
  echo "| ADR | Source line |"
  echo "|---|---|"
} > "$INDEX"

TMPLIST="$INDEX.tmp"
: > "$TMPLIST"
for f in "$TARGET_DIR"/ADR-*.md; do
  [[ -e "$f" ]] || continue
  base="$(basename "$f" .md)"
  src_line="$(awk '/^source_line:/ { print $2; exit }' "$f")"
  printf "%010d|| [\`%s\`](./%s.md) | %s |\n" "$src_line" "$base" "$base" "$src_line" >> "$TMPLIST"
done
sort "$TMPLIST" 2>/dev/null | sed 's/^[0-9]*|//' >> "$INDEX" || true
rm -f "$TMPLIST"

SRC_MD5_AFTER="$(md5 -q "$SRC" 2>/dev/null || md5sum "$SRC" | awk '{print $1}')"
if [[ "$SRC_MD5_BEFORE" != "$SRC_MD5_AFTER" ]]; then
  echo "ERROR: docs/DECISIONS.md changed during sharding — byte-preservation violated" >&2
  exit 3
fi

if [[ "${CHECK_MODE}" -eq 1 ]]; then
  if [[ -d "$OUT_DIR" ]] && diff -rq "$TARGET_DIR" "$OUT_DIR" > /dev/null 2>&1; then
    echo "OK: docs/decisions/ is up-to-date"
    rm -rf "$TARGET_DIR"
    exit 0
  else
    echo "DIFF: docs/decisions/ would change; run \`scripts/shard-decisions.sh\` to regenerate" >&2
    diff -rq "$TARGET_DIR" "$OUT_DIR" 2>&1 | head -20 >&2 || true
    rm -rf "$TARGET_DIR"
    exit 1
  fi
fi

count="$(find "$OUT_DIR" -maxdepth 1 -name 'ADR-*.md' | wc -l | tr -d ' ')"
echo "wrote $count decision fragments + INDEX.md to $OUT_DIR/"
