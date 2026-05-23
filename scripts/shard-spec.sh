#!/usr/bin/env bash
# shard-spec.sh — split docs/SPEC.md into docs/spec/<section>.md fragments.
#
# Generates a per-section read-view of docs/SPEC.md so the build loop's
# Tier-2 step can load only the §X.Y a slice references via
# `Read("docs/spec/6.4.md")` instead of offset/limit math on the mega-file.
#
# docs/SPEC.md remains the canonical write-target. This script only WRITES
# fragments; it never modifies docs/SPEC.md (byte-preservation discipline —
# checked in --check mode via md5).
#
# Usage:   scripts/shard-spec.sh          — regenerate docs/spec/*.md
#          scripts/shard-spec.sh --check  — verify idempotency without modifying tree
#
# Invariants:
# * Every `^## N` or `^### N.M` header in docs/SPEC.md (where N/M are digits)
#   produces exactly one docs/spec/<slug>.md fragment.
# * Slug is the section number with dots preserved: "6.4", "4.1", "13".
# * Each fragment carries frontmatter: id, source-line, header-level.
# * docs/spec/INDEX.md is regenerated as a flat table of contents.
# * Idempotent: running twice produces byte-identical output.
# * Pure bash + awk; no Python / Node.
# * docs/SPEC.md md5 MUST be unchanged after a run.

set -euo pipefail

CHECK_MODE=0
case "${1:-}" in
  --check) CHECK_MODE=1 ;;
  "")      ;;
  *)       echo "Usage: $0 [--check]" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$REPO_ROOT/docs/SPEC.md"
OUT_DIR="$REPO_ROOT/docs/spec"

if [[ ! -f "$SRC" ]]; then
  echo "ERROR: $SRC not found" >&2
  exit 1
fi

# Capture source md5 to enforce byte-preservation.
SRC_MD5_BEFORE="$(md5 -q "$SRC" 2>/dev/null || md5sum "$SRC" | awk '{print $1}')"

if [[ "${CHECK_MODE}" -eq 1 ]]; then
  TARGET_DIR="$(mktemp -d)"
else
  TARGET_DIR="$OUT_DIR"
  mkdir -p "$TARGET_DIR"
  # Clean stale fragments before regenerating (handles renamed/deleted sections).
  find "$TARGET_DIR" -maxdepth 1 -name '*.md' -delete
fi

awk -v out_dir="$TARGET_DIR" '
function slug(s,    t) {
  t = s
  gsub(/[^A-Za-z0-9._-]/, "-", t)
  gsub(/-+/, "-", t)
  sub(/^-/, "", t); sub(/-$/, "", t)
  return t
}
function flush() {
  if (current_id != "" && n > 0) {
    fname = out_dir "/" current_id ".md"
    printf "---\nid: §%s\nsource: docs/SPEC.md\nsource_line: %d\nheader_level: %d\n---\n\n", \
      current_id, current_start, current_level > fname
    for (i = 1; i <= n; i++) print buf[i] >> fname
    close(fname)
  }
}
# Match section headers: `## N.` or `### N.M` (digits, optional sub-number).
/^#+[ \t]+[0-9]+(\.[0-9]+)?[\.\)]?[ \t]/ {
  flush()
  delete buf
  n = 0
  match($0, /^#+/)
  current_level = RLENGTH
  # Capture the section number token (second whitespace-delimited field).
  num = $2
  # Strip trailing punctuation (".", ")").
  sub(/[\.\)]+$/, "", num)
  current_id = slug(num)
  current_start = NR
  buf[++n] = $0
  next
}
# Any other header at any level closes the current fragment cleanly.
/^#+[ \t]/ {
  flush()
  delete buf
  n = 0
  current_id = ""
  next
}
{
  if (current_id != "") buf[++n] = $0
}
END { flush() }
' "$SRC"

# Pass 2: emit INDEX.md
INDEX="$TARGET_DIR/INDEX.md"
{
  echo "# docs/spec/ Index — auto-generated"
  echo ""
  echo "**Generator:** \`scripts/shard-spec.sh\`."
  echo "**Source:** \`docs/SPEC.md\` (canonical write-target)."
  echo "**Cadence:** regenerate whenever \`docs/SPEC.md\` is edited (pre-commit \`shard-spec --check\` enforces freshness)."
  echo ""
  echo "Each fragment is the random-access read-view of one section of \`docs/SPEC.md\`. The loop's Tier-2 step reads exactly the section a slice's SPEC-REF cites, e.g. \`Read(\"docs/spec/6.4.md\")\` for §6.4."
  echo ""
  echo "| Fragment | § | Source line | Header level |"
  echo "|---|---|---|---|"
} > "$INDEX"

TMPLIST="$INDEX.tmp"
: > "$TMPLIST"
for f in "$TARGET_DIR"/*.md; do
  [[ -e "$f" ]] || continue
  base="$(basename "$f" .md)"
  [[ "$base" == "INDEX" ]] && continue
  src_line="$(awk '/^source_line:/ { print $2; exit }' "$f")"
  level="$(awk '/^header_level:/ { print $2; exit }' "$f")"
  printf "%010d|| [\`%s.md\`](./%s.md) | §%s | %s | %s |\n" \
    "$src_line" "$base" "$base" "$base" "$src_line" "$level" >> "$TMPLIST"
done
sort "$TMPLIST" 2>/dev/null | sed 's/^[0-9]*|//' >> "$INDEX" || true
rm -f "$TMPLIST"

# Enforce byte-preservation on the source.
SRC_MD5_AFTER="$(md5 -q "$SRC" 2>/dev/null || md5sum "$SRC" | awk '{print $1}')"
if [[ "$SRC_MD5_BEFORE" != "$SRC_MD5_AFTER" ]]; then
  echo "ERROR: docs/SPEC.md changed during sharding — byte-preservation violated" >&2
  exit 3
fi

if [[ "${CHECK_MODE}" -eq 1 ]]; then
  if [[ -d "$OUT_DIR" ]] && diff -rq "$TARGET_DIR" "$OUT_DIR" > /dev/null 2>&1; then
    echo "OK: docs/spec/ is up-to-date"
    rm -rf "$TARGET_DIR"
    exit 0
  else
    echo "DIFF: docs/spec/ would change; run \`scripts/shard-spec.sh\` to regenerate" >&2
    diff -rq "$TARGET_DIR" "$OUT_DIR" 2>&1 | head -20 >&2 || true
    rm -rf "$TARGET_DIR"
    exit 1
  fi
fi

count="$(find "$OUT_DIR" -maxdepth 1 -name '*.md' ! -name 'INDEX.md' | wc -l | tr -d ' ')"
echo "wrote $count spec fragments + INDEX.md to $OUT_DIR/"
