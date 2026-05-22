#!/usr/bin/env bash
# Download public-domain / open-license fixtures for testing.
# Files land in tests/fixtures/downloaded/ which is gitignored.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/tests/fixtures/downloaded"
mkdir -p "$OUT"

# 1. RFC 9110 — HTTP Semantics (public domain, IETF).
if [ ! -f "$OUT/rfc9110.txt" ]; then
  echo "Fetching RFC 9110 ..."
  curl -fsSL "https://www.rfc-editor.org/rfc/rfc9110.txt" -o "$OUT/rfc9110.txt"
fi

# 2. Project Gutenberg — "On the Origin of Species" by Darwin (public domain).
if [ ! -f "$OUT/origin_of_species.txt" ]; then
  echo "Fetching Origin of Species ..."
  curl -fsSL "https://www.gutenberg.org/cache/epub/1228/pg1228.txt" -o "$OUT/origin_of_species.txt"
fi

# 3. arXiv "Attention Is All You Need" (open-access).
if [ ! -f "$OUT/attention.pdf" ]; then
  echo "Fetching Attention Is All You Need (PDF) ..."
  curl -fsSL "https://arxiv.org/pdf/1706.03762" -o "$OUT/attention.pdf"
fi

cat > "$OUT/LICENSES.md" <<'EOF'
# Fixture licenses

- rfc9110.txt — IETF, public domain / Trust Legal Provisions.
- origin_of_species.txt — Project Gutenberg, public domain.
- attention.pdf — arXiv, open-access (CC-BY or author license).

These files are downloaded on demand and are not committed to the repository.
EOF

echo "Fixtures ready in $OUT"
