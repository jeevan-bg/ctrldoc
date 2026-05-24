#!/usr/bin/env bash
# Real-doc shakedown smoke.
#
# Drives every entry in the real-doc corpus
# (tests/fixtures/real_docs/MANIFEST.yaml) through the v1 substrate
# on the heuristic profile (no LLM, no Ollama, no network), validates
# ingest determinism by re-ingesting each doc into a sibling tree,
# and builds a workspace from the spec-vs-impl pair.
#
# The script delegates to the Python driver
# `ctrldoc.eval.real_doc_smoke`, which writes a summary JSON the
# script then validates. The driver is the source of truth — this
# shell wrapper exists to make the smoke invokable from CI and
# `scripts/` without a Python entry point.
#
# Usage:
#   scripts/real_doc_smoke.sh
#   scripts/real_doc_smoke.sh --output-root <dir> --summary-path <file>
#
# Exits 0 on success, non-zero on any substrate failure.

set -uo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -x .venv/bin/python ]]; then
  PY=".venv/bin/python"
else
  PY="python"
fi

OUTPUT_ROOT=""
SUMMARY_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --summary-path)
      SUMMARY_PATH="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '2,20p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${OUTPUT_ROOT}" ]]; then
  OUTPUT_ROOT="${REPO_ROOT}/runs/real_doc_smoke"
fi
if [[ -z "${SUMMARY_PATH}" ]]; then
  SUMMARY_PATH="${OUTPUT_ROOT}/summary.json"
fi

mkdir -p "${OUTPUT_ROOT}"

echo "=== ctrldoc real-doc smoke @ $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "    output_root=${OUTPUT_ROOT}"
echo "    summary=${SUMMARY_PATH}"
echo

"${PY}" -m ctrldoc.eval.real_doc_smoke \
  --output-root "${OUTPUT_ROOT}" \
  --summary-path "${SUMMARY_PATH}"
driver_rc=$?

if [[ ${driver_rc} -ne 0 ]]; then
  echo "=== driver exited ${driver_rc}; smoke FAILED"
  exit ${driver_rc}
fi

if [[ ! -f "${SUMMARY_PATH}" ]]; then
  echo "=== driver did not write ${SUMMARY_PATH}; smoke FAILED"
  exit 1
fi

# Surface a one-line per-substrate report so CI logs are readable.
"${PY}" - "${SUMMARY_PATH}" <<'PY'
import json
import sys

summary_path = sys.argv[1]
with open(summary_path, encoding="utf-8") as handle:
    payload = json.load(handle)

print(f"ingest_count        = {payload['ingest_count']}")
print(f"scan_count          = {payload['scan_count']}")
print(f"workspace_doc_count = {payload['workspace_doc_count']}")
print(f"determinism_ok      = {payload['determinism_ok']}")
print()
print("per-doc ingest signatures:")
for row in payload["ingests"]:
    print(
        f"  {row['doc_id']:<28} type={row['type']:<14} "
        f"sections={row['sections_parsed']:<3} chunks={row['chunks_indexed']:<3} "
        f"entities={row['entities_indexed']:<3} sig={row['signature_hash'][:12]}"
    )

if not payload["determinism_ok"]:
    print("\nFAIL: ingest signatures did not match across the determinism rerun")
    sys.exit(2)
PY
report_rc=$?

if [[ ${report_rc} -ne 0 ]]; then
  exit ${report_rc}
fi

echo
echo "=== real-doc smoke PASSED"
