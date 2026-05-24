#!/usr/bin/env bash
# Run every v1 eval-substrate baseline and aggregate their summary lines.
#
# Each `scripts/eval_v1_*.py` baseline drives a degenerate Protocol
# implementation through its substrate's runner, prints a single JSON
# summary line on stdout, and exits 0 on a clean run. This script
# invokes the five baselines sequentially, prints a per-substrate
# table, and exits non-zero if any baseline failed to execute or
# emit a parseable summary.
#
# Intended use:
#   - CI wiring check that the v1 substrates remain end-to-end driveable.
#   - Local baseline-vs-iteration delta inspection during S-125+ work.
#
# The script does NOT enforce release-gate thresholds — substrate
# `passed` flags are reported but never used as the script's exit
# code. Threshold enforcement is a release-time concern handled in
# the release smoke (Phase 23 / S-147), not in this baseline runner.

set -uo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -x .venv/bin/python ]]; then
  PY=".venv/bin/python"
else
  PY="python"
fi

BASELINES=(
  "eval_v1_claim_extraction.py"
  "eval_v1_cross_doc_coverage.py"
  "eval_v1_compare.py"
  "eval_v1_merge.py"
  "eval_v1_calibration.py"
)

echo "=== ctrldoc v1 eval-substrate baselines @ $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo

failures=0
declare -a SUMMARY_LINES=()

for baseline in "${BASELINES[@]}"; do
  script_path="scripts/${baseline}"
  if [[ ! -f "${script_path}" ]]; then
    echo "FAIL ${baseline}: script not found at ${script_path}"
    failures=$((failures + 1))
    continue
  fi
  raw=$("${PY}" "${script_path}" 2>&1)
  rc=$?
  if [[ ${rc} -ne 0 ]]; then
    echo "FAIL ${baseline}: exit ${rc}"
    echo "${raw}" | sed 's/^/    /'
    failures=$((failures + 1))
    continue
  fi
  # Validate the JSON line and extract `set_name` + `passed` for the table.
  parsed=$("${PY}" -c "
import json, sys
payload = json.loads(sys.argv[1].splitlines()[-1])
print(payload['set_name'], payload['cases'], payload['passed'])
" "${raw}" 2>/dev/null)
  if [[ -z "${parsed}" ]]; then
    echo "FAIL ${baseline}: unparseable summary line"
    echo "${raw}" | sed 's/^/    /'
    failures=$((failures + 1))
    continue
  fi
  set_name=$(echo "${parsed}" | awk '{print $1}')
  cases=$(echo "${parsed}" | awk '{print $2}')
  passed=$(echo "${parsed}" | awk '{print $3}')
  printf "OK   %-22s cases=%-4s baseline_passed=%s\n" "${set_name}" "${cases}" "${passed}"
  SUMMARY_LINES+=("${raw}")
done

echo
if [[ ${failures} -gt 0 ]]; then
  echo "=== ${failures} baseline(s) failed; smoke exit=1"
  exit 1
fi

echo "=== all 5 baselines emitted parseable summaries; smoke exit=0"
echo
echo "JSON summaries (one per substrate):"
for line in "${SUMMARY_LINES[@]}"; do
  echo "  ${line}"
done
