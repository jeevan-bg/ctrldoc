#!/usr/bin/env bash
# Run the remaining three smoke audits sequentially.
# Audit 01 already ran; we wire 02, 03, 04 here.
set -uo pipefail

cd /Users/jeevan/Documents/ctrldoc

PHASE0=/Users/jeevan/Downloads/ctrlmatrix_phase0
TARGET=/Users/jeevan/Downloads/THREAT_MODEL_v1_4.md

declare -a JOBS=(
  "02|${PHASE0}/02_trust_assumptions.md"
  "03|${PHASE0}/03_property_catalog.md"
  "04|${PHASE0}/04_exclusions_and_L1_contract.md"
)

for entry in "${JOBS[@]}"; do
  N="${entry%%|*}"
  CHECK="${entry##*|}"
  echo "=== audit ${N}: $(basename "${CHECK}") @ $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  rm -rf "runs/cli_smoke/${N}"
  .venv/bin/python -m ctrldoc --profile thrifty --max-cost-usd 2.00 --format json audit \
    --checklist "${CHECK}" \
    --target "${TARGET}" \
    --doc-id threat_model_v1_4 \
    --output-dir "runs/cli_smoke/${N}" \
    > "runs/cli_smoke/${N}_stdout.json" \
    2> "runs/cli_smoke/${N}_stderr.log"
  echo "    exit=$?"
  python_summary=$(.venv/bin/python -c "
import json
p = json.load(open('runs/cli_smoke/${N}_stdout.json'))
print('items=%d summary=%s' % (p['items_total'], p['summary']))
" 2>&1 || echo "summary-failed")
  echo "    ${python_summary}"
done

echo "=== all audits done @ $(date -u +%Y-%m-%dT%H:%M:%SZ)"
