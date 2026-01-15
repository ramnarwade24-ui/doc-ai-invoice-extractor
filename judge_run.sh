#!/usr/bin/env bash
set -euo pipefail

# Final judge runner (CPU-only, deterministic, offline-safe).
#
# Behavior:
# - Auto-detect dataset under data/pdfs
# - Run dataset_preflight.py (PDF openability + non-empty)
# - Run judge_mode.py in fast mode (no PaddleOCR unless explicitly enabled)
# - Print PASS/FAIL summary and exit 0/2

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INVOICES="${INVOICES:-data/pdfs}"
N="${N:-10}"
SEED="${SEED:-1337}"

cd "$REPO_ROOT"

if [[ ! -d "$INVOICES" ]]; then
  echo "[FAIL] Dataset folder not found: $INVOICES" >&2
  echo "       Expected PDFs under data/pdfs/." >&2
  exit 2
fi

set +e
python dataset_preflight.py --invoices "$INVOICES" --labels data/labels
PREFLIGHT_RC=$?
set -e

if [[ $PREFLIGHT_RC -ne 0 ]]; then
  echo "[FAIL] dataset_preflight.py failed (rc=$PREFLIGHT_RC)" >&2
  exit 2
fi

set +e
python judge_mode.py --invoices "$INVOICES" --n "$N" --seed "$SEED" --mode fast
JUDGE_RC=$?
set -e

if [[ $JUDGE_RC -eq 0 ]]; then
  echo "[PASS] Judge run succeeded." 
  echo "       Report: outputs/judge_report.json"
  exit 0
fi

echo "[FAIL] Judge run failed (rc=$JUDGE_RC)." >&2
echo "       Report: outputs/judge_report.json" >&2
exit 2
