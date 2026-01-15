#!/usr/bin/env bash
set -euo pipefail

# Submission packaging script.
#
# - Runs final_submission_check.py gate (schema/latency/cost/determinism; accuracy if labels exist)
# - Builds outputs/submission.zip
# - Prints checksum + size
#
# Offline-safe by default (PaddleOCR is disabled unless explicitly enabled).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INVOICES="${INVOICES:-data/pdfs}"
LABELS="${LABELS:-data/labels}"
SEED="${SEED:-1337}"
N="${N:-10}"

cd "$REPO_ROOT"

# Pick a sample PDF for submission schema validation.
SAMPLE_PDF=""
if compgen -G "$INVOICES/**/*.pdf" > /dev/null; then
  SAMPLE_PDF="$(find "$INVOICES" -type f -name '*.pdf' | sort | head -n 1)"
elif compgen -G "$INVOICES/*.pdf" > /dev/null; then
  SAMPLE_PDF="$(ls -1 "$INVOICES"/*.pdf | sort | head -n 1)"
fi

if [[ -z "$SAMPLE_PDF" ]]; then
  echo "[FAIL] No PDFs found under $INVOICES" >&2
  exit 2
fi

# Choose config for final check (prefer frozen final_config.json if present).
CONFIG="best_config.json"
if [[ -f "doc_ai_project/outputs/final_config.json" ]]; then
  CONFIG="outputs/final_config.json"
fi

echo "[INFO] Running final submission gate..."
set +e
if [[ -d "$LABELS" ]] && compgen -G "$LABELS/*.json" > /dev/null; then
  python final_submission_check.py --invoices "$INVOICES" --labels "$LABELS" --config "$CONFIG" -n "$N" --seed "$SEED"
else
  python final_submission_check.py --invoices "$INVOICES" --config "$CONFIG" -n "$N" --seed "$SEED"
fi
GATE_RC=$?
set -e

if [[ $GATE_RC -ne 0 ]]; then
  echo "[FAIL] final_submission_check.py failed (rc=$GATE_RC)." >&2
  echo "       See: doc_ai_project/outputs/final_submission_report.json" >&2
  exit 2
fi

echo "[INFO] Building outputs/submission.zip ..."
mkdir -p outputs

# Build zip using the project submission builder, but write to repo-root outputs/.
python doc_ai_project/submission.py \
  --out "../outputs/submission.zip" \
  --pdf "$SAMPLE_PDF" \
  --seed "$SEED"

ZIP_PATH="outputs/submission.zip"
if [[ ! -f "$ZIP_PATH" ]]; then
  echo "[FAIL] Missing zip at $ZIP_PATH" >&2
  exit 2
fi

SHA="$(sha256sum "$ZIP_PATH" | awk '{print $1}')"
SIZE_BYTES="$(stat -c%s "$ZIP_PATH")"

echo "[OK] Built: $ZIP_PATH"
echo "[OK] SHA256: $SHA"
echo "[OK] Size: ${SIZE_BYTES} bytes"
