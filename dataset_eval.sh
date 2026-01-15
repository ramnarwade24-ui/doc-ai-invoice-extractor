#!/usr/bin/env bash
set -euo pipefail

# Dataset pipeline runner.
#
# Default is the judge-safe path (fast, offline-safe, deterministic).
# To attempt accuracy-oriented OCR (requires local PaddleOCR models), use:
#   ./dataset_eval.sh accurate
# or:
#   PROFILE=accurate ./dataset_eval.sh

PROFILE="${1:-${PROFILE:-fast}}"
SEED="${SEED:-1337}"
INVOICES="${INVOICES:-data/pdfs}"
LABELS="${LABELS:-data/labels}"
EVAL_LIMIT="${EVAL_LIMIT:-50}"
TUNING_LIMIT="${TUNING_LIMIT:-50}"

if [[ "$PROFILE" != "fast" && "$PROFILE" != "accurate" ]]; then
  echo "[FAIL] PROFILE must be 'fast' or 'accurate' (got: $PROFILE)" >&2
  exit 2
fi

MODE="$PROFILE"
ENABLE_PADDLEOCR_ARGS=()
EVAL_DPI=150
EVAL_MAX_PAGES=1

if [[ "$PROFILE" == "accurate" ]]; then
  MODE="accurate"
  ENABLE_PADDLEOCR_ARGS=(--enable-paddleocr)
  EVAL_DPI=200
  EVAL_MAX_PAGES=5
fi

echo "[INFO] PROFILE=$PROFILE SEED=$SEED INVOICES=$INVOICES LABELS=$LABELS"
echo "[INFO] EVAL_LIMIT=$EVAL_LIMIT TUNING_LIMIT=$TUNING_LIMIT"

# 1) Convert images -> PDFs (optional; no-op if data/images missing)
python convert_images_to_pdf.py --input data/images --output data/pdfs || true

# 2) Dataset preflight (PDF openability + non-empty)
python dataset_preflight.py --invoices "$INVOICES" --labels "$LABELS"

# 3) Evaluate (accuracy only if labels exist)
python doc_ai_project/eval.py \
  --invoices "$INVOICES" \
  --labels "$LABELS" \
  --dpi "$EVAL_DPI" \
  --max-pages "$EVAL_MAX_PAGES" \
  --limit "$EVAL_LIMIT" \
  --run-mode submission \
  --seed "$SEED"

# 4) Tuning + selector (optional; requires labels)
if [[ -d "$LABELS" ]] && compgen -G "$LABELS/*.json" > /dev/null; then
  if [[ "${RUN_TUNING:-0}" == "1" ]]; then
    python doc_ai_project/tuning.py --invoices "$INVOICES" --labels "$LABELS" --seed "$SEED" --limit "$TUNING_LIMIT"
    python doc_ai_project/selector.py --tuning-report outputs/tuning_report.json --out-config best_config.json \
      --invoices "$INVOICES" --labels "$LABELS" --seed "$SEED" --limit "$EVAL_LIMIT"
  else
    echo "[INFO] Skipping tuning/selector (set RUN_TUNING=1 to enable)."
  fi
else
  echo "[INFO] Labels not present; skipping tuning/selector."
fi

# 5) Judge simulation (latency/cost/schema + optional accuracy)
python judge_mode.py --invoices "$INVOICES" --n 10 --seed "$SEED" --mode "$MODE" "${ENABLE_PADDLEOCR_ARGS[@]}"

# 6) Submission-style single run (writes JSON output; defaults are judge-safe)
# Replace invoice.pdf with a real path if needed.
if compgen -G "$INVOICES/*.pdf" > /dev/null; then
  SAMPLE_PDF="$(ls -1 "$INVOICES"/*.pdf | head -n 1)"
  python doc_ai_project/executable.py --pdf "$SAMPLE_PDF" --out outputs/submission_sample.json "${ENABLE_PADDLEOCR_ARGS[@]}" || true
  echo "[OK] Wrote submission sample: doc_ai_project/outputs/submission_sample.json"
else
  echo "[WARN] No PDFs found for submission sample step."
fi

echo "[OK] Dataset pipeline complete."
