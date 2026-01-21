#!/usr/bin/env bash
set -euo pipefail

# One-command offline EDA runner for jury evaluation.
# - Validates PNG-only dataset
# - Generates eda_report.json, eda_summary.csv, eda_profile.json
# - Exports eda_report.pdf

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${1:-outputs/eda}"

echo "[1/3] Dataset preflight (PNG-only)"
python dataset_preflight.py --invoices data/images --labels data/labels --out outputs/dataset_preflight.json

echo "[2/3] Running offline EDA -> ${OUT_DIR}"
python eda.py --images data/images --out "$OUT_DIR"

echo "[3/3] Exporting PDF report"
python eda.py --plot "$OUT_DIR/eda_report.json" --pdf "$OUT_DIR/eda_report.pdf"

echo "[OK] EDA artifacts written to: $OUT_DIR"
ls -1 "$OUT_DIR" | sed 's/^/ - /'
