#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "[demo] Converting images -> PDFs (data/images -> data/pdfs)"
python convert_images_to_pdf.py

echo "[demo] Preflight (fast sanity)"
python preflight.py --invoices data/pdfs

echo "[demo] Smoke test (10 deterministic PDFs)"
python smoke_test.py --invoices data/pdfs --n 10

echo "[demo] Demo runner (table + CSV)"
python demo_runner.py --invoices data/pdfs

echo "[demo] Launching Streamlit app"
exec streamlit run doc_ai_project/app.py
