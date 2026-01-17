# Doc AI Invoice Extractor

An intelligent Document AI system for multilingual invoice field extraction built for large-scale evaluation and enterprise-grade robustness.

---

## 🚀 Overview

Doc AI Invoice Extractor is a production-ready Document AI pipeline designed to automatically extract structured fields from invoice PDFs such as:

- Dealer Name  
- Model Name  
- Horse Power  
- Asset Cost  
- Signature / Stamp Presence  

The system is built with a deterministic, CPU-only architecture and supports large-scale evaluation, robustness testing, and judge-style verification.

It was developed as part of **Convolve 4.0 GenAI Track** and is designed for real-world deployment in document automation and financial workflows.

---

## 🎯 Use Cases

- Automated invoice processing for enterprises  
- OCR-based financial document analysis  
- AI-powered document digitization  
- Robotic Process Automation (RPA) pipelines  
- Audit and compliance automation  

---

## 🧠 Key Capabilities

- Multilingual OCR pipeline  
- Layout-aware invoice parsing  
- Deterministic evaluation framework  
- Robust noise stress testing  
- Judge-style schema & latency validation  
- Streamlit live demo interface  
- CSV / JSON export for downstream systems  

---

## 🏗 System Architecture

PDF → PyMuPDF (Image Extraction)  
OCR → PaddleOCR  
Parsing → Layout rules + fuzzy matching  
Validation → Schema + latency + cost gate  
Output → Strict JSON + CSV  

---

## 🛠 Tech Stack

- Python  
- PyMuPDF  
- PaddleOCR  
- Streamlit  
- OpenCV  
- NumPy  
- Deterministic evaluation framework  

---



# doc-ai-invoice-extractor
Intelligent Document AI system for multilingual invoice field extraction (Convolve 4.0 GenAI Track)

This repository contains the hackathon-ready project under [doc_ai_project/README.md](doc_ai_project/README.md).

Quick start:

```bash
cd doc_ai_project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python executable.py --pdf /path/to/invoice.pdf
streamlit run app.py
```

Robustness stress testing (deterministic noisy variants + report): see [doc_ai_project/README.md](doc_ai_project/README.md) and run `python robustness_eval.py --input /path/to/invoice.pdf`.

Dataset conversion (images → PDFs for evaluation pipeline):

```bash
pip install pillow
python convert_images_to_pdf.py
```

Place invoice scans under `data/images/` (JPG/PNG). Outputs are written to `data/pdfs/` (one PDF per image).

## Run on Official Dataset

Final submission (recommended): run end-to-end finalize + gating in one command.

- With labels: `python doc_ai_project/submission.py --finalize --final-check --invoices data/pdfs --labels data/labels`
- Without labels (accuracy check skipped): `python doc_ai_project/submission.py --finalize --final-check --invoices data/pdfs`
- Gate only: `python final_submission_check.py --invoices data/pdfs --config outputs/final_config.json`

Gate output: `doc_ai_project/outputs/final_submission_report.json`.

This repo expects the evaluator dataset as PDFs under `data/pdfs/` (nested folders supported).

```bash
# 1) Convert images -> PDFs (one PDF per image)
pip install pillow
python convert_images_to_pdf.py

# 2) One-command preflight (dataset exists, PDFs readable, 3 sample runs)
python preflight.py

# 3) Evaluate
python doc_ai_project/eval.py --invoices data/pdfs
# (Optional) with labels if you have them:
# python doc_ai_project/eval.py --invoices data/pdfs --labels data/labels

# 4) Tune (requires labels)
python doc_ai_project/tuning.py --invoices data/pdfs --labels data/labels

# 5) Select best config (optionally validate it on the dataset)
python doc_ai_project/selector.py --tuning-report outputs/tuning_report.json --out-config best_config.json
# python doc_ai_project/selector.py --tuning-report outputs/tuning_report.json --out-config best_config.json --invoices data/pdfs

# 6) Build submission zip
python doc_ai_project/submission.py

# 7) Evaluator dry-run (schema + latency + cost + bbox/presence)
# Provide a representative PDF from the dataset
python doc_ai_project/dry_run.py --pdf ../data/pdfs/<some_invoice>.pdf

# 8) Batch smoke test (N random PDFs, writes outputs/smoke_report.json)
python smoke_test.py --invoices data/pdfs -n 10
```

## Final Round Demo Guide

### What judges will see

- A deterministic, CPU-only pipeline that extracts: dealer name, model name, horse power, asset cost, signature/stamp presence.
- A clean “demo table” (CLI + Streamlit) plus downloadable CSV/JSON.
- A “judge simulation” scorecard that enforces schema/latency/cost and optionally reports accuracy if labels are present.

### 1) Dataset prep (images → PDFs)

```bash
pip install pillow
python convert_images_to_pdf.py
```

### 2) Preflight (fast evaluator sanity)

```bash
python preflight.py --invoices data/pdfs
```

### 3) CLI demo (table + CSV)

```bash
python demo_runner.py --invoices data/pdfs
# Optional deterministic subset:
# python demo_runner.py --invoices data/pdfs --limit 25 --seed 1337
```

Outputs:

- `outputs/demo_outputs/demo_results.csv`
- `outputs/demo_outputs/demo_summary.json`

### 4) Judge simulation (scorecard)

```bash
python judge_mode.py --invoices data/pdfs --n 10 --seed 1337
# With labels (adds Accuracy + DLA):
# python judge_mode.py --invoices data/pdfs --labels data/labels --n 10
```

Output: `outputs/judge_report.json`

### 5) Streamlit live demo (ZIP of PDFs)

```bash
streamlit run doc_ai_project/app.py
```

Then open **Judge Demo** tab → upload a ZIP of PDFs → run → download CSV/JSON.

**Config selection in UI**

- Default: uses `doc_ai_project/best_config.json` when present.
- You can upload a config JSON in the **Judge Demo** tab to override it.
- The UI shows the active config filename and its SHA256 so judges can see what was run.

**PASS/FAIL banner**

- **PASS** means: strict schema OK + average latency ≤ 30s + average cost < $0.01.
- **FAIL** shows the rule(s) that broke (e.g. schema validation errors, avg latency too high, avg cost too high).

### One-command demo

```bash
./run_demo.sh
```

### 2-minute explanation (talk track)

1) PDF → images via PyMuPDF (fast; respects max pages + latency budget).
2) OCR via PaddleOCR; for digital PDFs, lightweight PyMuPDF text extraction can be used.
3) Layout structuring + rules/fuzzy match for dealer/model; robust numeric parsing for HP and asset cost.
4) Optional signature/stamp detection (YOLO if weights present), always returning strict JSON.
5) Determinism: sorted dataset discovery + seeded sampling; evaluator-style checks for schema/latency/cost.

## Final Round Execution Guide

This section is the “one page” guide for the final round. The defaults are **CPU-only**, **deterministic**, and **offline-safe**.

### Dataset setup

- Place invoice PDFs under `data/pdfs/` (recursive folders supported)
- Optional ground truth JSON under `data/labels/` (same stem as PDF)
- See `data/README.md` for the official dataset layout and workflow

Quick dataset validation (opens PDFs via PyMuPDF):

```bash
python dataset_preflight.py --invoices data/pdfs --labels data/labels
```

### Judge run (PASS/FAIL)

Runs preflight + judge simulation in **fast** mode and prints a PASS/FAIL summary:

```bash
bash judge_run.sh
```

### Live demo (Streamlit)

Launches Streamlit (includes the **Judge Demo** tab):

```bash
bash demo_live.sh
```

### Leaderboard preview

After running `doc_ai_project/eval.py`, preview key metrics:

```bash
python leaderboard_preview.py --report doc_ai_project/outputs/eval_report.json
```

### Submission packaging

Runs the evaluator-grade gate and builds `outputs/submission.zip`:

```bash
bash submission_pack.sh
```

Notes:
- PaddleOCR stays disabled unless explicitly enabled with `--enable-paddleocr`.
- The gate report is written to `doc_ai_project/outputs/final_submission_report.json`.
