# 📄 Document AI Invoice Extraction System

## 🚀 Features
## 🚀 Overview
## 🎯 Use Cases
## 🧠 Key Capabilities
## 🏗️ System Architecture
## 🛠 Tech Stack
## 📂 Project Structure
## ⚙ Installation
## ▶ How To Run
## 🔎 OCR Configuration
## 📤 Output Format (JSON Schema)
## 🧪 EDA Workflow for Jury Evaluation
## 📝 Notes


📄 Document AI Invoice Extraction System (PNG)

A fully offline, deterministic Document AI system that extracts structured invoice data from scanned PNG images  documents using OCR and rule-based intelligence.

Designed for jury evaluation, PNG-only execution, and JSON schema output.

🚀 Features

✔ Works fully offline (no APIs, no cloud)
✔ Supports PNG and PDF invoices
✔ OCR using PaddleOCR (optional)
✔ Rule-based extraction (no training required)
✔ Deterministic and reproducible
✔ JSON output in required schema
✔ Jury-safe execution
✔ Cost + latency estimation
✔ Explainability overlays (optional)

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


## 🏗️ System Architecture

```text
Input (PNG / PDF)
        |
        v
Image Loader / PDF Renderer
        |
        v
OCR Engine (PaddleOCR or fallback)
        |
        v
Layout Analyzer
        |
        v
Field Extractor (Regex + Heuristics)
        |
        v
Validator + Confidence Scorer
        |
        v
JSON Output
```



## 🛠 Tech Stack

- Python  
- PyMuPDF  
- PaddleOCR  
- Streamlit  
- OpenCV  
- NumPy  
- Deterministic evaluation framework  

---
## 📂 Project Structure

```text
doc-ai-invoice-extractor/
├── executable.py          # Main entry point
├── requirements.txt       # Dependencies
├── README.md              # Documentation
├── doc_ai_project/
│   ├── extractor.py       # Field extraction logic
│   ├── ocr.py             # OCR engine
│   ├── layout.py          # Layout segmentation
│   ├── validation.py      # Output validation
│   ├── explainability.py  # Optional overlays
│   ├── data/
│   │   ├── images/         # PNG invoices
│   │   └── pdfs/           # PDF invoices
│   └── outputs/
│       └── result.json    # Output file
```


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

```

Robustness stress testing (deterministic noisy variants + report): see [doc_ai_project/README.md](doc_ai_project/README.md) and run `python robustness_eval.py --input /path/to/invoice.pdf`.

Dataset conversion (images → PDFs for evaluation pipeline):

```bash
pip install pillow
python convert_images_to_pdf.py
```

Place invoice scans under `data/images/` (JPG/PNG). Outputs are written to `data/pdfs/` (one PDF per image).

## EDA Workflow for Jury Evaluation

Final jury evaluation uses **PNG invoice images only** and will run the **extraction code only**.
EDA must be generated **offline** and submitted as a separate report.

Input Format
PNG Input
data/images/invoice.png

PDF Input
data/pdfs/invoice.pdf

📤 Output Format (JSON Schema)
{
  "doc_id": "invoice_001",
  "fields": {
    "dealer_name": "ABC Tractors Pvt Ltd",
    "model_name": "Mahindra 575 DI",
    "horse_power": 50,
    "asset_cost": 525000,
    "signature": {
      "present": false,
      "bbox": []
    },
    "stamp": {
      "present": false,
      "bbox": []
    }
  },
  "confidence": 0.52,
  "review_required": true,
  "processing_time_sec": 6.29,
  "cost_estimate_usd": 0.00089
}

⚙ Installation
1️⃣ Create Virtual Environment (Python 3.10 recommended)
py -3.10 -m venv venv
venv\Scripts\activate

2️⃣ Install Dependencies
pip install -r requirements.txt

▶ How To Run
🔹 Run on PNG (Jury Mode)
python executable.py --png data/images/invoice.png --out outputs/result.json


🔎 Enable OCR 

By default OCR is safe-mode. To enable PaddleOCR:

Windows:
set DOC_AI_ENABLE_PADDLEOCR=1

Linux / Mac:
export DOC_AI_ENABLE_PADDLEOCR=1

📌 Sample Execution
python executable.py --png data/images/sample.png --out outputs/result.json




### 1) Validate dataset (PNG-only)

```bash
python dataset_preflight.py --invoices data/images --labels data/labels
```

This fails fast if any **non-PNG** invoice files exist under `data/images/`.

### 2) Generate offline EDA report (one command)

```bash
./run_eda.sh
```

Outputs (default):
- `outputs/eda/eda_report.json`
- `outputs/eda/eda_summary.csv`
- `outputs/eda/eda_report.pdf`
- `outputs/eda/eda_profile.json`

Submit `eda_report.pdf` (and optionally `eda_summary.csv`) as your offline EDA deliverable.

```
📸 Sample input PNG + output JSON

<img width="1919" height="979" alt="image" src="https://github.com/user-attachments/assets/15c9e819-0c24-49d9-b97f-68e0df2c2b4f" />
<img width="1919" height="969" alt="image" src="https://github.com/user-attachments/assets/d15d284d-98c3-4d10-b0b1-0ee1843329e3" />
<img width="1918" height="942" alt="image" src="https://github.com/user-attachments/assets/741c63df-559f-4451-b7fc-849235c43ce1" />



Notes:
- PaddleOCR stays disabled unless explicitly enabled with `--enable-paddleocr`.
- The gate report is written to `doc_ai_project/outputs/final_submission_report.json`.
