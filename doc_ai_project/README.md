# DocAI Invoice Extractor (Convolve 4.0 GenAI Track)

Production-grade, modular Intelligent Document AI pipeline for multilingual invoice PDFs (English/Hindi/Gujarati) including digital, scanned, handwritten, and photographed documents.

## Architecture (ASCII)

```
PDF
  ↓ (PyMuPDF)
Images
  ↓ (PaddleOCR; fallback: PyMuPDF text extraction for digital PDFs)
OCR Words + Boxes
  ↓ (Region detection: header/body/table/footer; LayoutParser primitives when available)
Structured Layout
  ↓ (Rules + RapidFuzz + optional LLM hook)
Field Candidates
  ↓ (YOLOv8 optional)
Signature/Stamp
  ↓ (Validation + Postprocess)
Strict JSON Output
```

## Architecture (PNG)

Generate a PNG diagram:

```bash
python architecture_diagram.py
```

Output: `outputs/architecture_diagram.png`

## Output JSON (strict)

The CLI outputs:

```json
{
  "doc_id": "invoice_001",
  "fields": {
    "dealer_name": "ABC Tractors Pvt Ltd",
    "model_name": "Mahindra 575 DI",
    "horse_power": 50,
    "asset_cost": 525000,
    "signature": {"present": true, "bbox": [100, 200, 300, 250]},
    "stamp": {"present": true, "bbox": [400, 500, 500, 550]}
  },
  "confidence": 0.96,
  "processing_time_sec": 3.8,
  "cost_estimate_usd": 0.002
}
```

## Fields extracted

- Dealer Name: fuzzy match ≥90% vs master list (RapidFuzz)
- Model Name: exact match against master list (fallback to label-based extraction)
- Horse Power: numeric extraction (regex; e.g. `50 HP` → `50`)
- Asset Cost: digits-only (regex + label preference)
- Dealer Signature / Stamp: presence + bbox via YOLOv8 (optional weights)

## Quickstart

### 1) Install

```bash
cd doc_ai_project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

CPU-only/headless notes:

- Uses `opencv-python-headless` to avoid `libGL` runtime dependency.
- If PaddleOCR is missing, the pipeline falls back to PyMuPDF text extraction (works for digital PDFs; scanned/image PDFs still require OCR).

### 2) Run CLI

```bash
python executable.py --pdf /path/to/invoice.pdf --out outputs/result.json
```

Run with a frozen config (recommended for reproducible evaluation/submission):

```bash
python executable.py --config best_config.json --pdf /path/to/invoice.pdf --out outputs/result.json
```

By default the CLI also generates judge-ready artifacts under `outputs/`:

- `outputs/overlays/` (explainability overlays)
- `outputs/eda_outputs/` + `outputs/eda_summary.csv`
- `outputs/error_report.json` + `outputs/error_distribution.png`
- `outputs/architecture_diagram.png`

Optional signature/stamp detection:

```bash
python executable.py --pdf /path/to/invoice.pdf --yolo-weights /path/to/signature_stamp.pt
```

### 3) Run Streamlit demo

```bash
streamlit run app.py
```

## EDA & Error Analysis (bonus)

Every run appends a row to `outputs/runs.jsonl`. Generate plots:

```bash
python -c "from pathlib import Path; from eda import run_eda; run_eda(Path('outputs/runs.jsonl'), Path('outputs'))"
python -c "from pathlib import Path; from error_analysis import run_error_analysis; run_error_analysis(Path('outputs/runs.jsonl'), Path('outputs'))"
```

Artifacts are saved into `outputs/eda_outputs/` and `outputs/error_report.json`.

## Accuracy benchmarking (DLA)

Ground-truth format: create a JSON per PDF with the same stem under a labels folder.

Example `labels/invoice_001.json`:

```json
{
  "dealer_name": "ABC Tractors Pvt Ltd",
  "model_name": "Mahindra 575 DI",
  "horse_power": 50,
  "asset_cost": 525000,
  "signature": {"present": false},
  "stamp": {"present": false}
}
```

Run evaluation:

```bash
python eval.py --invoices /path/to/pdfs --labels /path/to/labels
```

Outputs:

- `outputs/eval_report.json`
- `outputs/leaderboard_metrics.json`
- `outputs/scorecard.png`

## Final hackathon score (leaderboard)

We compute a single **final_score** for leaderboard ranking:

$$\text{final\_score} = w_{dla}\cdot DLA + w_{lat}\cdot \text{latency\_score} + w_{cost}\cdot \text{cost\_score}$$

Defaults (override via `PipelineConfig`):

- `w_dla = 0.7`
- `w_lat = 0.2`
- `w_cost = 0.1`

Where latency/cost are normalized into $[0,1]$ with targets (defaults: 30s, $0.01):

- `latency_score = 1 - min(latency/latency_target, 1)` (uses p95 by default)
- `cost_score = 1 - min(cost/cost_target, 1)`

The scorecard image (`outputs/scorecard.png`) includes DLA, latency, cost, and final_score.

## Leaderboard simulation (top-5)

After auto-tuning, simulate the top-5 leaderboard rows from tuning trials:

```bash
python leaderboard.py --tuning-report outputs/tuning_report.json --top-k 5
```

Outputs:

- `outputs/leaderboard_simulation.json`
- `outputs/leaderboard_table.png`

## Auto-tuning

```bash
python tuning.py --invoices /path/to/pdfs --labels /path/to/labels
```

Outputs:

- `outputs/tuning_report.json`
- `outputs/best_config.json`

## Config auto-selector (submission constraints)

Auto-select the best tuned config that satisfies:

- DLA ≥ 95%
- Latency ≤ 30s
- Cost ≤ $0.01

```bash
python selector.py --tuning-report outputs/tuning_report.json --out-config best_config.json
```

Outputs:

- `best_config.json` (frozen for submission)
- `outputs/best_config_selected.json`
- `outputs/selector_report.json`

## Failure replay

```bash
python replay.py --invoices /path/to/pdfs --labels /path/to/labels --eval-report outputs/eval_report.json
```

Output:

- `outputs/replay_report.json`

## Determinism + CPU-only

- Pipeline seeds Python/numpy/torch (when available) using `PipelineConfig.seed`.
- YOLO inference is forced to CPU device.

## Real-world robustness hardening

OCR preprocessing (CPU-only, deterministic; OpenCV-backed when available):

- Auto-rotation (0/90/180/270) for rotated camera photos
- Shadow/illumination normalization for shadowed scans + faded ink
- Optional adaptive thresholding for uneven lighting
- Optional perspective correction (best-effort) for skewed camera captures

These are controlled via `PipelineConfig` flags like `ocr_autorotate`, `ocr_shadow_remove`, `ocr_adaptive_threshold`, `ocr_perspective_correct`.

## Deterministic noisy-document regression harness

We include a deterministic stress-test harness to ensure extraction remains stable under common real-world degradation (rotation, blur, shadows, perspective warp, JPEG artifacts).

### 1) Generate noisy variants

Creates `outputs/noisy_tests/<stem>/` with `clean.pdf` and multiple noisy PDFs + preview images:

```bash
python noisy_test.py --input /path/to/invoice.pdf --profile mild --seed 1337 --dpi 200
```

### 2) Evaluate robustness (clean vs noisy)

Runs the full pipeline on clean + variants and writes a report + table:

```bash
python robustness_eval.py --input /path/to/invoice.pdf --profile mild --seed 1337 --dpi 200
```

Outputs:

- `outputs/robustness_report.json`
- `outputs/robustness_table.png`

### 3) Regression gate (pre-submission)

Fail if any field mismatches vs the clean output, and optionally fail if a noisy variant becomes `review_required` when the clean run was not:

```bash
python regression.py --report outputs/robustness_report.json --require-no-new-review
```

This writes `outputs/regression_report.json` and exits non-zero on failure.

## Fail-safe mode (manual review)

If `confidence < review_conf_threshold` the output includes:

- `review_required: true`

This makes worst-quality invoices “fail-safe” instead of silently trusting low-confidence extraction.

## Final sanity validation

Validate schema + numeric sanity + dealer/model whitelist:

```bash
python final_checks.py --json outputs/result.json
```

## Submission dry-run (evaluator simulation)

Runs the CLI, enforces runtime <30s, validates output schema, and checks required submission files:

```bash
python dry_run.py --pdf outputs/sample_invoice.pdf --config best_config.json
```

## Submission mode (single-command build)

Build a deterministic CPU-only submission ZIP that includes:

- `executable.py`
- `requirements.txt`
- `utils/`
- `best_config.json`
- `README.md`

Single command:

```bash
python submission.py
```

Outputs:

- `outputs/submission.zip`
- `outputs/submission_validation.json` (schema validation run)

## Leaderboard strategy (why this system wins)

- High DLA through region-aware extraction (header/table/footer priors) and robust OCR fallback for digital PDFs.
- Low latency by CPU-first execution, deadline guardrails, and skipping expensive stages when unnecessary.
- Low cost via cost-aware toggles (YOLO optional, multi-scale only when enabled) and minimal retries.
- Reproducible: deterministic seeding + frozen `best_config.json` for consistent leaderboard scoring.

## Logs (debugging)

- Structured stage logs: `outputs/pipeline_logs.jsonl`
- Run summaries for EDA/error analysis: `outputs/runs.jsonl`

## Explainability (bonus)

- Bounding box overlays are saved to `outputs/overlays/`.
- The overlay includes extracted field line boxes and optional signature/stamp detections.

## Scaling strategy (banking-ready)

- Horizontal scaling: page-level parallelism and batch OCR.
- Cost control: CPU-first OCR; only run YOLO when required.
- Reliability: deterministic noise-regression harness + fail-safe `review_required` for low-confidence outputs.
- Monitoring & audit: persist `runs.jsonl` + per-field confidence attribution; keep robustness reports (`outputs/robustness_report.json`) for model/pipeline change reviews.

## Notes

- Provide your own master lists in `data/dealers_master_list.txt` and `data/models_master_list.txt` for higher accuracy.
- YOLO labels expected: `signature` and `stamp`.
