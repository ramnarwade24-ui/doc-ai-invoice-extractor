#!/usr/bin/env python3
"""Real-data error mining for the competition dataset.

Runs eval on a PDFs dataset and clusters failure cases into broad buckets:
- OCR errors
- Layout errors
- Model mismatches
- Dealer fuzzy mismatches
- Price parsing errors

Outputs:
- doc_ai_project/outputs/error_mining_report.json
- doc_ai_project/outputs/error_heatmap.png

Notes:
- If labels are not provided, clustering is based on pipeline/log signals and
  "suspicious" outputs (missing fields / low confidence).
- Deterministic: sorted file discovery + seeded operations.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _repo_root() -> Path:
	return Path(__file__).resolve().parent


def _doc_ai_dir(repo_root: Path) -> Path:
	return repo_root / "doc_ai_project"


def _load_json(path: Path) -> Dict[str, Any]:
	obj = json.loads(path.read_text(encoding="utf-8"))
	if not isinstance(obj, dict):
		raise ValueError(f"Expected JSON object at {path}")
	return obj


def _normalize_name(text: str) -> str:
	repo_root = _repo_root()
	import sys
	import importlib

	sys.path.insert(0, str(_doc_ai_dir(repo_root)))
	utils_text = importlib.import_module("utils.text")
	return utils_text.normalize_name(text or "")


def _fuzzy_ratio(a: str, b: str) -> float:
	from rapidfuzz import fuzz

	return float(fuzz.token_set_ratio(_normalize_name(a), _normalize_name(b)))


def _parse_pipeline_logs(log_path: Path) -> Dict[str, Dict[str, Any]]:
	"""Aggregate a few stage metrics keyed by doc_id."""
	by_doc: Dict[str, Dict[str, Any]] = {}
	if not log_path.exists():
		return by_doc

	for line in log_path.read_text(encoding="utf-8").splitlines():
		line = line.strip()
		if not line:
			continue
		try:
			obj = json.loads(line)
		except Exception:
			continue
		doc_id = obj.get("doc_id")
		if not doc_id:
			continue
		event = str(obj.get("event") or "")
		meta = by_doc.setdefault(str(doc_id), {})

		if event == "stage_ocr":
			meta["ocr_avg_word_conf"] = obj.get("avg_word_conf")
			meta["ocr_word_counts"] = obj.get("word_counts")
			meta["ocr_failures"] = obj.get("failures")
			meta["ocr_mode"] = obj.get("mode")
		elif event == "stage_layout":
			meta["layout_region_line_counts"] = obj.get("region_line_counts")
			meta["layout_language"] = obj.get("language")
		elif event == "stage_validation":
			meta["latency_ok"] = obj.get("latency_ok")
			meta["cost_ok"] = obj.get("cost_ok")
			meta["processing_time_sec"] = obj.get("processing_time_sec")
			meta["cost_estimate_usd"] = obj.get("cost_estimate_usd")

	return by_doc


@dataclass(frozen=True)
class Case:
	doc_id: str
	pdf: str
	confidence: float
	field_results: Dict[str, Any]
	pred_fields: Dict[str, Any]
	gt_fields: Optional[Dict[str, Any]]
	log_meta: Dict[str, Any]


def _load_pred_fields(doc_ai_outputs: Path, doc_id: str) -> Dict[str, Any]:
	pred_path = doc_ai_outputs / "eval_predictions" / f"{doc_id}.json"
	if not pred_path.exists():
		return {}
	try:
		obj = json.loads(pred_path.read_text(encoding="utf-8"))
		return obj.get("fields") or {}
	except Exception:
		return {}


def _load_gt_fields(labels_dir: Path, doc_id: str) -> Optional[Dict[str, Any]]:
	p = labels_dir / f"{doc_id}.json"
	if not p.exists():
		return None
	try:
		obj = json.loads(p.read_text(encoding="utf-8"))
		if isinstance(obj, dict) and "fields" in obj and isinstance(obj["fields"], dict):
			return obj["fields"]
		if isinstance(obj, dict):
			return obj
		return None
	except Exception:
		return None


def _is_failure(case: Case) -> bool:
	# With labels: failure = any incorrect field (skipping None)
	if case.gt_fields is not None:
		for meta in (case.field_results or {}).values():
			if isinstance(meta, dict) and meta.get("correct") is False:
				return True
		return False
	# Without labels: treat as suspicious if missing key fields or very low confidence
	fields = case.pred_fields or {}
	missing_core = sum(1 for k in ("dealer_name", "model_name", "horse_power", "asset_cost") if not fields.get(k))
	return bool(missing_core >= 2 or float(case.confidence or 0.0) < 0.55)


def _cluster_case(case: Case, *, dealer_thresh: float = 85.0, model_thresh: float = 85.0) -> str:
	meta = case.log_meta or {}
	pred = case.pred_fields or {}
	gt = case.gt_fields

	# 1) Price parsing errors
	if gt is not None:
		fr = case.field_results or {}
		asset_meta = fr.get("asset_cost") if isinstance(fr, dict) else None
		if isinstance(asset_meta, dict) and asset_meta.get("note") in {"int_parse_fail"}:
			return "price_parsing_errors"
		# common: predicted None/0 when GT exists
		if gt.get("asset_cost") not in (None, ""):
			if pred.get("asset_cost") in (None, 0, ""):
				return "price_parsing_errors"

	# 2) Dealer fuzzy mismatches
	if gt is not None and gt.get("dealer_name") not in (None, ""):
		pd = pred.get("dealer_name")
		if pd not in (None, ""):
			if _fuzzy_ratio(str(pd), str(gt.get("dealer_name"))) >= dealer_thresh:
				# If eval says wrong, call it a fuzzy mismatch bucket
				fr = case.field_results or {}
				dm = fr.get("dealer_name") if isinstance(fr, dict) else None
				if isinstance(dm, dict) and dm.get("correct") is False:
					return "dealer_fuzzy_mismatches"

	# 3) Model mismatches
	if gt is not None and gt.get("model_name") not in (None, ""):
		pm = pred.get("model_name")
		if pm not in (None, ""):
			if _fuzzy_ratio(str(pm), str(gt.get("model_name"))) >= model_thresh:
				mm = (case.field_results or {}).get("model_name") if isinstance(case.field_results, dict) else None
				if isinstance(mm, dict) and mm.get("correct") is False:
					return "model_mismatches"

	# 4) OCR errors (log-signal driven)
	ocr_fail = int(meta.get("ocr_failures") or 0)
	avg_conf = float(meta.get("ocr_avg_word_conf") or 0.0)
	word_counts = meta.get("ocr_word_counts") or []
	try:
		word_total = int(sum(int(x) for x in word_counts)) if isinstance(word_counts, list) else 0
	except Exception:
		word_total = 0
	if ocr_fail > 0 or avg_conf < 0.45 or word_total < 20:
		return "ocr_errors"

	# 5) Layout errors (log-signal driven)
	reg = meta.get("layout_region_line_counts") or {}
	if isinstance(reg, dict):
		try:
			table_lines = int(reg.get("table") or 0)
			all_lines = int(sum(int(v) for v in reg.values()))
		except Exception:
			table_lines, all_lines = 0, 0
		# very sparse structure tends to break heuristics
		if all_lines > 0 and all_lines < 25:
			return "layout_errors"
		if all_lines > 0 and table_lines == 0 and (pred.get("horse_power") is None or pred.get("asset_cost") is None):
			return "layout_errors"

	return "other"


def _render_heatmap(out_png: Path, counts: Dict[str, Dict[str, int]]) -> None:
	try:
		import matplotlib

		matplotlib.use("Agg")
		import matplotlib.pyplot as plt
		import numpy as np
	except Exception as e:
		raise RuntimeError(f"Heatmap rendering unavailable (matplotlib missing?): {e}")

	cats = ["ocr_errors", "layout_errors", "model_mismatches", "dealer_fuzzy_mismatches", "price_parsing_errors", "other"]
	fields = ["dealer_name", "model_name", "horse_power", "asset_cost", "signature", "stamp"]

	mat = np.zeros((len(cats), len(fields)), dtype=int)
	for i, c in enumerate(cats):
		for j, f in enumerate(fields):
			mat[i, j] = int((counts.get(c) or {}).get(f) or 0)

	fig = plt.figure(figsize=(10, 4.8))
	ax = fig.add_subplot(111)
	im = ax.imshow(mat, aspect="auto", cmap="magma")
	ax.set_xticks(range(len(fields)), labels=fields, rotation=25, ha="right")
	ax.set_yticks(range(len(cats)), labels=cats)
	ax.set_title("Error Heatmap (cluster × field)")
	for i in range(mat.shape[0]):
		for j in range(mat.shape[1]):
			ax.text(j, i, str(mat[i, j]), ha="center", va="center", color="white", fontsize=9)
	fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
	fig.tight_layout()
	out_png.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_png, dpi=160)
	plt.close(fig)


def _run_eval(repo_root: Path, invoices: str, labels: str, out_rel: str) -> Path:
	import subprocess

	doc_ai_dir = _doc_ai_dir(repo_root)
	cmd = ["python", "eval.py", "--invoices", invoices, "--out", out_rel]
	if labels:
		cmd += ["--labels", labels]
	subprocess.run(cmd, cwd=str(doc_ai_dir), check=True)
	return doc_ai_dir / out_rel


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Mine errors on a PDFs dataset and cluster failures")
	p.add_argument("--invoices", default="data/pdfs", help="Dataset PDFs path (repo-root relative)")
	p.add_argument("--labels", default="", help="Optional labels path (repo-root relative)")
	p.add_argument("--eval-out", default="outputs/eval_report.json", help="Eval report path (doc_ai_project relative)")
	p.add_argument("--out", default="outputs/error_mining_report.json", help="Output report path (doc_ai_project relative)")
	p.add_argument("--heatmap", default="outputs/error_heatmap.png", help="Heatmap PNG path (doc_ai_project relative)")
	return p.parse_args()


def main() -> int:
	args = parse_args()
	repo_root = _repo_root()
	doc_ai_dir = _doc_ai_dir(repo_root)
	outputs_dir = doc_ai_dir / "outputs"
	outputs_dir.mkdir(parents=True, exist_ok=True)

	eval_report_path = _run_eval(repo_root, invoices=str(args.invoices), labels=str(args.labels), out_rel=str(args.eval_out))
	rep = _load_json(eval_report_path)
	per_doc = rep.get("per_doc") or []
	labels_dir = (repo_root / args.labels) if args.labels else None

	logs = _parse_pipeline_logs(outputs_dir / "pipeline_logs.jsonl")

	cases: List[Case] = []
	for d in per_doc:
		if not isinstance(d, dict):
			continue
		doc_id = str(d.get("doc_id") or "")
		if not doc_id:
			continue
		pred_fields = _load_pred_fields(outputs_dir, doc_id)
		gt_fields = _load_gt_fields(labels_dir, doc_id) if labels_dir is not None else None
		cases.append(
			Case(
				doc_id=doc_id,
				pdf=str(d.get("pdf") or ""),
				confidence=float(d.get("confidence") or 0.0),
				field_results=d.get("field_results") or {},
				pred_fields=pred_fields,
				gt_fields=gt_fields,
				log_meta=logs.get(doc_id) or {},
			)
		)

	clusters: Dict[str, List[str]] = defaultdict(list)
	error_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
	failures = 0

	for c in cases:
		if not _is_failure(c):
			continue
		failures += 1
		bucket = _cluster_case(c)
		clusters[bucket].append(c.doc_id)

		# field-level heatmap counts
		fr = c.field_results or {}
		if c.gt_fields is not None and isinstance(fr, dict):
			for field, meta in fr.items():
				if isinstance(meta, dict) and meta.get("correct") is False:
					error_counts[bucket][str(field)] += 1
		else:
			# Without labels, treat missing core fields as errors
			for field in ("dealer_name", "model_name", "horse_power", "asset_cost"):
				if not (c.pred_fields or {}).get(field):
					error_counts[bucket][field] += 1

	for k in list(clusters.keys()):
		clusters[k] = sorted(set(clusters[k]))

	out_path = doc_ai_dir / args.out
	heatmap_path = doc_ai_dir / args.heatmap

	report = {
		"dataset": {
			"invoices": str(args.invoices),
			"labels": str(args.labels),
			"docs_total": int(len(cases)),
			"failures_detected": int(failures),
		},
		"clusters": {k: {"count": len(v), "doc_ids": v[:200]} for k, v in sorted(clusters.items())},
		"cluster_counts": {k: len(v) for k, v in sorted(clusters.items())},
		"heatmap_counts": {k: dict(v) for k, v in error_counts.items()},
		"notes": [
			"With labels: failures are incorrect fields per eval field_results.",
			"Without labels: failures are heuristic (missing fields / low confidence) and meant for triage.",
		],
	}

	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
	heatmap_ok = True
	heatmap_error = ""
	try:
		_render_heatmap(heatmap_path, counts=error_counts)
	except Exception as e:
		heatmap_ok = False
		heatmap_error = str(e)
		(heatmap_path.parent / "error_heatmap_unavailable.txt").write_text(
			"Heatmap was not generated.\n"
			f"Reason: {heatmap_error}\n\n"
			"To enable heatmap rendering, install matplotlib in your environment.\n",
			encoding="utf-8",
		)
		# Update report with a note (best-effort)
		try:
			report.setdefault("notes", []).append(f"Heatmap not generated: {heatmap_error}")
			out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
		except Exception:
			pass

	print(json.dumps({"report": str(out_path), "heatmap": str(heatmap_path), "heatmap_ok": heatmap_ok}, indent=2))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
