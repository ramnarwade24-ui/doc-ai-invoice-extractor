from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from executable import run_pipeline
from utils.config import PipelineConfig
from utils.determinism import set_deterministic
from utils.text import normalize_name, normalize_spaces
from utils.leaderboard_scoring import compute_final_score
from utils.scorecard import render_scorecard_png


FIELDS = ["dealer_name", "model_name", "horse_power", "asset_cost", "signature", "stamp"]


def _repo_root() -> Path:
	# doc_ai_project/ -> repo root
	return Path(__file__).resolve().parent.parent


def _discover_pdfs(invoices: str | Path, *, repo_root: Path) -> List[Path]:
	"""Discover PDFs deterministically.

	- If invoices is a directory: scans recursively for *.pdf
	- If invoices is a file: accepts a single PDF
	- Otherwise: treats as a glob pattern relative to repo root (supports **)
	"""
	inv_str = str(invoices)
	inv_path = Path(inv_str)
	resolved = inv_path if inv_path.is_absolute() else (repo_root / inv_path)

	if resolved.exists() and resolved.is_file():
		return [resolved] if resolved.suffix.lower() == ".pdf" else []
	if resolved.exists() and resolved.is_dir():
		return sorted([p for p in resolved.rglob("*.pdf") if p.is_file()])

	# Glob pattern (relative to repo root)
	return sorted([p for p in repo_root.glob(inv_str) if p.is_file() and p.suffix.lower() == ".pdf"])


def _limit_pdfs(pdfs: List[Path], *, limit: Optional[int], seed: int) -> List[Path]:
	if limit is None:
		return pdfs
	k = int(limit)
	if k <= 0:
		return []
	if k >= len(pdfs):
		return pdfs
	# deterministic sample from a deterministic ordering
	rng = random.Random(int(seed))
	return sorted(rng.sample(pdfs, k=k))


def _percentile(values: List[float], p: float) -> float:
	if not values:
		return 0.0
	vals = sorted(values)
	p = max(0.0, min(1.0, float(p)))
	idx = int(round((len(vals) - 1) * p))
	return float(vals[idx])


def _load_label(path: Path) -> Optional[Dict[str, Any]]:
	if not path.exists():
		return None
	try:
		obj = json.loads(path.read_text(encoding="utf-8"))
	except Exception:
		return None
	# Accept either {fields:{...}} or just fields dict
	if isinstance(obj, dict) and "fields" in obj and isinstance(obj["fields"], dict):
		return obj["fields"]
	if isinstance(obj, dict):
		return obj
	return None


def _norm_str(s: Optional[str]) -> str:
	if s is None:
		return ""
	return normalize_name(normalize_spaces(str(s)))


def _boolish(v: Any) -> Optional[bool]:
	if v is None:
		return None
	if isinstance(v, bool):
		return v
	if isinstance(v, (int, float)):
		return bool(v)
	if isinstance(v, str):
		t = v.strip().lower()
		if t in {"true", "yes", "y", "1"}:
			return True
		if t in {"false", "no", "n", "0"}:
			return False
	return None


def _compare_field(field: str, pred_fields: Dict[str, Any], gt_fields: Dict[str, Any]) -> Tuple[Optional[bool], str]:
	"""Returns (correct?, note). None means skipped (no ground truth)."""
	if field not in gt_fields:
		return None, "no_gt"
	gt_val = gt_fields.get(field)
	pred_val = pred_fields.get(field)

	if field in {"dealer_name", "model_name"}:
		if gt_val in (None, ""):
			return None, "gt_empty"
		return _norm_str(pred_val) == _norm_str(gt_val), "norm_str_eq"

	if field in {"horse_power", "asset_cost"}:
		if gt_val in (None, ""):
			return None, "gt_empty"
		try:
			return int(pred_val) == int(gt_val), "int_eq"
		except Exception:
			return False, "int_parse_fail"

	if field in {"signature", "stamp"}:
		# Ground truth can be boolean or {present: bool}
		gt_present = gt_val
		if isinstance(gt_val, dict):
			gt_present = gt_val.get("present")
		pred_present = pred_val
		if isinstance(pred_val, dict):
			pred_present = pred_val.get("present")
		gb = _boolish(gt_present)
		pb = _boolish(pred_present)
		if gb is None:
			return None, "no_gt_present"
		if pb is None:
			return False, "no_pred_present"
		return pb == gb, "present_eq"

	return None, "unsupported"


def evaluate_dataset(
	*,
	invoices_dir: Path,
	pdf_paths: Optional[List[Path]] = None,
	labels_dir: Optional[Path],
	config: PipelineConfig,
	outputs_dir: Path,
) -> Dict[str, Any]:
	set_deterministic(seed=config.seed, deterministic=config.deterministic)
	outputs_dir.mkdir(parents=True, exist_ok=True)
	pred_dir = outputs_dir / "eval_predictions"
	pred_dir.mkdir(parents=True, exist_ok=True)

	# Support nested folders (evaluator datasets often have structure)
	if pdf_paths is not None:
		pdfs = sorted([p for p in pdf_paths if p.is_file() and p.suffix.lower() == ".pdf"])
	else:
		pdfs = sorted([p for p in invoices_dir.rglob("*.pdf") if p.is_file()])
	if not pdfs:
		raise FileNotFoundError(f"No PDFs found under {invoices_dir}")

	per_doc: List[Dict[str, Any]] = []
	latencies: List[float] = []
	costs: List[float] = []

	field_correct: Dict[str, int] = {f: 0 for f in FIELDS}
	field_total: Dict[str, int] = {f: 0 for f in FIELDS}
	docs_with_gt = 0
	docs_all_fields_correct = 0

	for pdf in pdfs:
		doc_id = pdf.stem
		result = run_pipeline(pdf_path=str(pdf), doc_id=doc_id, config=config)
		(pred_dir / f"{doc_id}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

		latencies.append(float(result.get("processing_time_sec") or 0.0))
		costs.append(float(result.get("cost_estimate_usd") or 0.0))

		gt_fields = None
		if labels_dir is not None:
			gt_fields = _load_label(labels_dir / f"{doc_id}.json")

		pred_fields = result.get("fields") or {}
		field_results: Dict[str, Any] = {}
		has_gt = gt_fields is not None
		if has_gt:
			docs_with_gt += 1
			all_ok = True
			for f in FIELDS:
				ok, note = _compare_field(f, pred_fields, gt_fields)
				field_results[f] = {"correct": ok, "note": note}
				if ok is None:
					continue
				field_total[f] += 1
				if ok:
					field_correct[f] += 1
				else:
					all_ok = False
			if all_ok:
				docs_all_fields_correct += 1
		else:
			for f in FIELDS:
				field_results[f] = {"correct": None, "note": "no_labels"}

		per_doc.append(
			{
				"doc_id": doc_id,
				"pdf": str(pdf),
				"has_ground_truth": bool(has_gt),
				"confidence": result.get("confidence"),
				"latency_sec": result.get("processing_time_sec"),
				"cost_usd": result.get("cost_estimate_usd"),
				"field_results": field_results,
			}
		)

	field_accuracy: Dict[str, float] = {}
	field_error_rate: Dict[str, float] = {}
	for f in FIELDS:
		tot = max(1, field_total[f])
		acc = float(field_correct[f] / tot) if field_total[f] > 0 else 0.0
		field_accuracy[f] = acc
		field_error_rate[f] = float(1.0 - acc) if field_total[f] > 0 else 1.0

	dla = float(docs_all_fields_correct / max(1, docs_with_gt)) if docs_with_gt > 0 else 0.0

	summary = {
		"docs_total": int(len(pdfs)),
		"docs_with_ground_truth": int(docs_with_gt),
		"dla": float(dla),
		"field_accuracy": field_accuracy,
		"field_error_rate": field_error_rate,
		"latency_avg_sec": float(sum(latencies) / max(1, len(latencies))),
		"latency_p95_sec": _percentile(latencies, 0.95),
		"cost_avg_usd": float(sum(costs) / max(1, len(costs))),
	}

	final_score, components = compute_final_score(summary, config)
	summary["final_score"] = float(final_score)
	summary["final_score_components"] = components
	summary["final_score_targets"] = {
		"latency_target_sec": float(config.leaderboard_latency_target_sec),
		"cost_target_usd": float(config.leaderboard_cost_target_usd),
		"use_latency_p95": bool(config.leaderboard_use_latency_p95),
	}

	report = {
		"config": {
			"seed": config.seed,
			"deterministic": config.deterministic,
			"dpi": config.dpi,
			"max_pages": config.max_pages,
			"dealer_fuzzy_threshold": config.dealer_fuzzy_threshold,
			"region_weight_overrides": config.region_weight_overrides,
			"yolo_conf": config.yolo_conf,
			"yolo_iou": config.yolo_iou,
			"yolo_img_sizes": list(config.yolo_img_sizes),
			"ocr_preprocess_variants": [list(v) for v in config.ocr_preprocess_variants],
			"leaderboard_weight_dla": config.leaderboard_weight_dla,
			"leaderboard_weight_latency": config.leaderboard_weight_latency,
			"leaderboard_weight_cost": config.leaderboard_weight_cost,
			"leaderboard_latency_target_sec": config.leaderboard_latency_target_sec,
			"leaderboard_cost_target_usd": config.leaderboard_cost_target_usd,
			"leaderboard_use_latency_p95": config.leaderboard_use_latency_p95,
		},
		"summary": summary,
		"per_doc": per_doc,
	}
	return report


def _write_scorecard(outputs_dir: Path, summary: Dict[str, Any]) -> None:
	render_scorecard_png(summary=summary, out_path=outputs_dir / "scorecard.png")


def main() -> int:
	p = argparse.ArgumentParser(description="Evaluate DocAI invoice extractor")
	p.add_argument(
		"--invoices",
		default="data/pdfs",
		help=(
			"Dataset path for invoice PDFs. Accepts a directory (scans recursively), a single PDF, "
			"or a glob pattern like data/pdfs/**/*.pdf. Relative paths are resolved from repo root."
		),
	)
	p.add_argument(
		"--labels",
		default="",
		help=(
			"Folder containing ground-truth JSON (same stem as PDF). If omitted, accuracy metrics are 0. "
			"Relative paths are resolved from repo root."
		),
	)
	p.add_argument("--out", default="outputs/eval_report.json", help="Output eval report JSON")
	p.add_argument("--dpi", type=int, default=200)
	p.add_argument("--max-pages", type=int, default=5)
	p.add_argument("--dealer-threshold", type=int, default=90)
	p.add_argument(
		"--limit",
		type=int,
		default=None,
		help="Optional deterministic sample size (useful for quick CPU-only eval runs).",
	)
	p.add_argument(
		"--run-mode",
		default="normal",
		choices=("normal", "replay", "tuning", "submission", "judge", "demo"),
		help="Pipeline run mode (affects logging + some gating).",
	)
	p.add_argument("--seed", type=int, default=1337)
	args = p.parse_args()

	base_dir = Path(__file__).resolve().parent
	repo_root = _repo_root()
	outputs_dir = base_dir / "outputs"
	out_path = (base_dir / args.out) if not Path(args.out).is_absolute() else Path(args.out)
	out_path.parent.mkdir(parents=True, exist_ok=True)

	labels_dir = Path(args.labels) if args.labels else None
	if labels_dir is not None and not labels_dir.is_absolute():
		labels_dir = repo_root / labels_dir

	cfg = PipelineConfig(
		seed=args.seed,
		deterministic=True,
		run_mode=str(args.run_mode),
		dpi=args.dpi,
		max_pages=args.max_pages,
		dealer_fuzzy_threshold=args.dealer_threshold,
	)

	invoices = _discover_pdfs(args.invoices, repo_root=repo_root)
	invoices = _limit_pdfs(invoices, limit=args.limit, seed=int(args.seed))
	if not invoices:
		raise FileNotFoundError(
			"No PDFs found. Provide --invoices as a directory, a single PDF, or a glob pattern. "
			"Example: --invoices data/pdfs"
		)
	report = evaluate_dataset(
		invoices_dir=repo_root / "data" / "pdfs",
		pdf_paths=invoices,
		labels_dir=labels_dir,
		config=cfg,
		outputs_dir=outputs_dir,
	)

	out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

	# Leaderboard-friendly metrics
	leaderboard = {
		"dla": report["summary"]["dla"],
		"latency_avg_sec": report["summary"]["latency_avg_sec"],
		"latency_p95_sec": report["summary"]["latency_p95_sec"],
		"cost_avg_usd": report["summary"]["cost_avg_usd"],
		"final_score": report["summary"].get("final_score", 0.0),
		"final_score_components": report["summary"].get("final_score_components", {}),
		"field_error_rate": report["summary"]["field_error_rate"],
		"docs_with_ground_truth": report["summary"]["docs_with_ground_truth"],
	}
	(outputs_dir / "leaderboard_metrics.json").write_text(
		json.dumps(leaderboard, ensure_ascii=False, indent=2),
		encoding="utf-8",
	)

	_write_scorecard(outputs_dir, report["summary"])
	print(json.dumps(leaderboard, ensure_ascii=False, indent=2))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
