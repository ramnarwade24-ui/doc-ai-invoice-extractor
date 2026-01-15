#!/usr/bin/env python3
"""Confidence auto-calibration on a labeled dataset.

Fits a review boundary (review_required) based on per-document confidence to:
- maximize catching incorrect docs (review when wrong)
- minimize false review flags (review when correct)

Outputs:
- doc_ai_project/outputs/confidence_calibration.json

Notes:
- Requires eval report generated with labels so per-field correctness exists.
- Deterministic grid search over thresholds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _repo_root() -> Path:
	return Path(__file__).resolve().parent


def _doc_ai_dir(repo_root: Path) -> Path:
	return repo_root / "doc_ai_project"


def _load_json(path: Path) -> Dict[str, Any]:
	obj = json.loads(path.read_text(encoding="utf-8"))
	if not isinstance(obj, dict):
		raise ValueError(f"Expected JSON object at {path}")
	return obj


def _doc_correct(field_results: Dict[str, Any]) -> bool:
	# A doc is correct if every field with GT is correct.
	for meta in (field_results or {}).values():
		if isinstance(meta, dict) and meta.get("correct") is False:
			return False
	return True


def _metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
	prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
	rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
	f1 = float((2 * prec * rec) / (prec + rec)) if (prec + rec) > 0 else 0.0
	fp_rate = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
	fn_rate = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
	return {"precision": prec, "recall": rec, "f1": f1, "false_review_rate": fp_rate, "missed_bad_rate": fn_rate}


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Calibrate review_required confidence threshold")
	p.add_argument(
		"--eval-report",
		default="doc_ai_project/outputs/eval_report.json",
		help="Eval report JSON (repo-root relative)",
	)
	p.add_argument(
		"--out",
		default="doc_ai_project/outputs/confidence_calibration.json",
		help="Output calibration JSON (repo-root relative)",
	)
	p.add_argument("--min-threshold", type=float, default=0.40)
	p.add_argument("--max-threshold", type=float, default=0.95)
	p.add_argument("--step", type=float, default=0.01)
	return p.parse_args()


def main() -> int:
	args = parse_args()
	repo_root = _repo_root()
	eval_path = repo_root / args.eval_report
	out_path = repo_root / args.out

	rep = _load_json(eval_path)
	per_doc = rep.get("per_doc") or []

	rows: List[Tuple[float, bool]] = []  # (confidence, doc_correct)
	with_gt = 0
	for d in per_doc:
		if not isinstance(d, dict):
			continue
		if not bool(d.get("has_ground_truth")):
			continue
		fr = d.get("field_results") or {}
		conf = float(d.get("confidence") or 0.0)
		rows.append((conf, _doc_correct(fr)))
		with_gt += 1

	if not rows:
		out_path.parent.mkdir(parents=True, exist_ok=True)
		out_path.write_text(
			json.dumps(
				{
					"ok": False,
					"reason": "no_labeled_docs_in_eval_report",
					"hint": "Run eval with labels: python doc_ai_project/eval.py --invoices data/pdfs --labels data/labels",
				},
				indent=2,
			),
			encoding="utf-8",
		)
		print("[WARN] No labeled docs found in eval report; cannot calibrate.")
		return 2

	min_t = float(args.min_threshold)
	max_t = float(args.max_threshold)
	step = float(args.step)

	best = None
	best_key = (-1.0, -1.0, 1.0)  # (f1, recall, false_review_rate)
	cur = min_t
	trials: List[Dict[str, Any]] = []

	while cur <= max_t + 1e-9:
		tp = fp = fn = tn = 0
		for conf, correct in rows:
			review = bool(conf < cur)
			bad = not bool(correct)
			if review and bad:
				tp += 1
			elif review and (not bad):
				fp += 1
			elif (not review) and bad:
				fn += 1
			else:
				tn += 1
		m = _metrics(tp, fp, fn, tn)
		trial = {"threshold": round(cur, 4), "tp": tp, "fp": fp, "fn": fn, "tn": tn, **m}
		trials.append(trial)

		key = (float(m["f1"]), float(m["recall"]), -float(m["false_review_rate"]))
		if key > best_key:
			best_key = key
			best = trial

		cur += step

	report = {
		"ok": True,
		"docs_with_ground_truth": int(with_gt),
		"objective": "maximize F1 for flagging incorrect docs; tie-break by higher recall then lower false_review_rate",
		"recommended_review_conf_threshold": float(best["threshold"]) if best else 0.75,
		"best": best,
		"trials": trials,
	}

	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
	print(json.dumps({"out": str(out_path), "recommended": report["recommended_review_conf_threshold"]}, indent=2))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
