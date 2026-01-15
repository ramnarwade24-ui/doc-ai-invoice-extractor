#!/usr/bin/env python3
"""Judge mode CLI.

Simulates evaluator behavior on a deterministic sample of PDFs:
- Strict schema validation (Pydantic)
- Latency <= 30s
- Cost < $0.01
- Optional accuracy + DLA (document-level accuracy) if labels provided

Examples:
  python judge_mode.py --invoices data/pdfs --n 10 --seed 1337
  python judge_mode.py --invoices data/pdfs --labels data/labels --n 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _repo_root() -> Path:
	return Path(__file__).resolve().parent


def _ensure_docai_on_path() -> None:
	repo_root = _repo_root()
	doc_ai_dir = repo_root / "doc_ai_project"
	if str(doc_ai_dir) not in sys.path:
		sys.path.insert(0, str(doc_ai_dir))


def _try_rich_print_scorecard(lines: List[str], ok: bool) -> bool:
	try:
		from rich.console import Console
		from rich.panel import Panel

		console = Console()
		title = "PASS" if ok else "FAIL"
		style = "bold green" if ok else "bold red"
		console.print(Panel.fit("\n".join(lines), title=title, border_style=style))
		return True
	except Exception:
		return False


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(
		description="Simulate evaluator checks (schema/latency/cost) on a deterministic sample",
		epilog=(
			"Examples:\n"
			"  python judge_mode.py --invoices data/pdfs --n 10 --seed 1337\n"
			"  python judge_mode.py --invoices data/pdfs --labels data/labels --n 10\n"
		),
		formatter_class=argparse.RawTextHelpFormatter,
	)
	p.add_argument("--invoices", default="data/pdfs", help="Folder, PDF, or glob (repo-root relative)")
	p.add_argument("--labels", default="", help="Optional labels folder (repo-root relative)")
	p.add_argument("--n", type=int, default=10, help="Number of PDFs to test")
	p.add_argument("--seed", type=int, default=1337)
	p.add_argument("--config", default="", help="Optional config JSON (e.g. best_config.json)")
	p.add_argument(
		"--mode",
		choices=("fast", "accurate"),
		default="fast",
		help="Run mode preset: fast (default) keeps runs under 30s; accurate uses config defaults.",
	)
	p.add_argument(
		"--enable-paddleocr",
		action="store_true",
		help="Enable PaddleOCR (slower, higher accuracy)",
	)
	p.add_argument("--out", default="outputs/judge_report.json", help="Output report JSON (repo-root relative)")
	return p.parse_args()


def _as_bool_present(val: Any) -> Optional[bool]:
	if val is None:
		return None
	if isinstance(val, dict):
		if "present" in val:
			return bool(val.get("present"))
	return bool(val)


def _compare_pred_gt(pred_fields: Dict[str, Any], gt_fields: Dict[str, Any]) -> Tuple[int, int, bool]:
	"""Return (correct_count, total_count, doc_all_correct)."""
	from utils.demo_utils import normalized_equal

	correct = 0
	total = 0

	def check(cond: bool) -> None:
		nonlocal correct, total
		total += 1
		if cond:
			correct += 1

	# Names
	if gt_fields.get("dealer_name") not in (None, ""):
		check(normalized_equal(pred_fields.get("dealer_name"), gt_fields.get("dealer_name")))
	if gt_fields.get("model_name") not in (None, ""):
		check(normalized_equal(pred_fields.get("model_name"), gt_fields.get("model_name")))

	# Ints
	if gt_fields.get("horse_power") not in (None, ""):
		try:
			check(int(pred_fields.get("horse_power")) == int(gt_fields.get("horse_power")))
		except Exception:
			check(False)
	if gt_fields.get("asset_cost") not in (None, ""):
		try:
			check(int(pred_fields.get("asset_cost")) == int(gt_fields.get("asset_cost")))
		except Exception:
			check(False)

	# Presence
	for key in ("signature", "stamp"):
		if key in gt_fields and gt_fields.get(key) is not None:
			g = _as_bool_present(gt_fields.get(key))
			p = _as_bool_present(pred_fields.get(key))
			if g is None:
				continue
			check(bool(p) == bool(g))

	all_correct = bool(total > 0 and correct == total)
	return correct, total, all_correct


def main() -> int:
	args = parse_args()
	# Must be set before importing doc_ai_project modules so worker paths can enable OCR.
	if bool(getattr(args, "enable_paddleocr", False)):
		import os

		os.environ["DOC_AI_ENABLE_PADDLEOCR"] = "1"
	_ensure_docai_on_path()

	from utils.demo_utils import discover_pdfs, load_label_fields, run_pipeline, write_json

	repo_root = _repo_root()
	out_path = (repo_root / args.out) if not Path(args.out).is_absolute() else Path(args.out)

	pdfs = discover_pdfs(str(args.invoices), recursive=True, limit=int(args.n), seed=int(args.seed))
	if not pdfs:
		print("[FAIL] No PDFs found.")
		return 2

	results, errors = run_pipeline(pdfs, config_path=(str(args.config) if args.config else None), mode=str(args.mode))

	# Enforce latency/cost
	per_doc: List[Dict[str, Any]] = []
	passed = 0
	lat_sum = 0.0
	cost_sum = 0.0

	acc_correct = 0
	acc_total = 0
	dla_correct = 0
	dla_total = 0

	labels_dir: Optional[Path] = None
	if args.labels:
		labels_dir = Path(args.labels)
		if not labels_dir.is_absolute():
			labels_dir = repo_root / labels_dir
		if not labels_dir.exists():
			labels_dir = None

	# Index results by doc_id for accuracy lookup
	by_id = {r.get("doc_id"): r for r in results if isinstance(r, dict)}
	for pdf in pdfs:
		doc_id = pdf.stem
		err = next((e for e in errors if e.get("doc_id") == doc_id), None)
		row = by_id.get(doc_id)

		case: Dict[str, Any] = {"doc_id": doc_id, "pdf": str(pdf)}
		if err is not None:
			case.update({"ok": False, "error": err.get("error")})
			per_doc.append(case)
			continue

		lat = float(row.get("processing_time_sec") or 0.0)
		cost = float(row.get("cost_estimate_usd") or 0.0)
		lat_ok = bool(lat <= 30.0)
		cost_ok = bool(cost < 0.01)
		ok = bool(lat_ok and cost_ok)

		case.update(
			{
				"ok": ok,
				"latency_sec": lat,
				"cost_usd": cost,
				"latency_ok": lat_ok,
				"cost_ok": cost_ok,
				"confidence": row.get("confidence"),
			}
		)

		# Accuracy / DLA
		if labels_dir is not None:
			gt = load_label_fields(labels_dir, doc_id)
			if gt is not None:
				pred_fields = {
					"dealer_name": row.get("dealer_name"),
					"model_name": row.get("model_name"),
					"horse_power": row.get("horse_power"),
					"asset_cost": row.get("asset_cost"),
					"signature": {"present": bool(row.get("signature_present"))},
					"stamp": {"present": bool(row.get("stamp_present"))},
				}
				c, t, all_ok = _compare_pred_gt(pred_fields, gt)
				acc_correct += c
				acc_total += t
				dla_total += 1
				if all_ok:
					dla_correct += 1
				case["accuracy_fields_correct"] = int(c)
				case["accuracy_fields_total"] = int(t)
				case["dla_correct"] = bool(all_ok)

		per_doc.append(case)
		passed += 1 if ok else 0
		lat_sum += lat
		cost_sum += cost

	docs_tested = int(len(pdfs))
	pass_rate = float(passed / max(1, docs_tested))
	avg_lat = float(lat_sum / max(1, docs_tested))
	avg_cost = float(cost_sum / max(1, docs_tested))
	accuracy = float(acc_correct / acc_total) if acc_total > 0 else None
	dla = float(dla_correct / dla_total) if dla_total > 0 else None

	overall_ok = bool(len(errors) == 0 and passed == docs_tested)

	report: Dict[str, Any] = {
		"ok": overall_ok,
		"dataset": {"invoices": str(args.invoices), "labels": str(args.labels)},
		"sample": {"n": int(args.n), "seed": int(args.seed), "docs_tested": docs_tested},
		"checks": {
			"pass_rate": pass_rate,
			"avg_latency_sec": avg_lat,
			"avg_cost_usd": avg_cost,
			"schema_errors": int(len(errors)),
			"accuracy": accuracy,
			"dla": dla,
			"latency_limit_sec": 30.0,
			"cost_limit_usd": 0.01,
		},
		"per_doc": per_doc,
		"notes": [
			"Schema validation uses strict Pydantic model validation.",
			"Accuracy and DLA are computed only when labels are provided and match doc_id.json.",
		],
	}

	write_json(report, out_path)

	lines = [
		f"Docs tested: {docs_tested}",
		f"Pass rate: {pass_rate:.2%}",
		f"Avg latency: {avg_lat:.3f}s",
		f"Avg cost: ${avg_cost:.5f}",
	]
	if accuracy is not None:
		lines.append(f"Accuracy (field): {accuracy:.2%}")
	if dla is not None:
		lines.append(f"DLA (doc-level): {dla:.2%}")
	lines.append(f"Report: {str(out_path)}")

	printed = _try_rich_print_scorecard(lines, overall_ok)
	if not printed:
		print("SCORECARD")
		print("---------")
		for ln in lines:
			print(ln)
		print("PASS" if overall_ok else "FAIL")

	return 0 if overall_ok else 2


if __name__ == "__main__":
	raise SystemExit(main())
