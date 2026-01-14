from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

from eval import evaluate_dataset
from utils.config import PipelineConfig
from utils.determinism import set_deterministic


def _failed_doc_ids(eval_report: Dict[str, Any]) -> List[str]:
	failed: List[str] = []
	for d in eval_report.get("per_doc", []):
		if not d.get("has_ground_truth"):
			continue
		fr = d.get("field_results") or {}
		any_false = False
		for meta in fr.values():
			if isinstance(meta, dict) and meta.get("correct") is False:
				any_false = True
				break
		if any_false:
			failed.append(str(d.get("doc_id")))
	return failed


def main() -> int:
	p = argparse.ArgumentParser(description="Replay failed docs with alternate OCR/extraction config")
	p.add_argument("--invoices", required=True, help="Folder with PDFs")
	p.add_argument("--labels", required=True, help="Folder with GT JSON")
	p.add_argument("--eval-report", default="outputs/eval_report.json", help="Eval report JSON")
	p.add_argument("--out", default="outputs/replay_report.json", help="Replay report JSON")
	p.add_argument("--seed", type=int, default=1337)
	args = p.parse_args()

	base_dir = Path(__file__).resolve().parent
	outputs_dir = base_dir / "outputs"
	outputs_dir.mkdir(parents=True, exist_ok=True)

	eval_path = (base_dir / args.eval_report) if not Path(args.eval_report).is_absolute() else Path(args.eval_report)
	out_path = (base_dir / args.out) if not Path(args.out).is_absolute() else Path(args.out)
	out_path.parent.mkdir(parents=True, exist_ok=True)

	set_deterministic(seed=args.seed, deterministic=True)

	try:
		eval_report = json.loads(eval_path.read_text(encoding="utf-8"))
	except Exception as e:
		raise RuntimeError(f"Failed to read eval report: {eval_path} ({e})")

	failed = _failed_doc_ids(eval_report)

	invoices_dir = Path(args.invoices) if Path(args.invoices).is_absolute() else (base_dir / args.invoices)
	labels_dir = Path(args.labels) if Path(args.labels).is_absolute() else (base_dir / args.labels)

	# Baseline config (from eval report if present)
	base_cfg = PipelineConfig(seed=args.seed, deterministic=True, run_mode="replay")

	# Alternate configs: try more OCR variants + slightly relaxed dealer threshold
	alt_cfgs: List[PipelineConfig] = [
		replace(
			base_cfg,
			dealer_fuzzy_threshold=max(75, int(base_cfg.dealer_fuzzy_threshold) - 10),
			ocr_preprocess_variants=((True, True, True), (False, True, True), (True, False, True)),
		),
		replace(
			base_cfg,
			dealer_fuzzy_threshold=max(70, int(base_cfg.dealer_fuzzy_threshold) - 15),
			ocr_preprocess_variants=((False, True, True), (True, False, True)),
			region_weight_overrides={"header": {"header": 1.0, "body": 0.6, "*": 0.55}},
		),
	]

	# Evaluate only failed docs by creating a temporary invoices folder list
	# (Implementation: run full eval but filter by stems)
	pdfs = [p for p in invoices_dir.glob("*.pdf") if p.stem in set(failed)]
	if not pdfs:
		report = {"failed_docs": failed, "note": "No failed PDFs found in invoices folder."}
		out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
		print(json.dumps(report, ensure_ascii=False, indent=2))
		return 0

	tmp_dir = outputs_dir / "_replay_subset"
	tmp_dir.mkdir(parents=True, exist_ok=True)
	# Symlinks are not guaranteed; we just re-run using the full invoices dir and compare per-doc results.

	baseline = evaluate_dataset(invoices_dir=invoices_dir, labels_dir=labels_dir, config=base_cfg, outputs_dir=outputs_dir)
	base_by_id = {d["doc_id"]: d for d in baseline.get("per_doc", [])}

	attempts: List[Dict[str, Any]] = []
	for i, cfg in enumerate(alt_cfgs):
		rep = evaluate_dataset(invoices_dir=invoices_dir, labels_dir=labels_dir, config=cfg, outputs_dir=outputs_dir)
		attempts.append({"attempt": i, "config": rep["config"], "summary": rep["summary"], "per_doc": rep.get("per_doc", [])})

	improvements: Dict[str, Any] = {}
	for doc_id in failed:
		b = base_by_id.get(doc_id)
		if not b:
			continue
		best = {"baseline": b.get("field_results"), "best_attempt": None, "best_delta_correct": 0}
		base_correct = sum(1 for v in (b.get("field_results") or {}).values() if isinstance(v, dict) and v.get("correct") is True)
		for a in attempts:
			cand = next((d for d in a.get("per_doc", []) if d.get("doc_id") == doc_id), None)
			if not cand:
				continue
			cand_correct = sum(1 for v in (cand.get("field_results") or {}).values() if isinstance(v, dict) and v.get("correct") is True)
			delta = cand_correct - base_correct
			if delta > best["best_delta_correct"]:
				best["best_delta_correct"] = delta
				best["best_attempt"] = {"attempt": a["attempt"], "field_results": cand.get("field_results")}
		improvements[doc_id] = best

	report = {
		"failed_docs": failed,
		"baseline_summary": baseline.get("summary"),
		"attempt_summaries": [{"attempt": a["attempt"], "summary": a["summary"]} for a in attempts],
		"improvements": improvements,
		"notes": "Replay reruns the whole set but analyzes only previously-failed docs. Objective: improve number of correct fields.",
	}
	out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
	print(json.dumps({"failed_docs": len(failed), "improved_docs": sum(1 for v in improvements.values() if v.get("best_delta_correct", 0) > 0)}, ensure_ascii=False, indent=2))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
