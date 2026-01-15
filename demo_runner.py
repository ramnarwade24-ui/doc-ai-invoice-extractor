#!/usr/bin/env python3
"""Demo runner CLI.

Runs the pipeline on a folder/glob of PDFs and produces a clean table + CSV.
Deterministic: uses seeded discovery/sampling from utils.demo_utils.

Examples:
  python demo_runner.py --invoices data/pdfs
  python demo_runner.py --invoices data/pdfs --limit 25 --seed 1337
  python demo_runner.py --invoices "data/pdfs/**/*.pdf" --config best_config.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


def _repo_root() -> Path:
	return Path(__file__).resolve().parent


def _ensure_docai_on_path() -> None:
	repo_root = _repo_root()
	doc_ai_dir = repo_root / "doc_ai_project"
	if str(doc_ai_dir) not in sys.path:
		sys.path.insert(0, str(doc_ai_dir))


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(
		description="Run DocAI pipeline on a PDF dataset and write demo CSV",
		epilog=(
			"Examples:\n"
			"  python demo_runner.py --invoices data/pdfs\n"
			"  python demo_runner.py --invoices data/pdfs --limit 20 --seed 1337\n"
			"  python demo_runner.py --invoices 'data/pdfs/**/*.pdf' --config best_config.json\n"
		),
		formatter_class=argparse.RawTextHelpFormatter,
	)
	p.add_argument("--invoices", default="data/pdfs", help="Folder, PDF, or glob (repo-root relative)")
	p.add_argument("--out-csv", default="outputs/demo_outputs/demo_results.csv", help="CSV output path (repo-root relative)")
	p.add_argument("--limit", type=int, default=None, help="Optional deterministic sample size")
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
	return p.parse_args()


def main() -> int:
	args = parse_args()
	# Must be set before importing doc_ai_project modules so worker paths can enable OCR.
	if bool(getattr(args, "enable_paddleocr", False)):
		import os

		os.environ["DOC_AI_ENABLE_PADDLEOCR"] = "1"
	_ensure_docai_on_path()

	from utils.demo_utils import discover_pdfs, format_table, run_pipeline, write_csv, write_json

	repo_root = _repo_root()
	out_csv = Path(str(args.out_csv))
	out_summary = out_csv.with_name("demo_summary.json")

	pdfs = discover_pdfs(str(args.invoices), recursive=True, limit=args.limit, seed=int(args.seed))
	if not pdfs:
		print("[FAIL] No PDFs found.")
		return 2

	results, errors = run_pipeline(pdfs, config_path=(str(args.config) if args.config else None), mode=str(args.mode))

	# Table output
	if results:
		print(format_table(results))
	else:
		print("[warn] No results produced.")

	# Write outputs
	write_csv(results, out_csv)

	summary: Dict[str, Any] = {
		"dataset": {"invoices": str(args.invoices), "count": int(len(pdfs))},
		"config": str(args.config or ""),
		"seed": int(args.seed),
		"limit": (int(args.limit) if args.limit is not None else None),
		"success": bool(len(errors) == 0 and len(results) == len(pdfs) and len(results) > 0),
		"results_written_csv": str(out_csv),
		"errors": errors,
		"counts": {"ok": int(len(results)), "error": int(len(errors))},
	}
	write_json(summary, out_summary)

	if errors:
		print(f"[FAIL] {len(errors)} documents failed. Summary: {out_summary}")
		return 2

	print(f"[OK] Wrote CSV: {out_csv}")
	print(f"[OK] Wrote summary: {out_summary}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
