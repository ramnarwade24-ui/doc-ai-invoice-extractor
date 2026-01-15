#!/usr/bin/env python3
"""Leaderboard preview.

Loads an eval report JSON (from doc_ai_project/eval.py) and prints the key metrics
used for leaderboard-style comparison.

Examples:
  python leaderboard_preview.py
  python leaderboard_preview.py --report doc_ai_project/outputs/eval_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Print leaderboard metrics from eval_report.json")
	p.add_argument(
		"--report",
		default="doc_ai_project/outputs/eval_report.json",
		help="Path to eval_report.json",
	)
	return p.parse_args()


def _load(path: Path) -> Dict[str, Any]:
	obj = json.loads(path.read_text(encoding="utf-8"))
	if not isinstance(obj, dict):
		raise ValueError(f"Expected JSON object at {path}")
	return obj


def main() -> int:
	args = parse_args()
	report_path = Path(args.report)
	if not report_path.is_absolute():
		report_path = Path(__file__).resolve().parent / report_path
	if not report_path.exists():
		raise FileNotFoundError(f"Missing report: {report_path}")

	rep = _load(report_path)
	summary = rep.get("summary") or {}
	metrics = {
		"final_score": float(summary.get("final_score") or 0.0),
		"dla": float(summary.get("dla") or 0.0),
		"latency_avg_sec": float(summary.get("latency_avg_sec") or 0.0),
		"latency_p95_sec": float(summary.get("latency_p95_sec") or 0.0),
		"cost_avg_usd": float(summary.get("cost_avg_usd") or 0.0),
		"docs_total": int(summary.get("docs_total") or 0),
		"docs_with_ground_truth": int(summary.get("docs_with_ground_truth") or 0),
	}

	print(json.dumps(metrics, ensure_ascii=False, indent=2))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
