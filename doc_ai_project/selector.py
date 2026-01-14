from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.config import PipelineConfig
from utils.config_io import save_config_json
from utils.leaderboard_scoring import compute_final_score


def _load_json(path: Path) -> Dict[str, Any]:
	obj = json.loads(path.read_text(encoding="utf-8"))
	if not isinstance(obj, dict):
		raise ValueError(f"Expected JSON object at {path}")
	return obj


def _meets_constraints(summary: Dict[str, Any], cfg: PipelineConfig) -> bool:
	dla = float(summary.get("dla") or 0.0)
	lat = float(summary.get("latency_p95_sec") or summary.get("latency_avg_sec") or 0.0)
	cost = float(summary.get("cost_avg_usd") or 0.0)
	return (
		dla >= float(cfg.submission_min_dla)
		and lat <= float(cfg.submission_max_latency_sec)
		and cost <= float(cfg.submission_max_cost_usd)
	)


def select_best_config(*, tuning_report: Path, out_config: Path) -> Dict[str, Any]:
	"""Select the best trial meeting constraints and write frozen config JSON."""
	base_dir = Path(__file__).resolve().parent
	outputs_dir = base_dir / "outputs"
	outputs_dir.mkdir(parents=True, exist_ok=True)

	report_path = tuning_report
	if not report_path.is_absolute():
		report_path = base_dir / report_path

	tuning = _load_json(report_path)
	trials: List[Dict[str, Any]] = tuning.get("trials") or []
	if not trials:
		raise FileNotFoundError(f"No trials found in {report_path}")

	cfg_defaults = PipelineConfig()
	candidates: List[Dict[str, Any]] = []
	for t in trials:
		sumry = t.get("summary") or {}
		final = sumry.get("final_score")
		if final is None:
			final, _ = compute_final_score(sumry, cfg_defaults)
		candidates.append({"trial": t, "final_score": float(final), "dla": float(sumry.get("dla") or 0.0)})

	candidates.sort(key=lambda x: (x["final_score"], x["dla"]), reverse=True)

	selected: Optional[Dict[str, Any]] = None
	selected_reason = ""
	for c in candidates:
		sumry = (c["trial"].get("summary") or {})
		if _meets_constraints(sumry, cfg_defaults):
			selected = c["trial"]
			selected_reason = "meets_constraints"
			break

	if selected is None:
		selected = candidates[0]["trial"]
		selected_reason = "no_trial_met_constraints_selected_best_final_score"

	frozen_cfg = selected.get("config") or {}
	out_cfg_path = out_config
	if not out_cfg_path.is_absolute():
		out_cfg_path = base_dir / out_cfg_path
	out_cfg_path.write_text(json.dumps(frozen_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
	(outputs_dir / "best_config_selected.json").write_text(
		json.dumps(frozen_cfg, ensure_ascii=False, indent=2),
		encoding="utf-8",
	)

	report = {
		"selected_reason": selected_reason,
		"constraints": {
			"min_dla": cfg_defaults.submission_min_dla,
			"max_latency_sec": cfg_defaults.submission_max_latency_sec,
			"max_cost_usd": cfg_defaults.submission_max_cost_usd,
		},
		"selected_summary": selected.get("summary") or {},
		"selected_config_path": str(out_cfg_path),
		"source_tuning_report": str(report_path),
	}
	(outputs_dir / "selector_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
	return report


def main() -> int:
	p = argparse.ArgumentParser(description="Select best config that meets submission constraints")
	p.add_argument("--tuning-report", default="outputs/tuning_report.json")
	p.add_argument("--out-config", default="best_config.json", help="Frozen config path (relative to doc_ai_project)")
	args = p.parse_args()

	report = select_best_config(tuning_report=Path(args.tuning_report), out_config=Path(args.out_config))
	print(json.dumps(report, ensure_ascii=False, indent=2))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
