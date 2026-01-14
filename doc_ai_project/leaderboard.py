from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from utils.config import PipelineConfig
from utils.leaderboard_scoring import compute_final_score
from utils.table_viz import render_table_png


def _load_json(path: Path) -> Dict[str, Any]:
	obj = json.loads(path.read_text(encoding="utf-8"))
	if not isinstance(obj, dict):
		raise ValueError(f"Expected JSON object at {path}")
	return obj


def main() -> int:
	p = argparse.ArgumentParser(description="Leaderboard simulation from tuning trials")
	p.add_argument("--tuning-report", default="outputs/tuning_report.json", help="tuning_report.json path")
	p.add_argument("--top-k", type=int, default=5, help="How many rows to output")
	args = p.parse_args()

	base_dir = Path(__file__).resolve().parent
	report_path = Path(args.tuning_report)
	if not report_path.is_absolute():
		report_path = base_dir / report_path

	outputs_dir = base_dir / "outputs"
	outputs_dir.mkdir(parents=True, exist_ok=True)

	tuning = _load_json(report_path)
	trials: List[Dict[str, Any]] = tuning.get("trials") or []
	if not trials:
		raise FileNotFoundError(f"No trials found in {report_path}")

	cfg = PipelineConfig()  # just for weights/targets defaults

	scored: List[Dict[str, Any]] = []
	for t in trials:
		sumry = t.get("summary") or {}
		final = sumry.get("final_score")
		if final is None:
			final, comps = compute_final_score(sumry, cfg)
		else:
			comps = sumry.get("final_score_components") or {}
		scored.append(
			{
				"config": t.get("config") or {},
				"summary": sumry,
				"final_score": float(final),
				"final_score_components": comps,
			}
		)

	scored.sort(key=lambda r: (r.get("final_score", 0.0), r.get("summary", {}).get("dla", 0.0)), reverse=True)
	top = scored[: max(1, int(args.top_k))]

	# Simulate top-5 teams (deterministic): Team-1..Team-K = top configs
	sim_rows = []
	for i, row in enumerate(top, start=1):
		s = row.get("summary") or {}
		sim_rows.append(
			{
				"rank": i,
				"team": f"Team-{i}",
				"final_score": float(row.get("final_score") or 0.0),
				"dla": float(s.get("dla") or 0.0),
				"latency_p95_sec": float(s.get("latency_p95_sec") or 0.0),
				"cost_avg_usd": float(s.get("cost_avg_usd") or 0.0),
				"config": row.get("config") or {},
			}
		)

	sim_out = {
		"source": str(report_path),
		"weights": {
			"dla": cfg.leaderboard_weight_dla,
			"latency": cfg.leaderboard_weight_latency,
			"cost": cfg.leaderboard_weight_cost,
		},
		"targets": {
			"latency_target_sec": cfg.leaderboard_latency_target_sec,
			"cost_target_usd": cfg.leaderboard_cost_target_usd,
			"use_latency_p95": cfg.leaderboard_use_latency_p95,
		},
		"leaderboard": sim_rows,
	}

	(outputs_dir / "leaderboard_simulation.json").write_text(
		json.dumps(sim_out, ensure_ascii=False, indent=2),
		encoding="utf-8",
	)

	table_rows = []
	for r in sim_rows:
		table_rows.append(
			[
				str(r["rank"]),
				r["team"],
				f"{r['final_score']:.3f}",
				f"{r['dla']:.3f}",
				f"{r['latency_p95_sec']:.2f}",
				f"{r['cost_avg_usd']:.5f}",
			]
		)

	render_table_png(
		headers=["Rank", "Team", "Final", "DLA", "p95 Lat (s)", "Cost (USD)"],
		rows=table_rows,
		title="Leaderboard Simulation (Top configs from tuning)",
		out_path=outputs_dir / "leaderboard_table.png",
	)

	print(json.dumps({"rows": len(sim_rows), "out": "outputs/leaderboard_simulation.json"}, indent=2))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
