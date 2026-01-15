from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.config import PipelineConfig
from utils.config_io import load_config_json
from utils.leaderboard_scoring import compute_final_score


def _repo_root() -> Path:
	return Path(__file__).resolve().parent.parent


def _discover_pdfs(invoices: str | Path, *, repo_root: Path) -> List[Path]:
	inv_str = str(invoices)
	inv_path = Path(inv_str)
	resolved = inv_path if inv_path.is_absolute() else (repo_root / inv_path)
	if resolved.exists() and resolved.is_file():
		return [resolved] if resolved.suffix.lower() == ".pdf" else []
	if resolved.exists() and resolved.is_dir():
		return sorted([p for p in resolved.rglob("*.pdf") if p.is_file()])
	return sorted([p for p in repo_root.glob(inv_str) if p.is_file() and p.suffix.lower() == ".pdf"])


def _limit_pdfs(pdfs: List[Path], *, limit: int | None, seed: int) -> List[Path]:
	if limit is None:
		return pdfs
	k = int(limit)
	if k <= 0:
		return []
	if k >= len(pdfs):
		return pdfs
	rng = random.Random(int(seed))
	return sorted(rng.sample(pdfs, k=k))


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
	p.add_argument(
		"--invoices",
		default="",
		help=(
			"Optional dataset path for invoice PDFs to validate the selected config. "
			"Accepts a directory (recursive), single PDF, or glob like data/pdfs/**/*.pdf. "
			"Relative paths are resolved from repo root."
		),
	)
	p.add_argument(
		"--labels",
		default="",
		help="Optional labels folder for validation eval (same stem as PDF), relative to repo root.",
	)
	p.add_argument(
		"--limit",
		type=int,
		default=None,
		help="Optional deterministic sample size for validation eval (only used when --invoices is provided).",
	)
	p.add_argument("--seed", type=int, default=1337, help="Seed used for deterministic validation sampling")
	args = p.parse_args()

	report = select_best_config(tuning_report=Path(args.tuning_report), out_config=Path(args.out_config))

	# Optional evaluator-grade validation on an invoices dataset
	if args.invoices:
		from eval import evaluate_dataset
		from utils.determinism import set_deterministic

		base_dir = Path(__file__).resolve().parent
		repo_root = _repo_root()
		pdfs = _discover_pdfs(args.invoices, repo_root=repo_root)
		pdfs = _limit_pdfs(pdfs, limit=args.limit, seed=int(args.seed))
		if not pdfs:
			raise FileNotFoundError(
				"No PDFs found for --invoices. Example: --invoices data/pdfs"
			)
		labels_dir = Path(args.labels) if args.labels else None
		if labels_dir is not None and not labels_dir.is_absolute():
			labels_dir = repo_root / labels_dir
		try:
			cfg = load_config_json(Path(report["selected_config_path"]))
		except Exception:
			cfg = PipelineConfig()

		set_deterministic(seed=int(getattr(cfg, "seed", 1337)), deterministic=True)
		val = evaluate_dataset(
			invoices_dir=repo_root / "data" / "pdfs",
			pdf_paths=pdfs,
			labels_dir=labels_dir,
			config=cfg,
			outputs_dir=(base_dir / "outputs"),
		)
		report["validation_eval_summary"] = val.get("summary")
		(base_dir / "outputs" / "selector_report.json").write_text(
			json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
		)

	print(json.dumps(report, ensure_ascii=False, indent=2))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
